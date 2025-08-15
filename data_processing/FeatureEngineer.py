import pandas as pd
import numpy as np
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from .data_constants import (
    IDENTITY_BLACKLIST, PURE_GEOGRAPHIC_IDENTIFIERS, CLIMATE_GEOGRAPHIC_FEATURES,
    APPROVED_ECOLOGICAL_FEATURES, SPECIES_FUNCTIONAL_GROUPS, CATEGORICAL_ENCODINGS,
    BIOME_MAP, IGBP_MAP, LEAF_HABIT_MAP, PROBLEMATIC_COLUMNS_TO_EXCLUDE
)
from .error_utils import ErrorHandler
from .logging_utils import logger, LogPhase

class FeatureEngineer:
    """
    Optimized feature engineering class for SAPFLUXNET data processing.
    
    This class implements performance optimizations including:
    - Column existence and data caching
    - In-place operations to reduce memory usage
    - Vectorized operations for faster processing
    - Pre-computation of frequently used data
    """
    
    def __init__(self, df, config=None, memory_manager=None):
        """
        Initialize the FeatureEngineer with a DataFrame and configuration.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
            config (dict): Configuration dictionary for feature creation
            memory_manager (MemoryManager): MemoryManager instance for system memory operations
        """
        self.df = df
        self.config = config or self._default_config()
        self.memory_manager = memory_manager
        
        # Cache frequent attribute checks for performance FIRST (needed by other init methods)
        # Note: These will be False for dictionary configs, True for ProcessingConfig class
        self._has_get_feature_setting = hasattr(self.config, 'get_feature_setting')
        self._has_get_chunk_size = hasattr(self.config, 'get_chunk_size')
        
        # Initialize caches
        self._column_cache = {}
        self._data_cache = {}
        self._temporal_cache = {}
        self._gc_counter = 0
        
        # Pre-compute everything for performance
        self._precompute_columns()
        self._preprocess_timestamps()
        
        # Initialize memory monitoring
        self._initial_memory_mb = self._get_dataframe_memory_mb()
        self._should_use_chunking = self._determine_chunking_strategy()
        
        # Cache frequent column existence checks
        self._column_exists_cache = {
            'latitude': 'latitude' in self.df.columns,
            'mean_annual_temp': 'mean_annual_temp' in self.df.columns,
            'mean_annual_precip': 'mean_annual_precip' in self.df.columns,
            'leaf_habit': 'leaf_habit' in self.df.columns,
            'biome': 'biome' in self.df.columns,
            'igbp_class': 'igbp_class' in self.df.columns,
            'solar_TIMESTAMP': 'solar_TIMESTAMP' in self.df.columns
        }
    
    def _update_column_cache(self):
        """Update column existence cache after metadata is loaded"""
        self._column_exists_cache.update({
            'latitude': 'latitude' in self.df.columns,
            'mean_annual_temp': 'mean_annual_temp' in self.df.columns,
            'mean_annual_precip': 'mean_annual_precip' in self.df.columns,
            'leaf_habit': 'leaf_habit' in self.df.columns,
            'biome': 'biome' in self.df.columns,
            'igbp_class': 'igbp_class' in self.df.columns,
            'solar_TIMESTAMP': 'solar_TIMESTAMP' in self.df.columns
        })

    def _is_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check feature flag from config or use default."""
        try:
            if self._has_get_feature_setting:
                val = self.config.get_feature_setting(flag_name)
                return bool(val) if val is not None else default
            else:
                return bool(self.config.get(flag_name, default))
        except Exception:
            return default
    
    def _default_config(self):
        """Default configuration for feature creation"""
        return {
            'temporal_features': True,
            'rolling_windows': [3, 6, 12, 24, 48, 72, 168, 336, 720],
            'interaction_features': True,
            'metadata_features': True,
            'categorical_encoding': True,
            'seasonality_features': True,
            'domain_features': True,
            'parallel_processing': True,
            'max_parallel_workers': min(4, multiprocessing.cpu_count()),
            'memory_chunking': True,
            'chunk_size': 50000,  # Rows per chunk
            'memory_threshold_mb': 1000,  # Auto-enable chunking above this memory usage
            'adaptive_chunking': True  # Automatically adjust chunk size based on memory
        }
    
    def _precompute_columns(self):
        """Cache column existence and frequently used data"""
        # Cache column existence for all columns
        all_columns = set(self.df.columns)
        self._column_cache = {col: col in all_columns for col in all_columns}
        
        # Pre-cache frequently accessed columns (preserve NaN values)
        common_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        for col in common_cols:
            if self._column_cache.get(col):
                # Cache column data without imputing missing values (reference for read-only access)
                self._data_cache[col] = self.df[col]  # No copy needed for read-only operations
    
    def _preprocess_timestamps(self):
        """Pre-process timestamps once instead of multiple times"""
        # Import DataLoader for timestamp utilities (avoid circular imports)
        from .DataLoader import DataLoader
        
        # Use centralized timestamp utilities
        DataLoader.standardize_timestamp_columns(self.df, inplace=True)
        
        # Get primary timestamp for temporal cache
        main_timestamp = DataLoader.get_primary_timestamp_column(self.df)
        
        if main_timestamp:
            # Pre-compute common temporal features
            dt = self.df[main_timestamp].dt
            self._temporal_cache = {
                'hour': dt.hour,
                'day_of_year': dt.dayofyear,
                'month': dt.month,
                'year': dt.year,
                'day_of_week': dt.dayofweek
            }
    
    def _get_column(self, col_name, default_val=np.nan):
        """
        Optimized column access with caching - preserves missing values as np.nan.
        
        Args:
            col_name (str): Name of the column to retrieve
            default_val: Default value if column doesn't exist (should be np.nan)
            
        Returns:
            pd.Series: Column data with original NaN values preserved
        """
        # Check cache first
        if col_name in self._data_cache:
            return self._data_cache[col_name]
        
        # Check if column exists
        if col_name in self._column_cache and self._column_cache[col_name]:
            # Get column data without imputing missing values
            col_data = self.df[col_name]  # Use reference for read-only operations
            # Cache for future use
            self._data_cache[col_name] = col_data
            return col_data
        else:
            # Column doesn't exist, return default values (should be np.nan)
            return pd.Series([default_val] * len(self.df), index=self.df.index)
    
    def _get_system_memory_status(self):
        """Get system memory status via MemoryManager"""
        if self.memory_manager:
            return self.memory_manager.get_memory_status()
        else:
            # Fallback if no MemoryManager provided
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent_used': memory.percent
                }
            except:
                return {'available_gb': 4.0, 'used_gb': 0, 'percent_used': 0}
    
    def _get_dataframe_memory_mb(self):
        """Get DataFrame memory usage in MB"""
        return self.df.memory_usage(deep=True).sum() / 1024 / 1024
    
    def _determine_chunking_strategy(self):
        """Determine if chunking should be used based on data size and memory"""
        # Check if memory chunking is enabled using cached attribute check
        if self._has_get_feature_setting:
            if not self.config.get_feature_setting('memory_chunking'):
                return False
            memory_threshold = self.config.get_feature_setting('memory_threshold_mb') or 1000
        else:
            # Fallback for when config is a dict instead of ProcessingConfig class
            if not self.config.get('memory_chunking', True):
                return False
            memory_threshold = self.config.get('memory_threshold_mb', 1000)
        
        # Check DataFrame size
        df_memory_mb = self._get_dataframe_memory_mb()
        
        # Auto-enable chunking for large datasets
        if df_memory_mb > memory_threshold:
            logger.memory_info("Large dataset detected - enabling memory chunking", df_memory_mb)
            return True
        
        # Check available system memory via MemoryManager
        memory_status = self._get_system_memory_status()
        available_memory_mb = memory_status['available_gb'] * 1024
        
        if df_memory_mb > available_memory_mb * 0.3:  # Use chunking if DF > 30% of available memory
            logger.memory_info("Memory-constrained environment - enabling chunking", available_memory_mb / 1024)
            if self.memory_manager:
                self.memory_manager.force_memory_cleanup()
            return True
        
        return False
    
    def _calculate_optimal_chunk_size(self):
        """Calculate optimal chunk size based on available memory and data characteristics"""
        # Calculate optimal chunk size using standardized config access
        if hasattr(self.config, 'get_chunk_size'):
            base_chunk_size = self.config.get_chunk_size('medium_memory')
            adaptive_chunking = self.config.get_feature_setting('adaptive_chunking')
        else:
            # Fallback for dict config
            base_chunk_size = self.config.get('chunk_size', 50000)
            adaptive_chunking = self.config.get('adaptive_chunking', True)
        
        if not adaptive_chunking:
            return base_chunk_size
        
        try:
            # Get available memory via MemoryManager
            memory_status = self._get_system_memory_status()
            available_memory_mb = memory_status['available_gb'] * 1024
            df_memory_mb = self._get_dataframe_memory_mb()
            
            # Calculate memory per row
            memory_per_row_kb = (df_memory_mb * 1024) / len(self.df)
            
            # Target: use max 20% of available memory per chunk
            target_chunk_memory_mb = available_memory_mb * 0.2
            optimal_chunk_size = int((target_chunk_memory_mb * 1024) / memory_per_row_kb)
            
            # Clamp to reasonable bounds
            optimal_chunk_size = max(10000, min(optimal_chunk_size, 100000))
            
            logger.performance_info("Adaptive chunking", f"{optimal_chunk_size:,} rows per chunk")
            return optimal_chunk_size
            
        except Exception:
            return base_chunk_size
    
    def _create_temporal_features_batch(self):
        """Create all temporal features in one batch operation"""
        if not self._temporal_cache:
            print("    ⚠️  No temporal cache available, skipping temporal features")
            return
        
        print("    ⏰ Creating Temporal Features...")
        
        # Pre-compute all temporal values
        hour_rad = 2 * np.pi * self._temporal_cache['hour'] / 24
        day_rad = 2 * np.pi * self._temporal_cache['day_of_year'] / 365
        month_rad = 2 * np.pi * self._temporal_cache['month'] / 12
        
        # Batch assign all temporal features
        temporal_features = {
            'hour': self._temporal_cache['hour'],
            'day_of_year': self._temporal_cache['day_of_year'],
            'month': self._temporal_cache['month'],
            'year': self._temporal_cache['year'],
            'day_of_week': self._temporal_cache['day_of_week'],
            
            # Cyclical features
            'hour_sin': np.sin(hour_rad),
            'hour_cos': np.cos(hour_rad),
            'day_sin': np.sin(day_rad),
            'day_cos': np.cos(day_rad),
            'month_sin': np.sin(month_rad),
            'month_cos': np.cos(month_rad),
            
            # Boolean features
            'is_daylight': ((self._temporal_cache['hour'] >= 6) & (self._temporal_cache['hour'] <= 18)).astype(int),
            'is_peak_sunlight': ((self._temporal_cache['hour'] >= 10) & (self._temporal_cache['hour'] <= 16)).astype(int),
            'is_weekend': (self._temporal_cache['day_of_week'] >= 5).astype(int),
            'is_morning': ((self._temporal_cache['hour'] >= 6) & (self._temporal_cache['hour'] <= 12)).astype(int),
            'is_afternoon': ((self._temporal_cache['hour'] >= 12) & (self._temporal_cache['hour'] <= 18)).astype(int),
            'is_night': ((self._temporal_cache['hour'] < 6) | (self._temporal_cache['hour'] > 18)).astype(int),
            
            # Seasonal features
            'is_spring': ((self._temporal_cache['month'] >= 3) & (self._temporal_cache['month'] <= 5)).astype(int),
            'is_summer': ((self._temporal_cache['month'] >= 6) & (self._temporal_cache['month'] <= 8)).astype(int),
            'is_autumn': ((self._temporal_cache['month'] >= 9) & (self._temporal_cache['month'] <= 11)).astype(int),
            'is_winter': ((self._temporal_cache['month'] == 12) | (self._temporal_cache['month'] <= 2)).astype(int),
            
            # Time since sunrise/sunset
            'hours_since_sunrise': (self._temporal_cache['hour'] - 6) % 24,
            'hours_since_sunset': (self._temporal_cache['hour'] - 18) % 24
        }
        
        # Single batch assignment
        for name, values in temporal_features.items():
            self.df[name] = values
        
        # Handle solar timestamp features if available
        self._create_solar_temporal_features()
    
    def _create_solar_temporal_features(self):
        """Create solar-adjusted temporal features"""
        if self._column_exists_cache['solar_TIMESTAMP'] and not self.df['solar_TIMESTAMP'].isna().all():
            solar_dt = pd.to_datetime(self.df['solar_TIMESTAMP'], errors='coerce').dt
            self.df['solar_hour'] = solar_dt.hour.fillna(self._temporal_cache['hour'])
            self.df['solar_day_of_year'] = solar_dt.dayofyear.fillna(self._temporal_cache['day_of_year'])
        else:
            # No solar timestamp - use regular timestamp values
            self.df['solar_hour'] = self._temporal_cache['hour']
            self.df['solar_day_of_year'] = self._temporal_cache['day_of_year']
        
        # Create solar-adjusted cyclical features
        solar_hour_rad = 2 * np.pi * self.df['solar_hour'] / 24
        solar_day_rad = 2 * np.pi * self.df['solar_day_of_year'] / 365
        
        self.df['solar_hour_sin'] = np.sin(solar_hour_rad)
        self.df['solar_hour_cos'] = np.cos(solar_hour_rad)
        self.df['solar_day_sin'] = np.sin(solar_day_rad)
        self.df['solar_day_cos'] = np.cos(solar_day_rad)
    
    def create_all_features(self, metadata=None, timestamp_col=None, processing_mode='standard', parallel=True):
        """
        Main feature creation method with parallel processing optimizations.
        
        Args:
            metadata (dict): Site metadata
            timestamp_col (str): Timestamp column name
            processing_mode (str): 'standard' or 'streaming'
            parallel (bool): Whether to use parallel processing for independent feature groups
            
        Returns:
            pd.DataFrame: DataFrame with all features created
        """
        # Use key phase logging (always shown per user preference)
        logger.phase_start("Engineering Features")
        
        try:
            # Determine processing strategy based on data size and configuration
            chunk_threshold = (self.config.get_chunk_size('medium_memory') 
                             if self._has_get_chunk_size 
                             else self.config.get('chunk_size', 50000))
            parallel_enabled = (self.config.get_feature_setting('parallel_processing') 
                              if self._has_get_feature_setting 
                              else self.config.get('parallel_processing', True))
            
            use_chunking = self._should_use_chunking and len(self.df) > chunk_threshold
            use_parallel = (parallel and 
                           processing_mode == 'standard' and 
                           parallel_enabled)
            
            if use_chunking:
                return self._create_features_chunked(metadata, timestamp_col, processing_mode, use_parallel)
            elif use_parallel:
                return self._create_features_parallel(metadata, timestamp_col, processing_mode)
            else:
                return self._create_features_sequential(metadata, timestamp_col, processing_mode)
            
        finally:
            self._cleanup_intermediate_data()
            # Complete key phase logging
            logger.phase_complete("Engineering Features", f"{len(self.df):,} rows, {len(self.df.columns)} features")
    
    def _create_features_chunked(self, metadata=None, timestamp_col=None, processing_mode='standard', use_parallel=False):
        """
        Memory-efficient chunked feature creation for large datasets.
        
        Processes the DataFrame in chunks to minimize memory usage while maintaining
        all feature creation capabilities including parallel processing within chunks.
        """
        chunk_size = self._calculate_optimal_chunk_size()
        total_rows = len(self.df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Ceiling division
        
        print(f"    🧩 Processing {total_rows:,} rows in {num_chunks} chunks of {chunk_size:,} rows")
        print(f"    💾 Initial memory usage: {self._get_memory_usage_mb():.1f}MB")
        
        # Store original DataFrame and create empty result
        original_df = self.df
        processed_chunks = []
        
        try:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                logger.performance_info(f"Processing chunk {chunk_idx + 1}/{num_chunks}", f"rows {start_idx:,}-{end_idx:,}")
                
                # Create chunk DataFrame - copy needed since we modify the data
                chunk_df = original_df.iloc[start_idx:end_idx].copy()
                
                # Create new FeatureEngineer instance for this chunk
                chunk_engineer = FeatureEngineer(chunk_df, self.config)
                
                # Process chunk with same parameters (but disable chunking to avoid recursion)
                # Modify config directly instead of copying
                original_chunking = chunk_engineer.config.get('memory_chunking', True)
                chunk_engineer.config['memory_chunking'] = False
                
                # Process the chunk
                if use_parallel and processing_mode == 'standard':
                    processed_chunk = chunk_engineer._create_features_parallel(metadata, timestamp_col, processing_mode)
                else:
                    processed_chunk = chunk_engineer._create_features_sequential(metadata, timestamp_col, processing_mode)
                
                processed_chunks.append(processed_chunk)
                
                # Restore config setting
                chunk_engineer.config['memory_chunking'] = original_chunking
                
                # Memory cleanup after each chunk
                del chunk_df, chunk_engineer
                if self.memory_manager:
                    self.memory_manager.force_memory_cleanup()
                else:
                    gc.collect()
                
                # Memory monitoring
                memory_status = self._get_system_memory_status()
                current_memory = memory_status['used_gb']
                logger.memory_info(f"Chunk {chunk_idx + 1} complete", current_memory)
            
            # Combine all processed chunks
            print("    🔗 Combining processed chunks...")
            self.df = pd.concat(processed_chunks, ignore_index=True)
            
            # Final memory cleanup
            del processed_chunks, original_df
            if self.memory_manager:
                self.memory_manager.force_memory_cleanup()
            else:
                gc.collect()
            
            memory_status = self._get_system_memory_status()
            final_memory = memory_status['used_gb']
            print(f"    ✅ Chunked processing complete - Final memory: {final_memory:.1f}GB")
            
            return self.df
            
        except Exception as e:
            ErrorHandler.handle_processing_error("Chunked processing", e)
            # Restore original DataFrame
            self.df = original_df
            # Fallback to sequential processing
            print("    🔄 Falling back to sequential processing...")
            return self._create_features_sequential(metadata, timestamp_col, processing_mode)
    
    def _create_features_sequential(self, metadata=None, timestamp_col=None, processing_mode='standard'):
        """Sequential feature creation (original method)"""
        # Phase 1: Temporal Features (in-place)
        if self.config['temporal_features']:
            self._create_temporal_features_batch()
        
        # Phase 2: Environmental Features (in-place)
        self._create_environmental_features(processing_mode)
        
        # Phase 3: Interaction Features (in-place)
        if self.config['interaction_features']:
            self._create_interaction_features_vectorized()
        
        # Phase 4: Metadata Features (in-place)
        if self.config['metadata_features'] and metadata:
            self._create_metadata_features(metadata)
        
        # Phase 5: Domain Features (in-place)
        if self.config['domain_features'] and processing_mode == 'standard':
            self._create_domain_specific_features()
        
        # Phase 6: Seasonality Features (in-place)
        if self.config['seasonality_features'] and processing_mode == 'standard':
            self._create_seasonality_features()
        
        # Phase 7: Encoding Features
        if self.config['categorical_encoding']:
            self._encode_categorical_features()
        
        # Phase 8: Final cleanup
        self._drop_problematic_columns()
        self._ensure_consistent_schema()
        
                    # Feature engineering completion is now logged by key phase logging
        return self.df
    
    def _create_features_parallel(self, metadata=None, timestamp_col=None, processing_mode='standard'):
        """
        Parallel feature creation method - processes independent feature groups simultaneously.
        
        This method identifies feature groups that don't depend on each other and processes
        them in parallel threads for significant performance improvements.
        """
        print("    ⚡ Engineering Features (Parallel Mode)...")
        
        # Phase 1: Create temporal features if enabled
        if self.config.get('temporal_features', False):
            self._create_temporal_features_batch()
        
        # Phase 2: Create independent feature groups in parallel
        self._create_independent_features_parallel(metadata, processing_mode)
        
        # Phase 3: Create dependent features sequentially
        self._create_dependent_features_sequential(processing_mode)
        
        # Phase 4: Final cleanup
        self._drop_problematic_columns()
        self._ensure_consistent_schema()
        
                    # Feature engineering completion is now logged by key phase logging
        return self.df
    
    def _create_independent_features_parallel(self, metadata, processing_mode):
        """
        Create independent feature groups in parallel using ThreadPoolExecutor.
        
        These feature groups don't depend on each other and can be safely processed
        simultaneously for significant performance improvements.
        """
        # Use configured number of workers
        max_workers = (self.config.get_feature_setting('max_parallel_workers') or min(4, multiprocessing.cpu_count())
                      if hasattr(self.config, 'get_feature_setting') 
                      else self.config.get('max_parallel_workers', min(4, multiprocessing.cpu_count())))
        
        # Define independent feature creation tasks
        independent_tasks = []
        
        # Task 1: Environmental features (rolling, lagged, rate of change, cumulative)
        if processing_mode == 'standard' and (
            self.config.get('rolling_features', False) or 
            self.config.get('lagged_features', False) or 
            self.config.get('rate_of_change_features', False) or 
            self.config.get('cumulative_features', False)
        ):
            independent_tasks.append(('environmental', self._create_environmental_features_task, processing_mode))
        
        # Task 2: Interaction features
        if self.config.get('interaction_features', False):
            independent_tasks.append(('interaction', self._create_interaction_features_task, None))
        
        # Task 3: Domain-specific features
        if self.config.get('domain_features', False) and processing_mode == 'standard':
            independent_tasks.append(('domain', self._create_domain_features_task, None))
        
        # Task 4: Metadata features (if available)
        if self.config.get('metadata_features', True) and metadata:
            independent_tasks.append(('metadata', self._create_metadata_features_task, metadata))
        
        # Execute independent tasks in parallel
        if independent_tasks:
            print(f"    🚀 Running {len(independent_tasks)} feature groups in parallel ({max_workers} threads)...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                for task_name, task_func, task_args in independent_tasks:
                    future = executor.submit(task_func, task_args)
                    future_to_task[future] = task_name
                
                # Wait for all tasks to complete and handle results
                completed_tasks = []
                for future in future_to_task:
                    task_name = future_to_task[future]
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per task
                        completed_tasks.append(task_name)
                        print(f"    ✅ {task_name.capitalize()} features completed")
                    except Exception as e:
                        ErrorHandler.handle_feature_creation_error(task_name.capitalize(), e)
                        # Continue with other tasks
                
                print(f"    🎯 Parallel processing complete: {len(completed_tasks)}/{len(independent_tasks)} tasks successful")
    
    def _create_dependent_features_sequential(self, processing_mode):
        """
        Create features that depend on previously created features.
        These must be run sequentially after independent features are complete.
        """
        # Seasonality features (may depend on temporal features)
        if self.config.get('seasonality_features', False) and processing_mode == 'standard':
            self._create_seasonality_features()
        
        # Categorical encoding (depends on metadata features being complete)
        if self.config.get('categorical_encoding', True):
            self._encode_categorical_features()
    
    def _create_environmental_features_task(self, processing_mode):
        """Thread-safe wrapper for environmental features creation"""
        try:
            self._create_environmental_features(processing_mode)
            return True
        except Exception as e:
            print(f"    ⚠️  Environmental features error: {str(e)}")
            return False
    
    def _create_interaction_features_task(self, _):
        """Thread-safe wrapper for interaction features creation"""
        try:
            self._create_interaction_features_vectorized()
            return True
        except Exception as e:
            print(f"    ⚠️  Interaction features error: {str(e)}")
            return False
    
    def _create_domain_features_task(self, _):
        """Thread-safe wrapper for domain features creation"""
        try:
            self._create_domain_specific_features()
            return True
        except Exception as e:
            print(f"    ⚠️  Domain features error: {str(e)}")
            return False
    
    def _create_metadata_features_task(self, metadata):
        """Thread-safe wrapper for metadata features creation"""
        try:
            self._create_metadata_features(metadata)
            return True
        except Exception as e:
            print(f"    ⚠️  Metadata features error: {str(e)}")
            return False
    
    def _create_temporal_features_batch(self):
        """Create all temporal features in one batch operation"""
        if not self._temporal_cache:
            print("    ⚠️  No temporal cache available, skipping temporal features")
            return
        
        print("    ⏰ Creating Temporal Features...")
        
        # Pre-compute all temporal values
        hour_rad = 2 * np.pi * self._temporal_cache['hour'] / 24
        day_rad = 2 * np.pi * self._temporal_cache['day_of_year'] / 365
        month_rad = 2 * np.pi * self._temporal_cache['month'] / 12
        
        # Batch assign all temporal features
        temporal_features = {
            'hour': self._temporal_cache['hour'],
            'day_of_year': self._temporal_cache['day_of_year'],
            'month': self._temporal_cache['month'],
            'year': self._temporal_cache['year'],
            'day_of_week': self._temporal_cache['day_of_week'],
            
            # Cyclical features
            'hour_sin': np.sin(hour_rad),
            'hour_cos': np.cos(hour_rad),
            'day_sin': np.sin(day_rad),
            'day_cos': np.cos(day_rad),
            'month_sin': np.sin(month_rad),
            'month_cos': np.cos(month_rad),
            
            # Boolean features
            'is_daylight': ((self._temporal_cache['hour'] >= 6) & (self._temporal_cache['hour'] <= 18)).astype(int),
            'is_peak_sunlight': ((self._temporal_cache['hour'] >= 10) & (self._temporal_cache['hour'] <= 16)).astype(int),
            'is_weekend': (self._temporal_cache['day_of_week'] >= 5).astype(int),
            'is_morning': ((self._temporal_cache['hour'] >= 6) & (self._temporal_cache['hour'] <= 12)).astype(int),
            'is_afternoon': ((self._temporal_cache['hour'] >= 12) & (self._temporal_cache['hour'] <= 18)).astype(int),
            'is_night': ((self._temporal_cache['hour'] < 6) | (self._temporal_cache['hour'] > 18)).astype(int),
            
            # Seasonal features
            'is_spring': ((self._temporal_cache['month'] >= 3) & (self._temporal_cache['month'] <= 5)).astype(int),
            'is_summer': ((self._temporal_cache['month'] >= 6) & (self._temporal_cache['month'] <= 8)).astype(int),
            'is_autumn': ((self._temporal_cache['month'] >= 9) & (self._temporal_cache['month'] <= 11)).astype(int),
            'is_winter': ((self._temporal_cache['month'] == 12) | (self._temporal_cache['month'] <= 2)).astype(int),
            
            # Time since sunrise/sunset
            'hours_since_sunrise': (self._temporal_cache['hour'] - 6) % 24,
            'hours_since_sunset': (self._temporal_cache['hour'] - 18) % 24
        }
        
        # Single batch assignment
        for name, values in temporal_features.items():
            self.df[name] = values
        
        # Handle solar timestamp features if available
        self._create_solar_temporal_features()
    
    def _create_solar_temporal_features(self):
        """Create solar-adjusted temporal features"""
        if self._column_exists_cache['solar_TIMESTAMP'] and not self.df['solar_TIMESTAMP'].isna().all():
            solar_dt = pd.to_datetime(self.df['solar_TIMESTAMP'], errors='coerce').dt
            self.df['solar_hour'] = solar_dt.hour.fillna(self._temporal_cache['hour'])
            self.df['solar_day_of_year'] = solar_dt.dayofyear.fillna(self._temporal_cache['day_of_year'])
        else:
            # No solar timestamp - use regular timestamp values
            self.df['solar_hour'] = self._temporal_cache['hour']
            self.df['solar_day_of_year'] = self._temporal_cache['day_of_year']
        
        # Create solar-adjusted cyclical features
        solar_hour_rad = 2 * np.pi * self.df['solar_hour'] / 24
        solar_day_rad = 2 * np.pi * self.df['solar_day_of_year'] / 365
        
        self.df['solar_hour_sin'] = np.sin(solar_hour_rad)
        self.df['solar_hour_cos'] = np.cos(solar_hour_rad)
        self.df['solar_day_sin'] = np.sin(solar_day_rad)
        self.df['solar_day_cos'] = np.cos(solar_day_rad)
    
    def _create_environmental_features(self, processing_mode='standard'):
        """Create environmental features (rolling, lagged, rate of change, cumulative)"""
        print("    🌍 Creating Environmental Features...")
        
        # Get environmental columns
        env_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        
        # Create rolling features (conditional)
        if self.config.get('rolling_features', True):
            self._create_rolling_features_vectorized(env_cols)
        
        # Create lagged features (conditional and standard mode only)
        if processing_mode == 'standard':
            if self.config.get('lagged_features', True):
                self._create_lagged_features_vectorized(env_cols)
            if self.config.get('rate_of_change_features', True):
                self._create_rate_of_change_features_vectorized(env_cols)
            if self.config.get('cumulative_features', True):
                self._create_cumulative_features_vectorized()
    
    def _create_rolling_features_vectorized(self, env_cols):
        """Create rolling features using highly optimized single-pass operations"""
        windows = self.config['rolling_windows']
        
        for col in env_cols:
            if col in self._data_cache:
                col_data = self._data_cache[col]
                
                # Ultra-optimized: Process all windows and statistics in parallel batches
                # Process all windows for this column in batch for better memory efficiency
                rolling_objects = {w: col_data.rolling(w, min_periods=1) for w in windows}
                
                for window in windows:
                    rolling_obj = rolling_objects[window]
                    
                    # Pre-allocate feature dictionary for batch assignment
                    window_features = {}
                    
                    # Always compute mean and std
                    window_features[f'{col}_mean_{window}h'] = rolling_obj.mean()
                    window_features[f'{col}_std_{window}h'] = rolling_obj.std().fillna(0)
                    
                    # Conditionally compute min/max/range for larger windows
                    if window >= 72:
                        min_vals = rolling_obj.min()
                        max_vals = rolling_obj.max()
                        window_features[f'{col}_min_{window}h'] = min_vals
                        window_features[f'{col}_max_{window}h'] = max_vals
                        window_features[f'{col}_range_{window}h'] = max_vals - min_vals
                    
                    # Single batch assignment for all features in this window
                    for feature_name, feature_values in window_features.items():
                        self.df[feature_name] = feature_values
                    
                    # Memory cleanup for large windows
                    if window >= 336:
                        del window_features
                
                # Clean up rolling objects after processing all windows for this column
                del rolling_objects
                if len(windows) > 5:  # Only force cleanup for many windows
                    if self.memory_manager:
                        self.memory_manager.force_memory_cleanup()
                    else:
                        import gc
                        gc.collect()
    
    def _create_lagged_features_vectorized(self, env_cols):
        """Create lagged features using optimized batch operations"""
        lag_windows = [1, 2, 3, 6, 12, 24]
        
        for col in env_cols:
            if col in self._data_cache:
                col_data = self._data_cache[col]
                
                # Batch create all lag features for this column
                lag_features = {}
                for lag in lag_windows:
                    lag_features[f'{col}_lag_{lag}h'] = col_data.shift(lag)
                
                # Single batch assignment for all lag features
                for feature_name, feature_values in lag_features.items():
                    self.df[feature_name] = feature_values
    
    def _create_rate_of_change_features_vectorized(self, env_cols):
        """Create rate of change features using optimized batch operations"""
        rate_periods = [1, 6, 24]
        
        for col in env_cols:
            if col in self._data_cache:
                col_data = self._data_cache[col]
                
                # Batch create all rate of change features for this column
                rate_features = {}
                for period in rate_periods:
                    rate_features[f'{col}_rate_{period}h'] = col_data.diff(period)
                
                # Single batch assignment for all rate features
                for feature_name, feature_values in rate_features.items():
                    self.df[feature_name] = feature_values
    
    def _create_cumulative_features_vectorized(self):
        """Create cumulative features using optimized batch operations"""
        print("    📈 Creating Cumulative Features...")
        
        # Get precipitation and radiation data with cached lookup
        precip = self._get_column('precip', np.nan)
        sw_in = self._get_column('sw_in', np.nan)
        
        # OPTIMIZATION: Use dictionary comprehension for batch rolling creation
        cum_windows = [24, 72, 168]
        
        # Create all rolling objects at once for better memory management
        precip_rolling = {w: precip.rolling(w, min_periods=1) for w in cum_windows if w <= 168}
        sw_in_rolling = {w: sw_in.rolling(w, min_periods=1) for w in [24, 72]}
        
        # Batch create all cumulative features using pre-computed rolling objects
        cumulative_features = {}
        
        # Precipitation cumulative features (vectorized)
        cumulative_features.update({
            f'precip_cum_{w}h': rolling_obj.sum() 
            for w, rolling_obj in precip_rolling.items()
        })
        
        # Radiation cumulative features (vectorized)
        cumulative_features.update({
            f'sw_in_cum_{w}h': rolling_obj.sum() 
            for w, rolling_obj in sw_in_rolling.items()
        })
        
        # Single batch assignment for all cumulative features
        for feature_name, feature_values in cumulative_features.items():
            self.df[feature_name] = feature_values
        
        # Clean up rolling objects
        del precip_rolling, sw_in_rolling
    
    def _create_interaction_features_vectorized(self):
        """Create enhanced interaction features - ALWAYS CREATE ALL FEATURES"""
        print("    🔗 Creating Interaction Features...")
        
        # Get all required columns with np.nan as default for missing data
        required_cols = ['vpd', 'ppfd_in', 'ta', 'rh', 'swc_shallow', 'sw_in', 'ws']
        col_data = {col: self._get_column(col, np.nan) for col in required_cols}
        
        # Create all interactions in one batch - ALWAYS CREATE ALL FEATURES
        interactions = {
            'vpd_ppfd_interaction': col_data['vpd'] * col_data['ppfd_in'],
            'vpd_ta_interaction': col_data['vpd'] * col_data['ta'],
            'temp_humidity_ratio': col_data['ta'] / (col_data['rh'] + 1e-6),
            'water_stress_index': col_data['swc_shallow'] / (col_data['vpd'] + 1e-6),
            'light_efficiency': col_data['ppfd_in'] / (col_data['sw_in'] + 1e-6),
            'temp_soil_interaction': col_data['ta'] * col_data['swc_shallow'],
            'wind_vpd_interaction': col_data['ws'] * col_data['vpd'],
            'radiation_temp_interaction': col_data['sw_in'] * col_data['ta'],
            'humidity_soil_interaction': col_data['rh'] * col_data['swc_shallow']
        }
        
        # Batch assign
        for name, values in interactions.items():
            self.df[name] = values
    
    def _create_metadata_features(self, metadata):
        """Create features from metadata (reduced logging)"""
        print("    📊 Creating Metadata Features...")
        
        if not metadata:
            return
        
        # Site-level features
        if 'site' in metadata:
            site_data = metadata['site']
            
            # Geographic features
            if 'si_lat' in site_data and pd.notna(site_data['si_lat']):
                self.df['latitude'] = site_data['si_lat']
            if 'si_long' in site_data and pd.notna(site_data['si_long']):
                self.df['longitude'] = site_data['si_long']
            if 'si_elev' in site_data and pd.notna(site_data['si_elev']):
                self.df['elevation'] = site_data['si_elev']
            
            # Climate features
            if 'si_mat' in site_data and pd.notna(site_data['si_mat']):
                self.df['mean_annual_temp'] = site_data['si_mat']
            if 'si_map' in site_data and pd.notna(site_data['si_map']):
                self.df['mean_annual_precip'] = site_data['si_map']
            
            # Biome and land cover
            if 'si_biome' in site_data and pd.notna(site_data['si_biome']):
                self.df['biome'] = site_data['si_biome']
            if 'si_igbp' in site_data and pd.notna(site_data['si_igbp']):
                self.df['igbp_class'] = site_data['si_igbp']
            
            # Country
            if 'si_country' in site_data and pd.notna(site_data['si_country']):
                self.df['country'] = site_data['si_country']
            
            # Additional site metadata
            if 'si_code' in site_data and pd.notna(site_data['si_code']):
                self.df['site_code'] = site_data['si_code']
            if 'si_name' in site_data and pd.notna(site_data['si_name']):
                self.df['site_name'] = site_data['si_name']
            if 'is_inside_country' in site_data and pd.notna(site_data['is_inside_country']):
                self.df['is_inside_country'] = site_data['is_inside_country']
        
        # Stand-level features
        if 'stand' in metadata:
            stand_data = metadata['stand']
            
            # Stand characteristics
            if 'st_age' in stand_data and pd.notna(stand_data['st_age']):
                self.df['stand_age'] = stand_data['st_age']
            if 'st_basal_area' in stand_data and pd.notna(stand_data['st_basal_area']):
                self.df['basal_area'] = stand_data['st_basal_area']
            if 'st_density' in stand_data and pd.notna(stand_data['st_density']):
                self.df['tree_density'] = stand_data['st_density']
            if 'st_height' in stand_data and pd.notna(stand_data['st_height']):
                self.df['stand_height'] = stand_data['st_height']
            if 'st_lai' in stand_data and pd.notna(stand_data['st_lai']):
                self.df['leaf_area_index'] = stand_data['st_lai']
            
            # Soil characteristics
            if 'st_clay_perc' in stand_data and pd.notna(stand_data['st_clay_perc']):
                self.df['clay_percentage'] = stand_data['st_clay_perc']
            if 'st_sand_perc' in stand_data and pd.notna(stand_data['st_sand_perc']):
                self.df['sand_percentage'] = stand_data['st_sand_perc']
            if 'st_silt_perc' in stand_data and pd.notna(stand_data['st_silt_perc']):
                self.df['silt_percentage'] = stand_data['st_silt_perc']
            if 'st_soil_depth' in stand_data and pd.notna(stand_data['st_soil_depth']):
                self.df['soil_depth'] = stand_data['st_soil_depth']
            if 'st_USDA_soil_texture' in stand_data and pd.notna(stand_data['st_USDA_soil_texture']):
                self.df['soil_texture'] = stand_data['st_USDA_soil_texture']
            
            # Terrain and management
            if 'st_terrain' in stand_data and pd.notna(stand_data['st_terrain']):
                self.df['terrain'] = stand_data['st_terrain']
            if 'st_growth_condition' in stand_data and pd.notna(stand_data['st_growth_condition']):
                self.df['growth_condition'] = stand_data['st_growth_condition']
        
        # Species-level features
        if 'species' in metadata:
            species_data = metadata['species']
            
            # Species characteristics
            if 'sp_name' in species_data and pd.notna(species_data['sp_name']):
                self.df['species_name'] = species_data['sp_name']
            if 'sp_leaf_habit' in species_data and pd.notna(species_data['sp_leaf_habit']):
                self.df['leaf_habit'] = species_data['sp_leaf_habit']
            if 'sp_ntrees' in species_data and pd.notna(species_data['sp_ntrees']):
                self.df['n_trees'] = species_data['sp_ntrees']
        
        # Environmental metadata features
        if 'environmental' in metadata:
            env_data = metadata['environmental']
            
            # Measurement protocol features
            if 'env_timestep' in env_data and pd.notna(env_data['env_timestep']):
                self.df['measurement_timestep'] = env_data['env_timestep']
            if 'env_time_zone' in env_data and pd.notna(env_data['env_time_zone']):
                self.df['timezone'] = env_data['env_time_zone']
        
        # Plant metadata features - ULTRA-OPTIMIZED VECTORIZED IMPLEMENTATION
        if 'plants' in metadata and 'plant_id' in self.df.columns:
            plants_data = metadata['plants']
            
            # Ultra-fast vectorized approach using single merge operation
            if 'pl_code' in plants_data.columns:
                # Filter out rows with missing pl_code and prepare for merge
                plants_clean = plants_data.dropna(subset=['pl_code'])  # dropna() already returns a copy
                
                if not plants_clean.empty:
                    # Define essential plant columns to merge
                    essential_plant_cols = ['pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 
                                          'pl_bark_thick', 'pl_social', 'pl_species',
                                          'pl_sapw_area', 'pl_sapw_depth']
                    
                    # Select only the columns we need for merge (plus pl_code)
                    merge_cols = ['pl_code'] + [col for col in essential_plant_cols if col in plants_clean.columns]
                    plants_for_merge = plants_clean[merge_cols]  # Column selection doesn't need copy
                    
                    # OPTIMIZED VECTORIZED MERGE - Enhanced performance with direct assignment
                    # Use more efficient merge strategy without temporary DataFrame creation
                    
                    # OPTIMIZATION: Direct merge on main DataFrame index for better performance
                    # Set plant_id as temporary index for faster merge
                    df_indexed = self.df.set_index('plant_id', drop=False)
                    plants_indexed = plants_for_merge.set_index('pl_code')
                    
                    # Perform vectorized merge using index alignment (faster than explicit merge)
                    for col_name in essential_plant_cols:
                        if col_name in plants_indexed.columns:
                            # Use index-based assignment for maximum speed
                            self.df[col_name] = df_indexed.index.map(plants_indexed[col_name]).fillna(np.nan)
                        else:
                            # Column doesn't exist in plants data
                            self.df[col_name] = np.nan
                    
                    # Clean up temporary objects
                    del df_indexed, plants_indexed
                else:
                    # No valid plant data - set all plant columns to NaN
                    essential_plant_cols = ['pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 
                                          'pl_bark_thick', 'pl_social', 'pl_species',
                                          'pl_sapw_area', 'pl_sapw_depth']
                    for col_name in essential_plant_cols:
                        self.df[col_name] = np.nan
        
        # Update column cache after metadata is loaded (critical for Köppen-Geiger)
        self._update_column_cache()
        
        # Create derived metadata features
        self._create_derived_metadata_features()
    
    def _create_derived_metadata_features(self):
        """Create derived features from metadata"""
        # Absolute latitude (useful for climate modeling)
        if 'latitude' in self.df.columns:
            self.df['latitude_abs'] = abs(self.df['latitude'])
        
        # Köppen-Geiger classification (vectorized for performance)
        if (self._column_exists_cache['latitude'] and 
            self._column_exists_cache['mean_annual_temp'] and 
            self._column_exists_cache['mean_annual_precip']):
            self.df['koppen_geiger_code'] = self._classify_koppen_geiger_vectorized()
        
        # Aridity index (updated: provide PET-based proxy in addition to legacy)
        if (self._column_exists_cache['mean_annual_temp'] and 
            self._column_exists_cache['mean_annual_precip']):
            # Legacy simple proxy retained for backward compatibility
            self.df['aridity_index_legacy'] = self.df['mean_annual_precip'] / (self.df['mean_annual_temp'] + 10)

        # Compute extraterrestrial radiation (daily, FAO-56) and PET (Oudin) proxies if inputs available
        try:
            compute_ra = self._is_enabled('radiation_budget_features', False) or self._is_enabled('pet_features', False)
            if compute_ra and 'latitude' in self.df.columns and 'day_of_year' in self._temporal_cache:
                # Vectorized Ra (MJ m^-2 day^-1)
                lat_rad = np.deg2rad(self.df['latitude'].astype(float))
                J = self._temporal_cache['day_of_year']
                dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * J / 365.0)
                delta = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)
                omega_s = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1.0, 1.0))
                G_sc = 0.0820  # MJ m^-2 min^-1
                # 24*60/pi * G_sc * dr * [omega_s*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(omega_s)]
                Ra = (24.0 * 60.0 / np.pi) * G_sc * dr * (
                    omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
                )
                if self._is_enabled('radiation_budget_features', False) or self._is_enabled('faof56_support_terms', False):
                    self.df['ext_rad_fao56'] = Ra
                    # Daylight hours N = 24/pi * omega_s
                    N = (24.0 / np.pi) * omega_s
                    self.df['daylight_hours'] = N

                # PET Oudin (mm/day): PET = (Ra/lambda) * (T + 5)/100 if T > -5 else 0; lambda=2.45 MJ/kg
                if self._is_enabled('pet_features', False) and 'ta' in self.df.columns:
                    T = pd.to_numeric(self.df['ta'], errors='coerce')
                    lambda_mj = 2.45
                    pet = (Ra / lambda_mj) * ((T + 5.0) / 100.0)
                    pet = pet.where(T > -5.0, 0.0)
                    self.df['pet_oudin_mm_day'] = pet

                # Additional FAO-56 support variables where possible
                # Saturation vapor pressure and slope at air temperature
                if self._is_enabled('faof56_support_terms', False) and 'ta' in self.df.columns:
                    es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
                    delta = (4098.0 * es) / ((T + 237.3) ** 2)
                    self.df['sat_vapor_pressure_kpa'] = es
                    self.df['slope_vapor_pressure_curve'] = delta
                # Air pressure and psychrometric constant if elevation available
                if self._is_enabled('faof56_support_terms', False) and 'elevation' in self.df.columns:
                    z = pd.to_numeric(self.df['elevation'], errors='coerce')
                    P = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
                    gamma = 0.000665 * P
                    self.df['air_pressure_kpa'] = P
                    self.df['psychrometric_constant'] = gamma
        except Exception:
            # If any issue arises, skip without failing pipeline
            pass

        # Net radiation per step (approx. FAO-56 scaling). Produces MJ m^-2 per timestep.
        try:
            if not self._is_enabled('radiation_budget_features', False):
                raise Exception('radiation budget disabled')
            have_sw_col = 'sw_in' in self.df.columns
            if have_sw_col or ('ta' in self.df.columns and 'ext_rad_fao56' in self.df.columns):
                # Time step (hours)
                if 'measurement_timestep' in self.df.columns and pd.notna(self.df['measurement_timestep']).any():
                    dt_hours = pd.to_numeric(self.df['measurement_timestep'], errors='coerce') / 60.0
                    dt_hours = dt_hours.fillna(1.0)
                else:
                    dt_hours = pd.Series(1.0, index=self.df.index)

                # Shortwave incoming per step in MJ m^-2 (measured if available)
                if have_sw_col:
                    Rs_step = pd.to_numeric(self.df['sw_in'], errors='coerce') * (dt_hours * 3600.0) / 1e6
                else:
                    Rs_step = pd.Series(np.nan, index=self.df.index)

                # Clear-sky shortwave per step using Ra and elevation
                if 'ext_rad_fao56' in self.df.columns:
                    Ra_day = self.df['ext_rad_fao56']  # MJ m^-2 day^-1
                    N = self.df.get('daylight_hours', pd.Series(12.0, index=self.df.index))
                    Ra_per_hour = Ra_day / N  # MJ m^-2 h^-1 during daylight
                    # Scale per time step (can be <1h)
                    Ra_step = Ra_per_hour * dt_hours
                    if 'elevation' in self.df.columns:
                        z = pd.to_numeric(self.df['elevation'], errors='coerce').fillna(0.0)
                    else:
                        z = 0.0
                    Rso_step = (0.75 + 2e-5 * z) * Ra_step
                else:
                    Rso_step = Rs_step.replace(0, np.nan)

                # Fallback Rs estimation (Hargreaves–Samani) if measured sw_in missing
                # Rs_day = kRs * sqrt(Tmax - Tmin) * Ra_day; distribute across daylight hours
                if self._is_enabled('rs_fallback_estimation', False) and Rs_step.isna().any() and 'ta' in self.df.columns and 'year' in self._temporal_cache and 'day_of_year' in self._temporal_cache and 'ext_rad_fao56' in self.df.columns:
                    year_series = pd.Series(self._temporal_cache['year'], index=self.df.index)
                    doy_series = pd.Series(self._temporal_cache['day_of_year'], index=self.df.index)
                    T_num = pd.to_numeric(self.df['ta'], errors='coerce')
                    daily = pd.DataFrame({'year': year_series, 'doy': doy_series, 'ta': T_num, 'Ra_day': Ra_day, 'N': N})
                    agg = daily.groupby(['year', 'doy'], sort=False).agg(
                        Tmax=('ta', 'max'), Tmin=('ta', 'min'), Ra=('Ra_day', 'mean'), N=('N', 'mean')
                    ).reset_index()
                    agg['dtr'] = (agg['Tmax'] - agg['Tmin']).clip(lower=0.0)
                    kRs = 0.16  # inland default
                    agg['Rs_day_est'] = kRs * np.sqrt(agg['dtr']) * agg['Ra']
                    # Map back to rows
                    key = list(zip(year_series, doy_series))
                    key_df = pd.Series(key, index=self.df.index)
                    map_dict_Rs = { (r['year'], r['doy']): r['Rs_day_est'] for _, r in agg.iterrows() }
                    map_dict_N  = { (r['year'], r['doy']): max(0.1, r['N']) for _, r in agg.iterrows() }
                    Rs_day_est_series = key_df.map(map_dict_Rs)
                    N_series = key_df.map(map_dict_N)
                    # Distribute across daylight hours only
                    is_daylight = self.df.get('is_daylight', (pd.Series(self._temporal_cache.get('hour', 12), index=self.df.index).between(6, 18)).astype(int))
                    Rs_step_est = (Rs_day_est_series / N_series) * dt_hours * is_daylight
                    # Fill missing measured Rs_step with estimate
                    Rs_step = Rs_step.fillna(Rs_step_est)

                # Net shortwave
                albedo = 0.23  # FAO-56 reference; could be refined by land cover
                Rns_step = (1.0 - albedo) * Rs_step

                # Longwave
                T = pd.to_numeric(self.df.get('ta', pd.Series(np.nan, index=self.df.index)), errors='coerce')
                T_k = T + 273.16
                es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
                RH = pd.to_numeric(self.df.get('rh', pd.Series(np.nan, index=self.df.index)), errors='coerce')
                ea = (RH / 100.0) * es
                # Scale Stefan-Boltzmann daily constant by dt_hours/24
                sigma_daily = 4.903e-9  # MJ K^-4 m^-2 day^-1
                sigma_scaled = sigma_daily * (dt_hours / 24.0)
                cloud_term = (1.35 * (Rs_step / (Rso_step.replace(0, np.nan))) - 0.35).clip(lower=0.0)
                emissivity_term = (0.34 - 0.14 * np.sqrt(ea.clip(lower=0.0)))
                Rnl_step = sigma_scaled * (T_k ** 4) * emissivity_term * cloud_term
                # Net radiation
                self.df['net_shortwave_radiation'] = Rns_step
                self.df['net_longwave_radiation'] = Rnl_step
                self.df['net_radiation'] = Rns_step - Rnl_step
        except Exception:
            pass
        
        # Categorical encodings
        if self._column_exists_cache['leaf_habit']:
            self.df['leaf_habit_code'] = self.df['leaf_habit'].map(LEAF_HABIT_MAP).astype('float64')
        
        if self._column_exists_cache['biome']:
            self.df['biome_code'] = self.df['biome'].map(BIOME_MAP).astype('float64')
        
        if self._column_exists_cache['igbp_class']:
            self.df['igbp_code'] = self.df['igbp_class'].map(IGBP_MAP).astype('float64')
        
        # Tree size and age classes
        if 'pl_dbh' in self.df.columns:
            self.df['tree_size_class'] = pd.cut(
                self.df['pl_dbh'], 
                bins=[0, 10, 30, 50, 100, 1000], 
                labels=['Sapling', 'Small', 'Medium', 'Large', 'Very Large']
            )
        
        if 'pl_age' in self.df.columns:
            self.df['tree_age_class'] = pd.cut(
                self.df['pl_age'], 
                bins=[0, 20, 50, 100, 200, 1000], 
                labels=['Young', 'Mature', 'Old', 'Very Old', 'Ancient']
            )
        
        # Social status encoding
        if 'pl_social' in self.df.columns:
            social_map = {'dominant': 3, 'codominant': 2, 'intermediate': 1, 'suppressed': 0}
            self.df['social_status_code'] = self.df['pl_social'].map(social_map).astype('float64')
        
        # Derived ratios and indices
        if 'pl_sapw_area' in self.df.columns and 'pl_leaf_area' in self.df.columns:
            self.df['sapwood_leaf_ratio'] = self.df['pl_sapw_area'] / (self.df['pl_leaf_area'] + 1e-6)
        
        if 'pl_dbh' in self.df.columns and 'pl_height' in self.df.columns:
            self.df['tree_volume_index'] = (self.df['pl_dbh'] ** 2) * self.df['pl_height']
        
        # Timezone and measurement features
        if 'timezone' in self.df.columns:
            timezone_col = self.df['timezone'].astype(str)
            self.df['timezone_offset'] = timezone_col.str.extract(r'([+-]\d{2})').astype(float)
        
        if 'measurement_timestep' in self.df.columns:
            self.df['measurement_frequency'] = 60 / self.df['measurement_timestep']
    
    def _create_domain_specific_features(self):
        """Create domain-specific features for transpiration modeling - ALWAYS CREATE ALL FEATURES"""
        print("    🔬 Creating Domain Features...")
        
        # Get all environmental data with np.nan for missing values
        ppfd_in = self._get_column('ppfd_in', np.nan)
        sw_in = self._get_column('sw_in', np.nan)
        ta = self._get_column('ta', np.nan)
        vpd = self._get_column('vpd', np.nan)
        ws = self._get_column('ws', np.nan)
        ext_rad = self._get_column('ext_rad', np.nan)
        pl_dbh = self._get_column('pl_dbh', np.nan)
        pl_sapw_area = self._get_column('pl_sapw_area', np.nan)
        pl_leaf_area = self._get_column('pl_leaf_area', np.nan)
        
        # Light features - handle missing data appropriately
        self.df['ppfd_efficiency'] = ppfd_in / (sw_in + 1e-6)
        
        # Temperature features - ALWAYS CREATE
        ta = self._get_column('ta', np.nan)
        ta_median = ta.median() if not ta.isna().all() else 25.0  # Data-driven fallback
        self.df['temp_deviation'] = abs(ta - ta_median)
        
        # Physiological features
        self.df['stomatal_conductance_proxy'] = ppfd_in / (vpd + 1e-6)
            
        # Wind effects - ALWAYS CREATE
        ws = self._get_column('ws', np.nan)
        ws_max = ws.max() if len(ws) > 0 and ws.max() > 0 else 1.0
        self.df['wind_stress'] = ws / (ws_max + 1e-6)
        self.df['wind_vpd_interaction'] = ws * vpd
        
        # Enhanced interactions with extraterrestrial radiation - ALWAYS CREATE
        ext_rad = self._get_column('ext_rad', np.nan)
        self.df['stomatal_control_index'] = vpd * ppfd_in * ext_rad
        self.df['light_efficiency'] = ppfd_in / (ext_rad + 1e-6)
        
        # Tree-specific features - ALWAYS CREATE
        pl_dbh = self._get_column('pl_dbh', np.nan)
        self.df['tree_size_factor'] = np.log(pl_dbh + 1)
        
        pl_sapw_area = self._get_column('pl_sapw_area', np.nan)
        pl_leaf_area = self._get_column('pl_leaf_area', np.nan)
        self.df['sapwood_leaf_ratio'] = pl_sapw_area / (pl_leaf_area + 1e-6)
        self.df['transpiration_capacity'] = pl_sapw_area * ppfd_in / (vpd + 1e-6)
    
    def _create_seasonality_features(self):
        """
        Create seasonal temperature and precipitation range features for ecosystem clustering.
        These are site-level features that capture intra-annual variability.
        """
        print("    🌸 Creating Seasonality Features...")
        
        # Check if feature is enabled (if config system is available)
        try:
            # Check if seasonality features are enabled using standardized config access
            seasonality_enabled = True  # Default
            if hasattr(self, 'config') and self.config:
                if hasattr(self.config, 'get_feature_setting'):
                    seasonality_enabled = self.config.get_feature_setting('seasonality_features')
                else:
                    seasonality_enabled = self.config.get('seasonality_features', True)
            
            if not seasonality_enabled:
                return
        except:
            pass  # If no config system, continue creating features
        
        # ALWAYS CREATE seasonal features for consistency across all sites
        if 'site' not in self.df.columns:
            # No site column - use default values
            self.df['seasonal_temp_range'] = 0.0
            self.df['seasonal_precip_range'] = 0.0
            return
        
        # Check if we have the required data columns
        required_cols = ['month', 'ta', 'precip']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            # Missing required columns - use default values but still create features
            self.df['seasonal_temp_range'] = 0.0
            self.df['seasonal_precip_range'] = 0.0
            return
        
        # OPTIMIZED GROUPBY AGGREGATION WITH IMPROVED MEMORY EFFICIENCY
        # Enhanced two-step approach with better memory management and caching
        
        try:
            # OPTIMIZATION 1: Use more efficient aggregation with named aggregators
            # Create monthly aggregations with memory-efficient operations
            monthly_agg_dict = {
                'ta': 'mean',
                'precip': 'mean'
            }
            
            # Single groupby with optimized column selection to reduce memory
            monthly_stats = (self.df[['site', 'month', 'ta', 'precip']]
                           .groupby(['site', 'month'], sort=False)  # sort=False for performance
                           .agg(monthly_agg_dict)
                           .reset_index())
            
            # OPTIMIZATION 2: Use named aggregations for better performance
            site_agg_dict = {
                'ta_min': ('ta', 'min'),
                'ta_max': ('ta', 'max'), 
                'precip_min': ('precip', 'min'),
                'precip_max': ('precip', 'max')
            }
            
            # More efficient aggregation with named columns
            site_stats = (monthly_stats.groupby('site', sort=False)
                         .agg(**site_agg_dict)
                         .reset_index())
            
            # OPTIMIZATION 3: Vectorized range calculations (avoid intermediate column creation)
            seasonal_ranges = pd.DataFrame({
                'site': site_stats['site'],
                'seasonal_temp_range': site_stats['ta_max'] - site_stats['ta_min'],
                'seasonal_precip_range': site_stats['precip_max'] - site_stats['precip_min']
            })
            
            # Memory-efficient merge using optimized join
            self.df = self.df.merge(seasonal_ranges, on='site', how='left', copy=False)
            
        except Exception:
            # Fallback: Create default seasonality features for all sites
            unique_sites = self.df['site'].unique()
            seasonal_ranges = pd.DataFrame({
                'site': unique_sites,
                'seasonal_temp_range': 0.0,
                'seasonal_precip_range': 0.0
            })
            self.df = self.df.merge(seasonal_ranges, on='site', how='left')
        
        # OPTIMIZATION: More efficient fillna operation using loc indexer
        # Fill any remaining NaN values with 0 (optimized vectorized assignment)
        seasonal_cols = ['seasonal_temp_range', 'seasonal_precip_range']
        self.df.loc[:, seasonal_cols] = self.df.loc[:, seasonal_cols].fillna(0.0)
    
    def _encode_categorical_features(self):
        """
        Intelligently handle object columns - preserve continuous data, encode ONLY ecological categorical data
        
        CRITICAL: Prevents site identity memorization and geographic proxy overfitting by:
        1. Blacklisting identity features that cause overfitting
        2. Warning about geographic proxy features
        3. Only encoding ecologically meaningful categorical variables
        """
        logger.phase_start("Encoding Features")
        
        # STEP 1: Remove blacklisted features first (universal removal regardless of data type)
        blacklisted_to_remove = [col for col in IDENTITY_BLACKLIST if col in self.df.columns]
        if blacklisted_to_remove:
            self.df = self.df.drop(columns=blacklisted_to_remove)
            print(f"    🚫 Removing {len(blacklisted_to_remove)} blacklisted features: {blacklisted_to_remove}")
        
        # Initialize tracking lists at the beginning
        encoded_features = []
        skipped_identity_features = []
        skipped_pure_geographic_features = []
        allowed_climate_geographic_features = []
        dropped_features = []
        
        # Use imported encoding mappings for approved ecological categorical variables
        encodings = CATEGORICAL_ENCODINGS
        
        # 🌿 Process species names into functional groups - VECTORIZED IMPLEMENTATION
        if 'species_name' in self.df.columns:
            # Ultra-fast vectorized species classification
            species_col = self.df['species_name'].fillna('').astype(str).str.strip()
            
            # Step 1: Try exact match first using vectorized map
            functional_groups = species_col.map(SPECIES_FUNCTIONAL_GROUPS)
            
            # Step 2: For unmatched species, try genus-level matching (vectorized)
            unmatched_mask = functional_groups.isna()
            if unmatched_mask.any():
                # Extract genus (first word) using vectorized string operations
                genus_series = species_col[unmatched_mask].str.split().str[0]
                genus_matches = genus_series.map(SPECIES_FUNCTIONAL_GROUPS)
                functional_groups.loc[unmatched_mask] = genus_matches
            
            # Step 3: Fill remaining unmatched with 'unknown'
            functional_groups = functional_groups.fillna('unknown')
            
            # Single assignment
            self.df['species_functional_group'] = functional_groups
            self.df = self.df.drop('species_name', axis=1)  # Remove original species column
            encoded_features.append('species_name')
        
        # Explicitly encode Köppen-Geiger climate codes
        if 'koppen_geiger_code' in self.df.columns:
            # Use factorize for robust integer encoding of string categories
            self.df['koppen_geiger_code_encoded'], unique_koppen_codes = pd.factorize(self.df['koppen_geiger_code'])
            self.df = self.df.drop('koppen_geiger_code', axis=1)
            encoded_features.append('koppen_geiger_code')
        
        # Encode approved ecological categorical variables
        for col, mapping in encodings.items():
            if col in self.df.columns:
                self.df[f'{col}_code'] = self.df[col].map(mapping)
                # Drop original text column
                self.df = self.df.drop(col, axis=1)
                encoded_features.append(col)
        
        # REMOVED: Geographic proxy features from automatic encoding
        # These are now handled with warnings to prevent overfitting
        
        # Smart handling of remaining object columns
        text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        
        # Define columns that should be preserved as-is (not encoded or dropped)
        preserve_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'plant_id', 'sap_flow', 'site']
        
        # Define columns that should be converted to numeric (continuous data)
        numeric_cols = ['latitude', 'longitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
                       'stand_age', 'basal_area', 'tree_density', 'stand_height', 'leaf_area_index',
                       'clay_percentage', 'sand_percentage', 'silt_percentage', 'soil_depth',
                       'pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 'pl_bark_thick',
                       'pl_sapw_area', 'pl_sapw_depth', 'measurement_timestep', 'swc_shallow_depth',
                       'timezone_offset', 'measurement_frequency']
        
        # Process each object column with enhanced safety checks
        for col in text_columns:
            if col in preserve_cols:
                # Keep these columns as-is
                continue
                
            elif col in IDENTITY_BLACKLIST:
                # 🚨 CRITICAL: Skip identity features that cause overfitting
                self.df = self.df.drop(col, axis=1)
                skipped_identity_features.append(col)
                continue
                
            elif col in PURE_GEOGRAPHIC_IDENTIFIERS:
                # ⚠️ Skip pure geographic identifiers that hinder generalization
                self.df = self.df.drop(col, axis=1)
                skipped_pure_geographic_features.append(col)
                continue
                
            elif col in CLIMATE_GEOGRAPHIC_FEATURES:
                # ✅ Allow climate-based geographic features (ecological information)
                allowed_climate_geographic_features.append(col)
                # Continue to normal categorical processing
                pass
                
            elif col in numeric_cols:
                # Try to convert to numeric (continuous data)
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass
                    
            else:
                # Check if this looks like categorical data
                unique_values = self.df[col].unique()
                unique_count = len(unique_values)
                
                # Enhanced safety check: Look for patterns that suggest identity features
                col_lower = col.lower()
                is_likely_identity = any(pattern in col_lower for pattern in [
                    'name', 'code', 'id', 'identifier', 'paper', 'reference', 'study'
                ])
                
                if is_likely_identity and unique_count > 5:
                    # Likely an identity feature we missed - skip it
                    self.df = self.df.drop(col, axis=1)
                    skipped_identity_features.append(col)
                    continue
                
                # Conservative approach: Only encode if clearly ecological and low cardinality
                if unique_count <= 10 and not is_likely_identity:
                    # Very conservative encoding for small categorical variables
                    sample_values = unique_values[:5].tolist()  # More efficient
                    
                    # Only encode if values look clearly categorical (not numeric IDs)
                    if not all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values if pd.notna(val)):
                        encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                        self.df[f'{col}_code'] = self.df[col].map(encoding_map).astype('float64')
                        self.df = self.df.drop(col, axis=1)
                        encoded_features.append(col)
                    else:
                        # Looks like numeric IDs - drop it
                        self.df = self.df.drop(col, axis=1)
                        skipped_identity_features.append(col)
                        
                elif unique_count <= 50:
                    # Medium cardinality - be very conservative
                    # Check if values look like categories or IDs
                    sample_values = self.df[col].dropna().head(10)
                    if all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values):
                        # Looks like numeric IDs - drop it
                        self.df = self.df.drop(col, axis=1)
                        skipped_identity_features.append(col)
                    else:
                        # Could be ecological categorical but high cardinality is risky
                        self.df = self.df.drop(col, axis=1)
                        dropped_features.append(col)
                else:
                    # High cardinality - try to convert to numeric or drop
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    except:
                        # Drop high cardinality text columns
                        self.df = self.df.drop(col, axis=1)
                        dropped_features.append(col)
        
        # Summary logging (reduced verbosity)
        if encoded_features:
            logger.phase_complete("Encoding Features", f"{len(encoded_features)} features encoded, {len(skipped_identity_features) + len(skipped_pure_geographic_features)} blocked, {len(dropped_features)} dropped")
    
    def _drop_problematic_columns(self):
        """Drop columns that could cause issues in modeling"""
        # Problematic columns are now handled by key phase logging in _ensure_consistent_schema
        
        # Drop columns that are likely to cause overfitting or issues
        problematic_patterns = [
            'TIMESTAMP',  # Keep only the main timestamp
            '_flag',      # Quality flags
            '_qc',        # Quality control
            '_uncertainty', # Uncertainty measures
            'site_id',    # Site identifiers
            'plot_id',    # Plot identifiers
            'tree_id',    # Tree identifiers
        ]
        
        columns_to_drop = []
        for col in self.df.columns:
            col_lower = col.lower()
            for pattern in problematic_patterns:
                if pattern in col_lower and col != 'TIMESTAMP':
                    columns_to_drop.append(col)
                    break
        
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)
    
    def _ensure_consistent_schema(self):
        """Ensure consistent schema across all sites by adding missing columns with NA values"""
        logger.phase_start("Ensuring Data Compatibility")
        
        # Remove any problematic columns that might have been missed using centralized list
        problematic_columns = PROBLEMATIC_COLUMNS_TO_EXCLUDE
        
        # Ensure inconsistent columns are properly handled for XGBoost
        # These columns may be missing in some files but should be preserved as NaN
        xgboost_missing_columns = ['leaf_habit_code', 'soil_depth']
        for col in problematic_columns:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
                logger.warning(f"Removed problematic column from final schema: {col}", indent=2)
        
        # Define the base expected schema (always included)
        expected_columns = {
            # Core environmental variables (should exist in all sites)
            'ta': np.nan, 'rh': np.nan, 'vpd': np.nan, 'sw_in': np.nan, 'ws': np.nan, 
            'precip': np.nan, 'swc_shallow': np.nan, 'ppfd_in': np.nan,
            
            # Important soil texture features (useful when available)
            'clay_percentage': np.nan, 'sand_percentage': np.nan, 'silt_percentage': np.nan,
            
            # Important metadata features (useful when available)
            'latitude': np.nan, 'longitude': np.nan, 'elevation': np.nan,
            'mean_annual_temp': np.nan, 'mean_annual_precip': np.nan,
            'stand_age': np.nan, 'basal_area': np.nan, 'tree_density': np.nan,
            'stand_height': np.nan, 'leaf_area_index': np.nan,
            
            # Important plant features (useful when available)
            'pl_age': np.nan, 'pl_dbh': np.nan, 'pl_height': np.nan, 'pl_leaf_area': np.nan,
            'pl_sapw_area': np.nan, 'pl_sapw_depth': np.nan,
        }
        
        # Conditionally add temporal features if enabled
        if self.config.get('temporal_features', False):
            temporal_features = {
                # Solar timestamp features
                'solar_hour': np.nan, 'solar_day_of_year': np.nan, 'solar_hour_sin': np.nan,
                'solar_hour_cos': np.nan, 'solar_day_sin': np.nan, 'solar_day_cos': np.nan,
                
                # Enhanced temporal features
                'hour_sin': np.nan, 'hour_cos': np.nan, 'day_sin': np.nan, 'day_cos': np.nan,
                'month_sin': np.nan, 'month_cos': np.nan,
                'is_morning': np.nan, 'is_afternoon': np.nan, 'is_night': np.nan,
                'is_spring': np.nan, 'is_summer': np.nan, 'is_autumn': np.nan, 'is_winter': np.nan,
                'hours_since_sunrise': np.nan, 'hours_since_sunset': np.nan,
            }
            expected_columns.update(temporal_features)
        
        # Conditionally add interaction features if enabled
        if self.config.get('interaction_features', False):
            interaction_features = {
                'vpd_ppfd_interaction': np.nan, 'vpd_ta_interaction': np.nan, 'temp_humidity_ratio': np.nan,
                'water_stress_index': np.nan, 'light_efficiency': np.nan, 'temp_soil_interaction': np.nan,
                'wind_vpd_interaction': np.nan, 'radiation_temp_interaction': np.nan, 'humidity_soil_interaction': np.nan,
            }
            expected_columns.update(interaction_features)
        
        # Add missing columns with NA values
        for col, default_value in expected_columns.items():
            if col not in self.df.columns:
                self.df[col] = default_value
        
        # Add XGBoost-inconsistent columns if missing
        for col in xgboost_missing_columns:
            if col not in self.df.columns:
                self.df[col] = np.nan
                logger.warning(f"Added missing XGBoost column: {col}", indent=2)
            else:
                # Check if any changes are needed before making them
                needs_fixing = False
                
                # Ensure any invalid values are converted to NaN for XGBoost
                if self.df[col].dtype == 'object':
                    # For categorical columns, replace empty strings or invalid values with NaN
                    invalid_mask = self.df[col].isin(['', 'nan', 'None', 'NULL'])
                    if invalid_mask.any():
                        self.df[col] = self.df[col].replace(['', 'nan', 'None', 'NULL'], np.nan)
                        needs_fixing = True
                else:
                    # For numeric columns, ensure inf/-inf are converted to NaN
                    invalid_mask = np.isinf(self.df[col])
                    if invalid_mask.any():
                        self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                        needs_fixing = True
                
                # Only log message if actual changes were made
                if needs_fixing:
                    logger.warning(f"Fixed XGBoost compatibility for: {col}", indent=2)
        
        logger.phase_complete("Ensuring Data Compatibility", f"{len(self.df.columns)} columns finalized")
    
    def _classify_koppen_geiger_vectorized(self):
        """
        Vectorized Köppen-Geiger climate classification for better performance.
        Simplified version that's much faster than row-by-row apply().
        """
        temp = self.df['mean_annual_temp'].fillna(15.0)  # Fill missing with moderate temperature
        precip = self.df['mean_annual_precip'].fillna(1000.0)  # Fill missing with moderate precipitation
        lat = self.df['latitude'].fillna(0.0)  # Fill missing with equatorial
        
        # Vectorized classification using numpy.where
        climate = np.where(
            temp > 18,  # Tropical
            np.where(precip > 2000, 'Af', 'Aw'),  # Rainforest vs Savanna
            np.where(
                temp > 0,  # Temperate
                np.where(precip > 1000, 'Cf', 'Cs'),  # Oceanic vs Mediterranean
                np.where(temp > -3, 'Df', 'ET')  # Continental vs Tundra
            )
        )
        
        return climate
    
    def _classify_koppen_geiger(self, row):
        """
        Classify Köppen-Geiger climate zone based on latitude, mean annual temperature, and precipitation.
        
        This implementation provides ecological encoding that captures climate patterns
        without overfitting to specific site identities, addressing the site imbalance
        issue in spatial generalization.
        
        Parameters:
        -----------
        row : pandas.Series
            Row containing 'latitude', 'mean_annual_temp', 'mean_annual_precip'
            
        Returns:
        --------
        str : Köppen-Geiger climate code (e.g., 'Af', 'BWk', 'Cfb', 'Dfb')
        """
        try:
            lat = row['latitude']
            temp = row['mean_annual_temp']  # Annual mean temperature in °C
            precip = row['mean_annual_precip']  # Annual precipitation in mm
            
            # Handle missing values
            if pd.isna(lat) or pd.isna(temp) or pd.isna(precip):
                return 'Unknown'
            
            # Determine hemisphere for seasonal logic
            is_northern = lat >= 0
            
            # For simplification, estimate seasonal patterns based on latitude and hemisphere
            # In reality, this would use monthly data, but we approximate from annual values
            
            # Estimate seasonal temperature variation (larger variation at higher latitudes)
            temp_seasonal_range = min(30, abs(lat) * 0.8)  # Conservative estimate
            
            # Estimate seasonal precipitation patterns
            # We'll make simplified assumptions about precipitation seasonality
            summer_precip_ratio = 0.4 + 0.2 * (1 - abs(lat) / 90)  # More summer rain near equator
            winter_precip_ratio = 1 - summer_precip_ratio
            
            summer_precip = precip * summer_precip_ratio
            winter_precip = precip * winter_precip_ratio
            
            # Calculate coldest and warmest month temperatures (approximate)
            temp_coldest = temp - temp_seasonal_range / 2
            temp_warmest = temp + temp_seasonal_range / 2
            
            # Köppen-Geiger Classification Logic
            
            # A: Tropical climates (all months > 18°C)
            if temp_coldest > 18:
                if precip >= 2500:
                    return 'Af'  # Tropical rainforest
                elif precip >= 1000:
                    # Check for dry season (simplified)
                    if min(summer_precip, winter_precip) < 60:
                        if summer_precip > winter_precip:
                            return 'Aw'  # Tropical savanna, winter dry
                        else:
                            return 'As'  # Tropical savanna, summer dry
                    else:
                        return 'Am'  # Tropical monsoon
                else:
                    return 'Aw'  # Tropical savanna
            
            # B: Arid climates
            # Calculate aridity threshold
            if temp >= 0:
                aridity_threshold = 20 * temp + 280  # For predominantly summer rain
            else:
                aridity_threshold = 20 * temp + 280
            
            # Adjust for precipitation seasonality (simplified)
            if winter_precip > summer_precip * 2:  # Winter rain dominant
                aridity_threshold = 20 * temp + 140
            elif summer_precip > winter_precip * 2:  # Summer rain dominant  
                aridity_threshold = 20 * temp + 420
            
            if precip < aridity_threshold:
                if precip < aridity_threshold / 2:
                    # Desert climate (BW)
                    if temp >= 18:
                        return 'BWh'  # Hot desert
                    else:
                        return 'BWk'  # Cold desert
                else:
                    # Steppe climate (BS)
                    if temp >= 18:
                        return 'BSh'  # Hot steppe
                    else:
                        return 'BSk'  # Cold steppe
            
            # C: Temperate climates (coldest month -3°C to 18°C, warmest month > 10°C)
            if -3 <= temp_coldest <= 18 and temp_warmest > 10:
                # Determine precipitation pattern
                if winter_precip > summer_precip * 3:
                    precip_code = 's'  # Summer dry
                elif summer_precip > winter_precip * 10:
                    precip_code = 'w'  # Winter dry
                else:
                    precip_code = 'f'  # Fully humid
                
                # Determine temperature pattern
                if temp_warmest >= 22:
                    temp_code = 'a'  # Hot summer
                elif temp_warmest >= 10 and (temp_warmest - temp_coldest) >= 10:
                    temp_code = 'b'  # Warm summer
                else:
                    temp_code = 'c'  # Cool summer
                
                return f'C{precip_code}{temp_code}'
            
            # D: Continental climates (coldest month < -3°C, warmest month > 10°C)
            if temp_coldest < -3 and temp_warmest > 10:
                # Determine precipitation pattern
                if winter_precip > summer_precip * 3:
                    precip_code = 's'  # Summer dry
                elif summer_precip > winter_precip * 10:
                    precip_code = 'w'  # Winter dry
                else:
                    precip_code = 'f'  # Fully humid
                
                # Determine temperature pattern
                if temp_warmest >= 22:
                    temp_code = 'a'  # Hot summer
                elif temp_warmest >= 10 and (temp_warmest - temp_coldest) >= 10:
                    temp_code = 'b'  # Warm summer
                elif temp_coldest < -38:
                    temp_code = 'd'  # Extremely continental
                else:
                    temp_code = 'c'  # Cool summer
                
                return f'D{precip_code}{temp_code}'
            
            # E: Polar climates (warmest month < 10°C)
            if temp_warmest < 10:
                if temp_warmest > 0:
                    return 'ET'  # Tundra
                else:
                    return 'EF'  # Ice cap
            
            # Default fallback (should rarely occur)
            return 'Unknown'
            
        except Exception as e:
            # Handle any calculation errors gracefully
            return 'Unknown'
    
    def _cleanup_intermediate_data(self):
        """Clean up intermediate data to free memory"""
        # Clear caches after use
        if hasattr(self, '_temporal_cache'):
            del self._temporal_cache
        
        # Force garbage collection periodically
        self._gc_counter += 1
        if self._gc_counter % 10 == 0:  # Every 10 operations
            if self.memory_manager:
                self.memory_manager.force_memory_cleanup()
            else:
                gc.collect()