"""
SAPFLUXNET Data Processing Orchestrator

This module contains the main orchestrator class that coordinates all components
of the SAPFLUXNET data processing pipeline. It manages the workflow from site
discovery through feature engineering and file output.

Key Components Orchestrated:
- SiteAnalyzer: Site discovery and quality validation
- DataLoader: Data loading with quality flag filtering  
- MemoryManager: Adaptive memory management and optimization
- FileManager: File I/O operations and format handling
- FeatureEngineer: Complete feature engineering pipeline
- ErrorHandler: Standardized error handling
- Logger: Consistent logging and progress tracking

Main Classes:
- ProcessingConfig: Configuration management for all processing parameters
- SAPFLUXNETOrchestrator: Main orchestrator that coordinates the entire pipeline
"""

import pandas as pd
import os
from datetime import datetime
import warnings
import psutil
import gc

# Import our specialized component classes
from data_processing.MemoryManager import MemoryManager
from data_processing.FileManager import FileManager
from data_processing.SiteAnalyzer import SiteAnalyzer
from data_processing.DataLoader import DataLoader
from data_processing.FeatureEngineer import FeatureEngineer
from data_processing.logging_utils import logger
from data_processing.error_utils import ErrorHandler

warnings.filterwarnings('ignore')

class ProcessingConfig:
    """Configuration class for processing parameters - replaces hardcoded values"""
    
    # Memory thresholds (GB) - Same as v2 for compatibility
    MEMORY_THRESHOLDS = {
        'critical': 1.0,      # Critical memory situation - aggressive cleanup
        'low': 2.0,           # Low memory situation - moderate cleanup
        'moderate': 4.0,      # Moderate memory situation - light cleanup
        'streaming_low': 4.0, # Use streaming for low memory
        'streaming_medium': 6.0, # Use streaming for medium memory
        'memory_threshold': 6.0,  # Threshold for switching to aggressive memory management (all features still created)
    }
    
    # File size thresholds (MB)
    FILE_SIZE_THRESHOLDS = {
        'small': 50,          # Small files - standard processing
        'medium': 100,        # Medium files - consider streaming
        'large': 200,         # Large files - use streaming
        'very_large': 500,    # Very large files - aggressive streaming
    }
    
    # Chunk size settings - Same as v2 for compatibility
    CHUNK_SIZES = {
        'min': 10000,         # Minimum chunk size (rows)
        'max': 200000,        # Maximum chunk size (rows)
        'very_low_memory': 500,   # Very low memory chunk size
        'low_memory': 1000,       # Low memory chunk size
        'medium_memory': 1500,    # Medium memory chunk size
        'high_memory': 2000,      # High memory chunk size
        'large_dataset': 100000,  # Large dataset minimum chunk size
    }
    
    # Row limits for memory-constrained situations
    ROW_LIMITS = {
        'very_low_memory': 50000,   # Very low memory + large files
        'low_memory': 100000,       # Low memory + very large files
    }
    
    # Feature creation settings
    FEATURE_SETTINGS = {
        'max_lag_hours': 24,        # Maximum lag hours for all files
        'rolling_windows': [3, 6, 12, 24, 48, 72, 168, 336, 720],  # Enhanced rolling window sizes (added 7-day, 14-day, 30-day)
        'memory_percentage': 0.1,   # Target memory usage per chunk (10%)
        'estimated_memory_per_row_mb': 0.001,  # Estimated memory per row (1KB)
        'advanced_temporal_features': True,  # Enable advanced temporal features
        'interaction_features': True,  # Enable interaction features
        'rate_of_change_features': True,  # Enable rate of change features
        'cumulative_features': True,  # Enable cumulative features
        'seasonality_features': True,  # Enable seasonal temperature/precipitation range features
        'memory_chunking': True,    # Enable memory-based chunking
        'adaptive_chunking': True,  # Enable adaptive chunk sizing
        'parallel_processing': True,  # Enable parallel processing for feature groups
        'memory_threshold_mb': 1000,  # Memory threshold for chunking (MB)
        'max_parallel_workers': 4,  # Maximum parallel workers for feature creation
    }
    
    # I/O optimization settings
    IO_SETTINGS = {
        'buffer_size': 8192,        # File buffer size (8KB)
        'compression_level': 6,     # Gzip compression level
        'cleanup_frequency': 10,    # Memory cleanup every N chunks
    }
    
    # Quality flag settings
    QUALITY_FLAGS = {
        'bad_flags': ['OUT_WARN', 'RANGE_WARN'],  # Flags to filter out
        'min_file_size_mb': 0.2,    # Minimum file size for validation (uncompressed) - lowered from 1.0MB
        'min_file_size_compressed_mb': 0.1,  # Minimum file size for validation (compressed) - lowered from 0.3MB
    }
    
    @classmethod
    def get_memory_threshold(cls, threshold_name):
        """Get memory threshold by name"""
        return cls.MEMORY_THRESHOLDS.get(threshold_name, 4.0)
    
    @classmethod
    def get_file_size_threshold(cls, threshold_name):
        """Get file size threshold by name"""
        return cls.FILE_SIZE_THRESHOLDS.get(threshold_name, 100)
    
    @classmethod
    def get_chunk_size(cls, size_name):
        """Get chunk size by name"""
        return cls.CHUNK_SIZES.get(size_name, 1000)
    
    @classmethod
    def get_row_limit(cls, limit_name):
        """Get row limit by name"""
        return cls.ROW_LIMITS.get(limit_name, None)
    
    @classmethod
    def get_feature_setting(cls, setting_name):
        """Get feature setting by name"""
        return cls.FEATURE_SETTINGS.get(setting_name, None)
    
    @classmethod
    def get_io_setting(cls, setting_name):
        """Get I/O setting by name"""
        return cls.IO_SETTINGS.get(setting_name, None)
    
    @classmethod
    def get_quality_flag_setting(cls, setting_name):
        """Get quality flag setting by name"""
        return cls.QUALITY_FLAGS.get(setting_name, None)

class SAPFLUXNETOrchestrator:
    """Main orchestrator for SAPFLUXNET data processing pipeline - coordinates all components"""
    
    def __init__(self, output_dir='comprehensive_processed', chunk_size=1000, max_memory_gb=12, force_reprocess=False, skip_problematic_sites=True, use_quality_flags=True, compress_output=False, optimize_io=True, export_format='csv', config_overrides=None, clean_mode=False):
        # Store core configuration
        self.base_output_dir = output_dir
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.force_reprocess = force_reprocess
        self.skip_problematic_sites = skip_problematic_sites
        self.use_quality_flags = use_quality_flags
        self.compress_output = compress_output
        self.optimize_io = optimize_io
        self.export_format = export_format.lower()
        self.clean_mode = clean_mode
        
        # Apply configuration overrides if provided
        if config_overrides:
            self.apply_config_overrides(config_overrides)
        
        # Track processing statistics
        self.stats = {
            'successful_sites': 0,
            'failed_sites': 0,
            'total_rows': 0,
            'total_columns': 0,
            'memory_cleanups': 0,
            'io_operations': 0,
            'bytes_written': 0,
            'adaptive_decisions': []
        }
        
        # Initialize component classes
        logger.processing_start("SAPFLUXNET Processing Components")
        
        # File Manager (initialize first since other components depend on it)
        self.file_manager = FileManager(
            export_format=export_format,
            base_output_dir=output_dir,
            compress_output=compress_output,
            config=ProcessingConfig,
            stats=self.stats
        )
        
        # Memory Manager
        self.memory_manager = MemoryManager(
            max_memory_gb=max_memory_gb,
            config=ProcessingConfig,
            stats=self.stats,
            file_manager=self.file_manager
        )
        
        # Data Loader (initialize before SiteAnalyzer since SiteAnalyzer depends on it)
        self.data_loader = DataLoader(
            config=ProcessingConfig,
            memory_manager=self.memory_manager,
            use_quality_flags=use_quality_flags,
            file_manager=self.file_manager
        )
        
        # Site Analyzer
        self.site_analyzer = SiteAnalyzer(
            config=ProcessingConfig,
            skip_problematic_sites=skip_problematic_sites,
            clean_mode=clean_mode,
            force_reprocess=force_reprocess,
            output_dir=self.file_manager.output_dir,
            export_format=export_format,
            compress_output=compress_output,
            file_manager=self.file_manager,
            data_loader=self.data_loader
        )
        
        # Get system information for adaptive processing
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.init_system(self.system_memory_gb, self.available_memory_gb)
        logger.processing_complete("Component initialization", 5)  # 5 components initialized
        
        # Settings for adaptive processing (will be determined per site)
        self.adaptive_settings = {
            'use_streaming': None,
            'streaming_threshold_mb': 50,
            'max_lag_hours': None,
            'rolling_windows': None,
            'create_advanced_features': None,
            'create_domain_features': None,
            'chunk_size': None,
        }
    
    def apply_config_overrides(self, overrides):
        """Apply command-line overrides to configuration"""
        if overrides.get('memory_threshold'):
            ProcessingConfig.MEMORY_THRESHOLDS['memory_threshold'] = overrides['memory_threshold']
            # Also adjust streaming thresholds proportionally
            ProcessingConfig.MEMORY_THRESHOLDS['streaming_low'] = overrides['memory_threshold'] - 2
            ProcessingConfig.MEMORY_THRESHOLDS['streaming_medium'] = overrides['memory_threshold']
        
        if overrides.get('file_size_threshold'):
            ProcessingConfig.FILE_SIZE_THRESHOLDS['medium'] = overrides['file_size_threshold']
            ProcessingConfig.FILE_SIZE_THRESHOLDS['large'] = overrides['file_size_threshold'] * 2
        
        if overrides.get('chunk_size_override'):
            ProcessingConfig.CHUNK_SIZES['medium_memory'] = overrides['chunk_size_override']
            ProcessingConfig.CHUNK_SIZES['high_memory'] = overrides['chunk_size_override']
        
        if overrides.get('max_lag_hours'):
            ProcessingConfig.FEATURE_SETTINGS['max_lag_hours'] = overrides['max_lag_hours']
        
        if overrides.get('rolling_windows'):
            try:
                windows = [int(w.strip()) for w in overrides['rolling_windows'].split(',')]
                ProcessingConfig.FEATURE_SETTINGS['rolling_windows'] = windows
            except ValueError:
                print(f"⚠️  Invalid rolling windows format: {overrides['rolling_windows']}. Using default.")
        
        print(f"🔧 Applied configuration overrides: {overrides}")
    
    def orchestrate_complete_processing(self):
        """Orchestrate complete SAPFLUXNET processing pipeline using specialized component classes"""
        
        # Step 1: Analyze all sites using SiteAnalyzer
        print("🔍 Step 1: Analyzing site data quality...")
        analysis_summary = self.site_analyzer.analyze_all_sites_data_quality()
        
        print(f"\n🔍 Step 2: Finding sites to process...")
        all_sites = self.site_analyzer.get_all_sites()
        print(f"📊 Found {len(all_sites)} total sites")
        
        # Check which sites need processing vs which can be skipped
        sites_to_process, sites_to_skip = self.site_analyzer.get_processing_status(all_sites)
        
        if self.force_reprocess:
            print(f"🔄 Force reprocessing enabled - will process all valid sites")
            # Only include sites that have valid data for reprocessing
            valid_sites = [site for site in all_sites if site not in self.site_analyzer.sites_with_no_valid_data]
            sites_to_process = valid_sites
            sites_to_skip = [site for site in all_sites if site in self.site_analyzer.sites_with_no_valid_data]
        
        print(f"📋 Processing status:")
        print(f"  - Sites to process: {len(sites_to_process)}")
        print(f"  - Sites to skip: {len(sites_to_skip)}")
        
        if not sites_to_process:
            print(f"🎉 All valid sites already processed! No work to do.")
            return True
        
        print(f"\n🚀 Step 3: Starting processing of {len(sites_to_process)} sites...")
        
        successful_sites = []
        failed_sites = []
        
        for i, site in enumerate(sites_to_process, 1):
            print(f"\n[{i}/{len(sites_to_process)}] Processing {site}...")
            
            # Force memory cleanup before each site using MemoryManager
            self.memory_manager.force_memory_cleanup()
            
            # Use complete processing
            result = self.orchestrate_site_processing(site)
            
            if result is not None:
                # Save individual site file using FileManager
                file_extension = self.file_manager.get_output_file_extension()
                output_file = f'{self.file_manager.output_dir}/{site}_comprehensive{file_extension}'
                
                if isinstance(result, pd.DataFrame):
                    # Standard processing result - use FileManager
                    self.file_manager.save_dataframe_formatted(result, output_file, site)
                    
                    self.stats['total_rows'] += len(result)
                    self.stats['total_columns'] = max(self.stats['total_columns'], len(result.columns))
                    
                    # Clear memory immediately after saving
                    del result
                    self.memory_manager.check_memory_usage()
                else:
                    # Streaming processing result (already saved)
                    pass
                
                successful_sites.append(site)
                self.stats['successful_sites'] += 1
                
                # Memory cleanup after successful processing
                self.memory_manager.force_memory_cleanup()
                
            else:
                failed_sites.append(site)
                self.stats['failed_sites'] += 1
                
                # Memory cleanup after failed processing too
                self.memory_manager.force_memory_cleanup()
        
        print(f"\n{'='*60}")
        print(f"🎉 ADAPTIVE COMPLETE PROCESSING COMPLETE!")
        print(f"📊 Summary:")
        print(f"  - Total sites found: {len(all_sites)}")
        print(f"  - Sites processed: {len(sites_to_process)}")
        print(f"  - Sites skipped: {len(sites_to_skip)}")
        print(f"  - Successful: {self.stats['successful_sites']}")
        print(f"  - Failed: {self.stats['failed_sites']}")
        print(f"  - Total rows: {self.stats['total_rows']:,}")
        print(f"  - Max columns: {self.stats['total_columns']}")
        print(f"  - Memory cleanups: {self.stats['memory_cleanups']}")
        print(f"  - I/O operations: {self.stats['io_operations']}")
        print(f"  - Data written: {self.stats['bytes_written'] / (1024**2):.1f} MB")
        
        if failed_sites:
            print(f"\n❌ Failed sites:")
            for site in failed_sites:
                print(f"  - {site}")
        
        return len(successful_sites) > 0
    
    def orchestrate_site_processing(self, site):
        """Orchestrate single site processing with adaptive settings determined on-the-fly"""
        
        try:
            # Check for extremely large datasets and force streaming mode
            env_file = f'sapwood/{site}_env_data.csv'
            sapf_file = f'sapwood/{site}_sapf_data.csv'
            
            if os.path.exists(env_file) and os.path.exists(sapf_file):
                env_size_mb = os.path.getsize(env_file) / (1024**2)
                sapf_size_mb = os.path.getsize(sapf_file) / (1024**2)
                total_size_mb = env_size_mb + sapf_size_mb
                
                print(f"    📊 Dataset size: {total_size_mb:.1f}MB (env: {env_size_mb:.1f}MB, sapf: {sapf_size_mb:.1f}MB)")
                
                # Force streaming mode for very large datasets
                if total_size_mb > 50:  # If total size > 50MB, force streaming
                    print(f"    💾 Very large dataset detected - forcing streaming mode")
                    self.adaptive_settings['use_streaming'] = True
                    self.adaptive_settings['processing_mode'] = 'streaming'
                
                # Note: Previously skipped datasets > 100MB, but with abundant memory (500+ GB)
                # and streaming support, we can process all valid sites
                if total_size_mb > 100:
                    print(f"    💪 Large dataset detected ({total_size_mb:.1f}MB) - will use streaming mode")
            
            # Determine optimal settings for this site using MemoryManager
            success, self.adaptive_settings = self.memory_manager.determine_adaptive_settings(site, self.adaptive_settings)
            if not success:
                return None
            
            # Use adaptive settings
            if self.adaptive_settings['use_streaming']:
                return self._orchestrate_streaming_processing(site)
            else:
                return self._orchestrate_standard_processing(site)
                
        except Exception as e:
            print(f"  ❌ Error in adaptive processing for {site}: {str(e)}")
            return None
    
    def _orchestrate_standard_processing(self, site):
        """Orchestrate standard processing workflow using component classes and adaptive settings"""
        
        # Load metadata using FileManager
        metadata = self.file_manager.load_metadata(site)
        
        # EARLY VALIDATION: Check sap flow data using SiteAnalyzer
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        # Check if sap flow file exists
        if not os.path.exists(sapf_file):
            print(f"  ❌ Sap flow file not found: {sapf_file}")
            return None
        
        # Quick validation of sap flow data structure using SiteAnalyzer
        sapf_validation = self.site_analyzer.validate_sap_flow_data(sapf_file)
        if not sapf_validation['valid']:
            print(f"  ❌ Sap flow data validation failed: {sapf_validation['reason']}")
            return None
        
        # Now load environmental data with quality flag filtering using DataLoader
        env_data = self.data_loader.load_and_filter_data_with_flags(site, 'env')
        if env_data is None:
            return None
        
        # Find timestamp column using centralized utility
        from data_processing.DataLoader import DataLoader
        timestamp_col = DataLoader.get_primary_timestamp_column(env_data)
        
        if not timestamp_col:
            print(f"  ❌ No timestamp column found for {site}")
            return None
        
        # Load sap flow data with quality flag filtering using DataLoader
        sapf_data = self.data_loader.load_and_filter_data_with_flags(site, 'sapf')
        if sapf_data is None:
            return None
        
        # Find sap flow timestamp column using centralized utility
        sapf_timestamp_col = DataLoader.get_primary_timestamp_column(sapf_data)
        
        if not sapf_timestamp_col:
            print(f"  ❌ No timestamp column found in sap flow data for {site}")
            return None
        
        # Convert timestamps using centralized utility
        DataLoader.convert_timestamp_column(env_data, timestamp_col, inplace=True)
        DataLoader.convert_timestamp_column(sapf_data, sapf_timestamp_col, inplace=True)
        
        # Use the validated columns from earlier
        sapf_cols = sapf_validation['columns']
        
        # Optimized merging using time-based indexing with memory management
        
        # Convert timestamps to datetime for efficient merging
        env_data[timestamp_col] = pd.to_datetime(env_data[timestamp_col])
        sapf_data[sapf_timestamp_col] = pd.to_datetime(sapf_data[sapf_timestamp_col])
        
        # Set timestamp as index for faster merging
        env_data_indexed = env_data.set_index(timestamp_col)
        sapf_data_indexed = sapf_data.set_index(sapf_timestamp_col)
        
        # Clean up original dataframes immediately
        del env_data, sapf_data
        self.memory_manager.check_memory_usage()
        
        # Melt sap flow data with indexed timestamp
        sapf_long = sapf_data_indexed[sapf_cols].reset_index().melt(
            id_vars=[sapf_timestamp_col], 
            value_vars=sapf_cols,
            var_name='plant_id', 
            value_name='sap_flow'
        )
        
        # Clean up sapf_data_indexed immediately after melting
        del sapf_data_indexed
        self.memory_manager.check_memory_usage()
        
        # Merge using indexed join (more memory efficient)
        merged = sapf_long.merge(
            env_data_indexed.reset_index(), 
            left_on=sapf_timestamp_col, 
            right_on=timestamp_col, 
            how='inner'
        )
        
        # Clean up intermediate dataframes
        del sapf_long, env_data_indexed
        self.memory_manager.check_memory_usage()
        
        # Remove rows with missing sap flow data
        merged = merged.dropna(subset=['sap_flow'])
        
        if len(merged) == 0:
            print(f"  ❌ No valid data after merging for {site}")
            return None
        
        # Create features using FeatureEngineer class
        logger.performance_info("Starting feature creation", f"{len(merged):,} rows for {site}")
        
        # Initialize FeatureEngineer with the merged data, config, and MemoryManager
        # Pass None for config to use default dict config (avoids class method conflicts)
        feature_engineer = FeatureEngineer(merged, config=None, memory_manager=self.memory_manager)
        
        # Create all features using the optimized FeatureEngineer
        try:
            merged = feature_engineer.create_all_features(
                metadata=metadata, 
                timestamp_col=timestamp_col, 
                processing_mode='standard'
            )
        
            # Add site identifier
            merged['site'] = site
            
            logger.features_created(site, len(merged), len(merged.columns))
        
        except Exception as e:
            return ErrorHandler.handle_feature_creation_error("Site processing", e, site)
        
        # Clean up FeatureEngineer
        del feature_engineer
        self.memory_manager.check_memory_usage()
        
        # Features successfully created - final result logged by FeatureEngineer
        return merged
    
    def _orchestrate_streaming_processing(self, site):
        """Orchestrate streaming processing workflow using adaptive settings - no pre-loading"""
        
        # Load metadata first using FileManager
        metadata = self.file_manager.load_metadata(site)
        
        # EARLY VALIDATION: Check sap flow data before loading environmental data
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        # Check if sap flow file exists
        if not os.path.exists(sapf_file):
            print(f"  ❌ Sap flow file not found: {sapf_file}")
            return None
        
        # Quick validation of sap flow data structure
        sapf_validation = self.site_analyzer.validate_sap_flow_data(sapf_file)
        if not sapf_validation['valid']:
            print(f"  ❌ Sap flow data validation failed: {sapf_validation['reason']}")
            return None
        
        # Process in adaptive chunks
        chunk_size = self.adaptive_settings['chunk_size']
        file_extension = self.file_manager.get_output_file_extension()
        output_file = f'{self.file_manager.output_dir}/{site}_comprehensive{file_extension}'
        
        # Initialize output file
        first_chunk = True
        total_processed = 0
        
        # Stream through environmental data
        env_file = f'sapwood/{site}_env_data.csv'
        
        # Use the validated columns from earlier
        sapf_cols = sapf_validation['columns']
        
        # Get timestamp column from validation
        sapf_sample = pd.read_csv(sapf_file, nrows=100)
        sapf_timestamp_cols = [col for col in sapf_sample.columns if 'timestamp' in col.lower()]
        sapf_timestamp_col = sapf_timestamp_cols[0]
        
        # Create time-based index for efficient sap flow lookup
        sapf_time_index = {}
        
        # Stream through sap flow data to build time index (minimal memory usage)
        for sapf_chunk in pd.read_csv(sapf_file, chunksize=chunk_size):
            sapf_chunk[sapf_timestamp_col] = pd.to_datetime(sapf_chunk[sapf_timestamp_col])
            
            # Build time index for this chunk
            for _, row in sapf_chunk.iterrows():
                timestamp = row[sapf_timestamp_col]
                if timestamp not in sapf_time_index:
                    sapf_time_index[timestamp] = []
                
                # Store only essential data for each timestamp
                sapf_data = {}
                for col in sapf_cols:
                    if col in row and pd.notna(row[col]):
                        sapf_data[col] = row[col]
                
                if sapf_data:  # Only store if we have valid sap flow data
                    sapf_time_index[timestamp].append(sapf_data)
            
            # Memory cleanup after each chunk
            del sapf_chunk
            self.memory_manager.check_memory_usage()
        
        # Stream through environmental data
        for env_chunk_idx, env_chunk in enumerate(pd.read_csv(env_file, chunksize=chunk_size)):
            
            # Find timestamp column
            timestamp_cols = [col for col in env_chunk.columns if 'timestamp' in col.lower()]
            if not timestamp_cols:
                continue
            
            timestamp_col = timestamp_cols[0]
            env_chunk[timestamp_col] = pd.to_datetime(env_chunk[timestamp_col])
            
            # Process each row in the environmental chunk
            processed_rows = []
            
            for _, env_row in env_chunk.iterrows():
                timestamp = env_row[timestamp_col]
                
                # Look up sap flow data for this timestamp
                if timestamp in sapf_time_index:
                    sapf_entries = sapf_time_index[timestamp]
                    
                    # Create a row for each sap flow measurement at this timestamp
                    for sapf_entry in sapf_entries:
                        # Create combined row
                        combined_row = env_row.copy()
                        
                        # Add sap flow data
                        for plant_id, sap_flow_value in sapf_entry.items():
                            combined_row['plant_id'] = plant_id
                            combined_row['sap_flow'] = sap_flow_value
                            processed_rows.append(combined_row.copy())
            
            if not processed_rows:
                continue
            
            # Convert to DataFrame
            merged = pd.DataFrame(processed_rows)
            del processed_rows
            
            if len(sapf_chunk) == 0:
                continue
            
            # Melt sap flow data
            sapf_long = sapf_chunk.melt(
                id_vars=[sapf_timestamp_col], 
                value_vars=sapf_cols,
                var_name='plant_id', 
                value_name='sap_flow'
            )
            
            # Merge with environmental data
            merged = sapf_long.merge(
                env_chunk, 
                left_on=sapf_timestamp_col, 
                right_on=timestamp_col, 
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Remove missing sap flow data
            merged = merged.dropna(subset=['sap_flow'])
            
            if len(merged) == 0:
                continue
            
            # Use consolidated feature engineering (streaming mode)
            # Pass None for config to use default dict config (avoids class method conflicts)
            feature_engineer = FeatureEngineer(merged, config=None, memory_manager=self.memory_manager)
            merged = feature_engineer.create_all_features(merged, metadata, timestamp_col, processing_mode='streaming')
            del feature_engineer
            
            # Note: Seasonality features not available in streaming mode
            # (require site-level calculations that need full dataset)
            
            # Add site identifier
            merged['site'] = site
            
            # Save chunk with optimized I/O
            self.file_manager.save_streaming_chunk_optimized(merged, output_file, is_first_chunk=first_chunk)
            if first_chunk:
                first_chunk = False
            
            total_processed += len(merged)
            
            # Memory cleanup
            del env_chunk, merged
            self.memory_manager.check_memory_usage()
            
            # Continue processing all available data
        
        # Clean up time index
        del sapf_time_index
        self.memory_manager.check_memory_usage()
        
        if total_processed > 0:
            return True
        else:
            return None 