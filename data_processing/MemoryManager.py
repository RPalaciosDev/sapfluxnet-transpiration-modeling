import os
import gc
import gzip
import pandas as pd
import psutil
from pathlib import Path
from .error_utils import ErrorHandler
from .logging_utils import logger


class MemoryManager:
    """Handles system-wide memory monitoring, cleanup, and adaptive processing decisions"""
    
    def __init__(self, max_memory_gb=12, config=None, stats=None, file_manager=None):
        """
        Initialize MemoryManager
        
        Args:
            max_memory_gb: Maximum memory usage threshold in GB
            config: ProcessingConfig instance for accessing thresholds
            stats: Statistics dictionary to track memory cleanups
            file_manager: FileManager instance for file operations
        """
        self.max_memory_gb = max_memory_gb
        self.config = config
        self.stats = stats or {'memory_cleanups': 0}
        self.file_manager = file_manager
        
        # Get system information
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.initial_available_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.init_component("MemoryManager", f"{self.system_memory_gb:.1f}GB total, {self.initial_available_gb:.1f}GB available")
    
    def check_memory_usage(self):
        """Check current memory usage and force cleanup if needed"""
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        
        # Force garbage collection if memory usage is high
        if memory_gb > self.max_memory_gb:
            self.force_memory_cleanup()
            self.stats['memory_cleanups'] += 1
        
        # More aggressive cleanup for low memory situations
        if available_gb < 2:
            self.force_memory_cleanup()
            self.stats['memory_cleanups'] += 1
        
        return memory_gb
    
    def force_memory_cleanup(self):
        """Targeted memory cleanup based on actual memory pressure"""
        # Check current memory usage
        current_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Only force cleanup if memory is actually low
        if self.config and current_memory_gb < self.config.get_memory_threshold('critical'):
            # Critical memory situation - aggressive cleanup
            for i in range(3):
                gc.collect()
        elif self.config and current_memory_gb < self.config.get_memory_threshold('low'):
            # Low memory situation - moderate cleanup
            gc.collect()
        elif self.config and current_memory_gb < self.config.get_memory_threshold('moderate'):
            # Moderate memory situation - light cleanup
            gc.collect()
        elif not self.config and current_memory_gb < 2.0:
            # Fallback cleanup when no config is available
            gc.collect()
        # If memory is good (>4GB available), skip cleanup to avoid performance impact

    
    def load_data_in_chunks(self, file_path, chunk_size=None):
        """Load data in chunks with memory monitoring and compression support"""
        try:
            # Use FileManager for file path resolution if available
            if self.file_manager:
                exists, actual_file_path = self.file_manager.check_file_exists(file_path)
                if not exists:
                    return ErrorHandler.handle_file_not_found("data", file_path)
            else:
                # Fallback: Use basic file checking if no FileManager available
                if file_path.endswith('.gz') and os.path.exists(file_path):
                    actual_file_path = file_path
                elif os.path.exists(file_path + '.gz'):
                    actual_file_path = file_path + '.gz'
                elif not os.path.exists(file_path):
                    return ErrorHandler.handle_file_not_found("data", file_path)
                else:
                    actual_file_path = file_path
        
            # Get file size using FileManager if available
            if self.file_manager:
                file_size_mb = self.file_manager.get_file_size_mb(file_path, estimate_compressed=True)
            else:
                # Fallback: Get file size (account for compression) - simplified
                try:
                    file_size_bytes = os.path.getsize(actual_file_path)
                    # Estimate uncompressed size for .gz files
                    if actual_file_path.endswith('.gz'):
                        file_size_bytes *= 3  # Typical compression ratio
                    file_size_mb = file_size_bytes / (1024 * 1024)
                except Exception:
                    file_size_mb = 0.0
            
            # Use provided chunk_size or determine optimal chunk size
            if chunk_size is None:
                chunk_size = self.get_optimal_chunk_size_for_file(file_size_mb)
            
            # Adjust chunk size based on file size and available memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if self.config:
                if file_size_mb > self.config.get_file_size_threshold('large'):
                    chunk_size = min(chunk_size, self.config.get_chunk_size('very_low_memory'))
                elif file_size_mb > self.config.get_file_size_threshold('medium'):
                    chunk_size = min(chunk_size, self.config.get_chunk_size('low_memory'))
                elif file_size_mb > self.config.get_file_size_threshold('small'):
                    chunk_size = min(chunk_size, self.config.get_chunk_size('medium_memory'))
                
                # Further adjust based on available memory
                if available_memory_gb < self.config.get_memory_threshold('streaming_low'):
                    chunk_size = min(chunk_size, self.config.get_chunk_size('very_low_memory'))
                elif available_memory_gb < self.config.get_memory_threshold('streaming_medium'):
                    chunk_size = min(chunk_size, self.config.get_chunk_size('low_memory'))
            else:
                # Fallback logic when no config is available
                if file_size_mb > 200:  # Large files
                    chunk_size = min(chunk_size, 500)
                elif file_size_mb > 100:  # Medium files
                    chunk_size = min(chunk_size, 1000)
        
            chunks = []
            total_rows = 0
            
            # Use pandas read_csv with compression detection
            for i, chunk in enumerate(pd.read_csv(actual_file_path, chunksize=chunk_size)):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Check memory every few chunks
                if i % 3 == 0:
                    self.check_memory_usage()
                
                # Safety check - don't load too many chunks at once
                if len(chunks) > 30:
                    temp_df = pd.concat(chunks, ignore_index=True)
                    chunks = [temp_df]
                    self.force_memory_cleanup()
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            self.force_memory_cleanup()
            
            return df
                
        except Exception as e:
            return ErrorHandler.handle_data_loading_error("CSV", file_path, e)
    
    def determine_adaptive_settings(self, site, adaptive_settings=None):
        """Automatically determine optimal processing settings for a site"""
        
        # Initialize adaptive_settings if not provided
        if adaptive_settings is None:
            adaptive_settings = {}
        
        env_file = f'sapwood/{site}_env_data.csv'
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        # Use FileManager for file operations if available
        if self.file_manager:
            file_check = self.file_manager.check_files_exist([env_file, sapf_file], ['environmental data', 'sap flow data'])
            
            if not file_check[env_file]['exists'] or not file_check[sapf_file]['exists']:
                ErrorHandler.handle_processing_error("File validation", Exception("Missing required files"), site)
                return False, adaptive_settings
            
            # Get file sizes using FileManager
            env_size_mb = file_check[env_file]['size_mb']
            sapf_size_mb = file_check[sapf_file]['size_mb']
            total_size_mb = env_size_mb + sapf_size_mb
        else:
            # Fallback to basic file operations (minimal usage)
            try:
                env_exists = os.path.exists(env_file)
                sapf_exists = os.path.exists(sapf_file)
                
                if not env_exists or not sapf_exists:
                    ErrorHandler.handle_processing_error("File validation", Exception("Missing required files"), site)
                    return False, adaptive_settings
                
                # Get file sizes (fallback without FileManager)
                env_size_mb = os.path.getsize(env_file) / (1024 * 1024)
                sapf_size_mb = os.path.getsize(sapf_file) / (1024 * 1024)
                total_size_mb = env_size_mb + sapf_size_mb
            except Exception as e:
                ErrorHandler.handle_processing_error("File size calculation", e, site)
                return False, adaptive_settings
        
        # Get current memory availability
        current_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Initialize adaptive_decisions if not in stats
        if 'adaptive_decisions' not in self.stats:
            self.stats['adaptive_decisions'] = []
        
        # Adapt processing strategy based on memory and file size
        if self.config:
            # Use config thresholds
            if current_memory_gb < self.config.get_memory_threshold('streaming_low') and total_size_mb > self.config.get_file_size_threshold('small'):
                use_streaming = True
                reason = f"Very low memory ({current_memory_gb:.1f}GB) + large files ({total_size_mb:.1f}MB) - using streaming"
            elif current_memory_gb < self.config.get_memory_threshold('streaming_medium') and total_size_mb > self.config.get_file_size_threshold('medium'):
                use_streaming = True
                reason = f"Low memory ({current_memory_gb:.1f}GB) + large files ({total_size_mb:.1f}MB) - using streaming"
            elif total_size_mb > self.config.get_file_size_threshold('large'):
                use_streaming = True
                reason = f"Very large files ({total_size_mb:.1f}MB > {self.config.get_file_size_threshold('large')}MB threshold) - using streaming"
            else:
                use_streaming = False
                reason = f"Standard processing (memory: {current_memory_gb:.1f}GB, files: {total_size_mb:.1f}MB)"
            
            # Determine chunk size
            if current_memory_gb < self.config.get_memory_threshold('streaming_low'):
                chunk_size = self.config.get_chunk_size('very_low_memory')
            elif current_memory_gb < self.config.get_memory_threshold('streaming_medium'):
                chunk_size = self.config.get_chunk_size('low_memory')
            elif total_size_mb > self.config.get_file_size_threshold('large'):
                chunk_size = self.config.get_chunk_size('medium_memory')
            else:
                chunk_size = self.config.get_chunk_size('high_memory')
            
            # Feature settings
            max_lag_hours = self.config.get_feature_setting('max_lag_hours')
            rolling_windows = self.config.get_feature_setting('rolling_windows')
            memory_threshold = self.config.get_memory_threshold('memory_threshold')
            
            # Row limits
            if current_memory_gb < self.config.get_memory_threshold('streaming_low') and total_size_mb > self.config.get_file_size_threshold('medium'):
                max_rows = self.config.get_row_limit('very_low_memory')
                row_reason = f"Very low memory ({current_memory_gb:.1f}GB) + large files - limiting to {max_rows:,} rows"
            elif current_memory_gb < self.config.get_memory_threshold('streaming_medium') and total_size_mb > self.config.get_file_size_threshold('large'):
                max_rows = self.config.get_row_limit('low_memory')
                row_reason = f"Low memory ({current_memory_gb:.1f}GB) + very large files - limiting to {max_rows:,} rows"
            else:
                max_rows = None
                row_reason = f"Sufficient memory - processing all rows"
        else:
            # Fallback logic when no config is available
            if current_memory_gb < 4.0 and total_size_mb > 50:
                use_streaming = True
                reason = f"Low memory ({current_memory_gb:.1f}GB) + large files ({total_size_mb:.1f}MB) - using streaming"
            elif total_size_mb > 200:
                use_streaming = True
                reason = f"Very large files ({total_size_mb:.1f}MB) - using streaming"
            else:
                use_streaming = False
                reason = f"Standard processing (memory: {current_memory_gb:.1f}GB, files: {total_size_mb:.1f}MB)"
            
            chunk_size = 1000 if current_memory_gb < 4.0 else 1500
            max_lag_hours = 24
            rolling_windows = [3, 6, 12, 24]
            max_rows = None
            row_reason = "No config - processing all rows"
        
        # Update adaptive settings
        adaptive_settings.update({
            'use_streaming': use_streaming,
            'chunk_size': chunk_size,
            'max_lag_hours': max_lag_hours,
            'rolling_windows': rolling_windows,
            'create_advanced_features': True,
            'create_domain_features': True,
            'max_rows_per_site': max_rows
        })
        
        # Log decisions
        self.stats['adaptive_decisions'].append(f"{site}: streaming={use_streaming} ({reason})")
        self.stats['adaptive_decisions'].append(f"{site}: complete feature creation enabled")
        self.stats['adaptive_decisions'].append(f"{site}: row_limit={max_rows if max_rows else 'None'} ({row_reason})")
        
        return True, adaptive_settings

    def get_optimal_chunk_size(self, dataset_size, available_memory_gb):
        """Determine optimal chunk size for I/O operations based on dataset size and memory"""
        # Calculate optimal chunk size based on available memory and dataset size
        if self.config:
            # Target: Use configurable percentage of available memory for each chunk
            target_memory_mb = available_memory_gb * 1000 * self.config.get_feature_setting('memory_percentage')
            
            # Estimate memory per row from configuration
            estimated_memory_per_row_mb = self.config.get_feature_setting('estimated_memory_per_row_mb')
            
            # Calculate optimal chunk size
            optimal_chunk_size = int(target_memory_mb / estimated_memory_per_row_mb)
            
            # Apply reasonable bounds from configuration
            min_chunk_size = self.config.get_chunk_size('min')
            max_chunk_size = self.config.get_chunk_size('max')
            
            optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
            
            # For very large datasets, use larger chunks to reduce I/O operations
            if dataset_size > 1000000:  # >1M rows
                optimal_chunk_size = max(optimal_chunk_size, self.config.get_chunk_size('large_dataset'))
        else:
            # Fallback logic when no config is available
            target_memory_mb = available_memory_gb * 100  # 10% of available memory
            estimated_memory_per_row_mb = 0.001  # 1KB per row estimate
            optimal_chunk_size = int(target_memory_mb / estimated_memory_per_row_mb)
            optimal_chunk_size = max(10000, min(optimal_chunk_size, 200000))  # Bounds: 10K-200K
        
        return optimal_chunk_size
    
    def get_optimal_chunk_size_for_file(self, file_size_mb):
        """Helper method to determine chunk size based on file size"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if self.config:
            if file_size_mb > self.config.get_file_size_threshold('large'):
                return self.config.get_chunk_size('very_low_memory')
            elif file_size_mb > self.config.get_file_size_threshold('medium'):
                return self.config.get_chunk_size('low_memory')
            elif file_size_mb > self.config.get_file_size_threshold('small'):
                return self.config.get_chunk_size('medium_memory')
            else:
                return self.config.get_chunk_size('high_memory')
        else:
            # Fallback logic
            if file_size_mb > 200:  # Large files
                return 500
            elif file_size_mb > 100:  # Medium files
                return 1000
            else:
                return 1500
    
    def get_memory_status(self):
        """Get current memory status information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'cleanups_performed': self.stats.get('memory_cleanups', 0)
        }