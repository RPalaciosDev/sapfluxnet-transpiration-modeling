import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import gc
import warnings
import psutil
import gzip
from contextlib import contextmanager
warnings.filterwarnings('ignore')

class ProcessingConfig:
    """Configuration class for processing parameters - replaces hardcoded values"""
    
    # Memory thresholds (GB)
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
    
    # Chunk size settings
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
        'rolling_windows': [3, 6, 12, 24, 48, 72],  # Rolling window sizes
        'memory_percentage': 0.1,   # Target memory usage per chunk (10%)
        'estimated_memory_per_row_mb': 0.001,  # Estimated memory per row (1KB)
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
        'min_file_size_mb': 1.0,    # Minimum file size for validation (uncompressed)
        'min_file_size_compressed_mb': 0.3,  # Minimum file size for validation (compressed)
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

class MemoryEfficientSAPFLUXNETProcessor:
    """Complete SAPFLUXNET processor with adaptive memory management - creates all features consistently"""
    
    # Sites with no valid sap flow data - should be excluded from processing
    SITES_WITH_NO_VALID_DATA = {
        'AUS_CAN_ST2_MIX', 'AUS_CAR_THI_0P0', 'AUS_CAR_THI_TP0', 'AUS_CAR_THI_TPF', 'AUS_ELL_HB_HIG', 'AUS_RIC_EUC_ELE',
        'CAN_TUR_P39_POS', 'CAN_TUR_P74',
        'CHE_LOT_NOR',
        'DEU_HIN_OAK', 'DEU_HIN_TER', 'DEU_STE_2P3', 'DEU_STE_4P5',
        'ESP_CAN', 'ESP_GUA_VAL', 'ESP_TIL_PIN', 'ESP_TIL_OAK',
        'FIN_HYY_SME', 'FIN_PET',
        'FRA_FON', 'FRA_HES_HE2_NON',
        'GBR_GUI_ST2', 'GBR_GUI_ST3', 'GBR_DEV_CON', 'GBR_DEV_DRO',
        'GUF_GUY_ST2', 'GUF_NOU_PET',
        'JPN_EBE_SUG', 'JPN_EBE_HYB',
        'KOR_TAE_TC1_LOW', 'KOR_TAE_TC2_MED', 'KOR_TAE_TC3_EXT',
        'MEX_VER_BSJ', 'MEX_VER_BSM',
        'PRT_LEZ_ARN',
        'RUS_FYO', 'RUS_CHE_Y4',
        'SWE_NOR_ST1_AF1', 'SWE_NOR_ST1_AF2', 'SWE_NOR_ST1_BEF', 'SWE_NOR_ST2',
        'SWE_NOR_ST3', 'SWE_NOR_ST4_AFT', 'SWE_NOR_ST4_BEF', 'SWE_NOR_ST5_REF',
        'SWE_SKO_MIN', 'SWE_SKY_38Y', 'SWE_SKY_68Y',
        'USA_BNZ_BLA', 'USA_DUK_HAR', 'USA_HIL_HF1_POS', 'USA_HUY_LIN_NON',
        'USA_PAR_FER', 'USA_PJS_P04_AMB', 'USA_PJS_P08_AMB', 'USA_PJS_P12_AMB',
        'USA_SIL_OAK_1PR', 'USA_SIL_OAK_2PR', 'USA_SIL_OAK_POS',
        'USA_SMI_SCB', 'USA_SMI_SER', 'USA_SYL_HL1', 'USA_SYL_HL2',
        'USA_UMB_CON', 'USA_UMB_GIR', 'USA_WIL_WC1', 'USA_WIL_WC2',
        'USA_HIL_HF2',
        'UZB_YAN_DIS'
    }
    
    # Sites with extremely high quality flag rates (>50%) - should be excluded
    EXTREMELY_PROBLEMATIC_SITES = {
        'IDN_PON_STE',  # 63.1% flag rate - Extremely poor quality
        'ZAF_NOO_E3_IRR',  # 25.9% flag rate - Very poor quality  
        'GUF_GUY_GUY',  # 35.5% flag rate - Very poor quality
        'USA_NWH',  # 53.4% flag rate - Very poor quality
        'USA_TNP',  # 31.6% flag rate - Very poor quality
        'USA_TNY',  # 28.9% flag rate - Very poor quality
        'USA_WVF',  # 16.6% flag rate - Very poor quality
    }
    
    # Sites with high quality flag rates (20-50%) - should be excluded
    HIGH_PROBLEMATIC_SITES = {
        'USA_SYL_HL2',  # 16.0% flag rate - Poor quality
        'USA_WIL_WC2',  # 13.3% flag rate - Poor quality
        'CAN_TUR_P39_POS',  # 13.2% flag rate - Poor quality
        'CAN_TUR_P74',  # 15.8% flag rate - Poor quality
        'USA_PAR_FER',  # 16.7% flag rate - Poor quality
        'USA_TNB',  # 19.4% flag rate - Poor quality
        'USA_TNO',  # 19.3% flag rate - Poor quality
        'USA_UMB_GIR',  # 27.9% flag rate - Poor quality
    }
    
    # Sites with moderate quality issues (10-20%) - should be processed with warnings
    MODERATE_PROBLEMATIC_SITES = {
        'FRA_PUE',  # 9.1% flag rate - Moderate issues
        'CAN_TUR_P39_PRE',  # 9.2% flag rate - Moderate issues
        'FRA_HES_HE2_NON',  # 9.0% flag rate - Moderate issues
        'USA_DUK_HAR',  # 6.0% flag rate - Moderate issues
        'USA_UMB_CON',  # 1.6% flag rate - Moderate issues
        'USA_PJS_P12_AMB',  # 3.0% flag rate - Moderate issues
        'USA_SIL_OAK_2PR',  # 3.0% flag rate - Moderate issues
        'USA_SIL_OAK_1PR',  # 3.0% flag rate - Moderate issues
        'USA_PJS_P04_AMB',  # 2.2% flag rate - Moderate issues
        'USA_PJS_P08_AMB',  # 1.8% flag rate - Moderate issues
        'USA_SIL_OAK_POS',  # 3.5% flag rate - Moderate issues
    }
    
    # Combined list of all problematic sites
    PROBLEMATIC_SITES = EXTREMELY_PROBLEMATIC_SITES | HIGH_PROBLEMATIC_SITES | MODERATE_PROBLEMATIC_SITES
    
    def __init__(self, output_dir='comprehensive_processed', chunk_size=1000, max_memory_gb=12, force_reprocess=False, skip_problematic_sites=True, use_quality_flags=True, compress_output=False, optimize_io=True, export_format='csv', config_overrides=None):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb  # Leave 4GB buffer
        self.force_reprocess = force_reprocess  # Force reprocessing of all sites
        self.skip_problematic_sites = skip_problematic_sites  # Whether to skip problematic sites
        self.use_quality_flags = use_quality_flags  # Whether to filter out flagged data points
        self.compress_output = compress_output  # Whether to compress output files
        self.optimize_io = optimize_io  # Whether to use optimized I/O strategies
        self.export_format = export_format.lower()  # Export format: csv, parquet, feather, hdf5, pickle
        
        # Validate export format
        self.validate_export_format()
        
        # Apply configuration overrides if provided
        if config_overrides:
            self.apply_config_overrides(config_overrides)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get system information for adaptive processing
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"💻 System: {self.system_memory_gb:.1f}GB total, {self.available_memory_gb:.1f}GB available")
        print(f"💾 I/O: {'On' if self.optimize_io else 'Off'}, Compression: {'On' if self.compress_output else 'Off'}")
        print(f"📁 Export format: {self.export_format.upper()}")
        
        # Settings for complete feature creation with adaptive memory management
        self.adaptive_settings = {
            'use_streaming': None,  # Will be determined based on memory and file size
            'streaming_threshold_mb': 50,  # Lower threshold for memory-constrained systems
            'max_lag_hours': None,  # Will be set to full 24 hours
            'rolling_windows': None,  # Will be set to full range
            'create_advanced_features': None,  # Will always be True
            'create_domain_features': None,  # Will always be True
            'chunk_size': None,  # Will be determined based on memory and file size
        }
        
        # Track processing statistics
        self.stats = {
            'successful_sites': 0,
            'failed_sites': 0,
            'total_rows': 0,
            'total_columns': 0,
            'memory_cleanups': 0,
            'io_operations': 0,  # Track I/O operations
            'bytes_written': 0,  # Track bytes written
            'adaptive_decisions': []  # Track adaptive decisions made
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
    
    def validate_export_format(self):
        """Validate the export format and check for required dependencies"""
        valid_formats = ['csv', 'parquet', 'feather', 'hdf5', 'pickle']
        
        if self.export_format not in valid_formats:
            raise ValueError(f"Invalid export format: {self.export_format}. Valid formats: {valid_formats}")
        
        # Check for required dependencies
        if self.export_format == 'parquet':
            try:
                import pyarrow
            except ImportError:
                raise ImportError("pyarrow is required for parquet export. Install with: pip install pyarrow")
        
        elif self.export_format == 'feather':
            try:
                import pyarrow
            except ImportError:
                raise ImportError("pyarrow is required for feather export. Install with: pip install pyarrow")
        
        elif self.export_format == 'hdf5':
            try:
                import tables
            except ImportError:
                raise ImportError("tables (PyTables) is required for HDF5 export. Install with: pip install tables")
    
    def get_output_file_extension(self):
        """Get the appropriate file extension for the export format"""
        extensions = {
            'csv': '.csv',
            'parquet': '.parquet',
            'feather': '.feather',
            'hdf5': '.h5',
            'pickle': '.pkl'
        }
        return extensions.get(self.export_format, '.csv')
    
    def save_dataframe_formatted(self, df, output_file, site_name):
        """Save DataFrame in the specified format with optimized I/O"""
        
        # Add compression extension if needed
        if self.compress_output and self.export_format == 'csv':
            output_file += '.gz'
        
        try:
            if self.export_format == 'csv':
                if self.compress_output:
                    df.to_csv(output_file, index=False, compression='gzip')
                else:
                    df.to_csv(output_file, index=False)
                
            elif self.export_format == 'parquet':
                df.to_parquet(output_file, index=False, compression='snappy' if self.compress_output else None)
                
            elif self.export_format == 'feather':
                df.to_feather(output_file, compression='lz4' if self.compress_output else 'uncompressed')
                
            elif self.export_format == 'hdf5':
                df.to_hdf(output_file, key='data', mode='w', complevel=6 if self.compress_output else 0)
                
            elif self.export_format == 'pickle':
                df.to_pickle(output_file, compression='gzip' if self.compress_output else None)
            
            # Track statistics
            self.stats['io_operations'] += 1
            self.stats['bytes_written'] += os.path.getsize(output_file)
            
            print(f"  💾 Saved {site_name} in {self.export_format.upper()} format: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"  ❌ Error saving {site_name} in {self.export_format} format: {str(e)}")
            # Fallback to CSV if other format fails
            if self.export_format != 'csv':
                print(f"  🔄 Falling back to CSV format...")
                self.export_format = 'csv'
                self.save_dataframe_formatted(df, output_file.replace(self.get_output_file_extension(), '.csv'), site_name)
    
    @contextmanager
    def optimized_file_writer(self, file_path, mode='w'):
        """Context manager for optimized file writing with buffering and compression"""
        if self.compress_output and not file_path.endswith('.gz'):
            file_path += '.gz'
        
        if self.compress_output:
            # Use gzip compression with buffering
            with gzip.open(file_path, mode + 't', compresslevel=ProcessingConfig.get_io_setting('compression_level'), encoding='utf-8') as f:
                yield f
        else:
            # Use regular file with buffering
            with open(file_path, mode, buffering=ProcessingConfig.get_io_setting('buffer_size')) as f:
                yield f
    
    def get_optimal_chunk_size(self, dataset_size, available_memory_gb):
        """Determine optimal chunk size for I/O operations based on dataset size and memory"""
        if not self.optimize_io:
            return 50000  # Default chunk size
        
        # Calculate optimal chunk size based on available memory and dataset size
        # Target: Use configurable percentage of available memory for each chunk
        target_memory_mb = available_memory_gb * 1000 * ProcessingConfig.get_feature_setting('memory_percentage')
        
        # Estimate memory per row from configuration
        estimated_memory_per_row_mb = ProcessingConfig.get_feature_setting('estimated_memory_per_row_mb')
        
        # Calculate optimal chunk size
        optimal_chunk_size = int(target_memory_mb / estimated_memory_per_row_mb)
        
        # Apply reasonable bounds from configuration
        min_chunk_size = ProcessingConfig.get_chunk_size('min')
        max_chunk_size = ProcessingConfig.get_chunk_size('max')
        
        optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
        
        # For very large datasets, use larger chunks to reduce I/O operations
        if dataset_size > 1000000:  # >1M rows
            optimal_chunk_size = max(optimal_chunk_size, ProcessingConfig.get_chunk_size('large_dataset'))
        
        return optimal_chunk_size
    
    def save_dataframe_optimized(self, df, output_file, site_name):
        """Save DataFrame with optimized I/O strategy"""
        
        # Use the new formatted save method
        self.save_dataframe_formatted(df, output_file, site_name)
    
    def save_streaming_chunk_optimized(self, df, output_file, is_first_chunk=False):
        """Save streaming chunk with optimized I/O"""
        # For streaming, we'll use CSV format as it's the most compatible with append operations
        # Other formats don't support easy appending
        if is_first_chunk:
            df.to_csv(output_file, index=False, mode='w')
        else:
            df.to_csv(output_file, index=False, mode='a', header=False)
        
        self.stats['io_operations'] += 1
    
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
        if current_memory_gb < ProcessingConfig.get_memory_threshold('critical'):
            # Critical memory situation - aggressive cleanup
            for i in range(3):
                gc.collect()
        elif current_memory_gb < ProcessingConfig.get_memory_threshold('low'):
            # Low memory situation - moderate cleanup
            gc.collect()
        elif current_memory_gb < ProcessingConfig.get_memory_threshold('moderate'):
            # Moderate memory situation - light cleanup
            gc.collect()
        # If memory is good (>4GB available), skip cleanup to avoid performance impact
    
    def get_all_sites(self):
        """Get all sites from sapwood directory"""
        all_files = glob.glob('sapwood/*.csv')
        sites = set()
        
        for file in all_files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 2:
                if len(parts) == 4 and parts[2] in ['env', 'sapf', 'plant', 'site', 'species', 'stand']:
                    site = f"{parts[0]}_{parts[1]}"
                    sites.add(site)
                elif len(parts) >= 5 and parts[-2] in ['env', 'sapf', 'plant', 'site', 'species', 'stand']:
                    site = '_'.join(parts[:-2])
                    sites.add(site)
        
        return sorted(list(sites))
    
    def should_skip_site(self, site):
        """Check if site has already been processed and should be skipped"""
        # Check if site has no valid sap flow data - always skip these
        if site in self.SITES_WITH_NO_VALID_DATA:
            print(f"  🚫 Skipping {site} - No valid sap flow data")
            return True
        
        # Check if site is problematic and should be skipped
        if self.skip_problematic_sites and site in self.PROBLEMATIC_SITES:
            if site in self.EXTREMELY_PROBLEMATIC_SITES:
                print(f"  🚫 Skipping {site} - Extremely problematic site (>50% flag rate)")
                return True
            elif site in self.HIGH_PROBLEMATIC_SITES:
                print(f"  🚫 Skipping {site} - High problematic site (20-50% flag rate)")
                return True
            elif site in self.MODERATE_PROBLEMATIC_SITES:
                print(f"  ⚠️  Processing {site} with warnings - Moderate quality issues (10-20% flag rate)")
                # Don't skip moderate sites, but warn about them
        
        # If force reprocess is enabled, never skip
        if self.force_reprocess:
            return False
        
        # Check for file with appropriate extension
        file_extension = self.get_output_file_extension()
        output_file = f'{self.output_dir}/{site}_comprehensive{file_extension}'
        
        # Check if file exists
        if not os.path.exists(output_file):
            return False
        
        # Check file size to ensure it's not empty or corrupted (handle compression)
        actual_file_path = output_file
        if self.compress_output and self.export_format == 'csv' and os.path.exists(output_file + '.gz'):
            actual_file_path = output_file + '.gz'
        elif not os.path.exists(output_file):
            return False
        
        file_size = os.path.getsize(actual_file_path) / (1024 * 1024)  # MB
        
        # Skip if file exists and has reasonable size from configuration
        min_size_mb = (ProcessingConfig.get_quality_flag_setting('min_file_size_compressed_mb') 
                      if actual_file_path.endswith('.gz') 
                      else ProcessingConfig.get_quality_flag_setting('min_file_size_mb'))
        if file_size > min_size_mb:
            # Additional validation: check if file can be read and has expected columns
            if self.validate_existing_file(output_file):
                return True
            else:
                print(f"  ⚠️  Found corrupted file for {site} ({file_size:.1f}MB) - will reprocess")
                return False
        
        # If file is too small, it might be corrupted or incomplete
        print(f"  ⚠️  Found small/corrupted file for {site} ({file_size:.1f}MB) - will reprocess")
        return False
    
    def validate_existing_file(self, file_path):
        """Validate that an existing processed file is not corrupted (supports multiple formats)"""
        try:
            # Check for compressed version first (only for CSV)
            actual_file_path = file_path
            if self.compress_output and self.export_format == 'csv' and os.path.exists(file_path + '.gz'):
                actual_file_path = file_path + '.gz'
            elif not os.path.exists(file_path):
                return False
            
            # Try to read the first few rows to check if file is valid
            if self.export_format == 'csv':
                sample = pd.read_csv(actual_file_path, nrows=5)
            elif self.export_format == 'parquet':
                sample = pd.read_parquet(actual_file_path)
                sample = sample.head(5)
            elif self.export_format == 'feather':
                sample = pd.read_feather(actual_file_path)
                sample = sample.head(5)
            elif self.export_format == 'hdf5':
                sample = pd.read_hdf(actual_file_path, key='data')
                sample = sample.head(5)
            elif self.export_format == 'pickle':
                sample = pd.read_pickle(actual_file_path)
                sample = sample.head(5)
            
            # Check if file has expected structure
            if len(sample.columns) < 50:  # Should have many features
                return False
            
            # Check if it has the essential columns
            essential_cols = ['sap_flow', 'site']
            if not all(col in sample.columns for col in essential_cols):
                return False
            
            # Check if there's actual data
            if len(sample) == 0:
                return False
            
            return True
            
        except Exception as e:
            # If any error occurs, file is corrupted
            return False
    
    def validate_sap_flow_data(self, sapf_file):
        """Quickly validate sap flow data before loading environmental data"""
        try:
            # Read just a small sample to check structure
            sample = pd.read_csv(sapf_file, nrows=100)
            
            # Check if file has data
            if len(sample) == 0:
                return {'valid': False, 'reason': 'Empty file', 'columns': []}
            
            # Find timestamp column
            sapf_timestamp_cols = [col for col in sample.columns if 'timestamp' in col.lower()]
            if not sapf_timestamp_cols:
                return {'valid': False, 'reason': 'No timestamp column found', 'columns': []}
            
            sapf_timestamp_col = sapf_timestamp_cols[0]
            
            # Find valid sap flow columns
            sapf_cols = []
            for col in sample.columns:
                if col not in [sapf_timestamp_col, 'solar_TIMESTAMP'] and not col.lower().startswith('timestamp'):
                    sample_values = sample[col].dropna().head(10)
                    if len(sample_values) > 0 and pd.api.types.is_numeric_dtype(sample_values):
                        sapf_cols.append(col)
            
            if len(sapf_cols) == 0:
                return {'valid': False, 'reason': 'No valid sap flow columns found', 'columns': []}
            
            return {'valid': True, 'reason': 'Valid sap flow data', 'columns': sapf_cols}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Error reading file: {str(e)}', 'columns': []}
    
    def get_processing_status(self, all_sites):
        """Get processing status for all sites"""
        sites_to_process = []
        sites_to_skip = []
        
        for site in all_sites:
            if self.should_skip_site(site):
                sites_to_skip.append(site)
            else:
                sites_to_process.append(site)
        
        return sites_to_process, sites_to_skip
    
    def load_data_in_chunks(self, file_path):
        """Load data in chunks with memory monitoring and compression support"""
        try:
            # Handle compressed files
            actual_file_path = file_path
            if file_path.endswith('.gz') and os.path.exists(file_path):
                # File is already compressed
                pass
            elif os.path.exists(file_path + '.gz'):
                # Compressed version exists
                actual_file_path = file_path + '.gz'
            elif not os.path.exists(file_path):
                print(f"    ❌ File not found: {file_path}")
                return None
            
            # Get file size (account for compression)
            if actual_file_path.endswith('.gz'):
                # For compressed files, estimate original size
                with gzip.open(actual_file_path, 'rt') as f:
                    # Read first few lines to estimate compression ratio
                    sample_lines = [f.readline() for _ in range(100)]
                    if sample_lines:
                        compressed_size = os.path.getsize(actual_file_path)
                        # Rough estimate: compression ratio ~3:1 for CSV data
                        estimated_size = compressed_size * 3
                    else:
                        estimated_size = os.path.getsize(actual_file_path) * 3
            else:
                estimated_size = os.path.getsize(actual_file_path)
            
            file_size_mb = estimated_size / (1024 * 1024)
            
            # Adjust chunk size based on file size and available memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if file_size_mb > ProcessingConfig.get_file_size_threshold('large'):
                chunk_size = min(self.chunk_size, ProcessingConfig.get_chunk_size('very_low_memory'))
            elif file_size_mb > ProcessingConfig.get_file_size_threshold('medium'):
                chunk_size = min(self.chunk_size, ProcessingConfig.get_chunk_size('low_memory'))
            elif file_size_mb > ProcessingConfig.get_file_size_threshold('small'):
                chunk_size = min(self.chunk_size, ProcessingConfig.get_chunk_size('medium_memory'))
            else:
                chunk_size = self.chunk_size
            
            # Further adjust based on available memory
            if available_memory_gb < ProcessingConfig.get_memory_threshold('streaming_low'):
                chunk_size = min(chunk_size, ProcessingConfig.get_chunk_size('very_low_memory'))
            elif available_memory_gb < ProcessingConfig.get_memory_threshold('streaming_medium'):
                chunk_size = min(chunk_size, ProcessingConfig.get_chunk_size('low_memory'))
            
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
            print(f"    ❌ Error loading {file_path}: {str(e)}")
            return None
    
    def load_and_filter_data_with_flags(self, site, data_type='env'):
        """Load data and apply quality flag filtering"""
        
        # Load data
        data_file = f'sapwood/{site}_{data_type}_data.csv'
        flags_file = f'sapwood/{site}_{data_type}_flags.csv'
        
        if not os.path.exists(data_file):
            print(f"    ❌ {data_type} data file not found: {data_file}")
            return None
        
        # Load data
        data = self.load_data_in_chunks(data_file)
        if data is None:
            return None
        
        # Fix column naming issues
        if 'TIMESTAMP_solar' in data.columns:
            data = data.rename(columns={'TIMESTAMP_solar': 'solar_TIMESTAMP'})
            print(f"    🔧 Renamed TIMESTAMP_solar to solar_TIMESTAMP")
        
        # Exclude problematic columns that cause inconsistencies
        columns_to_exclude = ['pl_name', 'swc_deep', 'netrad', 'seasonal_leaf_area']
        for col in columns_to_exclude:
            if col in data.columns:
                data = data.drop(columns=[col])
                print(f"    🚫 Excluded problematic column: {col}")
        
        original_rows = len(data)
        
        # Apply quality flag filtering if flags file exists and filtering is enabled
        if os.path.exists(flags_file) and self.use_quality_flags:
            try:
                flags = pd.read_csv(flags_file)
                
                # Ensure data and flags have same number of rows
                if len(data) != len(flags):
                    # Align by timestamp if possible
                    if 'TIMESTAMP' in data.columns and 'TIMESTAMP' in flags.columns:
                        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
                        flags['TIMESTAMP'] = pd.to_datetime(flags['TIMESTAMP'])
                        data = data.merge(flags, on='TIMESTAMP', how='inner')
                        flags = data[flags.columns]
                        data = data.drop(columns=[col for col in flags.columns if col != 'TIMESTAMP'])
                    else:
                        return data
                
                # Find flag columns (exclude timestamp columns)
                flag_columns = [col for col in flags.columns if 'timestamp' not in col.lower()]
                
                if flag_columns:
                    # Create mask for rows with any bad flags from configuration
                    bad_flags = flags[flag_columns].isin(ProcessingConfig.get_quality_flag_setting('bad_flags'))
                    rows_with_flags = bad_flags.any(axis=1)
                    
                    flagged_count = rows_with_flags.sum()
                    
                    # Filter out flagged rows
                    data_clean = data[~rows_with_flags]
                    removed_count = original_rows - len(data_clean)
                    
                    # Memory cleanup
                    del flags, bad_flags, rows_with_flags
                    self.force_memory_cleanup()
                    
                    return data_clean
                else:
                    return data
                    
            except Exception as e:
                return data
        else:
            return data
    
    def create_advanced_temporal_features(self, df, timestamp_col):
        """Create optimized temporal features using existing data"""
        
        features = df.copy()
        
        # Convert timestamp if not already done
        if not pd.api.types.is_datetime64_any_dtype(features[timestamp_col]):
            features[timestamp_col] = pd.to_datetime(features[timestamp_col])
        
        dt = features[timestamp_col].dt
        
        # Essential time features (keep these - they're useful for lagging/rolling)
        features['hour'] = dt.hour
        features['day_of_year'] = dt.dayofyear
        features['month'] = dt.month
        features['year'] = dt.year
        features['day_of_week'] = dt.dayofweek
        
        # Use solar timestamp if available (more accurate than calculated features)
        if 'solar_TIMESTAMP' in df.columns:
            solar_dt = pd.to_datetime(df['solar_TIMESTAMP']).dt
            features['solar_hour'] = solar_dt.hour
            features['solar_day_of_year'] = solar_dt.dayofyear
            
            # Solar-adjusted cyclical features (more accurate than calculated)
            solar_hour_rad = 2 * np.pi * features['solar_hour'] / 24
            features['solar_hour_sin'] = np.sin(solar_hour_rad)
            features['solar_hour_cos'] = np.cos(solar_hour_rad)
            
            solar_day_rad = 2 * np.pi * features['solar_day_of_year'] / 365
            features['solar_day_sin'] = np.sin(solar_day_rad)
            features['solar_day_cos'] = np.cos(solar_day_rad)
        
        # Simple boolean features (keep these - they're useful)
        features['is_daylight'] = ((features['hour'] >= 6) & (features['hour'] <= 18)).astype(int)
        features['is_peak_sunlight'] = ((features['hour'] >= 10) & (features['hour'] <= 16)).astype(int)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Clean up intermediate variables
        del dt
        if 'solar_TIMESTAMP' in df.columns:
            del solar_dt, solar_hour_rad, solar_day_rad
        
        return features
    

    
    def create_interaction_features(self, df):
        """Create interaction features efficiently"""
        
        features = df.copy()
        
        # Note: All interaction features excluded - can be computed during training
        # VPD and radiation interaction
        # if 'vpd' in df.columns and 'ppfd_in' in df.columns:
        #     features['vpd_ppfd_interaction'] = df['vpd'] * df['ppfd_in']
        
        # Temperature and humidity ratio
        # if 'ta' in df.columns and 'vpd' in df.columns:
        #     features['temp_humidity_ratio'] = df['ta'] / (df['vpd'] + 1e-6)
        
        # Water stress index
        # if 'swc_shallow' in df.columns and 'vpd' in df.columns:
        #     features['water_stress_index'] = df['swc_shallow'] / (df['vpd'] + 1e-6)
        
        # Light efficiency
        # if 'ppfd_in' in df.columns and 'sw_in' in df.columns:
        #     features['light_efficiency'] = df['ppfd_in'] / (df['sw_in'] + 1e-6)
        
        
        return features
    
    def load_metadata(self, site):
        """Load site, stand, species, environmental, and plant metadata for a given site"""
        metadata = {}
        
        # Load site metadata
        site_md_file = f'sapwood/{site}_site_md.csv'
        if os.path.exists(site_md_file):
            try:
                site_md = pd.read_csv(site_md_file)
                if not site_md.empty:
                    metadata['site'] = site_md.iloc[0].to_dict()
            except Exception as e:
                pass
        
        # Load stand metadata
        stand_md_file = f'sapwood/{site}_stand_md.csv'
        if os.path.exists(stand_md_file):
            try:
                stand_md = pd.read_csv(stand_md_file)
                if not stand_md.empty:
                    metadata['stand'] = stand_md.iloc[0].to_dict()
            except Exception as e:
                pass
        
        # Load species metadata
        species_md_file = f'sapwood/{site}_species_md.csv'
        if os.path.exists(species_md_file):
            try:
                species_md = pd.read_csv(species_md_file)
                if not species_md.empty:
                    metadata['species'] = species_md.iloc[0].to_dict()
            except Exception as e:
                pass
        
        # Load environmental metadata
        env_md_file = f'sapwood/{site}_env_md.csv'
        if os.path.exists(env_md_file):
            try:
                env_md = pd.read_csv(env_md_file)
                if not env_md.empty:
                    metadata['environmental'] = env_md.iloc[0].to_dict()
            except Exception as e:
                pass
        
        # Load plant metadata (individual tree information)
        plant_md_file = f'sapwood/{site}_plant_md.csv'
        if os.path.exists(plant_md_file):
            try:
                plant_md = pd.read_csv(plant_md_file)
                if not plant_md.empty:
                    metadata['plants'] = plant_md
            except Exception as e:
                pass
        
        return metadata
    
    def create_metadata_features(self, df, metadata):
        """Create features from metadata"""
        
        features = df.copy()
        
        # Site-level features
        if 'site' in metadata:
            site_data = metadata['site']
            
            # Geographic features
            if 'si_lat' in site_data and pd.notna(site_data['si_lat']):
                features['latitude'] = site_data['si_lat']
            if 'si_long' in site_data and pd.notna(site_data['si_long']):
                features['longitude'] = site_data['si_long']
            if 'si_elev' in site_data and pd.notna(site_data['si_elev']):
                features['elevation'] = site_data['si_elev']
            
            # Climate features
            if 'si_mat' in site_data and pd.notna(site_data['si_mat']):
                features['mean_annual_temp'] = site_data['si_mat']
            if 'si_map' in site_data and pd.notna(site_data['si_map']):
                features['mean_annual_precip'] = site_data['si_map']
            
            # Biome and land cover
            if 'si_biome' in site_data and pd.notna(site_data['si_biome']):
                features['biome'] = site_data['si_biome']
            if 'si_igbp' in site_data and pd.notna(site_data['si_igbp']):
                features['igbp_class'] = site_data['si_igbp']
            
            # Country
            if 'si_country' in site_data and pd.notna(site_data['si_country']):
                features['country'] = site_data['si_country']
            
            # Additional site metadata (using actual column names)
            if 'si_code' in site_data and pd.notna(site_data['si_code']):
                features['site_code'] = site_data['si_code']
            if 'si_name' in site_data and pd.notna(site_data['si_name']):
                features['site_name'] = site_data['si_name']
            # Note: si_paper excluded - not predictive for modeling
            # if 'si_paper' in site_data and pd.notna(site_data['si_paper']):
            #     features['site_paper'] = site_data['si_paper']
            # Note: si_remarks excluded due to inconsistency issues
            if 'is_inside_country' in site_data and pd.notna(site_data['is_inside_country']):
                features['is_inside_country'] = site_data['is_inside_country']
        
        # Stand-level features
        if 'stand' in metadata:
            stand_data = metadata['stand']
            
            # Stand characteristics
            if 'st_age' in stand_data and pd.notna(stand_data['st_age']):
                features['stand_age'] = stand_data['st_age']
            if 'st_basal_area' in stand_data and pd.notna(stand_data['st_basal_area']):
                features['basal_area'] = stand_data['st_basal_area']
            if 'st_density' in stand_data and pd.notna(stand_data['st_density']):
                features['tree_density'] = stand_data['st_density']
            if 'st_height' in stand_data and pd.notna(stand_data['st_height']):
                features['stand_height'] = stand_data['st_height']
            if 'st_lai' in stand_data and pd.notna(stand_data['st_lai']):
                features['leaf_area_index'] = stand_data['st_lai']
            
            # Soil characteristics
            if 'st_clay_perc' in stand_data and pd.notna(stand_data['st_clay_perc']):
                features['clay_percentage'] = stand_data['st_clay_perc']
            if 'st_sand_perc' in stand_data and pd.notna(stand_data['st_sand_perc']):
                features['sand_percentage'] = stand_data['st_sand_perc']
            if 'st_silt_perc' in stand_data and pd.notna(stand_data['st_silt_perc']):
                features['silt_percentage'] = stand_data['st_silt_perc']
            if 'st_soil_depth' in stand_data and pd.notna(stand_data['st_soil_depth']):
                features['soil_depth'] = stand_data['st_soil_depth']
            if 'st_USDA_soil_texture' in stand_data and pd.notna(stand_data['st_USDA_soil_texture']):
                features['soil_texture'] = stand_data['st_USDA_soil_texture']
            
            # Terrain and management
            # Note: aspect excluded - redundant with terrain
            # if 'st_aspect' in stand_data and pd.notna(stand_data['st_aspect']):
            #     features['aspect'] = stand_data['st_aspect']
            if 'st_terrain' in stand_data and pd.notna(stand_data['st_terrain']):
                features['terrain'] = stand_data['st_terrain']
            if 'st_growth_condition' in stand_data and pd.notna(stand_data['st_growth_condition']):
                features['growth_condition'] = stand_data['st_growth_condition']
            
            # Additional stand metadata (using actual column names)
            # Note: st_name and st_remarks excluded due to inconsistency issues
            # Note: stand_soil_texture excluded - redundant with individual texture percentages
            # if 'st_soil_texture' in stand_data and pd.notna(stand_data['st_soil_texture']):
            #     features['stand_soil_texture'] = stand_data['st_soil_texture']
        
        # Species-level features
        if 'species' in metadata:
            species_data = metadata['species']
            
            # Species characteristics
            if 'sp_name' in species_data and pd.notna(species_data['sp_name']):
                features['species_name'] = species_data['sp_name']
            if 'sp_leaf_habit' in species_data and pd.notna(species_data['sp_leaf_habit']):
                features['leaf_habit'] = species_data['sp_leaf_habit']
            if 'sp_ntrees' in species_data and pd.notna(species_data['sp_ntrees']):
                features['n_trees'] = species_data['sp_ntrees']
            # Note: species_basal_area_perc excluded - redundant with basal_area
            # if 'sp_basal_area_perc' in species_data and pd.notna(species_data['sp_basal_area_perc']):
            #     features['species_basal_area_perc'] = species_data['sp_basal_area_perc']
        
        # Environmental metadata features
        if 'environmental' in metadata:
            env_data = metadata['environmental']
            
            # Measurement protocol features
            if 'env_timestep' in env_data and pd.notna(env_data['env_timestep']):
                features['measurement_timestep'] = env_data['env_timestep']
            if 'env_time_zone' in env_data and pd.notna(env_data['env_time_zone']):
                features['timezone'] = env_data['env_time_zone']
            # Note: daylight_time excluded - redundant with solar time features
            # if 'env_time_daylight' in env_data and pd.notna(env_data['env_time_daylight']):
            #     features['daylight_time'] = env_data['env_time_daylight']
            
            # Sensor depth information
            # Note: swc_shallow_depth excluded due to inconsistency issues (only 50% of sites have it)
            
            # Measurement context
            # Note: env_leafarea_seasonal and env_remarks excluded due to inconsistency issues
        
        # Plant metadata features (individual tree characteristics)
        if 'plants' in metadata and 'plant_id' in features.columns:
            plants_data = metadata['plants']
            
            # Create a mapping from plant_id to tree characteristics
            plant_features = {}
            
            for _, plant in plants_data.iterrows():
                plant_code = plant.get('pl_code', '')
                if plant_code:
                    # Focus on essential tree characteristics for transpiration modeling
                    plant_features[plant_code] = {
                        'pl_age': plant.get('pl_age'),
                        'pl_dbh': plant.get('pl_dbh'),
                        'pl_height': plant.get('pl_height'),
                        'pl_leaf_area': plant.get('pl_leaf_area'),
                        'pl_bark_thick': plant.get('pl_bark_thick'),
                        'pl_social': plant.get('pl_social'),
                        'pl_species': plant.get('pl_species'),
                        'pl_sapw_area': plant.get('pl_sapw_area'),
                        'pl_sapw_depth': plant.get('pl_sapw_depth'),
                        'pl_name': plant.get('pl_name')
                    }
            
            # Add essential plant metadata features (excluding sensor details, treatment codes, and problematic columns)
            essential_plant_cols = ['pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 
                                  'pl_bark_thick', 'pl_social', 'pl_species',
                                  'pl_sapw_area', 'pl_sapw_depth']
            # Note: pl_name excluded due to high cardinality and inconsistency issues
            
            for col_name in essential_plant_cols:
                features[col_name] = features['plant_id'].map(
                    lambda x: plant_features.get(x, {}).get(col_name) if x in plant_features else None
                )
        
        # Create derived features
        features = self.create_derived_metadata_features(features)
        
        return features
    
    def create_derived_metadata_features(self, df):
        """Create derived features from metadata"""
        features = df.copy()
        
        # Climate zone based on latitude
        if 'latitude' in features.columns:
            # Numeric climate zone for categorical learning (0=Temperate_South, 1=Tropical, 2=Temperate_North)
            features['climate_zone_code'] = pd.cut(
                features['latitude'], 
                bins=[-90, -23.5, 23.5, 90], 
                labels=[0, 1, 2]
            ).astype('float64')
            
            # Absolute latitude for continuous relationships
            features['latitude_abs'] = abs(features['latitude'])
            
            # Original categorical climate zone (for reference)
            features['climate_zone'] = pd.cut(
                features['latitude'], 
                bins=[-90, -23.5, 23.5, 90], 
                labels=['Temperate_South', 'Tropical', 'Temperate_North']
            )
        
        # Aridity index (simplified)
        if 'mean_annual_temp' in features.columns and 'mean_annual_precip' in features.columns:
            features['aridity_index'] = features['mean_annual_precip'] / (features['mean_annual_temp'] + 10)
        
        # Leaf habit encoding
        if 'leaf_habit' in features.columns:
            leaf_habit_map = {
                'cold deciduous': 1,
                'warm deciduous': 2,
                'evergreen': 3,
                'semi-deciduous': 4
            }
            features['leaf_habit_code'] = features['leaf_habit'].map(leaf_habit_map).astype('float64')
        
        # Biome encoding
        if 'biome' in features.columns:
            biome_map = {
                'Tropical and Subtropical Moist Broadleaf Forests': 1,
                'Tropical and Subtropical Dry Broadleaf Forests': 2,
                'Tropical and Subtropical Coniferous Forests': 3,
                'Temperate Broadleaf and Mixed Forests': 4,
                'Temperate Conifer Forests': 5,
                'Boreal Forests/Taiga': 6,
                'Tropical and Subtropical Grasslands, Savannas and Shrublands': 7,
                'Temperate Grasslands, Savannas and Shrublands': 8,
                'Flooded Grasslands and Savannas': 9,
                'Montane Grasslands and Shrublands': 10,
                'Tundra': 11,
                'Mediterranean Forests, Woodlands and Scrub': 12,
                'Deserts and Xeric Shrublands': 13,
                'Mangroves': 14,
                'Woodland/Shrubland': 15
            }
            features['biome_code'] = features['biome'].map(biome_map).astype('float64')
        
        # IGBP class encoding
        if 'igbp_class' in features.columns:
            igbp_map = {
                'ENF': 1,  # Evergreen Needleleaf Forests
                'EBF': 2,  # Evergreen Broadleaf Forests
                'DNF': 3,  # Deciduous Needleleaf Forests
                'DBF': 4,  # Deciduous Broadleaf Forests
                'MF': 5,   # Mixed Forests
                'CSH': 6,  # Closed Shrublands
                'OSH': 7,  # Open Shrublands
                'WSA': 8,  # Woody Savannas
                'SAV': 9,  # Savannas
                'GRA': 10, # Grasslands
                'WET': 11, # Permanent Wetlands
                'CRO': 12, # Croplands
                'URB': 13, # Urban and Built-up
                'CVM': 14, # Cropland/Natural Vegetation Mosaics
                'SNO': 15, # Snow and Ice
                'BSV': 16  # Barren or Sparsely Vegetated
            }
            features['igbp_code'] = features['igbp_class'].map(igbp_map).astype('float64')
        
        # Tree size class based on DBH
        if 'pl_dbh' in features.columns:
            features['tree_size_class'] = pd.cut(
                features['pl_dbh'], 
                bins=[0, 10, 30, 50, 100, 1000], 
                labels=['Sapling', 'Small', 'Medium', 'Large', 'Very Large']
            )
        
        # Tree age class
        if 'pl_age' in features.columns:
            features['tree_age_class'] = pd.cut(
                features['pl_age'], 
                bins=[0, 20, 50, 100, 200, 1000], 
                labels=['Young', 'Mature', 'Old', 'Very Old', 'Ancient']
            )
        
        # Social status encoding
        if 'pl_social' in features.columns:
            social_map = {
                'dominant': 3,
                'codominant': 2,
                'intermediate': 1,
                'suppressed': 0
            }
            features['social_status_code'] = features['pl_social'].map(social_map).astype('float64')
        
        # Sapwood efficiency (sapwood area per unit leaf area)
        if 'pl_sapw_area' in features.columns and 'pl_leaf_area' in features.columns:
            features['sapwood_leaf_ratio'] = features['pl_sapw_area'] / (features['pl_leaf_area'] + 1e-6)
        
        # Tree volume index (DBH² × height)
        if 'pl_dbh' in features.columns and 'pl_height' in features.columns:
            features['tree_volume_index'] = (features['pl_dbh'] ** 2) * features['pl_height']
        

        
        # Timezone encoding (simplified)
        if 'timezone' in features.columns:
            # Convert to string first to avoid dtype issues
            timezone_col = features['timezone'].astype(str)
            features['timezone_offset'] = timezone_col.str.extract(r'([+-]\d{2})').astype(float)
        
        # Measurement frequency encoding
        if 'measurement_timestep' in features.columns:
            features['measurement_frequency'] = 60 / features['measurement_timestep']  # measurements per hour
        
        return features
    
    def encode_categorical_features(self, df):
        """Intelligently handle object columns - preserve continuous data, encode categorical data"""
        
        features = df.copy()
        
        # Define encoding mappings for known categorical variables
        encodings = {
            'biome': {
                'Tropical and Subtropical Moist Broadleaf Forests': 1,
                'Tropical and Subtropical Dry Broadleaf Forests': 2,
                'Tropical and Subtropical Coniferous Forests': 3,
                'Temperate Broadleaf and Mixed Forests': 4,
                'Temperate Conifer Forests': 5,
                'Boreal Forests/Taiga': 6,
                'Tropical and Subtropical Grasslands, Savannas and Shrublands': 7,
                'Temperate Grasslands, Savannas and Shrublands': 8,
                'Flooded Grasslands and Savannas': 9,
                'Montane Grasslands and Shrublands': 10,
                'Tundra': 11,
                'Mediterranean Forests, Woodlands and Scrub': 12,
                'Deserts and Xeric Shrublands': 13,
                'Mangroves': 14,
                'Woodland/Shrubland': 15
            },
            'igbp_class': {
                'ENF': 1, 'EBF': 2, 'DNF': 3, 'DBF': 4, 'MF': 5,
                'CSH': 6, 'OSH': 7, 'WSA': 8, 'SAV': 9, 'GRA': 10,
                'WET': 11, 'CRO': 12, 'URB': 13, 'CVM': 14, 'SNO': 15, 'BSV': 16
            },
            'country': {
                'ARG': 1, 'AUS': 2, 'AUT': 3, 'BRA': 4, 'CAN': 5,
                'CHE': 6, 'CHN': 7, 'COL': 8, 'CZE': 9, 'DEU': 10,
                'ESP': 11, 'FIN': 12, 'FRA': 13, 'GBR': 14, 'GUF': 15,
                'JPN': 16, 'KOR': 17, 'MDG': 18, 'PRT': 19, 'RUS': 20,
                'SEN': 21, 'SWE': 22, 'USA': 23, 'ZAF': 24
            },
            'soil_texture': {
                'clay': 1, 'clay loam': 2, 'loam': 3, 'loamy sand': 4,
                'sandy clay': 5, 'sandy clay loam': 6, 'sandy loam': 7, 'sand': 8,
                'silt': 9, 'silt loam': 10, 'silty clay': 11, 'silty clay loam': 12
            },
            'aspect': {
                'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SW': 6, 'W': 7, 'NW': 8
            },
            'terrain': {
                'Flat': 1, 'Gentle slope (<2 %)': 2, 'Moderate slope (2-10 %)': 3,
                'Steep slope (>10 %)': 4, 'Valley': 5, 'Ridge': 6
            },
            'growth_condition': {
                'Naturally regenerated, managed': 1,
                'Naturally regenerated, unmanaged': 2,
                'Planted, managed': 3,
                'Planted, unmanaged': 4
            },
            'leaf_habit': {
                'cold deciduous': 1, 'warm deciduous': 2, 'evergreen': 3, 'semi-deciduous': 4
            },
            'pl_social': {
                'dominant': 3, 'codominant': 2, 'intermediate': 1, 'suppressed': 0
            },
            'climate_zone': {
                'Temperate_South': 0, 'Tropical': 1, 'Temperate_North': 2
            },
            'tree_size_class': {
                'Sapling': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Very Large': 4
            },
            'tree_age_class': {
                'Young': 0, 'Mature': 1, 'Old': 2, 'Very Old': 3, 'Ancient': 4
            }
        }
        
        # Encode known categorical variables
        for col, mapping in encodings.items():
            if col in features.columns:
                features[f'{col}_code'] = features[col].map(mapping)
                # Drop original text column
                features = features.drop(col, axis=1)
        
        # Smart handling of object columns - distinguish between categorical and continuous
        text_columns = [col for col in features.columns if features[col].dtype == 'object']
        
        # Define columns that should be preserved as-is (not encoded or dropped)
        preserve_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'plant_id', 'sap_flow', 'site']
        
        # Define columns that should be converted to numeric (continuous data)
        numeric_cols = ['latitude', 'longitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
                       'stand_age', 'basal_area', 'tree_density', 'stand_height', 'leaf_area_index',
                       'clay_percentage', 'sand_percentage', 'silt_percentage', 'soil_depth',
                       'pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 'pl_bark_thick',
                       'pl_sapw_area', 'pl_sapw_depth', 'measurement_timestep', 'swc_shallow_depth',
                       'timezone_offset', 'measurement_frequency']
        
        # Process each object column intelligently
        for col in text_columns:
            if col in preserve_cols:
                # Keep these columns as-is
                continue
            elif col in numeric_cols:
                # Try to convert to numeric (continuous data)
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    pass
            else:
                # Check if this looks like categorical data
                unique_values = features[col].unique()
                unique_count = len(unique_values)
                
                # Determine if this is likely categorical or continuous
                if unique_count <= 20:
                    # Likely categorical - encode it as numeric (not pandas categorical)
                    encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                    features[f'{col}_code'] = features[col].map(encoding_map).astype('float64')
                    features = features.drop(col, axis=1)
                elif unique_count <= 100:
                    # Moderate cardinality - could be categorical or continuous
                    # Check if values look like categories or numbers
                    sample_values = features[col].dropna().head(10)
                    if all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values):
                        # Looks like numeric data - convert to numeric
                        try:
                            features[col] = pd.to_numeric(features[col], errors='coerce')
                        except:
                            # If conversion fails, encode as categorical (numeric dtype)
                            encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                            features[f'{col}_code'] = features[col].map(encoding_map).astype('float64')
                            features = features.drop(col, axis=1)
                    else:
                        # Looks like categorical - encode it as numeric (not pandas categorical)
                        encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                        features[f'{col}_code'] = features[col].map(encoding_map).astype('float64')
                        features = features.drop(col, axis=1)
                else:
                    # High cardinality - likely continuous data
                    # Try to convert to numeric first
                    try:
                        features[col] = pd.to_numeric(features[col], errors='coerce')
                    except:
                        # If conversion fails, check if it's a timestamp or should be preserved
                        if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp', 'id', 'code']):
                            pass
                        else:
                            # Drop only if it's truly problematic
                            features = features.drop(col, axis=1)
        
        return features
    
    def create_domain_specific_features(self, df):
        """Create domain-specific features for transpiration modeling"""
        
        features = df.copy()
        
        # Water stress features
        # Note: water_stress_index excluded - VPD and swc_shallow are already available as individual features
        
        # Light features
        # Note: ppfd_efficiency excluded - can be computed during training as ppfd_in / sw_in
        # if 'ppfd_in' in df.columns and 'sw_in' in df.columns:
        #     # PPFD efficiency relative to total shortwave radiation
        #     features['ppfd_efficiency'] = df['ppfd_in'] / (df['sw_in'] + 1e-6)
        
        # Temperature features
        if 'ta' in df.columns:
            # Temperature deviation from optimal (25°C for photosynthesis)
            features['temp_deviation'] = abs(df['ta'] - 25)
        
        # Physiological features
        # Note: stomatal_conductance_proxy excluded - can be computed during training as ppfd_in / vpd
        # if 'vpd' in df.columns and 'ppfd_in' in df.columns:
        #     # Stomatal conductance proxy (key physiological control)
        #     features['stomatal_conductance_proxy'] = df['ppfd_in'] / (df['vpd'] + 1e-6)
            
        # Soil moisture features
        # Note: moisture_availability excluded - swc_shallow is already available as individual feature
        
        # Wind effects on transpiration
        # Note: wind_stress and wind_vpd_interaction excluded - can be computed during training
        # if 'ws' in df.columns:
        #     # Wind stress (higher wind = more transpiration)
        #     features['wind_stress'] = df['ws'] / (df['ws'].max() + 1e-6)
        #     
        #     # Wind × VPD interaction (wind enhances VPD effects)
        #     if 'vpd' in df.columns:
        #         features['wind_vpd_interaction'] = df['ws'] * df['vpd']
        
        # Precipitation effects
        # Note: recent_precip and precip_intensity excluded - can be computed during training
        # if 'precip' in df.columns:
        #     # Recent precipitation (lagged)
        #     features['recent_precip_1h'] = df['precip'].shift(1).fillna(0)
        #     features['recent_precip_6h'] = df['precip'].rolling(6, min_periods=1).sum()
        #     features['recent_precip_24h'] = df['precip'].rolling(24, min_periods=1).sum()
        #     
        #     # Precipitation intensity
        #     features['precip_intensity'] = df['precip'] / (df['precip'].rolling(6, min_periods=1).sum() + 1e-6)
        
        # Diurnal cycle features removed for simplicity
        
        # Seasonal water use patterns removed for simplicity
        
        # Use existing data efficiently - avoid redundant calculations
        # Note: netrad and swc_deep excluded due to inconsistency issues across sites
        # Note: stomatal_control_index and light_efficiency excluded - can be computed during training
        # if 'ext_rad' in df.columns:
        #     # Extraterrestrial radiation is the perfect seasonal signal
        #     # Use it directly instead of calculating seasonal features
        #     
        #     # Key interaction: VPD × PPFD × Solar potential (stomatal control)
        #     if 'vpd' in df.columns and 'ppfd_in' in df.columns:
        #         features['stomatal_control_index'] = df['vpd'] * df['ppfd_in'] * df['ext_rad']
        #     
        #     # Light efficiency relative to solar potential
        #     if 'ppfd_in' in df.columns:
        #         features['light_efficiency'] = df['ppfd_in'] / (df['ext_rad'] + 1e-6)
        
        # Tree-specific features (if available)
        if 'pl_dbh' in df.columns:
            # Tree size effect on transpiration
            features['tree_size_factor'] = np.log(df['pl_dbh'] + 1)  # Log scale for large trees
        
        if 'pl_sapw_area' in df.columns and 'pl_leaf_area' in df.columns:
            # Sapwood to leaf area ratio (important for transpiration scaling)
            features['sapwood_leaf_ratio'] = df['pl_sapw_area'] / (df['pl_leaf_area'] + 1e-6)
            
            # Transpiration capacity (sapwood area × environmental factors)
            if 'vpd' in df.columns and 'ppfd_in' in df.columns:
                features['transpiration_capacity'] = (
                    df['pl_sapw_area'] * 
                    df['ppfd_in'] / (df['vpd'] + 1e-6)
                )
        
        # Site-specific features removed for simplicity
        
        return features
    

    
    def process_all_sites(self):
        """Process all sites with complete feature creation and adaptive memory management"""
        print("🔍 Finding all sites...")
        all_sites = self.get_all_sites()
        print(f"📊 Found {len(all_sites)} total sites")
        
        # Check which sites need processing vs which can be skipped
        sites_to_process, sites_to_skip = self.get_processing_status(all_sites)
        
        if self.force_reprocess:
            print(f"🔄 Force reprocessing enabled - will process all {len(all_sites)} sites")
            sites_to_process = all_sites
            sites_to_skip = []
        
        print(f"📋 Processing status:")
        print(f"  - Sites to process: {len(sites_to_process)}")
        print(f"  - Sites to skip: {len(sites_to_skip)}")
        
        if not sites_to_process:
            print(f"🎉 All sites already processed! No work to do.")
            return True
        
        print(f"\n🚀 Starting processing of {len(sites_to_process)} sites...")
        
        successful_sites = []
        failed_sites = []
        
        for i, site in enumerate(sites_to_process, 1):
            print(f"\n[{i}/{len(sites_to_process)}] Processing {site}...")
            
            # Force memory cleanup before each site
            self.force_memory_cleanup()
            
            # Use complete processing
            result = self.process_site_adaptive(site)
            
            if result is not None:
                # Save individual site file
                file_extension = self.get_output_file_extension()
                output_file = f'{self.output_dir}/{site}_comprehensive{file_extension}'
                
                if isinstance(result, pd.DataFrame):
                    # Standard processing result - use optimized I/O
                    self.save_dataframe_optimized(result, output_file, site)
                    
                    self.stats['total_rows'] += len(result)
                    self.stats['total_columns'] = max(self.stats['total_columns'], len(result.columns))
                    
                    # Clear memory immediately after saving
                    del result
                    self.check_memory_usage()
                else:
                    # Streaming processing result (already saved)
                    pass
                
                successful_sites.append(site)
                self.stats['successful_sites'] += 1
                
            else:
                failed_sites.append(site)
                self.stats['failed_sites'] += 1
        
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
    
    def process_site_adaptive(self, site):
        """Process a single site with adaptive settings determined on-the-fly"""
        
        try:
            # Determine optimal settings for this site
            if not self.determine_adaptive_settings(site):
                return None
            
            # Use adaptive settings
            if self.adaptive_settings['use_streaming']:
                return self._process_site_streaming_adaptive(site)
            else:
                return self._process_site_standard_adaptive(site)
                
        except Exception as e:
            print(f"  ❌ Error in adaptive processing for {site}: {str(e)}")
            return None
    
    def _process_site_standard_adaptive(self, site):
        """Standard processing using adaptive settings"""
        
        # Load metadata first
        metadata = self.load_metadata(site)
        
        # EARLY VALIDATION: Check sap flow data before loading environmental data
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        # Check if sap flow file exists
        if not os.path.exists(sapf_file):
            print(f"  ❌ Sap flow file not found: {sapf_file}")
            return None
        
        # Quick validation of sap flow data structure
        sapf_validation = self.validate_sap_flow_data(sapf_file)
        if not sapf_validation['valid']:
            print(f"  ❌ Sap flow data validation failed: {sapf_validation['reason']}")
            return None
        
        # Now load environmental data with quality flag filtering
        env_data = self.load_and_filter_data_with_flags(site, 'env')
        if env_data is None:
            return None
        
        # Find timestamp column
        timestamp_cols = [col for col in env_data.columns if 'timestamp' in col.lower()]
        if not timestamp_cols:
            print(f"  ❌ No timestamp column found for {site}")
            return None
        
        timestamp_col = timestamp_cols[0]
        
        # Load sap flow data with quality flag filtering
        sapf_data = self.load_and_filter_data_with_flags(site, 'sapf')
        if sapf_data is None:
            return None
        
        # Find sap flow timestamp column
        sapf_timestamp_cols = [col for col in sapf_data.columns if 'timestamp' in col.lower()]
        if not sapf_timestamp_cols:
            print(f"  ❌ No timestamp column found in sap flow data for {site}")
            return None
        
        sapf_timestamp_col = sapf_timestamp_cols[0]
        
        # Convert timestamps
        env_data[timestamp_col] = pd.to_datetime(env_data[timestamp_col])
        sapf_data[sapf_timestamp_col] = pd.to_datetime(sapf_data[sapf_timestamp_col])
        
        # Use the validated columns from earlier
        sapf_cols = sapf_validation['columns']
        
        # Optimized merging using time-based indexing
        
        # Convert timestamps to datetime for efficient merging
        env_data[timestamp_col] = pd.to_datetime(env_data[timestamp_col])
        sapf_data[sapf_timestamp_col] = pd.to_datetime(sapf_data[sapf_timestamp_col])
        
        # Set timestamp as index for faster merging
        env_data_indexed = env_data.set_index(timestamp_col)
        sapf_data_indexed = sapf_data.set_index(sapf_timestamp_col)
        
        # Melt sap flow data with indexed timestamp
        sapf_long = sapf_data_indexed[sapf_cols].reset_index().melt(
            id_vars=[sapf_timestamp_col], 
            value_vars=sapf_cols,
            var_name='plant_id', 
            value_name='sap_flow'
        )
        
        # Merge using indexed join (more memory efficient)
        merged = sapf_long.merge(
            env_data_indexed.reset_index(), 
            left_on=sapf_timestamp_col, 
            right_on=timestamp_col, 
            how='inner'
        )
        
        # Clean up intermediate dataframes
        del sapf_long, env_data, sapf_data, env_data_indexed, sapf_data_indexed
        self.check_memory_usage()
        
        # Remove rows with missing sap flow data
        merged = merged.dropna(subset=['sap_flow'])
        
        if len(merged) == 0:
            print(f"  ❌ No valid data after merging for {site}")
            return None
        
        # Create features using adaptive settings
        
        # Stage 1: Temporal features
            merged = self.create_advanced_temporal_features(merged, timestamp_col)
        self.check_memory_usage()
        
        # Stage 2: Lagged features (adaptive)
        env_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        merged = self.create_lagged_features_adaptive(merged, env_cols)
        self.check_memory_usage()
        
        # Stage 3: Rolling features (adaptive)
        merged = self.create_rolling_features_adaptive(merged, env_cols)
        self.check_memory_usage()
        
        # Stage 4: Interaction features
        merged = self.create_interaction_features(merged)
        self.check_memory_usage()
        
        # Stage 5: Domain-specific features (adaptive)
        if self.adaptive_settings['create_domain_features']:
            merged = self.create_domain_specific_features(merged)
            self.check_memory_usage()
        
        # Stage 6: Metadata features
        merged = self.create_metadata_features(merged, metadata)
        self.check_memory_usage()
        
        # Encode categorical features
        merged = self.encode_categorical_features(merged)
        self.check_memory_usage()
        
        # Add site identifier
        merged['site'] = site
        
        # Ensure consistent schema across all sites
        merged = self.ensure_consistent_schema(merged)
        
        print(f"  ✅ {site}: {len(merged)} rows, {len(merged.columns)} columns")
        return merged
    
    def _process_site_streaming_adaptive(self, site):
        """True streaming processing using adaptive settings - no pre-loading"""
        
        # Load metadata first
        metadata = self.load_metadata(site)
        
        # EARLY VALIDATION: Check sap flow data before loading environmental data
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        # Check if sap flow file exists
        if not os.path.exists(sapf_file):
            print(f"  ❌ Sap flow file not found: {sapf_file}")
            return None
        
        # Quick validation of sap flow data structure
        sapf_validation = self.validate_sap_flow_data(sapf_file)
        if not sapf_validation['valid']:
            print(f"  ❌ Sap flow data validation failed: {sapf_validation['reason']}")
            return None
        
        # Process in adaptive chunks
        chunk_size = self.adaptive_settings['chunk_size']
        file_extension = self.get_output_file_extension()
        output_file = f'{self.output_dir}/{site}_comprehensive{file_extension}'
        
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
            self.check_memory_usage()
        
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
            
            # Create features (simplified for streaming)
                merged = self.create_advanced_temporal_features(merged, timestamp_col)
            
            # Add basic metadata features
            merged = self.create_metadata_features(merged, metadata)
            
            # Add site identifier
            merged['site'] = site
            
            # Save chunk with optimized I/O
            self.save_streaming_chunk_optimized(merged, output_file, is_first_chunk=first_chunk)
            if first_chunk:
                first_chunk = False
            
            total_processed += len(merged)
            
            # Memory cleanup
            del env_chunk, merged
            self.check_memory_usage()
            
            # Continue processing all available data
        
        # Clean up time index
        del sapf_time_index
        self.check_memory_usage()
        
        if total_processed > 0:
            return True
        else:
            return None
    

    
    def create_lagged_features_adaptive(self, df, env_cols):
        """Create lagged features using adaptive settings - ensures consistent schema"""
        
        features = df.copy()
        max_lag = self.adaptive_settings['max_lag_hours']
        
        # Create lags based on adaptive settings
        if max_lag >= 24:
            lags = [1, 2, 3, 6, 12, 24]
        elif max_lag >= 12:
            lags = [1, 2, 3, 6, 12]
        else:
            lags = [1, 2, 3, 6]
        
        # Define expected environmental variables (all sites should have these)
        expected_env_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        
        for col in expected_env_cols:
            for lag in lags:
                if lag <= max_lag:
                    lag_col_name = f'{col}_lag_{lag}h'
                    if col in df.columns:
                        # Column exists - create lag
                        features[lag_col_name] = df[col].shift(lag)
                    else:
                        # Column missing - fill with NA
                        features[lag_col_name] = np.nan
        
        return features
    
    def create_rolling_features_adaptive(self, df, env_cols):
        """Create rolling features using adaptive settings - ensures consistent schema"""
        
        features = df.copy()
        windows = self.adaptive_settings['rolling_windows']
        
        # Define expected environmental variables for rolling features
        expected_rolling_cols = ['ta', 'vpd', 'sw_in', 'rh']
        
        for col in expected_rolling_cols:
            for window in windows:
                mean_col_name = f'{col}_mean_{window}h'
                std_col_name = f'{col}_std_{window}h'
                
                if col in df.columns:
                    # Column exists - create rolling features
                    features[mean_col_name] = df[col].rolling(window, min_periods=1).mean()
                    features[std_col_name] = df[col].rolling(window, min_periods=1).std()
                else:
                    # Column missing - fill with NA
                    features[mean_col_name] = np.nan
                    features[std_col_name] = np.nan
        
        return features
    
    def determine_adaptive_settings(self, site):
        """Automatically determine optimal processing settings for a site"""
        
        env_file = f'sapwood/{site}_env_data.csv'
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        if not os.path.exists(env_file) or not os.path.exists(sapf_file):
            print(f"  ❌ Missing files for {site}")
            return False
        
        # Get file sizes
        env_size_mb = os.path.getsize(env_file) / (1024 * 1024)
        sapf_size_mb = os.path.getsize(sapf_file) / (1024 * 1024)
        total_size_mb = env_size_mb + sapf_size_mb
        
        # Get current memory availability
        current_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Adapt processing strategy based on memory and file size
        if current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_low') and total_size_mb > ProcessingConfig.get_file_size_threshold('small'):
            # Very low memory + large files: Use streaming
            use_streaming = True
            reason = f"Very low memory ({current_memory_gb:.1f}GB) + large files ({total_size_mb:.1f}MB) - using streaming"
        elif current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_medium') and total_size_mb > ProcessingConfig.get_file_size_threshold('medium'):
            # Low memory + large files: Use streaming
            use_streaming = True
            reason = f"Low memory ({current_memory_gb:.1f}GB) + large files ({total_size_mb:.1f}MB) - using streaming"
        elif total_size_mb > ProcessingConfig.get_file_size_threshold('large'):
            # Very large files: Use streaming regardless of memory
            use_streaming = True
            reason = f"Very large files ({total_size_mb:.1f}MB > {ProcessingConfig.get_file_size_threshold('large')}MB threshold) - using streaming"
        else:
            # Standard processing
            use_streaming = False
            reason = f"Standard processing (memory: {current_memory_gb:.1f}GB, files: {total_size_mb:.1f}MB)"
        
        self.adaptive_settings['use_streaming'] = use_streaming
        self.stats['adaptive_decisions'].append(f"{site}: streaming={use_streaming} ({reason})")
        
        # Adapt chunk size based on memory and file size
        if current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_low'):
            # Very low memory: Use very small chunks
            chunk_size = ProcessingConfig.get_chunk_size('very_low_memory')
        elif current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_medium'):
            # Low memory: Use small chunks
            chunk_size = ProcessingConfig.get_chunk_size('low_memory')
        elif total_size_mb > ProcessingConfig.get_file_size_threshold('large'):
            # Large files: Use moderate chunks
            chunk_size = ProcessingConfig.get_chunk_size('medium_memory')
        else:
            # Standard processing: Use larger chunks
            chunk_size = ProcessingConfig.get_chunk_size('high_memory')
        
        self.adaptive_settings['chunk_size'] = chunk_size
        
        # Always create full lag features regardless of file size
        max_lag_hours = ProcessingConfig.get_feature_setting('max_lag_hours')
        self.adaptive_settings['max_lag_hours'] = max_lag_hours
        
        # Always create full rolling windows regardless of file size
        rolling_windows = ProcessingConfig.get_feature_setting('rolling_windows')
        self.adaptive_settings['rolling_windows'] = rolling_windows
        
        # Always create all features - adapt processing strategy based on memory
        memory_threshold = ProcessingConfig.get_memory_threshold('memory_threshold')
        if current_memory_gb < memory_threshold:
            # Low memory: Use more aggressive memory management but still create all features
            create_advanced_features = True
            create_domain_features = True
            reason = f"Low memory ({current_memory_gb:.1f}GB < {memory_threshold}GB) - using aggressive memory management"
        else:
            # High memory: Use standard processing
            create_advanced_features = True
            create_domain_features = True
            reason = f"Sufficient memory ({current_memory_gb:.1f}GB >= {memory_threshold}GB) - standard processing"
        
        self.adaptive_settings['create_advanced_features'] = create_advanced_features
        self.adaptive_settings['create_domain_features'] = create_domain_features
        self.stats['adaptive_decisions'].append(f"{site}: complete feature creation enabled ({reason})")
        
        # Adapt row limits based on memory constraints
        if current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_low') and total_size_mb > ProcessingConfig.get_file_size_threshold('medium'):
            # Very low memory + large files: Limit rows
            max_rows = ProcessingConfig.get_row_limit('very_low_memory')
            reason = f"Very low memory ({current_memory_gb:.1f}GB) + large files - limiting to {ProcessingConfig.get_row_limit('very_low_memory'):,} rows"
        elif current_memory_gb < ProcessingConfig.get_memory_threshold('streaming_medium') and total_size_mb > ProcessingConfig.get_file_size_threshold('large'):
            # Low memory + very large files: Limit rows
            max_rows = ProcessingConfig.get_row_limit('low_memory')
            reason = f"Low memory ({current_memory_gb:.1f}GB) + very large files - limiting to {ProcessingConfig.get_row_limit('low_memory'):,} rows"
        else:
            # Process all rows
            max_rows = None
            reason = f"Sufficient memory - processing all rows"
        
        self.adaptive_settings['max_rows_per_site'] = max_rows
        self.stats['adaptive_decisions'].append(f"{site}: row_limit={max_rows if max_rows else 'None'} ({reason})")
        
        return True
    
    def ensure_consistent_schema(self, df):
        """Ensure consistent schema across all sites by adding missing columns with NA values"""
        
        # First, remove any problematic columns that might have been missed
        problematic_columns = [
            'pl_name', 'swc_deep', 'netrad', 'seasonal_leaf_area', 'seasonal_leaf_area_code',
            'stand_name_code', 'stand_remarks_code', 'site_remarks_code', 'env_remarks_code',
            'water_stress_index', 'moisture_availability', 'swc_shallow_depth',
            # Redundant features (can be computed during training)
            'wind_stress', 'wind_vpd_interaction', 'soil_texture_code', 'stand_soil_texture_code',
            'light_efficiency', 'ppfd_efficiency', 'stomatal_conductance_proxy', 'stomatal_control_index',
            'vpd_ppfd_interaction', 'precip_intensity', 'recent_precip_1h', 'recent_precip_6h', 
            'recent_precip_24h', 'aspect_code', 'species_basal_area_perc', 'site_paper_code',
            'terrain_code', 'temp_humidity_ratio', 'daylight_time'
        ]
        
        # Ensure inconsistent columns are properly handled for XGBoost
        # These columns may be missing in some files but should be preserved as NaN
        xgboost_missing_columns = ['leaf_habit_code', 'soil_depth']
        for col in problematic_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"    🚫 Removed problematic column from final schema: {col}")
        
        # Define the complete expected schema
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
            
            # Solar timestamp features (useful when available)
            'solar_hour': np.nan, 'solar_day_of_year': np.nan, 'solar_hour_sin': np.nan,
            'solar_hour_cos': np.nan, 'solar_day_sin': np.nan, 'solar_day_cos': np.nan,
        }
        
        # Add missing columns with NA values
        for col, default_value in expected_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        # Add XGBoost-inconsistent columns if missing
        for col in xgboost_missing_columns:
            if col not in df.columns:
                df[col] = np.nan
                print(f"    🔧 Added missing XGBoost column: {col}")
            else:
                # Ensure any invalid values are converted to NaN for XGBoost
                if df[col].dtype == 'object':
                    # For categorical columns, replace empty strings or invalid values with NaN
                    df[col] = df[col].replace(['', 'nan', 'None', 'NULL'], np.nan)
                else:
                    # For numeric columns, ensure inf/-inf are converted to NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                print(f"    🔧 Ensured XGBoost compatibility for: {col}")
        
        return df

def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process SAPFLUXNET data with complete feature creation')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of all sites (ignore existing files)')
    parser.add_argument('--output-dir', default='comprehensive_processed',
                       help='Output directory for processed files (default: comprehensive_processed)')
    parser.add_argument('--chunk-size', type=int, default=1500,
                       help='Chunk size for processing (default: 1500)')
    parser.add_argument('--max-memory', type=int, default=12,
                       help='Maximum memory usage in GB (default: 12)')
    parser.add_argument('--include-problematic', action='store_true',
                       help='Include problematic sites with high quality flag rates (sites with no valid data are always excluded)')
    parser.add_argument('--no-quality-flags', action='store_true',
                       help='Disable quality flag filtering (default: filter out OUT_WARN and RANGE_WARN data points)')
    parser.add_argument('--compress', action='store_true',
                       help='Compress output files with gzip (default: uncompressed)')
    parser.add_argument('--no-io-optimization', action='store_true',
                       help='Disable I/O optimization (default: enabled)')
    parser.add_argument('--memory-threshold', type=float, default=None,
                       help='Override memory threshold for aggressive memory management (GB, default: 6.0) - all features still created')
    parser.add_argument('--file-size-threshold', type=int, default=None,
                       help='Override file size threshold for streaming (MB)')
    parser.add_argument('--chunk-size-override', type=int, default=None,
                       help='Override chunk size for processing')
    parser.add_argument('--max-lag-hours', type=int, default=None,
                       help='Override maximum lag hours for feature creation')
    parser.add_argument('--rolling-windows', type=str, default=None,
                       help='Override rolling windows (comma-separated list, e.g., "3,6,12,24")')
    parser.add_argument('--export-format', choices=['csv', 'parquet', 'feather', 'hdf5', 'pickle'], default='csv',
                       help='Export format for processed files (default: csv)')

    
    args = parser.parse_args()
    
    print("🚀 Starting Adaptive Complete SAPFLUXNET Processing")
    print(f"⏰ Started at: {datetime.now()}")
    
    if args.force:
        print("🔄 Force reprocessing mode enabled")
    
    if not args.no_quality_flags:
        print("🏷️  Quality flag filtering enabled (removing OUT_WARN and RANGE_WARN data points)")
    else:
        print("⚠️  Quality flag filtering disabled (keeping all data points)")
    
    # Prepare configuration overrides
    config_overrides = {}
    if args.memory_threshold:
        config_overrides['memory_threshold'] = args.memory_threshold
    if args.file_size_threshold:
        config_overrides['file_size_threshold'] = args.file_size_threshold
    if args.chunk_size_override:
        config_overrides['chunk_size_override'] = args.chunk_size_override
    if args.max_lag_hours:
        config_overrides['max_lag_hours'] = args.max_lag_hours
    if args.rolling_windows:
        config_overrides['rolling_windows'] = args.rolling_windows
    
    # Create processor with adaptive memory management and I/O optimization
    processor = MemoryEfficientSAPFLUXNETProcessor(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_memory_gb=args.max_memory,
        force_reprocess=args.force,
        skip_problematic_sites=not args.include_problematic,
        use_quality_flags=not args.no_quality_flags,
        compress_output=args.compress,
        optimize_io=not args.no_io_optimization,
        export_format=args.export_format,
        config_overrides=config_overrides
    )
    
    # Process all sites with complete feature creation and adaptive memory management
    result = processor.process_all_sites()
    
    if result:
        print(f"\n✅ Adaptive complete processing completed successfully!")
        print(f"📄 Output directory: {processor.output_dir}")
        print(f"📊 All sites processed with complete feature sets")
        print(f"🧠 Memory management adapted to system capabilities")
        print(f"💾 I/O optimization: {'Enabled' if processor.optimize_io else 'Disabled'}")
        print(f"🗜️  Compression: {'Enabled' if processor.compress_output else 'Disabled'}")
        print(f"📁 Export format: {processor.export_format.upper()}")
        if processor.use_quality_flags:
            print(f"🏷️  Quality flag filtering applied (OUT_WARN and RANGE_WARN data removed)")
        else:
            print(f"⚠️  Quality flag filtering disabled (all data points included)")
        
        # Print excluded sites summary
        print(f"\n🚫 Excluded sites:")
        print(f"  - No valid sap flow data: {len(processor.SITES_WITH_NO_VALID_DATA)} sites")
        if processor.skip_problematic_sites:
            print(f"  - Extremely problematic (>50% flag rate): {len(processor.EXTREMELY_PROBLEMATIC_SITES)} sites")
            print(f"  - High problematic (20-50% flag rate): {len(processor.HIGH_PROBLEMATIC_SITES)} sites")
            print(f"  - Moderate problematic (10-20% flag rate): {len(processor.MODERATE_PROBLEMATIC_SITES)} sites")
            print(f"  - Total problematic sites: {len(processor.PROBLEMATIC_SITES)} sites")
            print(f"  💡 Use --include-problematic to process problematic sites anyway")
        else:
            print(f"  - Problematic sites included: {len(processor.PROBLEMATIC_SITES)} sites")
        
    else:
        print(f"\n❌ Adaptive complete processing failed")
        print(f"💡 Check memory availability and file integrity")
    
    print(f"\n⏰ Finished at: {datetime.now()}")

if __name__ == "__main__":
    main() 