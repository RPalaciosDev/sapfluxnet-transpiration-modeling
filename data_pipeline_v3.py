import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import gc
import warnings
import psutil
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
warnings.filterwarnings('ignore')

# ===============================================================================
# 🚨 CRITICAL OVERFITTING PROTECTION IMPLEMENTED (January 2025)
# ===============================================================================
# 
# This pipeline includes critical fixes to prevent site identity memorization
# and geographic proxy overfitting that was causing catastrophic spatial 
# generalization failure in machine learning models.
#
# FIXED ISSUES:
# 1. 🚨 Site Identity Memorization: site_code_code was causing models to learn
#    "if site=X, predict Y" instead of ecological relationships
# 2. ⚠️  Geographic Proxy Overfitting: timezone_code and country_code were
#    causing regional overfitting that hindered cross-regional generalization
# 3. 🛡️  Enhanced Safety Checks: Added pattern detection for identity features
#
# KEY CHANGES in encode_categorical_features():
# - IDENTITY_BLACKLIST: Prevents encoding of site_code, site_name, species_name
# - GEOGRAPHIC_PROXY_FEATURES: Blocks timezone, country encoding 
# - APPROVED_ECOLOGICAL_FEATURES: Only allows ecological categorical encoding
# - Enhanced logging: Shows what features are blocked and why
#
# IMPACT: This fixes Cluster 4's catastrophic failure (R² = -0.4085) and
# prevents future overfitting issues across all ecosystem clusters.
#
# ⚠️  DO NOT REMOVE THESE PROTECTIONS - They prevent catastrophic model failure
# ===============================================================================
#
# ===============================================================================
# 📅 DYNAMIC SITE QUALITY ANALYSIS IMPLEMENTED (January 2025)
# ===============================================================================
#
# Replaced hardcoded site exclusion lists with dynamic data quality analysis
# that examines actual site data to determine temporal coverage and data validity.
#
# DYNAMIC ANALYSIS FEATURES:
# - 🔍 Automated site discovery from sapwood directory
# - 📊 Real-time calculation of temporal coverage for each site
# - 🧪 Validation of sap flow data existence and quality
# - 📁 Comprehensive reporting with detailed analysis results
# - 🏷️  Dynamic categorization based on actual data characteristics
#
# TEMPORAL COVERAGE CATEGORIES (dynamically determined):
# - 🚫 NO VALID DATA: Sites missing env/sapf files or invalid data structure
# - 📉 INSUFFICIENT (<30 days): Excluded from spatial modeling
# - ⚠️  MODERATE (30-90 days): Processed with warnings about limited seasonal representation
# - ✅ ADEQUATE (≥90 days): Optimal for spatial modeling
#
# BENEFITS:
# - Eliminates dependency on hardcoded lists that become outdated
# - Automatically adapts to new data or changes in existing datasets
# - Provides detailed analysis reports for understanding site characteristics
# - Enables targeted data collection efforts for problematic sites
# - Improves maintainability and accuracy of site classification
#
# USE --analyze-only FLAG to run site analysis without processing
# Results saved to site_analysis_results/ with JSON, CSV, and TXT formats
# ===============================================================================

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

class MemoryEfficientSAPFLUXNETProcessor:
    """Complete SAPFLUXNET processor with adaptive memory management - creates all features consistently"""
    
    # Dynamic site classification - populated by analyze_site_data_quality()
    SITES_WITH_NO_VALID_DATA = set()
    SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE = set()
    SITES_WITH_MODERATE_TEMPORAL_COVERAGE = set()
    SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE = set()
    
    # Site analysis results storage
    site_analysis_results = {}
    
    # Sites with extremely high quality flag rates (>80%) - should be excluded
    EXTREMELY_PROBLEMATIC_SITES = {
        'IDN_PON_STE',  # 63.1% flag rate - Extremely poor quality
        'USA_NWH',  # 53.4% flag rate - Very poor quality
    }
    
    # Sites with high quality flag rates (50-80%) - should be excluded in clean mode
    HIGH_PROBLEMATIC_SITES = {
        'ZAF_NOO_E3_IRR',  # 25.9% flag rate - Very poor quality  
        'GUF_GUY_GUY',  # 35.5% flag rate - Very poor quality
        'USA_TNP',  # 31.6% flag rate - Very poor quality
        'USA_TNY',  # 28.9% flag rate - Very poor quality
        'USA_WVF',  # 16.6% flag rate - Very poor quality
    }
    
    # Sites with moderate quality issues (20-50%) - should be processed with warnings
    MODERATE_PROBLEMATIC_SITES = {
        'USA_SYL_HL2',  # 16.0% flag rate - Poor quality
        'USA_WIL_WC2',  # 13.3% flag rate - Poor quality
        'CAN_TUR_P39_POS',  # 13.2% flag rate - Poor quality
        'CAN_TUR_P74',  # 15.8% flag rate - Poor quality
        'USA_PAR_FER',  # 16.7% flag rate - Poor quality
        'USA_TNB',  # 19.4% flag rate - Poor quality
        'USA_TNO',  # 19.3% flag rate - Poor quality
        'USA_UMB_GIR',  # 27.9% flag rate - Poor quality
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
    
    # Combined list of all problematic sites (for quality flags only)
    PROBLEMATIC_SITES = EXTREMELY_PROBLEMATIC_SITES | HIGH_PROBLEMATIC_SITES | MODERATE_PROBLEMATIC_SITES
    
    def get_excluded_sites(self):
        """Get combined list of all sites to exclude from processing (dynamic)"""
        return self.SITES_WITH_NO_VALID_DATA | self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE
    
    def __init__(self, output_dir='comprehensive_processed', chunk_size=1000, max_memory_gb=12, force_reprocess=False, skip_problematic_sites=True, use_quality_flags=True, compress_output=False, optimize_io=True, export_format='csv', config_overrides=None, clean_mode=False):
        self.base_output_dir = output_dir
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb  # Leave 4GB buffer
        self.force_reprocess = force_reprocess  # Force reprocessing of all sites
        self.skip_problematic_sites = skip_problematic_sites  # Whether to skip problematic sites
        self.use_quality_flags = use_quality_flags  # Whether to filter out flagged data points
        self.compress_output = compress_output  # Whether to compress output files
        self.optimize_io = optimize_io  # Whether to use optimized I/O strategies
        self.export_format = export_format.lower()  # Export format: csv, parquet, feather, hdf5, pickle, libsvm
        self.clean_mode = clean_mode  # Clean mode: only exclude extremely problematic sites
        
        # Validate export format
        self.validate_export_format()
        
        # Create format-specific output directory
        self.output_dir = self.get_format_specific_output_dir()
        
        # Apply configuration overrides if provided
        if config_overrides:
            self.apply_config_overrides(config_overrides)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        valid_formats = ['csv', 'parquet', 'feather', 'hdf5', 'pickle', 'libsvm']
        
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
        
        elif self.export_format == 'libsvm':
            try:
                from sklearn.datasets import dump_svmlight_file
            except ImportError:
                raise ImportError("scikit-learn is required for libsvm export. Install with: pip install scikit-learn")
    
    def get_output_file_extension(self):
        """Get the appropriate file extension for the export format"""
        extensions = {
            'csv': '.csv',
            'parquet': '.parquet',
            'feather': '.feather',
            'hdf5': '.h5',
            'pickle': '.pkl',
            'libsvm': '.svm'
        }
        return extensions.get(self.export_format, '.csv')
    
    def get_format_specific_output_dir(self):
        """Get format-specific output directory name"""
        if self.base_output_dir == 'comprehensive_processed':
            # Use format-specific directory for default case
            return f'processed_{self.export_format}'
        else:
            # For custom directories, append format suffix
            return f'{self.base_output_dir}_{self.export_format}'
    
    def save_dataframe_formatted(self, df, output_file, site_name):
        """Save DataFrame in the specified format with optimized I/O"""
        
        # Add compression extension if needed
        if self.compress_output and self.export_format in ['csv', 'libsvm']:
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
                
            elif self.export_format == 'libsvm':
                self._save_libsvm_format(df, output_file, site_name)
            
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
    
    def _save_libsvm_format(self, df, output_file, site_name):
        """Save DataFrame in libsvm format for XGBoost external memory"""
        from sklearn.datasets import dump_svmlight_file
        import gzip
        import json
        
        # Identify target column
        target_col = 'sap_flow'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data for {site_name}")
        
        # Define columns to exclude from features
        exclude_cols = [
            'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
            'Unnamed: 0', target_col
        ]
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        # Save feature mapping (do this once per directory)
        output_dir = os.path.dirname(output_file)
        feature_mapping_file = os.path.join(output_dir, 'feature_mapping.json')
        
        if not os.path.exists(feature_mapping_file):
            # Create comprehensive feature mapping
            feature_mapping = {
                'target_column': target_col,
                'excluded_columns': exclude_cols,
                'feature_count': len(feature_cols),
                'features': {
                    # Map feature index to feature name
                    f'f{i}': feature_name for i, feature_name in enumerate(feature_cols)
                },
                'feature_names': feature_cols,  # List of feature names in order
                'created_by': 'comprehensive_processing_pipeline.py',
                'created_at': pd.Timestamp.now().isoformat(),
                'format': 'libsvm'
            }
            
            # Save feature mapping to JSON file
            with open(feature_mapping_file, 'w') as f:
                json.dump(feature_mapping, f, indent=2)
            
            print(f"    🗂️  Feature mapping saved: {feature_mapping_file}")
        
        # Prepare data
        df_clean = df.dropna(subset=[target_col])
        
        if len(df_clean) == 0:
            raise ValueError(f"No valid data after removing missing target values for {site_name}")
        
        # Fill missing values in features
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        # Save to libsvm format
        if self.compress_output:
            # Save to temporary file first, then compress
            temp_file = output_file.replace('.gz', '')
            dump_svmlight_file(X_array, y_array, temp_file)
            
            # Compress the file
            with open(temp_file, 'rb') as f_in:
                with gzip.open(output_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove temporary file
            os.remove(temp_file)
        else:
            dump_svmlight_file(X_array, y_array, output_file)
        
        print(f"    📊 LibSVM: {len(df_clean)} samples, {len(feature_cols)} features")
    
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
    
    def analyze_site_data_quality(self, site):
        """Analyze a single site's data quality and temporal coverage"""
        analysis_result = {
            'site': site,
            'has_env_data': False,
            'has_sapf_data': False,
            'has_valid_sapf_data': False,
            'temporal_coverage_days': 0,
            'sapf_columns_count': 0,
            'total_records': 0,
            'valid_sapf_records': 0,
            'data_quality_issues': [],
            'category': 'unknown',
            'exclude_reason': None
        }
        
        # Check if environmental data exists
        env_file = f'sapwood/{site}_env_data.csv'
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        if not os.path.exists(env_file):
            analysis_result['exclude_reason'] = 'No environmental data file'
            analysis_result['category'] = 'no_env_data'
            return analysis_result
        
        if not os.path.exists(sapf_file):
            analysis_result['exclude_reason'] = 'No sap flow data file'
            analysis_result['category'] = 'no_sapf_data'
            return analysis_result
        
        analysis_result['has_env_data'] = True
        analysis_result['has_sapf_data'] = True
        
        try:
            # Analyze sap flow data validity
            sapf_validation = self.validate_sap_flow_data(sapf_file)
            if not sapf_validation['valid']:
                analysis_result['exclude_reason'] = sapf_validation['reason']
                analysis_result['category'] = 'invalid_sapf_data'
                return analysis_result
            
            analysis_result['has_valid_sapf_data'] = True
            analysis_result['sapf_columns_count'] = len(sapf_validation['columns'])
            
            # Calculate temporal coverage
            temporal_info = self.calculate_temporal_coverage(site)
            if temporal_info['valid']:
                analysis_result['temporal_coverage_days'] = temporal_info['days']
            else:
                analysis_result['data_quality_issues'].append(f"Could not calculate temporal coverage: {temporal_info['reason']}")
                analysis_result['temporal_coverage_days'] = 0
            
            # Count total records (quick sample)
            try:
                sapf_sample = pd.read_csv(sapf_file, nrows=1000)
                # Estimate total records
                file_size = os.path.getsize(sapf_file)
                estimated_total = int((file_size / 1024) * 10)  # Rough estimate
                analysis_result['total_records'] = max(len(sapf_sample), estimated_total)
                
                # Count valid sap flow records
                sapf_cols = sapf_validation['columns']
                if len(sapf_cols) > 0:
                    valid_records = 0
                    for col in sapf_cols:
                        if col in sapf_sample.columns:
                            valid_records += sapf_sample[col].notna().sum()
                    analysis_result['valid_sapf_records'] = valid_records
                
            except Exception as e:
                analysis_result['data_quality_issues'].append(f"Error counting records: {str(e)}")
            
            # Categorize site based on analysis
            if analysis_result['temporal_coverage_days'] < 30:
                analysis_result['category'] = 'insufficient_temporal'
                analysis_result['exclude_reason'] = f"Insufficient temporal coverage ({analysis_result['temporal_coverage_days']:.1f} days < 30 days minimum)"
            elif 30 <= analysis_result['temporal_coverage_days'] < 90:
                analysis_result['category'] = 'moderate_temporal'
            else:
                analysis_result['category'] = 'adequate_temporal'
                
        except Exception as e:
            analysis_result['exclude_reason'] = f"Analysis error: {str(e)}"
            analysis_result['category'] = 'analysis_error'
            analysis_result['data_quality_issues'].append(str(e))
        
        return analysis_result
    
    def analyze_all_sites_data_quality(self):
        """Analyze data quality for all sites and populate classification sets"""
        print("🔍 Analyzing data quality for all sites...")
        
        all_sites = self.get_all_sites()
        print(f"📊 Found {len(all_sites)} potential sites")
        
        # Clear existing classifications
        self.SITES_WITH_NO_VALID_DATA.clear()
        self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE.clear()
        self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE.clear()
        self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE.clear()
        self.site_analysis_results.clear()
        
        # Analyze each site
        for i, site in enumerate(all_sites, 1):
            if i % 10 == 0 or i == len(all_sites):
                print(f"  📋 Analyzing site {i}/{len(all_sites)}: {site}")
            
            analysis = self.analyze_site_data_quality(site)
            self.site_analysis_results[site] = analysis
            
            # Classify site based on analysis
            if analysis['category'] in ['no_env_data', 'no_sapf_data', 'invalid_sapf_data', 'analysis_error']:
                self.SITES_WITH_NO_VALID_DATA.add(site)
            elif analysis['category'] == 'insufficient_temporal':
                self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE.add(site)
            elif analysis['category'] == 'moderate_temporal':
                self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE.add(site)
            elif analysis['category'] == 'adequate_temporal':
                self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE.add(site)
        
        # Print summary
        print(f"\n📊 Site Data Quality Analysis Complete:")
        print(f"  🚫 No valid data: {len(self.SITES_WITH_NO_VALID_DATA)} sites")
        print(f"  📉 Insufficient temporal (<30 days): {len(self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE)} sites")
        print(f"  ⚠️  Moderate temporal (30-90 days): {len(self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE)} sites")
        print(f"  ✅ Adequate temporal (≥90 days): {len(self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE)} sites")
        
        # Save detailed analysis results
        self.save_site_analysis_results()
        
        return {
            'total_sites': len(all_sites),
            'no_valid_data': len(self.SITES_WITH_NO_VALID_DATA),
            'insufficient_temporal': len(self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE),
            'moderate_temporal': len(self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE),
            'adequate_temporal': len(self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE)
        }
    
    def save_site_analysis_results(self):
        """Save detailed site analysis results to files"""
        output_dir = Path("site_analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = output_dir / f"site_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json_data = {
                'analysis_timestamp': timestamp,
                'total_sites_analyzed': len(self.site_analysis_results),
                'sites_with_no_valid_data': sorted(list(self.SITES_WITH_NO_VALID_DATA)),
                'sites_with_insufficient_temporal_coverage': sorted(list(self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE)),
                'sites_with_moderate_temporal_coverage': sorted(list(self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE)),
                'sites_with_adequate_temporal_coverage': sorted(list(self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE)),
                'detailed_results': self.site_analysis_results
            }
            json.dump(json_data, f, indent=2, default=str)
        
        # Save CSV summary
        csv_file = output_dir / f"site_analysis_summary_{timestamp}.csv"
        summary_data = []
        for site, analysis in self.site_analysis_results.items():
            summary_data.append({
                'site': site,
                'category': analysis['category'],
                'temporal_coverage_days': analysis['temporal_coverage_days'],
                'has_env_data': analysis['has_env_data'],
                'has_sapf_data': analysis['has_sapf_data'],
                'has_valid_sapf_data': analysis['has_valid_sapf_data'],
                'sapf_columns_count': analysis['sapf_columns_count'],
                'exclude_reason': analysis['exclude_reason'],
                'data_quality_issues': '; '.join(analysis['data_quality_issues']) if analysis['data_quality_issues'] else ''
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        
        # Save detailed report
        report_file = output_dir / f"site_analysis_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SAPFLUXNET Site Data Quality Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Sites Analyzed: {len(self.site_analysis_results)}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"✅ Sites with adequate temporal coverage (≥90 days): {len(self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE)}\n")
            f.write(f"⚠️  Sites with moderate temporal coverage (30-90 days): {len(self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE)}\n")
            f.write(f"📉 Sites with insufficient temporal coverage (<30 days): {len(self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE)}\n")
            f.write(f"🚫 Sites with no valid data: {len(self.SITES_WITH_NO_VALID_DATA)}\n\n")
            
            # Detailed breakdowns
            if self.SITES_WITH_NO_VALID_DATA:
                f.write("SITES WITH NO VALID DATA\n")
                f.write("-" * 25 + "\n")
                for site in sorted(self.SITES_WITH_NO_VALID_DATA):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['exclude_reason']}\n")
                f.write("\n")
            
            if self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE:
                f.write("SITES WITH INSUFFICIENT TEMPORAL COVERAGE (<30 days)\n")
                f.write("-" * 50 + "\n")
                for site in sorted(self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['temporal_coverage_days']:.1f} days\n")
                f.write("\n")
            
            if self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE:
                f.write("SITES WITH MODERATE TEMPORAL COVERAGE (30-90 days)\n")
                f.write("-" * 47 + "\n")
                for site in sorted(self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['temporal_coverage_days']:.1f} days\n")
                f.write("\n")
        
        print(f"📁 Site analysis results saved:")
        print(f"  📄 JSON: {json_file}")
        print(f"  📊 CSV: {csv_file}")
        print(f"  📝 Report: {report_file}")
    
    def should_skip_site(self, site):
        """Check if site has already been processed and should be skipped"""
        # Use dynamic analysis results if available
        if site in self.site_analysis_results:
            analysis = self.site_analysis_results[site]
            
            # Check if site has no valid data - always skip these
            if site in self.SITES_WITH_NO_VALID_DATA:
                print(f"  🚫 Skipping {site} - {analysis['exclude_reason']}")
                return True
            
            # Check if site has insufficient temporal coverage (<30 days) - always skip these
            if site in self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE:
                print(f"  🚫 Skipping {site} - Insufficient temporal coverage ({analysis['temporal_coverage_days']:.1f} days < 30 days)")
                return True
            
            # Sites with moderate coverage - process with warnings
            if site in self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE:
                print(f"  ⚠️  Processing {site} with temporal coverage warning - Moderate coverage ({analysis['temporal_coverage_days']:.1f} days)")
            
            # Sites with adequate coverage - process normally
            if site in self.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE:
                print(f"  ✅ {site} has adequate temporal coverage ({analysis['temporal_coverage_days']:.1f} days)")
        
        else:
            # Fallback to individual analysis if site wasn't analyzed yet
            print(f"  ⚠️  {site} not in analysis results, performing individual check...")
            
            # Check if site has no valid sap flow data - always skip these
            if site in self.SITES_WITH_NO_VALID_DATA:
                print(f"  🚫 Skipping {site} - No valid sap flow data (from previous analysis)")
                return True
            
            # Check if site has insufficient temporal coverage (<30 days) - always skip these
            if site in self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE:
                print(f"  🚫 Skipping {site} - Insufficient temporal coverage (<30 days, from previous analysis)")
                return True
            
            # Sites with moderate coverage - process with warnings
            if site in self.SITES_WITH_MODERATE_TEMPORAL_COVERAGE:
                print(f"  ⚠️  Processing {site} with temporal coverage warning - Moderate coverage (30-90 days, from previous analysis)")
        
        # Check if site is problematic and should be skipped (quality flags)
        if self.skip_problematic_sites and site in self.PROBLEMATIC_SITES:
            if site in self.EXTREMELY_PROBLEMATIC_SITES:
                print(f"  🚫 Skipping {site} - Extremely problematic site (>80% flag rate)")
                return True
            elif hasattr(self, 'clean_mode') and self.clean_mode:
                # In clean mode, only exclude extremely problematic sites
                if site in self.HIGH_PROBLEMATIC_SITES:
                    print(f"  ⚠️  Processing {site} in clean mode - High problematic site (50-80% flag rate)")
                elif site in self.MODERATE_PROBLEMATIC_SITES:
                    print(f"  ⚠️  Processing {site} in clean mode - Moderate quality issues (20-50% flag rate)")
            else:
                # Normal mode - exclude both high and extremely problematic sites
                if site in self.HIGH_PROBLEMATIC_SITES:
                    print(f"  🚫 Skipping {site} - High problematic site (50-80% flag rate)")
                    return True
                elif site in self.MODERATE_PROBLEMATIC_SITES:
                    print(f"  ⚠️  Processing {site} with warnings - Moderate quality issues (20-50% flag rate)")
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
            # Check for compressed version first (for CSV and libsvm)
            actual_file_path = file_path
            if self.compress_output and self.export_format in ['csv', 'libsvm'] and os.path.exists(file_path + '.gz'):
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
            elif self.export_format == 'libsvm':
                return self._validate_libsvm_file(actual_file_path)
            
            # Check if file has expected structure (not applicable for libsvm)
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
    
    def _validate_libsvm_file(self, file_path):
        """Validate libsvm format file"""
        try:
            from sklearn.datasets import load_svmlight_file
            import gzip
            
            # Try to load first few lines to check format
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    lines = [f.readline() for _ in range(5)]
            else:
                with open(file_path, 'r') as f:
                    lines = [f.readline() for _ in range(5)]
            
            # Check if lines look like libsvm format
            valid_lines = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Basic libsvm format check: should start with target value
                    parts = line.split(' ', 1)
                    if len(parts) >= 1:
                        try:
                            float(parts[0])  # Target should be numeric
                            valid_lines += 1
                        except ValueError:
                            pass
            
            # If we have at least 3 valid lines, consider it valid
            return valid_lines >= 3
            
        except Exception as e:
            return False
    
    def calculate_temporal_coverage(self, site):
        """Calculate temporal coverage duration for a site in days"""
        try:
            env_file = f'sapwood/{site}_env_data.csv'
            
            if not os.path.exists(env_file):
                return {'valid': False, 'days': 0, 'reason': 'Environmental file not found'}
            
            # Read a small sample to get timestamp column and check structure
            sample = pd.read_csv(env_file, nrows=100)
            
            if len(sample) == 0:
                return {'valid': False, 'days': 0, 'reason': 'Empty environmental file'}
            
            # Find timestamp column
            timestamp_cols = [col for col in sample.columns if 'timestamp' in col.lower() and not col.lower().startswith('solar')]
            if not timestamp_cols:
                return {'valid': False, 'days': 0, 'reason': 'No timestamp column found in environmental data'}
            
            timestamp_col = timestamp_cols[0]
            
            # Read first and last few rows to get time range (memory efficient)
            try:
                # Read first rows
                first_chunk = pd.read_csv(env_file, nrows=50)
                first_chunk[timestamp_col] = pd.to_datetime(first_chunk[timestamp_col])
                start_time = first_chunk[timestamp_col].min()
                
                # Read last rows using tail approach
                # For efficiency, read file in reverse to get last timestamp
                total_lines = sum(1 for _ in open(env_file)) - 1  # -1 for header
                skip_lines = max(0, total_lines - 50)  # Read last 50 lines
                
                last_chunk = pd.read_csv(env_file, skiprows=range(1, skip_lines + 1))
                last_chunk[timestamp_col] = pd.to_datetime(last_chunk[timestamp_col])
                end_time = last_chunk[timestamp_col].max()
                
                # Calculate duration
                duration = (end_time - start_time).total_seconds() / (24 * 3600)  # Convert to days
                
                return {'valid': True, 'days': duration, 'reason': f'Duration: {duration:.1f} days', 'start': start_time, 'end': end_time}
                
            except Exception as e:
                return {'valid': False, 'days': 0, 'reason': f'Error calculating duration: {str(e)}'}
                
        except Exception as e:
            return {'valid': False, 'days': 0, 'reason': f'Error reading environmental file: {str(e)}'}

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
        
        # Memory cleanup after loading
        self.check_memory_usage()
        
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
                    # Memory cleanup even when no flags
                    self.check_memory_usage()
                    return data
                    
            except Exception as e:
                # Memory cleanup on error
                self.check_memory_usage()
                return data
        else:
            # Memory cleanup when no flags file
            self.check_memory_usage()
            return data
    
    def create_all_features(self, df, metadata, timestamp_col, processing_mode='standard'):
        """
        Consolidated feature engineering orchestrator with phase-based logging.
        
        This method centralizes all feature creation and provides high-level progress updates
        instead of verbose per-feature logging, improving user experience.
        
        Args:
            df: Input dataframe
            metadata: Site metadata
            timestamp_col: Timestamp column name
            processing_mode: 'standard' or 'streaming' to control feature set
        """
        print("    🔧 Engineering Features...")
        
        # Use in-place operations to avoid unnecessary copying
        features = df
        
        # Phase 1: Temporal Features
        features = self.create_advanced_temporal_features(features, timestamp_col)
        self.check_memory_usage()
        
        # Phase 2: Environmental Rolling & Lagged Features  
        env_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        features = self.create_rolling_features_adaptive(features, env_cols)
        self.check_memory_usage()
        
        if processing_mode == 'standard':
            features = self.create_lagged_features_adaptive(features, env_cols)
            self.check_memory_usage()
        
        # Phase 3: Advanced Environmental Features (standard mode only)
        if processing_mode == 'standard':
            features = self.create_advanced_rolling_features(features)
            self.check_memory_usage()
            
            features = self.create_rate_of_change_features(features)
            self.check_memory_usage()
            
            features = self.create_cumulative_features(features)
            self.check_memory_usage()
        
        # Phase 4: Interaction & Domain Features
        features = self.create_interaction_features(features)
        self.check_memory_usage()
        
        if processing_mode == 'standard' and self.adaptive_settings['create_domain_features']:
            features = self.create_domain_specific_features(features)
            self.check_memory_usage()
        
        # Phase 5: Metadata Features
        features = self.create_metadata_features(features, metadata)
        self.check_memory_usage()
        
        # Phase 6: Encoding Features
        print("    🔢 Encoding Features...")
        features = self.encode_categorical_features(features)
        self.check_memory_usage()
        
        # Phase 7: Dropping Problematic Columns
        print("    🗑️  Dropping Problematic Columns...")
        features = self.drop_problematic_columns(features)
        self.check_memory_usage()
        
        # Phase 8: Ensuring Data Compatibility
        print("    ✅ Ensuring Data Compatibility...")
        features = self.ensure_consistent_schema(features)
        self.check_memory_usage()
        
        print("    ✅ Feature Engineering Complete")
        return features
    
    def drop_problematic_columns(self, df):
        """Drop columns that could cause issues in modeling"""
        features = df.copy()
        
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
        for col in features.columns:
            col_lower = col.lower()
            for pattern in problematic_patterns:
                if pattern in col_lower and col != 'TIMESTAMP':
                    columns_to_drop.append(col)
                    break
        
        if columns_to_drop:
            features = features.drop(columns=columns_to_drop)
        
        return features
    
    def create_advanced_temporal_features(self, df, timestamp_col):
        """Create enhanced temporal features (reduced logging)"""
        
        features = df.copy()
        
        # Convert timestamp if not already done
        if not pd.api.types.is_datetime64_any_dtype(features[timestamp_col]):
            features[timestamp_col] = pd.to_datetime(features[timestamp_col])
        
        dt = features[timestamp_col].dt
        
        # Essential time features
        features['hour'] = dt.hour
        features['day_of_year'] = dt.dayofyear
        features['month'] = dt.month
        features['year'] = dt.year
        features['day_of_week'] = dt.dayofweek
        
        # Enhanced cyclical features
        hour_rad = 2 * np.pi * features['hour'] / 24
        features['hour_sin'] = np.sin(hour_rad)
        features['hour_cos'] = np.cos(hour_rad)
        
        day_rad = 2 * np.pi * features['day_of_year'] / 365
        features['day_sin'] = np.sin(day_rad)
        features['day_cos'] = np.cos(day_rad)
        
        month_rad = 2 * np.pi * features['month'] / 12
        features['month_sin'] = np.sin(month_rad)
        features['month_cos'] = np.cos(month_rad)
        
        # Solar timestamp features if available
        if 'solar_TIMESTAMP' in df.columns:
            solar_dt = pd.to_datetime(df['solar_TIMESTAMP']).dt
            features['solar_hour'] = solar_dt.hour
            features['solar_day_of_year'] = solar_dt.dayofyear
            
            # Solar-adjusted cyclical features
            solar_hour_rad = 2 * np.pi * features['solar_hour'] / 24
            features['solar_hour_sin'] = np.sin(solar_hour_rad)
            features['solar_hour_cos'] = np.cos(solar_hour_rad)
            
            solar_day_rad = 2 * np.pi * features['solar_day_of_year'] / 365
            features['solar_day_sin'] = np.sin(solar_day_rad)
            features['solar_day_cos'] = np.cos(solar_day_rad)
        
        # Enhanced boolean features
        features['is_daylight'] = ((features['hour'] >= 6) & (features['hour'] <= 18)).astype(int)
        features['is_peak_sunlight'] = ((features['hour'] >= 10) & (features['hour'] <= 16)).astype(int)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_morning'] = ((features['hour'] >= 6) & (features['hour'] <= 12)).astype(int)
        features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] <= 18)).astype(int)
        features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 18)).astype(int)
        
        # Seasonal features
        features['is_spring'] = ((features['month'] >= 3) & (features['month'] <= 5)).astype(int)
        features['is_summer'] = ((features['month'] >= 6) & (features['month'] <= 8)).astype(int)
        features['is_autumn'] = ((features['month'] >= 9) & (features['month'] <= 11)).astype(int)
        features['is_winter'] = ((features['month'] == 12) | (features['month'] <= 2)).astype(int)
        
        # Time since sunrise/sunset
        features['hours_since_sunrise'] = (features['hour'] - 6) % 24
        features['hours_since_sunset'] = (features['hour'] - 18) % 24
        
        # Clean up intermediate variables
        del dt, hour_rad, day_rad, month_rad
        if 'solar_TIMESTAMP' in df.columns:
            del solar_dt, solar_hour_rad, solar_day_rad
        
        return features
    
    def create_interaction_features(self, df):
        """Create enhanced interaction features (reduced logging)"""
        
        features = df.copy()
        
        # VPD and radiation interaction
        if 'vpd' in df.columns and 'ppfd_in' in df.columns:
            features['vpd_ppfd_interaction'] = df['vpd'] * df['ppfd_in']
        
        # VPD and temperature interaction
        if 'vpd' in df.columns and 'ta' in df.columns:
            features['vpd_ta_interaction'] = df['vpd'] * df['ta']
        
        # Temperature and humidity ratio
        if 'ta' in df.columns and 'rh' in df.columns:
            features['temp_humidity_ratio'] = df['ta'] / (df['rh'] + 1e-6)
        
        # Water stress index
        if 'swc_shallow' in df.columns and 'vpd' in df.columns:
            features['water_stress_index'] = df['swc_shallow'] / (df['vpd'] + 1e-6)
        
        # Light efficiency
        if 'ppfd_in' in df.columns and 'sw_in' in df.columns:
            features['light_efficiency'] = df['ppfd_in'] / (df['sw_in'] + 1e-6)
        
        # Temperature and soil moisture interaction
        if 'ta' in df.columns and 'swc_shallow' in df.columns:
            features['temp_soil_interaction'] = df['ta'] * df['swc_shallow']
        
        # Wind and VPD interaction
        if 'ws' in df.columns and 'vpd' in df.columns:
            features['wind_vpd_interaction'] = df['ws'] * df['vpd']
        
        # Radiation and temperature interaction
        if 'sw_in' in df.columns and 'ta' in df.columns:
            features['radiation_temp_interaction'] = df['sw_in'] * df['ta']
        
        # Humidity and soil moisture interaction
        if 'rh' in df.columns and 'swc_shallow' in df.columns:
            features['humidity_soil_interaction'] = df['rh'] * df['swc_shallow']
        
        return features
    
    def create_rate_of_change_features(self, df):
        """Create rate of change features (reduced logging)"""
        
        features = df.copy()
        
        # Focus on key variables and time periods
        env_cols = ['rh', 'sw_in', 'vpd', 'ta']
        
        for col in env_cols:
            if col in df.columns:
                features[f'{col}_rate_1h'] = df[col].diff(1)
                features[f'{col}_rate_6h'] = df[col].diff(6)
                features[f'{col}_rate_24h'] = df[col].diff(24)
        
        return features
    
    def create_cumulative_features(self, df):
        """Create cumulative features (reduced logging)"""
        
        features = df.copy()
        
        # Cumulative precipitation
        if 'precip' in df.columns:
            features['precip_cum_24h'] = df['precip'].rolling(24, min_periods=1).sum()
            features['precip_cum_72h'] = df['precip'].rolling(72, min_periods=1).sum()
            features['precip_cum_168h'] = df['precip'].rolling(168, min_periods=1).sum()
        
        # Cumulative radiation
        if 'sw_in' in df.columns:
            features['sw_in_cum_24h'] = df['sw_in'].rolling(24, min_periods=1).sum()
            features['sw_in_cum_72h'] = df['sw_in'].rolling(72, min_periods=1).sum()
        
        return features
    
    def create_advanced_rolling_features(self, df):
        """Create enhanced rolling features (reduced logging)"""
        
        features = df.copy()
        
        # Focus on important windows and variables
        windows = [72, 168, 336]  # 3-day, 7-day, 14-day
        env_cols = ['rh', 'sw_in', 'vpd', 'ta']
        
        for col in env_cols:
            if col in df.columns:
                for window in windows:
                    features[f'{col}_mean_{window}h'] = df[col].rolling(window, min_periods=1).mean()
                    features[f'{col}_std_{window}h'] = df[col].rolling(window, min_periods=1).std()
                    
                    if window >= 72:
                        features[f'{col}_min_{window}h'] = df[col].rolling(window, min_periods=1).min()
                        features[f'{col}_max_{window}h'] = df[col].rolling(window, min_periods=1).max()
                        features[f'{col}_range_{window}h'] = features[f'{col}_max_{window}h'] - features[f'{col}_min_{window}h']
                    
                    if window == 336:  # Memory cleanup after largest window
                        gc.collect()
        
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
        """Create features from metadata (reduced logging)"""
        
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
            
            # Additional site metadata
            if 'si_code' in site_data and pd.notna(site_data['si_code']):
                features['site_code'] = site_data['si_code']
            if 'si_name' in site_data and pd.notna(site_data['si_name']):
                features['site_name'] = site_data['si_name']
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
            if 'st_terrain' in stand_data and pd.notna(stand_data['st_terrain']):
                features['terrain'] = stand_data['st_terrain']
            if 'st_growth_condition' in stand_data and pd.notna(stand_data['st_growth_condition']):
                features['growth_condition'] = stand_data['st_growth_condition']
        
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
        
        # Environmental metadata features
        if 'environmental' in metadata:
            env_data = metadata['environmental']
            
            # Measurement protocol features
            if 'env_timestep' in env_data and pd.notna(env_data['env_timestep']):
                features['measurement_timestep'] = env_data['env_timestep']
            if 'env_time_zone' in env_data and pd.notna(env_data['env_time_zone']):
                features['timezone'] = env_data['env_time_zone']
        
        # Plant metadata features
        if 'plants' in metadata and 'plant_id' in features.columns:
            plants_data = metadata['plants']
            
            # Create mapping from plant_id to tree characteristics
            plant_features = {}
            
            for _, plant in plants_data.iterrows():
                plant_code = plant.get('pl_code', '')
                if plant_code:
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
            
            # Add essential plant metadata features
            essential_plant_cols = ['pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 
                                  'pl_bark_thick', 'pl_social', 'pl_species',
                                  'pl_sapw_area', 'pl_sapw_depth']
            
            for col_name in essential_plant_cols:
                features[col_name] = features['plant_id'].map(
                    lambda x: plant_features.get(x, {}).get(col_name) if x in plant_features else None
                )
        
        # Create derived features
        features = self.create_derived_metadata_features(features)
        
        return features
    
    def create_derived_metadata_features(self, df):
        """Create derived features from metadata (reduced logging)"""
        features = df.copy()
        
        # Climate zone based on latitude
        if 'latitude' in features.columns:
            features['climate_zone_code'] = pd.cut(
                features['latitude'], 
                bins=[-90, -23.5, 23.5, 90], 
                labels=[0, 1, 2]
            ).astype('float64')
            
            features['latitude_abs'] = abs(features['latitude'])
            
            features['climate_zone'] = pd.cut(
                features['latitude'], 
                bins=[-90, -23.5, 23.5, 90], 
                labels=['Temperate_South', 'Tropical', 'Temperate_North']
            )
        
        # Köppen-Geiger classification
        if all(col in features.columns for col in ['latitude', 'mean_annual_temp', 'mean_annual_precip']):
            features['koppen_geiger_code'] = features.apply(self._classify_koppen_geiger, axis=1)
        
        # Aridity index
        if 'mean_annual_temp' in features.columns and 'mean_annual_precip' in features.columns:
            features['aridity_index'] = features['mean_annual_precip'] / (features['mean_annual_temp'] + 10)
        
        # Categorical encodings
        if 'leaf_habit' in features.columns:
            leaf_habit_map = {
                'cold deciduous': 1, 'warm deciduous': 2, 'evergreen': 3, 'semi-deciduous': 4
            }
            features['leaf_habit_code'] = features['leaf_habit'].map(leaf_habit_map).astype('float64')
        
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
        
        if 'igbp_class' in features.columns:
            igbp_map = {
                'ENF': 1, 'EBF': 2, 'DNF': 3, 'DBF': 4, 'MF': 5,
                'CSH': 6, 'OSH': 7, 'WSA': 8, 'SAV': 9, 'GRA': 10,
                'WET': 11, 'CRO': 12, 'URB': 13, 'CVM': 14, 'SNO': 15, 'BSV': 16
            }
            features['igbp_code'] = features['igbp_class'].map(igbp_map).astype('float64')
        
        # Tree size and age classes
        if 'pl_dbh' in features.columns:
            features['tree_size_class'] = pd.cut(
                features['pl_dbh'], 
                bins=[0, 10, 30, 50, 100, 1000], 
                labels=['Sapling', 'Small', 'Medium', 'Large', 'Very Large']
            )
        
        if 'pl_age' in features.columns:
            features['tree_age_class'] = pd.cut(
                features['pl_age'], 
                bins=[0, 20, 50, 100, 200, 1000], 
                labels=['Young', 'Mature', 'Old', 'Very Old', 'Ancient']
            )
        
        # Social status encoding
        if 'pl_social' in features.columns:
            social_map = {'dominant': 3, 'codominant': 2, 'intermediate': 1, 'suppressed': 0}
            features['social_status_code'] = features['pl_social'].map(social_map).astype('float64')
        
        # Derived ratios and indices
        if 'pl_sapw_area' in features.columns and 'pl_leaf_area' in features.columns:
            features['sapwood_leaf_ratio'] = features['pl_sapw_area'] / (features['pl_leaf_area'] + 1e-6)
        
        if 'pl_dbh' in features.columns and 'pl_height' in features.columns:
            features['tree_volume_index'] = (features['pl_dbh'] ** 2) * features['pl_height']
        
        # Timezone and measurement features
        if 'timezone' in features.columns:
            timezone_col = features['timezone'].astype(str)
            features['timezone_offset'] = timezone_col.str.extract(r'([+-]\d{2})').astype(float)
        
        if 'measurement_timestep' in features.columns:
            features['measurement_frequency'] = 60 / features['measurement_timestep']
        
        return features
    
    def encode_categorical_features(self, df):
        """
        Intelligently handle object columns - preserve continuous data, encode ONLY ecological categorical data
        
        CRITICAL: Prevents site identity memorization and geographic proxy overfitting by:
        1. Blacklisting identity features that cause overfitting
        2. Warning about geographic proxy features
        3. Only encoding ecologically meaningful categorical variables
        """
        
        features = df.copy()
        
        # Initialize tracking lists at the beginning
        encoded_features = []
        skipped_identity_features = []
        skipped_pure_geographic_features = []
        allowed_climate_geographic_features = []
        dropped_features = []
        
        # 🚨 CRITICAL: Identity features blacklist - NEVER encode these as predictive features
        IDENTITY_BLACKLIST = {
            'site_code', 'site_name', 'site_id', 'site_identifier',
            'plant_name', 'tree_name', 'tree_id', 'pl_name',
            # Note: 'species_name' moved to functional group processing instead of blocking
            'study_id', 'plot_id', 'station_id'
        }
        
        # ⚠️ Pure geographic identifiers - can hinder spatial generalization
        PURE_GEOGRAPHIC_IDENTIFIERS = {
            'timezone', 'country', 'continent', 'region', 'state', 'province'
        }
        
        # ✅ Climate-based geographic features (ecological information) - allowed with caution
        CLIMATE_GEOGRAPHIC_FEATURES = {
            'climate_zone', 'biome_region', 'koppen_class', 'climate_classification'
        }
        
        # ✅ Approved ecological categorical variables (safe to encode)
        APPROVED_ECOLOGICAL_FEATURES = {
            'biome', 'igbp_class', 'soil_texture', 'aspect', 'terrain', 
            'growth_condition', 'leaf_habit', 'pl_social', 'climate_zone',
            'tree_size_class', 'tree_age_class', 'koppen_geiger_code'
        }
        
        # 🌿 Species functional group mapping (ecological traits instead of specific species)
        # This prevents site-specific memorization while preserving ecological information
        SPECIES_FUNCTIONAL_GROUPS = {
            # Needleleaf Evergreen
            'Abies': 'needleleaf_evergreen', 'Picea': 'needleleaf_evergreen', 'Pinus': 'needleleaf_evergreen',
            'Pseudotsuga': 'needleleaf_evergreen', 'Tsuga': 'needleleaf_evergreen', 'Juniperus': 'needleleaf_evergreen',
            'Cupressus': 'needleleaf_evergreen', 'Thuja': 'needleleaf_evergreen', 'Taxus': 'needleleaf_evergreen',
            
            # Needleleaf Deciduous  
            'Larix': 'needleleaf_deciduous', 'Taxodium': 'needleleaf_deciduous',
            
            # Broadleaf Evergreen
            'Quercus ilex': 'broadleaf_evergreen', 'Quercus suber': 'broadleaf_evergreen',
            'Eucalyptus': 'broadleaf_evergreen', 'Acacia': 'broadleaf_evergreen',
            'Olea': 'broadleaf_evergreen', 'Arbutus': 'broadleaf_evergreen',
            'Ilex': 'broadleaf_evergreen', 'Magnolia': 'broadleaf_evergreen',
            
            # Broadleaf Deciduous Temperate
            'Quercus': 'broadleaf_deciduous_temperate', 'Fagus': 'broadleaf_deciduous_temperate',
            'Betula': 'broadleaf_deciduous_temperate', 'Acer': 'broadleaf_deciduous_temperate',
            'Populus': 'broadleaf_deciduous_temperate', 'Tilia': 'broadleaf_deciduous_temperate',
            'Fraxinus': 'broadleaf_deciduous_temperate', 'Castanea': 'broadleaf_deciduous_temperate',
            'Juglans': 'broadleaf_deciduous_temperate', 'Platanus': 'broadleaf_deciduous_temperate',
            
            # Broadleaf Deciduous Tropical
            'Cecropia': 'broadleaf_deciduous_tropical', 'Ficus': 'broadleaf_deciduous_tropical',
            'Terminalia': 'broadleaf_deciduous_tropical', 'Bombax': 'broadleaf_deciduous_tropical',
        }
        
        # Define encoding mappings for approved ecological categorical variables
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
            },
            'species_functional_group': {
                'needleleaf_evergreen': 1, 'needleleaf_deciduous': 2,
                'broadleaf_evergreen': 3, 'broadleaf_deciduous_temperate': 4, 
                'broadleaf_deciduous_tropical': 5, 'unknown': 0
            }
        }
        
        # 🌿 Process species names into functional groups (prevents site-specific memorization)
        if 'species_name' in features.columns:
            def classify_species_functional_group(species_name):
                if pd.isna(species_name):
                    return 'unknown'
                
                species_str = str(species_name).strip()
                
                # Try exact match first
                if species_str in SPECIES_FUNCTIONAL_GROUPS:
                    return SPECIES_FUNCTIONAL_GROUPS[species_str]
                
                # Try genus-level matching (first word)
                genus = species_str.split()[0] if ' ' in species_str else species_str
                if genus in SPECIES_FUNCTIONAL_GROUPS:
                    return SPECIES_FUNCTIONAL_GROUPS[genus]
                
                # Default for unmatched species
                return 'unknown'
            
            features['species_functional_group'] = features['species_name'].apply(classify_species_functional_group)
            features = features.drop('species_name', axis=1)  # Remove original species column
            encoded_features.append('species_name')
        
        # Explicitly encode Köppen-Geiger climate codes
        if 'koppen_geiger_code' in features.columns:
            # Use factorize for robust integer encoding of string categories
            features['koppen_geiger_code_encoded'], unique_koppen_codes = pd.factorize(features['koppen_geiger_code'])
            features = features.drop('koppen_geiger_code', axis=1)
            encoded_features.append('koppen_geiger_code')
        
        # Encode approved ecological categorical variables
        for col, mapping in encodings.items():
            if col in features.columns:
                features[f'{col}_code'] = features[col].map(mapping)
                # Drop original text column
                features = features.drop(col, axis=1)
                encoded_features.append(col)
        
        # REMOVED: Geographic proxy features from automatic encoding
        # These are now handled with warnings to prevent overfitting
        
        # Smart handling of remaining object columns
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
        
        # Process each object column with enhanced safety checks
        for col in text_columns:
            if col in preserve_cols:
                # Keep these columns as-is
                continue
                
            elif col in IDENTITY_BLACKLIST:
                # 🚨 CRITICAL: Skip identity features that cause overfitting
                features = features.drop(col, axis=1)
                skipped_identity_features.append(col)
                continue
                
            elif col in PURE_GEOGRAPHIC_IDENTIFIERS:
                # ⚠️ Skip pure geographic identifiers that hinder generalization
                features = features.drop(col, axis=1)
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
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    pass
                    
            else:
                # Check if this looks like categorical data
                unique_values = features[col].unique()
                unique_count = len(unique_values)
                
                # Enhanced safety check: Look for patterns that suggest identity features
                col_lower = col.lower()
                is_likely_identity = any(pattern in col_lower for pattern in [
                    'name', 'code', 'id', 'identifier', 'paper', 'reference', 'study'
                ])
                
                if is_likely_identity and unique_count > 5:
                    # Likely an identity feature we missed - skip it
                    features = features.drop(col, axis=1)
                    skipped_identity_features.append(col)
                    continue
                
                # Conservative approach: Only encode if clearly ecological and low cardinality
                if unique_count <= 10 and not is_likely_identity:
                    # Very conservative encoding for small categorical variables
                    sample_values = list(unique_values[:5])
                    
                    # Only encode if values look clearly categorical (not numeric IDs)
                    if not all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values if pd.notna(val)):
                        encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                        features[f'{col}_code'] = features[col].map(encoding_map).astype('float64')
                        features = features.drop(col, axis=1)
                        encoded_features.append(col)
                    else:
                        # Looks like numeric IDs - drop it
                        features = features.drop(col, axis=1)
                        skipped_identity_features.append(col)
                        
                elif unique_count <= 50:
                    # Medium cardinality - be very conservative
                    # Check if values look like categories or IDs
                    sample_values = features[col].dropna().head(10)
                    if all(str(val).replace('.', '').replace('-', '').isdigit() for val in sample_values):
                        # Looks like numeric IDs - drop it
                        features = features.drop(col, axis=1)
                        skipped_identity_features.append(col)
                    else:
                        # Could be ecological categorical but high cardinality is risky
                        features = features.drop(col, axis=1)
                        dropped_features.append(col)
                else:
                    # High cardinality - try to convert to numeric or drop
                    try:
                        features[col] = pd.to_numeric(features[col], errors='coerce')
                    except:
                        # Drop high cardinality text columns
                        features = features.drop(col, axis=1)
                        dropped_features.append(col)
        
        # Summary logging (reduced verbosity)
        if encoded_features:
            print(f"    ✅ Encoded {len(encoded_features)} features")
        if skipped_identity_features or skipped_pure_geographic_features:
            print(f"    🛡️  Blocked {len(skipped_identity_features) + len(skipped_pure_geographic_features)} identity/geographic features")
        if dropped_features:
            print(f"    🗑️  Dropped {len(dropped_features)} high-cardinality features")
        
        return features
    
    def create_domain_specific_features(self, df):
        """Create domain-specific features for transpiration modeling - ALWAYS CREATE ALL FEATURES"""
        
        features = df.copy()
        
        # Helper function to safely get and impute columns
        def safe_get_column(col_name, default_val=0):
            if col_name in df.columns:
                col_data = df[col_name].fillna(df[col_name].mean() if not df[col_name].isna().all() else default_val)
                return col_data
            else:
                return pd.Series([default_val] * len(df), index=df.index)
        
        # Light features - ALWAYS CREATE
        ppfd_in = safe_get_column('ppfd_in', 0)
        sw_in = safe_get_column('sw_in', 0)
        features['ppfd_efficiency'] = ppfd_in / (sw_in + 1e-6)
        
        # Temperature features - ALWAYS CREATE
        ta = safe_get_column('ta', 25)
        features['temp_deviation'] = abs(ta - 25)
        
        # Physiological features - ALWAYS CREATE
        vpd = safe_get_column('vpd', 0)
        features['stomatal_conductance_proxy'] = ppfd_in / (vpd + 1e-6)
            
        # Wind effects - ALWAYS CREATE
        ws = safe_get_column('ws', 0)
        ws_max = ws.max() if len(ws) > 0 and ws.max() > 0 else 1.0
        features['wind_stress'] = ws / (ws_max + 1e-6)
        features['wind_vpd_interaction'] = ws * vpd
        
        # Enhanced interactions with extraterrestrial radiation - ALWAYS CREATE
        ext_rad = safe_get_column('ext_rad', 0)
        features['stomatal_control_index'] = vpd * ppfd_in * ext_rad
        features['light_efficiency'] = ppfd_in / (ext_rad + 1e-6)
        
        # Tree-specific features - ALWAYS CREATE
        pl_dbh = safe_get_column('pl_dbh', 1)
        features['tree_size_factor'] = np.log(pl_dbh + 1)
        
        pl_sapw_area = safe_get_column('pl_sapw_area', 0)
        pl_leaf_area = safe_get_column('pl_leaf_area', 1)
        features['sapwood_leaf_ratio'] = pl_sapw_area / (pl_leaf_area + 1e-6)
        features['transpiration_capacity'] = pl_sapw_area * ppfd_in / (vpd + 1e-6)
        
        return features
    
    def create_seasonality_features(self, df):
        """
        Create seasonal temperature and precipitation range features for ecosystem clustering.
        These are site-level features that capture intra-annual variability.
        """
        if not ProcessingConfig.get_feature_setting('seasonality_features'):
            return df
        
        try:
            # ALWAYS CREATE seasonal features for consistency across all sites
            if 'site' not in df.columns:
                # No site column - use default values
                df['seasonal_temp_range'] = 0.0
                df['seasonal_precip_range'] = 0.0
                return df
            
            # Check if we have the required data columns
            required_cols = ['month', 'ta', 'precip']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Missing required columns - use default values but still create features
                df['seasonal_temp_range'] = 0.0
                df['seasonal_precip_range'] = 0.0
                return df
            
            # Calculate seasonal ranges for each site
            seasonality_data = []
            
            for site, site_group in df.groupby('site'):
                try:
                    # Calculate monthly means for temperature (with fallback)
                    if 'month' in site_group.columns and 'ta' in site_group.columns:
                        monthly_temp = site_group.groupby('month')['ta'].mean()
                        temp_range = monthly_temp.max() - monthly_temp.min() if len(monthly_temp) > 1 else 0.0
                    else:
                        temp_range = 0.0
                    
                    # Calculate monthly means for precipitation (with fallback)
                    if 'month' in site_group.columns and 'precip' in site_group.columns:
                        monthly_precip = site_group.groupby('month')['precip'].mean()
                        precip_range = monthly_precip.max() - monthly_precip.min() if len(monthly_precip) > 1 else 0.0
                    else:
                        precip_range = 0.0
                    
                    seasonality_data.append({
                        'site': site,
                        'seasonal_temp_range': temp_range,
                        'seasonal_precip_range': precip_range
                    })
                except Exception:
                    # Fallback for any site-specific errors
                    seasonality_data.append({
                        'site': site,
                        'seasonal_temp_range': 0.0,
                        'seasonal_precip_range': 0.0
                    })
            
            # Create seasonality DataFrame
            seasonality_df = pd.DataFrame(seasonality_data)
            
            # Merge back to main DataFrame (site-level features, so merge on 'site')
            df = df.merge(seasonality_df, on='site', how='left')
            
            # Fill any remaining NaN values with 0
            df['seasonal_temp_range'] = df['seasonal_temp_range'].fillna(0.0)
            df['seasonal_precip_range'] = df['seasonal_precip_range'].fillna(0.0)
            
            return df
            
        except Exception as e:
            # Add seasonal features with default values for consistency
            if 'seasonal_temp_range' not in df.columns:
                df['seasonal_temp_range'] = 0.0
            if 'seasonal_precip_range' not in df.columns:
                df['seasonal_precip_range'] = 0.0
            return df

    # NOTE: apply_balanced_sampling method removed - cannot work during data processing
    # Balanced sampling must happen AFTER clustering (which uses the processed data)
    # This is a chicken-and-egg problem: clusters are created FROM processed data,
    # so cluster information cannot be used DURING data processing.
    # 
    # Solution: Implement balanced sampling in the clustering or training stage,
    # not in the data processing pipeline.

    
    def process_all_sites(self):
        """Process all sites with complete feature creation and adaptive memory management"""
        
        # First, analyze all sites to populate dynamic classifications
        print("🔍 Step 1: Analyzing site data quality...")
        analysis_summary = self.analyze_all_sites_data_quality()
        
        print(f"\n🔍 Step 2: Finding sites to process...")
        all_sites = self.get_all_sites()
        print(f"📊 Found {len(all_sites)} total sites")
        
        # Check which sites need processing vs which can be skipped
        sites_to_process, sites_to_skip = self.get_processing_status(all_sites)
        
        if self.force_reprocess:
            print(f"🔄 Force reprocessing enabled - will process all valid sites")
            # Only include sites that have valid data for reprocessing
            valid_sites = [site for site in all_sites if site not in self.SITES_WITH_NO_VALID_DATA]
            sites_to_process = valid_sites
            sites_to_skip = [site for site in all_sites if site in self.SITES_WITH_NO_VALID_DATA]
        
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
                
                # Memory cleanup after successful processing
                self.force_memory_cleanup()
                
            else:
                failed_sites.append(site)
                self.stats['failed_sites'] += 1
                
                # Memory cleanup after failed processing too
                self.force_memory_cleanup()
        
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
        
        # Optimized merging using time-based indexing with memory management
        
        # Convert timestamps to datetime for efficient merging
        env_data[timestamp_col] = pd.to_datetime(env_data[timestamp_col])
        sapf_data[sapf_timestamp_col] = pd.to_datetime(sapf_data[sapf_timestamp_col])
        
        # Set timestamp as index for faster merging
        env_data_indexed = env_data.set_index(timestamp_col)
        sapf_data_indexed = sapf_data.set_index(sapf_timestamp_col)
        
        # Clean up original dataframes immediately
        del env_data, sapf_data
        self.check_memory_usage()
        
        # Melt sap flow data with indexed timestamp
        sapf_long = sapf_data_indexed[sapf_cols].reset_index().melt(
            id_vars=[sapf_timestamp_col], 
            value_vars=sapf_cols,
            var_name='plant_id', 
            value_name='sap_flow'
        )
        
        # Clean up sapf_data_indexed immediately after melting
        del sapf_data_indexed
        self.check_memory_usage()
        
        # Merge using indexed join (more memory efficient)
        merged = sapf_long.merge(
            env_data_indexed.reset_index(), 
            left_on=sapf_timestamp_col, 
            right_on=timestamp_col, 
            how='inner'
        )
        
        # Clean up intermediate dataframes
        del sapf_long, env_data_indexed
        self.check_memory_usage()
        
        # Remove rows with missing sap flow data
        merged = merged.dropna(subset=['sap_flow'])
        
        if len(merged) == 0:
            print(f"  ❌ No valid data after merging for {site}")
            return None
        
        # Create features using adaptive settings (v2 approach for better memory management)
        
        # Stage 1: Enhanced temporal features
        merged = self.create_advanced_temporal_features(merged, timestamp_col)
        self.check_memory_usage()
        
        # Stage 2: Basic rolling features (adaptive) - use existing function for memory efficiency
        env_cols = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        merged = self.create_rolling_features_adaptive(merged, env_cols)
        self.check_memory_usage()
        
        # Stage 3: Lagged features (adaptive)
        merged = self.create_lagged_features_adaptive(merged, env_cols)
        self.check_memory_usage()
        
        # Stage 4: Basic interaction features (memory-efficient subset)
        merged = self.create_interaction_features(merged)
        self.check_memory_usage()
        
        # Stage 5: Advanced features (ALL sites - with memory management)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        dataset_size = len(merged)
        
        print(f"    🔧 Creating advanced features for {site} ({dataset_size:,} rows, {available_memory_gb:.1f}GB available)")
        
        try:
            # Force memory cleanup before advanced features
            gc.collect()
            
            # Advanced rolling features (enhanced with longer windows)
            merged = self.create_advanced_rolling_features(merged)
            self.check_memory_usage()
            
            # Rate of change features
            merged = self.create_rate_of_change_features(merged)
            self.check_memory_usage()
            
            # Cumulative features
            merged = self.create_cumulative_features(merged)
            self.check_memory_usage()
            
            print(f"    ✅ Advanced features created for {site} ({len(merged):,} rows)")
            
        except Exception as e:
            print(f"    ❌ Advanced features failed for {site}: {e}")
            print(f"    🔄 Retrying with streaming approach...")
            
            # If advanced features fail, try streaming approach for large sites
            if dataset_size > 50000:
                try:
                    # Process in chunks for large sites
                    chunk_size = min(10000, dataset_size // 10)
                    advanced_features = []
                    
                    for i in range(0, dataset_size, chunk_size):
                        chunk = merged.iloc[i:i+chunk_size].copy()
                        
                        # Create advanced features for chunk
                        chunk = self.create_advanced_rolling_features(chunk)
                        chunk = self.create_rate_of_change_features(chunk)
                        chunk = self.create_cumulative_features(chunk)
                        
                        advanced_features.append(chunk)
                        
                        # Memory cleanup
                        del chunk
                        gc.collect()
                    
                    # Combine chunks
                    merged = pd.concat(advanced_features, ignore_index=True)
                    del advanced_features
                    gc.collect()
                    
                    print(f"    ✅ Advanced features created via streaming for {site}")
                    
                except Exception as e2:
                    print(f"    ❌ Streaming approach also failed for {site}: {e2}")
                    print(f"    📊 Continuing with basic feature set only")
                    gc.collect()
            else:
                print(f"    📊 Continuing with basic feature set only")
                gc.collect()
        
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
        
        # Stage 7: Seasonality features (for ecosystem clustering) - after site is added
        merged = self.create_seasonality_features(merged)
        self.check_memory_usage()
        
        # NOTE: Balanced sampling happens AFTER clustering, not during initial processing
        # Clusters don't exist yet during data pipeline - they're created from processed data
        
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
            
            # Use consolidated feature engineering (streaming mode)
            merged = self.create_all_features(merged, metadata, timestamp_col, processing_mode='streaming')
            
            # Note: Seasonality features not available in streaming mode
            # (require site-level calculations that need full dataset)
            
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
            'moisture_availability', 'swc_shallow_depth',
            # Redundant features (can be computed during training)
            'stand_soil_texture_code',
            'precip_intensity', 'recent_precip_1h', 'recent_precip_6h', 
            'recent_precip_24h', 'aspect_code', 'species_basal_area_perc', 'site_paper_code',
            'daylight_time'
            # Note: wind_stress, ppfd_efficiency, stomatal_conductance_proxy, stomatal_control_index
            # REMOVED from problematic list - these are scientifically important features (Jan 2025)
            # Note: soil_texture_code and terrain_code are kept - these are legitimate ecological features
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
            
            # Enhanced temporal features (new)
            'hour_sin': np.nan, 'hour_cos': np.nan, 'day_sin': np.nan, 'day_cos': np.nan,
            'month_sin': np.nan, 'month_cos': np.nan,
            'is_morning': np.nan, 'is_afternoon': np.nan, 'is_night': np.nan,
            'is_spring': np.nan, 'is_summer': np.nan, 'is_autumn': np.nan, 'is_winter': np.nan,
            'hours_since_sunrise': np.nan, 'hours_since_sunset': np.nan,
            
            # Interaction features (new)
            'vpd_ppfd_interaction': np.nan, 'vpd_ta_interaction': np.nan, 'temp_humidity_ratio': np.nan,
            'water_stress_index': np.nan, 'light_efficiency': np.nan, 'temp_soil_interaction': np.nan,
            'wind_vpd_interaction': np.nan, 'radiation_temp_interaction': np.nan, 'humidity_soil_interaction': np.nan,
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
                # Check if any changes are needed before making them
                needs_fixing = False
                
                # Ensure any invalid values are converted to NaN for XGBoost
                if df[col].dtype == 'object':
                    # For categorical columns, replace empty strings or invalid values with NaN
                    invalid_mask = df[col].isin(['', 'nan', 'None', 'NULL'])
                    if invalid_mask.any():
                        df[col] = df[col].replace(['', 'nan', 'None', 'NULL'], np.nan)
                        needs_fixing = True
                else:
                    # For numeric columns, ensure inf/-inf are converted to NaN
                    invalid_mask = np.isinf(df[col])
                    if invalid_mask.any():
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                        needs_fixing = True
                
                # Only print message if actual changes were made
                if needs_fixing:
                    print(f"    🔧 Fixed XGBoost compatibility for: {col}")
        
        return df
    
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

def main():
    """Main execution function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process SAPFLUXNET data with complete feature creation')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of all sites (ignore existing files)')
    parser.add_argument('--output-dir', default='comprehensive_processed',
                       help='Base output directory for processed files (default: comprehensive_processed). Format-specific directories will be created (e.g., processed_csv, processed_parquet, processed_libsvm)')
    parser.add_argument('--chunk-size', type=int, default=1500,
                       help='Chunk size for processing (default: 1500)')
    parser.add_argument('--max-memory', type=int, default=12,
                       help='Maximum memory usage in GB (default: 12)')
    parser.add_argument('--include-problematic', action='store_true',
                       help='Include problematic sites with high quality flag rates (sites with no valid data are always excluded)')
    parser.add_argument('--clean-mode', action='store_true',
                       help='Clean mode: only exclude extremely problematic sites (>80% flag rates). Includes moderate and high problematic sites.')
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
    parser.add_argument('--export-format', choices=['csv', 'parquet', 'feather', 'hdf5', 'pickle', 'libsvm'], default='csv',
                       help='Export format for processed files (default: csv)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only perform site data quality analysis without processing (saves results to site_analysis_results/)')

    
    args = parser.parse_args()
    
    print("🚀 Starting Adaptive Complete SAPFLUXNET Processing")
    print(f"⏰ Started at: {datetime.now()}")
    
    if args.force:
        print("🔄 Force reprocessing mode enabled")
    
    if args.clean_mode:
        print("🧹 Clean mode enabled - only excluding extremely problematic sites (>80% flag rates)")
        print("📊 Including moderate and high problematic sites for larger dataset")
    
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
        config_overrides=config_overrides,
        clean_mode=args.clean_mode
    )
    
    # Check if only analysis is requested
    if args.analyze_only:
        print("🔍 Running site data quality analysis only...")
        analysis_summary = processor.analyze_all_sites_data_quality()
        print(f"\n✅ Site analysis complete!")
        print(f"📁 Results saved in site_analysis_results/ directory")
        print(f"📊 Summary: {analysis_summary['total_sites']} sites analyzed")
        print(f"  - ✅ Adequate temporal coverage: {analysis_summary['adequate_temporal']} sites")
        print(f"  - ⚠️  Moderate temporal coverage: {analysis_summary['moderate_temporal']} sites")
        print(f"  - 📉 Insufficient temporal coverage: {analysis_summary['insufficient_temporal']} sites")
        print(f"  - 🚫 No valid data: {analysis_summary['no_valid_data']} sites")
        return
    
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
        print(f"🗂️  Directory structure: Format-specific directories created")
        if processor.use_quality_flags:
            print(f"🏷️  Quality flag filtering applied (OUT_WARN and RANGE_WARN data removed)")
        else:
            print(f"⚠️  Quality flag filtering disabled (all data points included)")
        
        # Print excluded sites summary (from dynamic analysis)
        print(f"\n🚫 Excluded sites (from dynamic analysis):")
        print(f"  - No valid data: {len(processor.SITES_WITH_NO_VALID_DATA)} sites")
        if len(processor.SITES_WITH_NO_VALID_DATA) > 0:
            # Show breakdown by reason
            no_data_reasons = {}
            for site in processor.SITES_WITH_NO_VALID_DATA:
                if site in processor.site_analysis_results:
                    reason = processor.site_analysis_results[site]['exclude_reason']
                    if reason not in no_data_reasons:
                        no_data_reasons[reason] = []
                    no_data_reasons[reason].append(site)
            
            for reason, sites in no_data_reasons.items():
                print(f"    └─ {reason}: {len(sites)} sites")
                if len(sites) <= 5:
                    print(f"       {', '.join(sorted(sites))}")
                else:
                    print(f"       {', '.join(sorted(sites[:5]))}... (+{len(sites)-5} more)")
        
        print(f"  - Insufficient temporal coverage (<30 days): {len(processor.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE)} sites")
        if len(processor.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE) > 0:
            insufficient_sites = sorted(list(processor.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE))
            print(f"    └─ Sites: {', '.join(insufficient_sites)}")
            # Show temporal coverage for these sites
            for site in insufficient_sites:
                if site in processor.site_analysis_results:
                    days = processor.site_analysis_results[site]['temporal_coverage_days']
                    print(f"       {site}: {days:.1f} days")
        
        if processor.skip_problematic_sites:
            if processor.clean_mode:
                print(f"  - Extremely problematic (>80% flag rate): {len(processor.EXTREMELY_PROBLEMATIC_SITES)} sites")
                print(f"  - High problematic (50-80% flag rate): {len(processor.HIGH_PROBLEMATIC_SITES)} sites (included in clean mode)")
                print(f"  - Moderate problematic (20-50% flag rate): {len(processor.MODERATE_PROBLEMATIC_SITES)} sites (included in clean mode)")
                print(f"  - Total problematic sites: {len(processor.PROBLEMATIC_SITES)} sites")
                print(f"  💡 Clean mode: Only excluding extremely problematic sites for larger dataset")
            else:
                print(f"  - Extremely problematic (>80% flag rate): {len(processor.EXTREMELY_PROBLEMATIC_SITES)} sites")
                print(f"  - High problematic (50-80% flag rate): {len(processor.HIGH_PROBLEMATIC_SITES)} sites")
                print(f"  - Moderate problematic (20-50% flag rate): {len(processor.MODERATE_PROBLEMATIC_SITES)} sites")
                print(f"  - Total problematic sites: {len(processor.PROBLEMATIC_SITES)} sites")
                print(f"  💡 Use --include-problematic to process problematic sites anyway")
                print(f"  💡 Use --clean-mode to only exclude extremely problematic sites")
        else:
            print(f"  - Problematic sites included: {len(processor.PROBLEMATIC_SITES)} sites")
        
        print(f"\n⚠️  Sites with moderate temporal coverage (30-90 days) processed with warnings:")
        print(f"  - Count: {len(processor.SITES_WITH_MODERATE_TEMPORAL_COVERAGE)} sites")
        print(f"  - These sites have adequate data but may have limited seasonal representation")
        if len(processor.SITES_WITH_MODERATE_TEMPORAL_COVERAGE) > 0:
            moderate_sites = sorted(list(processor.SITES_WITH_MODERATE_TEMPORAL_COVERAGE)[:10])
            print(f"  - Sites: {', '.join(moderate_sites)}{'...' if len(processor.SITES_WITH_MODERATE_TEMPORAL_COVERAGE) > 10 else ''}")
            # Show temporal coverage for these sites
            for site in moderate_sites:
                if site in processor.site_analysis_results:
                    days = processor.site_analysis_results[site]['temporal_coverage_days']
                    print(f"    └─ {site}: {days:.1f} days")
        
        print(f"\n✅ Sites with adequate temporal coverage (≥90 days):")
        print(f"  - Count: {len(processor.SITES_WITH_ADEQUATE_TEMPORAL_COVERAGE)} sites")
        print(f"  - These sites provide robust data for spatial modeling")
        
    else:
        print(f"\n❌ Adaptive complete processing failed")
        print(f"💡 Check memory availability and file integrity")
    
    print(f"\n⏰ Finished at: {datetime.now()}")

if __name__ == "__main__":
    main() 