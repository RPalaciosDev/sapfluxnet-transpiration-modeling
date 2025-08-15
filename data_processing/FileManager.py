import os
import gzip
import json
import pandas as pd
from contextlib import contextmanager
from pathlib import Path
from .data_constants import FILE_FORMAT_EXTENSIONS
from .error_utils import ErrorHandler
from .logging_utils import logger


class FileManager:
    """Handles all file I/O operations, format validation, and metadata loading"""
    
    def __init__(self, export_format='parquet', base_output_dir='comprehensive_processed', 
                 compress_output=False, config=None, stats=None):
        """
        Initialize FileManager
        
        Args:
            export_format: Export format ('csv', 'parquet', 'libsvm')
            base_output_dir: Base output directory
            compress_output: Whether to compress output files
            config: ProcessingConfig instance for accessing I/O settings
            stats: Statistics dictionary to track I/O operations
        """
        self.export_format = export_format.lower()
        self.base_output_dir = base_output_dir
        self.compress_output = compress_output
        self.config = config
        self.stats = stats or {'io_operations': 0, 'bytes_written': 0}
        
        # Validate format and dependencies
        self.validate_export_format()
        
        # Create format-specific output directory
        self.output_dir = self.get_format_specific_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.init_component("FileManager", f"{self.export_format.upper()} format, output: {self.output_dir}")
    
    def validate_export_format(self):
        """Validate the export format and check for required dependencies"""
        valid_formats = FILE_FORMAT_EXTENSIONS.keys()  # dict_keys is iterable
        
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
                raise ImportError("tables is required for HDF5 export. Install with: pip install tables")
        
        elif self.export_format == 'libsvm':
            try:
                from sklearn.datasets import dump_svmlight_file
            except ImportError:
                raise ImportError("scikit-learn is required for libsvm export. Install with: pip install scikit-learn")
    
    def get_output_file_extension(self):
        """Get the appropriate file extension for the export format"""
        return FILE_FORMAT_EXTENSIONS.get(self.export_format, '.csv')
    
    def get_format_specific_output_dir(self):
        """Get format-specific output directory name"""
        if self.base_output_dir == 'comprehensive_processed':
            # Use format-specific directory for default case
            return f'processed_{self.export_format}'
        else:
            # Check if format is already in the directory name to avoid duplication
            if self.base_output_dir.startswith(f'{self.export_format}_') or self.base_output_dir.endswith(f'_{self.export_format}'):
                # Format already included, use as-is
                return self.base_output_dir
            else:
                # For custom directories, append format suffix
                return f'{self.base_output_dir}_{self.export_format}'
    
    def standardize_features_to_reference(self, df, site_name):
        """Ensure all sites have the same features as reference site (THA_KHU)"""
        reference_file = os.path.join(self.get_format_specific_output_dir(), f'THA_KHU.{self.get_output_file_extension().lstrip(".")}')  
        
        if os.path.exists(reference_file) and site_name != 'THA_KHU':
            try:
                # Load reference features
                if self.export_format == 'parquet':
                    import pyarrow.parquet as pq
                    ref_file = pq.ParquetFile(reference_file)
                    reference_features = set(ref_file.schema.names)
                elif self.export_format == 'csv':
                    ref_df_sample = pd.read_csv(reference_file, nrows=0)  # Just get columns (header only)
                    reference_features = set(ref_df_sample.columns)
                elif self.export_format == 'libsvm':
                    # For libsvm format, skip standardization as it doesn't preserve column names
                    return df
                else:
                    # For other formats, skip standardization
                    return df
                
                current_features = set(df.columns)
                missing_features = reference_features - current_features
                extra_features = current_features - reference_features
                
                if missing_features or extra_features:
                    logger.performance_info(f"Standardizing features for {site_name}", f"{len(missing_features)} added, {len(extra_features)} removed")
                    if missing_features:
                        for feature in missing_features:
                            df[feature] = 0.0
                    if extra_features:
                        df = df.drop(columns=extra_features)  # set is iterable
                    
                    # Reorder columns to match reference
                    df = df.reindex(columns=sorted(reference_features), fill_value=0.0)
                
            except Exception as e:
                logger.warning(f"Could not standardize features for {site_name}: {e}", indent=2)
        
        return df
    
    def save_dataframe_formatted(self, df, output_file, site_name):
        """Save DataFrame in the specified format with optimized I/O"""
        
        # Standardize features to match reference site (THA_KHU)
        df = self.standardize_features_to_reference(df, site_name)
        
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
                df.to_feather(output_file)
                
            elif self.export_format == 'hdf5':
                df.to_hdf(output_file, key='data', mode='w', format='table', 
                         complevel=6 if self.compress_output else 0)
                
            elif self.export_format == 'pickle':
                if self.compress_output:
                    import pickle
                    import gzip
                    with gzip.open(output_file + '.gz', 'wb') as f:
                        pickle.dump(df, f)
                    output_file += '.gz'
                else:
                    df.to_pickle(output_file)
                
            elif self.export_format == 'libsvm':
                self._save_libsvm_format(df, output_file, site_name)
            
            else:
                raise ValueError(f"Unsupported export format: {self.export_format}")
            
            # Track statistics
            self.stats['io_operations'] += 1
            self.stats['bytes_written'] += self.get_file_size_bytes(output_file)
            
            size_mb = self.get_file_size_mb(output_file, estimate_compressed=False)
            logger.file_saved(site_name, size_mb, self.export_format)
            
        except Exception as e:
            logger.error(f"Error saving {site_name} in {self.export_format} format: {str(e)}")
            # Fallback to CSV if other format fails
            if self.export_format != 'csv':
                logger.warning("Falling back to CSV format...")
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
            
            logger.performance_info("Feature mapping", f"saved to {feature_mapping_file}")
        
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
        
        logger.performance_info("LibSVM export", f"{len(df_clean)} samples, {len(feature_cols)} features")
    
    @contextmanager
    def optimized_file_writer(self, file_path, mode='w'):
        """Context manager for optimized file writing with buffering and compression"""
        if self.compress_output and not file_path.endswith('.gz'):
            file_path += '.gz'
        
        if self.compress_output:
            # Use gzip compression with buffering
            compression_level = 6  # Default compression level
            if self.config and hasattr(self.config, 'get_io_setting'):
                compression_level = self.config.get_io_setting('compression_level') or 6
            with gzip.open(file_path, mode + 't', compresslevel=compression_level, encoding='utf-8') as f:
                yield f
        else:
            # Use regular file with buffering
            buffer_size = 8192  # Default buffer size
            if self.config and hasattr(self.config, 'get_io_setting'):
                buffer_size = self.config.get_io_setting('buffer_size') or 8192
            with open(file_path, mode, buffering=buffer_size) as f:
                yield f
    
    
    
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
    
    def get_file_stats(self):
        """Get current file I/O statistics"""
        return {
            'io_operations': self.stats.get('io_operations', 0),
            'bytes_written': self.stats.get('bytes_written', 0),
            'mb_written': self.stats.get('bytes_written', 0) / (1024 * 1024),
            'export_format': self.export_format,
            'compression_enabled': self.compress_output,
            'output_directory': self.output_dir
        }
    
    def reset_stats(self):
        """Reset file I/O statistics"""
        self.stats['io_operations'] = 0
        self.stats['bytes_written'] = 0
    
    def validate_file_exists(self, file_path, file_type="file"):
        """Validate that a file exists and is readable"""
        if not os.path.exists(file_path):
            logger.error(f"{file_type.capitalize()} not found: {file_path}", indent=2)
            return False
        
        try:
            # Try to get file size to ensure it's readable
            file_size_bytes = self.get_file_size_bytes(file_path)
            if file_size_bytes == 0:
                logger.warning(f"{file_type.capitalize()} is empty: {file_path}", indent=2)
                return False
            return True
        except Exception as e:
            logger.error(f"Cannot access {file_type}: {file_path} - {e}", indent=2)
            return False
    
    def check_file_exists(self, file_path, check_compressed=True):
        """
        Check if a file exists, optionally checking for compressed versions.
        
        Args:
            file_path (str): Path to the file
            check_compressed (bool): Whether to check for .gz compressed versions
            
        Returns:
            tuple: (exists, actual_path) - actual_path includes .gz if compressed version found
        """
        # Check original file first
        if os.path.exists(file_path):
            return True, file_path
        
        # Check compressed version if requested
        if check_compressed and os.path.exists(file_path + '.gz'):
            return True, file_path + '.gz'
        
        return False, file_path
    
    def get_file_size_mb(self, file_path, estimate_compressed=True):
        """
        Get file size in MB, handling compressed files intelligently.
        
        Args:
            file_path (str): Path to the file
            estimate_compressed (bool): Whether to estimate uncompressed size for .gz files
            
        Returns:
            float: File size in MB, or 0 if file doesn't exist
        """
        exists, actual_path = self.check_file_exists(file_path)
        if not exists:
            return 0.0
        
        try:
            file_size_bytes = os.path.getsize(actual_path)
            
            # Estimate uncompressed size for .gz files
            if estimate_compressed and actual_path.endswith('.gz'):
                # Typical compression ratio for CSV data is ~3:1
                file_size_bytes *= 3
            
            return file_size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def find_best_file_path(self, base_path, prefer_compressed=None):
        """
        Find the best available file path, considering compression preferences.
        
        Args:
            base_path (str): Base file path (without .gz extension)
            prefer_compressed (bool): True=prefer .gz, False=prefer uncompressed, None=auto-detect
            
        Returns:
            str or None: Best available file path, or None if no file found
        """
        uncompressed_exists = os.path.exists(base_path)
        compressed_exists = os.path.exists(base_path + '.gz')
        
        if not uncompressed_exists and not compressed_exists:
            return None
        
        # Auto-detect preference based on configuration
        if prefer_compressed is None:
            prefer_compressed = self.compress_output
        
        # Apply preference logic
        if prefer_compressed and compressed_exists:
            return base_path + '.gz'
        elif not prefer_compressed and uncompressed_exists:
            return base_path
        elif compressed_exists:
            return base_path + '.gz'
        elif uncompressed_exists:
            return base_path
        
        return None
    
    def check_files_exist(self, file_paths, file_types=None):
        """
        Check existence of multiple files efficiently.
        
        Args:
            file_paths (list): List of file paths to check
            file_types (list): Optional list of file type descriptions for error messages
            
        Returns:
            dict: {file_path: {'exists': bool, 'actual_path': str, 'size_mb': float}}
        """
        if file_types is None:
            file_types = ['file'] * len(file_paths)
        
        results = {}
        for i, file_path in enumerate(file_paths):
            file_type = file_types[i] if i < len(file_types) else 'file'
            
            exists, actual_path = self.check_file_exists(file_path)
            size_mb = self.get_file_size_mb(file_path) if exists else 0.0
            
            results[file_path] = {
                'exists': exists,
                'actual_path': actual_path,
                'size_mb': size_mb,
                'file_type': file_type
            }
        
        return results
    
    def validate_file_size(self, file_path, min_size_mb=None, max_size_mb=None):
        """
        Validate file size against optional min/max thresholds.
        
        Args:
            file_path (str): Path to the file
            min_size_mb (float): Minimum acceptable size in MB
            max_size_mb (float): Maximum acceptable size in MB
            
        Returns:
            dict: {'valid': bool, 'size_mb': float, 'reason': str}
        """
        exists, actual_path = self.check_file_exists(file_path)
        if not exists:
            return {'valid': False, 'size_mb': 0.0, 'reason': 'File does not exist'}
        
        size_mb = self.get_file_size_mb(file_path)
        
        if min_size_mb is not None and size_mb < min_size_mb:
            return {'valid': False, 'size_mb': size_mb, 'reason': f'File too small ({size_mb:.1f}MB < {min_size_mb:.1f}MB)'}
        
        if max_size_mb is not None and size_mb > max_size_mb:
            return {'valid': False, 'size_mb': size_mb, 'reason': f'File too large ({size_mb:.1f}MB > {max_size_mb:.1f}MB)'}
        
        return {'valid': True, 'size_mb': size_mb, 'reason': 'Valid file size'}
    
    def get_file_size_bytes(self, file_path):
        """
        Get file size in bytes, handling compressed files.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            int: File size in bytes, or 0 if file doesn't exist
        """
        exists, actual_path = self.check_file_exists(file_path)
        if not exists:
            return 0
        
        try:
            return os.path.getsize(actual_path)
        except Exception:
            return 0
    
    def estimate_rows_from_file_size(self, file_path, bytes_per_row=1000):
        """
        Estimate number of rows based on file size.
        
        Args:
            file_path (str): Path to the file
            bytes_per_row (int): Estimated bytes per row (default: 1KB for CSV)
            
        Returns:
            int: Estimated number of rows
        """
        size_mb = self.get_file_size_mb(file_path, estimate_compressed=True)
        if size_mb == 0:
            return 0
        
        size_bytes = size_mb * 1024 * 1024
        return int(size_bytes / bytes_per_row)
    
    def get_file_size_classification(self, file_path, thresholds=None):
        """
        Classify file size into categories (small, medium, large, etc.).
        
        Args:
            file_path (str): Path to the file
            thresholds (dict): Custom thresholds in MB. If None, uses default thresholds.
            
        Returns:
            dict: {'size_mb': float, 'classification': str, 'threshold_used': dict}
        """
        if thresholds is None:
            thresholds = {
                'small': 10,      # < 10MB
                'medium': 50,     # 10-50MB  
                'large': 200,     # 50-200MB
                'very_large': 500, # 200-500MB
                'huge': float('inf')  # > 500MB
            }
        
        size_mb = self.get_file_size_mb(file_path, estimate_compressed=True)
        
        if size_mb == 0:
            classification = 'empty'
        elif size_mb < thresholds['small']:
            classification = 'small'
        elif size_mb < thresholds['medium']:
            classification = 'medium'
        elif size_mb < thresholds['large']:
            classification = 'large'
        elif size_mb < thresholds['very_large']:
            classification = 'very_large'
        else:
            classification = 'huge'
        
        return {
            'size_mb': size_mb,
            'classification': classification,
            'threshold_used': thresholds
        }
    
    def compare_file_sizes(self, file_paths):
        """
        Compare sizes of multiple files.
        
        Args:
            file_paths (list): List of file paths to compare
            
        Returns:
            dict: Detailed size comparison information
        """
        results = {}
        total_size = 0
        
        for file_path in file_paths:
            size_mb = self.get_file_size_mb(file_path, estimate_compressed=True)
            classification = self.get_file_size_classification(file_path)
            
            results[file_path] = {
                'size_mb': size_mb,
                'classification': classification['classification'],
                'exists': size_mb > 0
            }
            total_size += size_mb
        
        # Find largest and smallest files
        valid_files = {k: v for k, v in results.items() if v['exists']}
        if valid_files:
            largest_file = max(valid_files.keys(), key=lambda x: valid_files[x]['size_mb'])
            smallest_file = min(valid_files.keys(), key=lambda x: valid_files[x]['size_mb'])
        else:
            largest_file = smallest_file = None
        
        return {
            'individual_files': results,
            'total_size_mb': total_size,
            'largest_file': largest_file,
            'smallest_file': smallest_file,
            'file_count': len(file_paths),
            'existing_count': len(valid_files)
        }