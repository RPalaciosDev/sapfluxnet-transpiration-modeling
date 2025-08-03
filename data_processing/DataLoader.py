import os
import pandas as pd
import numpy as np
from .error_utils import ErrorHandler
from .logging_utils import logger
from .data_constants import PROBLEMATIC_COLUMNS_TO_EXCLUDE, BAD_QUALITY_FLAGS


class DataLoader:
    """Handles data loading, quality flag filtering, and preprocessing"""
    
    def __init__(self, config=None, memory_manager=None, use_quality_flags=True, file_manager=None):
        """
        Initialize DataLoader
        
        Args:
            config: ProcessingConfig instance for accessing settings
            memory_manager: MemoryManager instance for memory operations
            use_quality_flags: Whether to filter out flagged data points
            file_manager: FileManager instance for file operations
        """
        self.config = config
        self.memory_manager = memory_manager
        self.use_quality_flags = use_quality_flags
        self.file_manager = file_manager
        
        logger.init_component("DataLoader", f"Quality flags {'enabled' if use_quality_flags else 'disabled'}")
    
    # =============================================================================
    # TIMESTAMP UTILITIES (Centralized to eliminate redundancy)
    # =============================================================================
    
    @staticmethod
    def find_timestamp_columns(df, exclude_solar=False):
        """
        Find timestamp columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to search
            exclude_solar (bool): Whether to exclude solar_TIMESTAMP columns
            
        Returns:
            list: List of timestamp column names
        """
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        
        if exclude_solar:
            timestamp_cols = [col for col in timestamp_cols if not col.lower().startswith('solar')]
        
        return timestamp_cols
    
    @staticmethod
    def get_primary_timestamp_column(df, exclude_solar=False):
        """
        Get the primary timestamp column from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to search
            exclude_solar (bool): Whether to exclude solar_TIMESTAMP columns
            
        Returns:
            str or None: Primary timestamp column name, or None if not found
        """
        timestamp_cols = DataLoader.find_timestamp_columns(df, exclude_solar)
        return timestamp_cols[0] if timestamp_cols else None
    
    @staticmethod
    def convert_timestamp_column(df, column_name, inplace=True):
        """
        Convert a column to datetime format.
        
        Args:
            df (pd.DataFrame): DataFrame containing the column
            column_name (str): Name of the column to convert
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pd.DataFrame: DataFrame with converted timestamp (if not inplace)
        """
        if not inplace:
            df = df.copy()
        
        if column_name in df.columns and not pd.api.types.is_datetime64_any_dtype(df[column_name]):
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        
        return None if inplace else df
    
    @staticmethod
    def standardize_timestamp_columns(df, inplace=True):
        """
        Standardize all timestamp columns in a DataFrame to datetime format.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pd.DataFrame: DataFrame with standardized timestamps (if not inplace)
        """
        if not inplace:
            df = df.copy()
        
        timestamp_cols = DataLoader.find_timestamp_columns(df)
        for col in timestamp_cols:
            DataLoader.convert_timestamp_column(df, col, inplace=True)
        
        return None if inplace else df
    
    @staticmethod
    def get_timestamp_range(df, timestamp_col=None):
        """
        Get the time range (start and end) from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            timestamp_col (str, optional): Timestamp column name. If None, auto-detects.
            
        Returns:
            dict: Dictionary with 'start', 'end', 'days', and 'column' information
        """
        if timestamp_col is None:
            timestamp_col = DataLoader.get_primary_timestamp_column(df, exclude_solar=True)
        
        if not timestamp_col or timestamp_col not in df.columns:
            return {'start': None, 'end': None, 'days': 0, 'column': None}
        
        # Ensure column is datetime
        DataLoader.convert_timestamp_column(df, timestamp_col, inplace=True)
        
        start_time = df[timestamp_col].min()
        end_time = df[timestamp_col].max()
        days = (end_time - start_time).days if start_time and end_time else 0
        
        return {
            'start': start_time,
            'end': end_time,
            'days': days,
            'column': timestamp_col
        }
    
    # =============================================================================
    # EXISTING METHODS
    # =============================================================================
    
    def load_and_filter_data_with_flags(self, site, data_type='env'):
        """Load data and apply quality flag filtering"""
        
        # Load data
        data_file = f'sapwood/{site}_{data_type}_data.csv'
        flags_file = f'sapwood/{site}_{data_type}_flags.csv'
        
        # Use FileManager for file existence checking if available
        if self.file_manager:
            exists, actual_path = self.file_manager.check_file_exists(data_file)
            if not exists:
                return ErrorHandler.handle_file_not_found(f"{data_type} data", data_file)
            data_file = actual_path  # Use the actual path (might include .gz)
        else:
            # Fallback: check file existence manually (final fallback)
            if not os.path.exists(data_file):
                return ErrorHandler.handle_file_not_found(f"{data_type} data", data_file)
        
        # Load data using memory manager if available
        if self.memory_manager:
            data = self.memory_manager.load_data_in_chunks(data_file)
        else:
            data = pd.read_csv(data_file)
            
        if data is None:
            return None
        
        # Memory cleanup after loading
        if self.memory_manager:
            self.memory_manager.check_memory_usage()
        
        # Fix column naming issues
        if 'TIMESTAMP_solar' in data.columns:
            data = data.rename(columns={'TIMESTAMP_solar': 'solar_TIMESTAMP'})
            logger.warning("Renamed TIMESTAMP_solar to solar_TIMESTAMP", indent=2)
        
        # Drop problematic columns (part of key phase)
        excluded_cols = [col for col in PROBLEMATIC_COLUMNS_TO_EXCLUDE if col in data.columns]
        if excluded_cols:
            logger.phase_start("Dropping Problematic Columns")
            data = data.drop(columns=excluded_cols)
            logger.phase_complete("Dropping Problematic Columns", f"{len(excluded_cols)} columns removed")
        
        original_rows = len(data)
        
        # Apply quality flag filtering if flags file exists and filtering is enabled
        # Check quality flags file exists and filter if enabled
        if self.file_manager:
            flags_exists, flags_actual_path = self.file_manager.check_file_exists(flags_file)
            if flags_exists and self.use_quality_flags:
                flags_file = flags_actual_path  # Use actual path (might include .gz)
        else:
            flags_exists = (self.file_manager.check_file_exists(flags_file)[0] if self.file_manager else os.path.exists(flags_file))
        
        if flags_exists and self.use_quality_flags:
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
                    bad_flags_list = BAD_QUALITY_FLAGS  # Use constants
                    if self.config and hasattr(self.config, 'get_quality_flag_setting'):
                        bad_flags_list = self.config.get_quality_flag_setting('bad_flags') or BAD_QUALITY_FLAGS
                    
                    bad_flags = flags[flag_columns].isin(bad_flags_list)
                    rows_with_flags = bad_flags.any(axis=1)
                    
                    flagged_count = rows_with_flags.sum()
                    
                    # Filter out flagged rows
                    data_clean = data[~rows_with_flags]
                    removed_count = original_rows - len(data_clean)
                    
                    # Memory cleanup
                    del flags, bad_flags, rows_with_flags
                    if self.memory_manager:
                        self.memory_manager.force_memory_cleanup()
                    
                    return data_clean
                else:
                    # Memory cleanup even when no flags
                    if self.memory_manager:
                        self.memory_manager.check_memory_usage()
                    return data
                    
            except Exception as e:
                # Memory cleanup on error
                if self.memory_manager:
                    self.memory_manager.check_memory_usage()
                return data
        else:
            # Memory cleanup when no flags file
            if self.memory_manager:
                self.memory_manager.check_memory_usage()
            return data
    
    def validate_data_files(self, site):
        """Validate that required data files exist for a site"""
        env_file = f'sapwood/{site}_env_data.csv'
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        if self.file_manager:
            file_check = self.file_manager.check_files_exist([env_file, sapf_file], ['environmental data', 'sap flow data'])
            files_status = {
                'env_exists': file_check[env_file]['exists'],
                'sapf_exists': file_check[sapf_file]['exists'],
                'env_file': file_check[env_file]['actual_path'],
                'sapf_file': file_check[sapf_file]['actual_path']
            }
        else:
            files_status = {
                'env_exists': os.path.exists(env_file),
                'sapf_exists': os.path.exists(sapf_file),
                'env_file': env_file,
                'sapf_file': sapf_file
            }
        
        return files_status
    
    def get_file_info(self, file_path):
        """Get basic information about a data file"""
        if self.file_manager:
            exists, actual_path = self.file_manager.check_file_exists(file_path)
            if not exists:
                return None
            file_size_mb = self.file_manager.get_file_size_mb(file_path)
        else:
            if not os.path.exists(file_path):
                return None
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            except Exception:
                file_size_mb = 0.0
        
        try:
            
            # Read a small sample to get basic info
            sample = pd.read_csv(file_path, nrows=10)
            
            return {
                'file_path': file_path,
                'size_mb': file_size_mb,
                'columns': sample.columns.tolist(),  # More efficient than list()
                'sample_rows': len(sample),
                'estimated_total_rows': int(file_size_mb * 1000)  # Rough estimate
            }
        except Exception as e:
            return {
                'file_path': file_path,
                'size_mb': 0,
                'error': str(e)
            }
    
    def sample_csv_file(self, file_path, sample_type='head', sample_size=100, specific_columns=None):
        """
        Unified method for sampling CSV files with various strategies.
        
        Args:
            file_path (str): Path to the CSV file
            sample_type (str): Type of sampling - 'head', 'tail', 'random', 'columns_only'
            sample_size (int): Number of rows to sample (ignored for 'columns_only')
            specific_columns (list): Specific columns to include in sample
            
        Returns:
            dict: Sample data and metadata
        """
        # Check file exists
        if self.file_manager:
            exists, actual_path = self.file_manager.check_file_exists(file_path)
            if not exists:
                return {'success': False, 'error': f'File not found: {file_path}'}
            file_path = actual_path
        else:
            if not os.path.exists(file_path):
                return {'success': False, 'error': f'File not found: {file_path}'}
        
        try:
            if sample_type == 'columns_only':
                # Get just column names (no data)
                sample = pd.read_csv(file_path, nrows=0)
                return {
                    'success': True,
                    'sample_type': sample_type,
                    'data': sample,
                    'columns': sample.columns.tolist(),  # More efficient than list()
                    'row_count': 0,
                    'column_count': len(sample.columns)
                }
            
            elif sample_type == 'head':
                # Get first N rows
                sample = pd.read_csv(file_path, nrows=sample_size)
                
            elif sample_type == 'tail':
                # Get last N rows (more complex for CSV)
                # First, estimate total rows
                if self.file_manager:
                    estimated_rows = self.file_manager.estimate_rows_from_file_size(file_path)
                else:
                    # Simple fallback estimation
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    estimated_rows = int(file_size_mb * 1000)
                
                skip_rows = max(0, estimated_rows - sample_size)
                sample = pd.read_csv(file_path, skiprows=range(1, skip_rows + 1))
                
            elif sample_type == 'random':
                # Random sampling requires loading more data first
                chunk_size = max(sample_size * 10, 10000)  # Load larger chunk for random sampling
                temp_data = pd.read_csv(file_path, nrows=chunk_size)
                
                if len(temp_data) <= sample_size:
                    sample = temp_data
                else:
                    sample = temp_data.sample(n=sample_size, random_state=42)
                    
            else:
                return {'success': False, 'error': f'Unsupported sample_type: {sample_type}'}
            
            # Filter to specific columns if requested
            if specific_columns:
                available_columns = [col for col in specific_columns if col in sample.columns]
                if available_columns:
                    sample = sample[available_columns]
                else:
                    return {'success': False, 'error': f'None of the requested columns found: {specific_columns}'}
            
            return {
                'success': True,
                'sample_type': sample_type,
                'data': sample,
                'columns': sample.columns.tolist(),  # More efficient than list()
                'row_count': len(sample),
                'column_count': len(sample.columns),
                'requested_size': sample_size,
                'actual_size': len(sample)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error sampling file: {str(e)}'}
    
    def analyze_data_structure(self, file_path, sample_size=100):
        """
        Analyze the structure and basic properties of a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            sample_size (int): Number of rows to sample for analysis
            
        Returns:
            dict: Comprehensive data structure analysis
        """
        sample_result = self.sample_csv_file(file_path, 'head', sample_size)
        
        if not sample_result['success']:
            return sample_result
        
        sample = sample_result['data']
        
        try:
            # Basic structure analysis
            analysis = {
                'success': True,
                'file_path': file_path,
                'total_columns': len(sample.columns),
                'sample_rows': len(sample),
                'columns': sample.columns.tolist(),  # More efficient than list()
                'data_types': dict(sample.dtypes.astype(str)),
                'memory_usage_mb': sample.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            # Identify special columns
            analysis['timestamp_columns'] = [col for col in sample.columns if 'timestamp' in col.lower()]
            analysis['numeric_columns'] = sample.select_dtypes(include=[np.number]).columns.tolist()
            analysis['text_columns'] = sample.select_dtypes(include=['object']).columns.tolist()
            
            # Check for missing values
            analysis['missing_values'] = dict(sample.isnull().sum())
            analysis['missing_percentage'] = {col: (count / len(sample)) * 100 
                                           for col, count in analysis['missing_values'].items()}
            
            # Get file size information if FileManager available
            if self.file_manager:
                size_info = self.file_manager.get_file_size_classification(file_path)
                analysis['file_size_mb'] = size_info['size_mb']
                analysis['file_size_classification'] = size_info['classification']
                analysis['estimated_total_rows'] = self.file_manager.estimate_rows_from_file_size(file_path)
            
            return analysis
            
        except Exception as e:
            return {'success': False, 'error': f'Error analyzing data structure: {str(e)}'}
    
    def validate_data_quality(self, file_path, expected_columns=None, min_rows=10):
        """
        Validate basic data quality of a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            expected_columns (list): Expected column names
            min_rows (int): Minimum expected number of rows
            
        Returns:
            dict: Data quality validation results
        """
        analysis = self.analyze_data_structure(file_path, sample_size=min(1000, min_rows * 2))
        
        if not analysis['success']:
            return analysis
        
        validation = {
            'success': True,
            'file_path': file_path,
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check minimum rows
        if analysis['sample_rows'] < min_rows:
            validation['issues'].append(f"Insufficient data: {analysis['sample_rows']} rows < {min_rows} minimum")
            validation['is_valid'] = False
        
        # Check expected columns
        if expected_columns:
            missing_columns = [col for col in expected_columns if col not in analysis['columns']]
            if missing_columns:
                validation['issues'].append(f"Missing expected columns: {missing_columns}")
                validation['is_valid'] = False
        
        # Check for completely empty columns
        empty_columns = [col for col, count in analysis['missing_values'].items() 
                        if count == analysis['sample_rows']]
        if empty_columns:
            validation['warnings'].append(f"Completely empty columns: {empty_columns}")
        
        # Check for mostly empty columns (>80% missing)
        mostly_empty = [col for col, pct in analysis['missing_percentage'].items() if pct > 80]
        if mostly_empty:
            validation['warnings'].append(f"Mostly empty columns (>80% missing): {mostly_empty}")
        
        # Check if file has timestamp columns
        if not analysis['timestamp_columns']:
            validation['warnings'].append("No timestamp columns detected")
        
        validation['analysis'] = analysis
        return validation