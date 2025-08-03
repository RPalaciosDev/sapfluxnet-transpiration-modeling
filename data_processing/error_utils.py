"""
Error handling utilities for standardized error reporting and handling.

This module provides consistent error handling patterns across all data processing components.
"""

from .logging_utils import logger


class ErrorHandler:
    """Standardized error handling utilities following Pattern 3 (FileManager style)"""
    
    @staticmethod
    def handle_file_not_found(file_type, file_path, indent=2):
        """
        Handle file not found errors with consistent formatting.
        
        Args:
            file_type (str): Type of file (e.g., 'data', 'metadata', 'configuration')
            file_path (str): Path to the file that was not found
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            None: Always returns None for file not found errors
        """
        logger.error(f"{file_type.capitalize()} file not found: {file_path}", indent=indent)
        return None
    
    @staticmethod
    def handle_processing_error(operation, error, site=None, indent=2, return_value=False):
        """
        Handle processing errors with consistent formatting.
        
        Args:
            operation (str): Description of the operation that failed
            error (Exception): The exception that occurred
            site (str, optional): Site identifier if relevant
            indent (int): Number of spaces for indentation (default: 2)
            return_value: Value to return (default: False for FileManager pattern)
            
        Returns:
            The specified return_value (typically False for FileManager pattern)
        """
        site_info = f" for {site}" if site else ""
        logger.error(f"{operation} failed{site_info}: {str(error)}", indent=indent)
        return return_value
    
    @staticmethod
    def handle_validation_error(validation_type, reason, site=None, indent=2):
        """
        Handle validation errors with consistent formatting.
        
        Args:
            validation_type (str): Type of validation that failed
            reason (str): Reason for validation failure
            site (str, optional): Site identifier if relevant
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            dict: Error result dictionary with consistent structure
        """
        site_info = f" for {site}" if site else ""
        logger.error(f"{validation_type} validation failed{site_info}: {reason}", indent=indent)
        return {'valid': False, 'reason': reason, 'error': True}
    
    @staticmethod
    def handle_data_loading_error(data_type, file_path, error, indent=2):
        """
        Handle data loading errors with consistent formatting.
        
        Args:
            data_type (str): Type of data being loaded
            file_path (str): Path to the file being loaded
            error (Exception): The exception that occurred
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            None: Always returns None for data loading errors
        """
        logger.error(f"Error loading {data_type} data from {file_path}: {str(error)}", indent=indent)
        return None
    
    @staticmethod
    def handle_memory_error(operation, error, indent=2):
        """
        Handle memory-related errors with consistent formatting.
        
        Args:
            operation (str): Description of the memory operation that failed
            error (Exception): The exception that occurred
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            False: Always returns False for memory errors
        """
        logger.error(f"Memory {operation} failed: {str(error)}", indent=indent)
        return False
    
    @staticmethod
    def handle_io_error(operation, file_path, error, indent=2):
        """
        Handle I/O errors with consistent formatting.
        
        Args:
            operation (str): Description of the I/O operation that failed
            file_path (str): Path to the file involved in the operation
            error (Exception): The exception that occurred
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            False: Always returns False for I/O errors (FileManager pattern)
        """
        logger.error(f"{operation} failed for {file_path}: {str(error)}", indent=indent)
        return False
    
    @staticmethod
    def handle_feature_creation_error(feature_type, error, site=None, indent=2):
        """
        Handle feature creation errors with consistent formatting.
        
        Args:
            feature_type (str): Type of features being created
            error (Exception): The exception that occurred
            site (str, optional): Site identifier if relevant
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            False: Always returns False for feature creation errors
        """
        site_info = f" for {site}" if site else ""
        logger.error(f"{feature_type} feature creation failed{site_info}: {str(error)}", indent=indent)
        return False
    
    @staticmethod
    def log_warning(message, indent=2):
        """
        Log warning messages with consistent formatting.
        
        Args:
            message (str): Warning message
            indent (int): Number of spaces for indentation (default: 2)
        """
        logger.warning(message, indent=indent)
    
    @staticmethod
    def log_success(message, indent=2):
        """
        Log success messages with consistent formatting.
        
        Args:
            message (str): Success message
            indent (int): Number of spaces for indentation (default: 2)
        """
        logger.info(message, indent=indent)
    
    @staticmethod
    def handle_generic_error(error, operation="Operation", return_value=False, indent=2):
        """
        Handle generic errors with consistent formatting (following FileManager pattern).
        
        Args:
            error (Exception): The exception that occurred
            operation (str): Description of the operation that failed
            return_value: Value to return (default: False for FileManager pattern)
            indent (int): Number of spaces for indentation (default: 2)
            
        Returns:
            The specified return_value (typically False for FileManager pattern)
        """
        logger.error(f"{operation} failed: {str(error)}", indent=indent)
        return return_value