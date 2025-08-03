"""
Consistent logging utilities for standardized, configurable messaging.

This module provides a unified logging interface that reduces verbose messaging
while maintaining essential information with configurable verbosity levels.
"""

from enum import Enum
from typing import Optional, Any


class LogLevel(Enum):
    """Logging levels for controlling message verbosity"""
    SILENT = 0      # No output except critical errors
    MINIMAL = 1     # Only essential information
    STANDARD = 2    # Normal operation details  
    VERBOSE = 3     # Detailed progress information
    DEBUG = 4       # Full diagnostic information


class SAPFLUXLogger:
    """
    Centralized logging utility with configurable verbosity levels.
    
    Follows user preference for reduced logging with key phase readouts:
    - Engineering Features
    - Encoding Features  
    - Dropping Problematic Columns
    - Ensuring Data Compatibility
    """
    
    def __init__(self, level: LogLevel = LogLevel.MINIMAL):
        """
        Initialize logger with specified verbosity level.
        
        Args:
            level: Logging level (default: MINIMAL per user preference)
        """
        self.level = level
        
        # Track progress through key phases
        self.current_phase = None
        self.phase_started = False
    
    def set_level(self, level: LogLevel):
        """Change logging level"""
        self.level = level
    
    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if message should be logged based on current level"""
        return self.level.value >= required_level.value
    
    def _format_message(self, emoji: str, message: str, indent: int = 0) -> str:
        """Format message with consistent style"""
        spaces = "  " * indent
        return f"{spaces}{emoji} {message}"
    
    # =============================================================================
    # KEY PHASE LOGGING (Always shown per user preference)
    # =============================================================================
    
    def phase_start(self, phase_name: str):
        """Log start of key processing phase (always shown)"""
        if phase_name != self.current_phase:
            self.current_phase = phase_name
            self.phase_started = True
            print(f"🔧 {phase_name}...")
    
    def phase_complete(self, phase_name: str, details: Optional[str] = None):
        """Log completion of key processing phase (always shown)"""
        if self.phase_started and phase_name == self.current_phase:
            detail_text = f" - {details}" if details else ""
            print(f"✅ {phase_name} complete{detail_text}")
            self.phase_started = False
            self.current_phase = None
    
    # =============================================================================
    # INITIALIZATION LOGGING (MINIMAL level)
    # =============================================================================
    
    def init_component(self, component: str, details: str = ""):
        """Log component initialization"""
        if self._should_log(LogLevel.MINIMAL):
            detail_text = f": {details}" if details else ""
            print(f"📋 {component} initialized{detail_text}")
    
    def init_system(self, memory_total: float, memory_available: float):
        """Log system information at startup"""
        if self._should_log(LogLevel.MINIMAL):
            print(f"💻 System: {memory_total:.1f}GB total, {memory_available:.1f}GB available")
    
    # =============================================================================
    # PROCESSING PROGRESS (STANDARD level)
    # =============================================================================
    
    def processing_start(self, operation: str, count: int = None):
        """Log start of processing operation"""
        if self._should_log(LogLevel.STANDARD):
            count_text = f" ({count} items)" if count else ""
            print(f"🚀 Starting {operation}{count_text}...")
    
    def processing_site(self, site: str, current: int, total: int, details: str = ""):
        """Log individual site processing"""
        if self._should_log(LogLevel.VERBOSE):
            detail_text = f" - {details}" if details else ""
            print(f"  [{current}/{total}] {site}{detail_text}")
    
    def processing_complete(self, operation: str, success_count: int, total_count: int = None):
        """Log completion of processing operation"""
        if self._should_log(LogLevel.MINIMAL):
            if total_count:
                print(f"🎉 {operation} complete: {success_count}/{total_count} successful")
            else:
                print(f"🎉 {operation} complete: {success_count} items processed")
    
    # =============================================================================
    # FILE OPERATIONS (VERBOSE level)
    # =============================================================================
    
    def file_saved(self, filename: str, size_mb: float, format_type: str):
        """Log file save operation"""
        if self._should_log(LogLevel.VERBOSE):
            print(f"  💾 Saved {filename} ({format_type.upper()}, {size_mb:.1f}MB)")
    
    def file_loaded(self, filename: str, rows: int, columns: int):
        """Log file load operation"""
        if self._should_log(LogLevel.DEBUG):
            print(f"  📂 Loaded {filename} ({rows:,} rows, {columns} columns)")
    
    # =============================================================================
    # WARNINGS AND ISSUES (STANDARD level)
    # =============================================================================
    
    def warning(self, message: str, indent: int = 1):
        """Log warning message"""
        if self._should_log(LogLevel.STANDARD):
            print(self._format_message("⚠️", message, indent))
    
    def skip_item(self, item: str, reason: str, indent: int = 1):
        """Log skipped item with reason"""
        if self._should_log(LogLevel.STANDARD):
            print(self._format_message("⏭️", f"Skipping {item}: {reason}", indent))
    
    def data_issue(self, issue: str, count: int = None, indent: int = 1):
        """Log data quality issues"""
        if self._should_log(LogLevel.STANDARD):
            count_text = f" ({count} items)" if count else ""
            print(self._format_message("⚠️", f"{issue}{count_text}", indent))
    
    # =============================================================================
    # ERRORS (ALWAYS shown)
    # =============================================================================
    
    def error(self, message: str, indent: int = 1):
        """Log error message (always shown)"""
        print(self._format_message("❌", message, indent))
    
    def critical_error(self, message: str):
        """Log critical error (always shown)"""
        print(f"💥 CRITICAL: {message}")
    
    # =============================================================================
    # MEMORY AND PERFORMANCE (DEBUG level)
    # =============================================================================
    
    def memory_info(self, operation: str, memory_gb: float, indent: int = 1):
        """Log memory usage information"""
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message("🧠", f"{operation}: {memory_gb:.1f}GB", indent))
    
    def performance_info(self, operation: str, details: str, indent: int = 1):
        """Log performance-related information"""
        if self._should_log(LogLevel.DEBUG):
            print(self._format_message("⚡", f"{operation}: {details}", indent))
    
    # =============================================================================
    # ADAPTIVE DECISIONS (VERBOSE level)
    # =============================================================================
    
    def adaptive_decision(self, decision: str, reason: str, indent: int = 1):
        """Log adaptive processing decisions"""
        if self._should_log(LogLevel.VERBOSE):
            print(self._format_message("🎯", f"{decision} ({reason})", indent))
    
    # =============================================================================
    # FEATURE ENGINEERING (Key phase - always shown)
    # =============================================================================
    
    def feature_group_start(self, group_name: str):
        """Log start of feature group creation (part of key phases)"""
        if self._should_log(LogLevel.VERBOSE):
            print(f"    🔧 {group_name}...")
    
    def feature_group_complete(self, group_name: str, count: int = None):
        """Log completion of feature group"""
        if self._should_log(LogLevel.VERBOSE):
            count_text = f" ({count} features)" if count else ""
            print(f"    ✅ {group_name}{count_text}")
    
    def features_created(self, site: str, rows: int, columns: int):
        """Log feature creation completion"""
        if self._should_log(LogLevel.STANDARD):
            print(f"  ✅ {site}: {rows:,} rows, {columns} features")
    
    # =============================================================================
    # SUMMARY AND STATISTICS (MINIMAL level)  
    # =============================================================================
    
    def summary(self, title: str, stats: dict):
        """Log summary statistics"""
        if self._should_log(LogLevel.MINIMAL):
            print(f"\n📊 {title}:")
            for key, value in stats.items():
                print(f"  • {key}: {value}")
    
    def analysis_summary(self, total: int, valid: int, skipped: int, failed: int):
        """Log analysis summary"""
        if self._should_log(LogLevel.MINIMAL):
            print(f"\n📈 Analysis Summary:")
            print(f"  • Total sites: {total}")
            print(f"  • Valid sites: {valid}")
            print(f"  • Skipped sites: {skipped}")
            print(f"  • Failed sites: {failed}")


# Global logger instance with user-preferred minimal logging
logger = SAPFLUXLogger(LogLevel.MINIMAL)


# Convenience functions for common operations
def set_log_level(level: LogLevel):
    """Set global logging level"""
    logger.set_level(level)


def log_key_phase(phase_name: str):
    """Start logging a key phase (always shown)"""
    logger.phase_start(phase_name)


def complete_key_phase(phase_name: str, details: str = None):
    """Complete logging a key phase (always shown)"""
    logger.phase_complete(phase_name, details)


# Context manager for key phases
class LogPhase:
    """Context manager for key processing phases"""
    
    def __init__(self, phase_name: str, completion_details: str = None):
        self.phase_name = phase_name
        self.completion_details = completion_details
    
    def __enter__(self):
        logger.phase_start(self.phase_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.phase_complete(self.phase_name, self.completion_details)
        else:
            logger.error(f"{self.phase_name} failed: {exc_val}")