"""
SAPFLUXNET Data Processing Pipeline

Main entry point for the SAPFLUXNET data processing pipeline.
This file serves as a clean CLI wrapper around the modular orchestrator.

The actual processing logic is handled by the orchestrator which coordinates
all specialized component classes for data processing.

Usage:
    python DataPipeline.py [options]
"""

import sys
import argparse
from datetime import datetime

# Import the orchestrator
from data_processing.Orchestrator import SAPFLUXNETOrchestrator


def main():
    """Main execution function - Clean CLI wrapper around the orchestrator"""
    
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
    
    print("🚀 Starting SAPFLUXNET Data Processing Pipeline")
    print(f"⏰ Started at: {datetime.now()}")
    
    # Display configuration
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
    
    try:
        # Create and run orchestrator
        orchestrator = SAPFLUXNETOrchestrator(
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
            analysis_summary = orchestrator.site_analyzer.analyze_all_sites_data_quality()
            print(f"\n✅ Site analysis complete!")
            print(f"📁 Results saved in site_analysis_results/ directory")
            print(f"📊 Summary: {analysis_summary['total_sites']} sites analyzed")
            print(f"  - ✅ Adequate temporal coverage: {analysis_summary['adequate_temporal']} sites")
            print(f"  - ⚠️  Moderate temporal coverage: {analysis_summary['moderate_temporal']} sites")
            print(f"  - 📉 Insufficient temporal coverage: {analysis_summary['insufficient_temporal']} sites")
            print(f"  - 🚫 No valid data: {analysis_summary['no_valid_data']} sites")
            return True
        
        # Run complete processing pipeline
        print("🎯 Delegating to orchestrator for complete processing...")
        result = orchestrator.orchestrate_complete_processing()
        
        if result:
            print(f"\n🎉 SAPFLUXNET data processing completed successfully!")
            print(f"📄 Output directory: {orchestrator.file_manager.output_dir}")
            print(f"📁 Export format: {orchestrator.export_format.upper()}")
            print(f"🎯 Pipeline orchestrated all components successfully")
        else:
            print(f"\n❌ Data processing pipeline failed")
            print(f"💡 Check orchestrator logs for detailed error information")
            return False
            
    except Exception as e:
        print(f"\n💥 Pipeline execution error: {str(e)}")
        print(f"💡 Check system resources and file permissions")
        return False
    
    finally:
        print(f"\n⏰ Pipeline finished at: {datetime.now()}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
