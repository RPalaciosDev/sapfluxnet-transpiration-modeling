import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import gzip
from .error_utils import ErrorHandler
from .logging_utils import logger, LogLevel


class SiteAnalyzer:
    """Handles site discovery, quality analysis, and validation"""
    
    def __init__(self, config, skip_problematic_sites=True, clean_mode=False, force_reprocess=False, 
                 output_dir='comprehensive_processed', export_format='csv', compress_output=False, 
                 file_manager=None, data_loader=None):
        self.config = config
        self.skip_problematic_sites = skip_problematic_sites
        self.clean_mode = clean_mode
        self.force_reprocess = force_reprocess
        self.output_dir = output_dir
        self.export_format = export_format.lower()
        self.compress_output = compress_output
        self.file_manager = file_manager
        self.data_loader = data_loader
        
        # Runtime site classification sets (populated by analyze_all_sites_data_quality)
        self.sites_with_no_valid_data = set()
        self.sites_with_insufficient_temporal_coverage = set()
        self.sites_with_moderate_temporal_coverage = set()
        self.sites_with_adequate_temporal_coverage = set()
        self.site_analysis_results = {}
        
        # Problematic site constants (quality flag based)
        self.EXTREMELY_PROBLEMATIC_SITES = {
            'IDN_PON_STE',  # 63.1% flag rate - Extremely poor quality
            'USA_NWH',  # 53.4% flag rate - Very poor quality
        }
        
        self.HIGH_PROBLEMATIC_SITES = {
            'ZAF_NOO_E3_IRR',  # 25.9% flag rate - Very poor quality  
            'GUF_GUY_GUY',  # 35.5% flag rate - Very poor quality
            'USA_TNP',  # 31.6% flag rate - Very poor quality
            'USA_TNY',  # 28.9% flag rate - Very poor quality
            'USA_WVF',  # 16.6% flag rate - Very poor quality
        }
        
        self.MODERATE_PROBLEMATIC_SITES = {
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
        
        self.PROBLEMATIC_SITES = (self.EXTREMELY_PROBLEMATIC_SITES | 
                                  self.HIGH_PROBLEMATIC_SITES | 
                                  self.MODERATE_PROBLEMATIC_SITES)
    
    def get_excluded_sites(self):
        """Get combined list of all sites to exclude from processing (dynamic)"""
        return self.sites_with_no_valid_data | self.sites_with_insufficient_temporal_coverage
    


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
        
        # Use FileManager for file existence checking if available
        if self.file_manager:
            file_check = self.file_manager.check_files_exist([env_file, sapf_file], ['environmental data', 'sap flow data'])
            
            if not file_check[env_file]['exists']:
                analysis_result['exclude_reason'] = 'No environmental data file'
                analysis_result['category'] = 'no_env_data'
                return analysis_result
            
            if not file_check[sapf_file]['exists']:
                analysis_result['exclude_reason'] = 'No sap flow data file'
                analysis_result['category'] = 'no_sapf_data'
                return analysis_result
        else:
            # Fallback: use FileManager methods directly
            if not self.file_manager or not self.file_manager.check_file_exists(env_file)[0]:
                analysis_result['exclude_reason'] = 'No environmental data file'
                analysis_result['category'] = 'no_env_data'
                return analysis_result
            
            if not self.file_manager or not self.file_manager.check_file_exists(sapf_file)[0]:
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
            
            # Count total records efficiently - check file size first
            try:
                file_size_mb = self.file_manager.get_file_size_mb(sapf_file) if self.file_manager else os.path.getsize(sapf_file) / (1024 * 1024)
                
                # Skip sampling for very small files
                if file_size_mb < 0.5:  # Less than 500KB is probably too small
                    analysis_result['total_records'] = 0
                    analysis_result['valid_sapf_records'] = 0
                    logger.debug(f"Sap flow file too small ({file_size_mb:.2f}MB) for {site}, skipping sampling")
                else:
                    # File is substantial - sample for quality check
                    if self.data_loader:
                        sample_result = self.data_loader.sample_csv_file(sapf_file, 'head', 1000)
                        if sample_result['success']:
                            sapf_sample = sample_result['data']
                            # Get estimated total using FileManager if available
                            if self.file_manager:
                                estimated_total = self.file_manager.estimate_rows_from_file_size(sapf_file)
                            else:
                                estimated_total = int(file_size_mb * 1000)
                            analysis_result['total_records'] = max(len(sapf_sample), estimated_total)
                        else:
                            analysis_result['data_quality_issues'].append(f"Could not sample sap flow data: {sample_result['error']}")
                            analysis_result['total_records'] = 0
                    else:
                        sapf_sample = pd.read_csv(sapf_file).head(1000)
                        estimated_total = int(file_size_mb * 1000)
                        analysis_result['total_records'] = max(len(sapf_sample), estimated_total)
                
                # Count valid sap flow records (only if we have a sample)
                sapf_cols = sapf_validation['columns']
                if len(sapf_cols) > 0 and 'sapf_sample' in locals():
                    valid_records = 0
                    for col in sapf_cols:
                        if col in sapf_sample.columns:
                            valid_records += sapf_sample[col].notna().sum()
                    analysis_result['valid_sapf_records'] = valid_records
                elif file_size_mb < 0.5:
                    # Already set above for small files
                    pass
                else:
                    analysis_result['valid_sapf_records'] = 0
                
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
            ErrorHandler.handle_processing_error("Site data quality analysis", e, site)
        
        return analysis_result
    
    def analyze_all_sites_data_quality(self):
        """Analyze data quality for all sites and populate classification sets"""
        all_sites = self.get_all_sites()
        logger.processing_start("Site analysis", len(all_sites))
        
        # Clear existing classifications
        self.sites_with_no_valid_data.clear()
        self.sites_with_insufficient_temporal_coverage.clear()
        self.sites_with_moderate_temporal_coverage.clear()
        self.sites_with_adequate_temporal_coverage.clear()
        self.site_analysis_results.clear()
        
        # Analyze each site
        for i, site in enumerate(all_sites, 1):
            # Only show progress for verbose logging
            logger.processing_site(site, i, len(all_sites))
            
            analysis = self.analyze_site_data_quality(site)
            self.site_analysis_results[site] = analysis
            
            # Classify site based on analysis
            if analysis['category'] in ['no_env_data', 'no_sapf_data', 'invalid_sapf_data', 'analysis_error']:
                self.sites_with_no_valid_data.add(site)
            elif analysis['category'] == 'insufficient_temporal':
                self.sites_with_insufficient_temporal_coverage.add(site)
            elif analysis['category'] == 'moderate_temporal':
                self.sites_with_moderate_temporal_coverage.add(site)
            elif analysis['category'] == 'adequate_temporal':
                self.sites_with_adequate_temporal_coverage.add(site)
        
        # Analysis complete - show summary
        logger.analysis_summary(
            total=len(all_sites),
            valid=len(self.sites_with_adequate_temporal_coverage) + len(self.sites_with_moderate_temporal_coverage),
            skipped=len(self.sites_with_no_valid_data) + len(self.sites_with_insufficient_temporal_coverage),
            failed=0
        )
        
        # Save detailed analysis results
        self.save_site_analysis_results()
        
        return {
            'total_sites': len(all_sites),
            'no_valid_data': len(self.sites_with_no_valid_data),
            'insufficient_temporal': len(self.sites_with_insufficient_temporal_coverage),
            'moderate_temporal': len(self.sites_with_moderate_temporal_coverage),
            'adequate_temporal': len(self.sites_with_adequate_temporal_coverage)
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
                            'sites_with_no_valid_data': sorted(self.sites_with_no_valid_data),
            'sites_with_insufficient_temporal_coverage': sorted(self.sites_with_insufficient_temporal_coverage),
            'sites_with_moderate_temporal_coverage': sorted(self.sites_with_moderate_temporal_coverage),
            'sites_with_adequate_temporal_coverage': sorted(self.sites_with_adequate_temporal_coverage),
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
            f.write(f"✅ Sites with adequate temporal coverage (≥90 days): {len(self.sites_with_adequate_temporal_coverage)}\n")
            f.write(f"⚠️  Sites with moderate temporal coverage (30-90 days): {len(self.sites_with_moderate_temporal_coverage)}\n")
            f.write(f"📉 Sites with insufficient temporal coverage (<30 days): {len(self.sites_with_insufficient_temporal_coverage)}\n")
            f.write(f"🚫 Sites with no valid data: {len(self.sites_with_no_valid_data)}\n\n")
            
            # Detailed breakdowns
            if self.sites_with_no_valid_data:
                f.write("SITES WITH NO VALID DATA\n")
                f.write("-" * 25 + "\n")
                for site in sorted(self.sites_with_no_valid_data):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['exclude_reason']}\n")
                f.write("\n")
            
            if self.sites_with_insufficient_temporal_coverage:
                f.write("SITES WITH INSUFFICIENT TEMPORAL COVERAGE (<30 days)\n")
                f.write("-" * 50 + "\n")
                for site in sorted(self.sites_with_insufficient_temporal_coverage):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['temporal_coverage_days']:.1f} days\n")
                f.write("\n")
            
            if self.sites_with_moderate_temporal_coverage:
                f.write("SITES WITH MODERATE TEMPORAL COVERAGE (30-90 days)\n")
                f.write("-" * 47 + "\n")
                for site in sorted(self.sites_with_moderate_temporal_coverage):
                    analysis = self.site_analysis_results[site]
                    f.write(f"  {site}: {analysis['temporal_coverage_days']:.1f} days\n")
                f.write("\n")
        
        logger.processing_complete("Site analysis results saved", 3)
        if logger._should_log(LogLevel.VERBOSE):
            print(f"  📄 JSON: {json_file}")
            print(f"  📊 CSV: {csv_file}")
            print(f"  📝 Report: {report_file}")

    def should_skip_site(self, site):
        """Check if site has already been processed and should be skipped"""
        # Use dynamic analysis results if available
        if site in self.site_analysis_results:
            analysis = self.site_analysis_results[site]
            
            # Check if site has no valid data - always skip these
            if site in self.sites_with_no_valid_data:
                logger.skip_item(site, analysis['exclude_reason'])
                return True
            
            # Check if site has insufficient temporal coverage (<30 days) - always skip these
            if site in self.sites_with_insufficient_temporal_coverage:
                logger.skip_item(site, f"Insufficient temporal coverage ({analysis['temporal_coverage_days']:.1f} days)")
                return True
            
            # Sites with moderate coverage - process with warnings (only in verbose mode)
            if site in self.sites_with_moderate_temporal_coverage:
                logger.warning(f"Processing {site} with moderate coverage ({analysis['temporal_coverage_days']:.1f} days)")
            
            # Sites with adequate coverage - no need to log every one
        
        else:
            # Fallback to individual analysis if site wasn't analyzed yet
            logger.warning(f"{site} not in analysis results, performing individual check")
            
            # Check if site has no valid sap flow data - always skip these
            if site in self.sites_with_no_valid_data:
                logger.skip_item(site, "No valid sap flow data (from previous analysis)")
                return True
            
            # Check if site has insufficient temporal coverage (<30 days) - always skip these
            if site in self.sites_with_insufficient_temporal_coverage:
                logger.skip_item(site, "Insufficient temporal coverage (<30 days, from previous analysis)")
                return True
            
            # Sites with moderate coverage - process with warnings
            if site in self.sites_with_moderate_temporal_coverage:
                logger.warning(f"Processing {site} with moderate coverage (30-90 days, from previous analysis)")
        
        # Check if site is problematic and should be skipped (quality flags)
        if self.skip_problematic_sites and site in self.PROBLEMATIC_SITES:
            if site in self.EXTREMELY_PROBLEMATIC_SITES:
                logger.skip_item(site, "Extremely problematic site (>80% flag rate)")
                return True
            elif hasattr(self, 'clean_mode') and self.clean_mode:
                # In clean mode, only exclude extremely problematic sites
                if site in self.HIGH_PROBLEMATIC_SITES:
                    logger.warning(f"Processing {site} in clean mode - High problematic site (50-80% flag rate)")
                elif site in self.MODERATE_PROBLEMATIC_SITES:
                    logger.warning(f"Processing {site} in clean mode - Moderate quality issues (20-50% flag rate)")
            else:
                # Normal mode - exclude both high and extremely problematic sites
                if site in self.HIGH_PROBLEMATIC_SITES:
                    logger.skip_item(site, "High problematic site (50-80% flag rate)")
                    return True
                elif site in self.MODERATE_PROBLEMATIC_SITES:
                    logger.warning(f"Processing {site} with warnings - Moderate quality issues (20-50% flag rate)")
                    # Don't skip moderate sites, but warn about them
        
        # If force reprocess is enabled, never skip
        if self.force_reprocess:
            return False
        
        # Check for file with appropriate extension
        file_extension = self.file_manager.get_output_file_extension() if self.file_manager else '.csv'
        output_file = f'{self.output_dir}/{site}{file_extension}'
        
        # Use FileManager for file checking if available
        if self.file_manager:
            exists, actual_file_path = self.file_manager.check_file_exists(output_file)
            if not exists:
                return False
            file_size = self.file_manager.get_file_size_mb(output_file, estimate_compressed=False)
        else:
            # Fallback logic
            if not (self.file_manager.check_file_exists(output_file)[0] if self.file_manager else os.path.exists(output_file)):
                return False
            
            # Check file size to ensure it's not empty or corrupted (handle compression)
            actual_file_path = output_file
            if self.file_manager:
                exists, actual_file_path = self.file_manager.check_file_exists(output_file, check_compressed=True)
                if not exists:
                    return False
                file_size = self.file_manager.get_file_size_mb(output_file, estimate_compressed=False)
            else:
                # Final fallback for systems without FileManager
                if self.compress_output and self.export_format == 'csv' and os.path.exists(output_file + '.gz'):
                    actual_file_path = output_file + '.gz'
                elif not os.path.exists(output_file):
                    return False
                file_size = os.path.getsize(actual_file_path) / (1024 * 1024)  # MB
        
        # Skip if file exists and has reasonable size from configuration
        if self.file_manager:
            # Use FileManager's size validation
            min_size_mb = (self.config.get_quality_flag_setting('min_file_size_compressed_mb') 
                          if actual_file_path.endswith('.gz') 
                          else self.config.get_quality_flag_setting('min_file_size_mb'))
            
            size_validation = self.file_manager.validate_file_size(output_file, min_size_mb=min_size_mb)
            if size_validation['valid']:
                # Additional validation: check if file can be read and has expected columns
                if self.validate_existing_file(output_file):
                    return True
                else:
                    logger.warning(f"Found corrupted file for {site} ({size_validation['size_mb']:.1f}MB) - will reprocess")
                    return False
            else:
                logger.warning(f"Found small/corrupted file for {site} ({size_validation['size_mb']:.1f}MB) - will reprocess")
                return False
        else:
            # Fallback logic
            min_size_mb = (self.config.get_quality_flag_setting('min_file_size_compressed_mb') 
                            if actual_file_path.endswith('.gz') 
                            else self.config.get_quality_flag_setting('min_file_size_mb'))
            if file_size > min_size_mb:
                # Additional validation: check if file can be read and has expected columns
                if self.validate_existing_file(output_file):
                    return True
                else:
                    logger.warning(f"Found corrupted file for {site} ({file_size:.1f}MB) - will reprocess")
                    return False
            
            # If file is too small, it might be corrupted or incomplete
            logger.warning(f"Found small/corrupted file for {site} ({file_size:.1f}MB) - will reprocess")
            return False

    def validate_existing_file(self, file_path):
        """Validate that an existing processed file is not corrupted (supports multiple formats)"""
        try:
            # Use FileManager for file path resolution if available
            if self.file_manager:
                exists, actual_file_path = self.file_manager.check_file_exists(file_path)
                if not exists:
                    return False
            else:
                # Fallback logic with minimal direct os.path usage
                if self.file_manager:
                    exists, actual_file_path = self.file_manager.check_file_exists(file_path, check_compressed=True)
                    if not exists:
                        return False
                else:
                    # Final fallback when no FileManager available
                    actual_file_path = file_path
                    if self.compress_output and self.export_format in ['csv', 'libsvm'] and os.path.exists(file_path + '.gz'):
                        actual_file_path = file_path + '.gz'
                    elif not os.path.exists(file_path):
                        return False
            
            # Try to read the first few rows to check if file is valid
            if self.export_format == 'csv':
                try:
                    # Quick validation - just try to read headers first
                    if self.data_loader:
                        headers_result = self.data_loader.sample_csv_file(actual_file_path, 'columns_only')
                        if not headers_result['success'] or len(headers_result['columns']) < 50:
                            return False
                        # Only sample data if headers look good
                        sample_result = self.data_loader.sample_csv_file(actual_file_path, 'head', 5)
                        if sample_result['success']:
                            sample = sample_result['data']
                        else:
                            return False
                    else:
                        sample = pd.read_csv(actual_file_path).head(5)  # Last resort
                except Exception:
                    return False
            elif self.export_format == 'parquet':
                sample = pd.read_parquet(actual_file_path)
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
            
            # Use FileManager for file existence checking if available
            if self.file_manager:
                exists, _ = self.file_manager.check_file_exists(env_file)
                if not exists:
                    return {'valid': False, 'days': 0, 'reason': 'Environmental file not found'}
            else:
                if not (self.file_manager.check_file_exists(env_file)[0] if self.file_manager else os.path.exists(env_file)):
                    return {'valid': False, 'days': 0, 'reason': 'Environmental file not found'}
            
            # Check file size first - if very small, skip expensive sampling
            file_size_mb = self.file_manager.get_file_size_mb(env_file) if self.file_manager else os.path.getsize(env_file) / (1024 * 1024)
            if file_size_mb < 0.1:  # Less than 100KB is probably empty/corrupt
                return {'valid': False, 'days': 0, 'reason': 'Environmental file too small'}
            
            # Only sample if we need to check structure - for timestamp column detection
            if self.data_loader:
                sample_result = self.data_loader.sample_csv_file(env_file, 'head', 100)
                if not sample_result['success']:
                    return {'valid': False, 'days': 0, 'reason': f'Could not sample environmental file: {sample_result["error"]}'}
                sample = sample_result['data']
            else:
                sample = pd.read_csv(env_file).head(100)  # Only sample when necessary
            
            if len(sample) == 0:
                return {'valid': False, 'days': 0, 'reason': 'Empty environmental file'}
            
            # Find timestamp column using centralized utility
            timestamp_col = self.data_loader.get_primary_timestamp_column(sample, exclude_solar=True) if self.data_loader else None
            
            if not timestamp_col:
                # Fallback for backward compatibility
                timestamp_cols = [col for col in sample.columns if 'timestamp' in col.lower() and not col.lower().startswith('solar')]
                timestamp_col = timestamp_cols[0] if timestamp_cols else None
                
            if not timestamp_col:
                return {'valid': False, 'days': 0, 'reason': 'No timestamp column found in environmental data'}
            
            # Read first and last few rows to get time range (memory efficient)
            try:
                # Read first rows
                if self.data_loader:
                    first_sample = self.data_loader.sample_csv_file(env_file, 'head', 50)
                    if not first_sample['success']:
                        return {'valid': False, 'days': 0, 'reason': f'Could not read first chunk: {first_sample["error"]}'}
                    first_chunk = first_sample['data']
                else:
                    if self.data_loader:
                        first_chunk_result = self.data_loader.sample_csv_file(env_file, 'head', 50)
                        if first_chunk_result['success']:
                            first_chunk = first_chunk_result['data']
                        else:
                            return {'valid': False, 'days': 0, 'reason': 'Could not read first chunk'}
                    else:
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
            if self.data_loader:
                sample_result = self.data_loader.sample_csv_file(sapf_file, 'head', 100)
                if not sample_result['success']:
                    return {'valid': False, 'reason': f'Could not sample sap flow file: {sample_result["error"]}', 'columns': []}
                sample = sample_result['data']
            else:
                if self.data_loader:
                    sample_result = self.data_loader.sample_csv_file(sapf_file, 'head', 100)
                    if sample_result['success']:
                        sample = sample_result['data']
                    else:
                        return {'valid': False, 'reason': f'Could not sample sap flow file: {sample_result["error"]}', 'columns': []}
                else:
                    sample = pd.read_csv(sapf_file, nrows=100)
            
            # Check if file has data
            if len(sample) == 0:
                return {'valid': False, 'reason': 'Empty file', 'columns': []}
            
            # Find timestamp column using centralized utility
            sapf_timestamp_col = self.data_loader.get_primary_timestamp_column(sample) if self.data_loader else None
            
            if not sapf_timestamp_col:
                # Fallback for backward compatibility
                sapf_timestamp_cols = [col for col in sample.columns if 'timestamp' in col.lower()]
                sapf_timestamp_col = sapf_timestamp_cols[0] if sapf_timestamp_cols else None
            
            if not sapf_timestamp_col:
                return {'valid': False, 'reason': 'No timestamp column found', 'columns': []}
            
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
        
        return sorted(sites)