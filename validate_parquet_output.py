#!/usr/bin/env python3
"""
Parquet Output Validation Script

This script reads through the largest parquet files created by the SAPFLUXNET pipeline
and validates that they were created as intended, checking for:
- File integrity and readability
- Expected column structure 
- Data quality metrics
- Feature completeness
- Memory usage and performance

Usage:
    python validate_parquet_output.py [--top-n 10] [--output-dir processed_parquet]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import psutil
import json


class ParquetValidator:
    """Validates parquet files created by the SAPFLUXNET pipeline"""
    
    def __init__(self, output_dir='processed_parquet', verbose=True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.validation_results = []
        
        # Expected feature categories based on the pipeline
        self.expected_feature_categories = {
            'temporal': ['hour', 'day_of_year', 'month', 'season', 'year'],
            'environmental': ['ta', 'rh', 'vpd', 'sw_in', 'precip', 'ws'],
            'rolling': ['_rolling_mean_', '_rolling_std_', '_rolling_min_', '_rolling_max_'],
            'lagged': ['_lag_', '_lag1h_', '_lag3h_', '_lag6h_', '_lag12h_', '_lag24h_'],
            'interaction': ['vpd_ta_interaction', 'sw_in_ta_interaction', 'precip_rh_interaction'],
            'rate_of_change': ['_roc_', '_change_'],
            'cumulative': ['_cumsum_', '_cumulative_'],
            'metadata': ['latitude', 'longitude', 'elevation', 'biome', 'igbp_class'],
            'categorical': ['biome_encoded', 'igbp_class_encoded', 'leaf_habit_encoded'],
            'target': ['sap_flux', 'sap_velocity']
        }
        
    def get_largest_files(self, top_n=10):
        """Get the N largest parquet files by size"""
        if not self.output_dir.exists():
            print(f"❌ Output directory not found: {self.output_dir}")
            return []
        
        parquet_files = list(self.output_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"❌ No parquet files found in {self.output_dir}")
            return []
        
        # Get file sizes and sort by size (descending)
        file_info = []
        for file_path in parquet_files:
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_info.append((file_path, size_mb))
            except Exception as e:
                print(f"⚠️  Error getting size for {file_path}: {e}")
        
        file_info.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Found {len(parquet_files)} parquet files")
        print(f"🔍 Analyzing top {min(top_n, len(file_info))} largest files:")
        
        for i, (file_path, size_mb) in enumerate(file_info[:top_n]):
            print(f"  {i+1:2d}. {file_path.name:30s} - {size_mb:8.1f}MB")
        
        return [info[0] for info in file_info[:top_n]]
    
    def validate_file_structure(self, file_path):
        """Validate basic file structure and readability"""
        validation = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'readable': False,
            'row_count': 0,
            'column_count': 0,
            'memory_usage_mb': 0,
            'load_time_seconds': 0,
            'errors': []
        }
        
        try:
            start_time = datetime.now()
            
            # Try to read the parquet file
            df = pd.read_parquet(file_path)
            
            load_time = (datetime.now() - start_time).total_seconds()
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            validation.update({
                'readable': True,
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage_mb': memory_usage,
                'load_time_seconds': load_time,
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes.astype(str))
            })
            
            return validation, df
            
        except Exception as e:
            validation['errors'].append(f"Failed to read file: {str(e)}")
            return validation, None
    
    def validate_data_quality(self, df, file_info):
        """Validate data quality and completeness"""
        quality_metrics = {
            'missing_values': {},
            'missing_percentage': {},
            'infinite_values': {},
            'data_ranges': {},
            'unique_counts': {},
            'suspicious_patterns': []
        }
        
        try:
            # Check for missing values
            missing = df.isnull().sum()
            total_rows = len(df)
            
            for col in df.columns:
                missing_count = missing[col]
                missing_pct = (missing_count / total_rows) * 100
                
                quality_metrics['missing_values'][col] = int(missing_count)
                quality_metrics['missing_percentage'][col] = round(missing_pct, 2)
                
                # Check for infinite values in numeric columns
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    inf_count = np.isinf(df[col]).sum()
                    quality_metrics['infinite_values'][col] = int(inf_count)
                    
                    # Get data ranges for numeric columns
                    if not df[col].empty and not df[col].isnull().all():
                        try:
                            quality_metrics['data_ranges'][col] = {
                                'min': float(df[col].min()),
                                'max': float(df[col].max()),
                                'mean': float(df[col].mean()),
                                'std': float(df[col].std())
                            }
                        except:
                            pass
                
                # Count unique values
                unique_count = df[col].nunique()
                quality_metrics['unique_counts'][col] = int(unique_count)
                
                # Flag suspicious patterns
                if missing_pct > 90:
                    quality_metrics['suspicious_patterns'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
                
                if unique_count == 1 and not df[col].isnull().all():
                    quality_metrics['suspicious_patterns'].append(f"Column '{col}' has only one unique value")
            
            # Check for completely duplicate rows
            duplicate_rows = df.duplicated().sum()
            if duplicate_rows > 0:
                duplicate_pct = (duplicate_rows / total_rows) * 100
                quality_metrics['suspicious_patterns'].append(f"{duplicate_rows} duplicate rows ({duplicate_pct:.1f}%)")
        
        except Exception as e:
            quality_metrics['error'] = str(e)
        
        return quality_metrics
    
    def validate_feature_completeness(self, df, file_info):
        """Validate that expected features are present"""
        feature_analysis = {
            'categories_found': {},
            'categories_missing': {},
            'unexpected_features': [],
            'feature_counts': {},
            'completeness_score': 0
        }
        
        try:
            columns = set(df.columns)
            total_expected = 0
            total_found = 0
            
            # Check each feature category
            for category, expected_patterns in self.expected_feature_categories.items():
                found_features = []
                
                for pattern in expected_patterns:
                    matching_cols = [col for col in columns if pattern in col.lower()]
                    found_features.extend(matching_cols)
                
                feature_analysis['categories_found'][category] = found_features
                feature_analysis['feature_counts'][category] = len(found_features)
                
                if not found_features:
                    feature_analysis['categories_missing'][category] = expected_patterns
                
                total_expected += len(expected_patterns)
                total_found += len(found_features)
            
            # Calculate completeness score
            if total_expected > 0:
                feature_analysis['completeness_score'] = round((total_found / total_expected) * 100, 1)
            
            # Identify unexpected features (not matching any known pattern)
            known_patterns = []
            for patterns in self.expected_feature_categories.values():
                known_patterns.extend(patterns)
            
            for col in columns:
                col_lower = col.lower()
                if not any(pattern in col_lower for pattern in known_patterns):
                    feature_analysis['unexpected_features'].append(col)
        
        except Exception as e:
            feature_analysis['error'] = str(e)
        
        return feature_analysis
    
    def validate_temporal_consistency(self, df, file_info):
        """Validate temporal data consistency"""
        temporal_analysis = {
            'timestamp_columns': [],
            'temporal_range': {},
            'temporal_gaps': [],
            'temporal_issues': []
        }
        
        try:
            # Find timestamp columns
            timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
            temporal_analysis['timestamp_columns'] = timestamp_cols
            
            if timestamp_cols:
                main_timestamp = timestamp_cols[0]
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[main_timestamp]):
                    df[main_timestamp] = pd.to_datetime(df[main_timestamp], errors='coerce')
                
                # Get temporal range
                if not df[main_timestamp].isnull().all():
                    temporal_analysis['temporal_range'] = {
                        'start': str(df[main_timestamp].min()),
                        'end': str(df[main_timestamp].max()),
                        'duration_days': (df[main_timestamp].max() - df[main_timestamp].min()).days
                    }
                    
                    # Check for large gaps (>24 hours)
                    df_sorted = df.sort_values(main_timestamp)
                    time_diffs = df_sorted[main_timestamp].diff()
                    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=24)]
                    
                    if len(large_gaps) > 0:
                        temporal_analysis['temporal_gaps'] = [
                            f"Gap of {gap.total_seconds()/3600:.1f} hours"
                            for gap in large_gaps.head(5)  # Show first 5 gaps
                        ]
                
                # Check for invalid timestamps
                invalid_timestamps = df[main_timestamp].isnull().sum()
                if invalid_timestamps > 0:
                    temporal_analysis['temporal_issues'].append(f"{invalid_timestamps} invalid timestamps")
        
        except Exception as e:
            temporal_analysis['error'] = str(e)
        
        return temporal_analysis
    
    def validate_single_file(self, file_path):
        """Validate a single parquet file comprehensively"""
        print(f"\n🔍 Validating: {file_path.name}")
        print(f"📂 Size: {file_path.stat().st_size / (1024 * 1024):.1f}MB")
        
        # Basic structure validation
        file_info, df = self.validate_file_structure(file_path)
        
        if not file_info['readable']:
            print(f"❌ Failed to read file: {file_info['errors']}")
            return file_info
        
        print(f"✅ Readable: {file_info['row_count']:,} rows × {file_info['column_count']} columns")
        print(f"⏱️  Load time: {file_info['load_time_seconds']:.2f}s")
        print(f"💾 Memory usage: {file_info['memory_usage_mb']:.1f}MB")
        
        # Data quality validation
        print("🔬 Analyzing data quality...")
        quality_metrics = self.validate_data_quality(df, file_info)
        
        # Feature completeness validation
        print("🧬 Analyzing feature completeness...")
        feature_analysis = self.validate_feature_completeness(df, file_info)
        
        # Temporal consistency validation
        print("⏰ Analyzing temporal consistency...")
        temporal_analysis = self.validate_temporal_consistency(df, file_info)
        
        # Combine all results
        validation_result = {
            'file_info': file_info,
            'quality_metrics': quality_metrics,
            'feature_analysis': feature_analysis,
            'temporal_analysis': temporal_analysis,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        self.print_validation_summary(validation_result)
        
        return validation_result
    
    def print_validation_summary(self, result):
        """Print a summary of validation results"""
        quality = result['quality_metrics']
        features = result['feature_analysis']
        temporal = result['temporal_analysis']
        
        print("\n📊 VALIDATION SUMMARY:")
        
        # Data Quality Summary
        if 'suspicious_patterns' in quality and quality['suspicious_patterns']:
            print("⚠️  Data Quality Issues:")
            for issue in quality['suspicious_patterns'][:5]:  # Show first 5 issues
                print(f"   • {issue}")
        else:
            print("✅ No major data quality issues detected")
        
        # Feature Completeness Summary
        if 'completeness_score' in features:
            score = features['completeness_score']
            if score >= 80:
                print(f"✅ Feature completeness: {score}% (Good)")
            elif score >= 60:
                print(f"⚠️  Feature completeness: {score}% (Acceptable)")
            else:
                print(f"❌ Feature completeness: {score}% (Poor)")
        
        # Show found feature categories
        if 'feature_counts' in features:
            print("🧬 Feature categories found:")
            for category, count in features['feature_counts'].items():
                if count > 0:
                    emoji = "✅" if count >= len(self.expected_feature_categories.get(category, [])) else "⚠️"
                    print(f"   {emoji} {category}: {count} features")
        
        # Temporal Summary
        if temporal.get('temporal_range'):
            range_info = temporal['temporal_range']
            print(f"⏰ Temporal range: {range_info['duration_days']} days")
            if temporal.get('temporal_gaps'):
                print(f"⚠️  Found {len(temporal['temporal_gaps'])} large temporal gaps")
        
        print("─" * 60)
    
    def generate_report(self, results, output_file=None):
        """Generate a comprehensive validation report"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"parquet_validation_report_{timestamp}.json"
        
        report = {
            'validation_summary': {
                'total_files_validated': len(results),
                'validation_timestamp': datetime.now().isoformat(),
                'system_info': {
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3)
                }
            },
            'file_results': results
        }
        
        # Calculate summary statistics
        readable_files = [r for r in results if r['file_info']['readable']]
        total_rows = sum(r['file_info']['row_count'] for r in readable_files)
        total_size_mb = sum(r['file_info']['file_size_mb'] for r in results)
        
        report['validation_summary'].update({
            'readable_files': len(readable_files),
            'total_rows_processed': total_rows,
            'total_size_mb': total_size_mb,
            'average_completeness_score': np.mean([
                r['feature_analysis'].get('completeness_score', 0) 
                for r in readable_files if 'completeness_score' in r['feature_analysis']
            ]) if readable_files else 0
        })
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        return report
    
    def run_validation(self, top_n=10):
        """Run complete validation on the largest parquet files"""
        print("🚀 Starting Parquet Output Validation")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"💾 System memory: {psutil.virtual_memory().total / (1024**3):.1f}GB total, {psutil.virtual_memory().available / (1024**3):.1f}GB available")
        
        # Get largest files
        largest_files = self.get_largest_files(top_n)
        
        if not largest_files:
            print("❌ No files to validate")
            return []
        
        # Validate each file
        results = []
        for i, file_path in enumerate(largest_files, 1):
            print(f"\n{'='*60}")
            print(f"📋 File {i}/{len(largest_files)}")
            
            try:
                result = self.validate_single_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Error validating {file_path.name}: {str(e)}")
                error_result = {
                    'file_info': {'file_path': str(file_path), 'readable': False, 'errors': [str(e)]},
                    'validation_timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        # Generate comprehensive report
        print(f"\n{'='*60}")
        print("📊 OVERALL VALIDATION SUMMARY")
        report = self.generate_report(results)
        
        readable_count = len([r for r in results if r['file_info']['readable']])
        print(f"✅ Successfully validated: {readable_count}/{len(results)} files")
        print(f"📊 Total rows processed: {report['validation_summary']['total_rows_processed']:,}")
        print(f"💾 Total size processed: {report['validation_summary']['total_size_mb']:.1f}MB")
        print(f"🧬 Average feature completeness: {report['validation_summary']['average_completeness_score']:.1f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Validate SAPFLUXNET parquet output files')
    parser.add_argument('--top-n', type=int, default=10, 
                       help='Number of largest files to validate (default: 10)')
    parser.add_argument('--output-dir', default='processed_parquet',
                       help='Directory containing parquet files (default: processed_parquet)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Create validator
    validator = ParquetValidator(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    # Run validation
    try:
        results = validator.run_validation(top_n=args.top_n)
        
        # Return appropriate exit code
        readable_count = len([r for r in results if r['file_info']['readable']])
        if readable_count == len(results):
            print("\n🎉 All files validated successfully!")
            sys.exit(0)
        elif readable_count > 0:
            print(f"\n⚠️  Partial success: {readable_count}/{len(results)} files validated")
            sys.exit(1)
        else:
            print("\n❌ Validation failed for all files")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()