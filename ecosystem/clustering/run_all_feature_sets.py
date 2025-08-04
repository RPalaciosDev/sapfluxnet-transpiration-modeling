#!/usr/bin/env python3
"""
Comprehensive Feature Set Comparison Script

Runs all available clustering feature sets on processed ecological data
and saves results with visualizations to organized directories.

This script will test all 12 feature sets:
- geographic, biome, climate, ecological, comprehensive
- performance, environmental, plant_functional  
- v2_core, v2_advanced, v2_hybrid, v3_hybrid

Usage:
    python run_all_feature_sets.py --data-dir ../../parquet_ecological
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

# Add the clustering directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_definitions import FeatureManager

class ComprehensiveFeatureSetRunner:
    """
    Runs all available clustering feature sets and organizes results
    """
    
    def __init__(self, data_dir, base_output_dir='../evaluation/comprehensive_clustering_comparison'):
        self.data_dir = Path(data_dir)
        self.base_output_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create main output directory
        self.comparison_dir = self.base_output_dir / f"comparison_{self.timestamp}"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature manager to get all available sets
        self.feature_manager = FeatureManager()
        self.feature_sets = list(self.feature_manager.feature_sets.keys())
        
        print(f"🚀 Comprehensive Feature Set Comparison")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.comparison_dir}")
        print(f">> Feature sets to test: {len(self.feature_sets)}")
        print(f"⏰ Started at: {datetime.now()}")
        
        # Results tracking
        self.results = {
            'metadata': {
                'timestamp': self.timestamp,
                'data_dir': str(self.data_dir),
                'output_dir': str(self.comparison_dir),
                'total_feature_sets': len(self.feature_sets)
            },
            'feature_sets': {},
            'summary': {}
        }
    
    def run_single_feature_set(self, feature_set_name, clusters='3,4,5,6'):
        """
        Run clustering for a single feature set
        """
        print(f"\n{'='*60}")
        print(f">> PROCESSING FEATURE SET: {feature_set_name.upper()}")
        print(f"{'='*60}")
        
        # Create dedicated output directory for this feature set
        feature_output_dir = self.comparison_dir / f"results_{feature_set_name}"
        feature_output_dir.mkdir(exist_ok=True)
        
        # Get feature set info for logging
        try:
            feature_set = self.feature_manager.get_feature_set(feature_set_name)
            print(f"📊 Features: {feature_set.feature_count} total")
            print(f"   • Numeric: {len(feature_set.numeric_features)}")
            print(f"   • Categorical: {len(feature_set.categorical_features)}")
            print(f"📝 Description: {feature_set.description}")
        except Exception as e:
            print(f"⚠️  Could not get feature set info: {e}")
        
        start_time = time.time()
        
        try:
            # Build command
            cmd = [
                'python', 'FlexibleClusteringPipeline.py',
                '--feature-set', feature_set_name,
                '--data-dir', str(self.data_dir),
                '--output-dir', str(feature_output_dir),
                '--clusters', clusters,
                '--visualize'
            ]
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            # Run the clustering pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {feature_set_name} completed in {elapsed_time:.1f}s")
                
                # Record success
                self.results['feature_sets'][feature_set_name] = {
                    'status': 'success',
                    'elapsed_time': elapsed_time,
                    'output_dir': str(feature_output_dir),
                    'stdout_lines': len(result.stdout.split('\n')),
                    'feature_count': feature_set.feature_count if 'feature_set' in locals() else 'unknown'
                }
                
                # Save stdout for debugging if needed
                with open(feature_output_dir / 'clustering_log.txt', 'w') as f:
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Elapsed time: {elapsed_time:.1f}s\n\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                
                return True
                
            else:
                print(f"❌ FAILED: {feature_set_name} failed after {elapsed_time:.1f}s")
                print(f"   Return code: {result.returncode}")
                
                # Print full error details for debugging
                print(f"\n=== FULL ERROR OUTPUT FOR {feature_set_name} ===")
                print(f"Command: {' '.join(cmd)}")
                if result.stdout:
                    print(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")
                print(f"=== END ERROR OUTPUT ===\n")
                
                # Record failure
                self.results['feature_sets'][feature_set_name] = {
                    'status': 'failed',
                    'elapsed_time': elapsed_time,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
                # Save error log
                with open(feature_output_dir / 'error_log.txt', 'w') as f:
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Elapsed time: {elapsed_time:.1f}s\n\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)
                    f.write("\n\nSTDOUT:\n")
                    f.write(result.stdout)
                
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"💥 EXCEPTION: {feature_set_name} crashed after {elapsed_time:.1f}s")
            print(f"   Error: {str(e)}")
            
            # Record exception
            self.results['feature_sets'][feature_set_name] = {
                'status': 'exception',
                'elapsed_time': elapsed_time,
                'error': str(e)
            }
            
            return False
    
    def run_all_feature_sets(self, clusters='3,4,5,6'):
        """
        Run clustering for all available feature sets
        """
        print(f"\n🎯 STARTING COMPREHENSIVE FEATURE SET COMPARISON")
        print(f"Feature sets to process: {', '.join(self.feature_sets)}")
        
        total_start_time = time.time()
        successful = 0
        failed = 0
        
        for i, feature_set_name in enumerate(self.feature_sets, 1):
            print(f"\n📈 Progress: {i}/{len(self.feature_sets)} feature sets")
            
            success = self.run_single_feature_set(feature_set_name, clusters)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Brief pause between runs
            if i < len(self.feature_sets):
                time.sleep(2)
        
        total_elapsed = time.time() - total_start_time
        
        # Update summary
        self.results['summary'] = {
            'total_time': total_elapsed,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.feature_sets) * 100
        }
        
        print(f"\n{'='*60}")
        print(f"🎉 COMPREHENSIVE COMPARISON COMPLETE!")
        print(f"{'='*60}")
        print(f"⏰ Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        print(f"✅ Successful: {successful}/{len(self.feature_sets)}")
        print(f"❌ Failed: {failed}/{len(self.feature_sets)}")
        print(f"📊 Success rate: {successful/len(self.feature_sets)*100:.1f}%")
        
        # Save comprehensive results
        self.save_results()
        self.create_summary_report()
    
    def save_results(self):
        """Save comprehensive results to JSON"""
        results_file = self.comparison_dir / 'comprehensive_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def create_summary_report(self):
        """Create a human-readable summary report"""
        report_file = self.comparison_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE FEATURE SET COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {self.results['metadata']['timestamp']}\n")
            f.write(f"Data directory: {self.results['metadata']['data_dir']}\n")
            f.write(f"Output directory: {self.results['metadata']['output_dir']}\n")
            f.write(f"Total feature sets: {self.results['metadata']['total_feature_sets']}\n\n")
            
            f.write("SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total time: {self.results['summary']['total_time']:.1f}s\n")
            f.write(f"Successful: {self.results['summary']['successful']}\n")
            f.write(f"Failed: {self.results['summary']['failed']}\n")
            f.write(f"Success rate: {self.results['summary']['success_rate']:.1f}%\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            for feature_set, result in self.results['feature_sets'].items():
                f.write(f"\n{feature_set.upper()}:\n")
                f.write(f"  Status: {result['status']}\n")
                f.write(f"  Time: {result['elapsed_time']:.1f}s\n")
                
                if result['status'] == 'success':
                    f.write(f"  Output: {result['output_dir']}\n")
                    if 'feature_count' in result:
                        f.write(f"  Features: {result['feature_count']}\n")
                elif result['status'] in ['failed', 'exception']:
                    f.write(f"  Error: {result['error'][:100]}...\n")
        
        print(f"📄 Summary report saved to: {report_file}")
    
    def create_directory_index(self):
        """Create an index of all result directories"""
        index_file = self.comparison_dir / 'directory_index.txt'
        
        with open(index_file, 'w') as f:
            f.write("DIRECTORY INDEX - CLUSTERING RESULTS\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Each feature set has its own results directory:\n\n")
            
            for feature_set in self.feature_sets:
                result_dir = f"results_{feature_set}"
                f.write(f"{feature_set:20} -> {result_dir}/\n")
                
                # List expected files
                expected_files = [
                    "flexible_site_clusters_*.csv",
                    "flexible_clustering_strategy_*.json", 
                    "preprocessing_summary_*.json",
                    "visualizations/",
                    "clustering_log.txt"
                ]
                
                for expected in expected_files:
                    f.write(f"{'':25} {expected}\n")
                f.write("\n")
        
        print(f"📋 Directory index saved to: {index_file}")


def main():
    parser = argparse.ArgumentParser(description='Run all clustering feature sets on ecological data')
    
    parser.add_argument('--data-dir', default='../../parquet_ecological',
                       help='Directory containing processed parquet files (default: ../../parquet_ecological)')
    
    parser.add_argument('--output-dir', default='../evaluation/comprehensive_clustering_comparison',
                       help='Base output directory for all results (default: ../evaluation/comprehensive_clustering_comparison)')
    
    parser.add_argument('--clusters', default='3,4,5,6',
                       help='Comma-separated list of cluster numbers to test (default: 3,4,5,6)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without actually running it')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print(f"   Make sure you've processed data with the ecological feature set first:")
        print(f"   python ../../DataPipeline.py --feature-set ecological --export-format parquet")
        sys.exit(1)
    
    # Check for parquet files
    parquet_files = list(Path(args.data_dir).glob('*.parquet'))
    if not parquet_files:
        print(f"❌ Error: No parquet files found in {args.data_dir}")
        print(f"   Make sure you've processed data first.")
        sys.exit(1)
    
    print(f"✅ Found {len(parquet_files)} parquet files in {args.data_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - showing what would be executed:")
        feature_manager = FeatureManager()
        for i, feature_set in enumerate(feature_manager.feature_sets.keys(), 1):
            print(f"  {i:2d}. {feature_set}")
        print(f"\nTotal: {len(feature_manager.feature_sets)} feature sets would be processed")
        return
    
    # Run comprehensive comparison
    runner = ComprehensiveFeatureSetRunner(args.data_dir, args.output_dir)
    runner.create_directory_index()  # Create index before running
    runner.run_all_feature_sets(args.clusters)


if __name__ == '__main__':
    main()