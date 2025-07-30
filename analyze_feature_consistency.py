#!/usr/bin/env python3
"""
Feature Consistency Analysis for SAPFLUXNET Parquet Files

This script analyzes all processed parquet files to identify:
1. Which features are missing from which sites
2. Feature count variations across sites
3. Common features across all sites
4. Site-specific feature patterns

Usage:
    python analyze_feature_consistency.py --parquet-dir processed_parquet --output results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import argparse
from datetime import datetime
import json

class FeatureConsistencyAnalyzer:
    """Analyze feature consistency across SAPFLUXNET parquet files"""
    
    def __init__(self, parquet_dir='processed_parquet', output_dir='feature_analysis'):
        self.parquet_dir = parquet_dir
        self.output_dir = output_dir
        
        # Storage for analysis results
        self.site_features = {}  # site -> set of features
        self.feature_sites = defaultdict(set)  # feature -> set of sites that have it
        self.site_info = {}  # site -> metadata (file size, row count, etc.)
        
        # Analysis results
        self.all_features = set()
        self.common_features = set()
        self.problematic_features = {}  # features missing from some sites
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_all_sites(self):
        """Analyze all parquet files to extract feature information"""
        print("🔍 SAPFLUXNET FEATURE CONSISTENCY ANALYSIS")
        print("=" * 60)
        print(f"📂 Parquet directory: {self.parquet_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Find all parquet files
        parquet_files = [f for f in os.listdir(self.parquet_dir) if f.endswith('_comprehensive.parquet')]
        parquet_files.sort()
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")
        
        print(f"📊 Found {len(parquet_files)} parquet files")
        print("\n🔍 Analyzing site features...")
        
        # Analyze each site
        for i, parquet_file in enumerate(parquet_files, 1):
            site_name = parquet_file.replace('_comprehensive.parquet', '')
            file_path = os.path.join(self.parquet_dir, parquet_file)
            
            print(f"[{i:3d}/{len(parquet_files)}] {site_name}", end="")
            
            try:
                # Load parquet file and get basic info
                df = pd.read_parquet(file_path)
                
                # Get features (exclude standard non-feature columns)
                exclude_cols = [
                    'sap_flow', 'site', 'TIMESTAMP', 'solar_TIMESTAMP', 
                    'plant_id', 'Unnamed: 0'
                ]
                exclude_suffixes = ['_flags', '_md']
                
                features = []
                for col in df.columns:
                    if col in exclude_cols:
                        continue
                    if any(col.endswith(suffix) for suffix in exclude_suffixes):
                        continue
                    features.append(col)
                
                feature_set = set(features)
                
                # Store site information
                self.site_features[site_name] = feature_set
                self.site_info[site_name] = {
                    'feature_count': len(features),
                    'total_columns': len(df.columns),
                    'row_count': len(df),
                    'file_size_mb': os.path.getsize(file_path) / (1024**2),
                    'has_target': 'sap_flow' in df.columns,
                    'valid_target_rows': len(df.dropna(subset=['sap_flow'])) if 'sap_flow' in df.columns else 0
                }
                
                # Update feature tracking
                self.all_features.update(feature_set)
                for feature in feature_set:
                    self.feature_sites[feature].add(site_name)
                
                print(f" - {len(features)} features, {len(df):,} rows, {self.site_info[site_name]['file_size_mb']:.1f} MB")
                
            except Exception as e:
                print(f" - ❌ ERROR: {e}")
                self.site_info[site_name] = {'error': str(e)}
                continue
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Total unique features found: {len(self.all_features)}")
        
        # Find common features (present in ALL sites)
        total_sites = len([s for s in self.site_info if 'error' not in self.site_info[s]])
        self.common_features = {f for f, sites in self.feature_sites.items() if len(sites) == total_sites}
        
        print(f"🔗 Common features (in all sites): {len(self.common_features)}")
        print(f"⚠️  Variable features: {len(self.all_features) - len(self.common_features)}")
        
    def identify_problematic_features(self):
        """Identify features that are missing from some sites"""
        print(f"\n🔍 Identifying problematic features...")
        
        total_sites = len([s for s in self.site_info if 'error' not in self.site_info[s]])
        
        for feature, sites_with_feature in self.feature_sites.items():
            missing_count = total_sites - len(sites_with_feature)
            if missing_count > 0:
                missing_sites = []
                for site in self.site_info:
                    if 'error' not in self.site_info[site] and site not in sites_with_feature:
                        missing_sites.append(site)
                
                self.problematic_features[feature] = {
                    'present_in': len(sites_with_feature),
                    'missing_from': missing_count,
                    'missing_sites': missing_sites,
                    'present_sites': list(sites_with_feature)
                }
        
        print(f"⚠️  Found {len(self.problematic_features)} features with inconsistent presence")
        
    def generate_reports(self):
        """Generate comprehensive analysis reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n📝 Generating reports...")
        
        # 1. Site Summary Report
        site_summary_file = os.path.join(self.output_dir, f'site_feature_summary_{timestamp}.csv')
        site_data = []
        
        for site, info in self.site_info.items():
            if 'error' in info:
                site_data.append({
                    'site': site,
                    'feature_count': 0,
                    'total_columns': 0,
                    'row_count': 0,
                    'file_size_mb': 0,
                    'has_target': False,
                    'valid_target_rows': 0,
                    'error': info['error']
                })
            else:
                site_data.append({
                    'site': site,
                    'feature_count': info['feature_count'],
                    'total_columns': info['total_columns'],
                    'row_count': info['row_count'],
                    'file_size_mb': round(info['file_size_mb'], 2),
                    'has_target': info['has_target'],
                    'valid_target_rows': info['valid_target_rows'],
                    'error': ''
                })
        
        site_df = pd.DataFrame(site_data)
        site_df = site_df.sort_values('feature_count', ascending=False)
        site_df.to_csv(site_summary_file, index=False)
        print(f"  ✅ Site summary: {site_summary_file}")
        
        # 2. Feature Presence Matrix (for problematic features only)
        if self.problematic_features:
            feature_matrix_file = os.path.join(self.output_dir, f'feature_presence_matrix_{timestamp}.csv')
            
            # Create matrix: sites x problematic features
            sites = [s for s in self.site_info if 'error' not in self.site_info[s]]
            sites.sort()
            
            problematic_feature_names = list(self.problematic_features.keys())
            problematic_feature_names.sort()
            
            matrix_data = []
            for site in sites:
                row = {'site': site}
                for feature in problematic_feature_names:
                    row[feature] = 1 if site in self.feature_sites[feature] else 0
                matrix_data.append(row)
            
            matrix_df = pd.DataFrame(matrix_data)
            matrix_df.to_csv(feature_matrix_file, index=False)
            print(f"  ✅ Feature presence matrix: {feature_matrix_file}")
        
        # 3. Problematic Features Report
        if self.problematic_features:
            problematic_file = os.path.join(self.output_dir, f'problematic_features_{timestamp}.csv')
            
            prob_data = []
            for feature, info in self.problematic_features.items():
                prob_data.append({
                    'feature': feature,
                    'present_in_sites': info['present_in'],
                    'missing_from_sites': info['missing_from'],
                    'coverage_percent': round(info['present_in'] / (info['present_in'] + info['missing_from']) * 100, 1),
                    'missing_sites': '; '.join(info['missing_sites'][:10]) + ('...' if len(info['missing_sites']) > 10 else '')
                })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('missing_from_sites', ascending=False)
            prob_df.to_csv(problematic_file, index=False)
            print(f"  ✅ Problematic features: {problematic_file}")
        
        # 4. Common Features List
        common_features_file = os.path.join(self.output_dir, f'common_features_{timestamp}.txt')
        with open(common_features_file, 'w') as f:
            f.write(f"# Common Features Across All Sites ({len(self.common_features)} features)\n")
            f.write(f"# Generated: {datetime.now()}\n\n")
            for feature in sorted(self.common_features):
                f.write(f"{feature}\n")
        print(f"  ✅ Common features list: {common_features_file}")
        
        # 5. JSON Summary for programmatic use
        summary_json_file = os.path.join(self.output_dir, f'feature_analysis_summary_{timestamp}.json')
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_sites_analyzed': len([s for s in self.site_info if 'error' not in self.site_info[s]]),
            'total_sites_with_errors': len([s for s in self.site_info if 'error' in self.site_info[s]]),
            'total_unique_features': len(self.all_features),
            'common_features_count': len(self.common_features),
            'problematic_features_count': len(self.problematic_features),
            'feature_count_range': {
                'min': min(info['feature_count'] for info in self.site_info.values() if 'error' not in info),
                'max': max(info['feature_count'] for info in self.site_info.values() if 'error' not in info),
                'mean': np.mean([info['feature_count'] for info in self.site_info.values() if 'error' not in info])
            },
            'common_features': sorted(list(self.common_features)),
            'most_problematic_features': sorted(
                [(f, info['missing_from']) for f, info in self.problematic_features.items()],
                key=lambda x: x[1], reverse=True
            )[:20]  # Top 20 most problematic
        }
        
        with open(summary_json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✅ JSON summary: {summary_json_file}")
        
    def print_summary(self):
        """Print analysis summary to console"""
        print(f"\n📊 FEATURE CONSISTENCY ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_sites = len([s for s in self.site_info if 'error' not in self.site_info[s]])
        error_sites = len([s for s in self.site_info if 'error' in self.site_info[s]])
        
        print(f"🏢 Sites analyzed: {total_sites}")
        if error_sites > 0:
            print(f"❌ Sites with errors: {error_sites}")
        
        print(f"🔢 Total unique features: {len(self.all_features)}")
        print(f"✅ Common features (all sites): {len(self.common_features)}")
        print(f"⚠️  Variable features: {len(self.all_features) - len(self.common_features)}")
        
        if self.site_info:
            feature_counts = [info['feature_count'] for info in self.site_info.values() if 'error' not in info]
            print(f"\n📈 Feature count statistics:")
            print(f"  Min: {min(feature_counts)} features")
            print(f"  Max: {max(feature_counts)} features")
            print(f"  Mean: {np.mean(feature_counts):.1f} features")
            print(f"  Std: {np.std(feature_counts):.1f} features")
        
        if self.problematic_features:
            print(f"\n⚠️  Top 10 most problematic features (missing from most sites):")
            sorted_problematic = sorted(
                self.problematic_features.items(),
                key=lambda x: x[1]['missing_from'], reverse=True
            )
            
            for i, (feature, info) in enumerate(sorted_problematic[:10], 1):
                coverage = info['present_in'] / (info['present_in'] + info['missing_from']) * 100
                print(f"  {i:2d}. {feature}: missing from {info['missing_from']} sites ({coverage:.1f}% coverage)")


def main():
    parser = argparse.ArgumentParser(description='Analyze feature consistency across SAPFLUXNET parquet files')
    parser.add_argument('--parquet-dir', default='processed_parquet',
                       help='Directory containing parquet files (default: processed_parquet)')
    parser.add_argument('--output-dir', default='feature_analysis',
                       help='Output directory for analysis results (default: feature_analysis)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FeatureConsistencyAnalyzer(args.parquet_dir, args.output_dir)
        
        # Run analysis
        analyzer.analyze_all_sites()
        analyzer.identify_problematic_features()
        
        # Generate reports
        analyzer.generate_reports()
        
        # Print summary
        analyzer.print_summary()
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {analyzer.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())