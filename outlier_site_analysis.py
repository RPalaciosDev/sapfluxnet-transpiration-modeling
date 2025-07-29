#!/usr/bin/env python3
"""
Outlier Site Analysis for Ecosystem Clustering
Identifies problematic sites that are contaminating clusters and causing poor performance
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierSiteAnalyzer:
    """Analyze sites to identify outliers that hurt cluster performance"""
    
    def __init__(self, parquet_dir='../../processed_parquet'):
        self.parquet_dir = parquet_dir
        self.target_col = 'sap_flow'
        
    def analyze_site_data_quality(self, site_code):
        """Analyze data quality issues for a specific site"""
        parquet_file = os.path.join(self.parquet_dir, f'{site_code}_comprehensive.parquet')
        
        if not os.path.exists(parquet_file):
            print(f"❌ File not found: {parquet_file}")
            return None
        
        print(f"\n🔍 Analyzing {site_code}...")
        
        try:
            # Load data in chunks to handle large files
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_parquet(parquet_file, chunksize=chunk_size):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            
            # Basic statistics
            total_rows = len(df)
            target_data = df[self.target_col].dropna()
            valid_rows = len(target_data)
            
            print(f"  📊 Total rows: {total_rows:,}")
            print(f"  ✅ Valid target rows: {valid_rows:,} ({valid_rows/total_rows*100:.1f}%)")
            
            # Target variable analysis
            if len(target_data) > 0:
                target_stats = target_data.describe()
                print(f"  📈 Target statistics:")
                print(f"    Mean: {target_stats['mean']:.4f}")
                print(f"    Std: {target_stats['std']:.4f}")
                print(f"    Min: {target_stats['min']:.4f}")
                print(f"    Max: {target_stats['max']:.4f}")
                
                # Check for extreme values
                q99 = target_data.quantile(0.99)
                q01 = target_data.quantile(0.01)
                extreme_high = len(target_data[target_data > q99])
                extreme_low = len(target_data[target_data < q01])
                
                print(f"  ⚠️  Extreme values (>99th percentile): {extreme_high}")
                print(f"  ⚠️  Extreme values (<1st percentile): {extreme_low}")
                
                # Check for constant values
                unique_values = target_data.nunique()
                if unique_values == 1:
                    print(f"  🚨 CRITICAL: Target is constant ({target_data.iloc[0]:.4f})")
                elif unique_values < 10:
                    print(f"  ⚠️  WARNING: Only {unique_values} unique target values")
                
                # Check for zero values
                zero_count = len(target_data[target_data == 0])
                zero_pct = zero_count / len(target_data) * 100
                print(f"  📊 Zero values: {zero_count:,} ({zero_pct:.1f}%)")
                
                # Temporal analysis
                if 'TIMESTAMP' in df.columns:
                    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                    df = df.sort_values('TIMESTAMP')
                    
                    # Check for temporal gaps
                    time_diff = df['TIMESTAMP'].diff()
                    large_gaps = len(time_diff[time_diff > pd.Timedelta(hours=24)])
                    print(f"  ⏰ Large temporal gaps (>24h): {large_gaps}")
                    
                    # Check data coverage period
                    start_date = df['TIMESTAMP'].min()
                    end_date = df['TIMESTAMP'].max()
                    total_days = (end_date - start_date).days
                    print(f"  📅 Data period: {start_date.date()} to {end_date.date()} ({total_days} days)")
                    
                    # Calculate data density
                    expected_half_hourly = total_days * 48  # 48 half-hour periods per day
                    actual_half_hourly = len(df)
                    density = actual_half_hourly / expected_half_hourly * 100
                    print(f"  📊 Data density: {density:.1f}% of expected half-hourly records")
                
                return {
                    'site': site_code,
                    'total_rows': total_rows,
                    'valid_rows': valid_rows,
                    'valid_pct': valid_rows/total_rows*100,
                    'mean_target': target_stats['mean'],
                    'std_target': target_stats['std'],
                    'min_target': target_stats['min'],
                    'max_target': target_stats['max'],
                    'unique_values': unique_values,
                    'zero_pct': zero_pct,
                    'extreme_high': extreme_high,
                    'extreme_low': extreme_low,
                    'data_density': density if 'TIMESTAMP' in df.columns else None
                }
            
        except Exception as e:
            print(f"  ❌ Error analyzing {site_code}: {e}")
            return None
    
    def identify_problematic_sites(self, cluster_results_file):
        """Identify sites that are likely causing poor cluster performance"""
        print(f"\n🎯 Identifying problematic sites from cluster results...")
        
        # Load cluster results
        results_df = pd.read_csv(cluster_results_file)
        
        # Find sites with very poor performance
        poor_performance = results_df[results_df['test_r2'] < -10]
        
        if len(poor_performance) > 0:
            print(f"  🚨 Found {len(poor_performance)} folds with very poor performance (R² < -10):")
            for _, row in poor_performance.iterrows():
                print(f"    Cluster {row['cluster']}, Site {row['test_site']}: R² = {row['test_r2']:.4f}")
        
        # Find sites with consistently poor performance
        site_performance = results_df.groupby('test_site')['test_r2'].agg(['mean', 'std', 'count'])
        consistently_poor = site_performance[site_performance['mean'] < -5]
        
        if len(consistently_poor) > 0:
            print(f"\n  ⚠️  Sites with consistently poor performance (mean R² < -5):")
            for site, stats in consistently_poor.iterrows():
                print(f"    {site}: mean R² = {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['count']} folds)")
        
        return list(consistently_poor.index) if len(consistently_poor) > 0 else []
    
    def analyze_site_patterns(self, problematic_sites):
        """Analyze patterns in problematic sites"""
        print(f"\n🔍 Analyzing patterns in {len(problematic_sites)} problematic sites...")
        
        site_analyses = []
        
        for site in problematic_sites:
            analysis = self.analyze_site_data_quality(site)
            if analysis:
                site_analyses.append(analysis)
        
        if site_analyses:
            df_analysis = pd.DataFrame(site_analyses)
            
            print(f"\n📊 Summary of problematic sites:")
            print(f"  Average valid data: {df_analysis['valid_pct'].mean():.1f}%")
            print(f"  Average zero percentage: {df_analysis['zero_pct'].mean():.1f}%")
            print(f"  Sites with <10 unique values: {len(df_analysis[df_analysis['unique_values'] < 10])}")
            print(f"  Sites with constant values: {len(df_analysis[df_analysis['unique_values'] == 1])}")
            
            # Identify common issues
            issues = []
            if df_analysis['valid_pct'].mean() < 50:
                issues.append("Low data quality (many missing values)")
            if df_analysis['zero_pct'].mean() > 50:
                issues.append("High percentage of zero values")
            if len(df_analysis[df_analysis['unique_values'] < 10]) > 0:
                issues.append("Limited target variability")
            if len(df_analysis[df_analysis['unique_values'] == 1]) > 0:
                issues.append("Constant target values")
            
            print(f"\n🚨 Common issues identified:")
            for issue in issues:
                print(f"  - {issue}")
            
            return df_analysis
        
        return None
    
    def create_site_filtering_strategy(self, analysis_df):
        """Create strategy to filter out problematic sites"""
        print(f"\n🎯 Creating site filtering strategy...")
        
        # Define filtering criteria
        filters = []
        
        # Filter 1: Very low data quality
        low_quality = analysis_df[analysis_df['valid_pct'] < 30]
        if len(low_quality) > 0:
            filters.append({
                'name': 'Low Data Quality',
                'sites': list(low_quality['site']),
                'criteria': 'valid_pct < 30%',
                'reason': 'Too many missing values'
            })
        
        # Filter 2: Constant or near-constant values
        constant_values = analysis_df[analysis_df['unique_values'] <= 3]
        if len(constant_values) > 0:
            filters.append({
                'name': 'Constant Values',
                'sites': list(constant_values['site']),
                'criteria': 'unique_values <= 3',
                'reason': 'Target has no variability'
            })
        
        # Filter 3: High percentage of zeros
        high_zeros = analysis_df[analysis_df['zero_pct'] > 80]
        if len(high_zeros) > 0:
            filters.append({
                'name': 'High Zero Percentage',
                'sites': list(high_zeros['site']),
                'criteria': 'zero_pct > 80%',
                'reason': 'Mostly zero values'
            })
        
        # Filter 4: Extreme outliers
        extreme_outliers = analysis_df[
            (analysis_df['std_target'] > 10) | 
            (analysis_df['max_target'] > 100) |
            (analysis_df['extreme_high'] > analysis_df['valid_rows'] * 0.1)
        ]
        if len(extreme_outliers) > 0:
            filters.append({
                'name': 'Extreme Outliers',
                'sites': list(extreme_outliers['site']),
                'criteria': 'high std/max values or >10% extreme values',
                'reason': 'Unrealistic or extreme values'
            })
        
        # Print filtering strategy
        print(f"  📋 Proposed filtering strategy:")
        for i, filter_info in enumerate(filters, 1):
            print(f"    {i}. {filter_info['name']}: {len(filter_info['sites'])} sites")
            print(f"       Criteria: {filter_info['criteria']}")
            print(f"       Reason: {filter_info['reason']}")
            print(f"       Sites: {', '.join(filter_info['sites'])}")
        
        # Get all sites to exclude
        all_excluded = set()
        for filter_info in filters:
            all_excluded.update(filter_info['sites'])
        
        print(f"\n  🚫 Total sites to exclude: {len(all_excluded)}")
        print(f"  📝 Sites: {', '.join(sorted(all_excluded))}")
        
        return filters, list(all_excluded)
    
    def create_improved_clustering_script(self, excluded_sites, output_file='clustering_v3.py'):
        """Create improved clustering script that excludes problematic sites"""
        print(f"\n🔧 Creating improved clustering script...")
        
        # Read original clustering script
        with open('ecosystem/clustering/clustering_v2.py', 'r') as f:
            original_content = f.read()
        
        # Add site filtering
        site_filter_section = f'''
        # SITE FILTERING - Exclude problematic sites identified by outlier analysis
        excluded_sites = {excluded_sites}
        
        def filter_problematic_sites(self, sites_df):
            """Filter out sites known to cause poor cluster performance"""
            print(f"🔍 Filtering out {len(excluded_sites)} problematic sites...")
            original_count = len(sites_df)
            
            # Remove excluded sites
            sites_df = sites_df[~sites_df['site'].isin(excluded_sites)]
            
            filtered_count = len(sites_df)
            print(f"  ✅ Removed {original_count - filtered_count} sites")
            print(f"  📊 Remaining sites: {filtered_count}")
            
            return sites_df
        '''
        
        # Insert the filtering function and modify the main clustering logic
        # This is a simplified version - you'd need to integrate it properly
        improved_content = original_content.replace(
            'def load_site_data(self):',
            site_filter_section + '\n\n    def load_site_data(self):'
        )
        
        # Add call to filtering function
        improved_content = improved_content.replace(
            'sites_df = pd.concat(site_data, ignore_index=True)',
            'sites_df = pd.concat(site_data, ignore_index=True)\n        sites_df = self.filter_problematic_sites(sites_df)'
        )
        
        # Write improved script
        with open(output_file, 'w') as f:
            f.write(improved_content)
        
        print(f"  ✅ Created improved clustering script: {output_file}")
        print(f"  💡 This script excludes {len(excluded_sites)} problematic sites")
        
        return output_file

def main():
    """Main analysis function"""
    print("🚀 Outlier Site Analysis for Ecosystem Clustering")
    print("=" * 60)
    
    analyzer = OutlierSiteAnalyzer()
    
    # Find latest cluster results
    results_files = glob.glob('ecosystem/models/results/parquet_spatial_validation/parquet_spatial_fold_results_*.csv')
    if not results_files:
        print("❌ No cluster results files found")
        return
    
    latest_results = max(results_files, key=os.path.getmtime)
    print(f"📊 Using results file: {os.path.basename(latest_results)}")
    
    # Identify problematic sites
    problematic_sites = analyzer.identify_problematic_sites(latest_results)
    
    if not problematic_sites:
        print("✅ No consistently problematic sites found")
        return
    
    # Analyze problematic sites
    analysis_df = analyzer.analyze_site_patterns(problematic_sites)
    
    if analysis_df is not None:
        # Create filtering strategy
        filters, excluded_sites = analyzer.create_site_filtering_strategy(analysis_df)
        
        # Create improved clustering script
        improved_script = analyzer.create_improved_clustering_script(excluded_sites)
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"  1. Use the improved clustering script: {improved_script}")
        print(f"  2. Re-run clustering with {len(excluded_sites)} sites excluded")
        print(f"  3. Re-run spatial validation to see if performance improves")
        print(f"  4. Consider manual review of excluded sites if needed")

if __name__ == "__main__":
    main()