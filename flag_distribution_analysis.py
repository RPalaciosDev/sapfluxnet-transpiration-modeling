#!/usr/bin/env python3
"""
SAPFLUXNET Flag Distribution Analysis

This script analyzes where flagged data points come from - are they spread out
across all sites or concentrated in specific sites?
"""

import pandas as pd
import glob
import os
from datetime import datetime
import numpy as np

def get_all_sites():
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
    return sorted(list(sites))

def analyze_site_flags(site):
    """Analyze flag distribution for a single site"""
    env_warn = 0
    sapf_warn = 0
    env_total = 0
    sapf_total = 0
    
    # Environmental flags
    env_flags_file = f'sapwood/{site}_env_flags.csv'
    if os.path.exists(env_flags_file):
        try:
            env_flags = pd.read_csv(env_flags_file)
            flag_columns = [col for col in env_flags.columns if 'timestamp' not in col.lower()]
            
            for col in flag_columns:
                total_values = len(env_flags[col].dropna())
                flagged_values = ((env_flags[col] == 'OUT_WARN') | (env_flags[col] == 'RANGE_WARN')).sum()
                
                env_total += total_values
                env_warn += flagged_values
                
        except Exception as e:
            print(f"    ❌ Error reading env flags for {site}: {str(e)}")
    
    # Sap flow flags
    sapf_flags_file = f'sapwood/{site}_sapf_flags.csv'
    if os.path.exists(sapf_flags_file):
        try:
            sapf_flags = pd.read_csv(sapf_flags_file)
            flag_columns = [col for col in sapf_flags.columns if 'timestamp' not in col.lower()]
            
            for col in flag_columns:
                total_values = len(sapf_flags[col].dropna())
                flagged_values = ((sapf_flags[col] == 'OUT_WARN') | (sapf_flags[col] == 'RANGE_WARN')).sum()
                
                sapf_total += total_values
                sapf_warn += flagged_values
                
        except Exception as e:
            print(f"    ❌ Error reading sapf flags for {site}: {str(e)}")
    
    return {
        'site': site,
        'env_warn': env_warn,
        'sapf_warn': sapf_warn,
        'env_total': env_total,
        'sapf_total': sapf_total,
        'total_warn': env_warn + sapf_warn,
        'total_data': env_total + sapf_total,
        'env_flag_percent': (env_warn / env_total * 100) if env_total > 0 else 0,
        'sapf_flag_percent': (sapf_warn / sapf_total * 100) if sapf_total > 0 else 0,
        'overall_flag_percent': ((env_warn + sapf_warn) / (env_total + sapf_total) * 100) if (env_total + sapf_total) > 0 else 0
    }

def main():
    print("🔍 Analyzing flag distribution across sites...")
    print(f"⏰ Started at: {datetime.now()}")
    
    all_sites = get_all_sites()
    print(f"📊 Found {len(all_sites)} total sites")
    
    # Analyze all sites
    site_results = []
    total_env_warn = 0
    total_sapf_warn = 0
    total_env_data = 0
    total_sapf_data = 0
    
    for i, site in enumerate(all_sites, 1):
        print(f"[{i}/{len(all_sites)}] {site}")
        result = analyze_site_flags(site)
        site_results.append(result)
        
        total_env_warn += result['env_warn']
        total_sapf_warn += result['sapf_warn']
        total_env_data += result['env_total']
        total_sapf_data += result['sapf_total']
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(site_results)
    
    print(f"\n{'='*80}")
    print(f"📊 FLAG DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"  🌡️  Environmental Data: {total_env_warn:,} flagged / {total_env_data:,} total = {(total_env_warn/total_env_data*100):.1f}%")
    print(f"  🌲 Sap Flow Data: {total_sapf_warn:,} flagged / {total_sapf_data:,} total = {(total_sapf_warn/total_sapf_data*100):.1f}%")
    print(f"  📊 Combined: {total_env_warn + total_sapf_warn:,} flagged / {total_env_data + total_sapf_data:,} total = {((total_env_warn + total_sapf_warn)/(total_env_data + total_sapf_data)*100):.1f}%")
    
    # Sites with flags
    sites_with_flags = df[df['total_warn'] > 0]
    sites_without_flags = df[df['total_warn'] == 0]
    
    print(f"\n🏷️  Site Distribution:")
    print(f"  📊 Sites with flags: {len(sites_with_flags)} ({len(sites_with_flags)/len(df)*100:.1f}%)")
    print(f"  ✅ Sites without flags: {len(sites_without_flags)} ({len(sites_without_flags)/len(df)*100:.1f}%)")
    
    # Top sites by flag count
    print(f"\n🔥 Top 10 Sites by Total Flag Count:")
    top_sites = df.nlargest(10, 'total_warn')[['site', 'total_warn', 'overall_flag_percent']]
    for _, row in top_sites.iterrows():
        print(f"  {row['site']}: {row['total_warn']:,} flags ({row['overall_flag_percent']:.1f}%)")
    
    # Top sites by flag percentage
    print(f"\n⚠️  Top 10 Sites by Flag Percentage:")
    top_percent_sites = df[df['total_data'] > 1000].nlargest(10, 'overall_flag_percent')[['site', 'total_warn', 'overall_flag_percent']]
    for _, row in top_percent_sites.iterrows():
        print(f"  {row['site']}: {row['total_warn']:,} flags ({row['overall_flag_percent']:.1f}%)")
    
    # Distribution analysis
    print(f"\n📊 Distribution Analysis:")
    
    # How many sites account for 50% of flags?
    df_sorted = df.sort_values('total_warn', ascending=False)
    cumulative_flags = df_sorted['total_warn'].cumsum()
    total_flags = df_sorted['total_warn'].sum()
    
    sites_for_50_percent = len(cumulative_flags[cumulative_flags <= total_flags * 0.5])
    sites_for_80_percent = len(cumulative_flags[cumulative_flags <= total_flags * 0.8])
    sites_for_90_percent = len(cumulative_flags[cumulative_flags <= total_flags * 0.9])
    
    print(f"  🎯 50% of flags come from: {sites_for_50_percent} sites ({sites_for_50_percent/len(df)*100:.1f}% of all sites)")
    print(f"  🎯 80% of flags come from: {sites_for_80_percent} sites ({sites_for_80_percent/len(df)*100:.1f}% of all sites)")
    print(f"  🎯 90% of flags come from: {sites_for_90_percent} sites ({sites_for_90_percent/len(df)*100:.1f}% of all sites)")
    
    # Flag concentration metrics
    print(f"\n📈 Concentration Metrics:")
    mean_flags_per_site = df['total_warn'].mean()
    median_flags_per_site = df['total_warn'].median()
    std_flags_per_site = df['total_warn'].std()
    
    print(f"  📊 Mean flags per site: {mean_flags_per_site:,.0f}")
    print(f"  📊 Median flags per site: {median_flags_per_site:,.0f}")
    print(f"  📊 Standard deviation: {std_flags_per_site:,.0f}")
    print(f"  📊 Coefficient of variation: {std_flags_per_site/mean_flags_per_site:.2f}")
    
    # Flag distribution categories
    print(f"\n📋 Flag Distribution Categories:")
    low_flag_sites = df[df['total_warn'] <= 1000]
    medium_flag_sites = df[(df['total_warn'] > 1000) & (df['total_warn'] <= 10000)]
    high_flag_sites = df[(df['total_warn'] > 10000) & (df['total_warn'] <= 50000)]
    very_high_flag_sites = df[df['total_warn'] > 50000]
    
    print(f"  🟢 Low flags (≤1K): {len(low_flag_sites)} sites ({len(low_flag_sites)/len(df)*100:.1f}%)")
    print(f"  🟡 Medium flags (1K-10K): {len(medium_flag_sites)} sites ({len(medium_flag_sites)/len(df)*100:.1f}%)")
    print(f"  🟠 High flags (10K-50K): {len(high_flag_sites)} sites ({len(high_flag_sites)/len(df)*100:.1f}%)")
    print(f"  🔴 Very high flags (>50K): {len(very_high_flag_sites)} sites ({len(very_high_flag_sites)/len(df)*100:.1f}%)")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'flag_distribution_analysis_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print(f"\n⏰ Finished at: {datetime.now()}")

if __name__ == "__main__":
    main() 