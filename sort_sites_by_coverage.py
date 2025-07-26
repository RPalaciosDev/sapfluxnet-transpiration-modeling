#!/usr/bin/env python3
"""
Sort SAPFLUXNET Sites by Temporal Coverage
==========================================

Analyzes and sorts all sites by their temporal coverage duration
to identify sites with insufficient data for spatial modeling.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_analysis_results():
    """Load the most recent comprehensive analysis results."""
    analysis_dir = Path("comprehensive_site_analysis")
    
    # Find the most recent analysis file
    json_files = list(analysis_dir.glob("comprehensive_analysis_*.json"))
    if not json_files:
        print("❌ No comprehensive analysis results found!")
        print("   Run comprehensive_site_analysis.py first")
        return None
    
    # Get most recent file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"📂 Loading analysis results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def extract_coverage_data(analysis_results):
    """Extract temporal coverage data for all sites."""
    coverage_data = []
    
    for site_code, analysis in analysis_results.items():
        if 'error' in analysis:
            continue
            
        temporal = analysis.get('temporal', {})
        environmental = analysis.get('environmental', {})
        metadata = analysis.get('metadata', {})
        
        # Extract key information
        duration_days = temporal.get('duration_days', 0)
        total_records = temporal.get('total_records', 0)
        start_date = temporal.get('start_date', 'Unknown')
        end_date = temporal.get('end_date', 'Unknown')
        sampling_freq = temporal.get('sampling_freq_minutes', 'Unknown')
        modeling_score = analysis.get('modeling_suitability_score', 0)
        
        # Environmental coverage
        critical_coverage = environmental.get('critical_coverage', 0)
        env_vars_present = environmental.get('critical_vars_present', 0)
        
        # Site metadata
        site_md = metadata.get('site', {}) or {}
        country = site_md.get('si_country', 'Unknown')
        biome = site_md.get('si_biome', 'Unknown')
        elevation = site_md.get('si_elev', 'Unknown')
        
        coverage_data.append({
            'site_code': site_code,
            'duration_days': duration_days,
            'total_records': total_records,
            'start_date': start_date,
            'end_date': end_date,
            'sampling_freq_min': sampling_freq,
            'modeling_score': modeling_score,
            'critical_env_coverage': critical_coverage,
            'env_vars_present': env_vars_present,
            'country': country,
            'biome': biome,
            'elevation': elevation
        })
    
    return coverage_data

def analyze_and_display_coverage(coverage_data):
    """Analyze and display sites sorted by temporal coverage."""
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(coverage_data)
    
    # Sort by duration (ascending - shortest first)
    df_sorted = df.sort_values('duration_days')
    
    print("\n" + "="*80)
    print("SAPFLUXNET SITES SORTED BY TEMPORAL COVERAGE")
    print("="*80)
    
    # Summary statistics
    print(f"\n📊 COVERAGE SUMMARY:")
    print(f"   Total sites analyzed: {len(df)}")
    print(f"   Shortest coverage: {df['duration_days'].min():.1f} days ({df_sorted.iloc[0]['site_code']})")
    print(f"   Longest coverage: {df['duration_days'].max():.1f} days ({df_sorted.iloc[-1]['site_code']})")
    print(f"   Average coverage: {df['duration_days'].mean():.1f} days")
    print(f"   Median coverage: {df['duration_days'].median():.1f} days")
    
    # Sites with insufficient coverage (< 30 days)
    insufficient = df[df['duration_days'] < 30]
    print(f"\n⚠️  SITES WITH INSUFFICIENT COVERAGE (< 30 days): {len(insufficient)}")
    
    # Coverage categories
    categories = [
        ("🔴 Very Short (< 7 days)", df['duration_days'] < 7),
        ("🟠 Short (7-30 days)", (df['duration_days'] >= 7) & (df['duration_days'] < 30)),
        ("🟡 Moderate (30-90 days)", (df['duration_days'] >= 30) & (df['duration_days'] < 90)),
        ("🟢 Good (90-365 days)", (df['duration_days'] >= 90) & (df['duration_days'] < 365)),
        ("🟢 Excellent (≥ 365 days)", df['duration_days'] >= 365)
    ]
    
    print(f"\n📈 COVERAGE CATEGORIES:")
    for label, mask in categories:
        count = mask.sum()
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} sites ({percentage:.1f}%)")
    
    # Show all sites sorted by coverage
    print(f"\n📋 ALL SITES SORTED BY TEMPORAL COVERAGE:")
    print(f"{'Rank':<4} {'Site Code':<20} {'Days':<8} {'Records':<8} {'Score':<6} {'Env%':<5} {'Country':<8} {'Biome':<15}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        # Color coding for coverage
        if row['duration_days'] < 7:
            icon = "🔴"
        elif row['duration_days'] < 30:
            icon = "🟠"
        elif row['duration_days'] < 90:
            icon = "🟡"
        else:
            icon = "🟢"
        
        env_pct = f"{row['critical_env_coverage']*100:.0f}%" if row['critical_env_coverage'] else "N/A"
        country = str(row['country'])[:7] if row['country'] != 'Unknown' else 'Unknown'
        biome = str(row['biome'])[:14] if row['biome'] != 'Unknown' else 'Unknown'
        
        print(f"{i:<4} {icon} {row['site_code']:<17} {row['duration_days']:<7.1f} {row['total_records']:<8} "
              f"{row['modeling_score']:<6.1f} {env_pct:<5} {country:<8} {biome:<15}")
    
    # Detailed breakdown of problematic sites
    print(f"\n🔍 DETAILED BREAKDOWN - SHORTEST COVERAGE SITES:")
    print(f"{'Site':<20} {'Duration':<10} {'Records':<10} {'Start Date':<12} {'End Date':<12} {'Freq(min)':<10}")
    print("-" * 85)
    
    shortest_20 = df_sorted.head(20)
    for _, row in shortest_20.iterrows():
        start = row['start_date'][:10] if row['start_date'] != 'Unknown' else 'Unknown'
        end = row['end_date'][:10] if row['end_date'] != 'Unknown' else 'Unknown'
        freq = f"{row['sampling_freq_min']}" if row['sampling_freq_min'] != 'Unknown' else 'Unknown'
        
        print(f"{row['site_code']:<20} {row['duration_days']:<10.1f} {row['total_records']:<10} "
              f"{start:<12} {end:<12} {freq:<10}")
    
    return df_sorted

def save_coverage_report(df_sorted):
    """Save detailed coverage analysis to file."""
    output_dir = Path("comprehensive_site_analysis")
    output_file = output_dir / f"temporal_coverage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Save to CSV
    df_sorted.to_csv(output_file, index=False)
    print(f"\n📝 Detailed coverage data saved to: {output_file}")
    
    # Also save a summary report
    summary_file = output_dir / f"coverage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SAPFLUXNET TEMPORAL COVERAGE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Sites: {len(df_sorted)}\n\n")
        
        f.write("COVERAGE STATISTICS:\n")
        f.write(f"  Minimum: {df_sorted['duration_days'].min():.1f} days\n")
        f.write(f"  Maximum: {df_sorted['duration_days'].max():.1f} days\n")
        f.write(f"  Mean: {df_sorted['duration_days'].mean():.1f} days\n")
        f.write(f"  Median: {df_sorted['duration_days'].median():.1f} days\n")
        f.write(f"  Std Dev: {df_sorted['duration_days'].std():.1f} days\n\n")
        
        f.write("SITES WITH INSUFFICIENT COVERAGE (< 30 days):\n")
        f.write("-" * 45 + "\n")
        insufficient = df_sorted[df_sorted['duration_days'] < 30]
        for _, row in insufficient.iterrows():
            f.write(f"{row['site_code']}: {row['duration_days']:.1f} days\n")
        
        f.write(f"\nTOP 20 SHORTEST COVERAGE SITES:\n")
        f.write("-" * 35 + "\n")
        for i, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
            f.write(f"{i:2d}. {row['site_code']}: {row['duration_days']:.1f} days "
                   f"({row['total_records']} records)\n")
    
    print(f"📝 Summary report saved to: {summary_file}")

if __name__ == "__main__":
    print("🚀 Analyzing SAPFLUXNET temporal coverage...")
    
    # Load analysis results
    results = load_analysis_results()
    if results is None:
        exit(1)
    
    # Extract coverage data
    coverage_data = extract_coverage_data(results)
    print(f"✓ Extracted data for {len(coverage_data)} sites")
    
    # Analyze and display
    df_sorted = analyze_and_display_coverage(coverage_data)
    
    # Save detailed report
    save_coverage_report(df_sorted)
    
    print(f"\n✅ Analysis complete!") 