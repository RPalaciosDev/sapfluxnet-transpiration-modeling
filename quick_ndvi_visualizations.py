#!/usr/bin/env python3
"""
Quick NDVI Visualization Script
Creates essential visualizations for SAPFLUXNET NDVI data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def create_quick_visualizations():
    """Create quick NDVI visualizations"""
    
    # Load data
    print("📊 Loading NDVI data...")
    df = pd.read_csv('SAPFLUXNET_Landsat_NDVI_AllSites.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    print(f"✅ Loaded {len(df):,} NDVI observations from {df['site'].nunique()} sites")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. NDVI Distribution
    axes[0, 0].hist(df['ndvi'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].set_title('NDVI Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('NDVI')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add interpretation lines
    axes[0, 0].axvline(0.1, color='red', linestyle='--', alpha=0.7, label='Sparse vegetation')
    axes[0, 0].axvline(0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate vegetation')
    axes[0, 0].axvline(0.5, color='darkgreen', linestyle='--', alpha=0.7, label='Dense vegetation')
    axes[0, 0].legend()
    
    # 2. Seasonal Patterns
    monthly_ndvi = df.groupby('month')['ndvi'].agg(['mean', 'std']).reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    axes[0, 1].errorbar(monthly_ndvi['month'], monthly_ndvi['mean'], 
                       yerr=monthly_ndvi['std'], marker='o', capsize=5, 
                       color='forestgreen', linewidth=2)
    axes[0, 1].set_title('Seasonal NDVI Patterns', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('NDVI (Mean ± Std)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(month_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Top Sites Time Series
    top_sites = df['site'].value_counts().head(6).index
    colors = plt.cm.Set3(np.linspace(0, 1, 6))
    
    for i, site in enumerate(top_sites):
        site_data = df[df['site'] == site].sort_values('date')
        axes[0, 2].plot(site_data['date'], site_data['ndvi'], 
                       alpha=0.8, label=site, color=colors[i], linewidth=1.5)
    
    axes[0, 2].set_title('NDVI Time Series - Top 6 Sites', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Date')
    axes[0, 2].set_ylabel('NDVI')
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cloud Cover vs NDVI
    axes[1, 0].scatter(df['cloud_cover'], df['ndvi'], alpha=0.5, s=2, color='steelblue')
    axes[1, 0].set_title('Cloud Cover vs NDVI', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Cloud Cover (%)')
    axes[1, 0].set_ylabel('NDVI')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['cloud_cover'], df['ndvi'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(df['cloud_cover'], p(df['cloud_cover']), "r--", alpha=0.8)
    
    # 5. Annual Trends
    annual_ndvi = df.groupby('year')['ndvi'].agg(['mean', 'std', 'count']).reset_index()
    
    axes[1, 1].errorbar(annual_ndvi['year'], annual_ndvi['mean'], 
                       yerr=annual_ndvi['std'], marker='o', capsize=5, 
                       color='darkgreen', linewidth=2)
    axes[1, 1].set_title('Annual NDVI Trends', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('NDVI (Mean ± Std)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Site Statistics
    site_stats = df.groupby('site')['ndvi'].agg(['mean', 'std', 'count']).reset_index()
    site_stats = site_stats[site_stats['count'] >= 5]  # Filter sites with enough data
    
    scatter = axes[1, 2].scatter(site_stats['mean'], site_stats['std'], 
                                s=site_stats['count']*3, alpha=0.6, 
                                c=site_stats['count'], cmap='viridis')
    
    axes[1, 2].set_title('Site NDVI: Mean vs Variability', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Mean NDVI')
    axes[1, 2].set_ylabel('NDVI Standard Deviation')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('Number of Observations')
    
    plt.tight_layout()
    plt.savefig('quick_ndvi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 NDVI Data Summary:")
    print(f"   Mean NDVI: {df['ndvi'].mean():.3f}")
    print(f"   Median NDVI: {df['ndvi'].median():.3f}")
    print(f"   Standard deviation: {df['ndvi'].std():.3f}")
    print(f"   Range: {df['ndvi'].min():.3f} to {df['ndvi'].max():.3f}")
    print(f"   Mean cloud cover: {df['cloud_cover'].mean():.1f}%")
    
    print(f"\n🌿 Vegetation Health Distribution:")
    dense_veg = (df['ndvi'] > 0.5).sum()
    moderate_veg = ((df['ndvi'] >= 0.3) & (df['ndvi'] <= 0.5)).sum()
    sparse_veg = ((df['ndvi'] >= 0.1) & (df['ndvi'] < 0.3)).sum()
    bare = (df['ndvi'] < 0.1).sum()
    
    print(f"   Dense vegetation (>0.5): {dense_veg:,} ({dense_veg/len(df)*100:.1f}%)")
    print(f"   Moderate vegetation (0.3-0.5): {moderate_veg:,} ({moderate_veg/len(df)*100:.1f}%)")
    print(f"   Sparse vegetation (0.1-0.3): {sparse_veg:,} ({sparse_veg/len(df)*100:.1f}%)")
    print(f"   Bare/water (<0.1): {bare:,} ({bare/len(df)*100:.1f}%)")
    
    print(f"\n🏆 Top 5 Sites by Data Volume:")
    top_sites_count = df['site'].value_counts().head(5)
    for i, (site, count) in enumerate(top_sites_count.items(), 1):
        print(f"   {i}. {site}: {count} observations")
    
    print(f"\n✅ Visualization saved as 'quick_ndvi_analysis.png'")
    
    return df

if __name__ == "__main__":
    df = create_quick_visualizations() 