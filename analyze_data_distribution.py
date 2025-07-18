#!/usr/bin/env python3
"""
Analyze data distribution by site and country for SAPFLUXNET dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def analyze_data_distribution():
    """Analyze data distribution by site and country."""
    
    print("Loading SAPFLUXNET data...")
    
    # Try to load the main data file
    data_file = "SAPFLUXNET_Landsat_NDVI_AllSites.csv"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return
    
    # Load data in chunks to handle large files
    chunk_size = 10000
    site_counts = Counter()
    country_counts = Counter()
    total_rows = 0
    
    print("Processing data in chunks...")
    
    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
        total_rows += len(chunk)
        
        # Count by site (assuming 'site' column exists)
        if 'site' in chunk.columns:
            site_counts.update(chunk['site'].value_counts().to_dict())
        
        # Count by country (assuming 'country' column exists)
        if 'country' in chunk.columns:
            country_counts.update(chunk['country'].value_counts().to_dict())
        
        # Alternative: try to extract site from other columns
        if 'site' not in chunk.columns:
            # Look for columns that might contain site information
            site_columns = [col for col in chunk.columns if 'site' in col.lower()]
            if site_columns:
                site_col = site_columns[0]
                site_counts.update(chunk[site_col].value_counts().to_dict())
    
    print(f"\nTotal data points: {total_rows:,}")
    
    # Site analysis
    if site_counts:
        print(f"\n{'='*60}")
        print("DATA DISTRIBUTION BY SITE")
        print(f"{'='*60}")
        
        # Convert to DataFrame for better display
        site_df = pd.DataFrame(list(site_counts.items()), columns=['Site', 'Count'])
        site_df = site_df.sort_values('Count', ascending=False)
        site_df['Percentage'] = (site_df['Count'] / total_rows * 100).round(2)
        
        print(f"Total unique sites: {len(site_df)}")
        print(f"\nTop 20 sites by data volume:")
        print(site_df.head(20).to_string(index=False))
        
        print(f"\nBottom 20 sites by data volume:")
        print(site_df.tail(20).to_string(index=False))
        
        # Summary statistics
        print(f"\nSite data statistics:")
        print(f"Mean data points per site: {site_df['Count'].mean():.1f}")
        print(f"Median data points per site: {site_df['Count'].median():.1f}")
        print(f"Standard deviation: {site_df['Count'].std():.1f}")
        print(f"Min data points per site: {site_df['Count'].min()}")
        print(f"Max data points per site: {site_df['Count'].max()}")
        
        # Sites with most data
        top_sites = site_df.head(10)
        print(f"\nTop 10 sites account for {top_sites['Percentage'].sum():.1f}% of total data")
        
        # Sites with least data
        bottom_sites = site_df.tail(10)
        print(f"Bottom 10 sites account for {bottom_sites['Percentage'].sum():.1f}% of total data")
        
        # Create visualizations
        create_site_visualizations(site_df, total_rows)
        
    else:
        print("No site information found in the dataset.")
    
    # Country analysis
    if country_counts:
        print(f"\n{'='*60}")
        print("DATA DISTRIBUTION BY COUNTRY")
        print(f"{'='*60}")
        
        country_df = pd.DataFrame(list(country_counts.items()), columns=['Country', 'Count'])
        country_df = country_df.sort_values('Count', ascending=False)
        country_df['Percentage'] = (country_df['Count'] / total_rows * 100).round(2)
        
        print(f"Total unique countries: {len(country_df)}")
        print(f"\nData distribution by country:")
        print(country_df.to_string(index=False))
        
        # Summary statistics
        print(f"\nCountry data statistics:")
        print(f"Mean data points per country: {country_df['Count'].mean():.1f}")
        print(f"Median data points per country: {country_df['Count'].median():.1f}")
        print(f"Standard deviation: {country_df['Count'].std():.1f}")
        
        # Countries with most data
        top_countries = country_df.head(5)
        print(f"\nTop 5 countries account for {top_countries['Percentage'].sum():.1f}% of total data")
        
        # Create country visualizations
        create_country_visualizations(country_df, total_rows)
        
    else:
        print("No country information found in the dataset.")
    
    # Try to find site information in other files
    print(f"\n{'='*60}")
    print("SEARCHING FOR SITE METADATA")
    print(f"{'='*60}")
    
    # Check if there are any metadata files
    metadata_files = [
        "site_date_summary.csv",
        "docs/data_info.md",
        "docs/comprehensive_data_documentation.md"
    ]
    
    for file_path in metadata_files:
        if os.path.exists(file_path):
            print(f"Found metadata file: {file_path}")
            try:
                if file_path.endswith('.csv'):
                    metadata = pd.read_csv(file_path)
                    print(f"Columns: {list(metadata.columns)}")
                    print(f"Shape: {metadata.shape}")
                    if len(metadata) > 0:
                        print("First few rows:")
                        print(metadata.head())
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Look for site-related information
                        if 'site' in content.lower():
                            print(f"Contains site information")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

def create_site_visualizations(site_df, total_rows):
    """Create visualizations for site data distribution."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Distribution by Site', fontsize=16, fontweight='bold')
    
    # 1. Top 20 sites bar plot
    top_20 = site_df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['Count'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['Site'], fontsize=8)
    axes[0, 0].set_xlabel('Number of Data Points')
    axes[0, 0].set_title('Top 20 Sites by Data Volume')
    axes[0, 0].invert_yaxis()
    
    # 2. Cumulative distribution
    cumulative_pct = np.cumsum(site_df['Percentage'])
    axes[0, 1].plot(range(1, len(cumulative_pct) + 1), cumulative_pct, 'b-', linewidth=2)
    axes[0, 1].axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50%')
    axes[0, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80%')
    axes[0, 1].set_xlabel('Number of Sites')
    axes[0, 1].set_ylabel('Cumulative Percentage of Data')
    axes[0, 1].set_title('Cumulative Data Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of data points per site
    axes[1, 0].hist(site_df['Count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Data Points per Site')
    axes[1, 0].set_ylabel('Number of Sites')
    axes[1, 0].set_title('Distribution of Data Points per Site')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Log-scale histogram for better visualization
    axes[1, 1].hist(site_df['Count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Data Points per Site (log scale)')
    axes[1, 1].set_ylabel('Number of Sites')
    axes[1, 1].set_title('Distribution of Data Points per Site (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('site_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Site distribution visualizations saved as 'site_data_distribution.png'")

def create_country_visualizations(country_df, total_rows):
    """Create visualizations for country data distribution."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Data Distribution by Country', fontsize=16, fontweight='bold')
    
    # 1. Country bar plot
    axes[0].barh(range(len(country_df)), country_df['Count'])
    axes[0].set_yticks(range(len(country_df)))
    axes[0].set_yticklabels(country_df['Country'], fontsize=10)
    axes[0].set_xlabel('Number of Data Points')
    axes[0].set_title('Data Points by Country')
    axes[0].invert_yaxis()
    
    # 2. Pie chart for top countries
    top_countries = country_df.head(10)
    other_count = country_df.iloc[10:]['Count'].sum() if len(country_df) > 10 else 0
    
    if other_count > 0:
        plot_data = pd.concat([top_countries, pd.DataFrame({'Country': ['Others'], 'Count': [other_count]})])
    else:
        plot_data = top_countries
    
    axes[1].pie(plot_data['Count'], labels=plot_data['Country'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Data Distribution by Country (Top 10 + Others)')
    
    plt.tight_layout()
    plt.savefig('country_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Country distribution visualizations saved as 'country_data_distribution.png'")

if __name__ == "__main__":
    analyze_data_distribution() 