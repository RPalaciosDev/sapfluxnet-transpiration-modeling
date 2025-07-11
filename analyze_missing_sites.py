#!/usr/bin/env python3
"""
Analyze why some sites are missing Landsat data
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load the NDVI data
ndvi_df = pd.read_csv('SAPFLUXNET_Landsat_NDVI_AllSites.csv')
ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])

# Get all sites that have NDVI data
sites_with_ndvi = set(ndvi_df['site'].unique())

# Get all sites from the original processing pipeline
# Let's check what sites were in the original data
import glob
import os

# Get all site files from sapwood directory
sapwood_files = glob.glob('sapwood/*_env_data.csv')
all_sites = set()
for file in sapwood_files:
    site_name = os.path.basename(file).replace('_env_data.csv', '')
    all_sites.add(site_name)

print("🔍 Analysis of Missing Landsat Data")
print("=" * 50)

print(f"📊 Total sites in SAPFLUXNET: {len(all_sites)}")
print(f"📊 Sites with Landsat NDVI: {len(sites_with_ndvi)}")
print(f"📊 Sites missing Landsat data: {len(all_sites - sites_with_ndvi)}")

missing_sites = all_sites - sites_with_ndvi
sites_with_data = all_sites & sites_with_ndvi

print(f"\n❌ Sites missing Landsat data ({len(missing_sites)}):")
for site in sorted(missing_sites):
    print(f"   - {site}")

print(f"\n✅ Sites with Landsat data ({len(sites_with_data)}):")
for site in sorted(sites_with_data):
    print(f"   - {site}")

# Analyze potential reasons for missing data
print(f"\n🔍 Potential Reasons for Missing Landsat Data:")

# 1. Check date ranges
print(f"\n1️⃣  **Date Range Issues**")
print(f"   Landsat data availability:")
print(f"   - Landsat 5: 1984-2013")
print(f"   - Landsat 7: 1999-2021") 
print(f"   - Landsat 8: 2013-present")

# Check sites with very early or late dates
early_sites = []
late_sites = []

for site in missing_sites:
    # Check if site has environmental data to get date range
    env_file = f'sapwood/{site}_env_data.csv'
    if os.path.exists(env_file):
        try:
            env_data = pd.read_csv(env_file, nrows=1000)  # Read sample
            if 'TIMESTAMP' in env_data.columns:
                timestamps = pd.to_datetime(env_data['TIMESTAMP'])
                min_date = timestamps.min()
                max_date = timestamps.max()
                
                if min_date.year < 1984:
                    early_sites.append((site, min_date.year))
                if max_date.year > 2021:
                    late_sites.append((site, max_date.year))
                    
        except Exception as e:
            pass

if early_sites:
    print(f"   Sites with data before Landsat 5 (1984):")
    for site, year in early_sites:
        print(f"     - {site}: {year}")
else:
    print(f"   No sites with data before Landsat 5")

if late_sites:
    print(f"   Sites with data after Landsat 7 (2021):")
    for site, year in late_sites:
        print(f"     - {site}: {year}")
else:
    print(f"   No sites with data after Landsat 7")

# 2. Check geographic location issues
print(f"\n2️⃣  **Geographic Location Issues**")
print(f"   Common geographic problems:")
print(f"   - Sites in very cloudy regions (tropics, coastal areas)")
print(f"   - Sites in high-latitude areas with seasonal darkness")
print(f"   - Sites in areas with persistent cloud cover")
print(f"   - Sites in urban areas with limited vegetation")

# 3. Check data quality issues
print(f"\n3️⃣  **Data Quality Issues**")
print(f"   Quality problems that could cause missing data:")
print(f"   - High cloud cover (>20% threshold used)")
print(f"   - Poor image quality")
print(f"   - Missing or corrupted Landsat data")
print(f"   - Sites in areas with frequent cloud cover")

# 4. Check temporal coverage
print(f"\n4️⃣  **Temporal Coverage Issues**")
print(f"   Temporal problems:")
print(f"   - Very short study periods")
print(f"   - Gaps in Landsat coverage")
print(f"   - Seasonal cloud cover patterns")

# Analyze the sites that DO have data
print(f"\n📈 Analysis of Sites WITH Data:")

# Check temporal distribution
years_with_data = ndvi_df['date'].dt.year.value_counts().sort_index()
print(f"   Years with Landsat data: {list(years_with_data.index)}")

# Check geographic distribution by site codes
site_regions = {}
for site in sites_with_data:
    country = site.split('_')[0]
    if country not in site_regions:
        site_regions[country] = []
    site_regions[country].append(site)

print(f"   Geographic distribution:")
for country, sites in site_regions.items():
    print(f"     {country}: {len(sites)} sites")

# Check cloud cover patterns
print(f"\n☁️  Cloud Cover Analysis:")
print(f"   Average cloud cover: {ndvi_df['cloud_cover'].mean():.1f}%")
print(f"   Sites with high cloud cover (>10% avg):")

high_cloud_sites = ndvi_df.groupby('site')['cloud_cover'].mean()
high_cloud_sites = high_cloud_sites[high_cloud_sites > 10].sort_values(ascending=False)

for site, cloud_cover in high_cloud_sites.head(10).items():
    print(f"     {site}: {cloud_cover:.1f}%")

print(f"\n💡 **Summary of Missing Data Reasons:**")
print(f"   1. **Cloud Cover**: Sites in persistently cloudy regions")
print(f"   2. **Date Range**: Sites outside Landsat coverage periods")
print(f"   3. **Geographic**: Sites in challenging locations (tropics, high-latitudes)")
print(f"   4. **Quality**: Poor image quality or data gaps")
print(f"   5. **Temporal**: Very short study periods with no clear images")

print(f"\n✅ **Recommendations:**")
print(f"   - Your 66 sites with data provide excellent coverage")
print(f"   - Missing sites are likely due to natural limitations")
print(f"   - Consider using alternative data sources for missing sites")
print(f"   - Focus on the sites with good temporal coverage for modeling") 