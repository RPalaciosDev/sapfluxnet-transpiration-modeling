#!/usr/bin/env python3
"""
Extract site geographic coordinates from full processed parquet files
For use with minimal feature clustering
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def extract_site_coordinates(full_data_dir='processed_parquet', output_file='site_coordinates.json'):
    """Extract longitude, latitude, elevation from full processed parquet files"""
    print(f"🌍 Extracting site coordinates from: {full_data_dir}")
    
    site_coords = {}
    parquet_files = sorted([f for f in os.listdir(full_data_dir) if f.endswith('_comprehensive.parquet')])
    
    for parquet_file in parquet_files:
        site_name = parquet_file.replace('_comprehensive.parquet', '')
        file_path = os.path.join(full_data_dir, parquet_file)
        
        try:
            # Load first row to get site-level metadata
            df_sample = pd.read_parquet(file_path, nrows=1)
            
            # Extract geographic coordinates
            coords = {}
            if 'longitude' in df_sample.columns and pd.notna(df_sample['longitude'].iloc[0]):
                coords['longitude'] = float(df_sample['longitude'].iloc[0])
            if 'latitude' in df_sample.columns and pd.notna(df_sample['latitude'].iloc[0]):
                coords['latitude'] = float(df_sample['latitude'].iloc[0])
            if 'elevation' in df_sample.columns and pd.notna(df_sample['elevation'].iloc[0]):
                coords['elevation'] = float(df_sample['elevation'].iloc[0])
            
            if coords:  # Only add if we have at least some coordinates
                site_coords[site_name] = coords
                print(f"  ✅ {site_name}: {coords}")
            else:
                print(f"  ⚠️  {site_name}: No coordinates found")
                
        except Exception as e:
            print(f"  ❌ {site_name}: Error loading - {e}")
            continue
    
    # Save coordinates
    with open(output_file, 'w') as f:
        json.dump(site_coords, f, indent=2)
    
    print(f"\n💾 Saved coordinates for {len(site_coords)} sites to: {output_file}")
    return site_coords

if __name__ == "__main__":
    extract_site_coordinates()