#!/usr/bin/env python3
"""Check if seasonal features exist in parquet files"""

import pandas as pd
import os

def check_seasonal_features():
    """Check if seasonal_temp_range exists in parquet files"""
    parquet_dir = '../../../processed_parquet'
    
    # Check a few files
    test_files = ['ESP_YUN_T1_THI_comprehensive.parquet', 'USA_TNO_comprehensive.parquet']
    
    for filename in test_files:
        file_path = os.path.join(parquet_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                has_seasonal_temp = 'seasonal_temp_range' in df.columns
                has_seasonal_precip = 'seasonal_precip_range' in df.columns
                
                print(f"📁 {filename}:")
                print(f"  📊 Total columns: {len(df.columns)}")
                print(f"  🌡️  seasonal_temp_range: {has_seasonal_temp}")
                print(f"  🌧️  seasonal_precip_range: {has_seasonal_precip}")
                
                if not has_seasonal_temp or not has_seasonal_precip:
                    seasonal_cols = [c for c in df.columns if 'seasonal' in c.lower()]
                    print(f"  📋 Available seasonal columns: {seasonal_cols}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
        else:
            print(f"❌ File not found: {filename}")

if __name__ == "__main__":
    check_seasonal_features()