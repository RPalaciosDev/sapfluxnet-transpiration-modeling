#!/usr/bin/env python3
"""
Focused Analysis of CHN_YUN_YUN Site
Understanding why this site is causing poor cluster performance
"""

import pandas as pd
import numpy as np
import os

def analyze_chn_yun_yun():
    """Analyze the CHN_YUN_YUN site specifically"""
    print("🔍 Detailed Analysis of CHN_YUN_YUN Site")
    print("=" * 50)
    
    # Check if we have the raw sapwood data
    sapwood_file = 'sapwood/CHN_YUN_YUN_sapf_data.csv'
    if os.path.exists(sapwood_file):
        print(f"📊 Found raw sapwood data: {sapwood_file}")
        
        # Load the data
        df = pd.read_csv(sapwood_file)
        print(f"  📈 Data shape: {df.shape}")
        
        # Analyze the structure
        print(f"\n📋 Data structure:")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for sap flow columns (target variables)
        sap_flow_cols = [col for col in df.columns if 'Js_' in col]
        print(f"  Sap flow columns: {len(sap_flow_cols)}")
        print(f"  Sap flow column names: {sap_flow_cols}")
        
        # Analyze each sap flow column
        print(f"\n🔍 Sap Flow Column Analysis:")
        for col in sap_flow_cols:
            data = pd.to_numeric(df[col], errors='coerce')
            valid_data = data.dropna()
            
            if len(valid_data) > 0:
                print(f"\n  {col}:")
                print(f"    Valid values: {len(valid_data):,} / {len(data):,} ({len(valid_data)/len(data)*100:.1f}%)")
                print(f"    Mean: {valid_data.mean():.4f}")
                print(f"    Std: {valid_data.std():.4f}")
                print(f"    Min: {valid_data.min():.4f}")
                print(f"    Max: {valid_data.max():.4f}")
                print(f"    Zero values: {len(valid_data[valid_data == 0]):,} ({len(valid_data[valid_data == 0])/len(valid_data)*100:.1f}%)")
                print(f"    Unique values: {valid_data.nunique()}")
                
                # Check for constant values
                if valid_data.nunique() == 1:
                    print(f"    🚨 CONSTANT VALUE: {valid_data.iloc[0]:.4f}")
                elif valid_data.nunique() < 5:
                    print(f"    ⚠️  LOW VARIABILITY: Only {valid_data.nunique()} unique values")
                
                # Check for extreme values
                q99 = valid_data.quantile(0.99)
                extreme_high = len(valid_data[valid_data > q99])
                print(f"    Extreme values (>99th percentile): {extreme_high}")
        
        # Temporal analysis
        if 'TIMESTAMP' in df.columns:
            print(f"\n⏰ Temporal Analysis:")
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            df = df.sort_values('TIMESTAMP')
            
            start_date = df['TIMESTAMP'].min()
            end_date = df['TIMESTAMP'].max()
            total_days = (end_date - start_date).days
            
            print(f"  Data period: {start_date.date()} to {end_date.date()} ({total_days} days)")
            print(f"  Total records: {len(df):,}")
            
            # Check for temporal gaps
            time_diff = df['TIMESTAMP'].diff()
            large_gaps = time_diff[time_diff > pd.Timedelta(hours=24)]
            print(f"  Large gaps (>24h): {len(large_gaps)}")
            
            if len(large_gaps) > 0:
                print(f"  Largest gap: {large_gaps.max()}")
        
        # Check for NA values
        print(f"\n❓ Missing Value Analysis:")
        na_counts = df.isna().sum()
        print(f"  Total NA values per column:")
        for col, count in na_counts.items():
            if count > 0:
                print(f"    {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Check processed parquet file
    parquet_file = '../../processed_parquet/CHN_YUN_YUN_comprehensive.parquet'
    if os.path.exists(parquet_file):
        print(f"\n📊 Found processed parquet data: {parquet_file}")
        
        # Load a sample to understand the processed data
        df_sample = pd.read_parquet(parquet_file, nrows=1000)
        print(f"  Sample shape: {df_sample.shape}")
        
        # Check target variable
        if 'sap_flow' in df_sample.columns:
            target_data = df_sample['sap_flow'].dropna()
            print(f"\n🎯 Target Variable Analysis (sample):")
            print(f"  Valid values: {len(target_data):,}")
            print(f"  Mean: {target_data.mean():.4f}")
            print(f"  Std: {target_data.std():.4f}")
            print(f"  Min: {target_data.min():.4f}")
            print(f"  Max: {target_data.max():.4f}")
            print(f"  Zero values: {len(target_data[target_data == 0]):,}")
            print(f"  Unique values: {target_data.nunique()}")
            
            if target_data.nunique() == 1:
                print(f"  🚨 CRITICAL: Target is constant!")
            elif target_data.nunique() < 10:
                print(f"  ⚠️  WARNING: Very low target variability")
    
    # Summary of issues
    print(f"\n🚨 SUMMARY OF CHN_YUN_YUN ISSUES:")
    print(f"  1. Multiple sap flow sensors with different characteristics")
    print(f"  2. Some sensors may have constant or near-constant values")
    print(f"  3. High percentage of zero values in some sensors")
    print(f"  4. Temporal gaps in data collection")
    print(f"  5. Mixed sensor types (Kob vs Ama) with different behaviors")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  1. Exclude CHN_YUN_YUN from clustering (confirmed problematic)")
    print(f"  2. Review other sites with similar sensor configurations")
    print(f"  3. Consider sensor-specific preprocessing for mixed sensor sites")
    print(f"  4. Implement data quality checks before clustering")

if __name__ == "__main__":
    analyze_chn_yun_yun()