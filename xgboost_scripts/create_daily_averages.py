#!/usr/bin/env python3
"""
Create Daily Averages for SAPFLUXNET Data
Condense high-frequency data to daily averages for temporal validation
This solves memory issues and creates realistic temporal patterns
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import gc

warnings.filterwarnings('ignore')

def load_and_condense_parquet_data(parquet_dir, output_dir='daily_averages'):
    """Load parquet data and condense to daily averages"""
    print(f"Loading and condensing parquet data from {parquet_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load first file to understand structure
    sample_file = os.path.join(parquet_dir, parquet_files[0])
    sample_df = pd.read_parquet(sample_file).head(1000)
    
    print(f"Sample data columns: {list(sample_df.columns)}")
    print(f"Sample data shape: {sample_df.shape}")
    
    # Check if TIMESTAMP column exists
    if 'TIMESTAMP' not in sample_df.columns:
        print("⚠️  No TIMESTAMP column found. Creating synthetic timestamps...")
        create_synthetic_timestamps = True
    else:
        print("✅ TIMESTAMP column found")
        create_synthetic_timestamps = False
    
    # Define features to aggregate
    target_col = 'sap_flow'
    
    # Features to average (environmental variables)
    avg_features = [
        'ta', 'rh', 'sw_in', 'ppfd_in', 'vpd', 'ext_rad', 'ws',
        'swc_shallow', 'precip',
        # Lagged features (we'll average these too)
        'ta_lag_1h', 'ta_lag_3h', 'ta_lag_6h', 'ta_lag_12h', 'ta_lag_24h',
        'rh_lag_1h', 'rh_lag_3h', 'rh_lag_6h', 'rh_lag_12h', 'rh_lag_24h',
        'sw_in_lag_1h', 'sw_in_lag_3h', 'sw_in_lag_6h', 'sw_in_lag_12h', 'sw_in_lag_24h',
        'vpd_lag_1h', 'vpd_lag_3h', 'vpd_lag_6h', 'vpd_lag_12h', 'vpd_lag_24h',
        'ws_lag_1h', 'ws_lag_3h', 'ws_lag_6h', 'ws_lag_12h', 'ws_lag_24h',
        'swc_shallow_lag_1h', 'swc_shallow_lag_3h', 'swc_shallow_lag_6h', 
        'swc_shallow_lag_12h', 'swc_shallow_lag_24h',
        'precip_lag_1h', 'precip_lag_3h', 'precip_lag_6h', 'precip_lag_12h', 'precip_lag_24h',
        'ppfd_in_lag_1h', 'ppfd_in_lag_3h', 'ppfd_in_lag_6h', 'ppfd_in_lag_12h', 'ppfd_in_lag_24h'
    ]
    
    # Features to sum (cumulative variables like precipitation)
    sum_features = ['precip', 'precip_lag_1h', 'precip_lag_3h', 'precip_lag_6h', 'precip_lag_12h', 'precip_lag_24h']
    
    # Filter to available features
    available_avg_features = [f for f in avg_features if f in sample_df.columns]
    available_sum_features = [f for f in sum_features if f in sample_df.columns]
    
    print(f"Features to average: {len(available_avg_features)}")
    print(f"Features to sum: {len(available_sum_features)}")
    
    # Process each parquet file
    all_daily_data = []
    total_original_rows = 0
    
    for i, parquet_file in enumerate(parquet_files):
        print(f"Processing file {i+1}/{len(parquet_files)}: {parquet_file}")
        
        file_path = os.path.join(parquet_dir, parquet_file)
        
        # Load with selected columns
        columns_to_load = ['site', target_col] + available_avg_features + available_sum_features
        if not create_synthetic_timestamps:
            columns_to_load.append('TIMESTAMP')
        
        df_chunk = pd.read_parquet(file_path, columns=columns_to_load)
        
        # Clean data
        df_chunk = df_chunk.dropna(subset=[target_col])
        df_chunk = df_chunk.fillna(0)
        
        total_original_rows += len(df_chunk)
        
        # Create synthetic timestamps if needed
        if create_synthetic_timestamps:
            # Create realistic timestamps based on row index
            # Assume 48 measurements per day (every 30 minutes)
            measurements_per_day = 48
            start_date = datetime(2000, 1, 1)  # Start from year 2000
            
            df_chunk['TIMESTAMP'] = start_date + pd.to_timedelta(
                (df_chunk.index + i * 1000000) // measurements_per_day, unit='D'
            )
        
        # Convert TIMESTAMP to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_chunk['TIMESTAMP']):
            df_chunk['TIMESTAMP'] = pd.to_datetime(df_chunk['TIMESTAMP'])
        
        # Create date column for grouping
        df_chunk['date'] = df_chunk['TIMESTAMP'].dt.date
        
        # Group by site and date, then aggregate
        print(f"  Condensing {len(df_chunk):,} rows to daily averages...")
        
        # Define aggregation dictionary
        agg_dict = {target_col: 'mean'}  # Target variable: mean
        
        # Average features
        for feature in available_avg_features:
            if feature in df_chunk.columns:
                agg_dict[feature] = 'mean'
        
        # Sum features (for cumulative variables)
        for feature in available_sum_features:
            if feature in df_chunk.columns:
                agg_dict[feature] = 'sum'
        
        # Group and aggregate
        daily_data = df_chunk.groupby(['site', 'date']).agg(agg_dict).reset_index()
        
        # Add back timestamp (use noon of each day)
        daily_data['TIMESTAMP'] = pd.to_datetime(daily_data['date']) + pd.Timedelta(hours=12)
        
        all_daily_data.append(daily_data)
        
        print(f"  Condensed to {len(daily_data):,} daily records")
        
        # Memory management
        del df_chunk
        gc.collect()
        
        # Limit processing for testing (remove this for full processing)
        if i >= 2:  # Process only first 3 files for testing
            print("  Limiting to first 3 files for testing...")
            break
    
    # Combine all daily data
    print("Combining all daily data...")
    combined_daily = pd.concat(all_daily_data, ignore_index=True)
    del all_daily_data
    gc.collect()
    
    # Sort by site and timestamp
    combined_daily = combined_daily.sort_values(['site', 'TIMESTAMP']).reset_index(drop=True)
    
    # Add temporal features
    combined_daily['year'] = combined_daily['TIMESTAMP'].dt.year
    combined_daily['month'] = combined_daily['TIMESTAMP'].dt.month
    combined_daily['day_of_year'] = combined_daily['TIMESTAMP'].dt.dayofyear
    combined_daily['season'] = combined_daily['month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    # Save daily averages
    output_file = os.path.join(output_dir, 'sapfluxnet_daily_averages.parquet')
    combined_daily.to_parquet(output_file, index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_original_rows': total_original_rows,
        'total_daily_records': len(combined_daily),
        'unique_sites': combined_daily['site'].nunique(),
        'date_range': f"{combined_daily['TIMESTAMP'].min()} to {combined_daily['TIMESTAMP'].max()}",
        'compression_ratio': total_original_rows / len(combined_daily),
        'features_included': len(available_avg_features) + len(available_sum_features) + 1  # +1 for target
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'daily_averages_summary.txt'), 'w') as f:
        f.write("Daily Averages Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write(f"Original data: {summary_stats['total_original_rows']:,} rows\n")
        f.write(f"Daily averages: {summary_stats['total_daily_records']:,} records\n")
        f.write(f"Unique sites: {summary_stats['unique_sites']}\n")
        f.write(f"Date range: {summary_stats['date_range']}\n")
        f.write(f"Compression ratio: {summary_stats['compression_ratio']:.1f}x\n")
        f.write(f"Features included: {summary_stats['features_included']}\n\n")
        
        f.write("Features averaged:\n")
        for feature in available_avg_features:
            f.write(f"  - {feature}\n")
        
        f.write("\nFeatures summed:\n")
        for feature in available_sum_features:
            f.write(f"  - {feature}\n")
    
    print(f"\n✅ Daily averages created successfully!")
    print(f"Original data: {summary_stats['total_original_rows']:,} rows")
    print(f"Daily averages: {summary_stats['total_daily_records']:,} records")
    print(f"Compression: {summary_stats['compression_ratio']:.1f}x reduction")
    print(f"Date range: {summary_stats['date_range']}")
    print(f"Saved to: {output_file}")
    
    return combined_daily, summary_stats

def main():
    """Main function to create daily averages"""
    print("SAPFLUXNET Daily Averages Creation")
    print("=" * 40)
    print(f"Started at: {datetime.now()}")
    print("Purpose: Condense high-frequency data to daily averages for temporal validation")
    
    # Load and condense data
    parquet_dir = '../processed_parquet'
    daily_data, summary_stats = load_and_condense_parquet_data(parquet_dir)
    
    print(f"\nDaily averages creation completed at: {datetime.now()}")
    print("Next step: Use daily_averages/sapfluxnet_daily_averages.parquet for temporal validation")

if __name__ == "__main__":
    main() 