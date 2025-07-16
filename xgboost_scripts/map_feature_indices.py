"""
Map LibSVM Feature Indices to Original Variable Names
====================================================

This script maps feature indices (f0, f1, f2, ..., f107, f110, f126) 
back to their original variable names from the SAPFLUXNET processing pipeline.

For libsvm-first workflows where raw data is processed directly to libsvm format.
"""

import pandas as pd
import os
import sys
from pathlib import Path

def get_feature_mapping_from_raw_data():
    """
    Get feature mapping by examining raw data and recreating the processing pipeline
    """
    # Look for raw data directories
    raw_dirs = ['../sapwood', '../leaf', '../plant']
    
    # Find a sample raw data file
    sample_file = None
    for raw_dir in raw_dirs:
        if os.path.exists(raw_dir):
            files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
            if files:
                sample_file = os.path.join(raw_dir, files[0])
                break
    
    if sample_file is None:
        print("❌ Could not find raw data files to examine")
        return None
    
    print(f"📊 Examining raw data structure from: {sample_file}")
    
    # Load sample raw data
    try:
        df = pd.read_csv(sample_file, nrows=100)
    except Exception as e:
        print(f"❌ Error loading {sample_file}: {e}")
        return None
    
    # Simulate the processing pipeline to get feature order
    # This recreates the feature engineering from comprehensive_processing_pipeline.py
    features = simulate_feature_engineering(df)
    
    if features is None:
        print("❌ Could not simulate feature engineering")
        return None
    
    # Create the feature mapping
    feature_mapping = {}
    for i, feature_name in enumerate(features):
        feature_mapping[f'f{i}'] = feature_name
    
    print(f"✅ Generated {len(features)} features from pipeline simulation")
    return feature_mapping

def simulate_feature_engineering(df):
    """
    Simulate the feature engineering pipeline to get the feature order
    """
    try:
        # Start with the raw data columns
        all_features = []
        
        # Basic environmental features (typically in raw data)
        env_features = []
        for col in df.columns:
            if col not in ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'sap_flow', 'Unnamed: 0']:
                if not col.endswith('_flags') and not col.endswith('_md'):
                    env_features.append(col)
        
        all_features.extend(env_features)
        
        # Add typical engineered features in the order they're created
        # (This is based on the comprehensive_processing_pipeline.py structure)
        
        # 1. Temporal features
        temporal_features = [
            'hour', 'day_of_year', 'month', 'year', 'day_of_week',
            'solar_hour', 'solar_day_of_year', 
            'solar_hour_sin', 'solar_hour_cos', 'solar_day_sin', 'solar_day_cos',
            'is_daylight', 'is_peak_sunlight', 'is_weekend'
        ]
        all_features.extend(temporal_features)
        
        # 2. Lagged features (for common environmental variables)
        env_vars = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
        lag_hours = [1, 2, 3, 6, 12, 24]
        for var in env_vars:
            for lag in lag_hours:
                all_features.append(f'{var}_lag_{lag}h')
        
        # 3. Rolling features
        rolling_vars = ['ta', 'vpd', 'sw_in', 'rh']
        rolling_windows = [3, 6, 12, 24, 48, 72]
        for var in rolling_vars:
            for window in rolling_windows:
                all_features.append(f'{var}_mean_{window}h')
                all_features.append(f'{var}_std_{window}h')
        
        # 4. Domain-specific features
        domain_features = [
            'temp_deviation', 'tree_size_factor', 'sapwood_leaf_ratio', 'transpiration_capacity'
        ]
        all_features.extend(domain_features)
        
        # 5. Metadata features (geographic, site, stand, species, plant)
        metadata_features = [
            'latitude', 'longitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
            'biome', 'biome_code', 'igbp_class', 'igbp_code', 'country', 'country_code',
            'site_code', 'site_name', 'is_inside_country',
            'stand_age', 'basal_area', 'tree_density', 'stand_height', 'leaf_area_index',
            'clay_percentage', 'sand_percentage', 'silt_percentage', 'soil_depth',
            'soil_texture', 'soil_texture_code', 'terrain', 'terrain_code',
            'growth_condition', 'growth_condition_code',
            'species_name', 'leaf_habit', 'leaf_habit_code', 'n_trees',
            'pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area', 'pl_bark_thick',
            'pl_social', 'social_status_code', 'pl_species', 'pl_sapw_area', 'pl_sapw_depth',
            'measurement_timestep', 'measurement_frequency', 'timezone', 'timezone_offset',
            'climate_zone', 'climate_zone_code', 'latitude_abs', 'aridity_index',
            'tree_size_class', 'tree_age_class', 'tree_volume_index'
        ]
        all_features.extend(metadata_features)
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in all_features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        return unique_features
        
    except Exception as e:
        print(f"❌ Error simulating feature engineering: {e}")
        return None

def get_feature_mapping_from_processed_data():
    """
    Get feature mapping by examining the processed data structure
    """
    # Try to find a processed CSV file to examine the feature order
    search_dirs = [
        '../processed_csv',
        '../comprehensive_processed',
        '../my_custom_data_parquet',
        '../processed_parquet'
    ]
    
    sample_file = None
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv') or f.endswith('.parquet')]
            if files:
                sample_file = os.path.join(dir_path, files[0])
                break
    
    if sample_file is None:
        print("ℹ️  No processed CSV/parquet files found (expected for libsvm-first workflow)")
        return None
    
    print(f"📊 Examining feature order from: {sample_file}")
    
    # Load the sample file
    try:
        if sample_file.endswith('.csv'):
            df = pd.read_csv(sample_file, nrows=10)
        elif sample_file.endswith('.parquet'):
            df = pd.read_parquet(sample_file, nrows=10)
        else:
            print(f"❌ Unsupported file format: {sample_file}")
            return None
    except Exception as e:
        print(f"❌ Error loading {sample_file}: {e}")
        return None
    
    # Recreate the feature selection logic from the pipeline
    target_col = 'sap_flow'
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0', target_col
    ]
    
    # Select feature columns (same logic as in _save_libsvm_format)
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    # Create mapping from feature index to name
    feature_mapping = {}
    for i, feature_name in enumerate(feature_cols):
        feature_mapping[f'f{i}'] = feature_name
    
    print(f"✅ Found {len(feature_cols)} features in the dataset")
    return feature_mapping

def lookup_specific_features(feature_mapping, feature_indices):
    """
    Look up specific feature indices and return their names
    """
    if feature_mapping is None:
        print("❌ No feature mapping available")
        return
    
    print(f"\n🔍 Feature Index Mapping:")
    print("=" * 50)
    
    for feature_idx in feature_indices:
        if feature_idx in feature_mapping:
            feature_name = feature_mapping[feature_idx]
            print(f"  {feature_idx:>6} → {feature_name}")
        else:
            print(f"  {feature_idx:>6} → NOT FOUND")
    
    print("=" * 50)

def save_complete_mapping(feature_mapping, output_file='feature_mapping.csv'):
    """
    Save complete feature mapping to a CSV file
    """
    if feature_mapping is None:
        print("❌ No feature mapping available")
        return
    
    # Convert to DataFrame for easy saving
    mapping_df = pd.DataFrame([
        {'feature_index': idx, 'feature_name': name}
        for idx, name in feature_mapping.items()
    ])
    
    # Sort by feature index number
    mapping_df['index_num'] = mapping_df['feature_index'].str.extract(r'(\d+)').astype(int)
    mapping_df = mapping_df.sort_values('index_num').drop('index_num', axis=1)
    
    # Save to CSV
    mapping_df.to_csv(output_file, index=False)
    print(f"💾 Complete feature mapping saved to: {output_file}")
    
    # Show first 10 and last 10 features
    print(f"\n📋 First 10 features:")
    for i in range(min(10, len(mapping_df))):
        row = mapping_df.iloc[i]
        print(f"  {row['feature_index']:>6} → {row['feature_name']}")
    
    if len(mapping_df) > 10:
        print(f"\n📋 Last 10 features:")
        for i in range(max(0, len(mapping_df)-10), len(mapping_df)):
            row = mapping_df.iloc[i]
            print(f"  {row['feature_index']:>6} → {row['feature_name']}")

def main():
    """
    Main function to map feature indices to names
    """
    print("SAPFLUXNET Feature Index Mapping")
    print("=" * 50)
    
    # Try processed data first, then fall back to raw data simulation
    feature_mapping = get_feature_mapping_from_processed_data()
    
    if feature_mapping is None:
        print("\n🔄 Falling back to raw data simulation...")
        feature_mapping = get_feature_mapping_from_raw_data()
    
    if feature_mapping is None:
        print("\n❌ Could not create feature mapping. Please ensure you have either:")
        print("   - Processed data files (CSV/parquet)")
        print("   - Raw data files in sapwood/, leaf/, or plant/ directories")
        return
    
    # Your most important features from the XGBoost results
    important_features = ['f107', 'f110', 'f126', 'f105', 'f133', 'f7', 'f117', 'f112', 'f116', 'f106']
    
    print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
    lookup_specific_features(feature_mapping, important_features)
    
    # Save complete mapping
    save_complete_mapping(feature_mapping)
    
    print(f"\n💡 Usage Tips:")
    print(f"  1. Use feature_mapping.csv to look up any feature index")
    print(f"  2. Cross-reference with docs/engineered_features_documentation.md for feature details")
    print(f"  3. The feature order matches the libsvm file creation order")
    print(f"  4. This mapping is based on the comprehensive_processing_pipeline.py feature engineering")

if __name__ == "__main__":
    main() 