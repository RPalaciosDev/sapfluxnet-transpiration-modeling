"""
Map LibSVM Feature Indices to Original Variable Names
====================================================

This script maps feature indices (f0, f1, f2, ..., f107, f110, f126) 
back to their original variable names from the SAPFLUXNET processing pipeline.
"""

import pandas as pd
import os
import sys
from pathlib import Path

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
        print("❌ Could not find any processed data files to examine feature order")
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
    mapping_df['index_num'] = mapping_df['feature_index'].str.extract('(\d+)').astype(int)
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
    
    # Get feature mapping from processed data
    feature_mapping = get_feature_mapping_from_processed_data()
    
    if feature_mapping is None:
        print("\n❌ Could not create feature mapping. Please ensure you have processed data files available.")
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

if __name__ == "__main__":
    main() 