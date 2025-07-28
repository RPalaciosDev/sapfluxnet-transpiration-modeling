#!/usr/bin/env python3
"""
Create final feature mapping by combining descriptions and order
This script creates the complete feature mapping for v2 pipeline
"""

import pandas as pd

def create_final_feature_mapping():
    """Create the final feature mapping by combining descriptions and order"""
    
    print("🔧 Creating final feature mapping...")
    
    # Load feature descriptions
    try:
        df_descriptions = pd.read_csv('v2_feature_descriptions.csv')
        descriptions_dict = dict(zip(df_descriptions['feature_name'], df_descriptions['description']))
        print(f"📖 Loaded {len(descriptions_dict)} feature descriptions")
    except FileNotFoundError:
        print("❌ Feature descriptions file not found. Run extract_v2_feature_descriptions.py first.")
        return None
    
    # Load feature order
    try:
        df_order = pd.read_csv('v2_feature_order.csv')
        print(f"📋 Loaded {len(df_order)} features in order")
    except FileNotFoundError:
        print("❌ Feature order file not found. Run get_v2_feature_order.py first.")
        return None
    
    # Create final mapping
    final_mapping = []
    
    for _, row in df_order.iterrows():
        feature_name = row['feature_name']
        feature_index = row['feature_index']
        
        # Get description
        description = descriptions_dict.get(feature_name, feature_name.replace('_', ' ').title())
        
        # Categorize feature
        category = categorize_feature(feature_name)
        
        final_mapping.append({
            'feature_index': feature_index,
            'feature_name': feature_name,
            'description': description,
            'category': category
        })
    
    # Create DataFrame
    df_final = pd.DataFrame(final_mapping)
    
    # Save to multiple formats
    df_final.to_csv('../xgboost_scripts/feature_mapping_v2_complete.csv', index=False)
    df_final.to_csv('feature_mapping_v2_final.csv', index=False)
    
    print(f"💾 Saved final mapping with {len(df_final)} features")
    print("📁 Files saved:")
    print("  - ../xgboost_scripts/feature_mapping_v2_complete.csv")
    print("  - feature_mapping_v2_final.csv")
    
    # Print summary by category
    category_counts = df_final['category'].value_counts()
    print("\n📊 Feature categories:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} features")
    
    # Print first 20 features
    print("\n📋 First 20 features in final mapping:")
    for i, row in df_final.head(20).iterrows():
        print(f"  {row['feature_index']}: {row['feature_name']} - {row['description']}")
    
    return df_final

def categorize_feature(feature_name):
    """Categorize a feature based on its name"""
    
    # Environmental variables
    if feature_name in ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'ppfd_in', 'ext_rad', 'swc_shallow']:
        return 'Environmental'
    
    # Temporal features
    if any(x in feature_name for x in ['hour', 'day', 'month', 'year', 'week', 'solar', 'season', 'is_']):
        return 'Temporal'
    
    # Lagged features
    if '_lag_' in feature_name:
        return 'Lagged'
    
    # Rolling window features
    if any(x in feature_name for x in ['_mean_', '_std_', '_min_', '_max_', '_range_']):
        return 'Rolling'
    
    # Rate of change features
    if '_rate_' in feature_name:
        return 'Rate of Change'
    
    # Cumulative features
    if '_cum_' in feature_name:
        return 'Cumulative'
    
    # Interaction features
    if '_interaction' in feature_name or '_ratio' in feature_name or '_index' in feature_name:
        return 'Interaction'
    
    # Seasonality features
    if 'seasonal_' in feature_name:
        return 'Seasonality'
    
    # Domain-specific features
    if feature_name in ['temp_deviation', 'tree_size_factor', 'sapwood_leaf_ratio', 'transpiration_capacity']:
        return 'Domain'
    
    # Geographic features
    if feature_name in ['latitude', 'longitude', 'elevation', 'country', 'timezone']:
        return 'Geographic'
    
    # Climate features
    if any(x in feature_name for x in ['climate', 'aridity', 'koppen']):
        return 'Climate'
    
    # Ecological features
    if any(x in feature_name for x in ['biome', 'igbp', 'species', 'leaf_habit']):
        return 'Ecological'
    
    # Structural features
    if any(x in feature_name for x in ['stand_', 'tree_', 'basal_', 'leaf_area', 'pl_']):
        return 'Structural'
    
    # Soil features
    if any(x in feature_name for x in ['soil', 'clay', 'sand', 'silt']):
        return 'Soil'
    
    # Metadata features
    if any(x in feature_name for x in ['measurement_', 'frequency', 'timestep']):
        return 'Metadata'
    
    # Identity features
    if any(x in feature_name for x in ['site_', 'code']):
        return 'Identity'
    
    # Encoded features
    if '_encoded' in feature_name or '_code' in feature_name:
        return 'Encoded'
    
    return 'Other'

if __name__ == "__main__":
    final_mapping = create_final_feature_mapping()
    
    if final_mapping is not None:
        print(f"\n✅ Successfully created final feature mapping with {len(final_mapping)} features")
        print("🎯 This mapping can now be used to interpret feature importance from your models!")
    else:
        print("❌ Failed to create final feature mapping") 