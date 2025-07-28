#!/usr/bin/env python3
"""
Map feature importance using the v2 feature mapping
This script shows how to use the new feature mapping to interpret results
"""

import pandas as pd
import os

def map_feature_importance(feature_importance_file, mapping_file='../xgboost_scripts/feature_mapping_v2_complete.csv'):
    """Map feature importance results using the v2 feature mapping"""
    
    print("🔍 Mapping feature importance using v2 feature mapping...")
    
    # Load the feature mapping
    try:
        df_mapping = pd.read_csv(mapping_file)
        mapping_dict = dict(zip(df_mapping['feature_index'], df_mapping['feature_name']))
        desc_dict = dict(zip(df_mapping['feature_index'], df_mapping['description']))
        cat_dict = dict(zip(df_mapping['feature_index'], df_mapping['category']))
        print(f"📖 Loaded mapping for {len(mapping_dict)} features")
    except FileNotFoundError:
        print(f"❌ Feature mapping file not found: {mapping_file}")
        return None
    
    # Load feature importance results
    try:
        df_importance = pd.read_csv(feature_importance_file)
        print(f"📊 Loaded feature importance for {len(df_importance)} features")
    except FileNotFoundError:
        print(f"❌ Feature importance file not found: {feature_importance_file}")
        return None
    
    # Map feature names
    mapped_results = []
    
    for _, row in df_importance.iterrows():
        feature_index = row['feature_index'] if 'feature_index' in row else row.iloc[0]
        importance_score = row['importance'] if 'importance' in row else row.iloc[1]
        
        # Get mapped information
        feature_name = mapping_dict.get(feature_index, f"Unknown feature: {feature_index}")
        description = desc_dict.get(feature_index, "No description available")
        category = cat_dict.get(feature_index, "Unknown category")
        
        mapped_results.append({
            'feature_index': feature_index,
            'feature_name': feature_name,
            'description': description,
            'category': category,
            'importance_score': importance_score
        })
    
    # Create DataFrame
    df_mapped = pd.DataFrame(mapped_results)
    
    # Sort by importance
    df_mapped = df_mapped.sort_values('importance_score', ascending=False)
    
    # Save mapped results
    output_file = feature_importance_file.replace('.csv', '_mapped.csv')
    df_mapped.to_csv(output_file, index=False)
    
    print(f"💾 Saved mapped results to: {output_file}")
    
    # Print top 20 features
    print("\n🏆 Top 20 most important features:")
    for i, row in df_mapped.head(20).iterrows():
        print(f"  {row['feature_index']}: {row['feature_name']} ({row['category']})")
        print(f"    Description: {row['description']}")
        print(f"    Importance: {row['importance_score']:.2f}")
        print()
    
    # Print summary by category
    category_importance = df_mapped.groupby('category')['importance_score'].sum().sort_values(ascending=False)
    print("📊 Total importance by category:")
    for category, total_importance in category_importance.items():
        print(f"  {category}: {total_importance:.2f}")
    
    return df_mapped

def demo_mapping():
    """Demonstrate the mapping with a sample feature importance file"""
    
    print("🎯 Demo: Mapping feature importance")
    
    # Check if we have any feature importance files
    importance_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'feature_importance' in file and file.endswith('.csv'):
                importance_files.append(os.path.join(root, file))
    
    if importance_files:
        print(f"📁 Found {len(importance_files)} feature importance files:")
        for i, file in enumerate(importance_files[:5]):  # Show first 5
            print(f"  {i+1}. {file}")
        
        # Use the first one as demo
        demo_file = importance_files[0]
        print(f"\n🎯 Using {demo_file} for demo mapping...")
        
        return map_feature_importance(demo_file)
    
    else:
        print("❌ No feature importance files found")
        print("💡 To test the mapping, run a model and save feature importance results")
        return None

if __name__ == "__main__":
    # Try to map existing feature importance files
    result = demo_mapping()
    
    if result is not None:
        print("\n✅ Feature importance mapping completed!")
        print("🎯 You can now use this mapping to interpret any feature importance results")
    else:
        print("\n💡 To use this mapping with your results:")
        print("  1. Run your model and save feature importance")
        print("  2. Call: map_feature_importance('your_importance_file.csv')")
        print("  3. The script will create a mapped version with descriptions") 