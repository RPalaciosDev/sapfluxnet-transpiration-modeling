#!/usr/bin/env python3
"""
Map temporal validation feature importance to actual feature names.
Uses the existing feature_mapping.csv to convert feature indices to names.
"""

import pandas as pd
import os
from datetime import datetime

def map_feature_importance():
    """Map feature importance results to actual feature names."""
    
    # Find the most recent temporal validation results
    results_dir = "external_memory_models/temporal_validation_chronological"
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        return
    
    # Find the most recent importance file
    importance_files = [f for f in os.listdir(results_dir) if '_importance_' in f and f.endswith('.csv')]
    if not importance_files:
        print("No importance files found!")
        return
    
    # Get the most recent one (they have timestamps)
    latest_file = sorted(importance_files)[-1]
    importance_file = os.path.join(results_dir, latest_file)
    
    print(f"Processing: {importance_file}")
    
    # Load the feature importance results
    importance_df = pd.read_csv(importance_file)
    
    # Load the feature mapping
    mapping_df = pd.read_csv("feature_mapping.csv")
    
    # Create a mapping dictionary
    feature_map = dict(zip(mapping_df['feature_index'], mapping_df['feature_name']))
    
    # Map the features
    importance_df['feature_name'] = importance_df['feature'].map(feature_map)
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Display top features
    print("\n" + "="*80)
    print("TOP FEATURES FROM TEMPORAL VALIDATION")
    print("="*80)
    
    print(f"\nTop 20 Most Important Features:")
    print("-" * 60)
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['feature']:>6} | {row['feature_name']:<30} | {row['importance']:>12.2f}")
    
    print(f"\nBottom 20 Least Important Features:")
    print("-" * 60)
    for i, row in importance_df.tail(20).iterrows():
        print(f"{row['feature']:>6} | {row['feature_name']:<30} | {row['importance']:>12.2f}")
    
    # Feature categories analysis
    print(f"\n" + "="*80)
    print("FEATURE CATEGORY ANALYSIS")
    print("="*80)
    
    # Group features by category
    categories = {
        'Meteorological': ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'ppfd_in', 'ext_rad'],
        'Temporal': ['hour', 'day_of_year', 'month', 'year', 'day_of_week', 'solar_hour', 'solar_day_of_year'],
        'Cyclical': ['solar_hour_sin', 'solar_hour_cos', 'solar_day_sin', 'solar_day_cos'],
        'Binary': ['is_daylight', 'is_peak_sunlight', 'is_weekend'],
        'Soil': ['swc_shallow'],
        'Lagged': ['lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h'],
        'Rolling_Stats': ['mean_3h', 'mean_6h', 'mean_12h', 'mean_24h', 'mean_48h', 'mean_72h', 
                         'std_3h', 'std_6h', 'std_12h', 'std_24h', 'std_48h', 'std_72h'],
        'Site_Characteristics': ['latitude', 'longitude', 'elevation', 'biome', 'country', 'site_code'],
        'Climate': ['mean_annual_temp', 'mean_annual_precip', 'climate_zone'],
        'Vegetation': ['basal_area', 'tree_density', 'stand_height', 'leaf_area_index', 'stand_age'],
        'Soil_Properties': ['clay_percentage', 'sand_percentage', 'silt_percentage', 'soil_depth', 'soil_texture'],
        'Tree_Properties': ['tree_size_factor', 'sapwood_leaf_ratio', 'transpiration_capacity', 'pl_age', 'pl_dbh'],
        'Derived': ['temp_deviation', 'aridity_index', 'tree_size_class', 'tree_age_class', 'tree_volume_index']
    }
    
    # Calculate category importance
    category_importance = {}
    for category, keywords in categories.items():
        category_features = importance_df[importance_df['feature_name'].str.contains('|'.join(keywords), case=False, na=False)]
        if not category_features.empty:
            total_importance = category_features['importance'].sum()
            avg_importance = category_features['importance'].mean()
            category_importance[category] = {
                'total_importance': total_importance,
                'avg_importance': avg_importance,
                'feature_count': len(category_features),
                'top_features': category_features.head(3)[['feature_name', 'importance']].to_dict('records')
            }
    
    # Sort categories by total importance
    sorted_categories = sorted(category_importance.items(), 
                             key=lambda x: x[1]['total_importance'], reverse=True)
    
    print(f"\nFeature Categories by Total Importance:")
    print("-" * 80)
    for category, stats in sorted_categories:
        print(f"\n{category}:")
        print(f"  Total Importance: {stats['total_importance']:>12.2f}")
        print(f"  Average Importance: {stats['avg_importance']:>12.2f}")
        print(f"  Feature Count: {stats['feature_count']:>3}")
        print(f"  Top Features:")
        for feat in stats['top_features']:
            print(f"    {feat['feature_name']:<25} | {feat['importance']:>10.2f}")
    
    # Save mapped results
    output_file = importance_file.replace('_importance_', '_importance_mapped_')
    importance_df.to_csv(output_file, index=False)
    print(f"\nMapped results saved to: {output_file}")
    
    return importance_df

if __name__ == "__main__":
    mapped_results = map_feature_importance() 