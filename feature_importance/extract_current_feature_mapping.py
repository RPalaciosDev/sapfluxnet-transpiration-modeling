#!/usr/bin/env python3
"""
Extract feature mapping from current parquet_ecological files
This script adapts the original feature mapping scripts to work with the current data pipeline
"""

import pandas as pd
import os
import glob
from pathlib import Path

def get_current_feature_order():
    """Get the actual feature order from current parquet_ecological files"""
    
    print("🔍 Getting feature order from parquet_ecological files...")
    
    try:
        # Find processed parquet files in the correct directory
        parquet_dir = Path('../parquet_ecological')
        if not parquet_dir.exists():
            print(f"❌ Parquet ecological directory not found: {parquet_dir}")
            return None
        
        # Find all parquet files
        parquet_files = list(parquet_dir.glob('*.parquet'))
        if not parquet_files:
            print("❌ No parquet files found in parquet_ecological")
            return None
        
        print(f"📁 Found {len(parquet_files)} processed parquet files")
        
        # Try to find a smaller file first for faster processing
        small_files = [f for f in parquet_files if f.stat().st_size < 5 * 1024 * 1024]  # < 5MB
        if small_files:
            # Sort by file size to get the smallest
            small_files.sort(key=lambda x: x.stat().st_size)
            sample_file = small_files[0]
            print(f"🎯 Using small file: {sample_file.name} ({sample_file.stat().st_size / 1024 / 1024:.1f}MB)")
        else:
            # Use the first file if no small files found
            sample_file = parquet_files[0]
            print(f"🎯 Using file: {sample_file.name}")
        
        # Read the file to get column order
        print("📖 Reading parquet file to extract feature order...")
        df = pd.read_parquet(sample_file)
        print(f"✅ Successfully read {sample_file.name}")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Total columns: {len(df.columns)}")
        
        # Get features in the order they appear (exclude non-feature columns)
        exclude_cols = {
            'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', 
            'ecosystem_cluster', 'sap_flow', 'sap_flow_original', 'sap_flow_quality',
            'is_inside_country', 'measurement_timestep', 'measurement_frequency',
            'timezone_offset'
        }
        
        # Also exclude any columns that end with common suffixes that aren't features
        exclude_suffixes = ['_flags', '_md', '_quality', '_original']
        
        feature_cols = []
        for col in df.columns:
            if (col not in exclude_cols and 
                not any(col.endswith(suffix) for suffix in exclude_suffixes)):
                feature_cols.append(col)
        
        print(f"📈 Found {len(feature_cols)} features in order")
        
        # Show sample of features
        print(f"\n📋 Sample of features found:")
        for i, feature in enumerate(feature_cols[:10]):
            print(f"  {i}: {feature}")
        if len(feature_cols) > 10:
            print(f"  ... ({len(feature_cols) - 10} more features)")
        
        # Create feature order mapping
        feature_order = []
        for i, feature in enumerate(feature_cols):
            feature_order.append({
                'feature_index': i,  # Use numeric index instead of f0, f1...
                'feature_name': feature,
                'column_index': i
            })
        
        # Save feature order
        df_order = pd.DataFrame(feature_order)
        df_order.to_csv('current_feature_order.csv', index=False)
        print("💾 Saved feature order to: current_feature_order.csv")
        
        # Verify with another file to ensure consistency
        print("\n🔍 Verifying feature order consistency...")
        if len(parquet_files) > 1:
            # Try a different file
            other_files = [f for f in parquet_files if f != sample_file][:2]  # Check up to 2 more files
            consistent = True
            for verify_file in other_files:
                print(f"🔍 Checking consistency with: {verify_file.name}")
                try:
                    df_verify = pd.read_parquet(verify_file)
                    verify_cols = [col for col in df_verify.columns 
                                 if col not in exclude_cols and 
                                 not any(col.endswith(suffix) for suffix in exclude_suffixes)]
                    
                    if verify_cols == feature_cols:
                        print(f"  ✅ Consistent with {verify_file.name}")
                    else:
                        print(f"  ⚠️  Different feature order in {verify_file.name}")
                        print(f"     Sample: {len(feature_cols)} features")
                        print(f"     Verify: {len(verify_cols)} features")
                        consistent = False
                        break
                except Exception as e:
                    print(f"  ⚠️  Could not verify with {verify_file.name}: {e}")
            
            if consistent:
                print("✅ Feature order is consistent across checked files")
        
        return feature_cols
        
    except Exception as e:
        print(f"❌ Error in get_current_feature_order: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_enhanced_feature_descriptions():
    """Generate enhanced descriptions for current features"""
    
    feature_descriptions = {}
    
    # Environmental variables (basic features)
    env_features = {
        'ta': 'Air temperature (°C)',
        'rh': 'Relative humidity (%)',
        'vpd': 'Vapor pressure deficit (kPa)', 
        'sw_in': 'Incoming shortwave radiation (W/m²)',
        'ws': 'Wind speed (m/s)',
        'precip': 'Precipitation (mm)',
        'ppfd_in': 'Photosynthetic photon flux density (μmol/m²/s)',
        'ext_rad': 'Extraterrestrial radiation (W/m²)',
        'swc_shallow': 'Shallow soil water content (m³/m³)'
    }
    feature_descriptions.update(env_features)
    
    # Geographic and site information
    geo_features = {
        'latitude': 'Site latitude (degrees)',
        'longitude': 'Site longitude (degrees)',
        'elevation': 'Site elevation (m above sea level)',
        'latitude_abs': 'Absolute latitude (distance from equator)',
        'mean_annual_temp': 'Mean annual temperature (°C)',
        'mean_annual_precip': 'Mean annual precipitation (mm)',
        'aridity_index': 'Aridity index (precipitation/potential ET)',
        'seasonal_temp_range': 'Seasonal temperature range (°C)',
        'seasonal_precip_range': 'Seasonal precipitation range (mm)'
    }
    feature_descriptions.update(geo_features)
    
    # Climate and ecological classification
    climate_features = {
        'koppen_geiger_code_encoded': 'Köppen-Geiger climate classification (encoded)',
        'biome_code': 'Biome classification code',
        'igbp_class_code': 'IGBP land cover class code',
        'climate_zone_code': 'Climate zone classification code'
    }
    feature_descriptions.update(climate_features)
    
    # Vegetation and forest structure
    veg_features = {
        'stand_age': 'Forest stand age (years)',
        'stand_height': 'Forest stand height (m)',
        'n_trees': 'Number of trees per unit area',
        'basal_area': 'Basal area (m²/ha)',
        'tree_density': 'Tree density (trees/ha)',
        'leaf_area_index': 'Leaf area index',
        'tree_volume_index': 'Tree volume index',
        'pl_age': 'Plant age (years)',
        'pl_dbh': 'Plant diameter at breast height (cm)',
        'pl_height': 'Plant height (m)',
        'pl_leaf_area': 'Plant leaf area (cm²)',
        'pl_bark_thick': 'Plant bark thickness (mm)',
        'pl_sapw_area': 'Plant sapwood area (cm²)',
        'pl_sapw_depth': 'Plant sapwood depth (cm)',
        'sapwood_leaf_ratio': 'Sapwood area to leaf area ratio'
    }
    feature_descriptions.update(veg_features)
    
    # Functional and taxonomic classification
    func_features = {
        'leaf_habit_code': 'Leaf habit classification (deciduous/evergreen)',
        'species_functional_group_code': 'Plant functional group classification',
        'pl_social_code': 'Plant social status code',
        'tree_size_class_code': 'Tree size class code',
        'tree_age_class_code': 'Tree age class code',
        'growth_condition_code': 'Growth condition classification'
    }
    feature_descriptions.update(func_features)
    
    # Soil properties
    soil_features = {
        'soil_depth': 'Soil depth (cm)',
        'soil_texture_code': 'Soil texture classification',
        'clay_percentage': 'Clay content (%)',
        'sand_percentage': 'Sand content (%)',
        'silt_percentage': 'Silt content (%)'
    }
    feature_descriptions.update(soil_features)
    
    # Topographic features
    topo_features = {
        'terrain_code': 'Terrain classification code'
    }
    feature_descriptions.update(topo_features)
    
    # Temporal features
    temporal_features = {
        'solar_hour': 'Solar hour of day',
        'solar_day_of_year': 'Solar day of year',
        'solar_hour_sin': 'Sine of solar hour (cyclical)',
        'solar_hour_cos': 'Cosine of solar hour (cyclical)',
        'solar_day_sin': 'Sine of solar day (cyclical)',
        'solar_day_cos': 'Cosine of solar day (cyclical)',
        'month_sin': 'Sine of month (cyclical)',
        'month_cos': 'Cosine of month (cyclical)',
        'is_morning': 'Morning time indicator (6-12h)',
        'is_afternoon': 'Afternoon time indicator (12-18h)',
        'is_night': 'Night time indicator',
        'is_spring': 'Spring season indicator',
        'is_summer': 'Summer season indicator',
        'is_autumn': 'Autumn season indicator',
        'is_winter': 'Winter season indicator',
        'hours_since_sunrise': 'Hours elapsed since sunrise',
        'hours_since_sunset': 'Hours elapsed since sunset'
    }
    feature_descriptions.update(temporal_features)
    
    # Interaction features
    interaction_features = {
        'vpd_ppfd_interaction': 'VPD × PPFD interaction term',
        'vpd_ta_interaction': 'VPD × Temperature interaction term',
        'temp_humidity_ratio': 'Temperature to humidity ratio',
        'water_stress_index': 'Combined water stress indicator',
        'light_efficiency': 'Light use efficiency proxy',
        'temp_soil_interaction': 'Temperature × soil moisture interaction',
        'wind_vpd_interaction': 'Wind speed × VPD interaction',
        'radiation_temp_interaction': 'Radiation × temperature interaction',
        'humidity_soil_interaction': 'Humidity × soil moisture interaction'
    }
    feature_descriptions.update(interaction_features)
    
    return feature_descriptions

def categorize_current_feature(feature_name):
    """Categorize a feature based on its name (updated for current data)"""
    
    # Environmental variables
    if feature_name in ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'ppfd_in', 'ext_rad', 'swc_shallow']:
        return 'Environmental'
    
    # Geographic features
    if feature_name in ['latitude', 'longitude', 'elevation', 'latitude_abs']:
        return 'Geographic'
    
    # Climate features
    if any(x in feature_name for x in ['mean_annual', 'aridity', 'seasonal', 'koppen', 'climate']):
        return 'Climate'
    
    # Ecological classification
    if any(x in feature_name for x in ['biome', 'igbp', 'leaf_habit', 'species_functional', 'growth_condition']):
        return 'Ecological'
    
    # Vegetation structure
    if any(x in feature_name for x in ['stand_', 'tree_', 'basal_', 'leaf_area', 'pl_', 'sapwood']):
        return 'Vegetation'
    
    # Soil features
    if any(x in feature_name for x in ['soil', 'clay', 'sand', 'silt']):
        return 'Soil'
    
    # Topographic features
    if 'terrain' in feature_name:
        return 'Topographic'
    
    # Temporal features
    if any(x in feature_name for x in ['solar_', 'month_', 'is_', 'hours_since']):
        return 'Temporal'
    
    # Interaction features
    if any(x in feature_name for x in ['_interaction', '_ratio', '_index', '_efficiency']):
        return 'Interaction'
    
    # Encoded features
    if '_code' in feature_name or '_encoded' in feature_name:
        return 'Encoded'
    
    return 'Other'

def create_current_feature_mapping():
    """Create comprehensive feature mapping for current data pipeline"""
    
    print("🔧 Creating feature mapping for current data pipeline...")
    
    # Get feature order
    feature_cols = get_current_feature_order()
    if not feature_cols:
        return None
    
    # Get descriptions
    descriptions = get_enhanced_feature_descriptions()
    
    # Create final mapping
    final_mapping = []
    
    for i, feature_name in enumerate(feature_cols):
        # Get description
        description = descriptions.get(feature_name, feature_name.replace('_', ' ').title())
        
        # Categorize feature
        category = categorize_current_feature(feature_name)
        
        final_mapping.append({
            'feature_index': i,
            'feature_name': feature_name,
            'description': description,
            'category': category
        })
    
    # Create DataFrame
    df_final = pd.DataFrame(final_mapping)
    
    # Save to multiple locations for compatibility
    output_files = [
        'feature_mapping_current.csv',
        'feature_mapping_v2_final.csv',  # For compatibility with existing scripts
        '../xgboost_scripts/feature_mapping_v2_complete.csv'  # For model training scripts
    ]
    
    for output_file in output_files:
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
            df_final.to_csv(output_file, index=False)
            print(f"💾 Saved mapping to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save to {output_file}: {e}")
    
    # Print summary
    print(f"\n📊 Feature mapping created with {len(df_final)} features")
    
    # Print summary by category
    category_counts = df_final['category'].value_counts()
    print("\n📊 Feature categories:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} features")
    
    # Print first 15 features
    print("\n📋 First 15 features in mapping:")
    for i, row in df_final.head(15).iterrows():
        print(f"  {row['feature_index']}: {row['feature_name']} - {row['description']}")
    
    return df_final

if __name__ == "__main__":
    print("🚀 Extracting feature mapping from current parquet_ecological files...")
    
    feature_mapping = create_current_feature_mapping()
    
    if feature_mapping is not None:
        print(f"\n✅ Successfully created feature mapping with {len(feature_mapping)} features")
        print("🎯 This mapping is now compatible with:")
        print("  - Clustering feature importance visualization")
        print("  - Model training feature importance mapping")
        print("  - All existing analysis scripts")
        print("\n💡 Next steps:")
        print("  1. Re-run your model training to get properly mapped feature importance")
        print("  2. Clustering visualizations will now show meaningful feature names")
    else:
        print("❌ Failed to create feature mapping")