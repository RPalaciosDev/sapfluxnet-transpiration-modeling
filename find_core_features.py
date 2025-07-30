#!/usr/bin/env python3
"""
Find features in our processed data that most closely match core transpiration drivers
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_sample_site():
    """Analyze a sample site to find relevant features"""
    
    # Load a sample site to see available features
    parquet_dir = Path('processed_parquet')
    sample_files = list(parquet_dir.glob('*_comprehensive.parquet'))
    
    if not sample_files:
        print("❌ No parquet files found in processed_parquet/")
        return None
    
    sample_file = sample_files[0]
    site_name = sample_file.stem.replace('_comprehensive', '')
    
    print(f"🔍 Analyzing sample site: {site_name}")
    
    # Load data
    df = pd.read_parquet(sample_file)
    
    print(f"📊 Total features: {len(df.columns)}")
    print(f"📊 Total rows: {len(df)}")
    
    return df

def find_core_features(df):
    """Map target physiological variables to available features"""
    
    core_mappings = {
        'Maximum Temperature': [],
        'Mean Temperature': [],
        'Mean VPD': [],
        'Incoming Solar Radiation': [],
        'Potential Evapotranspiration': [],
        'Soil Moisture': []
    }
    
    # Search patterns for each core variable
    search_patterns = {
        'Maximum Temperature': ['ta_max', 'temp_max', 'temperature_max', 'ta'],
        'Mean Temperature': ['ta_mean', 'ta', 'temp_mean', 'temperature'],
        'Mean VPD': ['vpd', 'vapor_pressure_deficit'],
        'Incoming Solar Radiation': ['sw_in', 'solar', 'radiation', 'ppfd_in', 'sw_'],
        'Potential Evapotranspiration': ['pet', 'evapotranspiration', 'et'],
        'Soil Moisture': ['swc', 'soil_water', 'moisture', 'swc_shallow']
    }
    
    print(f"\n🎯 MAPPING CORE TRANSPIRATION DRIVERS")
    print(f"=" * 60)
    
    for core_var, patterns in search_patterns.items():
        matches = []
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    # Get some basic stats
                    non_null_count = df[col].count()
                    mean_val = df[col].mean() if df[col].dtype in ['float64', 'int64'] else 'N/A'
                    
                    matches.append({
                        'feature': col,
                        'non_null_count': non_null_count,
                        'mean': mean_val,
                        'dtype': str(df[col].dtype)
                    })
                    break
        
        core_mappings[core_var] = matches
        
        print(f"\n📋 {core_var}:")
        if matches:
            for match in matches:
                print(f"  ✅ {match['feature']}")
                mean_str = f"{match['mean']:.3f}" if isinstance(match['mean'], (int, float)) else str(match['mean'])
                print(f"     Non-null: {match['non_null_count']:,} | Mean: {mean_str} | Type: {match['dtype']}")
        else:
            print(f"  ❌ No matches found")
    
    return core_mappings

def recommend_features(core_mappings):
    """Recommend the best feature for each core variable"""
    
    recommendations = {}
    
    print(f"\n🏆 RECOMMENDED CORE FEATURES")
    print(f"=" * 60)
    
    for core_var, matches in core_mappings.items():
        if not matches:
            print(f"\n❌ {core_var}: No suitable features found")
            continue
        
        # Simple heuristic: prefer features with most data
        best_match = max(matches, key=lambda x: x['non_null_count'])
        recommendations[core_var] = best_match['feature']
        
        print(f"\n✅ {core_var}:")
        print(f"   Recommended: {best_match['feature']}")
        print(f"   Data coverage: {best_match['non_null_count']:,} rows")
        
        if len(matches) > 1:
            print(f"   Alternatives: {', '.join([m['feature'] for m in matches if m['feature'] != best_match['feature']])}")
    
    return recommendations

def save_feature_mapping(recommendations):
    """Save the recommended feature mapping"""
    
    # Create a simple mapping file
    mapping_content = f"""# Core Transpiration Features Mapping
# Generated automatically from processed data analysis

CORE_FEATURES = {{
"""
    
    for core_var, feature in recommendations.items():
        mapping_content += f'    "{core_var}": "{feature}",\n'
    
    mapping_content += "}\n\n"
    
    # Add feature list for easy copying
    mapping_content += "# Feature list for minimal pipeline:\n"
    mapping_content += "MINIMAL_FEATURES = [\n"
    for feature in recommendations.values():
        mapping_content += f'    "{feature}",\n'
    mapping_content += "]\n"
    
    with open('core_feature_mapping.py', 'w') as f:
        f.write(mapping_content)
    
    print(f"\n💾 Feature mapping saved to: core_feature_mapping.py")

def main():
    """Main function"""
    print(f"🚀 CORE TRANSPIRATION FEATURES FINDER")
    print(f"=" * 60)
    print(f"Target: Find features matching core physiological drivers")
    
    # Analyze sample site
    df = analyze_sample_site()
    if df is None:
        return
    
    # Find matching features
    core_mappings = find_core_features(df)
    
    # Get recommendations
    recommendations = recommend_features(core_mappings)
    
    # Save mapping
    save_feature_mapping(recommendations)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📋 Found {len(recommendations)} core features")
    print(f"📁 Ready to create minimal pipeline")

if __name__ == "__main__":
    main()