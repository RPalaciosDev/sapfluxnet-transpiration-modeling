#!/usr/bin/env python3
"""
Get actual feature order from v2 pipeline processed files
This script reads processed parquet files to get the exact feature order
"""

import pandas as pd
import os
import glob
from pathlib import Path

def get_v2_feature_order():
    """Get the actual feature order from v2 pipeline processed files"""
    
    print("🔍 Getting feature order from processed parquet files...")
    
    try:
        # Find processed parquet files
        parquet_dir = Path('../processed_parquet')
        if not parquet_dir.exists():
            print(f"❌ Processed parquet directory not found: {parquet_dir}")
            return None
        
        # Find all comprehensive parquet files
        parquet_files = list(parquet_dir.glob('*_comprehensive.parquet'))
        if not parquet_files:
            print("❌ No comprehensive parquet files found")
            return None
        
        print(f"📁 Found {len(parquet_files)} processed parquet files")
        
        # Try to find a smaller file first for faster processing
        small_files = [f for f in parquet_files if f.stat().st_size < 50 * 1024 * 1024]  # < 50MB
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
            'ecosystem_cluster', 'sap_flow', 'sap_flow_original', 'sap_flow_quality'
        }
        
        # Also exclude any columns that end with common suffixes that aren't features
        exclude_suffixes = ['_flags', '_md', '_quality', '_original']
        
        feature_cols = []
        for col in df.columns:
            if (col not in exclude_cols and 
                not any(col.endswith(suffix) for suffix in exclude_suffixes)):
                feature_cols.append(col)
        
        print(f"📈 Found {len(feature_cols)} features in order")
        
        # Create feature order mapping
        feature_order = []
        for i, feature in enumerate(feature_cols):
            feature_order.append({
                'feature_index': f'f{i}',
                'feature_name': feature,
                'column_index': i
            })
        
        # Save feature order
        df_order = pd.DataFrame(feature_order)
        df_order.to_csv('v2_feature_order.csv', index=False)
        print("💾 Saved feature order to: v2_feature_order.csv")
        
        # Print first 20 features in order
        print("\n📋 First 20 features in order:")
        for i, feature in enumerate(feature_cols[:20]):
            print(f"  f{i}: {feature}")
        
        # Print last 10 features to show the end
        if len(feature_cols) > 20:
            print("\n📋 Last 10 features in order:")
            for i, feature in enumerate(feature_cols[-10:], start=len(feature_cols)-10):
                print(f"  f{i}: {feature}")
        
        # Verify with another file to ensure consistency
        print("\n🔍 Verifying feature order consistency...")
        if len(parquet_files) > 1:
            # Try a different file
            other_files = [f for f in parquet_files if f != sample_file]
            if other_files:
                verify_file = other_files[0]
                print(f"🔍 Checking consistency with: {verify_file.name}")
                try:
                    df_verify = pd.read_parquet(verify_file)
                    verify_cols = [col for col in df_verify.columns 
                                 if col not in exclude_cols and 
                                 not any(col.endswith(suffix) for suffix in exclude_suffixes)]
                    
                    if verify_cols == feature_cols:
                        print("✅ Feature order is consistent across files")
                    else:
                        print("⚠️  Feature order differs between files")
                        print(f"   Sample file has {len(feature_cols)} features")
                        print(f"   Verify file has {len(verify_cols)} features")
                        
                        # Find differences
                        sample_set = set(feature_cols)
                        verify_set = set(verify_cols)
                        only_in_sample = sample_set - verify_set
                        only_in_verify = verify_set - sample_set
                        
                        if only_in_sample:
                            print(f"   Only in sample: {len(only_in_sample)} features")
                        if only_in_verify:
                            print(f"   Only in verify: {len(only_in_verify)} features")
                except Exception as e:
                    print(f"⚠️  Could not verify with second file: {e}")
        
        return feature_cols
        
    except Exception as e:
        print(f"❌ Error in get_v2_feature_order: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting feature order extraction from processed files...")
    feature_order = get_v2_feature_order()
    
    if feature_order:
        print(f"\n✅ Successfully extracted {len(feature_order)} features in order")
        print("📁 Feature order saved to: v2_feature_order.csv")
        print("🎯 This feature order can now be used with the feature mapping system!")
    else:
        print("❌ Failed to extract feature order") 