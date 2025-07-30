#!/usr/bin/env python3
"""
Post-processing script to ensure ALL parquet files have identical features to THA_KHU.
This fixes existing files by adding missing features (filled with 0) and removing extra features.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

def fix_all_feature_consistency():
    """Fix all parquet files to have identical features to THA_KHU"""
    
    parquet_dir = "processed_parquet"
    reference_file = os.path.join(parquet_dir, "THA_KHU_comprehensive.parquet")
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return False
    
    # Load reference features from THA_KHU
    print(f"📂 Loading reference features from THA_KHU...")
    ref_parquet = pq.ParquetFile(reference_file)
    reference_features = list(ref_parquet.schema.names)
    reference_count = len(reference_features)
    
    print(f"📏 Reference feature count: {reference_count}")
    print(f"🔍 Checking all other parquet files...")
    
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    fixed_count = 0
    already_correct = 0
    
    for file_path in parquet_files:
        site_name = file_path.stem
        
        # Skip the reference file
        if site_name == 'THA_KHU_comprehensive':
            continue
            
        try:
            # Check current feature count
            current_parquet = pq.ParquetFile(file_path)
            current_features = list(current_parquet.schema.names)
            current_count = len(current_features)
            
            if current_count == reference_count and set(current_features) == set(reference_features):
                already_correct += 1
                continue
            
            print(f"  🔧 Fixing {site_name}: {current_count} → {reference_count} features")
            
            # Load the data
            df = pd.read_parquet(file_path)
            
            # Find missing and extra features
            current_feature_set = set(current_features)
            reference_feature_set = set(reference_features)
            
            missing_features = reference_feature_set - current_feature_set
            extra_features = current_feature_set - reference_feature_set
            
            # Add missing features with 0.0
            for feature in missing_features:
                df[feature] = 0.0
            
            # Remove extra features
            if extra_features:
                df = df.drop(columns=list(extra_features))
            
            # Reorder columns to match reference exactly
            df = df.reindex(columns=reference_features, fill_value=0.0)
            
            # Save the fixed file
            df.to_parquet(file_path, index=False, compression='snappy')
            
            # Verify the fix
            verify_parquet = pq.ParquetFile(file_path)
            verify_count = len(verify_parquet.schema.names)
            
            if verify_count == reference_count:
                print(f"    ✅ Fixed successfully: {verify_count} features")
                fixed_count += 1
            else:
                print(f"    ❌ Fix failed: still {verify_count} features")
                
        except Exception as e:
            print(f"    ❌ Error fixing {site_name}: {e}")
            continue
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files processed: {len(parquet_files) - 1}")  # -1 for THA_KHU
    print(f"   Already correct: {already_correct}")
    print(f"   Fixed: {fixed_count}")
    print(f"   Failed: {len(parquet_files) - 1 - already_correct - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Fixed {fixed_count} files to match THA_KHU's {reference_count} features")
        print(f"🚀 All files should now have consistent feature counts!")
        return True
    else:
        print(f"\n⚠️  No files needed fixing")
        return already_correct == len(parquet_files) - 1

if __name__ == "__main__":
    success = fix_all_feature_consistency()
    exit(0 if success else 1)