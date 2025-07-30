#!/usr/bin/env python3
"""
Direct fix: Standardize ALL parquet files to have identical features to THA_KHU.
Ensures features are in the same order and all accounted for.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def standardize_all_features():
    """Make all parquet files identical to THA_KHU in features and order"""
    
    parquet_dir = "processed_parquet"
    reference_file = os.path.join(parquet_dir, "THA_KHU_comprehensive.parquet")
    
    if not os.path.exists(reference_file):
        print(f"❌ Reference file not found: {reference_file}")
        return False
    
    # Get reference features and their exact order
    print(f"📂 Loading reference schema from THA_KHU...")
    ref_parquet = pq.ParquetFile(reference_file)
    reference_features = list(ref_parquet.schema.names)  # Preserves exact order
    reference_count = len(reference_features)
    
    print(f"📏 Reference: {reference_count} features in exact order")
    
    # Process all other files
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    processed = 0
    fixed = 0
    
    for file_path in parquet_files:
        site_name = file_path.stem
        
        # Skip reference file
        if site_name == 'THA_KHU_comprehensive':
            continue
        
        try:
            # Load current file
            df = pd.read_parquet(file_path)
            current_features = list(df.columns)
            current_count = len(current_features)
            
            # Check if already correct
            if (current_count == reference_count and 
                current_features == reference_features):
                processed += 1
                continue
            
            print(f"🔧 Fixing {site_name}: {current_count} → {reference_count} features")
            
            # Add missing features with 0.0
            missing = set(reference_features) - set(current_features)
            for feature in missing:
                df[feature] = 0.0
            
            # Remove extra features
            extra = set(current_features) - set(reference_features)
            if extra:
                df = df.drop(columns=list(extra))
            
            # Reorder columns to match reference EXACTLY
            df = df[reference_features]  # This preserves exact order
            
            # Save fixed file
            df.to_parquet(file_path, index=False, compression='snappy')
            
            # Verify fix
            verify_df = pd.read_parquet(file_path)
            if (len(verify_df.columns) == reference_count and 
                list(verify_df.columns) == reference_features):
                print(f"  ✅ Fixed: {len(verify_df.columns)} features in correct order")
                fixed += 1
            else:
                print(f"  ❌ Fix failed")
            
        except Exception as e:
            print(f"  ❌ Error fixing {site_name}: {e}")
            continue
        
        processed += 1
    
    print(f"\n📊 RESULTS:")
    print(f"  Files processed: {processed}")
    print(f"  Files fixed: {fixed}")
    print(f"  Reference features: {reference_count}")
    
    return fixed >= 0

if __name__ == "__main__":
    standardize_all_features()