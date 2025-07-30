#!/usr/bin/env python3
"""
Quick verification script to check that all processed parquet files have consistent column counts.
If they all have the same number of columns, the ensemble pipeline should work without issues.
"""

import os
import pandas as pd
from pathlib import Path

def verify_feature_consistency():
    """Check that all processed parquet files have the same number of columns"""
    
    parquet_dir = "processed_parquet"
    
    if not os.path.exists(parquet_dir):
        print(f"❌ Directory not found: {parquet_dir}")
        return False
    
    print(f"🔍 Checking feature consistency in: {parquet_dir}")
    
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {parquet_dir}")
        return False
    
    print(f"📊 Found {len(parquet_files)} parquet files")
    
    column_counts = {}
    inconsistent_files = []
    expected_count = None
    
    for file_path in parquet_files:
        try:
            # Read just the schema, not the data
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            col_count = len(parquet_file.schema.names)
            site_name = file_path.stem
            
            column_counts[site_name] = col_count
            
            if expected_count is None:
                expected_count = col_count
                print(f"📏 Expected column count: {expected_count} (from {site_name})")
            elif col_count != expected_count:
                inconsistent_files.append((site_name, col_count))
                
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False
    
    if inconsistent_files:
        print(f"\n❌ INCONSISTENT FEATURE COUNTS FOUND:")
        print(f"   Expected: {expected_count} columns")
        for site, count in inconsistent_files:
            print(f"   {site}: {count} columns (diff: {count - expected_count:+d})")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Consistent files: {len(parquet_files) - len(inconsistent_files)}")
        print(f"   Inconsistent files: {len(inconsistent_files)}")
        return False
    else:
        print(f"\n✅ ALL FILES HAVE CONSISTENT FEATURE COUNTS!")
        print(f"   Files checked: {len(parquet_files)}")
        print(f"   Columns per file: {expected_count}")
        print(f"   🚀 Ensemble pipeline should work without feature mismatch errors")
        return True

if __name__ == "__main__":
    success = verify_feature_consistency()
    exit(0 if success else 1)