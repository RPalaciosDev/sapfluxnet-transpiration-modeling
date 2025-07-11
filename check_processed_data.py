"""
Comprehensive Data Check Script for SAPFLUXNET Processed Data
Checks for categorical columns and other potential issues that might cause training failures.
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def check_single_file(file_path):
    """Check a single processed file for issues"""
    print(f"\n{'='*60}")
    print(f"Checking: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        # Load the file
        df = pd.read_parquet(file_path)
        print(f"✅ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for categorical columns
        categorical_cols = [col for col in df.columns if df[col].dtype.name == 'category']
        if categorical_cols:
            print(f"❌ Found {len(categorical_cols)} categorical columns:")
            for col in categorical_cols:
                print(f"  - {col}: {df[col].dtype}")
                print(f"    Categories: {list(df[col].cat.categories)}")
                print(f"    Has NaN: {df[col].isna().any()}")
        else:
            print("✅ No categorical columns found")
        
        # Check data types
        dtype_counts = df.dtypes.value_counts()
        print(f"\nData type distribution:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Check for object columns (should be minimal)
        object_cols = [col for col in df.columns if df[col].dtype == 'object']
        if object_cols:
            print(f"\n⚠️  Found {len(object_cols)} object columns:")
            for col in object_cols[:10]:  # Show first 10
                unique_vals = df[col].nunique()
                print(f"  - {col}: {unique_vals} unique values")
                if unique_vals <= 10:
                    print(f"    Values: {list(df[col].unique())}")
        else:
            print("✅ No object columns found")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if len(cols_with_missing) > 0:
            print(f"\n📊 Columns with missing values ({len(cols_with_missing)} columns):")
            for col in cols_with_missing.head(10).index:
                pct_missing = (missing_counts[col] / len(df)) * 100
                print(f"  - {col}: {missing_counts[col]} ({pct_missing:.1f}%)")
        else:
            print("✅ No missing values found")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            print(f"\n⚠️  Found infinite values in {len(inf_cols)} columns:")
            for col in inf_cols:
                inf_count = np.isinf(df[col]).sum()
                print(f"  - {col}: {inf_count} infinite values")
        else:
            print("✅ No infinite values found")
        
        # Check target column
        if 'sap_flow' in df.columns:
            sap_flow_stats = df['sap_flow'].describe()
            print(f"\n📊 Sap flow statistics:")
            print(f"  Count: {sap_flow_stats['count']}")
            print(f"  Mean: {sap_flow_stats['mean']:.4f}")
            print(f"  Std: {sap_flow_stats['std']:.4f}")
            print(f"  Min: {sap_flow_stats['min']:.4f}")
            print(f"  Max: {sap_flow_stats['max']:.4f}")
            print(f"  Missing: {df['sap_flow'].isnull().sum()}")
        else:
            print("❌ No 'sap_flow' target column found")
        
        # Check for problematic columns that should have been removed
        problematic_cols = [
            'pl_name', 'swc_deep', 'netrad', 'seasonal_leaf_area', 
            'water_stress_index', 'moisture_availability', 'wind_stress',
            'light_efficiency', 'ppfd_efficiency', 'stomatal_conductance_proxy'
        ]
        
        found_problematic = [col for col in problematic_cols if col in df.columns]
        if found_problematic:
            print(f"\n⚠️  Found problematic columns that should have been removed:")
            for col in found_problematic:
                print(f"  - {col}")
        else:
            print("✅ No problematic columns found")
        
        return {
            'file': os.path.basename(file_path),
            'rows': len(df),
            'columns': len(df.columns),
            'categorical_columns': len(categorical_cols),
            'object_columns': len(object_cols),
            'missing_columns': len(cols_with_missing),
            'infinite_columns': len(inf_cols),
            'has_target': 'sap_flow' in df.columns,
            'problematic_columns': len(found_problematic)
        }
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }

def check_all_files(data_dir='comprehensive_processed'):
    """Check all processed files"""
    print("SAPFLUXNET Processed Data Quality Check")
    print("=" * 60)
    
    # Find all parquet files
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Check each file
    results = []
    for file_path in parquet_files[:5]:  # Check first 5 files
        result = check_single_file(file_path)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_files = [r for r in results if 'error' not in r]
    failed_files = [r for r in results if 'error' in r]
    
    print(f"Successfully processed: {len(successful_files)}")
    print(f"Failed to process: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for result in failed_files:
            print(f"  - {result['file']}: {result['error']}")
    
    if successful_files:
        print(f"\nOverall statistics:")
        total_categorical = sum(r['categorical_columns'] for r in successful_files)
        total_object = sum(r['object_columns'] for r in successful_files)
        total_problematic = sum(r['problematic_columns'] for r in successful_files)
        files_with_target = sum(r['has_target'] for r in successful_files)
        
        print(f"  Total categorical columns: {total_categorical}")
        print(f"  Total object columns: {total_object}")
        print(f"  Total problematic columns: {total_problematic}")
        print(f"  Files with target column: {files_with_target}/{len(successful_files)}")
        
        if total_categorical > 0:
            print(f"\n❌ ISSUE: Found categorical columns in processed data!")
            print(f"   This will cause the training script to fail.")
            print(f"   The pipeline needs to be re-run to fix categorical encoding.")
        
        if total_object > 0:
            print(f"\n⚠️  WARNING: Found object columns in processed data.")
            print(f"   These should be encoded or removed.")
        
        if total_problematic > 0:
            print(f"\n⚠️  WARNING: Found problematic columns that should have been removed.")
        
        if files_with_target < len(successful_files):
            print(f"\n❌ ISSUE: Some files missing target column!")
        
        if total_categorical == 0 and total_object == 0 and total_problematic == 0:
            print(f"\n✅ All files look good for training!")

def check_specific_categorical_issue():
    """Check specifically for the categorical issue that caused the training failure"""
    print(f"\n{'='*60}")
    print("SPECIFIC CATEGORICAL ISSUE CHECK")
    print(f"{'='*60}")
    
    # Load a sample file
    parquet_files = glob.glob("comprehensive_processed/*.parquet")
    if not parquet_files:
        print("No parquet files found")
        return
    
    sample_file = parquet_files[0]
    print(f"Checking sample file: {os.path.basename(sample_file)}")
    
    try:
        df = pd.read_parquet(sample_file)
        
        # Try the exact operation that failed
        print("Testing fillna(0) operation...")
        
        # Check each column individually
        problematic_cols = []
        for col in df.columns:
            try:
                if df[col].dtype.name == 'category':
                    print(f"  Categorical column: {col}")
                    # Try fillna(0) on this column
                    df[col].fillna(0)
                    print(f"    ✅ fillna(0) works")
                else:
                    # Try on non-categorical columns too
                    df[col].fillna(0)
            except Exception as e:
                print(f"    ❌ fillna(0) failed on {col}: {str(e)}")
                problematic_cols.append(col)
        
        if problematic_cols:
            print(f"\nFound {len(problematic_cols)} problematic columns:")
            for col in problematic_cols:
                print(f"  - {col}: {df[col].dtype}")
                if df[col].dtype.name == 'category':
                    print(f"    Categories: {list(df[col].cat.categories)}")
        else:
            print("✅ No problematic columns found")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Check all files
    check_all_files()
    
    # Check specific categorical issue
    check_specific_categorical_issue() 