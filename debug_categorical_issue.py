"""
Debug script to trace where categorical columns are being created
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
import os

def check_dtypes(df, stage_name):
    """Check and report data types at each stage"""
    print(f"\n{'='*50}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*50}")
    
    if hasattr(df, 'dtypes'):
        # It's a pandas DataFrame
        dtype_counts = df.dtypes.value_counts()
        print(f"Data types:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Check for categorical columns specifically
        categorical_cols = [col for col in df.columns if df[col].dtype.name == 'category']
        if categorical_cols:
            print(f"\n❌ Found {len(categorical_cols)} categorical columns:")
            for col in categorical_cols:
                print(f"  - {col}: {df[col].dtype}")
                if hasattr(df[col], 'cat'):
                    try:
                        print(f"    Categories: {list(df[col].cat.categories)}")
                    except:
                        print(f"    Categories: <unknown - Dask categorical>")
        else:
            print("✅ No categorical columns found")
    else:
        print("This is a Dask DataFrame - checking dtypes")
        # For Dask DataFrames, check dtypes directly
        dtype_counts = df.dtypes.value_counts()
        print(f"Data types:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Check for categorical columns in Dask
        categorical_cols = [col for col in df.columns if df[col].dtype.name == 'category']
        if categorical_cols:
            print(f"\n❌ Found {len(categorical_cols)} categorical columns:")
            for col in categorical_cols:
                print(f"  - {col}: {df[col].dtype}")
                print(f"    This is a Dask categorical column")
        else:
            print("✅ No categorical columns found")

def debug_training_pipeline():
    """Debug the training pipeline step by step"""
    print("DEBUGGING CATEGORICAL COLUMN CREATION")
    print("=" * 60)
    
    # Load a single file first
    sample_file = "comprehensive_processed/ARG_MAZ_comprehensive.parquet"
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        return
    
    # Stage 1: Load single file
    print(f"Loading sample file: {sample_file}")
    df = pd.read_parquet(sample_file)
    check_dtypes(df, "1. Single file loaded")
    
    # Stage 2: Load with Dask
    print(f"\nLoading with Dask...")
    ddf = dd.read_parquet(sample_file, blocksize="25MB")
    check_dtypes(ddf, "2. Dask DataFrame loaded")
    
    # Stage 3: Select columns
    print(f"\nSelecting columns...")
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
    target_col = 'sap_flow'
    
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    ddf_subset = ddf[needed_cols]
    check_dtypes(ddf_subset, "3. Columns selected")
    
    # Stage 4: Try temporal split on small sample
    print(f"\nTesting temporal split...")
    
    def simple_temporal_split(partition_df):
        """Simple temporal split for testing"""
        # Remove rows with missing target
        partition_df = partition_df.dropna(subset=[target_col])
        
        if len(partition_df) == 0:
            return pd.DataFrame(columns=partition_df.columns)
        
        # Sort by timestamp if available
        if 'TIMESTAMP' in partition_df.columns:
            partition_df = partition_df.sort_values('TIMESTAMP')
        
        # Take first 80% for train
        split_idx = max(1, int(len(partition_df) * 0.8))
        return partition_df.iloc[:split_idx]
    
    train_ddf = ddf_subset.map_partitions(
        simple_temporal_split,
        meta=pd.DataFrame(columns=needed_cols)
    )
    
    check_dtypes(train_ddf, "4. After temporal split")
    
    # Stage 5: Try fillna operation
    print(f"\nTesting fillna operation...")
    
    try:
        # Try the standard fillna that was failing
        train_filled = train_ddf.fillna(0)
        check_dtypes(train_filled, "5. After fillna(0)")
        print("✅ Standard fillna(0) worked!")
        
    except Exception as e:
        print(f"❌ Standard fillna(0) failed: {str(e)}")
        
        # Try our safe fillna
        def safe_fillna_debug(partition_df):
            """Debug version of safe fillna"""
            print(f"Processing partition with {len(partition_df)} rows")
            
            for col in partition_df.columns:
                if partition_df[col].dtype.name == 'category':
                    print(f"  Found categorical column: {col}")
                    try:
                        print(f"    Categories: {list(partition_df[col].cat.categories)}")
                    except:
                        print(f"    Categories: <unknown>")
                    print(f"    Has NaN: {partition_df[col].isna().any()}")
                    
                    # Convert to numeric
                    if hasattr(partition_df[col], 'cat'):
                        partition_df[col] = partition_df[col].cat.codes.astype('float64')
                        print(f"    Converted to numeric codes")
                
                # Fill NaN values
                if partition_df[col].dtype in ['object']:
                    partition_df[col] = partition_df[col].fillna(str(0))
                else:
                    partition_df[col] = partition_df[col].fillna(0)
            
            return partition_df
        
        train_filled = train_ddf.map_partitions(
            safe_fillna_debug,
            meta=train_ddf._meta
        )
        
        check_dtypes(train_filled, "5. After safe fillna")
    
    # Stage 6: Try to_dask_array conversion
    print(f"\nTesting to_dask_array conversion...")
    
    try:
        # This is where the error occurred in the original script
        X_train = train_filled[feature_cols].to_dask_array(lengths=True)
        print("✅ to_dask_array conversion worked!")
        
    except Exception as e:
        print(f"❌ to_dask_array conversion failed: {str(e)}")
        
        # Try computing first
        print("Trying compute() first...")
        train_pd = train_filled.compute()
        check_dtypes(train_pd, "6. After compute()")
        
        print("✅ Compute worked - no categorical columns in final result")

def test_dask_categorical_behavior():
    """Test specifically how Dask handles categorical columns"""
    print("\n" + "="*60)
    print("TESTING DASK CATEGORICAL BEHAVIOR")
    print("="*60)
    
    # Load multiple files to see if Dask creates categoricals
    print("Loading multiple files with Dask...")
    
    try:
        ddf_all = dd.read_parquet("comprehensive_processed/*.parquet", blocksize="25MB")
        print(f"Loaded {ddf_all.npartitions} partitions from all files")
        
        check_dtypes(ddf_all, "All files loaded with Dask")
        
        # Test fillna on the multi-file dataset
        print("\nTesting fillna on multi-file dataset...")
        try:
            ddf_filled = ddf_all.fillna(0)
            print("✅ fillna(0) worked on multi-file dataset")
        except Exception as e:
            print(f"❌ fillna(0) failed on multi-file dataset: {str(e)}")
            
            # This is likely where the categorical issue occurs
            print("This is where the categorical issue occurs!")
            print("Dask creates categorical columns when loading multiple files")
            print("with different string values in the same column.")
            
    except Exception as e:
        print(f"Error loading multiple files: {str(e)}")

if __name__ == "__main__":
    debug_training_pipeline()
    test_dask_categorical_behavior() 