#!/usr/bin/env python3
"""Debug NaN issue in ensemble testing pipeline"""

import pandas as pd
import numpy as np

def test_feature_preparation(site='USA_TNO'):
    """Test the exact feature preparation logic from ensemble testing pipeline"""
    print(f"🔍 Testing feature preparation for {site}")
    
    # Load data
    df = pd.read_parquet(f'processed_parquet/{site}_comprehensive.parquet')
    print(f"📊 Original data: {len(df)} rows, {len(df.columns)} columns")
    
    # Remove rows with missing target values
    df = df.dropna(subset=['sap_flow'])
    print(f"📊 After target filtering: {len(df)} rows")
    
    # Prepare features - MATCH TRAINING PIPELINE EXACTLY
    exclude_cols = [
        'sap_flow', 'site', 'TIMESTAMP', 'solar_TIMESTAMP', 
        'plant_id', 'Unnamed: 0'
    ]
    
    # Also remove any columns ending with specific suffixes (like training pipeline)
    exclude_suffixes = ['_flags', '_md']
    
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(col.endswith(suffix) for suffix in exclude_suffixes):
            continue
        feature_cols.append(col)
    
    print(f"📊 Selected {len(feature_cols)} feature columns")
    
    X = df[feature_cols].copy()
    y = df['sap_flow'].copy()
    
    print(f"📊 Initial NaN values: {X.isnull().sum().sum()}")
    
    # Handle data types and NaN values - MATCH TRAINING PIPELINE EXACTLY
    print("🔧 Converting data types...")
    
    # Check data types before conversion
    print(f"  📊 Object columns: {len(X.select_dtypes(include=['object']).columns)}")
    print(f"  📊 Bool columns: {len(X.select_dtypes(include=[bool]).columns)}")
    
    # Convert boolean columns to numeric (True=1, False=0)
    for col in X.columns:
        if X[col].dtype == bool:
            print(f"    Converting bool column: {col}")
            X[col] = X[col].astype(int)
        elif X[col].dtype == 'object':
            print(f"    Converting object column: {col}")
            # Try to convert object columns to numeric, fill non-numeric with 0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"📊 After dtype conversion: {X.isnull().sum().sum()} NaN values")
    
    # Fill remaining NaN values with 0 (EXACTLY like training pipeline)
    X = X.fillna(0)
    print(f"📊 After final fillna(0): {X.isnull().sum().sum()} NaN values")
    
    # Check for infinite values
    inf_count = np.isinf(X.values).sum()
    print(f"📊 Infinite values: {inf_count}")
    
    # Check data types after all processing
    dtypes = X.dtypes.value_counts()
    print(f"📊 Final data types: {dtypes.to_dict()}")
    
    # Check if any columns still have issues
    problematic_cols = []
    for col in X.columns:
        if X[col].isnull().any():
            problematic_cols.append(col)
        elif np.isinf(X[col]).any():
            problematic_cols.append(f"{col} (inf)")
    
    if problematic_cols:
        print(f"❌ Problematic columns: {problematic_cols[:5]}")
    else:
        print("✅ All columns are clean")
    
    return X, y, feature_cols

if __name__ == "__main__":
    X, y, feature_cols = test_feature_preparation('USA_TNO')
    print(f"\n🎯 Final result: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target values: {len(y)} samples")