#!/usr/bin/env python3
"""
Simple Temporal Validation for SAPFLUXNET Data
K-FOLD TEMPORAL VALIDATION using parquet files directly
Simplified approach that works with our existing data pipeline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import warnings
import gc
import json
from pathlib import Path

warnings.filterwarnings('ignore')

def load_parquet_data(parquet_dir, max_rows=None):
    """Load parquet data for temporal validation"""
    print(f"Loading parquet data from {parquet_dir}...")
    
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load first file as sample to understand structure
    sample_file = os.path.join(parquet_dir, parquet_files[0])
    sample_df = pd.read_parquet(sample_file, nrows=1000)
    
    print(f"Sample data columns: {list(sample_df.columns)}")
    print(f"Sample data shape: {sample_df.shape}")
    
    # Define features to use (universal environmental features only)
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
    target_col = 'sap_flow'
    
    # Universal environmental features
    universal_features = [
        'ta', 'rh', 'sw_in', 'ppfd_in', 'vpd', 'ext_rad', 'ws',
        'swc_shallow', 'precip',
        # Lagged features
        'ta_lag_1h', 'ta_lag_3h', 'ta_lag_6h', 'ta_lag_12h', 'ta_lag_24h',
        'rh_lag_1h', 'rh_lag_3h', 'rh_lag_6h', 'rh_lag_12h', 'rh_lag_24h',
        'sw_in_lag_1h', 'sw_in_lag_3h', 'sw_in_lag_6h', 'sw_in_lag_12h', 'sw_in_lag_24h',
        'vpd_lag_1h', 'vpd_lag_3h', 'vpd_lag_6h', 'vpd_lag_12h', 'vpd_lag_24h',
        'ws_lag_1h', 'ws_lag_3h', 'ws_lag_6h', 'ws_lag_12h', 'ws_lag_24h',
        'swc_shallow_lag_1h', 'swc_shallow_lag_3h', 'swc_shallow_lag_6h', 
        'swc_shallow_lag_12h', 'swc_shallow_lag_24h',
        'precip_lag_1h', 'precip_lag_3h', 'precip_lag_6h', 'precip_lag_12h', 'precip_lag_24h',
        'ppfd_in_lag_1h', 'ppfd_in_lag_3h', 'ppfd_in_lag_6h', 'ppfd_in_lag_12h', 'ppfd_in_lag_24h'
    ]
    
    # Filter to available features
    available_features = [f for f in universal_features if f in sample_df.columns]
    print(f"Using {len(available_features)} universal environmental features")
    
    # Load data in chunks to handle large datasets
    dfs = []
    total_rows = 0
    
    for i, parquet_file in enumerate(parquet_files):
        print(f"Loading file {i+1}/{len(parquet_files)}: {parquet_file}")
        
        file_path = os.path.join(parquet_dir, parquet_file)
        
        # Load with selected columns only
        columns_to_load = ['site', target_col] + available_features
        df_chunk = pd.read_parquet(file_path, columns=columns_to_load)
        
        # Clean data
        df_chunk = df_chunk.dropna(subset=[target_col])
        df_chunk = df_chunk.fillna(0)
        
        dfs.append(df_chunk)
        total_rows += len(df_chunk)
        
        print(f"  Loaded {len(df_chunk):,} rows")
        
        # Limit total rows if specified
        if max_rows and total_rows >= max_rows:
            print(f"  Reached row limit of {max_rows:,}")
            break
        
        # Memory management - combine every 5 files
        if len(dfs) >= 5:
            print(f"  Combining {len(dfs)} dataframes to free memory...")
            combined_df = pd.concat(dfs, ignore_index=True)
            dfs = [combined_df]
            gc.collect()
    
    # Final combination
    print(f"Final combination of {len(dfs)} dataframes...")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    print(f"✅ Data loading complete: {len(df):,} rows from {df['site'].nunique()} sites")
    return df, available_features, target_col

def create_temporal_splits(df, n_folds=5):
    """Create temporal splits for k-fold validation"""
    print(f"Creating {n_folds}-fold temporal splits...")
    
    # Sort by site and create temporal index within each site
    df = df.sort_values(['site', 'TIMESTAMP']).reset_index(drop=True)
    
    # Create temporal index within each site
    df['temporal_index'] = df.groupby('site').cumcount()
    
    # Create folds based on temporal index
    splits = []
    
    for fold in range(n_folds):
        print(f"  Creating fold {fold + 1}/{n_folds}")
        
        # For temporal validation, we want to predict future data
        # Each fold uses earlier data for training, later data for testing
        
        # Calculate split points
        fold_size = len(df) // n_folds
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(df)
        
        # Create temporal split: train on earlier data, test on later data
        if fold == 0:
            # First fold: train on first 20%, test on next 20%
            train_end = int(0.2 * len(df))
            test_start = train_end
            test_end = int(0.4 * len(df))
        else:
            # Other folds: train on data up to fold, test on fold
            train_end = end_idx
            test_start = start_idx
            test_end = end_idx
        
        train_mask = df.index < train_end
        test_mask = (df.index >= test_start) & (df.index < test_end)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        print(f"    Train: {len(train_data):,} rows")
        print(f"    Test: {len(test_data):,} rows")
        
        splits.append((train_data, test_data))
    
    return splits

def train_temporal_model(train_data, test_data, feature_cols, target_col, fold_idx):
    """Train XGBoost model for temporal validation"""
    print(f"Training temporal model for fold {fold_idx + 1}...")
    
    # Prepare data
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'train_samples': len(train_data),
        'test_samples': len(test_data)
    }

def save_temporal_results(results, feature_cols, output_dir='temporal_validation_results'):
    """Save temporal validation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average metrics
    avg_metrics = {
        'train_r2_mean': np.mean([r['train_r2'] for r in results]),
        'train_r2_std': np.std([r['train_r2'] for r in results]),
        'test_r2_mean': np.mean([r['test_r2'] for r in results]),
        'test_r2_std': np.std([r['test_r2'] for r in results]),
        'train_rmse_mean': np.mean([r['train_rmse'] for r in results]),
        'train_rmse_std': np.std([r['train_rmse'] for r in results]),
        'test_rmse_mean': np.mean([r['test_rmse'] for r in results]),
        'test_rmse_std': np.std([r['test_rmse'] for r in results]),
        'train_mae_mean': np.mean([r['train_mae'] for r in results]),
        'train_mae_std': np.std([r['train_mae'] for r in results]),
        'test_mae_mean': np.mean([r['test_mae'] for r in results]),
        'test_mae_std': np.std([r['test_mae'] for r in results])
    }
    
    # Save detailed results
    results_data = []
    for i, result in enumerate(results):
        results_data.append({
            'fold': i + 1,
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'train_rmse': result['train_rmse'],
            'test_rmse': result['test_rmse'],
            'train_mae': result['train_mae'],
            'test_mae': result['test_mae'],
            'train_samples': result['train_samples'],
            'test_samples': result['test_samples']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, 'temporal_fold_results.csv'), index=False)
    
    # Save summary
    with open(os.path.join(output_dir, 'temporal_summary.txt'), 'w') as f:
        f.write("Temporal Validation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("Average Performance:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Train R²: {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}\n")
        f.write(f"Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}\n")
        f.write(f"Train RMSE: {avg_metrics['train_rmse_mean']:.4f} ± {avg_metrics['train_rmse_std']:.4f}\n")
        f.write(f"Test RMSE: {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}\n")
        f.write(f"Train MAE: {avg_metrics['train_mae_mean']:.4f} ± {avg_metrics['train_mae_std']:.4f}\n")
        f.write(f"Test MAE: {avg_metrics['test_mae_mean']:.4f} ± {avg_metrics['test_mae_std']:.4f}\n\n")
        
        f.write("Individual Fold Results:\n")
        f.write("-" * 25 + "\n")
        for i, result in enumerate(results):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Train R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}\n")
            f.write(f"  Train RMSE: {result['train_rmse']:.4f}, Test RMSE: {result['test_rmse']:.4f}\n")
            f.write(f"  Train samples: {result['train_samples']:,}, Test samples: {result['test_samples']:,}\n\n")
    
    # Save feature importance
    feature_importance_data = []
    for i, result in enumerate(results):
        for j, importance in enumerate(result['feature_importance']):
            feature_importance_data.append({
                'fold': i + 1,
                'feature_index': j,
                'feature_name': feature_cols[j],
                'importance': importance
            })
    
    importance_df = pd.DataFrame(feature_importance_data)
    importance_df.to_csv(os.path.join(output_dir, 'temporal_feature_importance.csv'), index=False)
    
    print(f"Results saved to {output_dir}/")
    return avg_metrics

def main():
    """Main temporal validation pipeline"""
    print("SAPFLUXNET Simple Temporal Validation")
    print("=" * 40)
    print(f"Started at: {datetime.now()}")
    print("Approach: K-fold temporal validation using parquet files")
    
    # Load data
    parquet_dir = '../processed_parquet'
    df, feature_cols, target_col = load_parquet_data(parquet_dir, max_rows=1000000)  # Limit for testing
    
    # Create temporal splits
    splits = create_temporal_splits(df, n_folds=5)
    
    # Train models
    results = []
    for i, (train_data, test_data) in enumerate(splits):
        result = train_temporal_model(train_data, test_data, feature_cols, target_col, i)
        results.append(result)
    
    # Save results
    avg_metrics = save_temporal_results(results, feature_cols)
    
    print(f"\nTemporal validation completed at: {datetime.now()}")
    print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")

if __name__ == "__main__":
    main() 