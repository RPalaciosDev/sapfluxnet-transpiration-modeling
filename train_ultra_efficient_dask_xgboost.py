#!/usr/bin/env python3
"""
Google Colab-Optimized Dask XGBoost Training for SAPFLUXNET Data
Ultra-conservative memory management to avoid worker kills.
"""

import dask.dataframe as dd
import dask.array as da
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import warnings
import gc
from dask.distributed import Client, LocalCluster
import psutil
warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def setup_dask_client(memory_limit_gb=None):
    """Setup Dask client with ultra-conservative memory constraints for Colab"""
    if memory_limit_gb is None:
        memory_limit_gb = max(1.0, get_available_memory_gb() * 0.6)  # Use only 60% of available memory
    
    print(f"Setting up Dask client with {memory_limit_gb:.1f}GB memory limit...")
    
    # Create local cluster with very conservative memory limits
    cluster = LocalCluster(
        n_workers=1,  # Single worker to avoid memory splitting
        threads_per_worker=2,  # Use multiple threads
        memory_limit=f"{memory_limit_gb}GB",
        silence_logs=False,
        dashboard_address=None,  # Disable dashboard to save memory
        processes=False  # Use threads instead of processes
    )
    
    client = Client(cluster)
    print(f"Dask client created: {client}")
    return client

def load_data_colab_safe(data_dir='comprehensive_processed', chunk_size_mb=50):
    """Load data with Google Colab-safe partitioning"""
    print(f"Loading data from {data_dir} with ultra-conservative partitioning (chunk_size: {chunk_size_mb}MB)...")
    
    # Read with very small partitions to avoid memory issues
    ddf = dd.read_parquet(
        f"{data_dir}/*_comprehensive.parquet",
        blocksize=f"{chunk_size_mb}MB"  # Very small chunks for Colab
    )
    
    print(f"Dask DataFrame created with {ddf.npartitions} partitions")
    print(f"Estimated partition size: ~{chunk_size_mb}MB each")
    
    # Get column info from first partition only
    sample = ddf.get_partition(0).compute()
    print(f"Columns: {len(sample.columns)}")
    print(f"Sample from first partition: {len(sample)} rows")
    
    return ddf

def prepare_features_colab_safe(ddf):
    """Prepare features without triggering expensive operations"""
    print("Preparing features with ultra-conservative operations...")
    
    # Get column info from metadata
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id']
    target_col = 'sap_flow'
    
    # Get all columns
    all_cols = ddf.columns.tolist()
    
    # Find numeric columns (assume most are numeric for efficiency)
    feature_cols = [col for col in all_cols 
                   if col not in exclude_cols + [target_col] 
                   and not col.endswith('_flags')  # Exclude flag columns
                   and not col.endswith('_md')]    # Exclude metadata columns
    
    print(f"Target variable: {target_col}")
    print(f"Features: {len(feature_cols)} columns")
    
    # DON'T use dropna() here - it triggers computation across all partitions
    # Instead, we'll handle missing values during training
    
    return ddf, feature_cols

def create_temporal_split_colab_safe(ddf, feature_cols, target_col, train_ratio=0.8):
    """Create temporal split with ultra-conservative memory management"""
    print(f"Creating temporal split (train_ratio={train_ratio}) with streaming...")
    
    # Select only needed columns to reduce memory
    needed_cols = [target_col] + feature_cols + ['site']
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    # Filter to only needed columns
    ddf_filtered = ddf[needed_cols]
    
    # Sort by site and timestamp (lazy operation)
    if 'TIMESTAMP' in ddf_filtered.columns:
        ddf_sorted = ddf_filtered.sort_values(['site', 'TIMESTAMP'])
    else:
        ddf_sorted = ddf_filtered.sort_values('site')
    
    # Create split function that works on each partition
    def split_partition_to_train(partition_df):
        """Extract training data from partition"""
        # Remove rows with missing target values first
        partition_df = partition_df.dropna(subset=[target_col])
        
        if len(partition_df) == 0:
            return pd.DataFrame(columns=partition_df.columns)
        
        train_list = []
        
        for site in partition_df['site'].unique():
            site_data = partition_df[partition_df['site'] == site].copy()
            
            if len(site_data) > 1:
                split_idx = max(1, int(len(site_data) * train_ratio))
                train_list.append(site_data.iloc[:split_idx])
            else:
                train_list.append(site_data)  # Single row goes to train
        
        return pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=partition_df.columns)
    
    def split_partition_to_test(partition_df):
        """Extract test data from partition"""
        # Remove rows with missing target values first
        partition_df = partition_df.dropna(subset=[target_col])
        
        if len(partition_df) == 0:
            return pd.DataFrame(columns=partition_df.columns)
        
        test_list = []
        
        for site in partition_df['site'].unique():
            site_data = partition_df[partition_df['site'] == site].copy()
            
            if len(site_data) > 1:
                split_idx = max(1, int(len(site_data) * train_ratio))
                if split_idx < len(site_data):
                    test_list.append(site_data.iloc[split_idx:])
        
        return pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=partition_df.columns)
    
    # Apply split functions to create train and test dataframes
    print("Creating train/test splits with streaming operations...")
    
    # Create train dataframe using map_partitions
    train_ddf = ddf_sorted.map_partitions(
        split_partition_to_train,
        meta=pd.DataFrame(columns=needed_cols)
    )
    
    # Create test dataframe using map_partitions
    test_ddf = ddf_sorted.map_partitions(
        split_partition_to_test,
        meta=pd.DataFrame(columns=needed_cols)
    )
    
    # DON'T compute lengths - this triggers the expensive operations!
    print("Split completed:")
    print("  Train: Created (length computation skipped to avoid memory issues)")
    print("  Test:  Created (length computation skipped to avoid memory issues)")
    
    return train_ddf, test_ddf

def train_colab_safe_xgboost(train_ddf, test_ddf, feature_cols, target_col, client):
    """Train XGBoost with Google Colab-safe memory management"""
    print("Training XGBoost with Google Colab-safe memory management...")
    
    # Handle missing values by filling with 0 (XGBoost can handle this)
    train_ddf = train_ddf.fillna(0)
    test_ddf = test_ddf.fillna(0)
    
    # Prepare data as Dask arrays (no .compute() calls!)
    X_train = train_ddf[feature_cols].to_dask_array(lengths=True)
    y_train = train_ddf[target_col].to_dask_array(lengths=True)
    X_test = test_ddf[feature_cols].to_dask_array(lengths=True)
    y_test = test_ddf[target_col].to_dask_array(lengths=True)
    
    print(f"Training data prepared (shape computation skipped)")
    print(f"Test data prepared (shape computation skipped)")
    
    # XGBoost parameters optimized for memory efficiency and Colab
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,          # Reduced for memory efficiency
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',   # Memory efficient
        'max_bin': 128,          # Reduced for memory efficiency
        'verbosity': 0,          # Reduce output
        'nthread': 2             # Limit threads
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Create DMatrix objects (lazy evaluation)
    print("Creating DMatrix objects...")
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
    
    print("Starting distributed training...")
    
    # Train the model with conservative settings
    output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=200,  # Reduced for memory efficiency
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,  # Reduced for faster training
        verbose_eval=False
    )
    
    model = output['booster']
    print("Training completed!")
    
    # Make predictions (still streaming!)
    print("Making predictions...")
    y_pred_train = xgb.dask.predict(client, model, dtrain)
    y_pred_test = xgb.dask.predict(client, model, dtest)
    
    # Only now do we compute the results for metrics
    print("Computing metrics...")
    y_train_actual = y_train.compute()
    y_test_actual = y_test.compute()
    y_pred_train_actual = y_pred_train.compute()
    y_pred_test_actual = y_pred_test.compute()
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual)),
        'train_mae': mean_absolute_error(y_train_actual, y_pred_train_actual),
        'test_mae': mean_absolute_error(y_test_actual, y_pred_test_actual),
        'train_r2': r2_score(y_train_actual, y_pred_train_actual),
        'test_r2': r2_score(y_test_actual, y_pred_test_actual)
    }
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model, metrics

def get_feature_importance(model, feature_cols):
    """Get and display feature importance"""
    try:
        importance_dict = model.get_score(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
    except:
        # Fallback if get_score doesn't work
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': [0.0] * len(feature_cols)
        })
    
    print(f"\nTop 15 Most Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def save_model_and_results(model, metrics, feature_importance, output_dir='colab_models'):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_colab_safe_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_path = f"{output_dir}/sapfluxnet_colab_features_{timestamp}.csv"
    feature_importance.to_csv(feature_path, index=False)
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_colab_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Google Colab-Safe Dask-XGBoost Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nModel and results saved:")
    print(f"  Model: {model_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main Google Colab-safe training pipeline"""
    print("SAPFLUXNET Google Colab-Safe Dask XGBoost Training")
    print("=" * 55)
    print(f"Started at: {datetime.now()}")
    
    # Check if we're in Google Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted.")
        data_dir = '/content/drive/MyDrive/comprehensive_processed'
    except ImportError:
        print("Not running in Google Colab, using local directory.")
        data_dir = 'comprehensive_processed'
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    if available_memory < 2.0:
        print("⚠️  Warning: Low memory available. Using ultra-conservative settings.")
        chunk_size = 30
        memory_limit = max(1.0, available_memory * 0.5)
    else:
        chunk_size = 50
        memory_limit = max(1.0, available_memory * 0.6)
    
    # Setup Dask client
    client = setup_dask_client(memory_limit_gb=memory_limit)
    
    try:
        # Step 1: Load data with Google Colab-safe partitioning
        ddf = load_data_colab_safe(data_dir, chunk_size_mb=chunk_size)
        
        # Step 2: Prepare features
        ddf_clean, feature_cols = prepare_features_colab_safe(ddf)
        
        # Step 3: Create temporal split (ultra-conservative)
        train_ddf, test_ddf = create_temporal_split_colab_safe(
            ddf_clean, feature_cols, 'sap_flow'
        )
        
        # Step 4: Train model (Google Colab-safe)
        model, metrics = train_colab_safe_xgboost(
            train_ddf, test_ddf, feature_cols, 'sap_flow', client
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 6: Save results
        model_path = save_model_and_results(model, metrics, feature_importance)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("💡 Try reducing chunk size or memory limit if this persists.")
        raise
    
    finally:
        # Clean up
        client.close()
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main()