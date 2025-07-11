#!/usr/bin/env python3
"""
Google Colab-Safe XGBoost Training for SAPFLUXNET Data
Designed to work within Colab's memory constraints by avoiding all memory traps.
"""

import dask.dataframe as dd
import dask.array as da
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def setup_minimal_dask_client():
    """Setup minimal Dask client for Google Colab"""
    available_memory = get_available_memory_gb()
    memory_limit = max(1.0, available_memory * 0.5)  # Use only 50% of available memory
    
    print(f"Available memory: {available_memory:.1f}GB")
    print(f"Setting up minimal Dask client with {memory_limit:.1f}GB memory limit...")
    
    # Create minimal local cluster
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,  # Single thread to minimize memory usage
        memory_limit=f"{memory_limit}GB",
        silence_logs=True,
        dashboard_address=None,  # Disable dashboard
        processes=False  # Use threads
    )
    
    client = Client(cluster)
    print(f"Dask client created successfully")
    return client

def load_data_minimal(data_dir):
    """Load data with minimal memory footprint"""
    print(f"Loading data from {data_dir} with minimal memory usage...")
    
    # Use very small chunks
    chunk_size = 30  # 30MB chunks
    
    try:
        ddf = dd.read_parquet(
            f"{data_dir}/*.parquet",
            blocksize=f"{chunk_size}MB"
        )
        print(f"Data loaded successfully with {ddf.npartitions} partitions")
        
        # Get sample to understand structure
        sample = ddf.get_partition(0).compute()
        print(f"Sample partition: {len(sample)} rows, {len(sample.columns)} columns")
        
        return ddf
        
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        print("Trying CSV files...")
        
        # Fallback to CSV
        ddf = dd.read_csv(
            f"{data_dir}/*.csv",
            blocksize=f"{chunk_size}MB"
        )
        print(f"CSV data loaded with {ddf.npartitions} partitions")
        return ddf

def prepare_features_minimal(ddf):
    """Prepare features with minimal operations"""
    print("Preparing features...")
    
    # Get column names from sample
    sample = ddf.get_partition(0).compute()
    all_cols = sample.columns.tolist()
    
    # Define columns to exclude
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0'  # Common index column
    ]
    
    # Find target column
    target_col = 'sap_flow'
    if target_col not in all_cols:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Select feature columns
    feature_cols = [col for col in all_cols 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)} columns")
    print(f"First 10 features: {feature_cols[:10]}")
    
    return ddf, feature_cols, target_col

def create_simple_split(ddf, feature_cols, target_col, train_ratio=0.8):
    """Create simple train/test split without temporal complexity"""
    print(f"Creating simple split (train_ratio={train_ratio})...")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    
    ddf_subset = ddf[needed_cols]
    
    # Simple partition-based split to avoid memory issues
    n_partitions = ddf_subset.npartitions
    train_partitions = int(n_partitions * train_ratio)
    
    print(f"Total partitions: {n_partitions}")
    print(f"Train partitions: {train_partitions}")
    print(f"Test partitions: {n_partitions - train_partitions}")
    
    # Split by partition index (not ideal but memory-safe)
    train_ddf = ddf_subset.get_partition(slice(0, train_partitions))
    test_ddf = ddf_subset.get_partition(slice(train_partitions, n_partitions))
    
    print("Split completed successfully")
    return train_ddf, test_ddf

def train_minimal_xgboost(train_ddf, test_ddf, feature_cols, target_col, client):
    """Train XGBoost with minimal memory usage"""
    print("Training XGBoost with minimal memory usage...")
    
    # Handle categorical columns and missing values properly
    def fix_categorical_and_fillna(df):
        """Fix categorical columns and fill missing values"""
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            if df_fixed[col].dtype.name == 'category':
                # Convert categorical to numeric, preserving the encoding
                df_fixed[col] = df_fixed[col].cat.codes
                # Replace -1 (missing category) with NaN, then fill with 0
                df_fixed[col] = df_fixed[col].replace(-1, np.nan)
        
        # Now fill all missing values with 0
        df_fixed = df_fixed.fillna(0)
        return df_fixed
    
    # Apply the fix to both train and test dataframes
    print("Converting categorical columns and filling missing values...")
    
    # Get a sample to determine the output meta
    sample_fixed = fix_categorical_and_fillna(train_ddf.get_partition(0).compute())
    
    train_ddf = train_ddf.map_partitions(fix_categorical_and_fillna, meta=sample_fixed)
    test_ddf = test_ddf.map_partitions(fix_categorical_and_fillna, meta=sample_fixed)
    
    # Convert to Dask arrays
    X_train = train_ddf[feature_cols].to_dask_array(lengths=True)
    y_train = train_ddf[target_col].to_dask_array(lengths=True)
    X_test = test_ddf[feature_cols].to_dask_array(lengths=True)
    y_test = test_ddf[target_col].to_dask_array(lengths=True)
    
    print("Data converted to Dask arrays")
    
    # Minimal XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 3,          # Very shallow trees
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 64,           # Very small bins
        'verbosity': 0,
        'nthread': 1             # Single thread
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Create DMatrix objects
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
    
    print("DMatrix objects created")
    
    # Train model
    print("Starting training...")
    output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=100,     # Fewer rounds
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    model = output['booster']
    print("Training completed!")
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = xgb.dask.predict(client, model, dtrain)
    y_pred_test = xgb.dask.predict(client, model, dtest)
    
    # Compute metrics
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

def save_results(model, metrics, feature_cols, output_dir='colab_models'):
    """Save model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_colab_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in training:\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Google Colab Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training completed: {datetime.now()}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nResults saved:")
    print(f"  Model: {model_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab-Safe XGBoost Training")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    
    # Check if we're in Google Colab
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted.")
        data_dir = '/content/drive/MyDrive/comprehensive_processed'
    except ImportError:
        print("Not running in Google Colab, using local directory.")
        data_dir = 'comprehensive_processed'
    
    # Setup minimal Dask client
    client = setup_minimal_dask_client()
    
    try:
        # Step 1: Load data
        ddf = load_data_minimal(data_dir)
        
        # Step 2: Prepare features
        ddf_clean, feature_cols, target_col = prepare_features_minimal(ddf)
        
        # Step 3: Create simple split
        train_ddf, test_ddf = create_simple_split(ddf_clean, feature_cols, target_col)
        
        # Step 4: Train model
        model, metrics = train_minimal_xgboost(
            train_ddf, test_ddf, feature_cols, target_col, client
        )
        
        # Step 5: Save results
        model_path = save_results(model, metrics, feature_cols)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("💡 Try reducing memory usage or check data format.")
        raise
    
    finally:
        # Clean up
        try:
            client.close()
        except:
            pass
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 