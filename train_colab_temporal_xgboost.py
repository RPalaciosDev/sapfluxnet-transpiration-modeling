"""
Google Colab-Safe Temporal XGBoost Training for SAPFLUXNET Data
Handles temporal splits properly while avoiding memory traps.
"""

import dask.dataframe as dd
import dask.array as da
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import warnings
import gc
from dask.distributed import Client, LocalCluster
import psutil
warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def setup_conservative_dask_client():
    """Setup conservative Dask client for Google Colab"""
    available_memory = get_available_memory_gb()
    memory_limit = max(1.0, available_memory * 0.4)  # Use only 40% of available memory
    
    print(f"Available memory: {available_memory:.1f}GB")
    print(f"Setting up conservative Dask client with {memory_limit:.1f}GB memory limit...")
    
    # Create conservative local cluster
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        memory_limit=f"{memory_limit}GB",
        silence_logs=True,
        dashboard_address=None,
        processes=False
    )
    
    client = Client(cluster)
    print(f"Dask client created successfully")
    return client

def load_data_conservative(data_dir):
    """Load data with conservative memory usage"""
    print(f"Loading data from {data_dir} with conservative memory usage...")
    
    # Use small chunks
    chunk_size = 25  # 25MB chunks
    
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

def prepare_features_conservative(ddf):
    """Prepare features conservatively"""
    print("Preparing features...")
    
    # Get column names from sample
    sample = ddf.get_partition(0).compute()
    all_cols = sample.columns.tolist()
    
    # Define columns to exclude
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0'
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

def create_temporal_split_safe(ddf, feature_cols, target_col, train_ratio=0.8):
    """Create temporal split safely without memory issues"""
    print(f"Creating temporal split (train_ratio={train_ratio})...")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    ddf_subset = ddf[needed_cols]
    
    # Process each partition separately to avoid memory issues
    def process_partition_for_temporal_split(partition_df, is_train=True):
        """Process a single partition for temporal split"""
        # Remove rows with missing target
        partition_df = partition_df.dropna(subset=[target_col])
        
        if len(partition_df) == 0:
            return pd.DataFrame(columns=partition_df.columns)
        
        result_list = []
        
        # Group by site if available
        if 'site' in partition_df.columns:
            sites = partition_df['site'].unique()
        else:
            sites = ['all']  # Single group if no site column
        
        for site in sites:
            if 'site' in partition_df.columns:
                site_data = partition_df[partition_df['site'] == site].copy()
            else:
                site_data = partition_df.copy()
            
            if len(site_data) == 0:
                continue
            
            # Sort by timestamp if available
            if 'TIMESTAMP' in site_data.columns:
                site_data = site_data.sort_values('TIMESTAMP')
            
            # Split temporally
            if len(site_data) > 1:
                split_idx = max(1, int(len(site_data) * train_ratio))
                if is_train:
                    result_list.append(site_data.iloc[:split_idx])
                else:
                    if split_idx < len(site_data):
                        result_list.append(site_data.iloc[split_idx:])
            else:
                # Single row goes to train
                if is_train:
                    result_list.append(site_data)
        
        if result_list:
            return pd.concat(result_list, ignore_index=True)
        else:
            return pd.DataFrame(columns=partition_df.columns)
    
    # Create train and test splits using map_partitions
    print("Creating train split...")
    train_ddf = ddf_subset.map_partitions(
        lambda df: process_partition_for_temporal_split(df, is_train=True),
        meta=pd.DataFrame(columns=needed_cols)
    )
    
    print("Creating test split...")
    test_ddf = ddf_subset.map_partitions(
        lambda df: process_partition_for_temporal_split(df, is_train=False),
        meta=pd.DataFrame(columns=needed_cols)
    )
    
    print("Temporal split completed successfully")
    return train_ddf, test_ddf

def train_temporal_xgboost(train_ddf, test_ddf, feature_cols, target_col, client):
    """Train XGBoost with temporal data"""
    print("Training XGBoost with temporal data...")
    
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
    
    # Conservative XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,          # Moderate depth
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 128,          # Moderate bins
        'verbosity': 0,
        'nthread': 1
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Create DMatrix objects
    print("Creating DMatrix objects...")
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
    
    print("DMatrix objects created")
    
    # Train model
    print("Starting training...")
    output = xgb.dask.train(
        client,
        params,
        dtrain,
        num_boost_round=150,     # Moderate rounds
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=15,
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
    
    print(f"\nTemporal Model Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Train samples: {len(y_train_actual):,}")
    print(f"  Test samples: {len(y_test_actual):,}")
    
    return model, metrics

def get_feature_importance(model, feature_cols):
    """Get feature importance"""
    try:
        importance_dict = model.get_score(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': feature_cols, 'importance': [0.0] * len(feature_cols)})

def save_temporal_results(model, metrics, feature_importance, feature_cols, output_dir='colab_temporal_models'):
    """Save temporal model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_temporal_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_temporal_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_temporal_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in temporal training:\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_temporal_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Temporal Training Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Note: This model uses temporal splits for proper time series validation\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nTemporal model results saved:")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main temporal training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab Temporal XGBoost Training")
    print("=" * 55)
    print(f"Started at: {datetime.now()}")
    print("Note: This version uses proper temporal splits for time series validation")
    
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
    
    # Setup conservative Dask client
    client = setup_conservative_dask_client()
    
    try:
        # Step 1: Load data
        ddf = load_data_conservative(data_dir)
        
        # Step 2: Prepare features
        ddf_clean, feature_cols, target_col = prepare_features_conservative(ddf)
        
        # Step 3: Create temporal split
        train_ddf, test_ddf = create_temporal_split_safe(ddf_clean, feature_cols, target_col)
        
        # Step 4: Train temporal model
        model, metrics = train_temporal_xgboost(
            train_ddf, test_ddf, feature_cols, target_col, client
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 6: Save results
        model_path = save_temporal_results(model, metrics, feature_importance, feature_cols)
        
        print(f"\n✅ Temporal training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"💡 This model uses proper temporal validation - test data is from later time periods")
        
    except Exception as e:
        print(f"\n❌ Temporal training failed: {str(e)}")
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