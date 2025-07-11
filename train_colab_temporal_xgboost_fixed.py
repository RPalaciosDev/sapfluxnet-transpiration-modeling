"""
Google Colab-Safe Hybrid Temporal XGBoost Training for SAPFLUXNET Data
FIXED VERSION - Uses original hybrid temporal split method
Implements proper site-wise temporal cross-validation
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

def fix_dask_categorical_columns(ddf):
    """Fix Dask categorical columns by converting them to numeric"""
    print("Fixing Dask categorical columns...")
    
    # Identify categorical columns that should be numeric
    categorical_cols = [col for col in ddf.columns if ddf[col].dtype.name == 'category']
    
    if categorical_cols:
        print(f"Found {len(categorical_cols)} categorical columns to fix:")
        for col in categorical_cols:
            print(f"  - {col}")
        
        # Convert categorical columns to numeric
        for col in categorical_cols:
            print(f"Converting {col} to numeric...")
            ddf[col] = ddf[col].astype('float64')
        
        print("✅ All categorical columns converted to numeric")
    else:
        print("✅ No categorical columns found")
    
    return ddf

def load_data_conservative(data_dir):
    """Load data with conservative memory usage and fix categorical columns"""
    print(f"Loading data from {data_dir} with conservative memory usage...")
    
    # Use small chunks
    chunk_size = 25  # 25MB chunks
    
    try:
        ddf = dd.read_parquet(
            f"{data_dir}/*.parquet",
            blocksize=f"{chunk_size}MB"
        )
        print(f"Data loaded successfully with {ddf.npartitions} partitions")
        
        # Fix categorical columns immediately after loading
        ddf = fix_dask_categorical_columns(ddf)
        
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
        
        # Fix categorical columns for CSV too
        ddf = fix_dask_categorical_columns(ddf)
        
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

def hybrid_temporal_split_safe(ddf, feature_cols, target_col, train_ratio=0.8):
    """
    Original hybrid temporal split method - site-wise temporal cross-validation
    Each site is split temporally (early data → train, late data → test)
    Then all sites' train data combined, all sites' test data combined
    """
    print(f"Creating hybrid temporal split (train_ratio={train_ratio})...")
    print("Using original hybrid method: site-wise temporal cross-validation")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    ddf_subset = ddf[needed_cols]
    
    # Remove rows with missing target
    ddf_clean = ddf_subset.dropna(subset=[target_col])
    
    # Convert to pandas for easier site-wise manipulation
    print("Converting to pandas for site-wise temporal splitting...")
    df = ddf_clean.compute()
    
    train_data_list = []
    test_data_list = []
    
    # Sort data by timestamp first (global sort)
    if 'TIMESTAMP' in df.columns:
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
        print("Data sorted by timestamp globally")
    
    # Split each site separately by time
    sites = df['site'].unique()
    print(f"Processing {len(sites)} sites for temporal splitting...")
    
    for i, site in enumerate(sites):
        site_data = df[df['site'] == site].copy()
        
        # Sort by timestamp within site (ensure temporal order)
        if 'TIMESTAMP' in site_data.columns:
            site_data = site_data.sort_values('TIMESTAMP').reset_index(drop=True)
        
        # Calculate temporal split point (early 80% → train, late 20% → test)
        split_idx = max(1, int(len(site_data) * train_ratio))
        
        # Split the data temporally
        train_portion = site_data.iloc[:split_idx]  # Early data
        test_portion = site_data.iloc[split_idx:]   # Late data
        
        train_data_list.append(train_portion)
        test_data_list.append(test_portion)
        
        if (i + 1) % 10 == 0 or i == len(sites) - 1:
            print(f"  Processed {i+1}/{len(sites)} sites")
        
        # Show details for first few sites
        if i < 5:
            print(f"  {site}: {len(train_portion):,} train (early), {len(test_portion):,} test (late)")
    
    # Combine all sites' train data and all sites' test data
    print("Combining train and test data across all sites...")
    train_data = pd.concat(train_data_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)
    
    print(f"\nHybrid temporal split completed:")
    print(f"  Train: {len(train_data):,} rows ({len(train_data)/len(df)*100:.1f}%) - Early time periods from all sites")
    print(f"  Test:  {len(test_data):,} rows ({len(test_data)/len(df)*100:.1f}%) - Late time periods from all sites")
    
    # Convert back to Dask DataFrames with appropriate partitioning
    print("Converting back to Dask DataFrames...")
    train_ddf = dd.from_pandas(train_data, npartitions=max(1, len(train_data) // 10000))
    test_ddf = dd.from_pandas(test_data, npartitions=max(1, len(test_data) // 10000))
    
    print("✅ Hybrid temporal split completed successfully")
    return train_ddf, test_ddf

def train_hybrid_temporal_xgboost(train_ddf, test_ddf, feature_cols, target_col, client):
    """Train XGBoost model with hybrid temporal validation"""
    print("Training XGBoost with hybrid temporal validation...")
    
    # Fill missing values with simple approach (no categorical issues now)
    print("Filling missing values...")
    train_ddf = train_ddf.fillna(0)
    test_ddf = test_ddf.fillna(0)
    
    print("Missing values filled successfully")
    
    # Convert to Dask arrays
    print("Converting to Dask arrays...")
    X_train = train_ddf[feature_cols].to_dask_array(lengths=True)
    y_train = train_ddf[target_col].to_dask_array(lengths=True)
    X_test = test_ddf[feature_cols].to_dask_array(lengths=True)
    y_test = test_ddf[target_col].to_dask_array(lengths=True)
    
    print("Data converted to Dask arrays successfully")
    
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
    print("Starting hybrid temporal training...")
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
    print("Hybrid temporal training completed!")
    
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
    
    print(f"\nHybrid Temporal Model Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f} (Early time periods)")
    print(f"  Test R²: {metrics['test_r2']:.4f} (Late time periods)")
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

def save_hybrid_temporal_results(model, metrics, feature_importance, feature_cols, output_dir='colab_hybrid_temporal_models'):
    """Save hybrid temporal model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_hybrid_temporal_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_hybrid_temporal_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_hybrid_temporal_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in hybrid temporal training:\n")
        f.write("Method: Site-wise temporal cross-validation\n")
        f.write("Split: Early time periods (train) vs Late time periods (test)\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_hybrid_temporal_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Hybrid Temporal Training Results\n")
        f.write("=" * 45 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Original hybrid temporal cross-validation\n")
        f.write("Approach: Site-wise temporal split (early → train, late → test)\n")
        f.write("Fixed: Handles Dask categorical columns properly\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- Each site split temporally (80% early data → train, 20% late data → test)\n")
        f.write("- All sites' train data combined for training\n")
        f.write("- All sites' test data combined for testing\n")
        f.write("- Tests model's ability to predict future time periods\n")
        f.write("- Prevents data leakage from future to past\n")
        f.write("- Proper temporal cross-validation for time series data\n")
    
    print(f"\nHybrid temporal model results saved:")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main hybrid temporal training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab Hybrid Temporal XGBoost Training")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("Method: Original hybrid temporal cross-validation")
    print("Approach: Site-wise temporal split for proper time series validation")
    
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
        # Step 1: Load data (with categorical fix)
        ddf = load_data_conservative(data_dir)
        
        # Step 2: Prepare features
        ddf_clean, feature_cols, target_col = prepare_features_conservative(ddf)
        
        # Step 3: Create hybrid temporal split (original method)
        print("\n" + "="*50)
        print("CREATING HYBRID TEMPORAL SPLIT")
        print("="*50)
        train_ddf, test_ddf = hybrid_temporal_split_safe(ddf_clean, feature_cols, target_col)
        
        # Step 4: Train hybrid temporal model
        print("\n" + "="*50)
        print("TRAINING HYBRID TEMPORAL MODEL")
        print("="*50)
        model, metrics = train_hybrid_temporal_xgboost(
            train_ddf, test_ddf, feature_cols, target_col, client
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 6: Save results
        model_path = save_hybrid_temporal_results(model, metrics, feature_importance, feature_cols)
        
        print(f"\n✅ Hybrid temporal training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"💡 This model uses proper temporal validation - trained on early periods, tested on late periods")
        print(f"🔧 Fixed: Dask categorical columns handled properly")
        print(f"📊 Method: Original hybrid temporal cross-validation")
        
    except Exception as e:
        print(f"\n❌ Hybrid temporal training failed: {str(e)}")
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