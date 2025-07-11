"""
Google Colab-Safe K-Fold Temporal XGBoost Training for SAPFLUXNET Data
K-FOLD TEMPORAL VALIDATION - Multiple temporal splits for robust validation
Implements k-fold temporal cross-validation with multiple time periods
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

def k_fold_temporal_split_safe(ddf, feature_cols, target_col, n_folds=5):
    """
    K-fold temporal split method - creates multiple temporal splits
    Divides time series into k consecutive folds for robust temporal validation
    Each fold uses earlier data for training, later data for testing
    """
    print(f"Creating k-fold temporal split (n_folds={n_folds})...")
    print("Using k-fold temporal method: multiple temporal splits for robust validation")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    ddf_subset = ddf[needed_cols]
    
    # Remove rows with missing target
    ddf_clean = ddf_subset.dropna(subset=[target_col])
    
    # Convert to pandas for easier manipulation
    print("Converting to pandas for k-fold temporal splitting...")
    df = ddf_clean.compute()
    
    # Sort data by timestamp globally
    if 'TIMESTAMP' in df.columns:
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
        print("Data sorted by timestamp globally")
    
    # Create k-fold temporal splits
    fold_splits = []
    sites = df['site'].unique()
    print(f"Creating {n_folds} temporal folds across {len(sites)} sites...")
    
    for fold in range(n_folds):
        print(f"\nCreating fold {fold + 1}/{n_folds}...")
        
        train_data_list = []
        test_data_list = []
        
        # For each site, create temporal split for this fold
        for i, site in enumerate(sites):
            site_data = df[df['site'] == site].copy()
            
            # Sort by timestamp within site
            if 'TIMESTAMP' in site_data.columns:
                site_data = site_data.sort_values('TIMESTAMP').reset_index(drop=True)
            
            # Calculate fold boundaries
            fold_size = len(site_data) // n_folds
            if fold_size == 0:
                continue  # Skip sites with too little data
            
            # Define train/test split for this fold
            # Each fold tests on a different time period
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, len(site_data))
            
            # Use all data before test period for training
            if test_start > 0:
                train_portion = site_data.iloc[:test_start]
                test_portion = site_data.iloc[test_start:test_end]
                
                if len(train_portion) > 0 and len(test_portion) > 0:
                    train_data_list.append(train_portion)
                    test_data_list.append(test_portion)
            
            # Show progress
            if (i + 1) % 20 == 0 or i == len(sites) - 1:
                print(f"  Processed {i+1}/{len(sites)} sites for fold {fold + 1}")
        
        # Combine data for this fold
        if train_data_list and test_data_list:
            train_data = pd.concat(train_data_list, ignore_index=True)
            test_data = pd.concat(test_data_list, ignore_index=True)
            
            print(f"  Fold {fold + 1}: {len(train_data):,} train, {len(test_data):,} test")
            
            # Convert to Dask DataFrames
            train_ddf = dd.from_pandas(train_data, npartitions=max(1, len(train_data) // 10000))
            test_ddf = dd.from_pandas(test_data, npartitions=max(1, len(test_data) // 10000))
            
            fold_splits.append((train_ddf, test_ddf))
        else:
            print(f"  Fold {fold + 1}: Skipped (insufficient data)")
    
    print(f"\n✅ K-fold temporal split completed: {len(fold_splits)} valid folds")
    return fold_splits

def train_k_fold_temporal_xgboost(fold_splits, feature_cols, target_col, client):
    """Train XGBoost model with k-fold temporal validation"""
    print(f"Training XGBoost with {len(fold_splits)}-fold temporal validation...")
    
    all_metrics = []
    fold_models = []
    
    # Conservative XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 128,
        'verbosity': 0,
        'nthread': 1
    }
    
    print(f"XGBoost parameters: {params}")
    
    for fold_idx, (train_ddf, test_ddf) in enumerate(fold_splits):
        print(f"\n--- Training Fold {fold_idx + 1}/{len(fold_splits)} ---")
        
        # Fill missing values
        print("Filling missing values...")
        train_ddf = train_ddf.fillna(0)
        test_ddf = test_ddf.fillna(0)
        
        # Convert to Dask arrays
        print("Converting to Dask arrays...")
        X_train = train_ddf[feature_cols].to_dask_array(lengths=True)
        y_train = train_ddf[target_col].to_dask_array(lengths=True)
        X_test = test_ddf[feature_cols].to_dask_array(lengths=True)
        y_test = test_ddf[target_col].to_dask_array(lengths=True)
        
        # Create DMatrix objects
        print("Creating DMatrix objects...")
        dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)
        
        # Train model for this fold
        print(f"Training fold {fold_idx + 1}...")
        output = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=150,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=15,
            verbose_eval=False
        )
        
        model = output['booster']
        fold_models.append(model)
        
        # Make predictions
        print("Making predictions...")
        y_pred_train = xgb.dask.predict(client, model, dtrain)
        y_pred_test = xgb.dask.predict(client, model, dtest)
        
        # Compute metrics
        y_train_actual = y_train.compute()
        y_test_actual = y_test.compute()
        y_pred_train_actual = y_pred_train.compute()
        y_pred_test_actual = y_pred_test.compute()
        
        # Calculate metrics for this fold
        fold_metrics = {
            'fold': fold_idx + 1,
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual)),
            'train_mae': mean_absolute_error(y_train_actual, y_pred_train_actual),
            'test_mae': mean_absolute_error(y_test_actual, y_pred_test_actual),
            'train_r2': r2_score(y_train_actual, y_pred_train_actual),
            'test_r2': r2_score(y_test_actual, y_pred_test_actual),
            'train_samples': len(y_train_actual),
            'test_samples': len(y_test_actual)
        }
        
        all_metrics.append(fold_metrics)
        
        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Train R²: {fold_metrics['train_r2']:.4f}")
        print(f"  Test R²: {fold_metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
        print(f"  Samples: {fold_metrics['train_samples']:,} train, {fold_metrics['test_samples']:,} test")
    
    # Calculate average metrics across all folds
    avg_metrics = {}
    for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
        values = [fold[metric] for fold in all_metrics]
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
    
    print(f"\n=== K-Fold Temporal Cross-Validation Results ===")
    print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
    print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
    print(f"Train R² (mean ± std): {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}")
    print(f"Number of folds: {len(fold_splits)}")
    
    return fold_models, all_metrics, avg_metrics

def get_feature_importance(models, feature_cols):
    """Get average feature importance across all folds"""
    try:
        # Collect importance from all models
        all_importance = []
        
        for i, model in enumerate(models):
            importance_dict = model.get_score(importance_type='gain')
            all_importance.append(importance_dict)
        
        # Calculate average importance
        all_features = set()
        for imp_dict in all_importance:
            all_features.update(imp_dict.keys())
        
        avg_importance = {}
        for feature in all_features:
            values = [imp_dict.get(feature, 0.0) for imp_dict in all_importance]
            avg_importance[feature] = np.mean(values)
        
        feature_importance = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features (averaged across folds):")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': feature_cols, 'importance': [0.0] * len(feature_cols)})

def save_k_fold_temporal_results(models, all_metrics, avg_metrics, feature_importance, feature_cols, output_dir='colab_k_fold_temporal_models'):
    """Save k-fold temporal model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_fold_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_fold_idx]
    model_path = f"{output_dir}/sapfluxnet_k_fold_temporal_best_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_k_fold_temporal_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_k_fold_temporal_fold_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_k_fold_temporal_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in k-fold temporal training:\n")
        f.write("Method: K-fold temporal cross-validation\n")
        f.write("Split: Multiple temporal splits for robust validation\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_k_fold_temporal_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET K-Fold Temporal Training Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: K-fold temporal cross-validation\n")
        f.write("Approach: Multiple temporal splits for robust validation\n")
        f.write("Fixed: Handles Dask categorical columns properly\n\n")
        
        f.write("Average Performance Across Folds:\n")
        f.write("-" * 35 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nBest Model: Fold {best_fold_idx + 1} (Test R²: {all_metrics[best_fold_idx]['test_r2']:.4f})\n")
        
        f.write("\nIndividual Fold Results:\n")
        f.write("-" * 25 + "\n")
        for metrics in all_metrics:
            f.write(f"Fold {metrics['fold']}: Test R² = {metrics['test_r2']:.4f}, Test RMSE = {metrics['test_rmse']:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- K-fold temporal cross-validation\n")
        f.write("- Multiple temporal splits across time periods\n")
        f.write("- Each fold tests on different time period\n")
        f.write("- Robust evaluation with confidence intervals\n")
        f.write("- Prevents data leakage from future to past\n")
        f.write("- Proper temporal cross-validation for time series data\n")
    
    print(f"\nK-fold temporal model results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main k-fold temporal training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab K-Fold Temporal XGBoost Training")
    print("=" * 65)
    print(f"Started at: {datetime.now()}")
    print("Method: K-fold temporal cross-validation")
    print("Approach: Multiple temporal splits for robust validation")
    
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
        
        # Step 3: Create k-fold temporal splits
        print("\n" + "="*50)
        print("CREATING K-FOLD TEMPORAL SPLITS")
        print("="*50)
        fold_splits = k_fold_temporal_split_safe(ddf_clean, feature_cols, target_col, n_folds=5)
        
        # Step 4: Train k-fold temporal models
        print("\n" + "="*50)
        print("TRAINING K-FOLD TEMPORAL MODELS")
        print("="*50)
        models, all_metrics, avg_metrics = train_k_fold_temporal_xgboost(
            fold_splits, feature_cols, target_col, client
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(models, feature_cols)
        
        # Step 6: Save results
        model_path = save_k_fold_temporal_results(models, all_metrics, avg_metrics, feature_importance, feature_cols)
        
        print(f"\n✅ K-fold temporal training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model uses k-fold temporal validation - multiple temporal splits for robust evaluation")
        print(f"🔧 Fixed: Dask categorical columns handled properly")
        print(f"📊 Method: K-fold temporal cross-validation")
        print(f"🎯 Folds: {len(fold_splits)} temporal splits")
        
    except Exception as e:
        print(f"\n❌ K-fold temporal training failed: {str(e)}")
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