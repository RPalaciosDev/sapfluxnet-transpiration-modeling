"""
Google Colab-Safe Spatial Validation XGBoost Training for SAPFLUXNET Data
LEAVE-ONE-SITE-OUT VALIDATION - Tests spatial generalization capability
Implements proper spatial cross-validation for predicting new sites
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

def leave_one_site_out_split_safe(ddf, feature_cols, target_col, max_sites=None):
    """
    Leave-One-Site-Out spatial validation - tests spatial generalization
    Each fold holds out one site for testing, trains on all other sites
    Tests ability to predict sap flow at completely new locations
    """
    print("Creating Leave-One-Site-Out spatial validation splits...")
    print("Using spatial validation method: predicting new sites")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    
    ddf_subset = ddf[needed_cols]
    
    # Remove rows with missing target
    ddf_clean = ddf_subset.dropna(subset=[target_col])
    
    # Convert to pandas for easier site-wise manipulation
    print("Converting to pandas for spatial splitting...")
    df = ddf_clean.compute()
    
    # Get unique sites
    sites = df['site'].unique()
    n_sites = len(sites)
    print(f"Found {n_sites} unique sites for spatial validation")
    
    # Limit number of sites if specified (for memory management)
    if max_sites and n_sites > max_sites:
        print(f"Limiting to {max_sites} sites for memory management...")
        sites = sites[:max_sites]
        df = df[df['site'].isin(sites)]
        n_sites = max_sites
    
    # Check site data distribution
    site_counts = df['site'].value_counts()
    print(f"Site data distribution:")
    print(f"  Mean: {site_counts.mean():.0f} records per site")
    print(f"  Min: {site_counts.min():,} records")
    print(f"  Max: {site_counts.max():,} records")
    
    # Filter out sites with too little data
    min_records = 100  # Minimum records per site
    valid_sites = site_counts[site_counts >= min_records].index.tolist()
    
    if len(valid_sites) < n_sites:
        print(f"Filtered to {len(valid_sites)} sites with ≥{min_records} records")
        sites = valid_sites
        df = df[df['site'].isin(sites)]
        n_sites = len(sites)
    
    # Create Leave-One-Site-Out splits
    spatial_splits = []
    
    print(f"\nCreating {n_sites} Leave-One-Site-Out splits...")
    
    for i, test_site in enumerate(sites):
        # Split data by site
        train_data = df[df['site'] != test_site].copy()
        test_data = df[df['site'] == test_site].copy()
        
        # Remove site column from features (to avoid data leakage)
        if 'site' in train_data.columns:
            train_data = train_data.drop('site', axis=1)
        if 'site' in test_data.columns:
            test_data = test_data.drop('site', axis=1)
        
        print(f"  Split {i+1}/{n_sites}: Test site {test_site}")
        print(f"    Train: {len(train_data):,} records from {n_sites-1} sites")
        print(f"    Test:  {len(test_data):,} records from 1 site ({test_site})")
        
        # Convert to Dask DataFrames with appropriate partitioning
        train_ddf = dd.from_pandas(train_data, npartitions=max(1, len(train_data) // 10000))
        test_ddf = dd.from_pandas(test_data, npartitions=max(1, len(test_data) // 10000))
        
        spatial_splits.append((train_ddf, test_ddf, test_site))
        
        # Show progress every 10 sites
        if (i + 1) % 10 == 0 or i == n_sites - 1:
            print(f"    Progress: {i+1}/{n_sites} splits created")
    
    print(f"\n✅ Leave-One-Site-Out spatial validation completed: {len(spatial_splits)} site splits")
    return spatial_splits

def train_spatial_validation_xgboost(spatial_splits, feature_cols, target_col, client, max_folds=None):
    """Train XGBoost model with Leave-One-Site-Out spatial validation"""
    n_splits = len(spatial_splits)
    
    # Limit number of folds for memory management
    if max_folds and n_splits > max_folds:
        print(f"Limiting to {max_folds} folds for memory management...")
        spatial_splits = spatial_splits[:max_folds]
        n_splits = max_folds
    
    print(f"Training XGBoost with {n_splits}-site spatial validation...")
    
    all_metrics = []
    fold_models = []
    site_results = []
    
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
    
    for fold_idx, (train_ddf, test_ddf, test_site) in enumerate(spatial_splits):
        print(f"\n--- Training Spatial Fold {fold_idx + 1}/{n_splits} (Test Site: {test_site}) ---")
        
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
        
        # Train model for this site fold
        print(f"Training spatial fold {fold_idx + 1} (excluding site {test_site})...")
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
            'test_site': test_site,
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
        site_results.append({
            'site': test_site,
            'test_r2': fold_metrics['test_r2'],
            'test_rmse': fold_metrics['test_rmse'],
            'test_samples': fold_metrics['test_samples']
        })
        
        print(f"Spatial Fold {fold_idx + 1} Results (Site: {test_site}):")
        print(f"  Train R²: {fold_metrics['train_r2']:.4f}")
        print(f"  Test R²: {fold_metrics['test_r2']:.4f} (New site prediction)")
        print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
        print(f"  Samples: {fold_metrics['train_samples']:,} train, {fold_metrics['test_samples']:,} test")
        
        # Memory cleanup
        del X_train, y_train, X_test, y_test, dtrain, dtest
        gc.collect()
    
    # Calculate average metrics across all folds
    avg_metrics = {}
    for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
        values = [fold[metric] for fold in all_metrics]
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
    
    # Site-specific analysis
    site_df = pd.DataFrame(site_results)
    best_sites = site_df.nlargest(5, 'test_r2')
    worst_sites = site_df.nsmallest(5, 'test_r2')
    
    print(f"\n=== Leave-One-Site-Out Spatial Validation Results ===")
    print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
    print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
    print(f"Train R² (mean ± std): {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}")
    print(f"Number of sites tested: {len(spatial_splits)}")
    
    print(f"\nBest Predicted Sites (Top 5):")
    for _, row in best_sites.iterrows():
        print(f"  {row['site']}: R² = {row['test_r2']:.4f}")
    
    print(f"\nWorst Predicted Sites (Bottom 5):")
    for _, row in worst_sites.iterrows():
        print(f"  {row['site']}: R² = {row['test_r2']:.4f}")
    
    return fold_models, all_metrics, avg_metrics, site_results

def get_feature_importance(models, feature_cols):
    """Get average feature importance across all spatial folds"""
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
        
        print(f"\nTop 15 Most Important Features (averaged across spatial folds):")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': feature_cols, 'importance': [0.0] * len(feature_cols)})

def save_spatial_validation_results(models, all_metrics, avg_metrics, feature_importance, site_results, feature_cols, output_dir='colab_spatial_validation_models'):
    """Save spatial validation model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_fold_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_fold_idx]
    best_site = all_metrics[best_fold_idx]['test_site']
    model_path = f"{output_dir}/sapfluxnet_spatial_validation_best_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_spatial_validation_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_spatial_validation_fold_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save site-specific results
    site_results_path = f"{output_dir}/sapfluxnet_spatial_validation_site_results_{timestamp}.csv"
    pd.DataFrame(site_results).to_csv(site_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_spatial_validation_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in spatial validation training:\n")
        f.write("Method: Leave-One-Site-Out cross-validation\n")
        f.write("Split: Train on all sites except one, test on held-out site\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_spatial_validation_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Spatial Validation Training Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Leave-One-Site-Out spatial cross-validation\n")
        f.write("Approach: Predict sap flow at completely new sites\n")
        f.write("Fixed: Handles Dask categorical columns properly\n\n")
        
        f.write("Average Performance Across Sites:\n")
        f.write("-" * 35 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nBest Model: Site {best_site} (Test R²: {all_metrics[best_fold_idx]['test_r2']:.4f})\n")
        
        f.write("\nSite-by-Site Results:\n")
        f.write("-" * 22 + "\n")
        for metrics in all_metrics:
            f.write(f"Site {metrics['test_site']}: Test R² = {metrics['test_r2']:.4f}, Test RMSE = {metrics['test_rmse']:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- Leave-One-Site-Out cross-validation\n")
        f.write("- Each fold excludes one site for testing\n")
        f.write("- Tests model's ability to predict at new locations\n")
        f.write("- Evaluates spatial generalization capability\n")
        f.write("- No spatial data leakage between train/test\n")
        f.write("- Critical for deploying models to new sites\n")
    
    print(f"\nSpatial validation model results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Site results: {site_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main spatial validation training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab Spatial Validation XGBoost Training")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print("Method: Leave-One-Site-Out spatial cross-validation")
    print("Approach: Test ability to predict sap flow at new sites")
    
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
        
        # Step 3: Create spatial validation splits
        print("\n" + "="*50)
        print("CREATING LEAVE-ONE-SITE-OUT SPLITS")
        print("="*50)
        spatial_splits = leave_one_site_out_split_safe(ddf_clean, feature_cols, target_col, max_sites=20)
        
        # Step 4: Train spatial validation models
        print("\n" + "="*50)
        print("TRAINING SPATIAL VALIDATION MODELS")
        print("="*50)
        models, all_metrics, avg_metrics, site_results = train_spatial_validation_xgboost(
            spatial_splits, feature_cols, target_col, client, max_folds=20
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(models, feature_cols)
        
        # Step 6: Save results
        model_path = save_spatial_validation_results(models, all_metrics, avg_metrics, feature_importance, site_results, feature_cols)
        
        print(f"\n✅ Spatial validation training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model tests spatial generalization - predicting sap flow at new sites")
        print(f"🔧 Fixed: Dask categorical columns handled properly")
        print(f"📊 Method: Leave-One-Site-Out spatial cross-validation")
        print(f"🎯 Sites tested: {len(spatial_splits)} different locations")
        
    except Exception as e:
        print(f"\n❌ Spatial validation training failed: {str(e)}")
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