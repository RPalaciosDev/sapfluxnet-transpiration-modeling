"""
Google Colab-Safe Rolling Window XGBoost Training for SAPFLUXNET Data
ROLLING WINDOW VALIDATION - Tests short-term forecasting and seasonal patterns
Implements time series forecasting validation with seasonal analysis
"""

import dask.dataframe as dd
import dask.array as da
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime, timedelta
import warnings
import gc
from dask.distributed import Client, LocalCluster
import psutil
import calendar
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

def rolling_window_split_safe(ddf, feature_cols, target_col, window_size_days=30, forecast_horizon_days=7, n_windows=10):
    """
    Rolling window validation - tests short-term forecasting capability
    Creates multiple time windows that slide forward through the time series
    Each window trains on historical data and predicts future periods
    """
    print(f"Creating rolling window validation splits...")
    print(f"Window size: {window_size_days} days, Forecast horizon: {forecast_horizon_days} days")
    print("Using rolling window method: short-term forecasting with seasonal analysis")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in ddf.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in ddf.columns:
        needed_cols.append('TIMESTAMP')
    
    ddf_subset = ddf[needed_cols]
    
    # Remove rows with missing target
    ddf_clean = ddf_subset.dropna(subset=[target_col])
    
    # Convert to pandas for easier temporal manipulation
    print("Converting to pandas for rolling window splitting...")
    df = ddf_clean.compute()
    
    # Ensure TIMESTAMP is datetime
    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
        print("Data sorted by timestamp")
    else:
        raise ValueError("TIMESTAMP column required for rolling window validation")
    
    # Add temporal features for seasonal analysis
    df['month'] = df['TIMESTAMP'].dt.month
    df['day_of_year'] = df['TIMESTAMP'].dt.dayofyear
    df['season'] = df['month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    
    # Get time range
    start_date = df['TIMESTAMP'].min()
    end_date = df['TIMESTAMP'].max()
    total_days = (end_date - start_date).days
    
    print(f"Data time range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    
    # Calculate window spacing to get desired number of windows
    min_window_spacing = window_size_days + forecast_horizon_days
    available_span = total_days - min_window_spacing
    
    if available_span <= 0:
        raise ValueError(f"Insufficient data for rolling windows. Need at least {min_window_spacing} days.")
    
    if n_windows > 1:
        window_spacing = available_span // (n_windows - 1)
    else:
        window_spacing = 0
    
    window_spacing = max(window_spacing, min_window_spacing)
    
    print(f"Window spacing: {window_spacing} days")
    
    # Create rolling window splits
    rolling_splits = []
    seasonal_info = []
    
    print(f"\nCreating {n_windows} rolling window splits...")
    
    for i in range(n_windows):
        # Calculate window start date
        window_start_date = start_date + timedelta(days=i * window_spacing)
        train_end_date = window_start_date + timedelta(days=window_size_days)
        test_start_date = train_end_date
        test_end_date = test_start_date + timedelta(days=forecast_horizon_days)
        
        # Check if we have enough data for this window
        if test_end_date > end_date:
            print(f"  Window {i+1}: Skipped (insufficient future data)")
            break
        
        # Filter data for this window
        train_mask = (df['TIMESTAMP'] >= window_start_date) & (df['TIMESTAMP'] < train_end_date)
        test_mask = (df['TIMESTAMP'] >= test_start_date) & (df['TIMESTAMP'] < test_end_date)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        if len(train_data) < 100 or len(test_data) < 10:
            print(f"  Window {i+1}: Skipped (insufficient data)")
            continue
        
        # Remove temporal columns that shouldn't be features
        feature_cols_clean = [col for col in feature_cols if col in train_data.columns]
        if 'site' in train_data.columns and 'site' not in feature_cols_clean:
            train_data = train_data.drop('site', axis=1)
            test_data = test_data.drop('site', axis=1)
        
        # Seasonal analysis
        train_season = train_data['season'].mode()[0] if len(train_data) > 0 else 0
        test_season = test_data['season'].mode()[0] if len(test_data) > 0 else 0
        train_month = train_data['month'].mode()[0] if len(train_data) > 0 else 0
        test_month = test_data['month'].mode()[0] if len(test_data) > 0 else 0
        
        # Remove temporal analysis columns from features
        temp_cols = ['TIMESTAMP', 'month', 'day_of_year', 'season']
        for col in temp_cols:
            if col in train_data.columns:
                train_data = train_data.drop(col, axis=1)
            if col in test_data.columns:
                test_data = test_data.drop(col, axis=1)
        
        print(f"  Window {i+1}: Train {window_start_date.date()} to {train_end_date.date()}")
        print(f"           Test {test_start_date.date()} to {test_end_date.date()}")
        print(f"           Train: {len(train_data):,} records (Season {train_season}, Month {train_month})")
        print(f"           Test:  {len(test_data):,} records (Season {test_season}, Month {test_month})")
        
        # Convert to Dask DataFrames
        train_ddf = dd.from_pandas(train_data, npartitions=max(1, len(train_data) // 5000))
        test_ddf = dd.from_pandas(test_data, npartitions=max(1, len(test_data) // 1000))
        
        # Store seasonal information
        seasonal_info.append({
            'window': i + 1,
            'train_start': window_start_date,
            'train_end': train_end_date,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'train_season': train_season,
            'test_season': test_season,
            'train_month': train_month,
            'test_month': test_month,
            'season_names': {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'},
            'is_seasonal_transition': train_season != test_season
        })
        
        rolling_splits.append((train_ddf, test_ddf, i + 1))
    
    print(f"\n✅ Rolling window validation completed: {len(rolling_splits)} valid windows")
    
    # Seasonal summary
    seasons_tested = [info['test_season'] for info in seasonal_info]
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    print(f"\nSeasonal coverage:")
    for season in [1, 2, 3, 4]:
        count = seasons_tested.count(season)
        print(f"  {season_names[season]}: {count} windows")
    
    transitions = sum([info['is_seasonal_transition'] for info in seasonal_info])
    print(f"  Seasonal transitions: {transitions} windows")
    
    return rolling_splits, seasonal_info

def train_rolling_window_xgboost(rolling_splits, seasonal_info, feature_cols, target_col, client):
    """Train XGBoost model with rolling window validation"""
    n_windows = len(rolling_splits)
    print(f"Training XGBoost with {n_windows}-window rolling validation...")
    
    all_metrics = []
    window_models = []
    seasonal_results = []
    
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
    
    for window_idx, (train_ddf, test_ddf, window_num) in enumerate(rolling_splits):
        print(f"\n--- Training Rolling Window {window_idx + 1}/{n_windows} ---")
        
        # Get seasonal info for this window
        season_info = seasonal_info[window_idx]
        
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
        
        # Train model for this window
        print(f"Training window {window_idx + 1} ({season_info['train_start'].date()} → {season_info['test_end'].date()})...")
        output = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=100,  # Fewer rounds for rolling window
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        model = output['booster']
        window_models.append(model)
        
        # Make predictions
        print("Making predictions...")
        y_pred_train = xgb.dask.predict(client, model, dtrain)
        y_pred_test = xgb.dask.predict(client, model, dtest)
        
        # Compute metrics
        y_train_actual = y_train.compute()
        y_test_actual = y_test.compute()
        y_pred_train_actual = y_pred_train.compute()
        y_pred_test_actual = y_pred_test.compute()
        
        # Calculate metrics for this window
        window_metrics = {
            'window': window_num,
            'train_start': season_info['train_start'],
            'test_start': season_info['test_start'],
            'train_season': season_info['train_season'],
            'test_season': season_info['test_season'],
            'train_month': season_info['train_month'],
            'test_month': season_info['test_month'],
            'is_seasonal_transition': season_info['is_seasonal_transition'],
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual)),
            'train_mae': mean_absolute_error(y_train_actual, y_pred_train_actual),
            'test_mae': mean_absolute_error(y_test_actual, y_pred_test_actual),
            'train_r2': r2_score(y_train_actual, y_pred_train_actual),
            'test_r2': r2_score(y_test_actual, y_pred_test_actual),
            'train_samples': len(y_train_actual),
            'test_samples': len(y_test_actual)
        }
        
        all_metrics.append(window_metrics)
        
        # Seasonal-specific results
        season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        seasonal_results.append({
            'window': window_num,
            'test_season_name': season_names[season_info['test_season']],
            'test_season': season_info['test_season'],
            'test_month': season_info['test_month'],
            'test_r2': window_metrics['test_r2'],
            'test_rmse': window_metrics['test_rmse'],
            'is_transition': season_info['is_seasonal_transition']
        })
        
        print(f"Window {window_num} Results:")
        print(f"  Train R²: {window_metrics['train_r2']:.4f}")
        print(f"  Test R²: {window_metrics['test_r2']:.4f} (Forecast: {season_names[season_info['test_season']]})")
        print(f"  Test RMSE: {window_metrics['test_rmse']:.4f}")
        print(f"  Seasonal transition: {season_info['is_seasonal_transition']}")
        print(f"  Samples: {window_metrics['train_samples']:,} train, {window_metrics['test_samples']:,} test")
        
        # Memory cleanup
        del X_train, y_train, X_test, y_test, dtrain, dtest
        gc.collect()
    
    # Calculate overall metrics
    avg_metrics = {}
    for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
        values = [window[metric] for window in all_metrics]
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
    
    # Seasonal analysis
    seasonal_df = pd.DataFrame(seasonal_results)
    season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    
    print(f"\n=== Rolling Window Forecasting Results ===")
    print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
    print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
    print(f"Number of forecast windows: {len(rolling_splits)}")
    
    print(f"\n=== Seasonal Forecasting Performance ===")
    for season in [1, 2, 3, 4]:
        season_data = seasonal_df[seasonal_df['test_season'] == season]
        if len(season_data) > 0:
            mean_r2 = season_data['test_r2'].mean()
            std_r2 = season_data['test_r2'].std()
            count = len(season_data)
            print(f"{season_names[season]}: R² = {mean_r2:.4f} ± {std_r2:.4f} ({count} windows)")
    
    # Seasonal transition analysis
    transition_data = seasonal_df[seasonal_df['is_transition'] == True]
    non_transition_data = seasonal_df[seasonal_df['is_transition'] == False]
    
    if len(transition_data) > 0 and len(non_transition_data) > 0:
        print(f"\n=== Seasonal Transition Analysis ===")
        print(f"Seasonal transitions: R² = {transition_data['test_r2'].mean():.4f} ± {transition_data['test_r2'].std():.4f} ({len(transition_data)} windows)")
        print(f"Within-season: R² = {non_transition_data['test_r2'].mean():.4f} ± {non_transition_data['test_r2'].std():.4f} ({len(non_transition_data)} windows)")
    
    return window_models, all_metrics, avg_metrics, seasonal_results

def get_feature_importance(models, feature_cols):
    """Get average feature importance across all rolling windows"""
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
        
        print(f"\nTop 15 Most Important Features (averaged across windows):")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': feature_cols, 'importance': [0.0] * len(feature_cols)})

def save_rolling_window_results(models, all_metrics, avg_metrics, feature_importance, seasonal_results, feature_cols, output_dir='colab_rolling_window_models'):
    """Save rolling window model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_window_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_window_idx]
    best_window = all_metrics[best_window_idx]['window']
    model_path = f"{output_dir}/sapfluxnet_rolling_window_best_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_rolling_window_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed window results
    window_results_path = f"{output_dir}/sapfluxnet_rolling_window_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(window_results_path, index=False)
    
    # Save seasonal analysis results
    seasonal_results_path = f"{output_dir}/sapfluxnet_rolling_window_seasonal_{timestamp}.csv"
    pd.DataFrame(seasonal_results).to_csv(seasonal_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_rolling_window_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in rolling window training:\n")
        f.write("Method: Rolling window time series validation\n")
        f.write("Split: Historical data → Future forecast periods\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_rolling_window_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Rolling Window Training Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Rolling window time series validation\n")
        f.write("Approach: Short-term forecasting with seasonal analysis\n")
        f.write("Fixed: Handles Dask categorical columns properly\n\n")
        
        f.write("Average Performance Across Windows:\n")
        f.write("-" * 35 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nBest Model: Window {best_window} (Test R²: {all_metrics[best_window_idx]['test_r2']:.4f})\n")
        
        f.write("\nWindow-by-Window Results:\n")
        f.write("-" * 25 + "\n")
        for metrics in all_metrics:
            season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
            season_name = season_names[metrics['test_season']]
            f.write(f"Window {metrics['window']}: Test R² = {metrics['test_r2']:.4f}, Season = {season_name}, Transition = {metrics['is_seasonal_transition']}\n")
        
        f.write("\nSeasonal Performance Analysis:\n")
        f.write("-" * 30 + "\n")
        seasonal_df = pd.DataFrame(seasonal_results)
        season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        for season in [1, 2, 3, 4]:
            season_data = seasonal_df[seasonal_df['test_season'] == season]
            if len(season_data) > 0:
                mean_r2 = season_data['test_r2'].mean()
                std_r2 = season_data['test_r2'].std()
                count = len(season_data)
                f.write(f"{season_names[season]}: R² = {mean_r2:.4f} ± {std_r2:.4f} ({count} windows)\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- Rolling window time series validation\n")
        f.write("- Multiple forecast windows sliding through time\n")
        f.write("- Tests short-term forecasting capability\n")
        f.write("- Reveals seasonal forecasting patterns\n")
        f.write("- Simulates operational forecasting scenarios\n")
        f.write("- Evaluates model performance across seasons\n")
    
    print(f"\nRolling window model results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Window results: {window_results_path}")
    print(f"  Seasonal analysis: {seasonal_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main rolling window training pipeline for Google Colab"""
    print("SAPFLUXNET Google Colab Rolling Window XGBoost Training")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print("Method: Rolling window time series validation")
    print("Approach: Short-term forecasting with seasonal pattern analysis")
    
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
        
        # Step 3: Create rolling window splits
        print("\n" + "="*50)
        print("CREATING ROLLING WINDOW SPLITS")
        print("="*50)
        rolling_splits, seasonal_info = rolling_window_split_safe(
            ddf_clean, feature_cols, target_col, 
            window_size_days=30, forecast_horizon_days=7, n_windows=12
        )
        
        # Step 4: Train rolling window models
        print("\n" + "="*50)
        print("TRAINING ROLLING WINDOW MODELS")
        print("="*50)
        models, all_metrics, avg_metrics, seasonal_results = train_rolling_window_xgboost(
            rolling_splits, seasonal_info, feature_cols, target_col, client
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(models, feature_cols)
        
        # Step 6: Save results
        model_path = save_rolling_window_results(models, all_metrics, avg_metrics, feature_importance, seasonal_results, feature_cols)
        
        print(f"\n✅ Rolling window training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model tests short-term forecasting - predicting sap flow days/weeks ahead")
        print(f"🔧 Fixed: Dask categorical columns handled properly")
        print(f"📊 Method: Rolling window time series validation")
        print(f"🎯 Windows tested: {len(rolling_splits)} forecast periods")
        print(f"🌱 Seasonal analysis: Performance across seasons and transitions")
        
    except Exception as e:
        print(f"\n❌ Rolling window training failed: {str(e)}")
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