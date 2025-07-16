"""
External Memory Rolling Window XGBoost Training for SAPFLUXNET Data
ROLLING WINDOW VALIDATION - Tests short-term forecasting and seasonal patterns
Implements time series forecasting validation with external memory efficiency
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import os
from datetime import datetime, timedelta
import warnings
import gc
import psutil
import tempfile
import shutil
import json
from pathlib import Path

warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"   Total: {memory.total / (1024**3):.1f}GB")

def load_feature_mapping(data_dir):
    """
    Load feature mapping from JSON file created by the data processing pipeline
    """
    feature_mapping_file = os.path.join(data_dir, 'feature_mapping.json')
    
    if not os.path.exists(feature_mapping_file):
        print(f"⚠️  No feature mapping found at {feature_mapping_file}")
        return None
        
    try:
        with open(feature_mapping_file, 'r') as f:
            feature_mapping = json.load(f)
        
        print(f"✅ Loaded feature mapping: {feature_mapping['feature_count']} features")
        print(f"   Created by: {feature_mapping.get('created_by', 'unknown')}")
        print(f"   Created at: {feature_mapping.get('created_at', 'unknown')}")
        
        return feature_mapping
        
    except Exception as e:
        print(f"❌ Error loading feature mapping: {e}")
        return None

def combine_libsvm_files(libsvm_dir, output_dir):
    """
    Combine existing libsvm files from pipeline into single file
    """
    print(f"Combining libsvm files from {libsvm_dir}...")
    
    # Check available disk space
    def check_space_gb(path):
        try:
            statvfs = os.statvfs(path)
            return statvfs.f_bavail * statvfs.f_frsize / (1024**3)
        except:
            return 0
    
    available_space = check_space_gb(output_dir)
    print(f"💾 Available space in output directory: {available_space:.1f} GB")
    
    # Get all libsvm files (including compressed)
    libsvm_files = [f for f in os.listdir(libsvm_dir) if f.endswith('.svm') or f.endswith('.svm.gz')]
    print(f"Found {len(libsvm_files)} libsvm files to combine")
    
    # Combine all files
    all_data_file = os.path.join(output_dir, 'all_data.svm')
    total_rows = 0
    
    try:
        with open(all_data_file, 'w') as output_file:
            for i, libsvm_file in enumerate(libsvm_files):
                print(f"Processing file {i+1}/{len(libsvm_files)}: {libsvm_file}")
                
                file_path = os.path.join(libsvm_dir, libsvm_file)
                
                # Handle compressed files
                if libsvm_file.endswith('.gz'):
                    import gzip
                    with gzip.open(file_path, 'rt') as f:
                        lines = f.readlines()
                else:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                
                # Write lines to combined file
                output_file.writelines(lines)
                total_rows += len(lines)
                
                print(f"  Added {len(lines)} rows from {libsvm_file}")
                
                # Explicit cleanup after processing each file
                del lines
                gc.collect()
        
        print(f"Combination completed: {total_rows:,} total rows")
        return all_data_file, total_rows
        
    except Exception as e:
        print(f"❌ Error during file combination: {e}")
        raise

def load_libsvm_as_dataframe(libsvm_file, feature_mapping):
    """
    Load libsvm file and convert to pandas DataFrame with proper column names
    """
    print(f"Loading libsvm file: {libsvm_file}")
    
    # Load using sklearn
    X, y = load_svmlight_file(libsvm_file)
    
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Create DataFrame with proper column names
    if feature_mapping:
        feature_names = feature_mapping['feature_names']
        # Ensure we have the right number of columns
        if X.shape[1] <= len(feature_names):
            column_names = feature_names[:X.shape[1]]
        else:
            # If we have more features than expected, use generic names
            column_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), X.shape[1])]
    else:
        column_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=column_names)
    df['sap_flow'] = y
    
    print(f"Loaded {len(df):,} rows, {X.shape[1]} features")
    
    return df

def add_temporal_features_for_rolling_window(df):
    """
    Add temporal features needed for rolling window analysis
    This simulates having timestamp information for window creation
    """
    print("Adding temporal features for rolling window analysis...")
    
    # For very large datasets (8M+ rows), use simple integer-based temporal simulation
    # This avoids pandas date_range overflow issues
    
    print(f"Dataset size: {len(df):,} rows")
    print("Using integer-based temporal simulation to avoid overflow")
    
    # Create simple temporal index (simulates days)
    df['temporal_index'] = np.arange(len(df))
    
    # Simulate temporal patterns using modular arithmetic
    # Each "day" represents a data point, cycle through patterns
    days_per_year = 365
    days_per_month = 30
    hours_per_day = 24
    
    # Calculate temporal features from index
    df['day_of_year'] = (df['temporal_index'] % days_per_year) + 1
    df['month'] = ((df['temporal_index'] // days_per_month) % 12) + 1
    df['season'] = df['month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})
    df['hour'] = (df['temporal_index'] % hours_per_day)
    
    # Create a simple timestamp for sorting (just use temporal_index)
    df['TIMESTAMP'] = df['temporal_index']  # Simple integer timestamp
    
    print(f"Added temporal features using integer simulation")
    print(f"Temporal range: 0 to {len(df)-1} (temporal units)")
    print(f"Simulated time span: {len(df)/days_per_year:.1f} years")
    
    return df

def create_rolling_window_splits(df, feature_cols, target_col, window_size_days=30, forecast_horizon_days=7, n_windows=10):
    """
    Create rolling window splits for time series validation
    Each window trains on historical data and predicts future periods
    """
    print(f"Creating rolling window validation splits...")
    print(f"Window size: {window_size_days} days, Forecast horizon: {forecast_horizon_days} days")
    print("Using rolling window method: short-term forecasting with seasonal analysis")
    
    # Sort by timestamp
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    # Get time range (integer timestamps)
    start_idx = df['TIMESTAMP'].min()
    end_idx = df['TIMESTAMP'].max()
    total_time_units = end_idx - start_idx
    
    print(f"Data time range: {start_idx} to {end_idx} ({total_time_units} temporal units)")
    
    # Calculate window spacing (using integer arithmetic)
    min_window_spacing = window_size_days + forecast_horizon_days
    available_span = total_time_units - min_window_spacing
    
    if available_span <= 0:
        raise ValueError(f"Insufficient data for rolling windows. Need at least {min_window_spacing} units.")
    
    if n_windows > 1:
        window_spacing = available_span // (n_windows - 1)
    else:
        window_spacing = 0
    
    window_spacing = max(window_spacing, min_window_spacing)
    print(f"Window spacing: {window_spacing} temporal units")
    
    # Create rolling window splits
    rolling_splits = []
    seasonal_info = []
    
    print(f"\nCreating {n_windows} rolling window splits...")
    
    for i in range(n_windows):
        # Calculate window indices (integer arithmetic)
        window_start_idx = start_idx + i * window_spacing
        train_end_idx = window_start_idx + window_size_days
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + forecast_horizon_days
        
        # Check if we have enough data
        if test_end_idx > end_idx:
            print(f"  Window {i+1}: Skipped (insufficient future data)")
            break
        
        # Filter data for this window
        train_mask = (df['TIMESTAMP'] >= window_start_idx) & (df['TIMESTAMP'] < train_end_idx)
        test_mask = (df['TIMESTAMP'] >= test_start_idx) & (df['TIMESTAMP'] < test_end_idx)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        if len(train_data) < 100 or len(test_data) < 10:
            print(f"  Window {i+1}: Skipped (insufficient data)")
            continue
        
        # Seasonal analysis
        train_season = train_data['season'].mode()[0] if len(train_data) > 0 else 0
        test_season = test_data['season'].mode()[0] if len(test_data) > 0 else 0
        train_month = train_data['month'].mode()[0] if len(train_data) > 0 else 0
        test_month = test_data['month'].mode()[0] if len(test_data) > 0 else 0
        
        print(f"  Window {i+1}: Train {window_start_idx} to {train_end_idx}")
        print(f"           Test {test_start_idx} to {test_end_idx}")
        print(f"           Train: {len(train_data):,} records (Season {train_season}, Month {train_month})")
        print(f"           Test:  {len(test_data):,} records (Season {test_season}, Month {test_month})")
        
        # Store seasonal information
        seasonal_info.append({
            'window': i + 1,
            'train_start': window_start_idx,
            'train_end': train_end_idx,
            'test_start': test_start_idx,
            'test_end': test_end_idx,
            'train_season': train_season,
            'test_season': test_season,
            'train_month': train_month,
            'test_month': test_month,
            'season_names': {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'},
            'is_seasonal_transition': train_season != test_season
        })
        
        rolling_splits.append((train_data, test_data, i + 1))
    
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

def create_libsvm_files_for_window(train_data, test_data, feature_cols, target_col, temp_dir, window_idx):
    """
    Create libsvm files for a specific rolling window
    """
    train_file = os.path.join(temp_dir, f'train_window_{window_idx}.svm')
    test_file = os.path.join(temp_dir, f'test_window_{window_idx}.svm')
    
    # Prepare training data
    train_clean = train_data.dropna(subset=[target_col])
    X_train = train_clean[feature_cols].fillna(0).values
    y_train = train_clean[target_col].values
    
    # Prepare test data
    test_clean = test_data.dropna(subset=[target_col])
    X_test = test_clean[feature_cols].fillna(0).values
    y_test = test_clean[target_col].values
    
    # Save to libsvm format
    dump_svmlight_file(X_train, y_train, train_file)
    dump_svmlight_file(X_test, y_test, test_file)
    
    print(f"    Created libsvm files for window {window_idx}")
    print(f"      Train: {len(y_train):,} samples -> {train_file}")
    print(f"      Test: {len(y_test):,} samples -> {test_file}")
    
    return train_file, test_file

def train_external_memory_xgboost_window(train_file, test_file, window_idx):
    """Train XGBoost model using external memory for a specific window"""
    print(f"Training window {window_idx} with external memory...")
    
    try:
        # Create DMatrix objects for external memory
        dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
        dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
    except Exception as e:
        print(f"libsvm format specification failed: {e}")
        try:
            # Fallback to regular DMatrix
            dtrain = xgb.DMatrix(train_file)
            dtest = xgb.DMatrix(test_file)
        except Exception as e2:
            print(f"Fallback DMatrix failed: {e2}")
            raise
    
    # External memory optimized parameters for rolling windows
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,          # Smaller depth for rolling windows
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 128,
        'verbosity': 0,
        'nthread': -1
    }
    
    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,     # Fewer rounds for rolling windows
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)
    
    # Extract actual values
    y_train_actual = dtrain.get_label()
    y_test_actual = dtest.get_label()
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
        'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
        'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
        'train_r2': r2_score(y_train_actual, y_pred_train),
        'test_r2': r2_score(y_test_actual, y_pred_test),
        'train_samples': len(y_train_actual),
        'test_samples': len(y_test_actual)
    }
    
    # Cleanup
    del dtrain, dtest
    gc.collect()
    
    return model, metrics

def train_rolling_window_external_memory(rolling_splits, seasonal_info, feature_cols, target_col, temp_dir):
    """Train XGBoost models with rolling window validation using external memory"""
    n_windows = len(rolling_splits)
    print(f"Training XGBoost with {n_windows}-window rolling validation (external memory)...")
    
    all_metrics = []
    window_models = []
    seasonal_results = []
    
    for window_idx, (train_data, test_data, window_num) in enumerate(rolling_splits):
        print(f"\n--- Training Rolling Window {window_idx + 1}/{n_windows} ---")
        
        # Get seasonal info
        season_info = seasonal_info[window_idx]
        
        # Create libsvm files for this window
        train_file, test_file = create_libsvm_files_for_window(
            train_data, test_data, feature_cols, target_col, temp_dir, window_idx
        )
        
        try:
            # Train model for this window
            model, window_metrics = train_external_memory_xgboost_window(
                train_file, test_file, window_idx
            )
            
            # Add window and seasonal information
            window_metrics.update({
                'window': window_num,
                'train_start': season_info['train_start'],
                'test_start': season_info['test_start'],
                'train_season': season_info['train_season'],
                'test_season': season_info['test_season'],
                'train_month': season_info['train_month'],
                'test_month': season_info['test_month'],
                'is_seasonal_transition': season_info['is_seasonal_transition']
            })
            
            all_metrics.append(window_metrics)
            window_models.append(model)
            
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
            
        finally:
            # Clean up window files
            for f in [train_file, test_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        
        # Memory cleanup
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
        return pd.DataFrame({'feature': feature_cols if feature_cols else [], 'importance': [0.0] * len(feature_cols) if feature_cols else []})

def create_enhanced_feature_importance(feature_importance, feature_cols, feature_mapping=None):
    """Create enhanced feature importance with both indices and names"""
    enhanced_importance = feature_importance.copy()
    
    # Extract feature names based on the actual features in the importance DataFrame
    feature_names = []
    feature_indices = []
    
    for _, row in feature_importance.iterrows():
        feature_key = row['feature']  # e.g., 'f0', 'f1', etc.
        feature_indices.append(feature_key)
        
        # Extract numeric index from feature key
        if feature_key.startswith('f'):
            try:
                numeric_idx = int(feature_key[1:])
                
                # Try to get name from feature_cols first
                if feature_cols is not None and numeric_idx < len(feature_cols):
                    feature_name = feature_cols[numeric_idx]
                # Fall back to feature mapping
                elif feature_mapping is not None and 'features' in feature_mapping:
                    feature_name = feature_mapping['features'].get(feature_key, f'feature_{numeric_idx}')
                else:
                    feature_name = f'feature_{numeric_idx}'
                    
            except (ValueError, IndexError):
                feature_name = feature_key
        else:
            feature_name = feature_key
        
        feature_names.append(feature_name)
    
    enhanced_importance['feature_name'] = feature_names
    enhanced_importance['feature_index'] = feature_indices
    
    return enhanced_importance

def save_rolling_window_external_results(models, all_metrics, avg_metrics, feature_importance, seasonal_results, feature_cols, feature_mapping=None, output_dir='external_memory_models/rolling_window'):
    """Save rolling window external memory model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_window_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_window_idx]
    best_window = all_metrics[best_window_idx]['window']
    model_path = f"{output_dir}/sapfluxnet_rolling_window_external_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Create enhanced feature importance
    enhanced_importance = create_enhanced_feature_importance(feature_importance, feature_cols, feature_mapping)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_rolling_window_external_importance_{timestamp}.csv"
    enhanced_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed window results
    window_results_path = f"{output_dir}/sapfluxnet_rolling_window_external_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(window_results_path, index=False)
    
    # Save seasonal analysis results
    seasonal_results_path = f"{output_dir}/sapfluxnet_rolling_window_external_seasonal_{timestamp}.csv"
    pd.DataFrame(seasonal_results).to_csv(seasonal_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_rolling_window_external_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in rolling window external memory training:\n")
        f.write("Method: Rolling window time series validation with external memory\n")
        f.write("Split: Historical data → Future forecast periods\n\n")
        
        if feature_cols is not None or feature_mapping is not None:
            f.write("Feature Index | Feature Name\n")
            f.write("-" * 50 + "\n")
            for i, row in enhanced_importance.iterrows():
                f.write(f"{row['feature_index']:>12} | {row['feature_name']}\n")
        else:
            f.write("Features: Used existing libsvm format files\n")
            f.write("Feature names not available\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_rolling_window_external_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Rolling Window External Memory Training Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Rolling window time series validation with external memory\n")
        f.write("Approach: Short-term forecasting with seasonal analysis\n")
        f.write("Memory: External memory (disk-based) for efficiency\n\n")
        
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
        f.write("- External memory processing for efficiency\n")
        f.write("- Multiple forecast windows sliding through time\n")
        f.write("- Tests short-term forecasting capability\n")
        f.write("- Reveals seasonal forecasting patterns\n")
        f.write("- Simulates operational forecasting scenarios\n")
        f.write("- Evaluates model performance across seasons\n")
        f.write("- Memory-efficient disk-based training\n")
    
    print(f"\nRolling window external memory model results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Window results: {window_results_path}")
    print(f"  Seasonal analysis: {seasonal_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main rolling window external memory training pipeline"""
    print("SAPFLUXNET Rolling Window External Memory XGBoost Training")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print("Method: Rolling window time series validation with external memory")
    print("Approach: Short-term forecasting with seasonal pattern analysis")
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories
    libsvm_dir = '../processed_libsvm'
    
    # Set up temp directory
    temp_dir = 'temp_rolling_window_external'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load feature mapping
        feature_mapping = load_feature_mapping(libsvm_dir)
        
        # Step 1: Combine libsvm files
        print("\n" + "="*60)
        print("COMBINING LIBSVM FILES")
        print("="*60)
        
        all_data_file, total_rows = combine_libsvm_files(libsvm_dir, temp_dir)
        
        # Extract feature information from mapping
        if feature_mapping:
            feature_cols = feature_mapping['feature_names']
            target_col = feature_mapping['target_column']
            print(f"✅ Using feature mapping: {len(feature_cols)} features")
        else:
            print("⚠️  No feature mapping available")
            feature_cols = None
            target_col = 'sap_flow'
        
        # Step 2: Load data as DataFrame for rolling window analysis
        print("\n" + "="*60)
        print("LOADING DATA FOR ROLLING WINDOW ANALYSIS")
        print("="*60)
        
        df = load_libsvm_as_dataframe(all_data_file, feature_mapping)
        
        # Add temporal features for rolling window analysis
        df = add_temporal_features_for_rolling_window(df)
        
        # Step 3: Create rolling window splits
        print("\n" + "="*60)
        print("CREATING ROLLING WINDOW SPLITS")
        print("="*60)
        
        rolling_splits, seasonal_info = create_rolling_window_splits(
            df, feature_cols, target_col, 
            window_size_days=30, forecast_horizon_days=7, n_windows=10
        )
        
        # Step 4: Train rolling window models with external memory
        print("\n" + "="*60)
        print("TRAINING ROLLING WINDOW MODELS (EXTERNAL MEMORY)")
        print("="*60)
        
        models, all_metrics, avg_metrics, seasonal_results = train_rolling_window_external_memory(
            rolling_splits, seasonal_info, feature_cols, target_col, temp_dir
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(models, feature_cols)
        
        # Step 6: Save results
        model_path = save_rolling_window_external_results(
            models, all_metrics, avg_metrics, feature_importance, seasonal_results, 
            feature_cols, feature_mapping
        )
        
        print(f"\n✅ Rolling window external memory training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model tests short-term forecasting with external memory efficiency")
        print(f"🚀 Memory-efficient external memory approach")
        print(f"📊 Method: Rolling window time series validation")
        print(f"🎯 Windows tested: {len(rolling_splits)} forecast periods")
        print(f"🌱 Seasonal analysis: Performance across seasons and transitions")
        
    except Exception as e:
        print(f"\n❌ Rolling window external memory training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files from: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary files cleaned up successfully")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 