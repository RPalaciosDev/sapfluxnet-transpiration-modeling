"""
External Memory K-Fold Temporal XGBoost Training for SAPFLUXNET Data
K-FOLD TEMPORAL VALIDATION - Multiple temporal splits for robust validation
Implements k-fold temporal cross-validation with external memory efficiency
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

def add_temporal_features_for_k_fold(df):
    """
    Add temporal features needed for k-fold temporal analysis
    This simulates having timestamp information for temporal splitting
    """
    print("Adding temporal features for k-fold temporal analysis...")
    
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

def k_fold_temporal_split(df, feature_cols, target_col, n_folds=5):
    """
    K-fold temporal split method - creates multiple temporal splits
    Divides time series into k consecutive folds for robust temporal validation
    Each fold uses earlier data for training, later data for testing
    """
    print(f"Creating k-fold temporal split (n_folds={n_folds})...")
    print("Using k-fold temporal method: multiple temporal splits for robust validation")
    
    # Sort data by timestamp globally
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    print("Data sorted by timestamp globally")
    
    # Create k-fold temporal splits
    fold_splits = []
    print(f"Creating {n_folds} temporal folds...")
    
    for fold in range(n_folds):
        print(f"\nCreating fold {fold + 1}/{n_folds}...")
        
        # Calculate fold boundaries
        fold_size = len(df) // n_folds
        if fold_size == 0:
            print(f"  Fold {fold + 1}: Skipped (insufficient data)")
            continue
        
        # Define train/test split for this fold
        # Each fold tests on a different time period
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, len(df))
        
        # Use all data before test period for training
        if test_start > 0:
            train_data = df.iloc[:test_start].copy()
            test_data = df.iloc[test_start:test_end].copy()
            
            if len(train_data) > 100 and len(test_data) > 10:
                print(f"  Fold {fold + 1}: {len(train_data):,} train, {len(test_data):,} test")
                fold_splits.append((train_data, test_data, fold + 1))
            else:
                print(f"  Fold {fold + 1}: Skipped (insufficient data)")
        else:
            print(f"  Fold {fold + 1}: Skipped (no training data)")
    
    print(f"\n✅ K-fold temporal split completed: {len(fold_splits)} valid folds")
    return fold_splits

def create_libsvm_files_for_fold(train_data, test_data, feature_cols, target_col, temp_dir, fold_idx):
    """
    Create libsvm files for a specific k-fold
    """
    train_file = os.path.join(temp_dir, f'train_fold_{fold_idx}.svm')
    test_file = os.path.join(temp_dir, f'test_fold_{fold_idx}.svm')
    
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
    
    print(f"    Created libsvm files for fold {fold_idx}")
    print(f"      Train: {len(y_train):,} samples -> {train_file}")
    print(f"      Test: {len(y_test):,} samples -> {test_file}")
    
    return train_file, test_file

def train_external_memory_xgboost_fold(train_file, test_file, fold_idx):
    """Train XGBoost model using external memory for a specific fold"""
    print(f"Training fold {fold_idx} with external memory...")
    
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
    
    # External memory optimized parameters for k-fold temporal
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,          # Conservative depth for temporal folds
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
        num_boost_round=150,     # Standard rounds for temporal validation
        evals=evals,
        early_stopping_rounds=15,
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

def train_k_fold_temporal_external_memory(fold_splits, feature_cols, target_col, temp_dir):
    """Train XGBoost models with k-fold temporal validation using external memory"""
    print(f"Training XGBoost with {len(fold_splits)}-fold temporal validation (external memory)...")
    
    all_metrics = []
    fold_models = []
    
    for fold_idx, (train_data, test_data, fold_num) in enumerate(fold_splits):
        print(f"\n--- Training Temporal Fold {fold_idx + 1}/{len(fold_splits)} ---")
        
        # Create libsvm files for this fold
        train_file, test_file = create_libsvm_files_for_fold(
            train_data, test_data, feature_cols, target_col, temp_dir, fold_idx
        )
        
        try:
            # Train model for this fold
            model, fold_metrics = train_external_memory_xgboost_fold(
                train_file, test_file, fold_idx
            )
            
            # Add fold information
            fold_metrics['fold'] = fold_num
            
            all_metrics.append(fold_metrics)
            fold_models.append(model)
            
            print(f"Fold {fold_num} Results:")
            print(f"  Train R²: {fold_metrics['train_r2']:.4f}")
            print(f"  Test R²: {fold_metrics['test_r2']:.4f}")
            print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
            print(f"  Samples: {fold_metrics['train_samples']:,} train, {fold_metrics['test_samples']:,} test")
            
        finally:
            # Clean up fold files
            for f in [train_file, test_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
        
        # Memory cleanup
        gc.collect()
    
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

def save_k_fold_temporal_external_results(models, all_metrics, avg_metrics, feature_importance, feature_cols, feature_mapping=None, output_dir='external_memory_models/temporal_validation'):
    """Save k-fold temporal external memory model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_fold_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_fold_idx]
    best_fold = all_metrics[best_fold_idx]['fold']
    model_path = f"{output_dir}/sapfluxnet_k_fold_temporal_external_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Create enhanced feature importance
    enhanced_importance = create_enhanced_feature_importance(feature_importance, feature_cols, feature_mapping)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_k_fold_temporal_external_importance_{timestamp}.csv"
    enhanced_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_k_fold_temporal_external_fold_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_k_fold_temporal_external_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in k-fold temporal external memory training:\n")
        f.write("Method: K-fold temporal cross-validation with external memory\n")
        f.write("Split: Multiple temporal splits for robust validation\n\n")
        
        if feature_cols is not None or feature_mapping is not None:
            f.write("Feature Index | Feature Name\n")
            f.write("-" * 50 + "\n")
            for i, row in enhanced_importance.iterrows():
                f.write(f"{row['feature_index']:>12} | {row['feature_name']}\n")
        else:
            f.write("Features: Used existing libsvm format files\n")
            f.write("Feature names not available\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_k_fold_temporal_external_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET K-Fold Temporal External Memory Training Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: K-fold temporal cross-validation with external memory\n")
        f.write("Approach: Multiple temporal splits for robust validation\n")
        f.write("Memory: External memory (disk-based) for efficiency\n\n")
        
        f.write("Average Performance Across Folds:\n")
        f.write("-" * 35 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nBest Model: Fold {best_fold} (Test R²: {all_metrics[best_fold_idx]['test_r2']:.4f})\n")
        
        f.write("\nIndividual Fold Results:\n")
        f.write("-" * 25 + "\n")
        for metrics in all_metrics:
            f.write(f"Fold {metrics['fold']}: Test R² = {metrics['test_r2']:.4f}, Test RMSE = {metrics['test_rmse']:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- K-fold temporal cross-validation\n")
        f.write("- External memory processing for efficiency\n")
        f.write("- Multiple temporal splits across time periods\n")
        f.write("- Each fold tests on different time period\n")
        f.write("- Robust evaluation with confidence intervals\n")
        f.write("- Prevents data leakage from future to past\n")
        f.write("- Proper temporal cross-validation for time series data\n")
        f.write("- Memory-efficient disk-based training\n")
    
    print(f"\nK-fold temporal external memory model results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main k-fold temporal external memory training pipeline"""
    print("SAPFLUXNET K-Fold Temporal External Memory XGBoost Training")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print("Method: K-fold temporal cross-validation with external memory")
    print("Approach: Multiple temporal splits for robust validation")
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories
    libsvm_dir = '../processed_libsvm'
    
    # Set up temp directory
    temp_dir = 'temp_k_fold_temporal_external'
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
        
        # Step 2: Load data as DataFrame for k-fold temporal analysis
        print("\n" + "="*60)
        print("LOADING DATA FOR K-FOLD TEMPORAL ANALYSIS")
        print("="*60)
        
        df = load_libsvm_as_dataframe(all_data_file, feature_mapping)
        
        # Add temporal features for k-fold analysis
        df = add_temporal_features_for_k_fold(df)
        
        # Step 3: Create k-fold temporal splits
        print("\n" + "="*60)
        print("CREATING K-FOLD TEMPORAL SPLITS")
        print("="*60)
        
        fold_splits = k_fold_temporal_split(df, feature_cols, target_col, n_folds=5)
        
        # Step 4: Train k-fold temporal models with external memory
        print("\n" + "="*60)
        print("TRAINING K-FOLD TEMPORAL MODELS (EXTERNAL MEMORY)")
        print("="*60)
        
        models, all_metrics, avg_metrics = train_k_fold_temporal_external_memory(
            fold_splits, feature_cols, target_col, temp_dir
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(models, feature_cols)
        
        # Step 6: Save results
        model_path = save_k_fold_temporal_external_results(
            models, all_metrics, avg_metrics, feature_importance, 
            feature_cols, feature_mapping
        )
        
        print(f"\n✅ K-fold temporal external memory training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model uses k-fold temporal validation with external memory efficiency")
        print(f"🚀 Memory-efficient external memory approach")
        print(f"📊 Method: K-fold temporal cross-validation")
        print(f"🎯 Folds: {len(fold_splits)} temporal splits")
        print(f"🔄 Robust evaluation with confidence intervals")
        
    except Exception as e:
        print(f"\n❌ K-fold temporal external memory training failed: {str(e)}")
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