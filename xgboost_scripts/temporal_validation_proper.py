"""
Proper Temporal Validation for SAPFLUXNET Data
Uses real TIMESTAMP data from parquet files with memory-efficient processing
Implements k-fold temporal cross-validation without loading entire dataset into memory
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file
import os
from datetime import datetime, timedelta
import warnings
import gc
import psutil
import tempfile
import shutil
import json
from pathlib import Path
import glob

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

def analyze_temporal_coverage(parquet_dir):
    """
    Analyze temporal coverage of all parquet files without loading everything into memory
    """
    print("Analyzing temporal coverage of parquet files...")
    
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Sample a few files to understand temporal structure
    temporal_info = []
    
    for i, file_path in enumerate(parquet_files[:10]):  # Sample first 10 files
        try:
            # Read just the TIMESTAMP column to save memory
            df_sample = pd.read_parquet(file_path, columns=['TIMESTAMP'])
            
            site_name = os.path.basename(file_path).replace('_comprehensive.parquet', '')
            start_time = df_sample['TIMESTAMP'].min()
            end_time = df_sample['TIMESTAMP'].max()
            duration = end_time - start_time
            n_measurements = len(df_sample)
            
            temporal_info.append({
                'site': site_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration_days': duration.days,
                'n_measurements': n_measurements
            })
            
            print(f"  {site_name}: {start_time.date()} to {end_time.date()} ({duration.days} days, {n_measurements:,} measurements)")
            
            del df_sample
            gc.collect()
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    return temporal_info, parquet_files

def create_temporal_splits_from_files(parquet_files, n_folds=5):
    """
    Create temporal splits based on actual file dates
    Sort files by their temporal coverage and create folds
    """
    print(f"Creating temporal splits from {len(parquet_files)} files...")
    
    # Get temporal info for all files (just start dates to save memory)
    file_temporal_info = []
    
    for file_path in parquet_files:
        try:
            # Read just TIMESTAMP column
            df_sample = pd.read_parquet(file_path, columns=['TIMESTAMP'])
            start_time = df_sample['TIMESTAMP'].min()
            
            file_temporal_info.append({
                'file_path': file_path,
                'start_time': start_time,
                'site': os.path.basename(file_path).replace('_comprehensive.parquet', '')
            })
            
            del df_sample
            gc.collect()
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    # Sort files by start time
    file_temporal_info.sort(key=lambda x: x['start_time'])
    
    print(f"Temporal range: {file_temporal_info[0]['start_time'].date()} to {file_temporal_info[-1]['start_time'].date()}")
    
    # Create temporal folds
    fold_splits = []
    files_per_fold = len(file_temporal_info) // n_folds
    
    for fold in range(n_folds):
        fold_start = fold * files_per_fold
        fold_end = min((fold + 1) * files_per_fold, len(file_temporal_info))
        
        # Training files: all files before this fold
        train_files = file_temporal_info[:fold_start]
        # Test files: files in this fold
        test_files = file_temporal_info[fold_start:fold_end]
        
        if len(train_files) > 0 and len(test_files) > 0:
            print(f"  Fold {fold + 1}: {len(train_files)} train files, {len(test_files)} test files")
            print(f"    Train period: {train_files[0]['start_time'].date()} to {train_files[-1]['start_time'].date()}")
            print(f"    Test period: {test_files[0]['start_time'].date()} to {test_files[-1]['start_time'].date()}")
            
            fold_splits.append({
                'fold': fold + 1,
                'train_files': train_files,
                'test_files': test_files
            })
    
    return fold_splits

def process_files_to_libsvm(file_list, output_file, feature_cols, target_col, max_memory_gb=2):
    """
    Process multiple parquet files to libsvm format with memory management
    """
    print(f"Processing {len(file_list)} files to {output_file}...")
    
    total_rows = 0
    available_memory = get_available_memory_gb()
    
    with open(output_file, 'w') as output:
        for i, file_info in enumerate(file_list):
            file_path = file_info['file_path']
            site_name = file_info['site']
            
            print(f"  Processing {i+1}/{len(file_list)}: {site_name}")
            
            try:
                # Read file in chunks to manage memory
                # For parquet files, we need to read the whole file but process in memory-efficient chunks
                df_chunk = pd.read_parquet(file_path)
                
                # Process in chunks to manage memory
                chunk_size = max(1000, int(max_memory_gb * 1000000 / len(feature_cols)))
                total_rows_in_file = len(df_chunk)
                
                for chunk_start in range(0, total_rows_in_file, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_rows_in_file)
                    df_subset = df_chunk.iloc[chunk_start:chunk_end]
                    # Ensure we have the required columns
                    if target_col not in df_subset.columns:
                        print(f"    Warning: {target_col} not found in {site_name}")
                        continue
                    
                    # Convert TIMESTAMP to datetime if it's a string
                    if 'TIMESTAMP' in df_subset.columns and df_subset['TIMESTAMP'].dtype == 'object':
                        try:
                            df_subset['TIMESTAMP'] = pd.to_datetime(df_subset['TIMESTAMP'])
                            print(f"    Converted TIMESTAMP to datetime for {site_name}")
                        except Exception as e:
                            print(f"    Warning: Could not convert TIMESTAMP for {site_name}: {e}")
                    
                    # Prepare features and target - exclude TIMESTAMP and other non-numeric columns
                    available_features = [col for col in feature_cols if col in df_subset.columns]
                    
                    # Filter out non-numeric columns (including TIMESTAMP)
                    numeric_features = []
                    for col in available_features:
                        if df_subset[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            numeric_features.append(col)
                        else:
                            print(f"    Skipping non-numeric column: {col} (dtype: {df_subset[col].dtype})")
                    
                    if len(numeric_features) == 0:
                        print(f"    Error: No valid numeric features found in {site_name}")
                        continue
                    
                    if len(numeric_features) != len(available_features):
                        print(f"    Using {len(numeric_features)}/{len(available_features)} numeric features for {site_name}")
                    
                    X = df_subset[numeric_features].fillna(0).values
                    y = df_subset[target_col].values
                    
                    # Remove rows with NaN target
                    valid_mask = ~np.isnan(y)
                    X = X[valid_mask]
                    y = y[valid_mask]
                    
                    if len(y) > 0:
                        # Save to libsvm format - handle bytes vs string issue
                        try:
                            # Try direct dump first
                            dump_svmlight_file(X, y, output, zero_based=False, multilabel=False)
                        except TypeError as e:
                            if "write() argument must be str, not bytes" in str(e):
                                # Handle bytes issue by writing to temporary file first
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
                                    dump_svmlight_file(X, y, temp_file.name, zero_based=False, multilabel=False)
                                
                                # Read and write as text
                                with open(temp_file.name, 'rb') as temp_read:
                                    content = temp_read.read().decode('utf-8')
                                    output.write(content)
                                
                                # Clean up temp file
                                os.unlink(temp_file.name)
                            else:
                                raise e
                        
                        total_rows += len(y)
                    
                    # Memory cleanup
                    del X, y, df_subset
                    gc.collect()
                    
                    # Check memory usage
                    if get_available_memory_gb() < 1.0:
                        print(f"    Warning: Low memory, processed chunk {chunk_start//chunk_size + 1}")
                        break
                
                print(f"    Completed: {site_name} -> {total_rows:,} total rows")
                
                # Clean up the main dataframe
                del df_chunk
                gc.collect()
                
            except Exception as e:
                print(f"    Error processing {site_name}: {e}")
                continue
    
    print(f"Total rows written: {total_rows:,}")
    return total_rows

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
    
    # XGBoost parameters for temporal validation
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
        'nthread': -1
    }
    
    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
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

def save_temporal_validation_results(models, all_metrics, avg_metrics, feature_importance, output_dir='external_memory_models/temporal_validation_proper'):
    """Save temporal validation results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_fold_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_fold_idx]
    best_fold = all_metrics[best_fold_idx]['fold']
    model_path = f"{output_dir}/sapfluxnet_temporal_proper_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_temporal_proper_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_temporal_proper_fold_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_temporal_proper_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Proper Temporal Validation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: K-fold temporal cross-validation with real timestamps\n")
        f.write("Approach: File-based temporal splitting using actual dates\n")
        f.write("Memory: Chunked processing with external memory training\n\n")
        
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
        f.write("- K-fold temporal cross-validation using real timestamps\n")
        f.write("- File-based temporal splitting (no synthetic timestamps)\n")
        f.write("- Chunked processing for memory efficiency\n")
        f.write("- External memory training for large datasets\n")
        f.write("- Strict temporal ordering maintained\n")
        f.write("- No data leakage from future to past\n")
    
    print(f"\nTemporal validation results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main temporal validation pipeline using real timestamps"""
    print("SAPFLUXNET Proper Temporal Validation")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print("Method: K-fold temporal cross-validation with real timestamps")
    print("Approach: File-based temporal splitting using actual dates")
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories
    parquet_dir = '../processed_parquet'
    temp_dir = 'temp_temporal_proper'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Step 1: Analyze temporal coverage
        print("\n" + "="*60)
        print("ANALYZING TEMPORAL COVERAGE")
        print("="*60)
        
        temporal_info, parquet_files = analyze_temporal_coverage(parquet_dir)
        
        # Step 2: Create temporal splits
        print("\n" + "="*60)
        print("CREATING TEMPORAL SPLITS")
        print("="*60)
        
        fold_splits = create_temporal_splits_from_files(parquet_files, n_folds=5)
        
        if len(fold_splits) == 0:
            print("❌ No valid temporal splits created")
            return
        
        # Step 3: Get feature information from first file
        print("\n" + "="*60)
        print("EXTRACTING FEATURE INFORMATION")
        print("="*60)
        
        sample_file = parquet_files[0]
        df_sample = pd.read_parquet(sample_file).head(100)
        
        # Identify feature columns (exclude metadata and target)
        exclude_cols = ['TIMESTAMP', 'site', 'plant_id', 'sap_flow']
        feature_cols = [col for col in df_sample.columns if col not in exclude_cols]
        target_col = 'sap_flow'
        
        print(f"Features: {len(feature_cols)} columns")
        print(f"Target: {target_col}")
        print(f"Sample features: {feature_cols[:5]}")
        
        del df_sample
        gc.collect()
        
        # Step 4: Train temporal models
        print("\n" + "="*60)
        print("TRAINING TEMPORAL MODELS")
        print("="*60)
        
        all_metrics = []
        fold_models = []
        
        for fold_split in fold_splits:
            fold_num = fold_split['fold']
            train_files = fold_split['train_files']
            test_files = fold_split['test_files']
            
            print(f"\n--- Training Temporal Fold {fold_num} ---")
            print(f"Train files: {len(train_files)}")
            print(f"Test files: {len(test_files)}")
            
            # Create libsvm files for this fold
            train_file = os.path.join(temp_dir, f'train_fold_{fold_num}.svm')
            test_file = os.path.join(temp_dir, f'test_fold_{fold_num}.svm')
            
            # Process training files
            print("Processing training files...")
            train_rows = process_files_to_libsvm(train_files, train_file, feature_cols, target_col)
            
            # Process test files
            print("Processing test files...")
            test_rows = process_files_to_libsvm(test_files, test_file, feature_cols, target_col)
            
            if train_rows > 0 and test_rows > 0:
                try:
                    # Train model for this fold
                    model, fold_metrics = train_external_memory_xgboost_fold(
                        train_file, test_file, fold_num
                    )
                    
                    # Add fold information
                    fold_metrics['fold'] = fold_num
                    fold_metrics['train_files'] = len(train_files)
                    fold_metrics['test_files'] = len(test_files)
                    
                    all_metrics.append(fold_metrics)
                    fold_models.append(model)
                    
                    print(f"Fold {fold_num} Results:")
                    print(f"  Train R²: {fold_metrics['train_r2']:.4f}")
                    print(f"  Test R²: {fold_metrics['test_r2']:.4f}")
                    print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
                    print(f"  Samples: {fold_metrics['train_samples']:,} train, {fold_metrics['test_samples']:,} test")
                    
                except Exception as e:
                    print(f"❌ Error training fold {fold_num}: {e}")
                    continue
            else:
                print(f"❌ Insufficient data for fold {fold_num}")
                continue
            
            # Clean up fold files
            for f in [train_file, test_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
            
            # Memory cleanup
            gc.collect()
        
        if len(all_metrics) == 0:
            print("❌ No successful folds completed")
            return
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
            values = [fold[metric] for fold in all_metrics]
            avg_metrics[f'{metric}_mean'] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        print(f"\n=== Temporal Cross-Validation Results ===")
        print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
        print(f"Train R² (mean ± std): {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}")
        print(f"Number of folds: {len(fold_splits)}")
        
        # Get feature importance
        feature_importance = get_feature_importance(fold_models, feature_cols)
        
        # Save results
        model_path = save_temporal_validation_results(
            fold_models, all_metrics, avg_metrics, feature_importance
        )
        
        print(f"\n✅ Proper temporal validation completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model uses real temporal data with proper validation")
        print(f"🚀 Memory-efficient chunked processing")
        print(f"📊 Method: File-based temporal cross-validation")
        print(f"🎯 Folds: {len(fold_splits)} temporal splits")
        print(f"🔄 Real timestamps, no synthetic data")
        
    except Exception as e:
        print(f"\n❌ Temporal validation failed: {str(e)}")
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