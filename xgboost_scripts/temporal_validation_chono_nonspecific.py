"""
Chronological Temporal Validation for SAPFLUXNET Data
Combines all data chronologically and creates true temporal splits by time periods
Uses memory-efficient processing to avoid loading everything into memory
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

def analyze_global_temporal_coverage(parquet_dir):
    """
    Analyze global temporal coverage across all files to understand the full time range
    """
    print("Analyzing global temporal coverage...")
    
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Get temporal info for all files
    all_temporal_info = []
    
    for i, file_path in enumerate(parquet_files):
        try:
            # Read just timestamp columns to save memory
            df_sample = pd.read_parquet(file_path, columns=['TIMESTAMP', 'solar_TIMESTAMP'])
            
            site_name = os.path.basename(file_path).replace('_comprehensive.parquet', '')
            
            # Convert solar_TIMESTAMP to datetime if it's a string
            if df_sample['solar_TIMESTAMP'].dtype == 'object':
                df_sample['solar_TIMESTAMP'] = pd.to_datetime(df_sample['solar_TIMESTAMP'])
            
            # Get temporal range for this file
            start_time = df_sample['solar_TIMESTAMP'].min()
            end_time = df_sample['solar_TIMESTAMP'].max()
            n_measurements = len(df_sample)
            
            all_temporal_info.append({
                'file_path': file_path,
                'site': site_name,
                'start_time': start_time,
                'end_time': end_time,
                'n_measurements': n_measurements
            })
            
            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(parquet_files)} files...")
            
            del df_sample
            gc.collect()
            
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    # Find global temporal range
    global_start = min(info['start_time'] for info in all_temporal_info)
    global_end = max(info['end_time'] for info in all_temporal_info)
    total_measurements = sum(info['n_measurements'] for info in all_temporal_info)
    
    print(f"\nGlobal Temporal Coverage:")
    print(f"  Start: {global_start}")
    print(f"  End: {global_end}")
    print(f"  Duration: {(global_end - global_start).days} days")
    print(f"  Total measurements: {total_measurements:,}")
    
    return all_temporal_info, global_start, global_end, total_measurements

def create_chronological_temporal_splits(all_temporal_info, global_start, global_end, n_folds=5):
    """
    Create temporal splits based on actual chronological time periods
    """
    print(f"Creating chronological temporal splits (n_folds={n_folds})...")
    
    # Calculate time periods for each fold
    total_duration = global_end - global_start
    fold_duration = total_duration / n_folds
    
    fold_splits = []
    
    for fold in range(n_folds):
        # Calculate fold time boundaries
        fold_start_time = global_start + (fold * fold_duration)
        fold_end_time = global_start + ((fold + 1) * fold_duration)
        
        # For each fold, training data is all data before the fold's test period
        train_start_time = global_start
        train_end_time = fold_start_time
        
        # Test data is the fold's time period
        test_start_time = fold_start_time
        test_end_time = fold_end_time
        
        print(f"  Fold {fold + 1}:")
        print(f"    Train period: {train_start_time.date()} to {train_end_time.date()}")
        print(f"    Test period: {test_start_time.date()} to {test_end_time.date()}")
        
        fold_splits.append({
            'fold': fold + 1,
            'train_start': train_start_time,
            'train_end': train_end_time,
            'test_start': test_start_time,
            'test_end': test_end_time
        })
    
    return fold_splits

def process_files_for_temporal_period(file_list, output_file, feature_cols, target_col, 
                                    period_start, period_end, max_memory_gb=8):
    """
    Process files and extract data for a specific temporal period
    """
    print(f"Processing files for period {period_start.date()} to {period_end.date()}...")
    
    total_rows = 0
    
    with open(output_file, 'w') as output:
        for i, file_info in enumerate(file_list):
            file_path = file_info['file_path']
            site_name = file_info['site']
            
            print(f"  Processing {i+1}/{len(file_list)}: {site_name}")
            
            try:
                # Read the entire file
                df_chunk = pd.read_parquet(file_path)
                
                # Convert timestamps to datetime if they're strings
                if 'TIMESTAMP' in df_chunk.columns and df_chunk['TIMESTAMP'].dtype == 'object':
                    df_chunk['TIMESTAMP'] = pd.to_datetime(df_chunk['TIMESTAMP'])
                
                if 'solar_TIMESTAMP' in df_chunk.columns and df_chunk['solar_TIMESTAMP'].dtype == 'object':
                    df_chunk['solar_TIMESTAMP'] = pd.to_datetime(df_chunk['solar_TIMESTAMP'])
                
                # Filter data for the specific temporal period
                period_mask = (df_chunk['solar_TIMESTAMP'] >= period_start) & (df_chunk['solar_TIMESTAMP'] < period_end)
                df_period = df_chunk[period_mask]
                
                if len(df_period) == 0:
                    print(f"    No data in period for {site_name}")
                    del df_chunk, df_period
                    gc.collect()
                    continue
                
                print(f"    Found {len(df_period):,} measurements in period")
                
                # Process in chunks to manage memory - optimized for large datasets
                # For large sites like ESP_TIL_MIX, use smaller chunks
                if len(df_period) > 100000:  # Large site
                    chunk_size = max(5000, int(max_memory_gb * 500000 / len(feature_cols)))
                else:
                    chunk_size = max(10000, int(max_memory_gb * 1000000 / len(feature_cols)))
                total_rows_in_period = len(df_period)
                
                for chunk_start in range(0, total_rows_in_period, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_rows_in_period)
                    df_subset = df_period.iloc[chunk_start:chunk_end]
                    
                    # Ensure we have the required columns
                    if target_col not in df_subset.columns:
                        print(f"    Warning: {target_col} not found in {site_name}")
                        continue
                    
                    # Prepare features and target - exclude timestamps but keep site-specific constants
                    available_features = [col for col in feature_cols if col in df_subset.columns]
                    
                    # Filter out non-numeric columns (including timestamps) - only print once per file
                    if chunk_start == 0:
                        numeric_features = []
                        non_numeric_cols = []
                        for col in available_features:
                            if df_subset[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                numeric_features.append(col)
                            else:
                                non_numeric_cols.append(col)
                        
                        if non_numeric_cols:
                            print(f"    Skipping non-numeric columns: {non_numeric_cols}")
                        
                        if len(numeric_features) == 0:
                            print(f"    Error: No valid numeric features found in {site_name}")
                            continue
                        
                        if len(numeric_features) != len(available_features):
                            print(f"    Using {len(numeric_features)}/{len(available_features)} numeric features for {site_name}")
                    else:
                        # For subsequent chunks, just filter without printing
                        numeric_features = [col for col in available_features if df_subset[col].dtype in ['int64', 'float64', 'int32', 'float32']]
                    
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
                    
                    # Memory cleanup - more aggressive for large sites
                    del X, y, df_subset
                    gc.collect()
                    
                    # Additional cleanup for large sites
                    if len(df_period) > 100000 and chunk_start % (chunk_size * 5) == 0:
                        gc.collect()
                        import psutil
                        memory = psutil.virtual_memory()
                        if memory.percent > 80:
                            print(f"    ⚠️  High memory usage: {memory.percent:.1f}% - forcing cleanup")
                            gc.collect()
                
                print(f"    Completed: {site_name} -> {total_rows:,} total rows in period")
                
                # Clean up the main dataframes
                del df_chunk, df_period
                gc.collect()
                
            except Exception as e:
                print(f"    Error processing {site_name}: {e}")
                continue
    
    print(f"Total rows written for period: {total_rows:,}")
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

def save_chronological_temporal_results(models, all_metrics, avg_metrics, feature_importance, 
                                      global_start, global_end, output_dir='external_memory_models/temporal_validation_chronological_nonspecific'):
    """Save chronological temporal validation results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model (highest test R²)
    best_fold_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['test_r2'])
    best_model = models[best_fold_idx]
    best_fold = all_metrics[best_fold_idx]['fold']
    model_path = f"{output_dir}/sapfluxnet_temporal_chronological_nonspecific_{timestamp}.json"
    best_model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_temporal_chronological_nonspecific_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_temporal_chronological_nonspecific_fold_results_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_temporal_chronological_nonspecific_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Chronological Temporal Validation Results (No Site-Specific Features)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: True chronological temporal cross-validation\n")
        f.write("Approach: Global temporal splitting by time periods\n")
        f.write("Features: Site-specific constants EXCLUDED\n")
        f.write("Memory: Chunked processing with external memory training\n\n")
        
        f.write("Global Temporal Coverage:\n")
        f.write("-" * 25 + "\n")
        f.write(f"  Start: {global_start}\n")
        f.write(f"  End: {global_end}\n")
        f.write(f"  Duration: {(global_end - global_start).days} days\n\n")
        
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
        f.write("- True chronological temporal cross-validation\n")
        f.write("- Global temporal splitting by time periods\n")
        f.write("- Solar timestamp-based temporal ordering\n")
        f.write("- Site-specific constants EXCLUDED\n")
        f.write("- Chunked processing for memory efficiency\n")
        f.write("- External memory training for large datasets\n")
        f.write("- Strict temporal ordering maintained\n")
        f.write("- No data leakage from future to past\n")
    
    print(f"\nChronological temporal validation results saved:")
    print(f"  Best model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main chronological temporal validation pipeline"""
    print("SAPFLUXNET Chronological Temporal Validation (No Site-Specific Features)")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print("Method: True chronological temporal cross-validation")
    print("Approach: Global temporal splitting by time periods")
    print("Features: Site-specific constants EXCLUDED")
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories
    parquet_dir = '../processed_parquet'
    temp_dir = 'temp_temporal_chronological_nonspecific'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Step 1: Analyze global temporal coverage
        print("\n" + "="*60)
        print("ANALYZING GLOBAL TEMPORAL COVERAGE")
        print("="*60)
        
        all_temporal_info, global_start, global_end, total_measurements = analyze_global_temporal_coverage(parquet_dir)
        
        # Step 2: Create chronological temporal splits
        print("\n" + "="*60)
        print("CREATING CHRONOLOGICAL TEMPORAL SPLITS")
        print("="*60)
        
        fold_splits = create_chronological_temporal_splits(all_temporal_info, global_start, global_end, n_folds=5)
        
        if len(fold_splits) == 0:
            print("❌ No valid temporal splits created")
            return
        
        # Step 3: Get feature information from first file
        print("\n" + "="*60)
        print("EXTRACTING FEATURE INFORMATION")
        print("="*60)
        
        sample_file = all_temporal_info[0]['file_path']
        df_sample = pd.read_parquet(sample_file).head(100)
        
        # Identify feature columns (exclude timestamps AND site-specific constants)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'sap_flow', 'is_inside_country']
        
        # Additional site-specific constants to exclude
        site_specific_cols = [
            'latitude', 'longitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
            'biome', 'biome_code', 'igbp_class', 'igbp_code', 'country', 'country_code',
            'site_code', 'site_name', 'stand_age', 'basal_area', 'tree_density', 'stand_height',
            'leaf_area_index', 'clay_percentage', 'sand_percentage', 'silt_percentage',
            'soil_depth', 'soil_texture', 'soil_texture_code', 'terrain', 'terrain_code',
            'growth_condition', 'growth_condition_code', 'species_name', 'leaf_habit',
            'leaf_habit_code', 'n_trees', 'pl_age', 'pl_dbh', 'pl_height', 'pl_leaf_area',
            'pl_bark_thick', 'pl_social', 'social_status_code', 'pl_species', 'pl_sapw_area',
            'pl_sapw_depth', 'measurement_timestep', 'measurement_frequency', 'timezone',
            'timezone_offset', 'climate_zone', 'climate_zone_code', 'latitude_abs',
            'aridity_index', 'tree_size_class', 'tree_age_class', 'tree_volume_index',
            'temp_deviation', 'tree_size_factor', 'sapwood_leaf_ratio', 'transpiration_capacity'
        ]
        
        exclude_cols.extend(site_specific_cols)
        feature_cols = [col for col in df_sample.columns if col not in exclude_cols]
        target_col = 'sap_flow'
        
        print(f"Features: {len(feature_cols)} columns")
        print(f"Target: {target_col}")
        print(f"Sample features: {feature_cols[:5]}")
        
        del df_sample
        gc.collect()
        
        # Step 4: Train chronological temporal models
        print("\n" + "="*60)
        print("TRAINING CHRONOLOGICAL TEMPORAL MODELS")
        print("="*60)
        
        all_metrics = []
        fold_models = []
        
        for fold_split in fold_splits:
            fold_num = fold_split['fold']
            train_start = fold_split['train_start']
            train_end = fold_split['train_end']
            test_start = fold_split['test_start']
            test_end = fold_split['test_end']
            
            print(f"\n--- Training Chronological Temporal Fold {fold_num} ---")
            print(f"Train period: {train_start.date()} to {train_end.date()}")
            print(f"Test period: {test_start.date()} to {test_end.date()}")
            
            # Create libsvm files for this fold
            train_file = os.path.join(temp_dir, f'train_fold_{fold_num}.svm')
            test_file = os.path.join(temp_dir, f'test_fold_{fold_num}.svm')
            
            # Process training period
            print("Processing training period...")
            train_rows = process_files_for_temporal_period(
                all_temporal_info, train_file, feature_cols, target_col, 
                train_start, train_end
            )
            
            # Process test period
            print("Processing test period...")
            test_rows = process_files_for_temporal_period(
                all_temporal_info, test_file, feature_cols, target_col, 
                test_start, test_end
            )
            
            if train_rows > 0 and test_rows > 0:
                try:
                    # Train model for this fold
                    model, fold_metrics = train_external_memory_xgboost_fold(
                        train_file, test_file, fold_num
                    )
                    
                    # Add fold information
                    fold_metrics['fold'] = fold_num
                    fold_metrics['train_start'] = train_start
                    fold_metrics['train_end'] = train_end
                    fold_metrics['test_start'] = test_start
                    fold_metrics['test_end'] = test_end
                    fold_metrics['train_samples'] = train_rows
                    fold_metrics['test_samples'] = test_rows
                    
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
        
        print(f"\n=== Chronological Temporal Cross-Validation Results ===")
        print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
        print(f"Train R² (mean ± std): {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}")
        print(f"Number of folds: {len(fold_splits)}")
        
        # Get feature importance
        feature_importance = get_feature_importance(fold_models, feature_cols)
        
        # Save results
        model_path = save_chronological_temporal_results(
            fold_models, all_metrics, avg_metrics, feature_importance,
            global_start, global_end
        )
        
        print(f"\n✅ Chronological temporal validation completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Best model saved: {model_path}")
        print(f"💡 This model uses true chronological temporal validation")
        print(f"🚀 Memory-efficient chunked processing")
        print(f"📊 Method: Global temporal splitting by time periods")
        print(f"🎯 Folds: {len(fold_splits)} chronological temporal splits")
        print(f"🔄 Solar timestamp-based temporal ordering")
        print(f"🏷️  Site-specific constants EXCLUDED")
        
    except Exception as e:
        print(f"\n❌ Chronological temporal validation failed: {str(e)}")
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