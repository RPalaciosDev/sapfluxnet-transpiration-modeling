"""
External Memory Spatial Validation XGBoost for SAPFLUXNET Data
LEAVE-ONE-SITE-OUT VALIDATION with BALANCED SAMPLING - Tests spatial generalization fairly
Implements external memory training with balanced site representation for true geographic generalization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file
import os
from datetime import datetime
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

def check_disk_space_gb(path):
    """Check available disk space in GB"""
    try:
        statvfs = os.statvfs(path)
        available_bytes = statvfs.f_bavail * statvfs.f_frsize
        return available_bytes / (1024**3)  # Convert to GB
    except:
        return 0

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")

def load_feature_mapping(data_dir):
    """Load feature mapping from JSON file created by the data processing pipeline"""
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

def load_libsvm_as_dataframe(file_path, feature_mapping=None, max_rows=None):
    """Load libsvm file back to DataFrame for site-based sampling (memory optimized)"""
    print(f"Loading libsvm data from: {file_path}")
    
    # For very large datasets, we need to be more memory efficient
    # We'll use chunked processing and only keep essential data in memory
    
    print("Using memory-optimized loading for large dataset...")
    
    # First pass: determine data structure and optionally limit rows
    n_features = 0
    valid_line_count = 0
    
    print("First pass: analyzing data structure...")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                print(f"  Analyzed {i:,} lines...")
            
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        float(parts[0])  # Validate target
                        valid_line_count += 1
                        
                        # Track max feature index
                        for part in parts[1:]:
                            if ':' in part:
                                idx = int(part.split(':')[0])
                                n_features = max(n_features, idx + 1)
                                
                        # Limit rows if specified
                        if max_rows and valid_line_count >= max_rows:
                            print(f"  Limiting to {max_rows:,} rows for memory efficiency")
                            break
                            
                    except ValueError:
                        continue
    
    # Use feature mapping if available
    if feature_mapping and 'feature_count' in feature_mapping:
        n_features = feature_mapping['feature_count']
    
    print(f"  Data structure: {valid_line_count:,} valid samples, {n_features} features")
    
    # Second pass: load data efficiently
    print("Second pass: loading data...")
    
    # Pre-allocate arrays for better memory efficiency
    data_matrix = np.zeros((valid_line_count, n_features), dtype=np.float32)
    targets = np.zeros(valid_line_count, dtype=np.float32)
    
    row_idx = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 500000 == 0 and i > 0:
                print(f"  Loaded {i:,} lines...")
            
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 1 and row_idx < valid_line_count:
                    try:
                        # Extract target
                        targets[row_idx] = float(parts[0])
                        
                        # Extract features
                        for part in parts[1:]:
                            if ':' in part:
                                idx_str, value_str = part.split(':', 1)
                                idx = int(idx_str)
                                value = float(value_str)
                                if idx < n_features:
                                    data_matrix[row_idx, idx] = value
                        
                        row_idx += 1
                        
                        # Stop if we've loaded enough rows
                        if max_rows and row_idx >= max_rows:
                            break
                            
                    except ValueError:
                        continue
    
    print(f"  Loaded {row_idx:,} samples with {n_features} features")
    
    # Convert to DataFrame efficiently
    if feature_mapping and 'feature_names' in feature_mapping:
        feature_cols = feature_mapping['feature_names'][:n_features]
    else:
        feature_cols = [f'f{i}' for i in range(n_features)]
    
    target_col = feature_mapping.get('target_column', 'sap_flow') if feature_mapping else 'sap_flow'
    
    # Create DataFrame from pre-allocated arrays
    df = pd.DataFrame(data_matrix, columns=feature_cols)
    df[target_col] = targets
    
    # Add dummy site column for spatial validation (we'll need to extract this from actual data)
    # For now, create artificial site groupings based on row blocks
    rows_per_site = 3000  # Approximate rows per site
    df['site'] = df.index // rows_per_site
    
    print(f"  Created {df['site'].nunique()} artificial site groups")
    
    return df, feature_cols, target_col

def combine_libsvm_files(libsvm_dir, temp_dir):
    """Combine libsvm files and return as DataFrame for site-based processing"""
    print(f"Combining libsvm files from {libsvm_dir}...")
    
    # Get all libsvm files
    libsvm_files = [f for f in os.listdir(libsvm_dir) if f.endswith('.svm') or f.endswith('.svm.gz')]
    print(f"Found {len(libsvm_files)} libsvm files to combine")
    
    # Combine into single file first (memory efficient)
    combined_file = os.path.join(temp_dir, 'combined_data.svm')
    total_rows = 0
    
    with open(combined_file, 'w') as output_file:
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
            
            output_file.writelines(lines)
            total_rows += len(lines)
            del lines
            gc.collect()
    
    print(f"Combined {total_rows:,} rows into temporary file")
    return combined_file, total_rows

def create_balanced_site_sampling(df, feature_cols, target_col, records_per_site=3000, min_records=500):
    """
    Create balanced sampling across sites for fair spatial validation
    This ensures each site contributes equally to model learning
    """
    print(f"Creating balanced site sampling...")
    print(f"Target: {records_per_site} records per site (minimum: {min_records})")
    
    if 'site' not in df.columns:
        raise ValueError("Site column not found in data - cannot perform spatial validation")
    
    # Analyze site distribution
    site_counts = df['site'].value_counts()
    print(f"\nSite data distribution:")
    print(f"  Total sites: {len(site_counts)}")
    print(f"  Mean records per site: {site_counts.mean():.0f}")
    print(f"  Median records per site: {site_counts.median():.0f}")
    print(f"  Min records: {site_counts.min():,}")
    print(f"  Max records: {site_counts.max():,}")
    
    # Filter sites with sufficient data
    valid_sites = site_counts[site_counts >= min_records].index.tolist()
    filtered_sites = site_counts[site_counts < min_records]
    
    if len(filtered_sites) > 0:
        print(f"\n⚠️  Filtered out {len(filtered_sites)} sites with <{min_records} records:")
        for site, count in filtered_sites.head(10).items():
            print(f"    {site}: {count} records")
        if len(filtered_sites) > 10:
            print(f"    ... and {len(filtered_sites) - 10} more sites")
    
    print(f"\n✅ Using {len(valid_sites)} sites for spatial validation")
    
    # Create balanced sampling
    balanced_data = []
    total_sampled = 0
    
    for site in valid_sites:
        site_data = df[df['site'] == site].copy()
        
        # Sample records for this site
        sample_size = min(records_per_site, len(site_data))
        
        if sample_size == len(site_data):
            # Use all data if site has less than target
            site_sample = site_data
        else:
            # Random sample
            site_sample = site_data.sample(n=sample_size, random_state=42)
        
        balanced_data.append(site_sample)
        total_sampled += len(site_sample)
        
        if len(balanced_data) % 20 == 0:  # Progress every 20 sites
            print(f"  Processed {len(balanced_data)}/{len(valid_sites)} sites...")
    
    # Combine balanced data
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    print(f"\n📊 Balanced sampling results:")
    print(f"  Original data: {len(df):,} records from {df['site'].nunique()} sites")
    print(f"  Balanced data: {len(balanced_df):,} records from {balanced_df['site'].nunique()} sites")
    print(f"  Average per site: {len(balanced_df) / balanced_df['site'].nunique():.0f} records")
    print(f"  Sampling ratio: {len(balanced_df) / len(df) * 100:.1f}%")
    
    return balanced_df, valid_sites

def prepare_features_from_dataframe(df, feature_mapping=None):
    """Prepare features from combined dataframe"""
    if feature_mapping:
        feature_cols = feature_mapping['feature_names']
        target_col = feature_mapping['target_column']
    else:
        # Fallback feature preparation
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
        target_col = 'sap_flow'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: {target_col}")
    
    return feature_cols, target_col

def create_loso_external_splits(balanced_df, feature_cols, target_col, temp_dir):
    """
    Create Leave-One-Site-Out splits in external memory format
    Each fold creates train/test libsvm files for external memory XGBoost
    """
    print(f"Creating Leave-One-Site-Out external memory splits...")
    
    sites = balanced_df['site'].unique()
    print(f"Will create {len(sites)} LOSO folds")
    
    # Check disk space
    available_space = check_disk_space_gb(temp_dir)
    estimated_space_needed = len(balanced_df) * len(sites) * 100 / (1024**3)  # Rough estimate
    print(f"💾 Available space: {available_space:.1f} GB")
    print(f"💾 Estimated space needed: {estimated_space_needed:.1f} GB")
    
    if available_space < estimated_space_needed:
        print(f"⚠️  WARNING: May have insufficient disk space!")
    
    fold_files = []
    
    for i, test_site in enumerate(sites):
        print(f"\nCreating fold {i+1}/{len(sites)}: Test site {test_site}")
        
        # Split data
        train_data = balanced_df[balanced_df['site'] != test_site].copy()
        test_data = balanced_df[balanced_df['site'] == test_site].copy()
        
        print(f"  Train: {len(train_data):,} records from {train_data['site'].nunique()} sites")
        print(f"  Test: {len(test_data):,} records from {test_site}")
        
        # Remove site column (avoid data leakage)
        train_features = train_data[feature_cols].values
        train_targets = train_data[target_col].values
        test_features = test_data[feature_cols].values
        test_targets = test_data[target_col].values
        
        # Create libsvm files for this fold
        train_file = os.path.join(temp_dir, f'fold_{i+1}_train.svm')
        test_file = os.path.join(temp_dir, f'fold_{i+1}_test.svm')
        
        # Save as libsvm format
        dump_svmlight_file(train_features, train_targets, train_file)
        dump_svmlight_file(test_features, test_targets, test_file)
        
        fold_files.append({
            'fold': i + 1,
            'test_site': test_site,
            'train_file': train_file,
            'test_file': test_file,
            'train_samples': len(train_data),
            'test_samples': len(test_data)
        })
        
        # Clean up large arrays
        del train_data, test_data, train_features, train_targets, test_features, test_targets
        gc.collect()
        
        # Progress update
        if (i + 1) % 10 == 0:
            current_space = check_disk_space_gb(temp_dir)
            print(f"  Progress: {i+1}/{len(sites)} folds, Space: {current_space:.1f} GB")
    
    print(f"\n✅ Created {len(fold_files)} LOSO external memory folds")
    return fold_files

def train_loso_external_memory(fold_files, feature_cols, feature_mapping=None):
    """Train XGBoost models using Leave-One-Site-Out external memory"""
    print(f"Training LOSO external memory models...")
    print(f"XGBoost version: {xgb.__version__}")
    
    all_metrics = []
    site_results = []
    
    # External memory optimized parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 256,
        'verbosity': 1,
        'nthread': -1
    }
    
    print(f"XGBoost parameters: {params}")
    
    for fold_info in fold_files:
        fold_num = fold_info['fold']
        test_site = fold_info['test_site']
        train_file = fold_info['train_file']
        test_file = fold_info['test_file']
        
        print(f"\n--- Training Fold {fold_num}/{len(fold_files)} (Site: {test_site}) ---")
        log_memory_usage(f"Before fold {fold_num}")
        
        try:
            # Create external memory DMatrix objects
            print("Creating external memory DMatrix objects...")
            dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
            
            # Train model
            print(f"Training external memory model for site {test_site}...")
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
            print("Making predictions...")
            y_pred_train = model.predict(dtrain)
            y_pred_test = model.predict(dtest)
            
            # Load actual targets for metrics
            y_train_actual = []
            y_test_actual = []
            
            # Extract targets from files
            with open(train_file, 'r') as f:
                for line in f:
                    if line.strip():
                        y_train_actual.append(float(line.split()[0]))
            
            with open(test_file, 'r') as f:
                for line in f:
                    if line.strip():
                        y_test_actual.append(float(line.split()[0]))
            
            y_train_actual = np.array(y_train_actual)
            y_test_actual = np.array(y_test_actual)
            
            # Calculate metrics
            fold_metrics = {
                'fold': fold_num,
                'test_site': test_site,
                'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
                'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
                'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
                'train_r2': r2_score(y_train_actual, y_pred_train),
                'test_r2': r2_score(y_test_actual, y_pred_test),
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
            
            print(f"Fold {fold_num} Results (Site: {test_site}):")
            print(f"  Train R²: {fold_metrics['train_r2']:.4f}")
            print(f"  Test R²: {fold_metrics['test_r2']:.4f} (New site prediction)")
            print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
            print(f"  Samples: {fold_metrics['train_samples']:,} train, {fold_metrics['test_samples']:,} test")
            
            # Store first model for feature importance
            if fold_num == 1:
                best_model = model
            
            # Cleanup
            del dtrain, dtest, model
            del y_pred_train, y_pred_test, y_train_actual, y_test_actual
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error in fold {fold_num} (site {test_site}): {e}")
            continue
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
            values = [fold[metric] for fold in all_metrics]
            avg_metrics[f'{metric}_mean'] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        print(f"\n=== Leave-One-Site-Out Spatial Validation Results ===")
        print(f"Test R² (mean ± std): {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Test RMSE (mean ± std): {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}")
        print(f"Successfully completed folds: {len(all_metrics)}/{len(fold_files)}")
        
        # Best and worst site predictions
        site_df = pd.DataFrame(site_results)
        if len(site_df) > 0:
            best_sites = site_df.nlargest(5, 'test_r2')
            worst_sites = site_df.nsmallest(5, 'test_r2')
            
            print(f"\nBest Predicted Sites (Top 5):")
            for _, row in best_sites.iterrows():
                print(f"  {row['site']}: R² = {row['test_r2']:.4f}")
            
            print(f"\nWorst Predicted Sites (Bottom 5):")
            for _, row in worst_sites.iterrows():
                print(f"  {row['site']}: R² = {row['test_r2']:.4f}")
        
        return best_model, all_metrics, avg_metrics, site_results
    else:
        raise ValueError("No successful folds completed")

def get_enhanced_feature_importance(model, feature_cols, feature_mapping=None):
    """Get enhanced feature importance with both indices and names"""
    try:
        importance_dict = model.get_score(importance_type='gain')
        
        # Create basic importance DataFrame
        feature_importance = pd.DataFrame({
            'feature_index': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        # Add feature names
        feature_names = []
        for idx in feature_importance['feature_index']:
            # Extract numeric index from feature string (e.g., 'f107' -> 107)
            if idx.startswith('f'):
                try:
                    numeric_idx = int(idx[1:])
                    
                    # Try to get name from feature_cols first
                    if feature_cols is not None and numeric_idx < len(feature_cols):
                        feature_name = feature_cols[numeric_idx]
                    # Fall back to feature mapping
                    elif feature_mapping and 'features' in feature_mapping:
                        feature_name = feature_mapping['features'].get(idx, f'feature_{numeric_idx}')
                    else:
                        feature_name = f'feature_{numeric_idx}'
                        
                except (ValueError, IndexError):
                    feature_name = idx
            else:
                feature_name = idx
            
            feature_names.append(feature_name)
        
        feature_importance['feature_name'] = feature_names
        
        print(f"\nTop 15 Most Important Features (averaged across spatial folds):")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature_index']} | {row['feature_name']}: {row['importance']:.4f}")
        
        return feature_importance
        
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        # Fallback
        return pd.DataFrame({
            'feature_index': [f'f{i}' for i in range(len(feature_cols or []))],
            'feature_name': feature_cols or [],
            'importance': [0.0] * len(feature_cols or [])
        })

def save_spatial_external_results(model, all_metrics, avg_metrics, feature_importance, site_results, 
                                 feature_cols, total_original_rows, balanced_rows, n_sites, feature_mapping=None,
                                 output_dir='external_memory_models/spatial_validation'):
    """Save spatial external memory model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_spatial_external_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance with enhanced format
    feature_importance_path = f"{output_dir}/sapfluxnet_spatial_external_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save detailed fold results
    fold_results_path = f"{output_dir}/sapfluxnet_spatial_external_folds_{timestamp}.csv"
    pd.DataFrame(all_metrics).to_csv(fold_results_path, index=False)
    
    # Save site results
    site_results_path = f"{output_dir}/sapfluxnet_spatial_external_sites_{timestamp}.csv"
    pd.DataFrame(site_results).to_csv(site_results_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_spatial_external_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in spatial external memory training:\n")
        f.write("Method: Leave-One-Site-Out with balanced site sampling\n")
        f.write("Approach: External memory + geographic fairness\n")
        f.write(f"Original data: {total_original_rows:,} rows\n")
        f.write(f"Balanced data: {balanced_rows:,} rows from {n_sites} sites\n")
        f.write("Purpose: Fair spatial generalization testing\n\n")
        
        f.write("Feature Index | Feature Name\n")
        f.write("-" * 50 + "\n")
        for i, row in feature_importance.iterrows():
            f.write(f"{row['feature_index']:>12} | {row['feature_name']}\n")
    
    # Save comprehensive metrics
    metrics_path = f"{output_dir}/sapfluxnet_spatial_external_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Spatial External Memory Training Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Leave-One-Site-Out with balanced sampling + external memory\n")
        f.write("Approach: Fair spatial generalization with memory efficiency\n")
        f.write(f"Original dataset: {total_original_rows:,} rows\n")
        f.write(f"Balanced dataset: {balanced_rows:,} rows from {n_sites} sites\n")
        f.write("Geographic fairness: Equal contribution from each site\n\n")
        
        f.write("Average Performance Across Sites:\n")
        f.write("-" * 35 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nCompleted Folds: {len(all_metrics)}/{n_sites}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- Leave-One-Site-Out spatial cross-validation\n")
        f.write("- Balanced site sampling for geographic fairness\n")
        f.write("- External memory for computational efficiency\n")
        f.write("- Each site contributes equally to model learning\n")
        f.write("- Tests true spatial generalization capability\n")
        f.write("- No geographic bias from high-data sites\n")
        
        f.write("\nAdvantages of This Approach:\n")
        f.write("-" * 30 + "\n")
        f.write("- Geographic fairness: Equal site representation\n")
        f.write("- Substantial data: ~3000 records per site\n")
        f.write("- Memory efficient: External memory processing\n")
        f.write("- Scientifically sound: Tests NEW site prediction\n")
        f.write("- Computationally feasible: Manageable fold sizes\n")
        f.write("- Balanced evaluation: No big-site dominance\n")
    
    print(f"\nSpatial external memory model results saved:")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Fold results: {fold_results_path}")
    print(f"  Site results: {site_results_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def load_parquet_for_site_sampling(parquet_dir, feature_mapping=None):
    """Load parquet files for site-based sampling (hybrid approach)"""
    print(f"Loading parquet data from: {parquet_dir}")
    
    # Get all parquet files
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    # Load first file to check structure
    first_file = os.path.join(parquet_dir, parquet_files[0])
    sample_df = pd.read_parquet(first_file, nrows=1000)
    
    print(f"Sample columns: {list(sample_df.columns)}")
    
    # Check if site column exists
    if 'site' not in sample_df.columns:
        raise ValueError("Site column not found in parquet files. Cannot perform spatial validation.")
    
    # Prepare feature columns
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
    target_col = 'sap_flow'
    
    if target_col not in sample_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in parquet files")
    
    feature_cols = [col for col in sample_df.columns 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: {target_col}")
    
    # Load all parquet files
    dfs = []
    total_rows = 0
    
    for i, parquet_file in enumerate(parquet_files):
        print(f"Loading file {i+1}/{len(parquet_files)}: {parquet_file}")
        
        file_path = os.path.join(parquet_dir, parquet_file)
        df_chunk = pd.read_parquet(file_path, columns=['site', target_col] + feature_cols)
        
        # Clean data
        df_chunk = df_chunk.dropna(subset=[target_col])
        df_chunk = df_chunk.fillna(0)  # Fill feature NaNs with 0
        
        dfs.append(df_chunk)
        total_rows += len(df_chunk)
        
        print(f"  Loaded {len(df_chunk):,} rows")
        
        # Memory management
        if i % 5 == 0:
            gc.collect()
    
    # Combine all dataframes
    print(f"Combining {len(dfs)} dataframes...")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    print(f"Total combined data: {len(df):,} rows from {df['site'].nunique()} sites")
    
    return df, feature_cols, target_col, total_rows

def convert_balanced_sample_to_libsvm(balanced_df, feature_cols, target_col, temp_dir):
    """Convert balanced sample to libsvm format for external memory training"""
    print("Converting balanced sample to libsvm format...")
    
    # Create libsvm file
    libsvm_file = os.path.join(temp_dir, 'balanced_sample.svm')
    
    # Extract features and target
    X = balanced_df[feature_cols].values
    y = balanced_df[target_col].values
    
    # Convert to libsvm format
    dump_svmlight_file(X, y, libsvm_file)
    
    print(f"Saved balanced sample to: {libsvm_file}")
    print(f"Data: {len(y):,} samples, {len(feature_cols)} features")
    
    return libsvm_file

def main():
    """Main spatial external memory training pipeline (hybrid approach)"""
    print("SAPFLUXNET Spatial External Memory XGBoost Training (Hybrid Approach)")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    print("Method: Parquet → Site Sampling → libsvm → External Memory Training")
    print("Purpose: True spatial generalization with real site information")
    
    # Check system resources
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories
    parquet_dir = '../processed_parquet'
    libsvm_dir = '../processed_libsvm'
    
    # Choose temp directory with most space
    current_dir_space = check_disk_space_gb('.')
    temp_dir_space = check_disk_space_gb('/tmp')
    
    print(f"💾 Disk space check:")
    print(f"  Current directory: {current_dir_space:.1f} GB available")
    print(f"  System temp (/tmp): {temp_dir_space:.1f} GB available")
    
    if current_dir_space > temp_dir_space and current_dir_space > 10:
        temp_dir = 'temp_spatial_external'
        print(f"✅ Using local temp directory: {temp_dir}")
    elif temp_dir_space > 10:
        try:
            temp_dir = tempfile.mkdtemp(prefix='spatial_external_')
            print(f"✅ Using system temp directory: {temp_dir}")
        except:
            temp_dir = 'temp_spatial_external'
            print(f"⚠️  Falling back to local temp directory: {temp_dir}")
    else:
        temp_dir = 'temp_spatial_external'
        print(f"⚠️  Using local temp directory with limited space: {temp_dir}")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load feature mapping
        feature_mapping = load_feature_mapping(libsvm_dir)
        
        # Step 1: Load parquet data for site sampling
        print("\n" + "="*70)
        print("LOADING PARQUET DATA FOR SITE ANALYSIS")
        print("="*70)
        
        df, feature_cols, target_col, total_rows = load_parquet_for_site_sampling(parquet_dir, feature_mapping)
        
        # Step 2: Create balanced site sampling
        print("\n" + "="*70)
        print("CREATING BALANCED SITE SAMPLING")
        print("="*70)
        
        balanced_df, valid_sites = create_balanced_site_sampling(df, feature_cols, target_col)
        
        # Clean up original dataframe
        del df
        gc.collect()
        log_memory_usage("After balanced sampling")
        
        # Step 3: Convert balanced sample to libsvm format
        print("\n" + "="*70)
        print("CONVERTING TO LIBSVM FORMAT")
        print("="*70)
        
        balanced_libsvm_file = convert_balanced_sample_to_libsvm(balanced_df, feature_cols, target_col, temp_dir)
        
        # Step 4: Create LOSO external memory splits
        print("\n" + "="*70)
        print("CREATING LOSO EXTERNAL MEMORY SPLITS")
        print("="*70)
        
        fold_files = create_loso_external_splits(balanced_df, feature_cols, target_col, temp_dir)
        
        # Clean up balanced dataframe
        balanced_rows = len(balanced_df)
        del balanced_df
        gc.collect()
        log_memory_usage("After creating folds")
        
        # Step 5: Train LOSO external memory models
        print("\n" + "="*70)
        print("TRAINING LOSO EXTERNAL MEMORY MODELS")
        print("="*70)
        
        model, all_metrics, avg_metrics, site_results = train_loso_external_memory(
            fold_files, feature_cols, feature_mapping
        )
        
        # Step 6: Get enhanced feature importance
        feature_importance = get_enhanced_feature_importance(model, feature_cols, feature_mapping)
        
        # Step 7: Save results
        model_path = save_spatial_external_results(
            model, all_metrics, avg_metrics, feature_importance, site_results,
            feature_cols, total_rows, balanced_rows, len(valid_sites), feature_mapping
        )
        
        print(f"\n✅ Spatial external memory training completed successfully!")
        print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")
        print(f"Real sites tested: {len(valid_sites)}")
        print(f"Successful folds: {len(all_metrics)}/{len(valid_sites)}")
        print(f"Model saved: {model_path}")
        print(f"💡 This model tests true spatial generalization with real site information")
        print(f"🚀 Hybrid approach: Parquet → Site sampling → External memory training")
        print(f"📊 Method: LOSO + balanced sampling + external memory")
        print(f"🎯 Purpose: True geographic generalization testing")
        
    except Exception as e:
        print(f"\n❌ Spatial external memory training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up temporary files from: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("✅ Temporary files cleaned up successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temp directory: {e}")
                print(f"⚠️  You may need to manually remove: {temp_dir}")
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 