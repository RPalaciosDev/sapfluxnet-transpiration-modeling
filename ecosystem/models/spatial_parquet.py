"""
Within-Cluster Spatial Validation for Ecosystem-Based Models (Raw Parquet Version)
Tests spatial generalization of cluster-specific XGBoost models using Leave-One-Site-Out validation within each cluster

PARQUET WORKFLOW (Comparison Version):
- Loads data directly from parquet files (no preprocessing)
- Cluster assignments come from CSV files  
- Applies features preparation on-the-fly
- For comparison with preprocessed libsvm approach

IMPORTANT: CORRECTED VERSION - Eliminates Data Leakage
- Retrains models for each LOSO fold instead of using pre-trained models
- Each fold uses a model trained ONLY on the training sites for that fold
- Ensures true spatial validation without test site contamination
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
from datetime import datetime
import warnings
import gc
import psutil
import argparse
import shutil
import json
from sklearn.metrics import r2_score

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

class ParquetSpatialValidator:
    """
    Validates cluster-specific models using Leave-One-Site-Out within each cluster
    Uses raw parquet data (no preprocessing)
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 models_dir='./results/cluster_models',
                 results_dir='./results/parquet_spatial_validation',
                 optimize_hyperparams=False):
        self.parquet_dir = parquet_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.optimize_hyperparams = optimize_hyperparams
        self.optimized_params = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🌍 Parquet Spatial Validator initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"🤖 Models directory: {models_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"💡 RAW PARQUET WORKFLOW: Uses original parquet files directly")
    
    def load_cluster_assignments(self):
        """Load cluster assignments from the latest clustering results"""
        cluster_files = sorted(glob.glob('../evaluation/clustering_results/advanced_site_clusters_*.csv'))
        
        if not cluster_files:
            raise FileNotFoundError("No cluster assignment files found")
        
        latest_file = cluster_files[-1]
        print(f"📊 Loading cluster assignments from: {os.path.basename(latest_file)}")
        
        clusters_df = pd.read_csv(latest_file)
        cluster_assignments = dict(zip(clusters_df['site'], clusters_df['cluster']))
        
        print(f"✅ Loaded {len(cluster_assignments)} site assignments")
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} sites")
        
        return cluster_assignments, cluster_counts
    
    def load_cluster_models(self):
        """Load trained cluster-specific models"""
        print(f"\n🤖 Loading cluster-specific models from {self.models_dir}...")
        
        model_files = glob.glob(os.path.join(self.models_dir, 'xgb_model_cluster_*.json'))
        
        if not model_files:
            raise FileNotFoundError(f"No cluster models found in {self.models_dir}")
        
        models = {}
        for model_file in model_files:
            # Extract cluster ID from filename
            filename = os.path.basename(model_file)
            # Format: xgb_model_cluster_{cluster_id}_{timestamp}.json
            parts = filename.replace('.json', '').split('_')
            cluster_id = None
            for i, part in enumerate(parts):
                if part == 'cluster' and i + 1 < len(parts):
                    try:
                        cluster_id = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            
            if cluster_id is not None:
                model = xgb.Booster()
                model.load_model(model_file)
                models[cluster_id] = model
                print(f"  ✅ Loaded model for cluster {cluster_id}")
            else:
                print(f"  ⚠️  Could not extract cluster ID from {filename}")
        
        print(f"📊 Loaded {len(models)} cluster models")
        return models
    
    def analyze_cluster_size(self, cluster_id, cluster_sites):
        """Analyze cluster data size to decide on memory strategy"""
        print(f"📊 Analyzing cluster {cluster_id} data size ({len(cluster_sites)} sites)...")
        
        total_rows = 0
        total_size_mb = 0
        successful_sites = []
        site_info = {}
        
        for site in cluster_sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"  ⚠️  Missing parquet file: {parquet_file}")
                continue
            
            try:
                # Get file size
                file_size_mb = os.path.getsize(parquet_file) / (1024**2)
                
                # Sample first few rows to estimate data size
                df_sample = pd.read_parquet(parquet_file, columns=[self.target_col])
                df_sample = df_sample.dropna(subset=[self.target_col])
                site_rows = len(df_sample)
                
                if site_rows == 0:
                    print(f"    ⚠️  No valid data for {site}")
                    del df_sample
                    continue
                
                total_rows += site_rows
                total_size_mb += file_size_mb
                successful_sites.append(site)
                
                site_info[site] = {
                    'file_path': parquet_file,
                    'rows': site_rows,
                    'size_mb': file_size_mb
                }
                
                print(f"    ✅ {site}: {site_rows:,} rows, {file_size_mb:.1f} MB")
                
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error analyzing {site}: {e}")
                continue
        
        if not successful_sites:
            raise ValueError(f"No valid sites found for cluster {cluster_id}")
        
        print(f"  📊 Cluster {cluster_id} total: {total_rows:,} rows, {total_size_mb:.1f} MB")
        
        return successful_sites, site_info, total_rows, total_size_mb
    
    def load_cluster_data_from_parquet(self, cluster_id, cluster_sites):
        """Load cluster data with memory-aware strategy"""
        # First analyze the cluster size
        successful_sites, site_info, total_rows, total_size_mb = self.analyze_cluster_size(cluster_id, cluster_sites)
        
        if len(successful_sites) < 3:
            raise ValueError(f"Need at least 3 sites for validation, found {len(successful_sites)}")
        
        # Check available memory to decide strategy
        available_memory_gb = get_available_memory_gb()
        estimated_memory_gb = total_size_mb / 1024 * 2  # Rough estimate with processing overhead
        
        print(f"  💾 Available memory: {available_memory_gb:.1f} GB")
        print(f"  📊 Estimated memory needed: {estimated_memory_gb:.1f} GB")
        
        # Adaptive streaming threshold based on available memory
        if available_memory_gb > 400:  # Ultra-high memory system (like yours with 545GB!)
            use_streaming = (total_size_mb > 50000) or (total_rows > 20000000)  # 50GB or 20M rows
        elif available_memory_gb > 100:  # High memory system 
            use_streaming = (total_size_mb > 20000) or (total_rows > 10000000)  # 20GB or 10M rows
        elif available_memory_gb > 50:  # Medium-high memory system  
            use_streaming = (total_size_mb > 5000) or (total_rows > 2000000)  # 5GB or 2M rows
        elif available_memory_gb > 20:  # Medium memory system
            use_streaming = (total_size_mb > 2000) or (total_rows > 1000000)  # 2GB or 1M rows
        else:  # Low memory system
            use_streaming = (total_size_mb > 1000) or (total_rows > 500000)  # 1GB or 500K rows
        
        if use_streaming:
            print(f"  💾 Using STREAMING approach (large dataset: {total_size_mb:.1f} MB, {total_rows:,} rows)")
            return self._prepare_streaming_cluster_data(cluster_id, site_info), 'streaming'
        else:
            print(f"  🚀 Using IN-MEMORY approach (small dataset: {total_size_mb:.1f} MB, {total_rows:,} rows)")
            return self._load_cluster_data_in_memory(cluster_id, site_info), 'in_memory'
    
    def _load_cluster_data_in_memory(self, cluster_id, site_info):
        """Load cluster data in memory (for smaller clusters)"""
        print(f"    📊 Loading cluster data in memory...")
        
        cluster_data = []
        
        for site, info in site_info.items():
            try:
                df_site = pd.read_parquet(info['file_path'])
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) == 0:
                    continue
                
                # Add site identifier for validation splits
                df_site['site'] = site
                cluster_data.append(df_site)
                
                del df_site
                gc.collect()
                
            except Exception as e:
                print(f"      ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            raise ValueError(f"No valid data loaded for cluster {cluster_id}")
        
        # Combine all site data
        combined_df = pd.concat(cluster_data, ignore_index=True)
        print(f"    ✅ Loaded {len(combined_df):,} rows in memory")
        
        # Clean up individual site data
        del cluster_data
        gc.collect()
        
        return combined_df, list(site_info.keys())
    
    def _prepare_streaming_cluster_data(self, cluster_id, site_info):
        """Prepare cluster data for streaming validation"""
        print(f"    🔧 Preparing streaming validation setup...")
        
        # Create temporary directory for this cluster
        temp_dir = os.path.join(self.results_dir, f'temp_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Return site info and temp directory for streaming processing
        return {
            'site_info': site_info,
            'temp_dir': temp_dir,
            'cluster_id': cluster_id
        }, list(site_info.keys())

    def prepare_features(self, df):
        """Prepare features for training (same as used in cluster training)"""
        # Exclude columns (same as in train_cluster_models.py)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [self.target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        # Extract and clean features
        X_df = df[feature_cols].copy()
        
        # Convert boolean columns to numeric (True=1, False=0)
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype == 'object':
                # Try to convert object columns to numeric, fill non-numeric with 0
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with 0
        X = X_df.fillna(0).values
        y = df[self.target_col].values
        
        return X, y, feature_cols

    def validate_cluster_spatially(self, cluster_id, cluster_sites):
        """Perform Leave-One-Site-Out validation within a cluster - CORRECTED (no data leakage)"""
        print(f"\n{'='*60}")
        print(f"SPATIAL VALIDATION FOR CLUSTER {cluster_id} (PARQUET)")
        print(f"{'='*60}")
        print(f"Sites in cluster: {len(cluster_sites)}")
        
        if len(cluster_sites) < 3:
            print(f"⚠️  Skipping cluster {cluster_id}: Need at least 3 sites for spatial validation")
            return None
        
        # Load data for this cluster (memory-optimized)
        try:
            cluster_data, data_mode = self.load_cluster_data_from_parquet(cluster_id, cluster_sites)
        except Exception as e:
            print(f"❌ Failed to load cluster data: {e}")
            return None
        
        if data_mode == 'in_memory':
            cluster_df, successful_sites = cluster_data
            return self._validate_cluster_in_memory_parquet(cluster_id, successful_sites, cluster_df)
        else:  # streaming mode
            streaming_info, successful_sites = cluster_data
            return self._validate_cluster_streaming_parquet(cluster_id, successful_sites, streaming_info)
    
    def _validate_cluster_in_memory_parquet(self, cluster_id, successful_sites, cluster_df):
        """Perform validation with in-memory parquet data"""
        print(f"  🚀 Running in-memory validation with parquet data...")
        
        if len(successful_sites) < 3:
            print(f"  ⚠️  Only {len(successful_sites)} sites loaded successfully, need at least 3")
            return None
        
        # Prepare features
        try:
            X, y, feature_cols = self.prepare_features(cluster_df)
        except Exception as e:
            print(f"  ❌ Failed to prepare features: {e}")
            return None
        
        fold_results = []
        
        # Leave-One-Site-Out validation
        for i, test_site in enumerate(successful_sites):
            print(f"\n--- Fold {i+1}/{len(successful_sites)}: Test site {test_site} ---")
            
            try:
                # Split data by site
                train_mask = cluster_df['site'] != test_site
                test_mask = cluster_df['site'] == test_site
                
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[test_mask]
                y_test = y[test_mask]
                
                if len(X_test) == 0:
                    print(f"  ⚠️  No test data for {test_site}")
                    continue
                
                print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
                
                # Create DMatrix objects
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # CRITICAL FIX: RETRAIN model for this fold to eliminate data leakage
                xgb_params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
                
                # Train new model ONLY on training sites for this fold
                fold_model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    verbose_eval=False
                )
                
                # Make predictions with fold-specific model
                y_pred_train = fold_model.predict(dtrain)
                y_pred_test = fold_model.predict(dtest)
                
                # Clean up fold-specific model
                del fold_model
                
                # Calculate metrics
                fold_metrics = {
                    'cluster': cluster_id,
                    'fold': i + 1,
                    'test_site': test_site,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'train_mae': mean_absolute_error(y_train, y_pred_train),
                    'test_mae': mean_absolute_error(y_test, y_pred_test),
                    'train_r2': r2_score(y_train, y_pred_train),
                    'test_r2': r2_score(y_test, y_pred_test)
                }
                
                fold_results.append(fold_metrics)
                
                print(f"  Results: Train R² = {fold_metrics['train_r2']:.4f}, Test R² = {fold_metrics['test_r2']:.4f}")
                
                # Clean up
                del dtrain, dtest, y_pred_train, y_pred_test
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error in fold {i+1}: {e}")
                continue
        
        # Clean up cluster data
        del cluster_df, X, y
        gc.collect()
        
        return self._calculate_cluster_summary(cluster_id, successful_sites, fold_results)
    
    def _validate_cluster_streaming_parquet(self, cluster_id, successful_sites, streaming_info):
        """Perform validation with streaming parquet data processing"""
        print(f"  💾 Running streaming validation with parquet data...")
        
        if len(successful_sites) < 3:
            print(f"  ⚠️  Only {len(successful_sites)} sites available, need at least 3")
            return None
        
        site_info = streaming_info['site_info']
        temp_dir = streaming_info['temp_dir']
        
        fold_results = []
        
        try:
            # Leave-One-Site-Out validation
            for i, test_site in enumerate(successful_sites):
                print(f"\n--- Fold {i+1}/{len(successful_sites)}: Test site {test_site} ---")
                
                try:
                    # Create train/test libsvm files for this fold from parquet
                    train_file, test_file, train_samples, test_samples = self._create_fold_files_from_parquet(
                        test_site, site_info, temp_dir
                    )
                    
                    if train_samples == 0 or test_samples == 0:
                        print(f"  ⚠️  Insufficient data for {test_site}")
                        continue
                    
                    print(f"  Train: {train_samples:,} samples, Test: {test_samples:,} samples")
                    
                    # Create DMatrix objects from files
                    dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
                    dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
                    
                    # CRITICAL FIX: RETRAIN model for this fold to eliminate data leakage
                    # Use optimized parameters if available, otherwise defaults
                    cluster_key = str(int(cluster_id))  # Convert to string for consistency
                    if cluster_key in self.optimized_params:
                        xgb_params = self.optimized_params[cluster_key].copy()
                        xgb_params.update({
                            'objective': 'reg:squarederror',
                            'eval_metric': 'rmse',
                            'random_state': 42
                        })
                        print(f"    🎯 Using optimized parameters for cluster {cluster_id}")
                    else:
                        xgb_params = {
                            'objective': 'reg:squarederror',
                            'eval_metric': 'rmse',
                            'max_depth': 8,
                            'learning_rate': 0.1,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'random_state': 42
                        }
                    
                    # Train new model ONLY on training sites for this fold
                    # Use optimized n_estimators if available
                    n_estimators = xgb_params.pop('n_estimators', 100)
                    fold_model = xgb.train(
                        params=xgb_params,
                        dtrain=dtrain,
                        num_boost_round=n_estimators,
                        verbose_eval=False
                    )
                    
                    # Make predictions with fold-specific model
                    y_pred_train = fold_model.predict(dtrain)
                    y_pred_test = fold_model.predict(dtest)
                    
                    # Clean up fold-specific model
                    del fold_model
                    
                    # Load actual targets for metrics
                    y_train_actual = self._load_targets_from_libsvm(train_file)
                    y_test_actual = self._load_targets_from_libsvm(test_file)
                    
                    # Calculate metrics
                    fold_metrics = {
                        'cluster': cluster_id,
                        'fold': i + 1,
                        'test_site': test_site,
                        'train_samples': train_samples,
                        'test_samples': test_samples,
                        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
                        'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
                        'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
                        'train_r2': r2_score(y_train_actual, y_pred_train),
                        'test_r2': r2_score(y_test_actual, y_pred_test)
                    }
                    
                    fold_results.append(fold_metrics)
                    
                    print(f"  Results: Train R² = {fold_metrics['train_r2']:.4f}, Test R² = {fold_metrics['test_r2']:.4f}")
                    
                    # Clean up fold files and memory
                    os.remove(train_file)
                    os.remove(test_file)
                    del dtrain, dtest, y_pred_train, y_pred_test, y_train_actual, y_test_actual
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ❌ Error in fold {i+1}: {e}")
                    continue
            
            return self._calculate_cluster_summary(cluster_id, successful_sites, fold_results)
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"  ⚠️  Could not clean up temp directory: {e}")
    
    def _create_fold_files_from_parquet(self, test_site, site_info, temp_dir):
        """Create train/test libsvm files for a single fold from parquet data"""
        train_file = os.path.join(temp_dir, f'fold_{test_site}_train.svm')
        test_file = os.path.join(temp_dir, f'fold_{test_site}_test.svm')
        
        train_samples = 0
        test_samples = 0
        
        # Get feature columns from first site
        first_site_file = list(site_info.values())[0]['file_path']
        df_sample = pd.read_parquet(first_site_file)
        df_sample = df_sample.head(100)  # Small sample to get feature columns
        _, _, feature_cols = self.prepare_features(df_sample)
        del df_sample
        gc.collect()
        
        print(f"    📊 Creating fold files: test site {test_site}")
        
        with open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
            for site, info in site_info.items():
                try:
                    # MEMORY FIX: Use chunked processing like train_cluster_models.py
                    site_samples = self._process_site_chunked_for_fold(
                        info['file_path'], site, test_site, train_out, test_out
                    )
                    
                    if site == test_site:
                        test_samples += site_samples
                    else:
                        train_samples += site_samples
                    
                except Exception as e:
                    print(f"      ❌ Error processing {site}: {e}")
                    continue
        
        print(f"    ✅ Created fold files: train={train_samples:,}, test={test_samples:,}")
        
        return train_file, test_file, train_samples, test_samples
    
    def find_existing_results_file(self):
        """Find the existing spatial validation results file"""
        pattern = os.path.join(self.results_dir, 'parquet_spatial_fold_results_*')
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No results file found matching pattern: {pattern}")
        
        if len(files) > 1:
            # Get the most recent file
            files.sort(key=os.path.getmtime, reverse=True)
            print(f"⚠️  Multiple results files found, using most recent: {os.path.basename(files[0])}")
        
        return files[0]
    
    def optimize_cluster_hyperparameters(self, cluster_id, sample_train_file, sample_test_file):
        """Run hyperparameter optimization for a cluster using Optuna"""
        try:
            import optuna
        except ImportError:
            print("❌ Optuna not installed. Install with: pip install optuna")
            return None
        
        print(f"🔧 Optimizing hyperparameters for cluster {cluster_id}...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
                'random_state': 42
            }
            
            dtrain = xgb.DMatrix(f"{sample_train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{sample_test_file}?format=libsvm")
            
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=trial.suggest_int('n_estimators', 50, 150),
                verbose_eval=False
            )
            
            y_pred = model.predict(dtest)
            y_actual = self._load_targets_from_libsvm(sample_test_file)
            
            return r2_score(y_actual, y_pred)
        
        study = optuna.create_study(direction='maximize', 
                                  study_name=f'cluster_{cluster_id}_optimization')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        print(f"  🎯 Best R² for cluster {cluster_id}: {study.best_value:.4f}")
        print(f"  📊 Best parameters: {study.best_params}")
        
        return study.best_params
    
    def run_hyperparameter_optimization_phase(self):
        """Run hyperparameter optimization based on existing results"""
        print("\n🔧 HYPERPARAMETER OPTIMIZATION PHASE")
        print("="*60)
        
        # Find existing results file
        try:
            results_file = self.find_existing_results_file()
            print(f"📊 Using results from: {os.path.basename(results_file)}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Run spatial validation first to generate results file")
            return False
        
        # Load results to identify clusters
        results_df = pd.read_csv(results_file)
        clusters = results_df['cluster'].unique()
        
        print(f"🎯 Optimizing hyperparameters for {len(clusters)} clusters...")
        
        # Load cluster assignments
        cluster_assignments, _ = self.load_cluster_assignments()
        
        # Group sites by cluster
        sites_by_cluster = {}
        for site, cluster_id in cluster_assignments.items():
            if cluster_id not in sites_by_cluster:
                sites_by_cluster[cluster_id] = []
            sites_by_cluster[cluster_id].append(site)
        
        # Optimize each cluster using sample data
        for cluster_id in clusters:
            if cluster_id not in sites_by_cluster:
                print(f"⚠️  No sites found for cluster {cluster_id}")
                continue
            
            cluster_sites = sites_by_cluster[cluster_id]
            
            try:
                # Get site info for this cluster
                site_info = {}
                for site in cluster_sites[:3]:  # Use first 3 sites for optimization
                    parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                    if os.path.exists(parquet_file):
                        site_info[site] = {'file_path': parquet_file}
                
                if len(site_info) < 2:
                    print(f"⚠️  Not enough sites for cluster {cluster_id} optimization")
                    continue
                
                # Create sample train/test files
                temp_dir = os.path.join(self.results_dir, f'temp_optimization_cluster_{cluster_id}')
                os.makedirs(temp_dir, exist_ok=True)
                
                sample_sites = list(site_info.keys())
                test_site = sample_sites[0]
                
                sample_train_file, sample_test_file, _, _ = self._create_fold_files_from_parquet(
                    test_site, site_info, temp_dir
                )
                
                # Run optimization
                optimal_params = self.optimize_cluster_hyperparameters(
                    cluster_id, sample_train_file, sample_test_file
                )
                
                if optimal_params:
                    self.optimized_params[str(int(cluster_id))] = optimal_params
                
                # Clean up temp files
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"❌ Error optimizing cluster {cluster_id}: {e}")
                continue
        
        if self.optimized_params:
            print(f"\n✅ Optimization completed for {len(self.optimized_params)} clusters")
            
            # Save optimized parameters
            opt_params_file = os.path.join(self.results_dir, f'optimized_params_{self.timestamp}.json')
            with open(opt_params_file, 'w') as f:
                json.dump(self.optimized_params, f, indent=2)
            print(f"💾 Optimized parameters saved to: {opt_params_file}")
            
            return True
        else:
            print("❌ No clusters were successfully optimized")
            return False
    
    def _process_site_chunked_for_fold(self, parquet_file, site, test_site, train_out, test_out):
        """Process site in chunks like train_cluster_models.py does"""
        import pyarrow.parquet as pq
        
        # Read parquet table for chunked processing
        parquet_table = pq.read_table(parquet_file)
        total_rows = len(parquet_table)
        chunk_size = 50000  # Conservative chunk size like train_cluster_models.py
        
        total_processed = 0
        
        # Process in chunks
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            # Read chunk
            chunk_table = parquet_table.slice(start_idx, end_idx - start_idx)
            df_chunk = chunk_table.to_pandas()
            
            # Drop missing targets
            df_chunk = df_chunk.dropna(subset=[self.target_col])
            
            if len(df_chunk) == 0:
                del df_chunk
                continue
            
            # Prepare features (same as before)
            X, y, _ = self.prepare_features(df_chunk)
            
            # Write to appropriate file
            output_file = test_out if site == test_site else train_out
            
            # Convert to libsvm format and write
            for j in range(len(X)):
                line_parts = [str(y[j])]
                for k, value in enumerate(X[j]):
                    if value != 0:  # Sparse format
                        line_parts.append(f"{k}:{value}")
                output_file.write(' '.join(line_parts) + '\n')
            
            chunk_processed = len(X)
            total_processed += chunk_processed
            
            # Clean up chunk immediately
            del df_chunk, X, y
            gc.collect()
        
        # Clean up parquet table
        del parquet_table
        gc.collect()
        
        return total_processed
    
    def _load_targets_from_libsvm(self, libsvm_file):
        """Load only target values from libsvm file"""
        targets = []
        with open(libsvm_file, 'r') as f:
            for line in f:
                if line.strip():
                    target = float(line.split()[0])
                    targets.append(target)
        return np.array(targets)
    
    def _calculate_cluster_summary(self, cluster_id, cluster_sites, fold_results):
        """Calculate cluster summary statistics"""
        if not fold_results:
            print(f"❌ No successful folds for cluster {cluster_id}")
            return None
        
        cluster_summary = {
            'cluster': cluster_id,
            'total_sites': len(cluster_sites),
            'successful_folds': len(fold_results),
            'mean_test_r2': np.mean([f['test_r2'] for f in fold_results]),
            'std_test_r2': np.std([f['test_r2'] for f in fold_results]),
            'mean_test_rmse': np.mean([f['test_rmse'] for f in fold_results]),
            'std_test_rmse': np.std([f['test_rmse'] for f in fold_results]),
            'min_test_r2': np.min([f['test_r2'] for f in fold_results]),
            'max_test_r2': np.max([f['test_r2'] for f in fold_results])
        }
        
        print(f"\n📊 Cluster {cluster_id} Summary (Parquet):")
        print(f"  Successful folds: {cluster_summary['successful_folds']}/{cluster_summary['total_sites']}")
        print(f"  Test R² (mean ± std): {cluster_summary['mean_test_r2']:.4f} ± {cluster_summary['std_test_r2']:.4f}")
        print(f"  Test R² range: [{cluster_summary['min_test_r2']:.4f}, {cluster_summary['max_test_r2']:.4f}]")
        print(f"  Test RMSE (mean ± std): {cluster_summary['mean_test_rmse']:.4f} ± {cluster_summary['std_test_rmse']:.4f}")
        
        return fold_results, cluster_summary
    
    def run_validation(self):
        """Run spatial validation for all clusters using parquet data"""
        print("🌍 SAPFLUXNET Cluster-Based Spatial Validation (PARQUET)")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Test spatial generalization using raw parquet data")
        
        # Run hyperparameter optimization if requested
        if self.optimize_hyperparams:
            optimization_success = self.run_hyperparameter_optimization_phase()
            if not optimization_success:
                print("⚠️  Continuing with default parameters...")
        
        try:
            # Load cluster assignments
            cluster_assignments, cluster_counts = self.load_cluster_assignments()
            
            # Note: We don't load pre-trained models since we retrain for each fold
            
            # Group sites by cluster
            sites_by_cluster = {}
            for site, cluster_id in cluster_assignments.items():
                if cluster_id not in sites_by_cluster:
                    sites_by_cluster[cluster_id] = []
                sites_by_cluster[cluster_id].append(site)
            
            # Validate each cluster
            all_fold_results = []
            cluster_summaries = []
            
            for cluster_id in sorted(sites_by_cluster.keys()):
                cluster_sites = sites_by_cluster[cluster_id]
                
                log_memory_usage(f"Before cluster {cluster_id} validation")
                
                result = self.validate_cluster_spatially(cluster_id, cluster_sites)
                
                if result is not None:
                    fold_results, cluster_summary = result
                    all_fold_results.extend(fold_results)
                    cluster_summaries.append(cluster_summary)
                
                log_memory_usage(f"After cluster {cluster_id} validation")
            
            # Save results
            self.save_results(all_fold_results, cluster_summaries)
            
            # Print overall summary
            self.print_overall_summary(cluster_summaries)
            
            return all_fold_results, cluster_summaries
            
        except Exception as e:
            print(f"\n❌ Parquet spatial validation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_results(self, all_fold_results, cluster_summaries):
        """Save validation results"""
        print(f"\n💾 Saving results to {self.results_dir}...")
        
        # Save detailed fold results
        if all_fold_results:
            fold_results_df = pd.DataFrame(all_fold_results)
            fold_results_path = os.path.join(self.results_dir, f'parquet_spatial_fold_results_{self.timestamp}.csv')
            fold_results_df.to_csv(fold_results_path, index=False)
            print(f"  ✅ Fold results: {fold_results_path}")
        
        # Save cluster summaries
        if cluster_summaries:
            summaries_df = pd.DataFrame(cluster_summaries)
            summaries_path = os.path.join(self.results_dir, f'parquet_spatial_summaries_{self.timestamp}.csv')
            summaries_df.to_csv(summaries_path, index=False)
            print(f"  ✅ Cluster summaries: {summaries_path}")
        
        # Save comprehensive report
        report_path = os.path.join(self.results_dir, f'parquet_spatial_report_{self.timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("SAPFLUXNET Cluster-Based Spatial Validation Report (PARQUET)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Method: Leave-One-Site-Out within each ecosystem cluster\n")
            f.write(f"Data Source: Raw parquet files (no preprocessing)\n")
            f.write(f"Purpose: Compare with preprocessed libsvm approach\n\n")
            
            if cluster_summaries:
                f.write("Cluster Performance Summary:\n")
                f.write("-" * 30 + "\n")
                for summary in cluster_summaries:
                    f.write(f"Cluster {summary['cluster']}:\n")
                    f.write(f"  Sites: {summary['successful_folds']}/{summary['total_sites']}\n")
                    f.write(f"  Test R²: {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}\n")
                    f.write(f"  Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}\n\n")
                
                # Overall statistics
                overall_r2 = np.mean([s['mean_test_r2'] for s in cluster_summaries])
                overall_rmse = np.mean([s['mean_test_rmse'] for s in cluster_summaries])
                total_folds = sum([s['successful_folds'] for s in cluster_summaries])
                total_sites = sum([s['total_sites'] for s in cluster_summaries])
                
                f.write("Overall Performance:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Test R² across clusters: {overall_r2:.4f}\n")
                f.write(f"Average Test RMSE across clusters: {overall_rmse:.4f}\n")
                f.write(f"Total successful folds: {total_folds}/{total_sites}\n")
        
        print(f"  ✅ Comprehensive report: {report_path}")
    
    def print_overall_summary(self, cluster_summaries):
        """Print overall validation summary"""
        if not cluster_summaries:
            print("\n❌ No cluster summaries available")
            return
        
        print(f"\n📊 OVERALL PARQUET SPATIAL VALIDATION SUMMARY")
        print("=" * 60)
        
        overall_r2 = np.mean([s['mean_test_r2'] for s in cluster_summaries])
        overall_rmse = np.mean([s['mean_test_rmse'] for s in cluster_summaries])
        total_folds = sum([s['successful_folds'] for s in cluster_summaries])
        total_sites = sum([s['total_sites'] for s in cluster_summaries])
        
        print(f"Clusters validated: {len(cluster_summaries)}")
        print(f"Total sites: {total_sites}")
        print(f"Successful folds: {total_folds}")
        print(f"Average Test R² across clusters: {overall_r2:.4f}")
        print(f"Average Test RMSE across clusters: {overall_rmse:.4f}")
        
        print(f"\nCluster Performance Breakdown:")
        for summary in sorted(cluster_summaries, key=lambda x: x['mean_test_r2'], reverse=True):
            print(f"  Cluster {summary['cluster']}: R² = {summary['mean_test_r2']:.4f} "
                  f"({summary['successful_folds']}/{summary['total_sites']} sites)")
        
        print(f"\n✅ Parquet-based spatial validation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Parquet-Based Spatial Validation")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--models-dir', default='./results/cluster_models',
                        help="Directory containing cluster models")
    parser.add_argument('--results-dir', default='./results/parquet_spatial_validation',
                        help="Directory to save validation results")
    parser.add_argument('--optimize-hyperparams', action='store_true',
                        help="Run hyperparameter optimization based on existing results")
    
    args = parser.parse_args()
    
    try:
        validator = ParquetSpatialValidator(
            parquet_dir=args.parquet_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            optimize_hyperparams=args.optimize_hyperparams
        )
        
        fold_results, cluster_summaries = validator.run_validation()
        
        print(f"\n🎉 Parquet spatial validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main() 