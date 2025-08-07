"""
Standalone Hyperparameter Optimization for Ecosystem Cluster Models

This script runs hyperparameter optimization for XGBoost models using Optuna.
It reads existing spatial validation results and optimizes parameters for each cluster.

Usage:
    python hyperparameter_optimizer.py --results-dir ./results/parquet_spatial_validation/hybrid_v3_20250806_202623
    python hyperparameter_optimizer.py --results-dir ./results/parquet_spatial_validation/biome_20250806_192728 --n-trials 200
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
import os
import glob
import json
import shutil
import argparse
from datetime import datetime
import warnings
import gc
import subprocess

warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """Standalone hyperparameter optimizer for ecosystem cluster models"""
    
    def __init__(self, results_dir, parquet_dir='../../parquet_ecological', 
                 cluster_file=None, n_trials=None, force_gpu=False):
        self.results_dir = results_dir
        self.parquet_dir = parquet_dir
        self.cluster_file = cluster_file
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.optimized_params = {}
        self.force_gpu = force_gpu
        self.target_col = 'sap_flow'
        
        # GPU Detection and Configuration
        self.use_gpu = False
        self.gpu_id = 0
        self.detect_gpu()
        
        # Set number of trials based on GPU availability
        if n_trials is None:
            self.n_trials = 100 if self.use_gpu else 50
        else:
            self.n_trials = n_trials
        
        print(f"🔧 Hyperparameter Optimizer initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"⚡ GPU Acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        print(f"🎯 Trials per cluster: {self.n_trials}")
    
    def detect_gpu(self):
        """Detect GPU availability for XGBoost"""
        print("🔍 Checking GPU and CUDA availability...")
        
        # Check if CUDA is available
        cuda_available = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cuda_available = True
                print("  ✅ NVIDIA GPU detected via nvidia-smi")
                # Extract GPU info
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line:
                        gpu_name = line.split('|')[1].strip() if '|' in line else line.strip()
                        print(f"  🎮 GPU: {gpu_name}")
                        break
            else:
                print("  ❌ nvidia-smi not found or failed")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  ⚠️  Could not run nvidia-smi: {e}")
        
        # Try XGBoost GPU detection
        try:
            print(f"  📦 XGBoost version: {xgb.__version__}")
            
            # Test GPU training capability
            try:
                import numpy as np
                test_data = np.random.rand(10, 5)
                test_labels = np.random.rand(10)
                test_dmatrix = xgb.DMatrix(test_data, label=test_labels)
                
                gpu_params = {
                    'objective': 'reg:squarederror',
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'verbosity': 0
                }
                
                print("  🧪 Testing GPU training...")
                test_model = xgb.train(gpu_params, test_dmatrix, num_boost_round=1, verbose_eval=False)
                self.use_gpu = True
                print("  🚀 GPU acceleration VERIFIED and enabled!")
                print(f"  💾 GPU will be used for XGBoost training (gpu_id={self.gpu_id})")
                
                # Clean up
                del test_model, test_dmatrix, test_data, test_labels
                
            except Exception as gpu_test_error:
                print(f"  ❌ GPU training test failed: {gpu_test_error}")
                if self.force_gpu:
                    print("  🔧 Force GPU flag set - enabling GPU despite test failure")
                    self.use_gpu = True
                else:
                    print("  💻 Falling back to CPU")
                    self.use_gpu = False
                
        except ImportError:
            print("  ❌ XGBoost not installed")
            self.use_gpu = False
        except Exception as e:
            print(f"  ❌ GPU detection failed: {e}")
            self.use_gpu = False
        
        # Final status
        if self.use_gpu:
            print("  🎯 FINAL STATUS: GPU acceleration ENABLED")
        else:
            print("  🎯 FINAL STATUS: CPU-only mode")
    
    def find_existing_results_file(self):
        """Find the existing spatial validation results file"""
        pattern = os.path.join(self.results_dir, 'parquet_spatial_fold_results_*')
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No results file found matching pattern: {pattern}")
        
        if len(files) > 1:
            files.sort(key=os.path.getmtime, reverse=True)
            print(f"⚠️  Multiple results files found, using most recent: {os.path.basename(files[0])}")
        
        return files[0]
    
    def load_cluster_assignments(self):
        """Load cluster assignments from specified file or latest clustering results"""
        if self.cluster_file:
            if not os.path.exists(self.cluster_file):
                raise FileNotFoundError(f"Specified cluster file not found: {self.cluster_file}")
            latest_file = self.cluster_file
            print(f"📊 Using specified cluster file: {os.path.basename(latest_file)}")
        else:
            # Auto-detect latest clustering results
            cluster_files = []
            
            # Try flexible clustering results (new format)
            flexible_files = sorted(glob.glob('../evaluation/clustering_results/*/flexible_site_clusters_*.csv'))
            cluster_files.extend(flexible_files)
            
            # Also try legacy advanced clustering results
            advanced_files = sorted(glob.glob('../evaluation/clustering_results/advanced_site_clusters_*.csv'))
            cluster_files.extend(advanced_files)
            
            if not cluster_files:
                raise FileNotFoundError("No cluster assignment files found")
            
            # Sort by modification time to get the most recent
            cluster_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = cluster_files[0]
            print(f"📊 Auto-detected latest cluster file: {os.path.basename(latest_file)}")
        
        clusters_df = pd.read_csv(latest_file)
        cluster_assignments = dict(zip(clusters_df['site'], clusters_df['cluster']))
        
        print(f"✅ Loaded {len(cluster_assignments)} site assignments")
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} sites")
        
        return cluster_assignments, cluster_counts
    
    def prepare_features(self, df):
        """Prepare features for training (same as used in cluster training)"""
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', 'ecosystem_cluster']
        
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
    
    def create_sample_libsvm_files(self, cluster_sites, cluster_id):
        """Create sample train/test libsvm files for optimization"""
        temp_dir = os.path.join(self.results_dir, f'temp_optimization_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use first 3 sites for optimization
        sample_sites = cluster_sites[:3]
        if len(sample_sites) < 2:
            raise ValueError(f"Not enough sites for cluster {cluster_id} optimization")
        
        test_site = sample_sites[0]  # Use first site as test
        train_sites = sample_sites[1:]  # Rest as training
        
        train_file = os.path.join(temp_dir, f'train_cluster_{cluster_id}.svm')
        test_file = os.path.join(temp_dir, f'test_cluster_{cluster_id}.svm')
        
        # Create training file
        with open(train_file, 'w') as f_train:
            for site in train_sites:
                # Handle both cases: site names with and without .parquet extension
                if site.endswith('.parquet'):
                    parquet_file = os.path.join(self.parquet_dir, site)
                else:
                    parquet_file = os.path.join(self.parquet_dir, f"{site}.parquet")
                
                if os.path.exists(parquet_file):
                    self._process_site_to_libsvm(parquet_file, f_train)
        
        # Create test file
        with open(test_file, 'w') as f_test:
            # Handle both cases: site names with and without .parquet extension
            if test_site.endswith('.parquet'):
                test_parquet = os.path.join(self.parquet_dir, test_site)
            else:
                test_parquet = os.path.join(self.parquet_dir, f"{test_site}.parquet")
            
            if os.path.exists(test_parquet):
                self._process_site_to_libsvm(test_parquet, f_test)
        
        return train_file, test_file, temp_dir
    
    def _process_site_to_libsvm(self, parquet_file, output_file):
        """Process a single site parquet file and write to libsvm format"""
        try:
            # Read parquet file in chunks to manage memory
            import pyarrow.parquet as pq
            parquet_table = pq.read_table(parquet_file)
            total_rows = len(parquet_table)
            chunk_size = 10000  # Smaller chunks for optimization
            
            processed_rows = 0
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk_table = parquet_table.slice(start_idx, end_idx - start_idx)
                df_chunk = chunk_table.to_pandas()
                
                # Drop missing targets
                df_chunk = df_chunk.dropna(subset=[self.target_col])
                
                if len(df_chunk) == 0:
                    del df_chunk
                    continue
                
                # Sample to reduce size for optimization (take every 10th row)
                if len(df_chunk) > 100:
                    df_chunk = df_chunk.iloc[::10]
                
                # Prepare features
                X, y, _ = self.prepare_features(df_chunk)
                
                # Write to libsvm format
                for i in range(len(X)):
                    line_parts = [str(y[i])]
                    for j, value in enumerate(X[i]):
                        if value != 0:  # Sparse format
                            line_parts.append(f"{j}:{value}")
                    output_file.write(' '.join(line_parts) + '\n')
                
                processed_rows += len(X)
                
                # Clean up
                del df_chunk, X, y
                gc.collect()
            
            # Clean up
            del parquet_table
            gc.collect()
            
            return processed_rows
            
        except Exception as e:
            print(f"      ❌ Error processing {parquet_file}: {e}")
            return 0
    
    def _load_targets_from_libsvm(self, libsvm_file):
        """Load only target values from libsvm file"""
        targets = []
        with open(libsvm_file, 'r') as f:
            for line in f:
                if line.strip():
                    target = float(line.split()[0])
                    targets.append(target)
        return np.array(targets)
    
    def optimize_cluster_hyperparameters(self, cluster_id, sample_train_file, sample_test_file):
        """Run hyperparameter optimization for a cluster using Optuna"""
        try:
            import optuna
        except ImportError:
            print("❌ Optuna not installed. Install with: pip install optuna")
            return None
        
        print(f"🔧 Optimizing hyperparameters for cluster {cluster_id} ({'GPU-accelerated' if self.use_gpu else 'CPU-only'})...")
        
        def objective(trial):
            # GPU-optimized parameter ranges
            if self.use_gpu:
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'gpu_hist',
                    'gpu_id': self.gpu_id,
                    'max_depth': trial.suggest_int('max_depth', 6, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 6),
                    'random_state': 42,
                    'verbosity': 0
                }
            else:
                # CPU parameters
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'hist',
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 5),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
            
            dtrain = xgb.DMatrix(f"{sample_train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{sample_test_file}?format=libsvm")
            
            # GPU can handle more trees efficiently
            if self.use_gpu:
                num_boost_round = trial.suggest_int('n_estimators', 100, 300)
            else:
                num_boost_round = trial.suggest_int('n_estimators', 50, 150)
            
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=False
            )
            
            y_pred = model.predict(dtest)
            y_actual = self._load_targets_from_libsvm(sample_test_file)
            
            return r2_score(y_actual, y_pred)
        
        study = optuna.create_study(direction='maximize', 
                                  study_name=f'cluster_{cluster_id}_optimization')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"  🎯 Best R² for cluster {cluster_id}: {study.best_value:.4f}")
        print(f"  📊 Best parameters: {study.best_params}")
        
        return study.best_params
    
    def run_optimization(self):
        """Run hyperparameter optimization for all clusters"""
        print("\n🔧 HYPERPARAMETER OPTIMIZATION")
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
        
        # Optimize each cluster
        for cluster_id in clusters:
            if cluster_id not in sites_by_cluster:
                print(f"⚠️  No sites found for cluster {cluster_id}")
                continue
            
            cluster_sites = sites_by_cluster[cluster_id]
            
            try:
                # Filter sites that have parquet files
                available_sites = []
                for site in cluster_sites:
                    # Handle both cases: site names with and without .parquet extension
                    if site.endswith('.parquet'):
                        parquet_file = os.path.join(self.parquet_dir, site)
                        site_name = site
                    else:
                        parquet_file = os.path.join(self.parquet_dir, f"{site}.parquet")
                        site_name = site
                    
                    if os.path.exists(parquet_file):
                        available_sites.append(site_name)
                
                if len(available_sites) < 2:
                    print(f"⚠️  Not enough sites with data for cluster {cluster_id} optimization")
                    continue
                
                print(f"\n🎯 Optimizing cluster {cluster_id} ({len(available_sites)} sites available)")
                
                # Create sample train/test files
                train_file, test_file, temp_dir = self.create_sample_libsvm_files(available_sites, cluster_id)
                
                # Run optimization
                optimal_params = self.optimize_cluster_hyperparameters(
                    cluster_id, train_file, test_file
                )
                
                if optimal_params:
                    self.optimized_params[str(int(cluster_id))] = optimal_params
                
                # Clean up temp files
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
            except Exception as e:
                print(f"❌ Error optimizing cluster {cluster_id}: {e}")
                continue
        
        # Save results
        if self.optimized_params:
            print(f"\n✅ Optimization completed for {len(self.optimized_params)} clusters")
            
            # Save optimized parameters
            opt_params_file = os.path.join(self.results_dir, f'optimized_params_{self.timestamp}.json')
            with open(opt_params_file, 'w') as f:
                json.dump(self.optimized_params, f, indent=2)
            print(f"💾 Optimized parameters saved to: {opt_params_file}")
            
            # Print summary
            print(f"\n📊 OPTIMIZATION SUMMARY")
            print("="*30)
            for cluster_id, params in self.optimized_params.items():
                print(f"Cluster {cluster_id}:")
                print(f"  Max depth: {params.get('max_depth', 'N/A')}")
                print(f"  Learning rate: {params.get('learning_rate', 'N/A'):.3f}")
                print(f"  N estimators: {params.get('n_estimators', 'N/A')}")
            
            return True
        else:
            print("❌ No clusters were successfully optimized")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Standalone Hyperparameter Optimization")
    parser.add_argument('--results-dir', required=True,
                        help="Directory containing spatial validation results")
    parser.add_argument('--parquet-dir', default='../../parquet_ecological',
                        help="Directory containing parquet files")
    parser.add_argument('--cluster-file', default=None,
                        help="Path to specific cluster assignment CSV file")
    parser.add_argument('--n-trials', type=int, default=None,
                        help="Number of optimization trials per cluster (default: 100 for GPU, 50 for CPU)")
    parser.add_argument('--force-gpu', action='store_true',
                        help="Force GPU usage even if detection fails")
    
    args = parser.parse_args()
    
    try:
        optimizer = HyperparameterOptimizer(
            results_dir=args.results_dir,
            parquet_dir=args.parquet_dir,
            cluster_file=args.cluster_file,
            n_trials=args.n_trials,
            force_gpu=args.force_gpu
        )
        
        success = optimizer.run_optimization()
        
        if success:
            print(f"\n🎉 Hyperparameter optimization completed successfully!")
        else:
            print(f"\n❌ Hyperparameter optimization failed!")
            
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()
