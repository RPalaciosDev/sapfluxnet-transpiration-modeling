"""
Train Cluster Models with Optimized Hyperparameters

This script trains new XGBoost models for each ecosystem cluster using 
the hyperparameters optimized by hyperparameter_optimizer.py.

Usage:
    python train_optimized_cluster_models.py --optimized-params ./results/parquet_spatial_validation/hybrid_v3_20250806_202623/optimized_params_20250806_215855.json
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
import json
import argparse
from datetime import datetime
import warnings
import gc
import subprocess

warnings.filterwarnings('ignore')

class OptimizedClusterTrainer:
    """Train cluster models using optimized hyperparameters"""
    
    def __init__(self, optimized_params_file, parquet_dir='../../parquet_ecological',
                 cluster_file=None, output_dir=None, force_gpu=False):
        
        self.parquet_dir = parquet_dir
        self.cluster_file = cluster_file
        self.force_gpu = force_gpu
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.target_col = 'sap_flow'
        
        # Load optimized parameters
        with open(optimized_params_file, 'r') as f:
            self.optimized_params = json.load(f)
        
        print(f"✅ Loaded optimized parameters for {len(self.optimized_params)} clusters")
        
        # Set output directory
        if output_dir is None:
            self.output_dir = f'./results/optimized_cluster_models_{self.timestamp}'
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GPU Detection
        self.use_gpu = False
        self.gpu_id = 0
        self.detect_gpu()
        
        print(f"🤖 Optimized Cluster Trainer initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚡ GPU Acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
    
    def detect_gpu(self):
        """Detect GPU availability for XGBoost"""
        print("🔍 Checking GPU and CUDA availability...")
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("  ✅ NVIDIA GPU detected via nvidia-smi")
            else:
                print("  ❌ nvidia-smi not found or failed")
        except:
            print("  ⚠️  Could not run nvidia-smi")
        
        # Try XGBoost GPU detection
        try:
            print(f"  📦 XGBoost version: {xgb.__version__}")
            
            # Test GPU training capability
            try:
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
                
        except Exception as e:
            print(f"  ❌ GPU detection failed: {e}")
            self.use_gpu = False
        
        if self.use_gpu:
            print("  🎯 FINAL STATUS: GPU acceleration ENABLED")
        else:
            print("  🎯 FINAL STATUS: CPU-only mode")
    
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
    
    def load_cluster_data(self, cluster_sites, cluster_id):
        """Load and combine data for a cluster"""
        print(f"📊 Loading data for cluster {cluster_id} ({len(cluster_sites)} sites)...")
        
        cluster_data = []
        successful_sites = []
        
        for site in cluster_sites:
            # Handle both cases: site names with and without .parquet extension
            if site.endswith('.parquet'):
                parquet_file = os.path.join(self.parquet_dir, site)
            else:
                parquet_file = os.path.join(self.parquet_dir, f"{site}.parquet")
            
            if not os.path.exists(parquet_file):
                print(f"  ⚠️  Missing parquet file: {parquet_file}")
                continue
            
            try:
                df_site = pd.read_parquet(parquet_file)
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) == 0:
                    print(f"  ⚠️  No valid data for {site}")
                    continue
                
                # Add site identifier
                df_site['site'] = site
                cluster_data.append(df_site)
                successful_sites.append(site)
                
                print(f"    ✅ {site}: {len(df_site):,} rows")
                
                del df_site
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            raise ValueError(f"No valid data loaded for cluster {cluster_id}")
        
        # Combine all site data
        combined_df = pd.concat(cluster_data, ignore_index=True)
        print(f"  ✅ Combined data: {len(combined_df):,} rows from {len(successful_sites)} sites")
        
        # Clean up
        del cluster_data
        gc.collect()
        
        return combined_df, successful_sites
    
    def train_cluster_model(self, cluster_id, cluster_sites):
        """Train a model for a specific cluster using optimized parameters"""
        print(f"\n{'='*60}")
        print(f"TRAINING CLUSTER {cluster_id} MODEL")
        print(f"{'='*60}")
        
        # Check if we have optimized parameters for this cluster
        cluster_key = str(cluster_id)
        if cluster_key not in self.optimized_params:
            print(f"⚠️  No optimized parameters found for cluster {cluster_id}, skipping")
            return None
        
        try:
            # Load cluster data
            cluster_df, successful_sites = self.load_cluster_data(cluster_sites, cluster_id)
            
            if len(successful_sites) < 2:
                print(f"⚠️  Not enough sites for cluster {cluster_id}, need at least 2")
                return None
            
            # Prepare features
            X, y, feature_cols = self.prepare_features(cluster_df)
            print(f"  📊 Features: {len(feature_cols)}, Samples: {len(X):,}")
            
            # Get optimized parameters
            opt_params = self.optimized_params[cluster_key].copy()
            
            # Add GPU/CPU specific parameters
            if self.use_gpu:
                opt_params.update({
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'gpu_hist',
                    'gpu_id': self.gpu_id,
                    'random_state': 42,
                    'verbosity': 1
                })
                print(f"  🚀 Using GPU acceleration with optimized parameters")
            else:
                opt_params.update({
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'hist',
                    'n_jobs': -1,
                    'random_state': 42,
                    'verbosity': 1
                })
                print(f"  💻 Using CPU with optimized parameters")
            
            # Extract n_estimators separately
            n_estimators = opt_params.pop('n_estimators', 100)
            
            print(f"  🎯 Optimized parameters:")
            print(f"    Max depth: {opt_params['max_depth']}")
            print(f"    Learning rate: {opt_params['learning_rate']:.4f}")
            print(f"    N estimators: {n_estimators}")
            print(f"    Subsample: {opt_params['subsample']:.4f}")
            print(f"    Colsample bytree: {opt_params['colsample_bytree']:.4f}")
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
            
            # Train model
            print(f"  🔧 Training model...")
            model = xgb.train(
                params=opt_params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                verbose_eval=False
            )
            
            # Make predictions and calculate metrics
            y_pred = model.predict(dtrain)
            
            metrics = {
                'cluster': cluster_id,
                'sites': len(successful_sites),
                'samples': len(X),
                'features': len(feature_cols),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
            
            print(f"  📊 Model Performance:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    R²: {metrics['r2']:.4f}")
            
            # Save model
            model_file = os.path.join(self.output_dir, f'optimized_xgb_model_cluster_{cluster_id}_{self.timestamp}.json')
            model.save_model(model_file)
            print(f"  💾 Model saved: {os.path.basename(model_file)}")
            
            # Save feature importance
            importance_file = os.path.join(self.output_dir, f'optimized_feature_importance_cluster_{cluster_id}_{self.timestamp}.csv')
            importance_dict = model.get_score(importance_type='weight')
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance_dict.get(feature, 0)}
                for feature in feature_cols
            ]).sort_values('importance', ascending=False)
            importance_df.to_csv(importance_file, index=False)
            print(f"  📊 Feature importance saved: {os.path.basename(importance_file)}")
            
            # Clean up
            del cluster_df, X, y, dtrain, model
            gc.collect()
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error training cluster {cluster_id}: {e}")
            return None
    
    def train_all_clusters(self):
        """Train models for all clusters with optimized parameters"""
        print(f"🤖 TRAINING OPTIMIZED CLUSTER MODELS")
        print("="*60)
        print(f"Started at: {datetime.now()}")
        
        # Load cluster assignments
        cluster_assignments, cluster_counts = self.load_cluster_assignments()
        
        # Group sites by cluster
        sites_by_cluster = {}
        for site, cluster_id in cluster_assignments.items():
            if cluster_id not in sites_by_cluster:
                sites_by_cluster[cluster_id] = []
            sites_by_cluster[cluster_id].append(site)
        
        # Train models for each cluster
        all_metrics = []
        
        for cluster_id in sorted(sites_by_cluster.keys()):
            cluster_sites = sites_by_cluster[cluster_id]
            
            metrics = self.train_cluster_model(cluster_id, cluster_sites)
            if metrics:
                all_metrics.append(metrics)
        
        # Save summary
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            summary_file = os.path.join(self.output_dir, f'optimized_training_summary_{self.timestamp}.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"\n💾 Training summary saved: {summary_file}")
            
            # Print overall summary
            print(f"\n📊 TRAINING SUMMARY")
            print("="*30)
            print(f"Models trained: {len(all_metrics)}")
            print(f"Average R²: {summary_df['r2'].mean():.4f}")
            print(f"Average RMSE: {summary_df['rmse'].mean():.4f}")
            
            for _, row in summary_df.iterrows():
                print(f"Cluster {int(row['cluster'])}: R² = {row['r2']:.4f}, Sites = {int(row['sites'])}")
        
        print(f"\n✅ Optimized cluster training completed!")
        return all_metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Cluster Models with Optimized Hyperparameters")
    parser.add_argument('--optimized-params', required=True,
                        help="Path to optimized parameters JSON file")
    parser.add_argument('--parquet-dir', default='../../parquet_ecological',
                        help="Directory containing parquet files")
    parser.add_argument('--cluster-file', default=None,
                        help="Path to specific cluster assignment CSV file")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory for trained models")
    parser.add_argument('--force-gpu', action='store_true',
                        help="Force GPU usage even if detection fails")
    
    args = parser.parse_args()
    
    try:
        trainer = OptimizedClusterTrainer(
            optimized_params_file=args.optimized_params,
            parquet_dir=args.parquet_dir,
            cluster_file=args.cluster_file,
            output_dir=args.output_dir,
            force_gpu=args.force_gpu
        )
        
        metrics = trainer.train_all_clusters()
        
        if metrics:
            print(f"\n🎉 Successfully trained {len(metrics)} optimized cluster models!")
        else:
            print(f"\n❌ No models were successfully trained!")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
