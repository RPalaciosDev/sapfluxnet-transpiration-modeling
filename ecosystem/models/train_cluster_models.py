"""
Memory-Optimized Cluster-Specific XGBoost Training for SAPFLUXNET Data
Trains separate XGBoost models for each ecosystem cluster with external memory support
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.model_selection import train_test_split
import os
import glob
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

def get_total_memory_gb():
    """Get total system memory in GB"""
    return psutil.virtual_memory().total / (1024**3)

def calculate_optimal_memory_usage():
    """Calculate optimal memory usage based on available RAM"""
    total_memory = get_total_memory_gb()
    available_memory = get_available_memory_gb()
    
    print(f"💾 Memory Analysis:")
    print(f"  Total RAM: {total_memory:.1f} GB")
    print(f"  Available RAM: {available_memory:.1f} GB")
    print(f"  Used RAM: {total_memory - available_memory:.1f} GB")
    
    # Use 70% of available memory for optimal performance
    optimal_usage = available_memory * 0.7
    print(f"  Recommended usage: {optimal_usage:.1f} GB (70% of available)")
    
    return optimal_usage

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")

def check_disk_space_gb(path):
    """Check available disk space in GB"""
    try:
        if os.name == 'nt':  # Windows
            import shutil
            total, used, free = shutil.disk_usage(path)
            return free / (1024**3)
        else:  # Unix/Linux
            statvfs = os.statvfs(path)
            available_bytes = statvfs.f_bavail * statvfs.f_frsize
            return available_bytes / (1024**3)
    except:
        return 0

class MemoryOptimizedClusterTrainer:
    """
    Memory-optimized trainer for cluster-specific XGBoost models
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', results_dir='./results/cluster_models'):
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🚀 Memory-Optimized Cluster Trainer initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"🎯 Target column: {self.target_col}")
        print(f"🏷️  Cluster column: {self.cluster_col}")
    
    def analyze_data_requirements(self):
        """Analyze data size and memory requirements"""
        print("\n📊 Analyzing data requirements...")
        
        parquet_files = [f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')]
        total_files = len(parquet_files)
        
        # Sample a few files to estimate size
        sample_files = parquet_files[:min(5, total_files)]
        total_rows = 0
        total_size_mb = 0
        
        for parquet_file in sample_files:
            file_path = os.path.join(self.parquet_dir, parquet_file)
            try:
                # Get file size
                file_size_mb = os.path.getsize(file_path) / (1024**2)
                total_size_mb += file_size_mb
                
                # Sample rows
                df_sample = pd.read_parquet(file_path, columns=[self.cluster_col, self.target_col])
                total_rows += len(df_sample)
                
                print(f"  📄 {parquet_file}: {len(df_sample):,} rows, {file_size_mb:.1f} MB")
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error analyzing {parquet_file}: {e}")
        
        # Estimate total data size
        avg_size_mb = total_size_mb / len(sample_files)
        estimated_total_size_gb = (avg_size_mb * total_files) / 1024
        
        avg_rows = total_rows / len(sample_files)
        estimated_total_rows = int(avg_rows * total_files)
        
        print(f"\n📈 Data size estimates:")
        print(f"  Total files: {total_files}")
        print(f"  Estimated total rows: {estimated_total_rows:,}")
        print(f"  Estimated total size: {estimated_total_size_gb:.1f} GB")
        
        return estimated_total_size_gb, estimated_total_rows, total_files
    
    def load_cluster_info_memory_efficient(self):
        """Load cluster information without loading full data"""
        print("\n🔍 Loading cluster information...")
        
        cluster_info = {}
        parquet_files = sorted([f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')])
        
        for parquet_file in parquet_files:
            site_name = parquet_file.replace('_comprehensive.parquet', '')
            file_path = os.path.join(self.parquet_dir, parquet_file)
            
            try:
                # Load only cluster and target columns for analysis
                df_sample = pd.read_parquet(file_path, columns=[self.cluster_col, self.target_col])
                
                if self.cluster_col not in df_sample.columns:
                    print(f"  ⚠️  {site_name}: Missing {self.cluster_col} column")
                    continue
                
                cluster_id = df_sample[self.cluster_col].iloc[0]  # All rows should have same cluster
                valid_rows = len(df_sample.dropna(subset=[self.target_col]))
                
                if cluster_id not in cluster_info:
                    cluster_info[cluster_id] = {'sites': [], 'total_rows': 0}
                
                cluster_info[cluster_id]['sites'].append(site_name)
                cluster_info[cluster_id]['total_rows'] += valid_rows
                
                print(f"  ✅ {site_name}: Cluster {cluster_id}, {valid_rows:,} valid rows")
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error loading {site_name}: {e}")
                continue
        
        print(f"\n📊 Cluster distribution:")
        for cluster_id, info in sorted(cluster_info.items()):
            print(f"  Cluster {cluster_id}: {len(info['sites'])} sites, {info['total_rows']:,} total rows")
            print(f"    Sites: {', '.join(info['sites'][:5])}{'...' if len(info['sites']) > 5 else ''}")
        
        return cluster_info
    
    def prepare_cluster_data_external_memory(self, cluster_id, cluster_sites, temp_dir):
        """Prepare cluster data in external memory format"""
        print(f"\n🔧 Preparing external memory data for cluster {cluster_id}...")
        
        # Create temporary libsvm file for this cluster
        cluster_file = os.path.join(temp_dir, f'cluster_{cluster_id}_data.svm')
        
        all_features = []
        total_rows = 0
        
        with open(cluster_file, 'w') as output_file:
            for site in cluster_sites:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                
                if not os.path.exists(parquet_file):
                    print(f"    ⚠️  Missing: {parquet_file}")
                    continue
                
                try:
                    # Load site data
                    df_site = pd.read_parquet(parquet_file)
                    
                    # Filter for this cluster and valid target
                    df_site = df_site[df_site[self.cluster_col] == cluster_id]
                    df_site = df_site.dropna(subset=[self.target_col])
                    
                    if len(df_site) == 0:
                        continue
                    
                    # Prepare features
                    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
                    feature_cols = [col for col in df_site.columns 
                                   if col not in exclude_cols + [self.target_col]
                                   and not col.endswith('_flags')
                                   and not col.endswith('_md')]
                    
                    if not all_features:
                        all_features = feature_cols
                        print(f"    📊 Using {len(all_features)} features")
                    
                    # Extract features and target
                    X = df_site[feature_cols].fillna(0).values
                    y = df_site[self.target_col].values
                    
                    # Convert to libsvm format and append to file
                    for i in range(len(X)):
                        line_parts = [str(y[i])]
                        for j, value in enumerate(X[i]):
                            if value != 0:  # Sparse format
                                line_parts.append(f"{j}:{value}")
                        output_file.write(' '.join(line_parts) + '\n')
                    
                    total_rows += len(X)
                    print(f"    ✅ {site}: {len(X):,} rows")
                    
                    del df_site, X, y
                    gc.collect()
                    
                except Exception as e:
                    print(f"    ❌ Error processing {site}: {e}")
                    continue
        
        print(f"  📄 Created external memory file: {cluster_file}")
        print(f"  📊 Total rows: {total_rows:,}, Features: {len(all_features)}")
        
        return cluster_file, all_features, total_rows
    
    def train_cluster_model_external_memory(self, cluster_id, cluster_file, feature_cols, total_rows):
        """Train XGBoost model using external memory"""
        print(f"\n🚀 Training external memory model for cluster {cluster_id}...")
        
        # Load data for train/test split
        print("  📊 Loading data for train/test split...")
        X, y = load_svmlight_file(cluster_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Create temporary files for train/test
        temp_train_file = cluster_file.replace('.svm', '_train.svm')
        temp_test_file = cluster_file.replace('.svm', '_test.svm')
        
        dump_svmlight_file(X_train, y_train, temp_train_file)
        dump_svmlight_file(X_test, y_test, temp_test_file)
        
        print(f"  📊 Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")
        
        # Clean up arrays
        del X, y, X_train, X_test, y_train, y_test
        gc.collect()
        
        # XGBoost parameters optimized for memory
        available_memory = get_available_memory_gb()
        if available_memory > 8:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'max_bin': 256,
                'verbosity': 1,
                'nthread': -1
            }
        else:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'max_bin': 128,
                'verbosity': 1,
                'nthread': -1
            }
        
        print(f"  🔧 XGBoost parameters: max_depth={params['max_depth']}, max_bin={params['max_bin']}")
        
        # Create DMatrix objects for external memory training
        dtrain = xgb.DMatrix(f"{temp_train_file}?format=libsvm")
        dtest = xgb.DMatrix(f"{temp_test_file}?format=libsvm")
        
        # Train model
        print(f"  🏋️  Training XGBoost model...")
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=15,
            verbose_eval=20
        )
        
        # Make predictions
        print(f"  📊 Making predictions...")
        y_pred_train = model.predict(dtrain)
        y_pred_test = model.predict(dtest)
        
        # Load actual targets for metrics
        _, y_train_actual = load_svmlight_file(temp_train_file)
        _, y_test_actual = load_svmlight_file(temp_test_file)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))
        train_r2 = r2_score(y_train_actual, y_pred_train)
        test_r2 = r2_score(y_test_actual, y_pred_test)
        train_mae = mean_absolute_error(y_train_actual, y_pred_train)
        test_mae = mean_absolute_error(y_test_actual, y_pred_test)
        
        metrics = {
            'cluster': cluster_id,
            'total_rows': total_rows,
            'train_samples': len(y_train_actual),
            'test_samples': len(y_test_actual),
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_count': len(feature_cols),
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else 200
        }
        
        print(f"  📊 Cluster {cluster_id} Results:")
        print(f"    Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"    Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        
        # Save model
        model_path = os.path.join(self.results_dir, f'xgb_model_cluster_{cluster_id}_{self.timestamp}.json')
        model.save_model(model_path)
        print(f"  💾 Model saved: {model_path}")
        
        # Save feature importance
        importance = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature_index': list(importance.keys()),
            'feature_name': [feature_cols[int(k[1:])] if k.startswith('f') and int(k[1:]) < len(feature_cols) else k 
                           for k in importance.keys()],
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(self.results_dir, f'feature_importance_cluster_{cluster_id}_{self.timestamp}.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"  💾 Feature importance saved: {importance_path}")
        
        # Clean up temporary files
        for temp_file in [temp_train_file, temp_test_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Clean up memory
        del dtrain, dtest, model, y_pred_train, y_pred_test, y_train_actual, y_test_actual
        gc.collect()
        
        return metrics
    
    def train_all_cluster_models(self):
        """Train models for all clusters with memory optimization"""
        print("🚀 Starting Memory-Optimized Cluster Model Training")
        print("=" * 60)
        
        # Analyze data requirements
        estimated_size_gb, estimated_rows, total_files = self.analyze_data_requirements()
        optimal_memory = calculate_optimal_memory_usage()
        
        # Check if we need external memory approach
        use_external_memory = estimated_size_gb > optimal_memory
        print(f"\n💾 Memory strategy: {'External Memory' if use_external_memory else 'In-Memory'}")
        
        # Load cluster information
        cluster_info = self.load_cluster_info_memory_efficient()
        
        if not cluster_info:
            raise ValueError("No clusters found! Make sure parquet files have ecosystem_cluster column.")
        
        # Set up temporary directory for external memory files
        available_space = check_disk_space_gb('.')
        print(f"💾 Available disk space: {available_space:.1f} GB")
        
        temp_dir = os.path.join(self.results_dir, 'temp_external_memory')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            all_metrics = []
            
            # Train model for each cluster
            for cluster_id, info in sorted(cluster_info.items()):
                print(f"\n{'='*60}")
                print(f"TRAINING CLUSTER {cluster_id} MODEL")
                print(f"{'='*60}")
                print(f"Sites: {len(info['sites'])}, Total rows: {info['total_rows']:,}")
                
                log_memory_usage(f"Before cluster {cluster_id}")
                
                # Prepare external memory data
                cluster_file, feature_cols, total_rows = self.prepare_cluster_data_external_memory(
                    cluster_id, info['sites'], temp_dir
                )
                
                if total_rows == 0:
                    print(f"  ⚠️  No valid data for cluster {cluster_id}, skipping...")
                    continue
                
                # Train model
                metrics = self.train_cluster_model_external_memory(
                    cluster_id, cluster_file, feature_cols, total_rows
                )
                
                all_metrics.append(metrics)
                
                # Clean up cluster file
                if os.path.exists(cluster_file):
                    os.remove(cluster_file)
                
                log_memory_usage(f"After cluster {cluster_id}")
                gc.collect()
            
            # Save summary metrics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_path = os.path.join(self.results_dir, f'cluster_model_metrics_{self.timestamp}.csv')
                metrics_df.to_csv(metrics_path, index=False)
                print(f"\n💾 All metrics saved: {metrics_path}")
                
                # Print summary
                print(f"\n📊 TRAINING SUMMARY")
                print(f"=" * 30)
                print(f"Total clusters trained: {len(all_metrics)}")
                print(f"Average test R²: {metrics_df['test_r2'].mean():.4f} ± {metrics_df['test_r2'].std():.4f}")
                print(f"Average test RMSE: {metrics_df['test_rmse'].mean():.4f} ± {metrics_df['test_rmse'].std():.4f}")
                
                for _, row in metrics_df.iterrows():
                    print(f"  Cluster {int(row['cluster'])}: R² = {row['test_r2']:.4f}, RMSE = {row['test_rmse']:.4f}")
                
                print(f"\n✅ Memory-optimized cluster training completed successfully!")
                return metrics_df
            else:
                print(f"\n❌ No models were trained successfully")
                return None
                
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                print(f"\n🧹 Cleaning up temporary files...")
                try:
                    shutil.rmtree(temp_dir)
                    print("✅ Temporary files cleaned up")
                except Exception as e:
                    print(f"⚠️  Warning: Could not clean up temp directory: {e}")

def main():
    """Main function to run memory-optimized cluster training"""
    print("🌍 SAPFLUXNET Memory-Optimized Cluster Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Initialize trainer
        trainer = MemoryOptimizedClusterTrainer()
        
        # Train all cluster models
        metrics_df = trainer.train_all_cluster_models()
        
        if metrics_df is not None:
            print(f"\n🎉 Training completed successfully!")
            print(f"📁 Results saved to: {trainer.results_dir}")
        else:
            print(f"\n❌ Training failed - no models were created")
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main()
