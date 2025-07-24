"""
Simplified Cluster-Specific XGBoost Training for SAPFLUXNET Data
Trains separate XGBoost models for each ecosystem cluster from preprocessed libsvm data
Preprocessing should be done separately using preprocess_cluster_data.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_svmlight_file
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
import argparse

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

class ClusterModelTrainer:
    """
    Simplified trainer for cluster-specific XGBoost models
    Trains from preprocessed libsvm files only
    """
    
    def __init__(self, results_dir='./results/cluster_models'):
        self.results_dir = results_dir
        self.preprocessed_dir = os.path.join(results_dir, 'preprocessed_libsvm')
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🚀 Cluster Model Trainer initialized")
        print(f"📁 Results directory: {results_dir}")
        print(f"📁 Preprocessed directory: {self.preprocessed_dir}")

    def check_preprocessed_files(self):
        """Check for available preprocessed files"""
        print("\n🔍 Checking for preprocessed files...")
        
        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessed directory not found: {self.preprocessed_dir}")
        
        # Find all libsvm and metadata files
        libsvm_files = glob.glob(os.path.join(self.preprocessed_dir, 'cluster_*_clean.svm'))
        metadata_files = glob.glob(os.path.join(self.preprocessed_dir, 'cluster_*_metadata.json'))
        
        if not libsvm_files:
            raise FileNotFoundError(f"No preprocessed libsvm files found in {self.preprocessed_dir}")
        
        available_clusters = {}
        
        for libsvm_file in libsvm_files:
            # Extract cluster ID from filename
            filename = os.path.basename(libsvm_file)
            # Format: cluster_{cluster_id}_clean.svm
            try:
                cluster_id = int(filename.split('_')[1])
            except (IndexError, ValueError):
                print(f"  ⚠️  Could not extract cluster ID from {filename}")
                continue
                
            metadata_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_metadata.json')
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
                    available_clusters[cluster_id] = {
                        'libsvm_file': libsvm_file,
                        'metadata': metadata,
                        'size_mb': file_size_mb
                    }
                    print(f"  ✅ Cluster {cluster_id}: {file_size_mb:.1f} MB, {metadata['total_rows']:,} rows, {metadata['feature_count']} features")
                except Exception as e:
                    print(f"  ❌ Error reading metadata for cluster {cluster_id}: {e}")
                    continue
            else:
                print(f"  ⚠️  Missing metadata file for cluster {cluster_id}")
                continue
        
        if not available_clusters:
            raise ValueError("No valid preprocessed clusters found!")
        
        print(f"\n📊 Found {len(available_clusters)} preprocessed clusters")
        total_size_mb = sum(info['size_mb'] for info in available_clusters.values())
        print(f"💾 Total data size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        
        return available_clusters

    def train_cluster_model_from_libsvm(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train XGBoost model from preprocessed libsvm file"""
        print(f"\n🚀 Training model for cluster {cluster_id}...")
        
        # Check if we should use external memory based on file size and available memory
        file_size_gb = os.path.getsize(libsvm_file) / (1024**3)
        available_memory = get_available_memory_gb()
        
        print(f"  📊 File size: {file_size_gb:.2f} GB")
        print(f"  💾 Available memory: {available_memory:.1f} GB")
        
        # Use external memory if file is large relative to available memory
        use_external_memory = file_size_gb > (available_memory * 0.3)
        
        if use_external_memory:
            print(f"  🔧 Using EXTERNAL MEMORY training")
            return self._train_external_memory(cluster_id, libsvm_file, feature_cols, total_rows)
        else:
            print(f"  🚀 Using IN-MEMORY training")
            return self._train_in_memory(cluster_id, libsvm_file, feature_cols, total_rows)

    def _train_in_memory(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train model in memory (for smaller datasets)"""
        print("  📊 Loading data into memory...")
        X, y = load_svmlight_file(libsvm_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"  📊 Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")
        
        # XGBoost parameters
        params = self._get_xgboost_params()
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
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
        
        # Make predictions and calculate metrics
        y_pred_train = model.predict(dtrain)
        y_pred_test = model.predict(dtest)
        
        metrics = self._calculate_metrics(cluster_id, total_rows, len(feature_cols),
                                        y_train, y_test, y_pred_train, y_pred_test)
        
        # Save model and importance
        self._save_model_and_importance(cluster_id, model, feature_cols, metrics)
        
        # Clean up memory
        del dtrain, dtest, model, X, y, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
        gc.collect()
        
        return metrics

    def _train_external_memory(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train model using external memory (for large datasets)"""
        # Create temporary directory for train/test splits
        temp_dir = os.path.join(self.results_dir, f'temp_training_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            print("  📊 Creating train/test splits...")
            
            # Create train/test split files
            train_file, test_file, train_samples, test_samples = self._create_external_train_test_split(
                libsvm_file, temp_dir, self.test_size, self.random_state
            )
            
            print(f"  📊 Train: {train_samples:,} samples, Test: {test_samples:,} samples")
            
            # XGBoost parameters
            params = self._get_xgboost_params()
            
            # Create DMatrix objects from files (external memory)
            dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
            
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
            y_train_actual = self._load_targets_from_libsvm(train_file)
            y_test_actual = self._load_targets_from_libsvm(test_file)
            
            metrics = self._calculate_metrics(cluster_id, total_rows, len(feature_cols),
                                            y_train_actual, y_test_actual, y_pred_train, y_pred_test)
            
            # Save model and importance
            self._save_model_and_importance(cluster_id, model, feature_cols, metrics)
            
            # Clean up memory
            del dtrain, dtest, model, y_pred_train, y_pred_test, y_train_actual, y_test_actual
            gc.collect()
            
            return metrics
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                print(f"  🧹 Cleaning up temporary files...")
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not clean up temp directory: {e}")

    def _get_xgboost_params(self):
        """Get XGBoost parameters optimized for available memory"""
        available_memory = get_available_memory_gb()
        if available_memory > 20:  # High memory system
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'max_bin': 512,
                'verbosity': 1,
                'nthread': -1
            }
        elif available_memory > 8:
            return {
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
            return {
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

    def _create_external_train_test_split(self, libsvm_file, temp_dir, test_size, random_state):
        """Create train/test split files without loading everything into memory"""
        train_file = os.path.join(temp_dir, 'train.svm')
        test_file = os.path.join(temp_dir, 'test.svm')
        
        # First pass: count total lines
        total_lines = 0
        with open(libsvm_file, 'r') as f:
            for line in f:
                if line.strip():
                    total_lines += 1
        
        # Calculate split indices
        np.random.seed(random_state)
        test_indices = set(np.random.choice(total_lines, size=int(total_lines * test_size), replace=False))
        
        # Second pass: split data
        train_samples = 0
        test_samples = 0
        
        with open(libsvm_file, 'r') as input_file, \
             open(train_file, 'w') as train_out, \
             open(test_file, 'w') as test_out:
            
            for i, line in enumerate(input_file):
                if line.strip():
                    if i in test_indices:
                        test_out.write(line)
                        test_samples += 1
                    else:
                        train_out.write(line)
                        train_samples += 1
        
        return train_file, test_file, train_samples, test_samples

    def _load_targets_from_libsvm(self, libsvm_file):
        """Load only target values from libsvm file (memory efficient)"""
        targets = []
        with open(libsvm_file, 'r') as f:
            for line in f:
                if line.strip():
                    target = float(line.split()[0])
                    targets.append(target)
        return np.array(targets)

    def _calculate_metrics(self, cluster_id, total_rows, feature_count, y_train, y_test, y_pred_train, y_pred_test):
        """Calculate training metrics"""
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        metrics = {
            'cluster': cluster_id,
            'total_rows': total_rows,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_count': feature_count,
            'best_iteration': 200
        }
        
        print(f"  📊 Cluster {cluster_id} Results:")
        print(f"    Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        print(f"    Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        
        return metrics

    def _save_model_and_importance(self, cluster_id, model, feature_cols, metrics):
        """Save model and feature importance"""
        # Save model
        model_path = os.path.join(self.results_dir, f'xgb_model_cluster_{cluster_id}_{self.timestamp}.json')
        model.save_model(model_path)
        print(f"  💾 Model saved: {model_path}")
        
        # Save feature importance
        try:
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
        except Exception as e:
            print(f"  ⚠️  Could not save feature importance: {e}")

    def train_all_cluster_models(self):
        """Train models for all available preprocessed clusters"""
        print("🚀 Starting Cluster Model Training from Preprocessed Data")
        print("=" * 60)
        
        # Check for preprocessed files
        available_clusters = self.check_preprocessed_files()
        
        all_metrics = []
        
        for cluster_id, cluster_info in sorted(available_clusters.items()):
            print(f"\n{'='*60}")
            print(f"TRAINING CLUSTER {cluster_id} MODEL")
            print(f"{'='*60}")
            
            cluster_file = cluster_info['libsvm_file']
            feature_cols = cluster_info['metadata']['feature_names']
            total_rows = cluster_info['metadata']['total_rows']
            
            print(f"File: {os.path.basename(cluster_file)}")
            print(f"Rows: {total_rows:,}, Features: {len(feature_cols)}")
            
            log_memory_usage(f"Before training cluster {cluster_id}")
            
            try:
                # Train model from preprocessed data
                metrics = self.train_cluster_model_from_libsvm(
                    cluster_id, cluster_file, feature_cols, total_rows
                )
                
                all_metrics.append(metrics)
                log_memory_usage(f"After training cluster {cluster_id}")
                
            except Exception as e:
                print(f"❌ Error training cluster {cluster_id}: {e}")
                continue
        
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
            
            print(f"\n✅ Cluster training completed successfully!")
            print(f"📁 Results saved to: {self.results_dir}")
            return metrics_df
        else:
            print(f"\n❌ No models were trained successfully")
            return None

def main():
    """Main function for cluster model training"""
    parser = argparse.ArgumentParser(description="Cluster Model Training from Preprocessed Data")
    parser.add_argument('--results-dir', default='./results/cluster_models',
                        help="Directory containing preprocessed data and to save results")
    
    args = parser.parse_args()
    
    print("🚀 SAPFLUXNET Cluster Model Training")
    print("=" * 40)
    print(f"Started at: {datetime.now()}")
    print(f"\n💡 Note: Data should be preprocessed first using: python preprocess_cluster_data.py")
    
    try:
        # Initialize trainer
        trainer = ClusterModelTrainer(results_dir=args.results_dir)
        
        # Train all available clusters
        metrics_df = trainer.train_all_cluster_models()
        
        if metrics_df is not None:
            print(f"\n🎉 Training completed successfully!")
            print(f"📁 Results saved to: {trainer.results_dir}")
            print(f"💡 Next step: Run spatial validation using: python cluster_spatial_validation.py")
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