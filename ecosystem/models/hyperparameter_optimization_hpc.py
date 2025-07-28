#!/usr/bin/env python3
"""
HPC-Optimized Hyperparameter Optimization for Cluster-Specific Models
Designed for high-performance computing environments with parallel processing
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import optuna
import os
import json
import glob
from datetime import datetime
import warnings
import multiprocessing as mp
import psutil
import argparse
warnings.filterwarnings('ignore')

class HPCClusterHyperparameterOptimizer:
    """
    HPC-optimized hyperparameter optimization for each cluster
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 results_dir='./results/hyperparameter_optimization',
                 n_jobs=None):
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(results_dir, exist_ok=True)
        
        # Determine optimal number of parallel jobs
        if n_jobs is None:
            self.n_jobs = min(mp.cpu_count(), 16)  # Cap at 16 for stability
        else:
            self.n_jobs = n_jobs
            
        # Set XGBoost to use available cores efficiently
        os.environ['OMP_NUM_THREADS'] = str(max(1, self.n_jobs // 3))
        
        print(f"🔧 HPC Hyperparameter Optimizer initialized")
        print(f"💻 Available CPUs: {mp.cpu_count()}")
        print(f"🔄 Using {self.n_jobs} parallel jobs")
        print(f"🧠 Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"📁 Results will be saved to: {results_dir}")
    
    def estimate_memory_usage(self, n_samples, n_features):
        """Estimate memory usage for optimization"""
        # Data storage (float64)
        data_mb = (n_samples * n_features * 8) / (1024**2)
        # XGBoost model overhead
        model_mb = 50
        # Optuna overhead  
        optuna_mb = 20
        # Safety buffer
        buffer_mb = 100
        
        total_mb = data_mb + model_mb + optuna_mb + buffer_mb
        return total_mb
    
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
    
    def load_cluster_data_sample(self, cluster_id, cluster_sites, max_sites=5, max_samples_per_site=10000):
        """Load a representative sample of data for each cluster for optimization"""
        print(f"📊 Loading sample data for Cluster {cluster_id}...")
        
        cluster_data = []
        sites_loaded = 0
        total_samples = 0
        
        for site in cluster_sites:
            if sites_loaded >= max_sites:
                break
                
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                continue
            
            try:
                # Load and sample data
                df_site = pd.read_parquet(parquet_file)
                df_site = df_site.dropna(subset=['sap_flow'])
                
                if len(df_site) == 0:
                    continue
                
                # Sample if too large
                if len(df_site) > max_samples_per_site:
                    df_site = df_site.sample(n=max_samples_per_site, random_state=42)
                
                # Add site identifier
                df_site['site'] = site
                cluster_data.append(df_site)
                sites_loaded += 1
                total_samples += len(df_site)
                
                print(f"    ✅ {site}: {len(df_site):,} samples")
                
            except Exception as e:
                print(f"    ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            raise ValueError(f"No valid data loaded for cluster {cluster_id}")
        
        # Combine all site data
        combined_df = pd.concat(cluster_data, ignore_index=True)
        
        # Estimate memory usage
        n_features_est = len([col for col in combined_df.columns 
                             if not col.endswith('_flags') and not col.endswith('_md')]) - 5  # Approximate
        memory_mb = self.estimate_memory_usage(len(combined_df), n_features_est)
        
        print(f"  📊 Total samples for Cluster {cluster_id}: {len(combined_df):,}")
        print(f"  💾 Estimated memory usage: {memory_mb:.1f} MB")
        
        return combined_df
    
    def prepare_features(self, df):
        """Prepare features for training (same as used in spatial validation)"""
        # Exclude columns (same as in spatial_parquet.py)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', 'ecosystem_cluster']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + ['sap_flow']
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
        y = df['sap_flow'].values
        
        return X, y, feature_cols
    
    def optimize_cluster_hyperparameters(self, cluster_id, X, y, n_trials=100):
        """Use Optuna to find optimal hyperparameters for a specific cluster"""
        print(f"\n🔧 Optimizing hyperparameters for Cluster {cluster_id}...")
        print(f"   Data shape: {X.shape}")
        print(f"   Optimization trials: {n_trials}")
        print(f"   Parallel jobs: {self.n_jobs}")
        
        # Split data for optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"   Train: {X_train.shape[0]:,} samples")
        print(f"   Validation: {X_val.shape[0]:,} samples")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'n_jobs': 1,  # Each trial uses 1 core, Optuna handles parallelization
                'random_state': 42,
                'verbosity': 0
            }
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train model with early stopping
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=trial.suggest_int('n_estimators', 50, 500),
                evals=[(dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Make predictions and calculate R²
            y_pred = model.predict(dval)
            r2 = r2_score(y_val, y_pred)
            
            return r2
        
        # Run optimization with parallel processing
        study = optuna.create_study(
            direction='maximize',
            study_name=f'cluster_{cluster_id}_{self.timestamp}',
            # Use TPE sampler for better parallel performance
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize with parallel jobs
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"   ✅ Best R² for Cluster {cluster_id}: {best_score:.4f}")
        print(f"   🎯 Best parameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")
        
        return best_params, best_score, study
    
    def run_optimization_for_all_clusters(self, n_trials=100):
        """Run hyperparameter optimization for all clusters"""
        print("🚀 Starting HPC hyperparameter optimization for all clusters")
        print("=" * 60)
        
        # Load cluster assignments
        cluster_assignments, cluster_counts = self.load_cluster_assignments()
        
        # Group sites by cluster
        sites_by_cluster = {}
        for site, cluster_id in cluster_assignments.items():
            if cluster_id not in sites_by_cluster:
                sites_by_cluster[cluster_id] = []
            sites_by_cluster[cluster_id].append(site)
        
        # Optimize each cluster
        optimization_results = {}
        
        for cluster_id in sorted(sites_by_cluster.keys()):
            cluster_sites = sites_by_cluster[cluster_id]
            
            print(f"\n{'='*40}")
            print(f"OPTIMIZING CLUSTER {cluster_id}")
            print(f"{'='*40}")
            
            try:
                # Load sample data for this cluster
                cluster_df = self.load_cluster_data_sample(cluster_id, cluster_sites)
                
                # Prepare features
                X, y, feature_cols = self.prepare_features(cluster_df)
                
                # Optimize hyperparameters
                start_time = datetime.now()
                best_params, best_score, study = self.optimize_cluster_hyperparameters(
                    cluster_id, X, y, n_trials
                )
                end_time = datetime.now()
                
                optimization_time = (end_time - start_time).total_seconds() / 60
                
                # Store results
                optimization_results[cluster_id] = {
                    'best_params': best_params,
                    'best_r2': best_score,
                    'n_features': len(feature_cols),
                    'n_samples': len(X),
                    'n_sites_sampled': len(cluster_df['site'].unique()),
                    'optimization_time_minutes': optimization_time,
                    'n_trials': n_trials
                }
                
                print(f"   ⏱️  Optimization time: {optimization_time:.1f} minutes")
                
                # Clean up memory
                del cluster_df, X, y
                
            except Exception as e:
                print(f"❌ Failed to optimize Cluster {cluster_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results
        self.save_optimization_results(optimization_results)
        
        return optimization_results
    
    def save_optimization_results(self, results):
        """Save optimization results to files"""
        print(f"\n💾 Saving optimization results...")
        
        # Save detailed results as JSON
        results_file = os.path.join(self.results_dir, f'hpc_hyperparameter_optimization_{self.timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Detailed results: {results_file}")
        
        # Save summary as CSV
        summary_data = []
        for cluster_id, result in results.items():
            row = {
                'cluster': cluster_id,
                'best_r2': result['best_r2'],
                'n_features': result['n_features'],
                'n_samples': result['n_samples'],
                'n_sites_sampled': result['n_sites_sampled'],
                'optimization_time_minutes': result['optimization_time_minutes'],
                'n_trials': result['n_trials']
            }
            # Add hyperparameters
            row.update(result['best_params'])
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.results_dir, f'hpc_hyperparameter_summary_{self.timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"   ✅ Summary table: {summary_file}")
        
        # Save optimized parameters for spatial validation
        spatial_params_file = os.path.join(self.results_dir, f'hpc_optimized_spatial_params_{self.timestamp}.py')
        self.generate_spatial_validation_params(results, spatial_params_file)
        print(f"   ✅ Spatial validation params: {spatial_params_file}")
        
        # Save resource usage summary
        self.save_resource_summary(results)
    
    def save_resource_summary(self, results):
        """Save resource usage summary for HPC reporting"""
        total_time = sum([r['optimization_time_minutes'] for r in results.values()])
        total_trials = sum([r['n_trials'] for r in results.values()])
        
        resource_file = os.path.join(self.results_dir, f'hpc_resource_summary_{self.timestamp}.txt')
        with open(resource_file, 'w') as f:
            f.write("HPC Hyperparameter Optimization Resource Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"CPUs used: {self.n_jobs}\n")
            f.write(f"Total optimization time: {total_time:.1f} minutes\n")
            f.write(f"Total trials: {total_trials}\n")
            f.write(f"Average time per trial: {total_time/total_trials:.2f} minutes\n")
            f.write(f"Clusters optimized: {len(results)}\n\n")
            
            f.write("Per-cluster performance:\n")
            for cluster_id, result in results.items():
                f.write(f"  Cluster {cluster_id}: {result['optimization_time_minutes']:.1f} min, "
                       f"R² = {result['best_r2']:.4f}\n")
        
        print(f"   ✅ Resource summary: {resource_file}")
    
    def generate_spatial_validation_params(self, results, output_file):
        """Generate Python code with optimized parameters for spatial validation"""
        with open(output_file, 'w') as f:
            f.write("# HPC-Optimized XGBoost Parameters for Spatial Validation\n")
            f.write(f"# Generated on: {datetime.now()}\n")
            f.write(f"# Based on HPC hyperparameter optimization results\n")
            f.write(f"# CPUs used: {self.n_jobs}\n\n")
            
            f.write("OPTIMIZED_CLUSTER_PARAMS = {\n")
            
            for cluster_id, result in results.items():
                f.write(f"    {cluster_id}: {{\n")
                f.write("        'objective': 'reg:squarederror',\n")
                f.write("        'eval_metric': 'rmse',\n")
                
                params = result['best_params']
                for param, value in params.items():
                    if param == 'n_jobs':  # Skip n_jobs from optimization
                        continue
                    if isinstance(value, str):
                        f.write(f"        '{param}': '{value}',\n")
                    else:
                        f.write(f"        '{param}': {value},\n")
                
                f.write("        'random_state': 42,\n")
                f.write("        'verbosity': 0\n")
                f.write("    },\n")
            
            f.write("}\n\n")
            
            f.write("# Performance summary:\n")
            for cluster_id, result in results.items():
                f.write(f"# Cluster {cluster_id}: R² = {result['best_r2']:.4f} "
                       f"({result['n_samples']:,} samples, {result['optimization_time_minutes']:.1f} min)\n")

def main():
    """Main HPC optimization workflow"""
    parser = argparse.ArgumentParser(description="HPC Hyperparameter Optimization")
    parser.add_argument('--n-trials', type=int, default=100, 
                       help="Number of optimization trials per cluster")
    parser.add_argument('--n-jobs', type=int, default=None,
                       help="Number of parallel jobs (default: auto-detect)")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                       help="Directory containing parquet files")
    
    args = parser.parse_args()
    
    print("🔧 HPC XGBoost Hyperparameter Optimization for Ecosystem Clusters")
    print("=" * 60)
    
    optimizer = HPCClusterHyperparameterOptimizer(
        parquet_dir=args.parquet_dir,
        n_jobs=args.n_jobs
    )
    
    # Run optimization
    start_time = datetime.now()
    results = optimizer.run_optimization_for_all_clusters(n_trials=args.n_trials)
    end_time = datetime.now()
    
    total_time = (end_time - start_time).total_seconds() / 60
    
    print(f"\n🎉 HPC Optimization completed!")
    print(f"⏱️  Total time: {total_time:.1f} minutes")
    print(f"📊 Results summary:")
    
    for cluster_id, result in results.items():
        print(f"   Cluster {cluster_id}: R² = {result['best_r2']:.4f} "
              f"({result['optimization_time_minutes']:.1f} min)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Review the optimized parameters in the results directory")
    print(f"   2. Update spatial_parquet.py with the optimized parameters")
    print(f"   3. Re-run spatial validation to see improved performance")

if __name__ == "__main__":
    main() 