"""
Within-Cluster Spatial Validation for Ecosystem-Based Models
Tests spatial generalization of cluster-specific XGBoost models using Leave-One-Site-Out validation within each cluster
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
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

class ClusterSpatialValidator:
    """
    Validates cluster-specific models using Leave-One-Site-Out within each cluster
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 models_dir='./results/cluster_models',
                 results_dir='./results/cluster_spatial_validation'):
        self.parquet_dir = parquet_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🌍 Cluster Spatial Validator initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"🤖 Models directory: {models_dir}")
        print(f"📁 Results directory: {results_dir}")
    
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
    
    def load_cluster_data(self, cluster_id, cluster_sites):
        """Load and prepare data for a specific cluster - MEMORY OPTIMIZED"""
        print(f"📊 Loading data for cluster {cluster_id} ({len(cluster_sites)} sites)...")
        
        # First pass: analyze data size and check availability
        site_info = {}
        total_estimated_rows = 0
        
        for site in cluster_sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"  ⚠️  Missing: {parquet_file}")
                continue
            
            try:
                # Quick sample to check structure and estimate size
                df_sample = pd.read_parquet(parquet_file, columns=[self.cluster_col, self.target_col, 'site'])
                df_sample = df_sample[df_sample[self.cluster_col] == cluster_id]
                df_sample = df_sample.dropna(subset=[self.target_col])
                
                if len(df_sample) == 0:
                    print(f"  ⚠️  {site}: No valid data for cluster {cluster_id}")
                    continue
                
                site_info[site] = {
                    'file_path': parquet_file,
                    'estimated_rows': len(df_sample)
                }
                total_estimated_rows += len(df_sample)
                print(f"    ✅ {site}: ~{len(df_sample):,} rows")
                
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error analyzing {site}: {e}")
                continue
        
        if not site_info:
            raise ValueError(f"No valid data found for cluster {cluster_id}")
        
        print(f"  📊 Total estimated: {total_estimated_rows:,} rows from {len(site_info)} sites")
        
        # Check if we should use in-memory or streaming approach
        available_memory = get_available_memory_gb()
        estimated_memory_gb = total_estimated_rows * 100 * 8 / (1024**3)  # Rough estimate
        
        print(f"  💾 Estimated memory needed: {estimated_memory_gb:.1f} GB")
        print(f"  💾 Available memory: {available_memory:.1f} GB")
        
        if estimated_memory_gb < available_memory * 0.3:  # Use in-memory if < 30% of RAM
            print(f"  🚀 Using IN-MEMORY approach")
            return self._load_cluster_data_in_memory(cluster_id, site_info)
        else:
            print(f"  💾 Using STREAMING approach")
            return self._prepare_cluster_data_streaming(cluster_id, site_info)
    
    def _load_cluster_data_in_memory(self, cluster_id, site_info):
        """Load cluster data in memory (for smaller datasets)"""
        cluster_data = []
        
        for site, info in site_info.items():
            try:
                df_site = pd.read_parquet(info['file_path'])
                df_site = df_site[df_site[self.cluster_col] == cluster_id]
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) > 0:
                    cluster_data.append(df_site)
                
            except Exception as e:
                print(f"  ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            raise ValueError(f"No valid data loaded for cluster {cluster_id}")
        
        # Combine all data
        combined_df = pd.concat(cluster_data, ignore_index=True)
        print(f"  ✅ Loaded {len(combined_df):,} rows in memory")
        
        return combined_df, 'in_memory'
    
    def _prepare_cluster_data_streaming(self, cluster_id, site_info):
        """Prepare cluster data for streaming validation (for larger datasets)"""
        print(f"  🔧 Preparing streaming validation setup...")
        
        # Create temporary directory for this cluster
        temp_dir = os.path.join(self.results_dir, f'temp_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Return site info and temp directory for streaming processing
        return {
            'site_info': site_info,
            'temp_dir': temp_dir,
            'cluster_id': cluster_id
        }, 'streaming'
    
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
    
    def validate_cluster_spatially(self, cluster_id, cluster_sites, model):
        """Perform Leave-One-Site-Out validation within a cluster - MEMORY OPTIMIZED"""
        print(f"\n{'='*60}")
        print(f"SPATIAL VALIDATION FOR CLUSTER {cluster_id}")
        print(f"{'='*60}")
        print(f"Sites in cluster: {len(cluster_sites)}")
        
        if len(cluster_sites) < 3:
            print(f"⚠️  Skipping cluster {cluster_id}: Need at least 3 sites for spatial validation")
            return None
        
        # Load data for this cluster (optimized)
        cluster_data, data_mode = self.load_cluster_data(cluster_id, cluster_sites)
        
        if data_mode == 'in_memory':
            return self._validate_cluster_in_memory(cluster_id, cluster_sites, model, cluster_data)
        else:  # streaming mode
            return self._validate_cluster_streaming(cluster_id, cluster_sites, model, cluster_data)
    
    def _validate_cluster_in_memory(self, cluster_id, cluster_sites, model, cluster_df):
        """Perform validation with in-memory data"""
        print(f"  🚀 Running in-memory validation...")
        
        # Prepare features once
        X, y, feature_cols = self.prepare_features(cluster_df)
        
        fold_results = []
        
        # Leave-One-Site-Out validation
        for i, test_site in enumerate(cluster_sites):
            print(f"\n--- Fold {i+1}/{len(cluster_sites)}: Test site {test_site} ---")
            
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
                
                # Make predictions
                y_pred_train = model.predict(dtrain)
                y_pred_test = model.predict(dtest)
                
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
        
        return self._calculate_cluster_summary(cluster_id, cluster_sites, fold_results)
    
    def _validate_cluster_streaming(self, cluster_id, cluster_sites, model, cluster_info):
        """Perform validation with streaming data processing"""
        print(f"  💾 Running streaming validation...")
        
        site_info = cluster_info['site_info']
        temp_dir = cluster_info['temp_dir']
        
        fold_results = []
        
        try:
            # Leave-One-Site-Out validation
            for i, test_site in enumerate(cluster_sites):
                if test_site not in site_info:
                    print(f"  ⚠️  No data available for test site {test_site}")
                    continue
                
                print(f"\n--- Fold {i+1}/{len(cluster_sites)}: Test site {test_site} ---")
                
                try:
                    # Create train/test data for this fold
                    train_file, test_file, train_samples, test_samples = self._create_fold_files_streaming(
                        cluster_id, test_site, site_info, temp_dir
                    )
                    
                    if train_samples == 0 or test_samples == 0:
                        print(f"  ⚠️  Insufficient data for {test_site}")
                        continue
                    
                    print(f"  Train: {train_samples:,} samples, Test: {test_samples:,} samples")
                    
                    # Create DMatrix objects from files
                    dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
                    dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
                    
                    # Make predictions
                    y_pred_train = model.predict(dtrain)
                    y_pred_test = model.predict(dtest)
                    
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
            
            return self._calculate_cluster_summary(cluster_id, cluster_sites, fold_results)
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"  ⚠️  Could not clean up temp directory: {e}")
    
    def _create_fold_files_streaming(self, cluster_id, test_site, site_info, temp_dir):
        """Create train/test libsvm files for a single fold"""
        train_file = os.path.join(temp_dir, f'fold_{test_site}_train.svm')
        test_file = os.path.join(temp_dir, f'fold_{test_site}_test.svm')
        
        train_samples = 0
        test_samples = 0
        
        # Get feature columns from first site
        first_site_file = list(site_info.values())[0]['file_path']
        df_sample = pd.read_parquet(first_site_file, nrows=100)
        _, _, feature_cols = self.prepare_features(df_sample)
        del df_sample
        gc.collect()
        
        with open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
            for site, info in site_info.items():
                try:
                    # Load site data
                    df_site = pd.read_parquet(info['file_path'])
                    df_site = df_site[df_site[self.cluster_col] == cluster_id]
                    df_site = df_site.dropna(subset=[self.target_col])
                    
                    if len(df_site) == 0:
                        continue
                    
                    # Prepare features
                    X, y, _ = self.prepare_features(df_site)
                    
                    # Write to appropriate file
                    output_file = test_out if site == test_site else train_out
                    sample_count = test_samples if site == test_site else train_samples
                    
                    # Convert to libsvm format and write
                    for j in range(len(X)):
                        line_parts = [str(y[j])]
                        for k, value in enumerate(X[j]):
                            if value != 0:  # Sparse format
                                line_parts.append(f"{k}:{value}")
                        output_file.write(' '.join(line_parts) + '\n')
                    
                    if site == test_site:
                        test_samples += len(X)
                    else:
                        train_samples += len(X)
                    
                    del df_site, X, y
                    gc.collect()
                    
                except Exception as e:
                    print(f"    ❌ Error processing {site}: {e}")
                    continue
        
        return train_file, test_file, train_samples, test_samples
    
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
        
        print(f"\n📊 Cluster {cluster_id} Summary:")
        print(f"  Successful folds: {cluster_summary['successful_folds']}/{cluster_summary['total_sites']}")
        print(f"  Test R² (mean ± std): {cluster_summary['mean_test_r2']:.4f} ± {cluster_summary['std_test_r2']:.4f}")
        print(f"  Test R² range: [{cluster_summary['min_test_r2']:.4f}, {cluster_summary['max_test_r2']:.4f}]")
        print(f"  Test RMSE (mean ± std): {cluster_summary['mean_test_rmse']:.4f} ± {cluster_summary['std_test_rmse']:.4f}")
        
        return fold_results, cluster_summary
    
    def run_validation(self):
        """Run spatial validation for all clusters"""
        print("🌍 SAPFLUXNET Cluster-Based Spatial Validation")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Test spatial generalization of ecosystem-specific models")
        
        try:
            # Load cluster assignments
            cluster_assignments, cluster_counts = self.load_cluster_assignments()
            
            # Load cluster models
            models = self.load_cluster_models()
            
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
                if cluster_id not in models:
                    print(f"\n⚠️  No model found for cluster {cluster_id}, skipping...")
                    continue
                
                cluster_sites = sites_by_cluster[cluster_id]
                model = models[cluster_id]
                
                log_memory_usage(f"Before cluster {cluster_id} validation")
                
                result = self.validate_cluster_spatially(cluster_id, cluster_sites, model)
                
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
            print(f"\n❌ Cluster spatial validation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_results(self, all_fold_results, cluster_summaries):
        """Save validation results"""
        print(f"\n💾 Saving results to {self.results_dir}...")
        
        # Save detailed fold results
        if all_fold_results:
            fold_results_df = pd.DataFrame(all_fold_results)
            fold_results_path = os.path.join(self.results_dir, f'cluster_spatial_fold_results_{self.timestamp}.csv')
            fold_results_df.to_csv(fold_results_path, index=False)
            print(f"  ✅ Fold results: {fold_results_path}")
        
        # Save cluster summaries
        if cluster_summaries:
            summaries_df = pd.DataFrame(cluster_summaries)
            summaries_path = os.path.join(self.results_dir, f'cluster_spatial_summaries_{self.timestamp}.csv')
            summaries_df.to_csv(summaries_path, index=False)
            print(f"  ✅ Cluster summaries: {summaries_path}")
        
        # Save comprehensive report
        report_path = os.path.join(self.results_dir, f'cluster_spatial_report_{self.timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("SAPFLUXNET Cluster-Based Spatial Validation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Method: Leave-One-Site-Out within each ecosystem cluster\n")
            f.write(f"Purpose: Test spatial generalization of cluster-specific models\n\n")
            
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
        
        print(f"\n📊 OVERALL CLUSTER SPATIAL VALIDATION SUMMARY")
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
        
        print(f"\n✅ Cluster-based spatial validation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cluster-Based Spatial Validation")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--models-dir', default='./results/cluster_models',
                        help="Directory containing cluster models")
    parser.add_argument('--results-dir', default='./results/cluster_spatial_validation',
                        help="Directory to save validation results")
    
    args = parser.parse_args()
    
    try:
        validator = ClusterSpatialValidator(
            parquet_dir=args.parquet_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
        
        fold_results, cluster_summaries = validator.run_validation()
        
        print(f"\n🎉 Cluster spatial validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main() 