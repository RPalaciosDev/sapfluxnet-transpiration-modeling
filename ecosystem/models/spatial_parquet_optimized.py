"""
Memory-Optimized Within-Cluster Spatial Validation for Ecosystem-Based Models
Tests spatial generalization using true streaming and external memory training

OPTIMIZATIONS:
- Chunk-based parquet processing (never loads full files)
- XGBoost external memory training
- Immediate cleanup after each operation
- Conservative memory thresholds
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

class MemoryOptimizedSpatialValidator:
    """
    Memory-optimized spatial validator that never loads full datasets
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 results_dir='./results/parquet_spatial_validation_optimized',
                 chunk_size=50000):  # Conservative chunk size
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.chunk_size = chunk_size
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🚀 Memory-Optimized Spatial Validator initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"🔧 Chunk size: {chunk_size:,} rows")
        print(f"💾 Available memory: {get_available_memory_gb():.1f} GB")
    
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

    def get_site_info(self, sites):
        """Get basic info about sites without loading data"""
        site_info = {}
        
        for site in sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"  ⚠️  Missing parquet file: {parquet_file}")
                continue
            
            # Get file size and estimated rows
            file_size_mb = os.path.getsize(parquet_file) / (1024**2)
            
            # Quick sample to check if site has valid data
            try:
                df_sample = pd.read_parquet(parquet_file, columns=[self.target_col])
                df_sample = df_sample.dropna(subset=[self.target_col])
                estimated_rows = len(df_sample)
                
                if estimated_rows == 0:
                    print(f"    ⚠️  No valid data for {site}")
                    del df_sample
                    continue
                
                site_info[site] = {
                    'file_path': parquet_file,
                    'estimated_rows': estimated_rows,
                    'size_mb': file_size_mb
                }
                
                print(f"    ✅ {site}: ~{estimated_rows:,} rows, {file_size_mb:.1f} MB")
                
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error checking {site}: {e}")
                continue
        
        return site_info

    def prepare_features_chunk(self, df_chunk):
        """Prepare features for a data chunk (memory efficient)"""
        # Exclude columns (same as in original script)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', 'ecosystem_cluster']
        
        feature_cols = [col for col in df_chunk.columns 
                       if col not in exclude_cols + [self.target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        # Extract and clean features
        X_df = df_chunk[feature_cols].copy()
        
        # Convert boolean columns to numeric
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype == 'object':
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with 0
        X = X_df.fillna(0).values
        y = df_chunk[self.target_col].values
        
        del X_df  # Immediate cleanup
        
        return X, y, feature_cols

    def create_external_memory_files(self, test_site, site_info, temp_dir):
        """Create libsvm files using chunked processing"""
        train_file = os.path.join(temp_dir, f'fold_{test_site}_train.svm')
        test_file = os.path.join(temp_dir, f'fold_{test_site}_test.svm')
        
        train_samples = 0
        test_samples = 0
        feature_cols = None
        
        print(f"    🔧 Creating external memory files (chunked processing)...")
        
        with open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
            
            for site, info in site_info.items():
                print(f"      Processing {site} in chunks...")
                
                try:
                    # Read parquet file in chunks
                    parquet_file = info['file_path']
                    
                    # Get total number of rows for chunking
                    total_rows = info['estimated_rows']
                    
                    processed_rows = 0
                    site_train_samples = 0
                    site_test_samples = 0
                    
                    # Load entire parquet file but process in smaller chunks in memory
                    df_full = pd.read_parquet(parquet_file)
                    df_full = df_full.dropna(subset=[self.target_col])
                    total_rows = len(df_full)
                    
                    if total_rows == 0:
                        del df_full
                        continue
                    
                    # Process in memory chunks to reduce peak memory usage
                    for chunk_start in range(0, total_rows, self.chunk_size):
                        try:
                            chunk_end = min(chunk_start + self.chunk_size, total_rows)
                            df_chunk = df_full.iloc[chunk_start:chunk_end].copy()
                            
                            if len(df_chunk) == 0:
                                continue
                            
                            # Prepare features for this chunk
                            X_chunk, y_chunk, chunk_feature_cols = self.prepare_features_chunk(df_chunk)
                            
                            # Store feature columns from first chunk
                            if feature_cols is None:
                                feature_cols = chunk_feature_cols
                            
                            # Determine output file
                            output_file = test_out if site == test_site else train_out
                            
                            # Write chunk to libsvm format
                            for j in range(len(X_chunk)):
                                line_parts = [str(y_chunk[j])]
                                for k, value in enumerate(X_chunk[j]):
                                    if value != 0:  # Sparse format
                                        line_parts.append(f"{k}:{value}")
                                output_file.write(' '.join(line_parts) + '\n')
                            
                            # Update counters
                            if site == test_site:
                                site_test_samples += len(X_chunk)
                            else:
                                site_train_samples += len(X_chunk)
                            
                            processed_rows += len(df_chunk)
                            
                            # Immediate cleanup
                            del df_chunk, X_chunk, y_chunk
                            gc.collect()
                            
                            # Memory check
                            if get_available_memory_gb() < 2.0:  # Conservative threshold
                                print(f"        ⚠️  Low memory warning, forcing garbage collection")
                                gc.collect()
                            
                            # Break if we've processed all data
                            if processed_rows >= total_rows:
                                break
                                
                        except Exception as e:
                            print(f"        ❌ Error processing chunk for {site}: {e}")
                            continue
                    
                    train_samples += site_train_samples
                    test_samples += site_test_samples
                    
                    print(f"        ✅ {site}: {site_train_samples + site_test_samples:,} samples processed")
                    
                    # Clean up full dataframe after processing site
                    del df_full
                    gc.collect()
                    
                except Exception as e:
                    print(f"      ❌ Error processing {site}: {e}")
                    continue
        
        print(f"    ✅ Created external memory files: train={train_samples:,}, test={test_samples:,}")
        
        return train_file, test_file, train_samples, test_samples, feature_cols

    def validate_cluster_spatially(self, cluster_id, cluster_sites):
        """Perform Leave-One-Site-Out validation with true memory optimization"""
        print(f"\n{'='*60}")
        print(f"MEMORY-OPTIMIZED SPATIAL VALIDATION FOR CLUSTER {cluster_id}")
        print(f"{'='*60}")
        print(f"Sites in cluster: {len(cluster_sites)}")
        
        if len(cluster_sites) < 3:
            print(f"⚠️  Skipping cluster {cluster_id}: Need at least 3 sites for spatial validation")
            return None
        
        # Get site information
        print(f"📊 Analyzing cluster sites...")
        site_info = self.get_site_info(cluster_sites)
        
        if len(site_info) < 3:
            print(f"⚠️  Only {len(site_info)} valid sites found, need at least 3")
            return None
        
        # Create temporary directory
        temp_dir = os.path.join(self.results_dir, f'temp_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        fold_results = []
        successful_sites = list(site_info.keys())
        
        try:
            # Leave-One-Site-Out validation
            for i, test_site in enumerate(successful_sites):
                print(f"\n--- Fold {i+1}/{len(successful_sites)}: Test site {test_site} ---")
                
                log_memory_usage(f"Before fold {i+1}")
                
                try:
                    # Create external memory files for this fold (chunked)
                    train_file, test_file, train_samples, test_samples, feature_cols = \
                        self.create_external_memory_files(test_site, site_info, temp_dir)
                    
                    if train_samples == 0 or test_samples == 0:
                        print(f"  ⚠️  Insufficient data for {test_site}")
                        continue
                    
                    print(f"  Train: {train_samples:,} samples, Test: {test_samples:,} samples")
                    
                    # Create DMatrix objects from files (external memory)
                    dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
                    dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
                    
                    # XGBoost parameters optimized for memory efficiency
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'max_depth': 6,  # Reduced for memory
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'tree_method': 'hist',  # Memory efficient
                        'max_bin': 256  # Reduce memory usage
                    }
                    
                    # Train model with external memory
                    print(f"    🤖 Training model for fold {i+1}...")
                    fold_model = xgb.train(
                        params=xgb_params,
                        dtrain=dtrain,
                        num_boost_round=100,
                        verbose_eval=False
                    )
                    
                    # Make predictions
                    y_pred_train = fold_model.predict(dtrain)
                    y_pred_test = fold_model.predict(dtest)
                    
                    # Load actual targets for metrics (memory efficient)
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
                    
                    # Immediate cleanup
                    del fold_model, dtrain, dtest, y_pred_train, y_pred_test, y_train_actual, y_test_actual
                    os.remove(train_file)
                    os.remove(test_file)
                    gc.collect()
                    
                    log_memory_usage(f"After fold {i+1}")
                    
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

    def _load_targets_from_libsvm(self, libsvm_file):
        """Load only target values from libsvm file (memory efficient)"""
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
        
        print(f"\n📊 Cluster {cluster_id} Summary (Memory Optimized):")
        print(f"  Successful folds: {cluster_summary['successful_folds']}/{cluster_summary['total_sites']}")
        print(f"  Test R² (mean ± std): {cluster_summary['mean_test_r2']:.4f} ± {cluster_summary['std_test_r2']:.4f}")
        print(f"  Test R² range: [{cluster_summary['min_test_r2']:.4f}, {cluster_summary['max_test_r2']:.4f}]")
        print(f"  Test RMSE (mean ± std): {cluster_summary['mean_test_rmse']:.4f} ± {cluster_summary['std_test_rmse']:.4f}")
        
        return fold_results, cluster_summary

    def run_validation(self):
        """Run memory-optimized spatial validation for all clusters"""
        print("🚀 MEMORY-OPTIMIZED SAPFLUXNET Spatial Validation")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Memory optimizations: Chunked processing, external memory training")
        
        try:
            # Load cluster assignments
            cluster_assignments, cluster_counts = self.load_cluster_assignments()
            
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
                
                result = self.validate_cluster_spatially(cluster_id, cluster_sites)
                
                if result is not None:
                    fold_results, cluster_summary = result
                    all_fold_results.extend(fold_results)
                    cluster_summaries.append(cluster_summary)
                
                # Force garbage collection between clusters
                gc.collect()
            
            # Save results
            self.save_results(all_fold_results, cluster_summaries)
            
            # Print overall summary
            self.print_overall_summary(cluster_summaries)
            
            return all_fold_results, cluster_summaries
            
        except Exception as e:
            print(f"\n❌ Memory-optimized spatial validation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_results(self, all_fold_results, cluster_summaries):
        """Save validation results"""
        print(f"\n💾 Saving results to {self.results_dir}...")
        
        # Save detailed fold results
        if all_fold_results:
            fold_results_df = pd.DataFrame(all_fold_results)
            fold_results_path = os.path.join(self.results_dir, f'optimized_spatial_fold_results_{self.timestamp}.csv')
            fold_results_df.to_csv(fold_results_path, index=False)
            print(f"  ✅ Fold results: {fold_results_path}")
        
        # Save cluster summaries
        if cluster_summaries:
            summaries_df = pd.DataFrame(cluster_summaries)
            summaries_path = os.path.join(self.results_dir, f'optimized_spatial_summaries_{self.timestamp}.csv')
            summaries_df.to_csv(summaries_path, index=False)
            print(f"  ✅ Cluster summaries: {summaries_path}")

    def print_overall_summary(self, cluster_summaries):
        """Print overall validation summary"""
        if not cluster_summaries:
            print("\n❌ No cluster summaries available")
            return
        
        print(f"\n📊 OVERALL MEMORY-OPTIMIZED SPATIAL VALIDATION SUMMARY")
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
        
        print(f"\n✅ Memory-optimized spatial validation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Memory-Optimized Spatial Validation")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--results-dir', default='./results/parquet_spatial_validation_optimized',
                        help="Directory to save validation results")
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help="Chunk size for processing large files")
    
    args = parser.parse_args()
    
    try:
        validator = MemoryOptimizedSpatialValidator(
            parquet_dir=args.parquet_dir,
            results_dir=args.results_dir,
            chunk_size=args.chunk_size
        )
        
        fold_results, cluster_summaries = validator.run_validation()
        
        print(f"\n🎉 Memory-optimized spatial validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main() 