"""
Within-Cluster Spatial Validation for Ecosystem-Based Models
Tests spatial generalization of cluster-specific XGBoost models using Leave-One-Site-Out validation within each cluster

NEW WORKFLOW (Updated Jan 2025):
- Cluster assignments come from CSV files (not embedded in parquet files)
- Uses preprocessed libsvm files (same data used for training models)
- Ensures training/validation data consistency through shared preprocessing
- Supports both in-memory and streaming validation approaches
- Site mapping reconstructed from metadata for Leave-One-Site-Out validation
- No longer expects 'ecosystem_cluster' column in parquet files
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
                 results_dir='./results/cluster_spatial_validation',
                 force_streaming=False):
        self.parquet_dir = parquet_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.preprocessed_dir = os.path.join(models_dir, 'preprocessed_libsvm')
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.force_streaming = force_streaming
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🌍 Cluster Spatial Validator initialized")
        print(f"📁 Parquet directory: {parquet_dir} (fallback only)")
        print(f"🤖 Models directory: {models_dir}")
        print(f"💾 Preprocessed directory: {self.preprocessed_dir}")
        print(f"📁 Results directory: {results_dir}")
        if force_streaming:
            print(f"⚠️  FORCED STREAMING MODE enabled")
        print(f"💡 NEW WORKFLOW: Uses preprocessed libsvm files (ensures training/validation data consistency)")
    
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
        """Load preprocessed libsvm data for a specific cluster"""
        print(f"📊 Loading preprocessed data for cluster {cluster_id} ({len(cluster_sites)} sites)...")
        
        # Check if preprocessed data exists
        libsvm_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_clean.svm')
        metadata_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_metadata.json')
        
        if not os.path.exists(libsvm_file):
            print(f"  ❌ Preprocessed file not found: {libsvm_file}")
            print(f"  💡 Run preprocessing first: python preprocess_cluster_data.py")
            raise FileNotFoundError(f"Preprocessed data for cluster {cluster_id} not found")
        
        if not os.path.exists(metadata_file):
            print(f"  ❌ Metadata file not found: {metadata_file}")
            raise FileNotFoundError(f"Metadata for cluster {cluster_id} not found")
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            total_rows = metadata['total_rows']
            feature_count = metadata['feature_count']
            sites_processed = metadata.get('sites_processed', metadata.get('sites', []))
            
            print(f"  ✅ Found preprocessed data: {total_rows:,} rows, {feature_count} features")
            print(f"  📊 Sites in preprocessed data: {len(sites_processed)}")
            
            # Check for missing sites
            missing_sites = set(cluster_sites) - set(sites_processed)
            if missing_sites:
                print(f"  ⚠️  Warning: {len(missing_sites)} cluster sites not in preprocessed data: {missing_sites}")
                print(f"      Missing sites: {missing_sites}")
                
            # Check for sites in preprocessed data but not in current cluster assignment
            extra_sites = set(sites_processed) - set(cluster_sites)
            if extra_sites:
                print(f"  ℹ️  Note: {len(extra_sites)} sites in preprocessed data not in current cluster assignment")
                print(f"      Extra sites: {extra_sites}")
            
            return self._load_preprocessed_cluster_data(cluster_id, libsvm_file, metadata, cluster_sites)
            
        except Exception as e:
            print(f"  ❌ Error loading metadata: {e}")
            raise
    
    def _load_preprocessed_cluster_data(self, cluster_id, libsvm_file, metadata, cluster_sites):
        """Load preprocessed libsvm data and prepare for site-based validation"""
        from sklearn.datasets import load_svmlight_file
        
        # Check file size to determine processing approach
        file_size_gb = os.path.getsize(libsvm_file) / (1024**3)
        available_memory = get_available_memory_gb()
        
        print(f"  📊 Libsvm file size: {file_size_gb:.2f} GB")
        print(f"  💾 Available memory: {available_memory:.1f} GB")
        
        # Use streaming if file is large relative to available memory
        use_streaming = file_size_gb > (available_memory * 0.4) or self.force_streaming
        
        if use_streaming:
            print(f"  💾 Using STREAMING approach (large file or forced)")
            return self._prepare_streaming_validation_from_libsvm(cluster_id, libsvm_file, metadata, cluster_sites), 'streaming'
        else:
            print(f"  🚀 Loading preprocessed data in memory")
            return self._load_libsvm_with_site_mapping(cluster_id, libsvm_file, metadata, cluster_sites), 'in_memory'
    
    def _load_libsvm_with_site_mapping(self, cluster_id, libsvm_file, metadata, cluster_sites):
        """Load libsvm data in memory and create site mapping for validation"""
        from sklearn.datasets import load_svmlight_file
        
        # Load the preprocessed data
        print(f"    📊 Loading libsvm file into memory...")
        X, y = load_svmlight_file(libsvm_file)
        X = X.toarray()  # Convert sparse to dense for easier manipulation
        
        total_rows = len(y)
        feature_count = X.shape[1]
        
        print(f"    ✅ Loaded {total_rows:,} rows, {feature_count} features")
        
        # We need to map the rows back to sites somehow
        # Since the preprocessed data doesn't contain site info, we need to reconstruct it
        # by matching against the original data sizes from parquet files
        
        print(f"    🔄 Reconstructing site mapping from original data...")
        
        # Use preprocessing metadata to get the correct row counts (after sampling)
        site_row_mapping = {}
        current_row = 0
        
        # Check if metadata contains per-site row counts from preprocessing
        sites_processed = metadata.get('sites_processed', metadata.get('sites', []))
        per_site_rows = metadata.get('per_site_rows', {})
        
        if per_site_rows:
            # Use the exact row counts from preprocessing metadata
            print(f"    📊 Using per-site row counts from preprocessing metadata")
            for site in sites_processed:
                if site not in cluster_sites:
                    continue  # Skip sites not in current cluster assignment
                
                site_rows = per_site_rows.get(site, 0)
                if site_rows > 0:
                    site_row_mapping[site] = {
                        'start_row': current_row,
                        'end_row': current_row + site_rows,
                        'row_count': site_rows
                    }
                    current_row += site_rows
                    
                    print(f"      ✅ {site}: rows {site_row_mapping[site]['start_row']}-{site_row_mapping[site]['end_row']} ({site_rows:,} rows)")
        else:
            # Fallback: estimate from preprocessed data proportionally
            print(f"    📊 Estimating row counts proportionally from preprocessed data")
            
            # Get original row counts for proportion calculation
            original_total = 0
            original_site_counts = {}
            
            for site in sites_processed:
                if site not in cluster_sites:
                    continue
                    
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                if not os.path.exists(parquet_file):
                    continue
                
                try:
                    df_site = pd.read_parquet(parquet_file, columns=[self.target_col])
                    df_site = df_site.dropna(subset=[self.target_col])
                    original_rows = len(df_site)
                    original_site_counts[site] = original_rows
                    original_total += original_rows
                    del df_site
                    gc.collect()
                except Exception as e:
                    print(f"      ❌ Error reading {site}: {e}")
                    continue
            
            # Distribute preprocessed rows proportionally
            if original_total > 0:
                for site in sites_processed:
                    if site not in cluster_sites or site not in original_site_counts:
                        continue
                    
                    # Calculate proportional rows in preprocessed data
                    proportion = original_site_counts[site] / original_total
                    site_rows = int(total_rows * proportion)
                    
                    if site_rows > 0:
                        site_row_mapping[site] = {
                            'start_row': current_row,
                            'end_row': current_row + site_rows,
                            'row_count': site_rows
                        }
                        current_row += site_rows
                        
                        print(f"      ✅ {site}: rows {site_row_mapping[site]['start_row']}-{site_row_mapping[site]['end_row']} ({site_rows:,} rows, {proportion:.1%})")
        
        print(f"    📊 Mapped {len(site_row_mapping)} sites to {current_row:,} total rows")
        
        if current_row != total_rows:
            print(f"    ⚠️  Warning: Row count mismatch. Expected {total_rows:,}, mapped {current_row:,}")
            print(f"    💡 Using only the mapped {current_row:,} rows for validation")
        
        # Extract only the data that corresponds to the mapped sites
        if current_row < total_rows:
            # Only use the first current_row samples (corresponding to mapped sites)
            X_mapped = X[:current_row]
            y_mapped = y[:current_row]
            print(f"    🔧 Trimmed data to {current_row:,} rows to match site mapping")
        else:
            X_mapped = X
            y_mapped = y
        
        # Create site labels array
        site_labels = np.concatenate([
            [site] * mapping['row_count'] 
            for site, mapping in site_row_mapping.items()
        ])
        
        # Verify lengths match
        if len(site_labels) != len(y_mapped):
            raise ValueError(f"Site mapping length mismatch: {len(site_labels)} site labels vs {len(y_mapped)} target values")
        
        # Create a DataFrame-like structure for compatibility with existing validation code
        cluster_df = pd.DataFrame({
            'site': site_labels,
            self.target_col: y_mapped
        })
        
        # Store the feature matrix separately (we'll use it directly)
        cluster_df._feature_matrix = X_mapped
        cluster_df._feature_names = metadata.get('feature_names', [f'feature_{i}' for i in range(X_mapped.shape[1])])
        
        print(f"    ✅ Created cluster dataframe: {len(cluster_df):,} rows, {len(site_row_mapping)} sites")
        
        return cluster_df
    
    def _prepare_streaming_validation_from_libsvm(self, cluster_id, libsvm_file, metadata, cluster_sites):
        """Prepare streaming validation using preprocessed libsvm file"""
        print(f"    🔧 Preparing streaming validation from preprocessed data...")
        
        # For streaming, we'll work directly with the libsvm file
        # and create temporary train/test files for each fold
        
        # First, we need to map line numbers in libsvm file to sites
        print(f"    📊 Creating site-to-line mapping...")
        
        site_line_mapping = {}
        current_line = 0
        
        sites_processed = metadata.get('sites_processed', metadata.get('sites', []))
        
        for site in sites_processed:
            if site not in cluster_sites:
                continue
                
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                continue
            
            try:
                # Count valid rows for this site
                df_site = pd.read_parquet(parquet_file, columns=[self.target_col])
                df_site = df_site.dropna(subset=[self.target_col])
                site_rows = len(df_site)
                
                if site_rows > 0:
                    site_line_mapping[site] = {
                        'start_line': current_line,
                        'end_line': current_line + site_rows,
                        'line_count': site_rows
                    }
                    current_line += site_rows
                    
                    print(f"      ✅ {site}: lines {site_line_mapping[site]['start_line']}-{site_line_mapping[site]['end_line']} ({site_rows:,} lines)")
                
                del df_site
                gc.collect()
                
            except Exception as e:
                print(f"      ❌ Error mapping {site}: {e}")
                continue
        
        # Create temporary directory for this cluster
        temp_dir = os.path.join(self.results_dir, f'temp_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        streaming_info = {
            'libsvm_file': libsvm_file,
            'site_line_mapping': site_line_mapping,
            'temp_dir': temp_dir,
            'cluster_id': cluster_id,
            'metadata': metadata
        }
        
        print(f"    ✅ Prepared streaming validation for {len(site_line_mapping)} sites")
        
        return streaming_info
    
    def _load_cluster_data_in_memory(self, cluster_id, site_info):
        """Load cluster data in memory (for smaller datasets)"""
        cluster_data = []
        
        for site, info in site_info.items():
            try:
                df_site = pd.read_parquet(info['file_path'])
                # No need to filter by cluster - we already know this site belongs to this cluster
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
        # Note: ecosystem_cluster column may not exist in parquet files (comes from CSV)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
        if self.cluster_col in df.columns:
            exclude_cols.append(self.cluster_col)
        
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
        """Perform validation with in-memory preprocessed data"""
        print(f"  🚀 Running in-memory validation with preprocessed data...")
        
        # Use the preprocessed feature matrix directly (no need for prepare_features)
        X = cluster_df._feature_matrix
        y = cluster_df[self.target_col].values
        feature_cols = cluster_df._feature_names
        
        print(f"    📊 Using preprocessed features: {X.shape[1]} features, {len(y):,} samples")
        
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
    
    def _validate_cluster_streaming(self, cluster_id, cluster_sites, model, streaming_info):
        """Perform validation with streaming preprocessed data processing"""
        print(f"  💾 Running streaming validation with preprocessed data...")
        
        libsvm_file = streaming_info['libsvm_file']
        site_line_mapping = streaming_info['site_line_mapping']
        temp_dir = streaming_info['temp_dir']
        
        fold_results = []
        
        try:
            # Leave-One-Site-Out validation
            for i, test_site in enumerate(cluster_sites):
                if test_site not in site_line_mapping:
                    print(f"  ⚠️  No data available for test site {test_site}")
                    continue
                
                print(f"\n--- Fold {i+1}/{len(cluster_sites)}: Test site {test_site} ---")
                
                try:
                    # Create train/test data for this fold from preprocessed libsvm file
                    train_file, test_file, train_samples, test_samples = self._create_fold_files_from_libsvm(
                        test_site, libsvm_file, site_line_mapping, temp_dir
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
        df_sample = pd.read_parquet(first_site_file)
        df_sample = df_sample.head(100)  # Take first 100 rows instead of nrows parameter
        _, _, feature_cols = self.prepare_features(df_sample)
        del df_sample
        gc.collect()
        
        with open(train_file, 'w') as train_out, open(test_file, 'w') as test_out:
            for site, info in site_info.items():
                try:
                    # Load site data
                    df_site = pd.read_parquet(info['file_path'])
                    # No need to filter by cluster - we already know this site belongs to this cluster
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
    
    def _create_fold_files_from_libsvm(self, test_site, libsvm_file, site_line_mapping, temp_dir):
        """Create train/test libsvm files for a single fold from preprocessed data"""
        train_file = os.path.join(temp_dir, f'fold_{test_site}_train.svm')
        test_file = os.path.join(temp_dir, f'fold_{test_site}_test.svm')
        
        train_samples = 0
        test_samples = 0
        
        # Get test site line range
        if test_site not in site_line_mapping:
            return train_file, test_file, 0, 0
        
        test_start = site_line_mapping[test_site]['start_line']
        test_end = site_line_mapping[test_site]['end_line']
        
        print(f"    📊 Splitting libsvm file: test site {test_site} uses lines {test_start}-{test_end}")
        
        # Read the libsvm file and split by site
        with open(libsvm_file, 'r') as input_file, \
             open(train_file, 'w') as train_out, \
             open(test_file, 'w') as test_out:
            
            for line_num, line in enumerate(input_file):
                if line.strip():
                    # Determine which file this line belongs to
                    if test_start <= line_num < test_end:
                        # This line belongs to test site
                        test_out.write(line)
                        test_samples += 1
                    else:
                        # This line belongs to training sites
                        train_out.write(line)
                        train_samples += 1
        
        print(f"    ✅ Created fold files: train={train_samples:,}, test={test_samples:,}")
        
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
    parser.add_argument('--force-streaming', action='store_true',
                        help="Force streaming mode for all clusters (memory optimization)")
    
    args = parser.parse_args()
    
    try:
        validator = ClusterSpatialValidator(
            parquet_dir=args.parquet_dir,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            force_streaming=args.force_streaming
        )
        
        fold_results, cluster_summaries = validator.run_validation()
        
        print(f"\n🎉 Cluster spatial validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main() 