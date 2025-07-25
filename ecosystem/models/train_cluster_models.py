"""
Memory-Optimized Cluster-Specific XGBoost Training for SAPFLUXNET Data
Trains separate XGBoost models for each ecosystem cluster with external memory support
Two-stage approach: 1) Preprocess to libsvm once, 2) Train models from preprocessed data
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
import argparse

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
    Two-stage approach: preprocessing + training
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', results_dir='./results/cluster_models'):
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.preprocessed_dir = os.path.join(results_dir, 'preprocessed_libsvm')
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Adaptive chunk size based on available memory
        available_memory = get_available_memory_gb()
        if available_memory > 20:  # High memory system (like yours with 27GB)
            self.chunk_size = 200000  # 200K rows per chunk
            print(f"🚀 HIGH-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        elif available_memory > 10:
            self.chunk_size = 100000  # 100K rows per chunk
            print(f"⚡ MEDIUM-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        else:
            self.chunk_size = 50000   # 50K rows per chunk
            print(f"💾 LOW-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        print(f"🚀 Memory-Optimized Cluster Trainer initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"📁 Preprocessed directory: {self.preprocessed_dir}")
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
    
    def load_preprocessed_files(self):
        """Load information about existing preprocessed libsvm files"""
        print("\n🔍 Loading preprocessed libsvm files...")
        
        if not os.path.exists(self.preprocessed_dir):
            print(f"  ❌ Preprocessed directory not found: {self.preprocessed_dir}")
            print(f"  💡 Run preprocessing first: python preprocess_cluster_data.py")
            return {}
        
        preprocessed_files = {}
        
        # Find all libsvm files
        libsvm_files = glob.glob(os.path.join(self.preprocessed_dir, 'cluster_*_clean.svm'))
        
        if not libsvm_files:
            print(f"  ❌ No preprocessed libsvm files found in {self.preprocessed_dir}")
            print(f"  💡 Run preprocessing first: python preprocess_cluster_data.py")
            return {}
        
        for libsvm_file in sorted(libsvm_files):
            # Extract cluster ID from filename
            filename = os.path.basename(libsvm_file)
            cluster_id = int(filename.replace('cluster_', '').replace('_clean.svm', ''))
            
            # Find corresponding metadata file
            metadata_file = libsvm_file.replace('_clean.svm', '_metadata.json')
            
            if not os.path.exists(metadata_file):
                print(f"  ⚠️  Missing metadata for cluster {cluster_id}")
                continue
            
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get file size
                file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
                
                preprocessed_files[cluster_id] = {
                    'libsvm_file': libsvm_file,
                    'metadata': metadata,
                    'size_mb': file_size_mb
                }
                
                print(f"  ✅ Cluster {cluster_id}: {metadata['total_rows']:,} rows, {file_size_mb:.1f} MB")
                print(f"     Sites: {', '.join(metadata['sites_processed'][:3])}{'...' if len(metadata['sites_processed']) > 3 else ''}")
                
            except Exception as e:
                print(f"  ❌ Error loading metadata for cluster {cluster_id}: {e}")
                continue
        
        if not preprocessed_files:
            print(f"  ❌ No valid preprocessed files found!")
            print(f"  💡 Run preprocessing first: python preprocess_cluster_data.py")
        else:
            total_size_mb = sum(info['size_mb'] for info in preprocessed_files.values())
            total_rows = sum(info['metadata']['total_rows'] for info in preprocessed_files.values())
            print(f"\n📊 Preprocessed data summary:")
            print(f"   Clusters: {len(preprocessed_files)}")
            print(f"   Total rows: {total_rows:,}")
            print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        
        return preprocessed_files
    
    def check_preprocessed_files_exist(self, cluster_info):
        """Check if preprocessed libsvm files already exist"""
        print("\n🔍 Checking for existing preprocessed files...")
        
        existing_files = {}
        missing_clusters = []
        
        for cluster_id in cluster_info.keys():
            libsvm_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_clean.svm')
            metadata_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_metadata.json')
            
            if os.path.exists(libsvm_file) and os.path.exists(metadata_file):
                # Check metadata
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
                    existing_files[cluster_id] = {
                        'libsvm_file': libsvm_file,
                        'metadata': metadata,
                        'size_mb': file_size_mb
                    }
                    print(f"  ✅ Cluster {cluster_id}: Found preprocessed file ({file_size_mb:.1f} MB, {metadata['total_rows']:,} rows)")
                except Exception as e:
                    print(f"  ⚠️  Cluster {cluster_id}: File exists but metadata corrupted: {e}")
                    missing_clusters.append(cluster_id)
            else:
                print(f"  ❌ Cluster {cluster_id}: Missing preprocessed files")
                missing_clusters.append(cluster_id)
        
        return existing_files, missing_clusters
    
    def preprocess_cluster_to_libsvm(self, cluster_id, cluster_sites):
        """Preprocess a single cluster to clean libsvm format"""
        print(f"\n🔧 Preprocessing cluster {cluster_id} to libsvm format...")
        
        # Output files
        libsvm_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_clean.svm')
        metadata_file = os.path.join(self.preprocessed_dir, f'cluster_{cluster_id}_metadata.json')
        
        all_features = []
        total_rows = 0
        
        with open(libsvm_file, 'w') as output_file:
            for site_idx, site in enumerate(cluster_sites):
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                
                if not os.path.exists(parquet_file):
                    print(f"    ⚠️  Missing: {parquet_file}")
                    continue
                
                try:
                    print(f"    🔄 Processing {site} ({site_idx+1}/{len(cluster_sites)})...")
                    
                    # Check site size first
                    df_info = pd.read_parquet(parquet_file, columns=[self.cluster_col, self.target_col])
                    df_info = df_info[df_info[self.cluster_col] == cluster_id]
                    df_info = df_info.dropna(subset=[self.target_col])
                    site_total_rows = len(df_info)
                    
                    if site_total_rows == 0:
                        print(f"      ⚠️  No valid data for {site}")
                        del df_info
                        gc.collect()
                        continue
                    
                    print(f"      📊 {site}: {site_total_rows:,} rows total")
                    del df_info
                    gc.collect()
                    
                    # Process site (chunked or in-memory based on size and available memory)
                    if site_total_rows > self.chunk_size:
                        print(f"      🔄 Using chunked processing ({self.chunk_size:,} rows per chunk)")
                        site_rows_processed = self._process_site_chunked_to_libsvm(
                            parquet_file, cluster_id, output_file, all_features
                        )
                    else:
                        print(f"      🔄 Loading site in memory")
                        site_rows_processed = self._process_site_in_memory_to_libsvm(
                            parquet_file, cluster_id, output_file, all_features
                        )
                    
                    total_rows += site_rows_processed
                    print(f"      ✅ {site}: {site_rows_processed:,} rows processed")
                    
                    # Force garbage collection after each site
                    gc.collect()
                    
                    # Log memory usage every 5 sites
                    if (site_idx + 1) % 5 == 0:
                        log_memory_usage(f"After {site_idx + 1} sites in cluster {cluster_id}")
                    
                except Exception as e:
                    print(f"    ❌ Error processing {site}: {e}")
                    continue
        
        # Save metadata (convert numpy types to native Python types for JSON serialization)
        metadata = {
            'cluster_id': int(cluster_id),  # Convert to native int
            'total_rows': int(total_rows),  # Convert to native int
            'feature_count': int(len(all_features)),  # Convert to native int
            'feature_names': [str(feature) for feature in all_features],  # Convert to native strings
            'sites': [str(site) for site in cluster_sites],  # Convert to native strings
            'created_at': datetime.now().isoformat(),
            'chunk_size_used': int(self.chunk_size)  # Convert to native int
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
        print(f"  ✅ Preprocessed cluster {cluster_id}: {total_rows:,} rows, {len(all_features)} features, {file_size_mb:.1f} MB")
        
        return libsvm_file, metadata
    
    def _process_site_chunked_to_libsvm(self, parquet_file, cluster_id, output_file, all_features):
        """Process a large site in chunks to avoid memory issues"""
        total_processed = 0
        
        # Read parquet file in chunks using pyarrow for better memory control
        import pyarrow.parquet as pq
        
        parquet_table = pq.read_table(parquet_file)
        total_rows = len(parquet_table)
        
        # Process in chunks
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            
            # Read chunk
            chunk_table = parquet_table.slice(start_idx, end_idx - start_idx)
            df_chunk = chunk_table.to_pandas()
            
            # Filter for this cluster and valid target
            df_chunk = df_chunk[df_chunk[self.cluster_col] == cluster_id]
            df_chunk = df_chunk.dropna(subset=[self.target_col])
            
            if len(df_chunk) == 0:
                del df_chunk
                continue
            
            # Prepare features
            exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
            feature_cols = [col for col in df_chunk.columns 
                           if col not in exclude_cols + [self.target_col]
                           and not col.endswith('_flags')
                           and not col.endswith('_md')]
            
            if not all_features:
                all_features.extend(feature_cols)
            
            # Extract and clean features
            X_df = df_chunk[feature_cols].copy()
            
            # Convert boolean columns to numeric (True=1, False=0)
            for col in X_df.columns:
                if X_df[col].dtype == bool:
                    X_df[col] = X_df[col].astype(int)
                elif X_df[col].dtype == 'object':
                    # Try to convert object columns to numeric, fill non-numeric with 0
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
            
            # Fill remaining NaN values with 0
            X = X_df.fillna(0).values
            y = df_chunk[self.target_col].values
            
            # Convert to libsvm format and append to file
            for i in range(len(X)):
                line_parts = [str(y[i])]
                for j, value in enumerate(X[i]):
                    if value != 0:  # Sparse format
                        line_parts.append(f"{j}:{value}")
                output_file.write(' '.join(line_parts) + '\n')
            
            chunk_processed = len(X)
            total_processed += chunk_processed
            
            # Clean up chunk
            del df_chunk, X_df, X, y
            gc.collect()
            
            if start_idx % (self.chunk_size * 3) == 0:  # Log every 3 chunks for high-memory systems
                print(f"        📊 Processed {total_processed:,}/{total_rows:,} rows...")
        
        # Clean up parquet table
        del parquet_table
        gc.collect()
        
        return total_processed
    
    def _process_site_in_memory_to_libsvm(self, parquet_file, cluster_id, output_file, all_features):
        """Process a small site in memory"""
        # Load site data
        df_site = pd.read_parquet(parquet_file)
        
        # Filter for this cluster and valid target
        df_site = df_site[df_site[self.cluster_col] == cluster_id]
        df_site = df_site.dropna(subset=[self.target_col])
        
        if len(df_site) == 0:
            del df_site
            return 0
        
        # Prepare features
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
        feature_cols = [col for col in df_site.columns 
                       if col not in exclude_cols + [self.target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        if not all_features:
            all_features.extend(feature_cols)
        
        # Extract and clean features
        X_df = df_site[feature_cols].copy()
        
        # Convert boolean columns to numeric (True=1, False=0)
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype == 'object':
                # Try to convert object columns to numeric, fill non-numeric with 0
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with 0
        X = X_df.fillna(0).values
        y = df_site[self.target_col].values
        
        # Convert to libsvm format and append to file
        for i in range(len(X)):
            line_parts = [str(y[i])]
            for j, value in enumerate(X[i]):
                if value != 0:  # Sparse format
                    line_parts.append(f"{j}:{value}")
            output_file.write(' '.join(line_parts) + '\n')
        
        rows_processed = len(X)
        
        # Clean up
        del df_site, X_df, X, y
        gc.collect()
        
        return rows_processed
    
    def preprocess_all_clusters(self, cluster_info, force_reprocess=False):
        """Preprocess all clusters to libsvm format"""
        print("\n🔧 PREPROCESSING STAGE: Converting clusters to clean libsvm format")
        print("=" * 70)
        
        # Check existing files
        existing_files, missing_clusters = self.check_preprocessed_files_exist(cluster_info)
        
        if force_reprocess:
            print("🔄 Force reprocessing enabled - will recreate all files")
            clusters_to_process = list(cluster_info.keys())
        else:
            clusters_to_process = missing_clusters
            print(f"📊 Found {len(existing_files)} existing, {len(missing_clusters)} missing preprocessed files")
        
        if not clusters_to_process:
            print("✅ All clusters already preprocessed!")
            return existing_files
        
        # Process missing clusters
        preprocessed_files = existing_files.copy()
        
        for cluster_id in clusters_to_process:
            info = cluster_info[cluster_id]
            print(f"\n{'='*60}")
            print(f"PREPROCESSING CLUSTER {cluster_id}")
            print(f"{'='*60}")
            print(f"Sites: {len(info['sites'])}, Total rows: {info['total_rows']:,}")
            
            log_memory_usage(f"Before preprocessing cluster {cluster_id}")
            
            try:
                libsvm_file, metadata = self.preprocess_cluster_to_libsvm(cluster_id, info['sites'])
                
                file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
                preprocessed_files[cluster_id] = {
                    'libsvm_file': libsvm_file,
                    'metadata': metadata,
                    'size_mb': file_size_mb
                }
                
                log_memory_usage(f"After preprocessing cluster {cluster_id}")
                
            except Exception as e:
                print(f"❌ Error preprocessing cluster {cluster_id}: {e}")
                continue
        
        print(f"\n✅ Preprocessing completed!")
        print(f"📊 Total preprocessed clusters: {len(preprocessed_files)}")
        total_size_mb = sum(info['size_mb'] for info in preprocessed_files.values())
        print(f"💾 Total preprocessed data size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        
        return preprocessed_files
    
    def train_cluster_model_from_libsvm(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train XGBoost model from preprocessed libsvm file using external memory"""
        print(f"\n🚀 Training model for cluster {cluster_id} from preprocessed data...")
        
        # Check if we should use external memory based on file size and available memory
        file_size_gb = os.path.getsize(libsvm_file) / (1024**3)
        available_memory = get_available_memory_gb()
        
        print(f"  📊 Libsvm file size: {file_size_gb:.2f} GB")
        print(f"  💾 Available memory: {available_memory:.1f} GB")
        
        # Use external memory if file is large relative to available memory
        use_external_memory = file_size_gb > (available_memory * 0.3)  # Use external if file > 30% of RAM
        
        if use_external_memory:
            print(f"  🔧 Using EXTERNAL MEMORY training (file too large for RAM)")
            return self._train_external_memory(cluster_id, libsvm_file, feature_cols, total_rows)
        else:
            print(f"  🚀 Using IN-MEMORY training (file fits comfortably in RAM)")
            return self._train_in_memory(cluster_id, libsvm_file, feature_cols, total_rows)
    
    def _train_in_memory(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train model in memory (for smaller datasets)"""
        print("  📊 Loading data into memory for train/test split...")
        X, y = load_svmlight_file(libsvm_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"  📊 Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")
        
        # XGBoost parameters
        params = self._get_xgboost_params()
        print(f"  🔧 XGBoost parameters: max_depth={params['max_depth']}, max_bin={params['max_bin']}")
        
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
        import tempfile
        
        # Create temporary directory for train/test splits
        temp_dir = os.path.join(self.results_dir, f'temp_training_cluster_{cluster_id}')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            print("  📊 Creating train/test splits with external memory...")
            
            # Create train/test split files without loading everything into memory
            train_file, test_file, train_samples, test_samples = self._create_external_train_test_split(
                libsvm_file, temp_dir, self.test_size, self.random_state
            )
            
            print(f"  📊 Train: {train_samples:,} samples, Test: {test_samples:,} samples")
            
            # XGBoost parameters
            params = self._get_xgboost_params()
            print(f"  🔧 XGBoost parameters: max_depth={params['max_depth']}, max_bin={params['max_bin']}")
            
            # Create DMatrix objects from files (external memory)
            dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
            
            # Train model
            print(f"  🏋️  Training XGBoost model with external memory...")
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
            
            # Load actual targets for metrics (only targets, not features)
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
                print(f"  🧹 Cleaning up temporary training files...")
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
        print("    🔍 Counting total samples...")
        total_lines = 0
        with open(libsvm_file, 'r') as f:
            for line in f:
                if line.strip():
                    total_lines += 1
        
        # Calculate split indices
        np.random.seed(random_state)
        test_indices = set(np.random.choice(total_lines, size=int(total_lines * test_size), replace=False))
        
        print(f"    📊 Splitting {total_lines:,} samples into train/test...")
        
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
        
        print(f"    ✅ Created train file: {train_samples:,} samples")
        print(f"    ✅ Created test file: {test_samples:,} samples")
        
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
            'best_iteration': 200  # Default since we may not have the model object
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
    
    def train_all_cluster_models(self, force_reprocess=False):
        """Train models for all clusters with two-stage approach"""
        print("🚀 Starting Memory-Optimized Cluster Model Training (Two-Stage Approach)")
        print("=" * 80)
        
        # Analyze data requirements
        estimated_size_gb, estimated_rows, total_files = self.analyze_data_requirements()
        optimal_memory = calculate_optimal_memory_usage()
        
        print(f"\n💾 Two-stage approach: Preprocess once, train multiple times")
        print(f"📊 Estimated data size: {estimated_size_gb:.1f} GB")
        print(f"🧠 Optimal memory usage: {optimal_memory:.1f} GB")
        
        # Load preprocessed files (NEW WORKFLOW: preprocessing done separately)
        preprocessed_files = self.load_preprocessed_files()
        
        if not preprocessed_files:
            raise ValueError("No preprocessed libsvm files found! Run preprocessing first:\n" +
                           "  python preprocess_cluster_data.py --cluster-csv path/to/clusters.csv")
        
        try:
            # Skip preprocessing stage - already done by preprocess_cluster_data.py
            print(f"✅ Using {len(preprocessed_files)} preprocessed cluster files")
            
            # STAGE 2: Train models from preprocessed data
            print("\n🏋️  TRAINING STAGE: Training models from preprocessed data")
            print("=" * 70)
            
            all_metrics = []
            
            for cluster_id, preprocessed_info in sorted(preprocessed_files.items()):
                print(f"\n{'='*60}")
                print(f"TRAINING CLUSTER {cluster_id} MODEL")
                print(f"{'='*60}")
                
                cluster_file = preprocessed_info['libsvm_file']
                feature_cols = preprocessed_info['metadata']['feature_names']
                total_rows = preprocessed_info['metadata']['total_rows']
                
                print(f"Preprocessed file: {os.path.basename(cluster_file)}")
                print(f"Total rows: {total_rows:,}, Features: {len(feature_cols)}")
                
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
                
                print(f"\n✅ Two-stage cluster training completed successfully!")
                print(f"📁 Results saved to: {self.results_dir}")
                print(f"💡 Preprocessed files saved for future use in: {self.preprocessed_dir}")
                return metrics_df
            else:
                print(f"\n❌ No models were trained successfully")
                return None
                
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise

def main():
    """Main function to run memory-optimized cluster training"""
    parser = argparse.ArgumentParser(description="Memory-Optimized Cluster Model Training")
    parser.add_argument('--mode', choices=['preprocess', 'train', 'both'], default='both',
                        help="Mode: 'preprocess' (DEPRECATED - use preprocess_cluster_data.py), 'train' (train from existing), 'both' (default)")
    parser.add_argument('--force-reprocess', action='store_true',
                        help="Force reprocessing of all clusters even if files exist")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--results-dir', default='./results/cluster_models',
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    print("🌍 SAPFLUXNET Memory-Optimized Cluster Model Training")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Started at: {datetime.now()}")
    print()
    print("📋 NEW WORKFLOW (Updated Jan 2025):")
    print("   1. First run: python preprocess_cluster_data.py --cluster-csv path/to/clusters.csv")
    print("   2. Then run:  python train_cluster_models.py [--mode train]")
    print("   💡 Preprocessing is now separate for better balanced sampling!")
    
    try:
        # Initialize trainer
        trainer = MemoryOptimizedClusterTrainer(
            parquet_dir=args.parquet_dir,
            results_dir=args.results_dir
        )
        
        if args.mode == 'preprocess':
            # Preprocessing is now handled by separate script
            print("\n🔧 PREPROCESSING MODE: Use separate preprocessing script")
            print("❌ Preprocessing is now handled by a separate script for better control!")
            print("\n💡 SOLUTION:")
            print("   Run: python preprocess_cluster_data.py --cluster-csv path/to/clusters.csv")
            print("   Or:  python preprocess_cluster_data.py --cluster-csv auto")
            print("\n📋 This provides:")
            print("   - Balanced sampling options")
            print("   - Better memory management") 
            print("   - Non-destructive workflow")
            return
            
        elif args.mode == 'train':
            # Only train from existing libsvm files
            print("\n🏋️  TRAINING MODE: Training from existing preprocessed files")
            preprocessed_files = trainer.load_preprocessed_files()
            
            if not preprocessed_files:
                print("❌ No preprocessed files found!")
                print("\n💡 SOLUTION:")
                print("   Run: python preprocess_cluster_data.py --cluster-csv path/to/clusters.csv")
                print("   Or:  python preprocess_cluster_data.py --cluster-csv auto")
                return
            
            # Train from existing files
            all_metrics = []
            for cluster_id, preprocessed_info in sorted(preprocessed_files.items()):
                print(f"\n{'='*60}")
                print(f"TRAINING CLUSTER {cluster_id} MODEL")
                print(f"{'='*60}")
                
                try:
                    metrics = trainer.train_cluster_model_from_libsvm(
                        cluster_id, 
                        preprocessed_info['libsvm_file'],
                        preprocessed_info['metadata']['feature_names'],
                        preprocessed_info['metadata']['total_rows']
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"❌ Error training cluster {cluster_id}: {e}")
                    continue
            
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_path = os.path.join(trainer.results_dir, f'cluster_model_metrics_{trainer.timestamp}.csv')
                metrics_df.to_csv(metrics_path, index=False)
                print(f"\n✅ Training completed! Results saved to: {metrics_path}")
            
        else:  # both
            # Full two-stage approach
            metrics_df = trainer.train_all_cluster_models(args.force_reprocess)
            
            if metrics_df is not None:
                print(f"\n🎉 Two-stage training completed successfully!")
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
