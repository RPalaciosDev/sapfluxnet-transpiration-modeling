"""
GPU-Optimized Cluster-Specific XGBoost Training for SAPFLUXNET Data
Trains separate XGBoost models for each ecosystem cluster with GPU acceleration
Modernized approach: Direct parquet processing with site split support (no LibSVM dependency)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import glob
import subprocess
from datetime import datetime
import warnings
import gc
import psutil
import tempfile
import shutil
import json
import pickle
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

class GPUOptimizedClusterTrainer:
    """
    GPU-optimized trainer for cluster-specific XGBoost models
    Direct parquet processing with site split support (no LibSVM dependency)
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 results_dir='./results/cluster_models',
                 cluster_csv=None, 
                 site_split_file=None,
                 force_gpu=False):
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.cluster_csv = cluster_csv
        self.site_split_file = site_split_file
        self.target_col = 'sap_flow'
        self.test_size = 0.2
        self.random_state = 42
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.force_gpu = force_gpu
        
        # Site split data
        self.train_sites = None
        self.test_sites = None
        self.cluster_assignments = {}
        
        # GPU Detection and Configuration
        self.use_gpu = False
        self.gpu_id = 0
        self._detect_gpu()
        
        # Adaptive memory settings
        self._configure_memory_settings()
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        
        # Load site split and cluster assignments
        if site_split_file:
            self._load_site_split()
        if cluster_csv:
            self._load_cluster_assignments()
        
        print(f"🚀 GPU-Optimized Cluster Trainer initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"🎯 Target column: {self.target_col}")
        print(f"🎮 GPU enabled: {self.use_gpu}")
        if self.train_sites:
            print(f"📊 Train sites: {len(self.train_sites)}")
            print(f"📊 Test sites: {len(self.test_sites) if self.test_sites else 0}")
            print(f"📊 Clusters: {len(set(self.cluster_assignments.values())) if self.cluster_assignments else 0}")
    
    def _detect_gpu(self):
        """Detect and configure GPU support"""
        print("🔍 Checking GPU and CUDA availability...")
        
        # Check if CUDA is available first
        cuda_available = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cuda_available = True
                print("  ✅ NVIDIA GPU detected via nvidia-smi")
                # Extract GPU info from nvidia-smi output
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
            import xgboost as xgb
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
                print(f"  💾 GPU will be used for XGBoost training (gpu_id={self.gpu_id})")
                
                # Clean up test objects
                del test_model, test_dmatrix, test_data, test_labels
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ GPU training test failed: {e}")
                if self.force_gpu:
                    print("  🔥 Force GPU flag enabled - will attempt GPU training anyway")
                    self.use_gpu = True
                else:
                    print("  🔄 Falling back to CPU training")
                    self.use_gpu = False
                    
        except ImportError as e:
            print(f"  ❌ XGBoost import failed: {e}")
            self.use_gpu = False
        
        if not self.use_gpu and not cuda_available:
            print("  💻 Using CPU-only training")
    
    def _configure_memory_settings(self):
        """Configure memory settings based on available system memory"""
        available_memory = get_available_memory_gb()
        
        # Adaptive chunk size and streaming thresholds
        if available_memory > 400:  # Ultra-high memory system (like yours with 545GB!)
            self.chunk_size = 500000  # 500K rows per chunk
            self.streaming_threshold_mb = 50000  # 50GB
            self.streaming_threshold_rows = 20000000  # 20M rows
            print(f"🚀 ULTRA-HIGH-MEMORY mode: {self.chunk_size:,} rows per chunk")
        elif available_memory > 100:  # High memory system 
            self.chunk_size = 300000  # 300K rows per chunk
            self.streaming_threshold_mb = 20000  # 20GB
            self.streaming_threshold_rows = 10000000  # 10M rows
            print(f"🚀 HIGH-MEMORY mode: {self.chunk_size:,} rows per chunk")
        elif available_memory > 50:  # Medium-high memory system  
            self.chunk_size = 200000  # 200K rows per chunk
            self.streaming_threshold_mb = 5000  # 5GB
            self.streaming_threshold_rows = 2000000  # 2M rows
            print(f"⚡ MEDIUM-HIGH-MEMORY mode: {self.chunk_size:,} rows per chunk")
        elif available_memory > 20:  # Medium memory system
            self.chunk_size = 100000  # 100K rows per chunk
            self.streaming_threshold_mb = 2000  # 2GB
            self.streaming_threshold_rows = 1000000  # 1M rows
            print(f"⚡ MEDIUM-MEMORY mode: {self.chunk_size:,} rows per chunk")
        else:  # Low memory system
            self.chunk_size = 50000   # 50K rows per chunk
            self.streaming_threshold_mb = 1000  # 1GB
            self.streaming_threshold_rows = 500000  # 500K rows
            print(f"💾 LOW-MEMORY mode: {self.chunk_size:,} rows per chunk")
    
    def _load_site_split(self):
        """Load site split from JSON file"""
        if not os.path.exists(self.site_split_file):
            raise FileNotFoundError(f"Site split file not found: {self.site_split_file}")
        
        print(f"📂 Loading site split from: {self.site_split_file}")
        with open(self.site_split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_sites = set(split_data['train_sites'])
        self.test_sites = set(split_data['test_sites'])
        
        print(f"  ✅ Loaded site split:")
        print(f"    📊 Train sites: {len(self.train_sites)}")
        print(f"    📊 Test sites: {len(self.test_sites)}")
        print(f"    📅 Split created: {split_data['metadata']['timestamp']}")
        print(f"    🎲 Random seed: {split_data['metadata']['random_seed']}")
    
    def _load_cluster_assignments(self):
        """Load cluster assignments from CSV file"""
        if not os.path.exists(self.cluster_csv):
            raise FileNotFoundError(f"Cluster CSV file not found: {self.cluster_csv}")
        
        print(f"📂 Loading cluster assignments from: {self.cluster_csv}")
        cluster_df = pd.read_csv(self.cluster_csv)
        
        if 'site' not in cluster_df.columns or 'cluster' not in cluster_df.columns:
            raise ValueError("Cluster CSV must have 'site' and 'cluster' columns")
        
        self.cluster_assignments = dict(zip(cluster_df['site'], cluster_df['cluster']))
        clusters = set(self.cluster_assignments.values())
        
        print(f"  ✅ Loaded cluster assignments:")
        print(f"    📊 Total sites: {len(self.cluster_assignments)}")
        print(f"    📊 Clusters: {sorted(clusters)}")
        
        # Filter by training sites if site split is loaded
        if self.train_sites:
            train_cluster_assignments = {site: cluster for site, cluster in self.cluster_assignments.items() 
                                       if site in self.train_sites}
            train_clusters = set(train_cluster_assignments.values())
            
            print(f"  🎯 Training sites only:")
            print(f"    📊 Train sites with clusters: {len(train_cluster_assignments)}")
            print(f"    📊 Train clusters: {sorted(train_clusters)}")
            
            # Show cluster distribution for training sites
            for cluster_id in sorted(train_clusters):
                sites_in_cluster = [site for site, cluster in train_cluster_assignments.items() 
                                  if cluster == cluster_id]
                print(f"    📊 Cluster {cluster_id}: {len(sites_in_cluster)} train sites")
            
            self.cluster_assignments = train_cluster_assignments
    
    def analyze_cluster_data_requirements(self):
        """Analyze data size and memory requirements for each cluster"""
        print("\n📊 Analyzing cluster data requirements...")
        
        if not self.cluster_assignments:
            print("  ⚠️  No cluster assignments loaded - analyzing all parquet files")
            return self._analyze_all_parquet_files()
        
        cluster_info = {}
        total_size_mb = 0
        total_rows = 0
        
        # Group sites by cluster
        clusters = {}
        for site, cluster_id in self.cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(site)
        
        print(f"  📊 Analyzing {len(clusters)} clusters...")
        
        for cluster_id, sites in clusters.items():
            cluster_size_mb = 0
            cluster_rows = 0
            valid_sites = []
            
            print(f"  📂 Cluster {cluster_id}: {len(sites)} sites")
            
            for site in sites:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                
                if not os.path.exists(parquet_file):
                    print(f"    ❌ Missing: {site}")
                    continue
                
                try:
                    # Get file size
                    file_size_mb = os.path.getsize(parquet_file) / (1024**2)
                    cluster_size_mb += file_size_mb
                    
                    # Sample rows (just for target column to check data availability)
                    df_sample = pd.read_parquet(parquet_file, columns=[self.target_col])
                    valid_rows = len(df_sample.dropna(subset=[self.target_col]))
                    cluster_rows += valid_rows
                    
                    valid_sites.append(site)
                    
                    del df_sample
                    gc.collect()
                    
                except Exception as e:
                    print(f"    ❌ Error analyzing {site}: {e}")
                    continue
            
            cluster_info[cluster_id] = {
                'sites': valid_sites,
                'size_mb': cluster_size_mb,
                'rows': cluster_rows
            }
            
            total_size_mb += cluster_size_mb
            total_rows += cluster_rows
            
            print(f"    ✅ {len(valid_sites)} valid sites, {cluster_rows:,} rows, {cluster_size_mb:.1f} MB")
        
        print(f"\n📈 Total data requirements:")
        print(f"  Total valid sites: {sum(len(info['sites']) for info in cluster_info.values())}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        
        return cluster_info
    
    def _analyze_all_parquet_files(self):
        """Fallback method to analyze all parquet files when no cluster assignments"""
        parquet_files = [f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')]
        total_files = len(parquet_files)
        
        # Sample a few files to estimate size
        sample_files = parquet_files[:min(10, total_files)]
        total_rows = 0
        total_size_mb = 0
        
        for parquet_file in sample_files:
            file_path = os.path.join(self.parquet_dir, parquet_file)
            try:
                # Get file size
                file_size_mb = os.path.getsize(file_path) / (1024**2)
                total_size_mb += file_size_mb
                
                # Sample rows
                df_sample = pd.read_parquet(file_path, columns=[self.target_col])
                valid_rows = len(df_sample.dropna(subset=[self.target_col]))
                total_rows += valid_rows
                
                print(f"  📄 {parquet_file}: {valid_rows:,} valid rows, {file_size_mb:.1f} MB")
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error analyzing {parquet_file}: {e}")
        
        # Estimate total data size
        if len(sample_files) > 0:
            avg_size_mb = total_size_mb / len(sample_files)
            estimated_total_size_gb = (avg_size_mb * total_files) / 1024
            
            avg_rows = total_rows / len(sample_files)
            estimated_total_rows = int(avg_rows * total_files)
        else:
            estimated_total_size_gb = 0
            estimated_total_rows = 0
        
        print(f"\n📈 Data size estimates:")
        print(f"  Total files: {total_files}")
        print(f"  Estimated total rows: {estimated_total_rows:,}")
        print(f"  Estimated total size: {estimated_total_size_gb:.1f} GB")
        
        return {'all_files': {
            'sites': [f.replace('_comprehensive.parquet', '') for f in parquet_files],
            'size_mb': estimated_total_size_gb * 1024,
            'rows': estimated_total_rows
        }}
    
    def get_xgboost_params(self):
        """Get XGBoost parameters optimized for GPU or CPU"""
        if self.use_gpu:
            # GPU-optimized parameters
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',
                'gpu_id': self.gpu_id,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': 1,
                'n_jobs': -1
            }
            print(f"  🎮 Using GPU-optimized XGBoost parameters")
        else:
            # CPU-optimized parameters
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'verbosity': 1,
                'n_jobs': -1
            }
            print(f"  💻 Using CPU-optimized XGBoost parameters")
        
        return params
    
    def load_cluster_data_from_parquet(self, cluster_id, cluster_sites):
        """Load cluster data with memory-aware strategy"""
        print(f"\n📊 Loading data for cluster {cluster_id}...")
        
        # Analyze cluster size first
        total_size_mb = 0
        total_rows = 0
        valid_sites = []
        
        for site in cluster_sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"    ❌ Missing: {site}")
                continue
                
            try:
                file_size_mb = os.path.getsize(parquet_file) / (1024**2)
                df_sample = pd.read_parquet(parquet_file, columns=[self.target_col])
                valid_rows = len(df_sample.dropna(subset=[self.target_col]))
                
                total_size_mb += file_size_mb
                total_rows += valid_rows
                valid_sites.append(site)
                
                print(f"    ✅ {site}: {valid_rows:,} rows, {file_size_mb:.1f} MB")
                
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error analyzing {site}: {e}")
                continue
        
        if len(valid_sites) < 2:
            raise ValueError(f"Need at least 2 sites for training, found {len(valid_sites)} in cluster {cluster_id}")
        
        # Check if we should use streaming or in-memory approach
        available_memory_gb = get_available_memory_gb()
        estimated_memory_gb = total_size_mb / 1024 * 2  # Rough estimate with processing overhead
        
        print(f"  💾 Available memory: {available_memory_gb:.1f} GB")
        print(f"  📊 Estimated memory needed: {estimated_memory_gb:.1f} GB")
        
        use_streaming = (total_size_mb > self.streaming_threshold_mb) or (total_rows > self.streaming_threshold_rows)
        
        if use_streaming:
            print(f"  💾 Using STREAMING approach (large dataset: {total_size_mb:.1f} MB, {total_rows:,} rows)")
            return self._load_cluster_data_streaming(cluster_id, valid_sites)
        else:
            print(f"  🚀 Using IN-MEMORY approach (small dataset: {total_size_mb:.1f} MB, {total_rows:,} rows)")
            return self._load_cluster_data_in_memory(cluster_id, valid_sites)
    
    def _load_cluster_data_in_memory(self, cluster_id, sites):
        """Load cluster data in memory (for smaller clusters)"""
        print(f"    📊 Loading cluster data in memory...")
        
        cluster_data = []
        
        for site in sites:
            try:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                df_site = pd.read_parquet(parquet_file)
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) == 0:
                    continue
                
                # Add site identifier for potential debugging
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
        
        return combined_df, sites
    
    def _load_cluster_data_streaming(self, cluster_id, sites):
        """Prepare cluster data for streaming processing"""
        print(f"    🔧 Preparing streaming processing setup...")
        
        # For streaming, we return the site list and let the training method handle chunked loading
        return None, sites  # None indicates streaming mode
    
    def prepare_features(self, df):
        """Prepare features for training by removing non-feature columns"""
        # Remove non-feature columns
        exclude_cols = [
            self.target_col, 'site', 'TIMESTAMP', 'solar_TIMESTAMP', 
            'plant_id', 'Unnamed: 0'
        ]
        
        # Also remove any columns ending with specific suffixes
        exclude_suffixes = ['_flags', '_md']
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if any(col.endswith(suffix) for suffix in exclude_suffixes):
                continue
            feature_cols.append(col)
        
        return df[feature_cols], feature_cols
    
    def train_cluster_model_from_parquet(self, cluster_id, cluster_sites):
        """Train XGBoost model for a specific cluster using parquet data"""
        print(f"\n🎯 Training model for cluster {cluster_id}...")
        
        try:
            # Load cluster data
            cluster_data, sites = self.load_cluster_data_from_parquet(cluster_id, cluster_sites)
            
            if cluster_data is None:
                # Streaming mode
                return self._train_cluster_streaming(cluster_id, sites)
            else:
                # In-memory mode
                return self._train_cluster_in_memory(cluster_id, cluster_data, sites)
                
        except Exception as e:
            print(f"    ❌ Training failed for cluster {cluster_id}: {e}")
            return None
    
    def _train_cluster_in_memory(self, cluster_id, df, sites):
        """Train cluster model with in-memory data"""
        print(f"    🚀 Training in-memory model for cluster {cluster_id}...")
        
        # Prepare features
        X, feature_cols = self.prepare_features(df)
        y = df[self.target_col]
        
        print(f"    📊 Training data: {len(X):,} rows, {len(feature_cols)} features")
        print(f"    📊 Sites: {len(sites)} ({', '.join(sites[:5])}{'...' if len(sites) > 5 else ''})")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        
        print(f"    📊 Train: {len(X_train):,} rows, Test: {len(X_test):,} rows")
        
        # Get XGBoost parameters
        xgb_params = self.get_xgboost_params()
        
        # Train model
        print(f"    🎯 Training XGBoost model...")
        log_memory_usage("Before training")
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        
        log_memory_usage("After training")
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(cluster_id, len(X), len(feature_cols), 
                                        y_train, y_test, y_pred_train, y_pred_test)
        
        # Save model and results
        model_info = self._save_model_and_importance(cluster_id, model, feature_cols, metrics, sites)
        
        # Clean up
        del X, y, X_train, X_test, y_train, y_test, df
        gc.collect()
        
        print(f"    ✅ Cluster {cluster_id} training completed!")
        print(f"    📊 Train R²: {metrics['train_r2']:.3f}, Test R²: {metrics['test_r2']:.3f}")
        
        return model_info
    
    def _train_cluster_streaming(self, cluster_id, sites):
        """Train cluster model with streaming data (for large clusters)"""
        print(f"    💾 Training streaming model for cluster {cluster_id}...")
        
        # For streaming mode, we need to implement chunked loading and training
        # This is more complex and would require XGBoost's incremental learning
        # For now, let's implement a simplified version that loads data in chunks
        
        print(f"    ⚠️  Streaming training not yet implemented - falling back to chunked in-memory")
        
        # Load data in smaller chunks and combine
        cluster_data = []
        total_rows = 0
        
        for site in sites:
            try:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                
                # Load in chunks
                chunk_iter = pd.read_parquet(parquet_file, chunksize=self.chunk_size)
                
                for chunk in chunk_iter:
                    chunk = chunk.dropna(subset=[self.target_col])
                    if len(chunk) > 0:
                        cluster_data.append(chunk)
                        total_rows += len(chunk)
                    
                    # Prevent memory overflow
                    if total_rows > 5000000:  # 5M row limit for safety
                        print(f"    ⚠️  Reached row limit, stopping at {total_rows:,} rows")
                        break
                
                if total_rows > 5000000:
                    break
                    
            except Exception as e:
                print(f"    ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            raise ValueError(f"No data loaded for cluster {cluster_id}")
        
        # Combine chunks
        combined_df = pd.concat(cluster_data, ignore_index=True)
        print(f"    ✅ Loaded {len(combined_df):,} rows via chunked streaming")
        
        # Clean up chunks
        del cluster_data
        gc.collect()
        
        # Now train with the combined data
        return self._train_cluster_in_memory(cluster_id, combined_df, sites)
    
    def _calculate_metrics(self, cluster_id, total_rows, feature_count, y_train, y_test, y_pred_train, y_pred_test):
        """Calculate training metrics"""
        metrics = {
            'cluster_id': cluster_id,
            'total_rows': total_rows,
            'feature_count': feature_count,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def _save_model_and_importance(self, cluster_id, model, feature_cols, metrics, sites):
        """Save trained model and feature importance"""
        # Save model in pickle format (for ensemble pipeline compatibility)
        model_file = os.path.join(self.results_dir, f'cluster_{cluster_id}_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"  💾 Model saved: {model_file}")
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_file = os.path.join(self.results_dir, f'feature_importance_cluster_{cluster_id}_{self.timestamp}.csv')
        importance_df.to_csv(importance_file, index=False)
        print(f"  📊 Feature importance saved: {importance_file}")
        
        # Save individual cluster metrics
        metrics_with_info = {
            **metrics,
            'cluster_id': cluster_id,
            'n_sites': len(sites),
            'sites': ', '.join(sites),
            'n_features': len(feature_cols),
            'timestamp': self.timestamp
        }
        
        individual_metrics_file = os.path.join(self.results_dir, f'cluster_{cluster_id}_metrics.json')
        with open(individual_metrics_file, 'w') as f:
            json.dump(metrics_with_info, f, indent=2)
        print(f"  📈 Metrics saved: {individual_metrics_file}")
        
        # Also save to combined metrics file
        combined_metrics_file = os.path.join(self.results_dir, f'cluster_model_metrics_{self.timestamp}.csv')
        
        # Append or create combined metrics file
        if os.path.exists(combined_metrics_file):
            existing_metrics = pd.read_csv(combined_metrics_file)
            metrics_df = pd.concat([existing_metrics, pd.DataFrame([metrics_with_info])], ignore_index=True)
        else:
            metrics_df = pd.DataFrame([metrics_with_info])
        
        metrics_df.to_csv(combined_metrics_file, index=False)
        
        return {
            'cluster_id': cluster_id,
            'model_file': model_file,
            'importance_file': importance_file,
            'metrics_file': individual_metrics_file,
            'metrics': metrics_with_info,
            'sites': sites,
            'feature_count': len(feature_cols)
        }
    
    def _old_load_preprocessed_files_DEPRECATED(self):
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
    
    def _old_train_cluster_model_from_libsvm_DEPRECATED(self, cluster_id, libsvm_file, feature_cols, total_rows):
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
            return self._old_train_external_memory_DEPRECATED(cluster_id, libsvm_file, feature_cols, total_rows)
        else:
            print(f"  🚀 Using IN-MEMORY training (file fits comfortably in RAM)")
            return self._old_train_in_memory_DEPRECATED(cluster_id, libsvm_file, feature_cols, total_rows)
    
    def _old_train_in_memory_DEPRECATED(self, cluster_id, libsvm_file, feature_cols, total_rows):
        """Train model in memory (for smaller datasets)"""
        print("  📊 Loading data into memory for train/test split...")
        # X, y = load_svmlight_file(libsvm_file)  # DEPRECATED - method not used
        raise NotImplementedError("This method is deprecated. Use train_cluster_model_from_parquet instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"  📊 Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")
        
        # XGBoost parameters
        params = self._old_get_xgboost_params_DEPRECATED()
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
    
    def _old_train_external_memory_DEPRECATED(self, cluster_id, libsvm_file, feature_cols, total_rows):
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
            params = self._old_get_xgboost_params_DEPRECATED()
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
    
    def _old_get_xgboost_params_DEPRECATED(self):
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
    
    def _save_model_and_importance_DEPRECATED(self, cluster_id, model, feature_cols, metrics):
        """DEPRECATED: Save model and feature importance with mapping - use new version instead"""
        raise NotImplementedError("This method is deprecated - use the new _save_model_and_importance method")
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
            
            # Save basic importance file
            importance_path = os.path.join(self.results_dir, f'feature_importance_cluster_{cluster_id}_{self.timestamp}.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"  💾 Feature importance saved: {importance_path}")
            
            # Create mapped version with descriptions and categories
            mapped_importance = self._map_feature_importance(importance_df, cluster_id)
            if mapped_importance is not None:
                mapped_path = os.path.join(self.results_dir, f'feature_importance_cluster_{cluster_id}_{self.timestamp}_mapped.csv')
                mapped_importance.to_csv(mapped_path, index=False)
                print(f"  📊 Mapped feature importance saved: {mapped_path}")
                
                # Print top features summary
                print(f"  🏆 Top 10 features for cluster {cluster_id}:")
                for i, row in mapped_importance.head(10).iterrows():
                    print(f"    {row['feature_index']}: {row['feature_name']} ({row['category']}) - {row['importance_score']:.2f}")
                
        except Exception as e:
            print(f"  ⚠️  Could not save feature importance: {e}")
    
    def _map_feature_importance(self, importance_df, cluster_id):
        """Map feature importance using the v2 feature mapping"""
        try:
            # Try to load the feature mapping from the feature_importance directory
            mapping_paths = [
                '../../feature_importance/feature_mapping_v2_final.csv',
                '../../feature_importance/xgboost_scripts/feature_mapping_v2_complete.csv',
                '../feature_importance/feature_mapping_v2_final.csv',
                './feature_importance/feature_mapping_v2_final.csv'
            ]
            
            mapping_file = None
            for path in mapping_paths:
                if os.path.exists(path):
                    mapping_file = path
                    break
            
            if mapping_file is None:
                print(f"  ⚠️  Feature mapping file not found. Tried paths: {mapping_paths}")
                print(f"  💡 To enable feature mapping, ensure feature_mapping_v2_final.csv exists in feature_importance directory")
                return None
            
            # Load the feature mapping
            df_mapping = pd.read_csv(mapping_file)
            mapping_dict = dict(zip(df_mapping['feature_index'], df_mapping['feature_name']))
            desc_dict = dict(zip(df_mapping['feature_index'], df_mapping['description']))
            cat_dict = dict(zip(df_mapping['feature_index'], df_mapping['category']))
            
            # Map feature names and add descriptions/categories
            mapped_results = []
            
            for _, row in importance_df.iterrows():
                feature_index = row['feature_index']
                importance_score = row['importance']
                
                # Get mapped information
                feature_name = mapping_dict.get(feature_index, row['feature_name'])
                description = desc_dict.get(feature_index, "No description available")
                category = cat_dict.get(feature_index, "Unknown category")
                
                mapped_results.append({
                    'feature_index': feature_index,
                    'feature_name': feature_name,
                    'description': description,
                    'category': category,
                    'importance_score': importance_score
                })
            
            # Create DataFrame and sort by importance
            df_mapped = pd.DataFrame(mapped_results)
            df_mapped = df_mapped.sort_values('importance_score', ascending=False)
            
            # Print summary by category
            category_importance = df_mapped.groupby('category')['importance_score'].sum().sort_values(ascending=False)
            print(f"  📊 Total importance by category for cluster {cluster_id}:")
            for category, total_importance in category_importance.head(5).items():
                print(f"    {category}: {total_importance:.2f}")
            
            return df_mapped
            
        except Exception as e:
            print(f"  ⚠️  Could not map feature importance: {e}")
            return None
    
    def train_all_cluster_models(self, force_reprocess=False):
        """Train models for all clusters using direct parquet processing"""
        print("🚀 Starting GPU-Optimized Cluster Model Training")
        print("=" * 80)
        print(f"🎮 GPU enabled: {self.use_gpu}")
        print(f"📊 Site split: {bool(self.train_sites)}")
        print(f"📊 Cluster assignments: {bool(self.cluster_assignments)}")
        
        if not self.cluster_assignments:
            raise ValueError("No cluster assignments loaded! Provide cluster_csv parameter or call _load_cluster_assignments()")
        
        # Analyze data requirements
        cluster_info = self.analyze_cluster_data_requirements()
        
        try:
            print(f"\n🏋️  TRAINING STAGE: Direct parquet processing with GPU acceleration")
            print("=" * 70)
            
            all_model_info = []
            all_metrics = []
            
            # Group sites by cluster
            clusters = {}
            for site, cluster_id in self.cluster_assignments.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(site)
            
            # Train each cluster
            for cluster_id in sorted(clusters.keys()):
                cluster_sites = clusters[cluster_id]
                
                print(f"\n{'='*60}")
                print(f"TRAINING CLUSTER {cluster_id} MODEL")
                print(f"{'='*60}")
                print(f"Sites: {len(cluster_sites)} ({', '.join(cluster_sites[:3])}{'...' if len(cluster_sites) > 3 else ''})")
                
                log_memory_usage(f"Before training cluster {cluster_id}")
                
                try:
                    # Train model from parquet data
                    model_info = self.train_cluster_model_from_parquet(cluster_id, cluster_sites)
                    
                    if model_info:
                        all_model_info.append(model_info)
                        all_metrics.append(model_info['metrics'])
                        log_memory_usage(f"After training cluster {cluster_id}")
                    else:
                        print(f"    ❌ Training failed for cluster {cluster_id}")
                    
                except Exception as e:
                    print(f"    ❌ Error training cluster {cluster_id}: {e}")
                    continue
            
            # Save summary and print results
            if all_metrics:
                print(f"\n📊 TRAINING SUMMARY")
                print(f"=" * 30)
                print(f"Total clusters trained: {len(all_metrics)}")
                
                # Calculate summary statistics
                train_r2_scores = [m['train_r2'] for m in all_metrics]
                test_r2_scores = [m['test_r2'] for m in all_metrics]
                test_rmse_scores = [m['test_rmse'] for m in all_metrics]
                
                print(f"Average train R²: {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
                print(f"Average test R²: {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
                print(f"Average test RMSE: {np.mean(test_rmse_scores):.4f} ± {np.std(test_rmse_scores):.4f}")
                
                print(f"\nCluster Performance:")
                for model_info in all_model_info:
                    metrics = model_info['metrics']
                    print(f"  Cluster {metrics['cluster_id']}: Train R² = {metrics['train_r2']:.4f}, Test R² = {metrics['test_r2']:.4f}, RMSE = {metrics['test_rmse']:.4f}")
                
                print(f"\n✅ GPU-optimized cluster training completed successfully!")
                print(f"📁 Results saved to: {self.results_dir}")
                print(f"🎮 GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
                
                if self.train_sites:
                    print(f"🎯 Trained on {len(self.train_sites)} training sites only")
                    print(f"🔒 {len(self.test_sites)} test sites withheld for ensemble validation")
                
                return all_model_info
            else:
                print(f"\n❌ No models were trained successfully")
                return None
                
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise

def main():
    """Main function to run GPU-optimized cluster training"""
    parser = argparse.ArgumentParser(description="GPU-Optimized Cluster Model Training")
    parser.add_argument('--cluster-csv', required=True,
                        help="Path to cluster assignments CSV file")
    parser.add_argument('--site-split-file', 
                        help="Path to site split JSON file (for train-only training)")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--results-dir', default='./results/cluster_models',
                        help="Directory to save results")
    parser.add_argument('--force-gpu', action='store_true',
                        help="Force GPU usage even if detection fails")
    parser.add_argument('--force-reprocess', action='store_true',
                        help="Force reprocessing even if models exist")
    
    args = parser.parse_args()
    
    print("🌍 SAPFLUXNET GPU-Optimized Cluster Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    print("📋 MODERNIZED WORKFLOW:")
    print("   ✅ Direct parquet processing (no LibSVM dependency)")
    print("   ✅ GPU acceleration with automatic detection")
    print("   ✅ Site split support for train-only models")
    print("   ✅ Memory-efficient chunked processing")
    print("   ✅ Advanced feature preparation")
    print()
    
    try:
        # Initialize trainer
        trainer = GPUOptimizedClusterTrainer(
            parquet_dir=args.parquet_dir,
            results_dir=args.results_dir,
            cluster_csv=args.cluster_csv,
            site_split_file=args.site_split_file,
            force_gpu=args.force_gpu
        )
        
        # Train all cluster models
        model_info_list = trainer.train_all_cluster_models(args.force_reprocess)
        
        if model_info_list:
            print(f"\n🎉 GPU-optimized training completed successfully!")
            print(f"📁 Results saved to: {trainer.results_dir}")
            print(f"🎮 GPU acceleration: {'Enabled' if trainer.use_gpu else 'Disabled'}")
            
            if trainer.train_sites:
                print(f"🎯 Trained on {len(trainer.train_sites)} training sites only")
                print(f"🔒 {len(trainer.test_sites)} test sites withheld for ensemble validation")
                print(f"📊 Ready for ensemble testing pipeline!")
            else:
                print(f"📊 Trained on all available sites")
                
        else:
            print(f"\n❌ Training failed - no models were created")
            return 1
                
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nFinished at: {datetime.now()}")
    return 0

if __name__ == "__main__":
    main()
