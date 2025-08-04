"""
Memory-Optimized Cluster Data Preprocessing for SAPFLUXNET
Preprocesses parquet files into libsvm format for cluster-specific training
Separated from training to allow for better control and cleanup

NEW FEATURES:
- Non-destructive workflow: Reads cluster assignments from CSV instead of modifying parquet files
- Balanced sampling: Addresses site imbalance within clusters using median-based sampling
- On-the-fly cluster assignment: Assigns cluster labels during preprocessing without altering original data

USAGE:
1. With cluster CSV file:
   python preprocess_cluster_data.py --cluster-csv ../evaluation/clustering_results/advanced_site_clusters_20250723_215127.csv

2. With balanced sampling enabled:
   python preprocess_cluster_data.py --cluster-csv path/to/clusters.csv --apply-sampling

3. Legacy mode (reads cluster labels from parquet files):
   python preprocess_cluster_data.py

4. Auto-find latest cluster CSV:
   python preprocess_cluster_data.py --cluster-csv auto
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import warnings
import gc
import psutil
import shutil
import json
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')

def find_latest_cluster_csv(base_dir='../evaluation/clustering_results'):
    """Find the latest cluster assignments CSV file"""
    csv_pattern = os.path.join(base_dir, 'advanced_site_clusters_*.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f'No advanced_site_clusters_*.csv found in {base_dir}')
    
    latest_csv = csv_files[-1]
    print(f"🔍 Auto-found latest cluster CSV: {latest_csv}")
    return latest_csv

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def get_total_memory_gb():
    """Get total system memory in GB"""
    return psutil.virtual_memory().total / (1024**3)

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")

class ClusterDataPreprocessor:
    """
    Preprocesses cluster data to libsvm format for memory-efficient training
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', results_dir='./results/cluster_models', 
                 apply_sampling=False, min_fraction=0.3, sampling_seed=42, cluster_csv_path=None):
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.preprocessed_dir = os.path.join(results_dir, 'preprocessed_libsvm')
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Cluster assignments from CSV
        self.cluster_csv_path = cluster_csv_path
        self.cluster_assignments = None
        
        # Balanced sampling parameters
        self.apply_sampling = apply_sampling
        self.min_fraction = min_fraction
        self.sampling_seed = sampling_seed
        
        # Adaptive chunk size based on available memory
        available_memory = get_available_memory_gb()
        if available_memory > 20:  # High memory system
            self.chunk_size = 200000  # 200K rows per chunk
            print(f"🚀 HIGH-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        elif available_memory > 10:
            self.chunk_size = 100000  # 100K rows per chunk
            print(f"⚡ MEDIUM-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        else:
            self.chunk_size = 50000   # 50K rows per chunk
            print(f"💾 LOW-MEMORY mode: Using {self.chunk_size:,} rows per chunk")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Load cluster assignments if CSV path provided
        if self.cluster_csv_path:
            self.cluster_assignments = self.load_cluster_assignments()
        
        print(f"🔧 Cluster Data Preprocessor initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"📁 Results directory: {results_dir}")
        print(f"📁 Preprocessed directory: {self.preprocessed_dir}")
        print(f"🎯 Target column: {self.target_col}")
        print(f"🏷️  Cluster column: {self.cluster_col}")
        if self.cluster_csv_path:
            print(f"📊 Cluster assignments: {self.cluster_csv_path}")
            if self.cluster_assignments is not None:
                print(f"   Loaded {len(self.cluster_assignments)} site assignments")
        print(f"⚖️  Apply balanced sampling: {apply_sampling}")
        if apply_sampling:
            print(f"   Min fraction: {min_fraction}")
            print(f"   Sampling seed: {sampling_seed}")

    def load_cluster_assignments(self):
        """Load cluster assignments from CSV file"""
        if not self.cluster_csv_path:
            return None
            
        print(f"\n📊 Loading cluster assignments from {self.cluster_csv_path}...")
        
        if not os.path.exists(self.cluster_csv_path):
            print(f"❌ Cluster CSV file not found: {self.cluster_csv_path}")
            return None
        
        try:
            cluster_df = pd.read_csv(self.cluster_csv_path)
            
            # Validate required columns
            if 'site' not in cluster_df.columns or 'cluster' not in cluster_df.columns:
                raise ValueError("CSV must contain 'site' and 'cluster' columns")
            
            # Convert to dictionary for fast lookup
            cluster_dict = dict(zip(cluster_df['site'], cluster_df['cluster']))
            
            print(f"✅ Loaded cluster assignments for {len(cluster_dict)} sites")
            
            # Show cluster distribution
            cluster_counts = cluster_df['cluster'].value_counts().sort_index()
            print("📈 Cluster distribution:")
            for cluster_id, count in cluster_counts.items():
                print(f"   Cluster {cluster_id}: {count} sites")
            
            return cluster_dict
            
        except Exception as e:
            print(f"❌ Error loading cluster assignments: {e}")
            return None

    def analyze_data_requirements(self):
        """Analyze data size and memory requirements"""
        print("\n📊 Analyzing data requirements...")
        
        parquet_files = [f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')]
        total_files = len(parquet_files)
        
        if total_files == 0:
            raise ValueError(f"No parquet files found in {self.parquet_dir}")
        
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
        """Load cluster information using CSV assignments or from parquet files"""
        print("\n🔍 Loading cluster information...")
        
        cluster_info = {}
        parquet_files = sorted([f for f in os.listdir(self.parquet_dir) if f.endswith('.parquet')])
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.parquet_dir}")
        
        # Use CSV assignments if available, otherwise fall back to reading from parquet
        if self.cluster_assignments is not None:
            print("📊 Using cluster assignments from CSV file...")
            
            for parquet_file in parquet_files:
                site_name = parquet_file.replace('_comprehensive.parquet', '')
                file_path = os.path.join(self.parquet_dir, parquet_file)
                
                # Look up cluster assignment
                if site_name not in self.cluster_assignments:
                    print(f"  ⚠️  {site_name}: No cluster assignment found in CSV")
                    continue
                
                cluster_id = self.cluster_assignments[site_name]
                
                try:
                    # Load only target column for row counting
                    df_sample = pd.read_parquet(file_path, columns=[self.target_col])
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
        
        else:
            print("📊 Reading cluster assignments from parquet files (legacy mode)...")
            
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
        
        if not cluster_info:
            raise ValueError("No valid cluster information found! Make sure cluster assignments are available via CSV or parquet files have ecosystem_cluster column.")
        
        print(f"\n📊 Cluster distribution:")
        for cluster_id, info in sorted(cluster_info.items()):
            print(f"  Cluster {cluster_id}: {len(info['sites'])} sites, {info['total_rows']:,} total rows")
            print(f"    Sites: {', '.join(info['sites'][:5])}{'...' if len(info['sites']) > 5 else ''}")
        
        return cluster_info

    def clean_previous_preprocessing(self):
        """Clean all previous preprocessing files"""
        if not os.path.exists(self.preprocessed_dir):
            print("📁 No preprocessing directory found - nothing to clean")
            return
        
        print(f"\n🧹 Cleaning previous preprocessing files from {self.preprocessed_dir}...")
        
        # Find all preprocessing files
        libsvm_files = glob.glob(os.path.join(self.preprocessed_dir, '*.svm'))
        metadata_files = glob.glob(os.path.join(self.preprocessed_dir, '*_metadata.json'))
        temp_dirs = [d for d in os.listdir(self.preprocessed_dir) 
                    if os.path.isdir(os.path.join(self.preprocessed_dir, d)) and d.startswith('temp_')]
        
        total_files = len(libsvm_files) + len(metadata_files)
        total_size_mb = 0
        
        # Calculate total size
        for file_path in libsvm_files + metadata_files:
            if os.path.exists(file_path):
                total_size_mb += os.path.getsize(file_path) / (1024**2)
        
        if total_files == 0 and len(temp_dirs) == 0:
            print("✅ No preprocessing files found - directory is already clean")
            return
        
        print(f"🗑️  Found {total_files} files ({total_size_mb:.1f} MB) and {len(temp_dirs)} temp directories to remove")
        
        # Remove libsvm files
        for file_path in libsvm_files:
            try:
                os.remove(file_path)
                print(f"  ✅ Removed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ❌ Error removing {os.path.basename(file_path)}: {e}")
        
        # Remove metadata files
        for file_path in metadata_files:
            try:
                os.remove(file_path)
                print(f"  ✅ Removed: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ❌ Error removing {os.path.basename(file_path)}: {e}")
        
        # Remove temp directories
        for temp_dir in temp_dirs:
            temp_path = os.path.join(self.preprocessed_dir, temp_dir)
            try:
                shutil.rmtree(temp_path)
                print(f"  ✅ Removed temp directory: {temp_dir}")
            except Exception as e:
                print(f"  ❌ Error removing temp directory {temp_dir}: {e}")
        
        print(f"✅ Cleanup completed! Freed ~{total_size_mb:.1f} MB of disk space")

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
        processed_sites = []
        per_site_rows = {}  # Track row counts per site for validation script
        
        # Calculate cluster-wide sampling statistics if sampling is enabled
        sampling_limits = None
        if self.apply_sampling:
            sampling_limits = self._calculate_cluster_sampling_limits(cluster_id, cluster_sites)
        
        with open(libsvm_file, 'w') as output_file:
            for site_idx, site in enumerate(cluster_sites):
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                
                if not os.path.exists(parquet_file):
                    print(f"    ⚠️  Missing: {parquet_file}")
                    continue
                
                try:
                    print(f"    🔄 Processing {site} ({site_idx+1}/{len(cluster_sites)})...")
                    
                    # Check site size first - no cluster filtering needed now
                    df_info = pd.read_parquet(parquet_file, columns=[self.target_col])
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
                            parquet_file, cluster_id, output_file, all_features, sampling_limits
                        )
                    else:
                        print(f"      🔄 Loading site in memory")
                        site_rows_processed = self._process_site_in_memory_to_libsvm(
                            parquet_file, cluster_id, output_file, all_features, sampling_limits
                        )
                    
                    if site_rows_processed > 0:
                        total_rows += site_rows_processed
                        processed_sites.append(site)
                        per_site_rows[site] = site_rows_processed  # Save for validation script
                        print(f"      ✅ {site}: {site_rows_processed:,} rows processed")
                    else:
                        print(f"      ⚠️  {site}: No rows processed after filtering/sampling")
                    
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
            'cluster_id': int(cluster_id),
            'total_rows': int(total_rows),
            'feature_count': int(len(all_features)),
            'feature_names': [str(feature) for feature in all_features],
            'sites_requested': [str(site) for site in cluster_sites],
            'sites_processed': [str(site) for site in processed_sites],
            'sites_skipped': [str(site) for site in cluster_sites if site not in processed_sites],
            'per_site_rows': {str(site): int(rows) for site, rows in per_site_rows.items()},  # For validation script
            'created_at': datetime.now().isoformat(),
            'chunk_size_used': int(self.chunk_size),
            'balanced_sampling_applied': bool(self.apply_sampling),
            'sampling_parameters': {
                'min_fraction': float(self.min_fraction),
                'sampling_seed': int(self.sampling_seed)
            } if self.apply_sampling else None
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_size_mb = os.path.getsize(libsvm_file) / (1024**2)
        print(f"  ✅ Preprocessed cluster {cluster_id}: {total_rows:,} rows, {len(all_features)} features, {file_size_mb:.1f} MB")
        print(f"     Processed {len(processed_sites)}/{len(cluster_sites)} sites")
        if len(processed_sites) < len(cluster_sites):
            skipped = [site for site in cluster_sites if site not in processed_sites]
            print(f"     Skipped sites: {', '.join(skipped[:3])}{'...' if len(skipped) > 3 else ''}")
        
        return libsvm_file, metadata

    def _process_site_chunked_to_libsvm(self, parquet_file, cluster_id, output_file, all_features, sampling_limits):
        """Process a large site in chunks to avoid memory issues"""
        total_processed = 0
        site_name = os.path.basename(parquet_file).replace('_comprehensive.parquet', '')
        
        # Read parquet file in chunks using pyarrow for better memory control
        import pyarrow.parquet as pq
        
        parquet_table = pq.read_table(parquet_file)
        total_rows = len(parquet_table)
        
        # Collect all chunks for this site to apply balanced sampling at site level
        site_chunks = []
        
        # Process in chunks
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            
            # Read chunk
            chunk_table = parquet_table.slice(start_idx, end_idx - start_idx)
            df_chunk = chunk_table.to_pandas()
            
            # Filter for valid target (no cluster filtering needed - we assign cluster on-the-fly)
            df_chunk = df_chunk.dropna(subset=[self.target_col])
            
            if len(df_chunk) == 0:
                del df_chunk
                continue
            
            # Add site column for balanced sampling
            df_chunk['site'] = site_name
            site_chunks.append(df_chunk)
            
            # Clean up chunk
            del df_chunk
            gc.collect()
        
        if not site_chunks:
            del parquet_table
            gc.collect()
            return 0
        
        # Combine all chunks for this site
        df_site = pd.concat(site_chunks, ignore_index=True)
        del site_chunks
        gc.collect()
        
        # Apply sampling if limits are provided
        if sampling_limits and site_name in sampling_limits:
            target_size = sampling_limits[site_name]
            current_size = len(df_site)
            
            if current_size > target_size:
                print(f"        🎯 Applying sampling: {current_size:,} → {target_size:,}")
                np.random.seed(self.sampling_seed)
                df_site = df_site.sample(n=target_size, random_state=self.sampling_seed).reset_index(drop=True)
            else:
                print(f"        ✅ No sampling needed: {current_size:,} ≤ {target_size:,}")
        elif self.apply_sampling:
            print(f"        ⚠️  No sampling limit found for {site_name}")
        
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
        
        total_processed = len(X)
        
        # Clean up
        del df_site, X_df, X, y, parquet_table
        gc.collect()
        
        return total_processed

    def _process_site_in_memory_to_libsvm(self, parquet_file, cluster_id, output_file, all_features, sampling_limits):
        """Process a small site in memory"""
        site_name = os.path.basename(parquet_file).replace('_comprehensive.parquet', '')
        
        # Load site data
        df_site = pd.read_parquet(parquet_file)
        
        # Filter for valid target (no cluster filtering needed - we assign cluster on-the-fly)
        df_site = df_site.dropna(subset=[self.target_col])
        
        if len(df_site) == 0:
            del df_site
            return 0
        
        # Apply sampling if limits are provided
        if sampling_limits and site_name in sampling_limits:
            target_size = sampling_limits[site_name]
            current_size = len(df_site)
            
            if current_size > target_size:
                print(f"        🎯 Applying sampling: {current_size:,} → {target_size:,}")
                np.random.seed(self.sampling_seed)
                df_site = df_site.sample(n=target_size, random_state=self.sampling_seed).reset_index(drop=True)
            else:
                print(f"        ✅ No sampling needed: {current_size:,} ≤ {target_size:,}")
        elif self.apply_sampling:
            print(f"        ⚠️  No sampling limit found for {site_name}")
        
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

    def preprocess_all_clusters(self, force_reprocess=False, clean_first=False):
        """Preprocess all clusters to libsvm format"""
        print("\n🔧 PREPROCESSING STAGE: Converting clusters to clean libsvm format")
        print("=" * 70)
        
        # Clean previous files if requested
        if clean_first:
            self.clean_previous_preprocessing()
        
        # Load cluster information
        cluster_info = self.load_cluster_info_memory_efficient(self.cluster_file)
        
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

    def apply_balanced_sampling_to_cluster_data(self, df, cluster_id):
        """
        Apply median-based balanced sampling within a cluster
        
        This addresses site imbalance by:
        1. Finding median sample size within the cluster
        2. Setting max samples per site to median value
        3. Setting min samples per site to fraction of median
        4. Randomly sampling to achieve balance
        
        Args:
            df: DataFrame with cluster data and site identifiers
            cluster_id: The cluster ID being processed
            
        Returns:
            DataFrame with balanced sampling applied
        """
        if not self.apply_sampling:
            return df
        
        if 'site' not in df.columns:
            print(f"      ⚠️  No site column found - skipping sampling for cluster {cluster_id}")
            return df
        
        original_size = len(df)
        print(f"      🎯 Applying balanced sampling to cluster {cluster_id}...")
        
        # Check if this is actually multi-site data or single site
        unique_sites = df['site'].unique()
        if len(unique_sites) == 1:
            print(f"      ℹ️  Single site ({unique_sites[0]}) - no site balancing needed")
            return df
        
        # Calculate site sample sizes within this cluster
        site_sizes = df.groupby('site').size().sort_values(ascending=False)
        
        if len(site_sizes) <= 1:
            print(f"      ⚠️  Only {len(site_sizes)} site(s) in cluster {cluster_id} - no sampling needed")
            return df
        
        # Calculate median sample size for this cluster
        median_size = int(site_sizes.median())
        min_size = max(int(median_size * self.min_fraction), 1)  # At least 1 sample per site
        
        print(f"      📊 Sites: {len(site_sizes)}, Samples: {len(df):,}")
        print(f"      📈 Min size: {site_sizes.min()}, Max size: {site_sizes.max()}, Median: {median_size}")
        print(f"      🎯 Target range: {min_size} - {median_size} samples per site")
        
        # Check if sampling is actually needed
        sites_need_downsampling = (site_sizes > median_size).sum()
        if sites_need_downsampling == 0:
            print(f"      ✅ No sites exceed median size - no sampling needed")
            return df
        
        # Apply sampling to each site in this cluster
        balanced_sites = []
        sites_modified = 0
        
        np.random.seed(self.sampling_seed)  # Set seed for reproducible sampling
        
        for site_name, site_group in df.groupby('site'):
            site_size = len(site_group)
            
            if site_size > median_size:
                # Downsample large sites
                sampled_site = site_group.sample(n=median_size, random_state=self.sampling_seed)
                sites_modified += 1
                print(f"        ↓ {site_name}: {site_size:,} → {median_size:,} (downsampled)")
            
            elif site_size < min_size:
                # Keep small sites as-is (could implement upsampling with SMOTE if needed)
                sampled_site = site_group.copy()
                print(f"        ↑ {site_name}: {site_size:,} → {site_size:,} (kept small)")
            
            else:
                # Site is within target range
                sampled_site = site_group.copy()
                print(f"        = {site_name}: {site_size:,} → {site_size:,} (no change)")
            
            balanced_sites.append(sampled_site)
        
        # Combine all sites
        balanced_df = pd.concat(balanced_sites, ignore_index=True)
        final_size = len(balanced_df)
        
        # Report statistics
        final_site_sizes = balanced_df.groupby('site').size()
        print(f"      ✅ Sampling complete: {original_size:,} → {final_size:,} samples ({final_size/original_size*100:.1f}% retained)")
        print(f"         Modified {sites_modified} sites (downsampled)")
        print(f"         New size range: {final_site_sizes.min()} - {final_site_sizes.max()}")
        
        return balanced_df

    def _calculate_cluster_sampling_limits(self, cluster_id, cluster_sites):
        """Calculate sampling limits for all sites in a cluster"""
        print(f"      📊 Calculating cluster-wide sampling limits for cluster {cluster_id}...")
        
        site_sizes = {}
        
        # First pass: collect sample sizes for all sites in cluster
        for site in cluster_sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"        ⚠️  Missing: {parquet_file}")
                continue
                
            try:
                # Load only target column to count valid rows
                df_info = pd.read_parquet(parquet_file, columns=[self.target_col])
                df_info = df_info.dropna(subset=[self.target_col])
                site_sizes[site] = len(df_info)
                del df_info
                gc.collect()
                
            except Exception as e:
                print(f"        ❌ Error analyzing {site}: {e}")
                continue
        
        if len(site_sizes) <= 1:
            print(f"        ℹ️  Only {len(site_sizes)} valid site(s) - no sampling limits needed")
            return None
        
        # Calculate cluster statistics
        sizes_array = np.array(list(site_sizes.values()))
        median_size = int(np.median(sizes_array))
        min_size = max(int(median_size * self.min_fraction), 1)
        
        print(f"        📈 Sites: {len(site_sizes)}, Median: {median_size:,}, Min threshold: {min_size:,}")
        print(f"        📊 Size range: {sizes_array.min():,} - {sizes_array.max():,}")
        
        # Calculate sampling limits for each site
        sampling_limits = {}
        sites_to_downsample = 0
        
        for site, size in site_sizes.items():
            if size > median_size:
                sampling_limits[site] = median_size
                sites_to_downsample += 1
                print(f"          ↓ {site}: {size:,} → {median_size:,} (downsample)")
            else:
                sampling_limits[site] = size  # Keep original size
                print(f"          = {site}: {size:,} → {size:,} (keep)")
        
        print(f"        🎯 Will downsample {sites_to_downsample}/{len(site_sizes)} sites")
        
        return sampling_limits

def main():
    """Main function for cluster data preprocessing"""
    parser = argparse.ArgumentParser(description="Cluster Data Preprocessing for SAPFLUXNET")
    parser.add_argument('--action', choices=['preprocess', 'clean', 'status'], default='preprocess',
                        help="Action: 'preprocess' (convert to libsvm), 'clean' (remove previous files), 'status' (check existing files)")
    parser.add_argument('--force-reprocess', action='store_true',
                        help="Force reprocessing of all clusters even if files exist")
    parser.add_argument('--clean-first', action='store_true',
                        help="Clean previous preprocessing files before starting")
    parser.add_argument('--parquet-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--results-dir', default='./results/cluster_models',
                        help="Directory to save results")
    parser.add_argument('--cluster-csv', type=str,
                        help="Path to CSV file with cluster assignments (site,cluster columns) or 'auto' for auto-find")
    parser.add_argument('--apply-sampling', action='store_true',
                        help="Apply balanced sampling to address site imbalance within clusters")
    parser.add_argument('--min-fraction', type=float, default=0.3,
                        help="Minimum fraction of median samples per site (default: 0.3)")
    parser.add_argument('--sampling-seed', type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    
    args = parser.parse_args()
    
    print("🔧 SAPFLUXNET Cluster Data Preprocessing")
    print("=" * 50)
    print(f"Action: {args.action.upper()}")
    print(f"Started at: {datetime.now()}")
    
    # Show configuration
    cluster_csv_path = args.cluster_csv
    if args.cluster_csv == 'auto':
        try:
            cluster_csv_path = find_latest_cluster_csv()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("💡 Try specifying the cluster CSV path manually with --cluster-csv")
            return
    
    if cluster_csv_path:
        print(f"📊 Cluster assignments: {cluster_csv_path}")
    else:
        print("📊 Cluster assignments: Reading from parquet files (legacy mode)")
    
    if args.apply_sampling:
        print(f"⚖️  Balanced sampling: ENABLED")
        print(f"   Min fraction: {args.min_fraction}")
        print(f"   Sampling seed: {args.sampling_seed}")
    else:
        print(f"⚖️  Balanced sampling: DISABLED")
    
    try:
        # Initialize preprocessor
        preprocessor = ClusterDataPreprocessor(
            parquet_dir=args.parquet_dir,
            results_dir=args.results_dir,
            apply_sampling=args.apply_sampling,
            min_fraction=args.min_fraction,
            sampling_seed=args.sampling_seed,
            cluster_csv_path=cluster_csv_path
        )
        
        if args.action == 'clean':
            # Only clean previous files
            preprocessor.clean_previous_preprocessing()
            print("✅ Cleanup completed!")
            
        elif args.action == 'status':
            # Only check status
            print("\n📊 Checking preprocessing status...")
            try:
                cluster_info = preprocessor.load_cluster_info_memory_efficient(args.cluster_csv if args.cluster_csv != 'auto' else None)
                existing_files, missing_clusters = preprocessor.check_preprocessed_files_exist(cluster_info)
                
                print(f"\n📈 Status Summary:")
                print(f"  Total clusters: {len(cluster_info)}")
                print(f"  Preprocessed: {len(existing_files)}")
                print(f"  Missing: {len(missing_clusters)}")
                
                if existing_files:
                    total_size_mb = sum(info['size_mb'] for info in existing_files.values())
                    print(f"  Total preprocessed size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
                
                if missing_clusters:
                    print(f"  Missing clusters: {missing_clusters}")
                    
            except Exception as e:
                print(f"❌ Error checking status: {e}")
            
        else:  # preprocess
            # Run preprocessing
            preprocessed_files = preprocessor.preprocess_all_clusters(
                force_reprocess=args.force_reprocess,
                clean_first=args.clean_first
            )
            
            if preprocessed_files:
                print(f"\n🎉 Preprocessing completed successfully!")
                print(f"📁 Preprocessed files saved to: {preprocessor.preprocessed_dir}")
                print(f"💡 Run the training script next: python train_cluster_models.py --mode train")
                
                # Show final statistics
                total_size_mb = sum(info['size_mb'] for info in preprocessed_files.values())
                total_rows = sum(info['metadata']['total_rows'] for info in preprocessed_files.values())
                print(f"\n📊 Final Statistics:")
                print(f"   Clusters processed: {len(preprocessed_files)}")
                print(f"   Total rows: {total_rows:,}")
                print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
                
                if args.apply_sampling:
                    print(f"   Balanced sampling: Applied with min_fraction={args.min_fraction}")
                
            else:
                print(f"\n❌ Preprocessing failed - no files were created")
                
    except Exception as e:
        print(f"\n❌ Preprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 