#!/usr/bin/env python3
"""
Ensemble Testing Pipeline for SAPFLUXNET Spatial Validation

This script tests trained cluster models on withheld test sites using multiple
ensemble weighting strategies. It identifies outlier sites and provides comprehensive
metrics both including and excluding outliers.

Key Features:
- Loads trained cluster models from all clusters
- Tests on withheld sites (20% of data) to prevent data leakage
- Implements multiple ensemble weighting strategies:
  * Confidence-weighted (based on prediction uncertainty)
  * Historical performance-weighted (based on cluster validation performance)
  * Distance-based weighting (based on similarity to cluster centroids)
  * Equal weighting (baseline)
- Identifies outlier sites with weak cluster relationships
- Provides comprehensive metrics with and without outliers
- GPU acceleration support

Usage:
    python ensemble_test_pipeline.py --site-split-file ../../site_split_assignment.json \
                                    --cluster-csv ../evaluation/clustering_results/advanced_site_clusters_*.csv \
                                    --models-dir ./results/cluster_models \
                                    --parquet-dir ../../processed_parquet \
                                    --results-dir ./results/ensemble_validation

Author: AI Assistant
Date: 2025-07-29
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import psutil
import gc
import subprocess

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def log_memory_usage(step_name):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    print(f"  💾 {step_name}: {memory.percent:.1f}% used ({memory.available/1024**3:.1f} GB available)")

class EnsembleTestPipeline:
    """
    Comprehensive ensemble testing pipeline for cluster-based SAPFLUXNET models
    """
    
    def __init__(self, 
                 site_split_file='../../site_split_assignment.json',
                 cluster_csv=None,
                 models_dir='./results/cluster_models',
                 parquet_dir='../../processed_parquet',
                 results_dir='./results/ensemble_validation',
                 force_gpu=False,
                 outlier_threshold=0.3):
        """
        Initialize ensemble testing pipeline
        
        Args:
            site_split_file: Path to site split JSON file
            cluster_csv: Path to cluster assignments CSV
            models_dir: Directory containing trained cluster models
            parquet_dir: Directory containing parquet data files
            results_dir: Directory to save ensemble results
            force_gpu: Force GPU usage even if not detected
            outlier_threshold: Threshold for identifying outlier sites (distance-based)
        """
        self.site_split_file = site_split_file
        self.cluster_csv = cluster_csv
        self.models_dir = models_dir
        self.parquet_dir = parquet_dir
        self.results_dir = results_dir
        self.force_gpu = force_gpu
        self.outlier_threshold = outlier_threshold
        
        # Data containers
        self.train_sites = []
        self.test_sites = []
        self.cluster_assignments = {}
        self.cluster_models = {}
        self.cluster_performance = {}
        self.cluster_centroids = {}
        self.outlier_sites = set()
        
        # Target column
        self.target_col = 'sap_flow'
        
        # Initialize
        self._detect_gpu()
        self._configure_memory_settings()
        self._load_site_split()
        self._load_cluster_assignments()
        self._create_results_dir()
        
        print(f"🚀 ENSEMBLE TESTING PIPELINE INITIALIZED")
        print(f"=" * 60)
        print(f"🎮 GPU enabled: {self.use_gpu}")
        print(f"📊 Test sites: {len(self.test_sites)}")
        print(f"📊 Clusters: {len(set(self.cluster_assignments.values())) if self.cluster_assignments else 0}")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🎯 Outlier threshold: {self.outlier_threshold}")

    def _detect_gpu(self):
        """Detect and configure GPU usage"""
        self.use_gpu = False
        self.gpu_id = 0
        
        if self.force_gpu:
            print("🎮 GPU usage forced by user")
            self.use_gpu = True
            return
        
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("🔍 No NVIDIA GPU detected (nvidia-smi failed)")
                return
            
            # Test XGBoost GPU functionality
            print("🔍 Testing XGBoost GPU functionality...")
            X_test = np.random.random((100, 10))
            y_test = np.random.random(100)
            
            test_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'objective': 'reg:squarederror',
                'n_estimators': 10,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**test_params)
            model.fit(X_test, y_test)
            
            self.use_gpu = True
            print("✅ GPU detection successful - XGBoost GPU acceleration enabled")
            
        except Exception as e:
            print(f"❌ GPU detection failed: {e}")
            print("🔄 Falling back to CPU processing")

    def _configure_memory_settings(self):
        """Configure memory settings based on available RAM"""
        available_gb = get_available_memory_gb()
        
        if available_gb > 32:
            self.chunk_size = 100000
            self.streaming_threshold_mb = 2000
            print(f"💾 Ultra-High memory mode: {available_gb:.1f} GB available")
        elif available_gb > 16:
            self.chunk_size = 50000
            self.streaming_threshold_mb = 1000
            print(f"💾 High memory mode: {available_gb:.1f} GB available")
        elif available_gb > 8:
            self.chunk_size = 25000
            self.streaming_threshold_mb = 500
            print(f"💾 Medium memory mode: {available_gb:.1f} GB available")
        else:
            self.chunk_size = 10000
            self.streaming_threshold_mb = 200
            print(f"💾 Low memory mode: {available_gb:.1f} GB available")

    def _load_site_split(self):
        """Load train/test site split from JSON file"""
        if not os.path.exists(self.site_split_file):
            raise FileNotFoundError(f"Site split file not found: {self.site_split_file}")
        
        print(f"📂 Loading site split from: {self.site_split_file}")
        
        with open(self.site_split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_sites = split_data['train_sites']
        self.test_sites = split_data['test_sites']
        
        print(f"✅ Loaded site split:")
        print(f"  📊 Train sites: {len(self.train_sites)}")
        print(f"  📊 Test sites: {len(self.test_sites)}")
        print(f"  📅 Split created: {split_data['metadata']['timestamp']}")

    def _load_cluster_assignments(self):
        """Load cluster assignments from CSV file"""
        if not self.cluster_csv:
            # Auto-detect latest cluster file
            cluster_files = [f for f in os.listdir('../evaluation/clustering_results') 
                           if f.startswith('advanced_site_clusters_') and f.endswith('.csv')]
            if not cluster_files:
                raise FileNotFoundError("No cluster assignment files found")
            
            # Use the most recent file
            cluster_files.sort(reverse=True)
            self.cluster_csv = os.path.join('../evaluation/clustering_results', cluster_files[0])
        
        print(f"📂 Loading cluster assignments from: {self.cluster_csv}")
        
        df = pd.read_csv(self.cluster_csv)
        
        # Only load assignments for training sites (test sites weren't clustered)
        for _, row in df.iterrows():
            if row['site'] in self.train_sites:
                self.cluster_assignments[row['site']] = row['cluster']
        
        # Group sites by cluster for centroids calculation
        clusters = {}
        for site, cluster_id in self.cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(site)
        
        print(f"✅ Loaded cluster assignments:")
        for cluster_id in sorted(clusters.keys()):
            print(f"  📊 Cluster {cluster_id}: {len(clusters[cluster_id])} sites")

    def _create_results_dir(self):
        """Create results directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{self.results_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Results directory created: {self.results_dir}")

    def load_trained_models(self):
        """Load all trained cluster models and their performance metrics"""
        print(f"\n🔧 Loading trained cluster models...")
        
        # Find all cluster model files
        model_files = []
        for file in os.listdir(self.models_dir):
            if file.startswith('cluster_') and file.endswith('_model.pkl'):
                cluster_id = int(file.split('_')[1])
                model_files.append((cluster_id, file))
        
        model_files.sort()  # Sort by cluster ID
        
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {self.models_dir}")
        
        # Load models and performance metrics
        for cluster_id, model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            
            try:
                # Load model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.cluster_models[cluster_id] = model
                
                # Load performance metrics
                metrics_file = os.path.join(self.models_dir, f'cluster_{cluster_id}_metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    self.cluster_performance[cluster_id] = metrics
                else:
                    print(f"  ⚠️  No metrics file found for cluster {cluster_id}")
                    self.cluster_performance[cluster_id] = {'test_r2': 0.0}
                
                print(f"  ✅ Cluster {cluster_id}: Model loaded (R²={self.cluster_performance[cluster_id].get('test_r2', 0.0):.3f})")
                
            except Exception as e:
                print(f"  ❌ Failed to load cluster {cluster_id} model: {e}")
                continue
        
        print(f"✅ Loaded {len(self.cluster_models)} cluster models")

    def calculate_cluster_centroids(self):
        """Calculate cluster centroids from training data for distance-based weighting"""
        print(f"\n🎯 Calculating cluster centroids...")
        
        # Features to use for centroid calculation (ecological features)
        # Try to use all features, but handle missing ones gracefully
        centroid_features = [
            'longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
            'seasonal_temp_range', 'seasonal_precip_range'
        ]
        
        for cluster_id in self.cluster_models.keys():
            cluster_sites = [site for site, cid in self.cluster_assignments.items() if cid == cluster_id]
            
            if not cluster_sites:
                continue
            
            # Load data for cluster sites
            cluster_data = []
            for site in cluster_sites:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                if os.path.exists(parquet_file):
                    try:
                        df = pd.read_parquet(parquet_file)
                        if len(df) > 0:
                            # Use mean values for this site, fill missing features with 0
                            site_means = []
                            for feature in centroid_features:
                                if feature in df.columns:
                                    site_means.append(df[feature].mean())
                                else:
                                    site_means.append(0.0)  # Default for missing features
                            cluster_data.append(site_means)
                    except Exception as e:
                        print(f"    ❌ Error loading {site}: {e}")
                        continue
            
            if cluster_data:
                # Calculate centroid as mean of all site means
                centroid = np.mean(cluster_data, axis=0)
                self.cluster_centroids[cluster_id] = centroid
                print(f"    ✅ Cluster {cluster_id}: {len(cluster_sites)} sites, {len(centroid_features)} features")
            else:
                print(f"    ❌ Cluster {cluster_id}: No valid data for centroid calculation")
        
        print(f"✅ Calculated {len(self.cluster_centroids)} cluster centroids")

    def load_test_site_data(self, site):
        """Load and prepare data for a test site"""
        parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
        
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Parquet file not found for site {site}")
        
        # Load full dataset
        df = pd.read_parquet(parquet_file)
        
        # Remove rows with missing target values
        df = df.dropna(subset=[self.target_col])
        
        if len(df) == 0:
            raise ValueError(f"No valid data found for site {site}")
        
        # Prepare features - MATCH TRAINING PIPELINE EXACTLY
        exclude_cols = [
            self.target_col, 'site', 'TIMESTAMP', 'solar_TIMESTAMP', 
            'plant_id', 'Unnamed: 0'
        ]
        
        # Also remove any columns ending with specific suffixes (like training pipeline)
        exclude_suffixes = ['_flags', '_md']
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if any(col.endswith(suffix) for suffix in exclude_suffixes):
                continue
            feature_cols.append(col)
        
        X = df[feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Handle data types and NaN values - MATCH TRAINING PIPELINE EXACTLY
        print(f"  🔍 Before cleaning: {X.isnull().sum().sum()} NaN values")
        
        # Convert boolean columns to numeric (True=1, False=0)
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            elif X[col].dtype == 'object':
                # Try to convert object columns to numeric, fill non-numeric with 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with 0 (EXACTLY like training pipeline)
        X = X.fillna(0)
        
        # Check for any remaining issues
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        print(f"  🔍 After cleaning: {nan_count} NaN values, {inf_count} infinite values")
        
        if nan_count > 0:
            print(f"  ❌ Columns with NaN: {X.columns[X.isnull().any()].tolist()}")
        if inf_count > 0:
            print(f"  ❌ Columns with inf: {X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()}")
            # Replace infinite values with 0
            X = X.replace([np.inf, -np.inf], 0)
        
        return X, y, feature_cols

    def calculate_site_cluster_distances(self, site, X):
        """Calculate distances from test site to all cluster centroids"""
        if not self.cluster_centroids:
            return {}
        
        # Features for distance calculation (matching centroid features)
        # Try to use all features, but handle missing ones gracefully
        distance_features = [
            'longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip',
            'seasonal_temp_range', 'seasonal_precip_range'
        ]
        
        # Get site features (use mean values, fill missing with 0)
        site_features = []
        for feature in distance_features:
            if feature in X.columns:
                site_features.append(X[feature].mean())
            else:
                site_features.append(0.0)  # Default value for missing features
        
        site_features = np.array(site_features).reshape(1, -1)
        
        # Calculate distances to all centroids
        distances = {}
        for cluster_id, centroid in self.cluster_centroids.items():
            centroid = np.array(centroid).reshape(1, -1)
            distance = euclidean_distances(site_features, centroid)[0][0]
            distances[cluster_id] = distance
        
        return distances

    def predict_with_cluster_model(self, cluster_id, X):
        """Make predictions using a specific cluster model"""
        if cluster_id not in self.cluster_models:
            raise ValueError(f"No model found for cluster {cluster_id}")
        
        model = self.cluster_models[cluster_id]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate prediction confidence (inverse of prediction variance)
        # For XGBoost, we can use the standard deviation of tree predictions as uncertainty
        try:
            # Get predictions from individual trees
            pred_contribs = model.predict(X, pred_contribs=True)
            pred_variance = np.var(pred_contribs, axis=1)
            confidence = 1.0 / (1.0 + pred_variance)  # Higher variance = lower confidence
        except:
            # Fallback: use constant confidence
            confidence = np.ones(len(y_pred)) * 0.5
        
        return y_pred, confidence

    def calculate_ensemble_weights(self, site, distances):
        """Calculate ensemble weights using different strategies"""
        weights = {}
        
        # Strategy 1: Equal weighting (baseline)
        equal_weights = {cid: 1.0 / len(self.cluster_models) for cid in self.cluster_models.keys()}
        weights['equal'] = equal_weights
        
        # Strategy 2: Historical performance weighting
        total_performance = sum(max(0.0, perf.get('test_r2', 0.0)) for perf in self.cluster_performance.values())
        if total_performance > 0:
            perf_weights = {
                cid: max(0.0, self.cluster_performance[cid].get('test_r2', 0.0)) / total_performance
                for cid in self.cluster_models.keys()
            }
        else:
            perf_weights = equal_weights.copy()
        weights['performance'] = perf_weights
        
        # Strategy 3: Distance-based weighting (inverse distance)
        if distances:
            # Convert distances to weights (closer = higher weight)
            max_distance = max(distances.values())
            inv_distances = {cid: max_distance - dist + 0.001 for cid, dist in distances.items()}
            total_inv_distance = sum(inv_distances.values())
            
            if total_inv_distance > 0:
                distance_weights = {cid: inv_dist / total_inv_distance for cid, inv_dist in inv_distances.items()}
            else:
                distance_weights = equal_weights.copy()
        else:
            distance_weights = equal_weights.copy()
        weights['distance'] = distance_weights
        
        # Strategy 4: Confidence-weighted (will be calculated during prediction)
        weights['confidence'] = {}  # Will be filled during prediction
        
        return weights

    def test_site_with_ensemble(self, site):
        """Test a single site using all ensemble strategies"""
        print(f"\n🧪 Testing site: {site}")
        
        try:
            # Load site data
            X, y_true, feature_cols = self.load_test_site_data(site)
            print(f"  📊 Loaded {len(X)} samples with {len(feature_cols)} features")
            
            # Calculate distances to cluster centroids
            distances = self.calculate_site_cluster_distances(site, X)
            
            # Check if site is an outlier
            if distances:
                min_distance = min(distances.values())
                is_outlier = min_distance > self.outlier_threshold
                if is_outlier:
                    self.outlier_sites.add(site)
                    print(f"  🚨 OUTLIER DETECTED: Min distance {min_distance:.3f} > threshold {self.outlier_threshold}")
            else:
                is_outlier = False
            
            # Calculate ensemble weights
            ensemble_weights = self.calculate_ensemble_weights(site, distances)
            
            # Make predictions with each cluster model
            cluster_predictions = {}
            cluster_confidences = {}
            
            for cluster_id in self.cluster_models.keys():
                try:
                    y_pred, confidence = self.predict_with_cluster_model(cluster_id, X)
                    cluster_predictions[cluster_id] = y_pred
                    cluster_confidences[cluster_id] = confidence
                    
                    # Calculate individual cluster metrics
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    print(f"    📈 Cluster {cluster_id}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
                    
                except Exception as e:
                    print(f"    ❌ Cluster {cluster_id} prediction failed: {e}")
                    continue
            
            if not cluster_predictions:
                raise ValueError("No successful cluster predictions")
            
            # Calculate confidence-based weights
            if cluster_confidences:
                total_confidence = sum(np.mean(conf) for conf in cluster_confidences.values())
                if total_confidence > 0:
                    ensemble_weights['confidence'] = {
                        cid: np.mean(conf) / total_confidence 
                        for cid, conf in cluster_confidences.items()
                    }
                else:
                    ensemble_weights['confidence'] = ensemble_weights['equal'].copy()
            else:
                ensemble_weights['confidence'] = ensemble_weights['equal'].copy()
            
            # Calculate ensemble predictions
            ensemble_results = {}
            
            for strategy, weights in ensemble_weights.items():
                if not weights:
                    continue
                
                # Weighted average of predictions
                ensemble_pred = np.zeros(len(y_true))
                total_weight = 0
                
                for cluster_id, weight in weights.items():
                    if cluster_id in cluster_predictions:
                        ensemble_pred += weight * cluster_predictions[cluster_id]
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred /= total_weight
                
                # Calculate ensemble metrics
                r2 = r2_score(y_true, ensemble_pred)
                rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
                mae = mean_absolute_error(y_true, ensemble_pred)
                
                ensemble_results[strategy] = {
                    'predictions': ensemble_pred,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'weights': weights,
                    'n_samples': len(y_true)
                }
                
                print(f"  🎯 {strategy.capitalize()}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            
            return {
                'site': site,
                'is_outlier': is_outlier,
                'distances': distances,
                'y_true': y_true,
                'cluster_predictions': cluster_predictions,
                'cluster_confidences': cluster_confidences,
                'ensemble_results': ensemble_results,
                'n_samples': len(y_true),
                'feature_count': len(feature_cols)
            }
            
        except Exception as e:
            print(f"  ❌ Testing failed for site {site}: {e}")
            return None

    def run_ensemble_testing(self):
        """Run ensemble testing on all test sites"""
        print(f"\n🚀 STARTING ENSEMBLE TESTING")
        print(f"=" * 60)
        
        # Load models and calculate centroids
        self.load_trained_models()
        self.calculate_cluster_centroids()
        
        # Test all sites
        all_results = []
        failed_sites = []
        
        # Process regular sites first, outliers last
        regular_sites = []
        potential_outliers = []
        
        # Pre-screen sites for potential outliers (quick distance check)
        print(f"\n🔍 Pre-screening sites for outliers...")
        for site in self.test_sites:
            try:
                parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
                if os.path.exists(parquet_file):
                    # Quick load for distance calculation
                    df = pd.read_parquet(parquet_file, columns=[
                        'longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip'
                    ])
                    
                    if len(df) > 0:
                        distances = self.calculate_site_cluster_distances(site, df)
                        if distances and min(distances.values()) > self.outlier_threshold:
                            potential_outliers.append(site)
                        else:
                            regular_sites.append(site)
                    else:
                        regular_sites.append(site)  # Default to regular
                else:
                    failed_sites.append(site)
            except:
                regular_sites.append(site)  # Default to regular if error
        
        print(f"  📊 Regular sites: {len(regular_sites)}")
        print(f"  🚨 Potential outliers: {len(potential_outliers)}")
        
        # Process regular sites first
        print(f"\n📊 TESTING REGULAR SITES")
        print(f"=" * 40)
        
        for i, site in enumerate(regular_sites, 1):
            print(f"\n[{i}/{len(regular_sites)}] Processing regular site: {site}")
            log_memory_usage(f"Before site {site}")
            
            result = self.test_site_with_ensemble(site)
            if result:
                all_results.append(result)
            else:
                failed_sites.append(site)
            
            # Memory cleanup
            gc.collect()
        
        # Process outlier sites last
        if potential_outliers:
            print(f"\n🚨 TESTING POTENTIAL OUTLIER SITES")
            print(f"=" * 40)
            
            for i, site in enumerate(potential_outliers, 1):
                print(f"\n[{i}/{len(potential_outliers)}] Processing outlier site: {site}")
                log_memory_usage(f"Before outlier site {site}")
                
                result = self.test_site_with_ensemble(site)
                if result:
                    all_results.append(result)
                else:
                    failed_sites.append(site)
                
                # Memory cleanup
                gc.collect()
        
        print(f"\n✅ ENSEMBLE TESTING COMPLETED")
        print(f"  📊 Successful sites: {len(all_results)}")
        print(f"  ❌ Failed sites: {len(failed_sites)}")
        print(f"  🚨 Outlier sites: {len(self.outlier_sites)}")
        
        if failed_sites:
            print(f"  ❌ Failed sites: {', '.join(failed_sites)}")
        
        return all_results

    def calculate_comprehensive_metrics(self, results):
        """Calculate comprehensive metrics including and excluding outliers"""
        print(f"\n📊 CALCULATING COMPREHENSIVE METRICS")
        print(f"=" * 50)
        
        # Separate regular and outlier results
        regular_results = [r for r in results if not r['is_outlier']]
        outlier_results = [r for r in results if r['is_outlier']]
        
        print(f"📊 Regular sites: {len(regular_results)}")
        print(f"🚨 Outlier sites: {len(outlier_results)}")
        
        metrics_summary = {
            'total_sites': len(results),
            'regular_sites': len(regular_results),
            'outlier_sites': len(outlier_results),
            'outlier_site_names': [r['site'] for r in outlier_results],
            'strategies': {}
        }
        
        # Calculate metrics for each ensemble strategy
        if results:
            strategies = list(results[0]['ensemble_results'].keys())
            
            for strategy in strategies:
                print(f"\n🎯 Strategy: {strategy.upper()}")
                
                # Metrics including all sites
                all_r2 = []
                all_rmse = []
                all_mae = []
                all_samples = 0
                
                for result in results:
                    if strategy in result['ensemble_results']:
                        ensemble_result = result['ensemble_results'][strategy]
                        all_r2.append(ensemble_result['r2'])
                        all_rmse.append(ensemble_result['rmse'])
                        all_mae.append(ensemble_result['mae'])
                        all_samples += ensemble_result['n_samples']
                
                # Metrics excluding outliers
                regular_r2 = []
                regular_rmse = []
                regular_mae = []
                regular_samples = 0
                
                for result in regular_results:
                    if strategy in result['ensemble_results']:
                        ensemble_result = result['ensemble_results'][strategy]
                        regular_r2.append(ensemble_result['r2'])
                        regular_rmse.append(ensemble_result['rmse'])
                        regular_mae.append(ensemble_result['mae'])
                        regular_samples += ensemble_result['n_samples']
                
                # Store metrics
                strategy_metrics = {
                    'all_sites': {
                        'mean_r2': np.mean(all_r2) if all_r2 else 0,
                        'std_r2': np.std(all_r2) if all_r2 else 0,
                        'median_r2': np.median(all_r2) if all_r2 else 0,
                        'mean_rmse': np.mean(all_rmse) if all_rmse else 0,
                        'std_rmse': np.std(all_rmse) if all_rmse else 0,
                        'median_rmse': np.median(all_rmse) if all_rmse else 0,
                        'mean_mae': np.mean(all_mae) if all_mae else 0,
                        'std_mae': np.std(all_mae) if all_mae else 0,
                        'median_mae': np.median(all_mae) if all_mae else 0,
                        'total_samples': all_samples,
                        'n_sites': len(all_r2)
                    },
                    'regular_sites_only': {
                        'mean_r2': np.mean(regular_r2) if regular_r2 else 0,
                        'std_r2': np.std(regular_r2) if regular_r2 else 0,
                        'median_r2': np.median(regular_r2) if regular_r2 else 0,
                        'mean_rmse': np.mean(regular_rmse) if regular_rmse else 0,
                        'std_rmse': np.std(regular_rmse) if regular_rmse else 0,
                        'median_rmse': np.median(regular_rmse) if regular_rmse else 0,
                        'mean_mae': np.mean(regular_mae) if regular_mae else 0,
                        'std_mae': np.std(regular_mae) if regular_mae else 0,
                        'median_mae': np.median(regular_mae) if regular_mae else 0,
                        'total_samples': regular_samples,
                        'n_sites': len(regular_r2)
                    }
                }
                
                metrics_summary['strategies'][strategy] = strategy_metrics
                
                # Print summary
                print(f"  📊 All sites ({len(all_r2)}):")
                print(f"    R²: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f} (median: {np.median(all_r2):.3f})")
                print(f"    RMSE: {np.mean(all_rmse):.3f} ± {np.std(all_rmse):.3f}")
                print(f"    MAE: {np.mean(all_mae):.3f} ± {np.std(all_mae):.3f}")
                
                if regular_r2:
                    print(f"  📊 Regular sites only ({len(regular_r2)}):")
                    print(f"    R²: {np.mean(regular_r2):.3f} ± {np.std(regular_r2):.3f} (median: {np.median(regular_r2):.3f})")
                    print(f"    RMSE: {np.mean(regular_rmse):.3f} ± {np.std(regular_rmse):.3f}")
                    print(f"    MAE: {np.mean(regular_mae):.3f} ± {np.std(regular_mae):.3f}")
        
        return metrics_summary

    def save_results(self, results, metrics_summary):
        """Save comprehensive ensemble testing results"""
        print(f"\n💾 SAVING RESULTS")
        print(f"=" * 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_results = []
        for result in results:
            site_result = {
                'site': result['site'],
                'is_outlier': result['is_outlier'],
                'n_samples': result['n_samples'],
                'feature_count': result['feature_count'],
                'distances_to_clusters': result['distances']
            }
            
            # Add ensemble results
            for strategy, ensemble_result in result['ensemble_results'].items():
                site_result[f'{strategy}_r2'] = ensemble_result['r2']
                site_result[f'{strategy}_rmse'] = ensemble_result['rmse']
                site_result[f'{strategy}_mae'] = ensemble_result['mae']
                site_result[f'{strategy}_weights'] = ensemble_result['weights']
            
            detailed_results.append(site_result)
        
        # Save to CSV
        results_df = pd.DataFrame(detailed_results)
        results_file = os.path.join(self.results_dir, f'ensemble_test_results_{timestamp}.csv')
        results_df.to_csv(results_file, index=False)
        print(f"  ✅ Detailed results: {results_file}")
        
        # Save metrics summary
        metrics_file = os.path.join(self.results_dir, f'ensemble_metrics_summary_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"  ✅ Metrics summary: {metrics_file}")
        
        # Save full results (including predictions)
        full_results_file = os.path.join(self.results_dir, f'ensemble_full_results_{timestamp}.pkl')
        with open(full_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"  ✅ Full results: {full_results_file}")
        
        # Create summary report
        self._create_summary_report(results, metrics_summary, timestamp)
        
        return results_file, metrics_file

    def _create_summary_report(self, results, metrics_summary, timestamp):
        """Create a human-readable summary report"""
        report_file = os.path.join(self.results_dir, f'ensemble_summary_report_{timestamp}.txt')
        
        with open(report_file, 'w') as f:
            f.write("SAPFLUXNET ENSEMBLE TESTING SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n")
            f.write(f"Outlier Threshold: {self.outlier_threshold}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total test sites: {metrics_summary['total_sites']}\n")
            f.write(f"Regular sites: {metrics_summary['regular_sites']}\n")
            f.write(f"Outlier sites: {metrics_summary['outlier_sites']}\n")
            
            if metrics_summary['outlier_site_names']:
                f.write(f"Outlier site names: {', '.join(metrics_summary['outlier_site_names'])}\n")
            f.write("\n")
            
            f.write("ENSEMBLE STRATEGY PERFORMANCE\n")
            f.write("-" * 35 + "\n")
            
            for strategy, metrics in metrics_summary['strategies'].items():
                f.write(f"\n{strategy.upper()} STRATEGY:\n")
                
                # All sites
                all_metrics = metrics['all_sites']
                f.write(f"  All Sites ({all_metrics['n_sites']} sites, {all_metrics['total_samples']:,} samples):\n")
                f.write(f"    R²: {all_metrics['mean_r2']:.3f} ± {all_metrics['std_r2']:.3f} (median: {all_metrics['median_r2']:.3f})\n")
                f.write(f"    RMSE: {all_metrics['mean_rmse']:.3f} ± {all_metrics['std_rmse']:.3f}\n")
                f.write(f"    MAE: {all_metrics['mean_mae']:.3f} ± {all_metrics['std_mae']:.3f}\n")
                
                # Regular sites only
                regular_metrics = metrics['regular_sites_only']
                if regular_metrics['n_sites'] > 0:
                    f.write(f"  Regular Sites Only ({regular_metrics['n_sites']} sites, {regular_metrics['total_samples']:,} samples):\n")
                    f.write(f"    R²: {regular_metrics['mean_r2']:.3f} ± {regular_metrics['std_r2']:.3f} (median: {regular_metrics['median_r2']:.3f})\n")
                    f.write(f"    RMSE: {regular_metrics['mean_rmse']:.3f} ± {regular_metrics['std_rmse']:.3f}\n")
                    f.write(f"    MAE: {regular_metrics['mean_mae']:.3f} ± {regular_metrics['std_mae']:.3f}\n")
            
            f.write("\nCLUSTER MODEL INFORMATION\n")
            f.write("-" * 30 + "\n")
            for cluster_id, performance in self.cluster_performance.items():
                f.write(f"Cluster {cluster_id}: R² = {performance.get('test_r2', 0.0):.3f}\n")
            
            f.write(f"\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Find best strategy
            best_strategy = None
            best_r2 = -999
            
            for strategy, metrics in metrics_summary['strategies'].items():
                regular_r2 = metrics['regular_sites_only']['mean_r2']
                if regular_r2 > best_r2:
                    best_r2 = regular_r2
                    best_strategy = strategy
            
            if best_strategy:
                f.write(f"Best performing strategy: {best_strategy.upper()} (R² = {best_r2:.3f} on regular sites)\n")
            
            if metrics_summary['outlier_sites'] > 0:
                f.write(f"Consider additional analysis for {metrics_summary['outlier_sites']} outlier sites\n")
                f.write("Outlier sites may benefit from specialized models or different clustering approaches\n")
        
        print(f"  ✅ Summary report: {report_file}")

def main():
    """Main function for ensemble testing pipeline"""
    parser = argparse.ArgumentParser(description='Ensemble Testing Pipeline for SAPFLUXNET Spatial Validation')
    
    parser.add_argument('--site-split-file', type=str, default='../../site_split_assignment.json',
                        help='Path to site split JSON file')
    parser.add_argument('--cluster-csv', type=str, default=None,
                        help='Path to cluster assignments CSV (auto-detect if not provided)')
    parser.add_argument('--models-dir', type=str, default='./results/cluster_models',
                        help='Directory containing trained cluster models')
    parser.add_argument('--parquet-dir', type=str, default='../../processed_parquet',
                        help='Directory containing parquet data files')
    parser.add_argument('--results-dir', type=str, default='./results/ensemble_validation',
                        help='Directory to save ensemble results')
    parser.add_argument('--force-gpu', action='store_true',
                        help='Force GPU usage even if not detected')
    parser.add_argument('--outlier-threshold', type=float, default=0.3,
                        help='Threshold for identifying outlier sites (distance-based)')
    
    args = parser.parse_args()
    
    print("🚀 SAPFLUXNET ENSEMBLE TESTING PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Initialize pipeline
        pipeline = EnsembleTestPipeline(
            site_split_file=args.site_split_file,
            cluster_csv=args.cluster_csv,
            models_dir=args.models_dir,
            parquet_dir=args.parquet_dir,
            results_dir=args.results_dir,
            force_gpu=args.force_gpu,
            outlier_threshold=args.outlier_threshold
        )
        
        # Run ensemble testing
        results = pipeline.run_ensemble_testing()
        
        if results:
            # Calculate comprehensive metrics
            metrics_summary = pipeline.calculate_comprehensive_metrics(results)
            
            # Save results
            results_file, metrics_file = pipeline.save_results(results, metrics_summary)
            
            print(f"\n🎉 ENSEMBLE TESTING COMPLETED SUCCESSFULLY!")
            print(f"📁 Results saved to: {pipeline.results_dir}")
            print(f"📊 Tested {len(results)} sites")
            print(f"🚨 Identified {len(pipeline.outlier_sites)} outlier sites")
            
            # Print best strategy
            best_strategy = None
            best_r2 = -999
            for strategy, metrics in metrics_summary['strategies'].items():
                regular_r2 = metrics['regular_sites_only']['mean_r2']
                if regular_r2 > best_r2:
                    best_r2 = regular_r2
                    best_strategy = strategy
            
            if best_strategy:
                print(f"🏆 Best strategy: {best_strategy.upper()} (R² = {best_r2:.3f} on regular sites)")
        
        else:
            print(f"\n❌ No successful ensemble tests completed")
            return 1
        
    except Exception as e:
        print(f"\n❌ Ensemble testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nFinished at: {datetime.now()}")
    return 0

if __name__ == "__main__":
    main()
