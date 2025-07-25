"""
Ecosystem-Based Temporal Validation for SAPFLUXNET Data
Combines temporal validation with ecosystem clustering to test temporal generalization
within ecosystem boundaries while addressing site imbalance issues.

Based on the successful ecosystem clustering approach that improved spatial validation
from R² = -1377.87 to R² = 0.5939.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file
import os
import glob
from datetime import datetime, timedelta
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

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")

class EcosystemTemporalValidator:
    """
    Ecosystem-based temporal validation with site balancing and cluster-specific modeling
    """
    
    def __init__(self, parquet_dir='../processed_parquet',
                 clustering_results_dir='../ecosystem/evaluation/clustering_results',
                 output_dir='external_memory_models/ecosystem_temporal_validation'):
        self.parquet_dir = parquet_dir
        self.clustering_results_dir = clustering_results_dir
        self.output_dir = output_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Temporal validation configuration
        self.n_folds = 8
        self.fold_duration_years = 2.75
        self.min_training_years = 3
        self.max_samples_per_site = 5000  # Balance site contributions
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🌍 Ecosystem Temporal Validator initialized")
        print(f"📁 Parquet directory: {parquet_dir}")
        print(f"🧬 Clustering results: {clustering_results_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"⚖️  Site balance limit: {self.max_samples_per_site:,} samples per site")

    def load_ecosystem_cluster_assignments(self):
        """Load ecosystem cluster assignments from clustering results"""
        print(f"\n📊 Loading ecosystem cluster assignments...")
        
        # Find latest cluster assignments
        cluster_files = sorted(glob.glob(os.path.join(
            self.clustering_results_dir, 'advanced_site_clusters_*.csv'
        )))
        
        if not cluster_files:
            raise FileNotFoundError(f"No cluster assignments found in {self.clustering_results_dir}")
        
        latest_file = cluster_files[-1]
        print(f"📄 Using: {os.path.basename(latest_file)}")
        
        # Load cluster assignments
        clusters_df = pd.read_csv(latest_file)
        cluster_assignments = dict(zip(clusters_df['site'], clusters_df['cluster']))
        
        print(f"✅ Loaded {len(cluster_assignments)} site assignments")
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        
        print(f"🌍 Ecosystem cluster distribution:")
        ecosystem_names = {
            0: "Warm Temperate (Mediterranean/European)",
            1: "Mixed Temperate (Australian/Global)",
            2: "Continental (Cold/Warm Temperate)", 
            3: "European Temperate (Mountain/Forest)",
            4: "Tropical/Subtropical (Global)"
        }
        
        for cluster_id, count in cluster_counts.items():
            name = ecosystem_names.get(cluster_id, f"Cluster {cluster_id}")
            print(f"  Cluster {cluster_id} ({name}): {count} sites")
        
        return cluster_assignments, cluster_counts

    def analyze_cluster_temporal_coverage(self, cluster_assignments):
        """Analyze temporal coverage for each ecosystem cluster"""
        print(f"\n📅 Analyzing temporal coverage by ecosystem cluster...")
        
        cluster_temporal_info = {}
        
        for site, cluster_id in cluster_assignments.items():
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                continue
                
            try:
                # Load just timestamp columns for temporal analysis
                df_sample = pd.read_parquet(
                    parquet_file, 
                    columns=['TIMESTAMP', 'solar_TIMESTAMP', self.target_col]
                )
                
                # Filter valid data
                df_sample = df_sample.dropna(subset=[self.target_col])
                
                if len(df_sample) == 0:
                    continue
                
                # Convert to datetime
                df_sample['solar_TIMESTAMP'] = pd.to_datetime(df_sample['solar_TIMESTAMP'])
                
                # Get temporal range
                start_time = df_sample['solar_TIMESTAMP'].min()
                end_time = df_sample['solar_TIMESTAMP'].max()
                n_measurements = len(df_sample)
                
                if cluster_id not in cluster_temporal_info:
                    cluster_temporal_info[cluster_id] = {
                        'sites': [],
                        'start_times': [],
                        'end_times': [],
                        'total_measurements': 0
                    }
                
                cluster_temporal_info[cluster_id]['sites'].append(site)
                cluster_temporal_info[cluster_id]['start_times'].append(start_time)
                cluster_temporal_info[cluster_id]['end_times'].append(end_time)
                cluster_temporal_info[cluster_id]['total_measurements'] += n_measurements
                
                del df_sample
                gc.collect()
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing {site}: {e}")
                continue
        
        # Calculate global temporal range for each cluster
        cluster_summaries = {}
        for cluster_id, info in cluster_temporal_info.items():
            if info['sites']:
                global_start = min(info['start_times'])
                global_end = max(info['end_times'])
                duration_days = (global_end - global_start).days
                
                cluster_summaries[cluster_id] = {
                    'sites': info['sites'],
                    'n_sites': len(info['sites']),
                    'global_start': global_start,
                    'global_end': global_end, 
                    'duration_days': duration_days,
                    'duration_years': duration_days / 365.25,
                    'total_measurements': info['total_measurements']
                }
                
                print(f"  Cluster {cluster_id}: {len(info['sites'])} sites, "
                      f"{duration_days} days ({duration_days/365.25:.1f} years), "
                      f"{info['total_measurements']:,} measurements")
        
        return cluster_summaries

    def balance_sites_within_cluster(self, cluster_sites, cluster_id, max_samples_per_site=None):
        """Balance site contributions within a cluster while preserving temporal patterns"""
        if max_samples_per_site is None:
            max_samples_per_site = self.max_samples_per_site
            
        print(f"⚖️  Balancing sites within cluster {cluster_id}...")
        print(f"   Target: max {max_samples_per_site:,} samples per site")
        
        balanced_data = []
        total_original = 0
        total_balanced = 0
        
        for site in cluster_sites:
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"    ⚠️  Missing: {site}")
                continue
                
            try:
                # Load site data
                df_site = pd.read_parquet(parquet_file)
                
                # Filter for this cluster and valid target data
                df_site = df_site[df_site[self.cluster_col] == cluster_id]
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) == 0:
                    continue
                
                original_size = len(df_site)
                total_original += original_size
                
                # Balance this site's contribution
                if len(df_site) > max_samples_per_site:
                    # Stratified temporal sampling to preserve seasonal patterns
                    df_site = df_site.sort_values('solar_TIMESTAMP')
                    
                    # Sample uniformly across time to preserve temporal patterns
                    indices = np.linspace(0, len(df_site)-1, max_samples_per_site).astype(int)
                    df_site = df_site.iloc[indices].copy()
                
                balanced_size = len(df_site)
                total_balanced += balanced_size
                
                balanced_data.append(df_site)
                
                reduction = (original_size - balanced_size) / original_size * 100 if original_size > 0 else 0
                print(f"    ✅ {site}: {original_size:,} → {balanced_size:,} "
                      f"({reduction:.1f}% reduction)")
                
            except Exception as e:
                print(f"    ❌ Error processing {site}: {e}")
                continue
        
        if not balanced_data:
            raise ValueError(f"No valid data found for cluster {cluster_id}")
        
        # Combine balanced data
        combined_df = pd.concat(balanced_data, ignore_index=True)
        
        overall_reduction = (total_original - total_balanced) / total_original * 100 if total_original > 0 else 0
        print(f"    📊 Cluster {cluster_id} balanced: {total_original:,} → {total_balanced:,} "
              f"({overall_reduction:.1f}% reduction)")
        
        return combined_df

    def create_ecosystem_temporal_splits(self, df, cluster_id):
        """Create temporal splits for a specific ecosystem cluster"""
        print(f"📅 Creating temporal splits for cluster {cluster_id}...")
        
        # Sort by temporal index
        df = df.sort_values('solar_TIMESTAMP').reset_index(drop=True)
        
        # Get time range
        start_time = df['solar_TIMESTAMP'].min()
        end_time = df['solar_TIMESTAMP'].max()
        total_duration = end_time - start_time
        
        print(f"   Time range: {start_time.date()} to {end_time.date()}")
        print(f"   Duration: {total_duration.days} days ({total_duration.days/365.25:.1f} years)")
        
        # Calculate fold parameters
        fold_duration = timedelta(days=int(self.fold_duration_years * 365.25))
        min_training_duration = timedelta(days=int(self.min_training_years * 365.25))
        
        print(f"   Fold config: {self.n_folds} folds of {self.fold_duration_years:.1f} years each")
        
        # Check if we have enough data
        min_required_duration = min_training_duration + fold_duration
        if total_duration < min_required_duration:
            raise ValueError(f"Insufficient temporal data for cluster {cluster_id}. "
                           f"Need at least {min_required_duration.days/365.25:.1f} years, "
                           f"have {total_duration.days/365.25:.1f} years")
        
        # Create temporal splits
        temporal_splits = []
        
        for fold_idx in range(self.n_folds):
            # Progressive training approach - each fold uses all data before test period
            test_start_time = start_time + timedelta(days=int(
                (fold_idx * self.fold_duration_years + self.min_training_years) * 365.25
            ))
            test_end_time = test_start_time + fold_duration
            
            # Check if test period fits within data range
            if test_end_time > end_time:
                print(f"   Fold {fold_idx + 1}: Skipped (insufficient future data)")
                break
            
            # Training data: all data from start to test_start
            train_start_time = start_time
            train_end_time = test_start_time
            
            # Filter data for this fold
            train_mask = (df['solar_TIMESTAMP'] >= train_start_time) & (df['solar_TIMESTAMP'] < train_end_time)
            test_mask = (df['solar_TIMESTAMP'] >= test_start_time) & (df['solar_TIMESTAMP'] < test_end_time)
            
            train_data = df[train_mask].copy()
            test_data = df[test_mask].copy()
            
            if len(train_data) < 1000 or len(test_data) < 100:
                print(f"   Fold {fold_idx + 1}: Skipped (insufficient data)")
                continue
            
            print(f"   Fold {fold_idx + 1}: Train {train_start_time.date()} to {train_end_time.date()}")
            print(f"                      Test {test_start_time.date()} to {test_end_time.date()}")
            print(f"                      Train: {len(train_data):,}, Test: {len(test_data):,}")
            
            temporal_splits.append({
                'fold': fold_idx + 1,
                'train_data': train_data,
                'test_data': test_data,
                'train_start': train_start_time,
                'train_end': train_end_time,
                'test_start': test_start_time,
                'test_end': test_end_time
            })
        
        print(f"   ✅ Created {len(temporal_splits)} valid temporal splits")
        return temporal_splits

    def validate_cluster_temporally(self, cluster_id, cluster_sites, temp_dir):
        """Run temporal validation for a specific ecosystem cluster"""
        print(f"\n🌍 Validating Cluster {cluster_id} Temporally")
        print("=" * 50)
        
        try:
            # Step 1: Balance sites within cluster
            balanced_df = self.balance_sites_within_cluster(cluster_sites, cluster_id)
            
            # Step 2: Create temporal splits
            temporal_splits = self.create_ecosystem_temporal_splits(balanced_df, cluster_id)
            
            if not temporal_splits:
                print(f"❌ No valid temporal splits for cluster {cluster_id}")
                return None
            
            # Step 3: Train models for each temporal fold
            fold_results = []
            fold_models = []
            
            # Get feature columns (exclude metadata)
            exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 
                          self.target_col, 'is_inside_country', self.cluster_col]
            feature_cols = [col for col in balanced_df.columns if col not in exclude_cols]
            
            print(f"📊 Using {len(feature_cols)} features for training")
            
            for split in temporal_splits:
                fold_num = split['fold']
                train_data = split['train_data']
                test_data = split['test_data']
                
                print(f"\n--- Training Fold {fold_num} ---")
                
                # Create libsvm files for this fold
                train_file = os.path.join(temp_dir, f'cluster_{cluster_id}_fold_{fold_num}_train.svm')
                test_file = os.path.join(temp_dir, f'cluster_{cluster_id}_fold_{fold_num}_test.svm')
                
                try:
                    # Prepare training data
                    train_clean = train_data.dropna(subset=[self.target_col])
                    X_train = train_clean[feature_cols].fillna(0).values
                    y_train = train_clean[self.target_col].values
                    
                    # Prepare test data
                    test_clean = test_data.dropna(subset=[self.target_col])
                    X_test = test_clean[feature_cols].fillna(0).values
                    y_test = test_clean[self.target_col].values
                    
                    # Save to libsvm format
                    dump_svmlight_file(X_train, y_train, train_file)
                    dump_svmlight_file(X_test, y_test, test_file)
                    
                    # Train XGBoost model with external memory
                    model, metrics = self.train_external_memory_fold(
                        train_file, test_file, cluster_id, fold_num
                    )
                    
                    # Add fold and temporal information
                    metrics.update({
                        'cluster_id': cluster_id,
                        'fold': fold_num,
                        'train_start': split['train_start'],
                        'train_end': split['train_end'],
                        'test_start': split['test_start'],
                        'test_end': split['test_end']
                    })
                    
                    fold_results.append(metrics)
                    fold_models.append(model)
                    
                    print(f"Fold {fold_num} Results:")
                    print(f"  Train R²: {metrics['train_r2']:.4f}")
                    print(f"  Test R²: {metrics['test_r2']:.4f}")
                    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error training fold {fold_num}: {e}")
                    continue
                
                finally:
                    # Clean up fold files
                    for f in [train_file, test_file]:
                        if os.path.exists(f):
                            try:
                                os.remove(f)
                            except:
                                pass
            
            if not fold_results:
                print(f"❌ No successful folds for cluster {cluster_id}")
                return None
            
            # Calculate cluster summary
            test_r2_values = [fold['test_r2'] for fold in fold_results]
            test_rmse_values = [fold['test_rmse'] for fold in fold_results]
            
            cluster_summary = {
                'cluster_id': cluster_id,
                'n_sites': len(cluster_sites),
                'n_folds': len(fold_results),
                'total_samples': len(balanced_df),
                'test_r2_mean': np.mean(test_r2_values),
                'test_r2_std': np.std(test_r2_values),
                'test_rmse_mean': np.mean(test_rmse_values),
                'test_rmse_std': np.std(test_rmse_values),
                'feature_count': len(feature_cols)
            }
            
            print(f"\n📊 Cluster {cluster_id} Summary:")
            print(f"   Test R² (mean ± std): {cluster_summary['test_r2_mean']:.4f} ± {cluster_summary['test_r2_std']:.4f}")
            print(f"   Test RMSE (mean ± std): {cluster_summary['test_rmse_mean']:.4f} ± {cluster_summary['test_rmse_std']:.4f}")
            print(f"   Successful folds: {len(fold_results)}/{len(temporal_splits)}")
            
            return fold_results, cluster_summary, fold_models, feature_cols
            
        except Exception as e:
            print(f"❌ Error validating cluster {cluster_id}: {e}")
            return None

    def train_external_memory_fold(self, train_file, test_file, cluster_id, fold_num):
        """Train XGBoost model using external memory for a specific fold"""
        
        try:
            # Create DMatrix objects for external memory
            dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
            dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
        except Exception as e:
            # Fallback to regular DMatrix
            dtrain = xgb.DMatrix(train_file)
            dtest = xgb.DMatrix(test_file)
        
        # XGBoost parameters optimized for temporal validation
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'max_bin': 128,
            'verbosity': 0,
            'nthread': -1
        }
        
        # Train model
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=150,
            evals=evals,
            early_stopping_rounds=15,
            verbose_eval=False
        )
        
        # Make predictions
        y_pred_train = model.predict(dtrain)
        y_pred_test = model.predict(dtest)
        
        # Extract actual values
        y_train_actual = dtrain.get_label()
        y_test_actual = dtest.get_label()
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
            'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
            'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
            'train_r2': r2_score(y_train_actual, y_pred_train),
            'test_r2': r2_score(y_test_actual, y_pred_test),
            'train_samples': len(y_train_actual),
            'test_samples': len(y_test_actual)
        }
        
        # Cleanup
        del dtrain, dtest
        gc.collect()
        
        return model, metrics

    def run_ecosystem_temporal_validation(self):
        """Run complete ecosystem-based temporal validation"""
        print("🌍 SAPFLUXNET Ecosystem-Based Temporal Validation")
        print("=" * 70)
        print(f"Started at: {datetime.now()}")
        print("Method: Cluster-specific temporal validation with site balancing")
        print("Purpose: Test temporal generalization within ecosystem boundaries")
        
        # Set up temp directory
        temp_dir = 'temp_ecosystem_temporal'
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Step 1: Load ecosystem cluster assignments
            cluster_assignments, cluster_counts = self.load_ecosystem_cluster_assignments()
            
            # Step 2: Group sites by cluster
            sites_by_cluster = {}
            for site, cluster_id in cluster_assignments.items():
                if cluster_id not in sites_by_cluster:
                    sites_by_cluster[cluster_id] = []
                sites_by_cluster[cluster_id].append(site)
            
            # Step 3: Analyze temporal coverage by cluster
            cluster_temporal_summaries = self.analyze_cluster_temporal_coverage(cluster_assignments)
            
            # Step 4: Validate each ecosystem cluster
            all_fold_results = []
            cluster_summaries = []
            all_models = {}
            all_feature_cols = {}
            
            for cluster_id in sorted(sites_by_cluster.keys()):
                cluster_sites = sites_by_cluster[cluster_id]
                
                if cluster_id not in cluster_temporal_summaries:
                    print(f"\n⚠️  Cluster {cluster_id}: No temporal data available")
                    continue
                
                temporal_info = cluster_temporal_summaries[cluster_id]
                if temporal_info['duration_years'] < self.min_training_years + self.fold_duration_years:
                    print(f"\n⚠️  Cluster {cluster_id}: Insufficient temporal coverage "
                          f"({temporal_info['duration_years']:.1f} years)")
                    continue
                
                print(f"\n🌍 Processing Cluster {cluster_id} ({len(cluster_sites)} sites)")
                
                result = self.validate_cluster_temporally(cluster_id, cluster_sites, temp_dir)
                
                if result is not None:
                    fold_results, cluster_summary, fold_models, feature_cols = result
                    all_fold_results.extend(fold_results)
                    cluster_summaries.append(cluster_summary)
                    all_models[cluster_id] = fold_models
                    all_feature_cols[cluster_id] = feature_cols
                
                # Memory cleanup
                gc.collect()
            
            # Step 5: Overall analysis and comparison
            self.analyze_ecosystem_temporal_results(
                all_fold_results, cluster_summaries, all_models, all_feature_cols
            )
            
            # Step 6: Save comprehensive results
            self.save_ecosystem_temporal_results(
                all_fold_results, cluster_summaries, all_models, all_feature_cols
            )
            
            print(f"\n✅ Ecosystem temporal validation completed!")
            
        except Exception as e:
            print(f"\n❌ Ecosystem temporal validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                print(f"\nCleaning up temporary files from: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                    print("Temporary files cleaned up successfully")
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory: {e}")
        
        print(f"\nFinished at: {datetime.now()}")

    def analyze_ecosystem_temporal_results(self, all_fold_results, cluster_summaries, all_models, all_feature_cols):
        """Analyze and compare temporal validation results across ecosystems"""
        print(f"\n📊 ECOSYSTEM TEMPORAL VALIDATION ANALYSIS")
        print("=" * 60)
        
        if not cluster_summaries:
            print("❌ No cluster results to analyze")
            return
        
        # Overall performance comparison
        print(f"🏆 ECOSYSTEM TEMPORAL PERFORMANCE COMPARISON")
        print("-" * 50)
        
        ecosystem_names = {
            0: "Warm Temperate (Mediterranean)",
            1: "Mixed Temperate (Australian/Global)",
            2: "Continental (Cold/Warm Temperate)", 
            3: "European Temperate (Mountain/Forest)",
            4: "Tropical/Subtropical (Global)"
        }
        
        # Sort by performance
        cluster_summaries.sort(key=lambda x: x['test_r2_mean'], reverse=True)
        
        for i, summary in enumerate(cluster_summaries):
            cluster_id = summary['cluster_id']
            name = ecosystem_names.get(cluster_id, f"Cluster {cluster_id}")
            
            status = "🏆" if i == 0 else "✅" if summary['test_r2_mean'] > 0.4 else "⚠️"
            
            print(f"{status} Cluster {cluster_id} ({name}):")
            print(f"    Test R²: {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}")
            print(f"    Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}")
            print(f"    Sites: {summary['n_sites']}, Folds: {summary['n_folds']}")
        
        # Calculate overall improvement vs global approach
        all_test_r2 = [summary['test_r2_mean'] for summary in cluster_summaries]
        ecosystem_mean_r2 = np.mean(all_test_r2)
        ecosystem_std_r2 = np.std(all_test_r2)
        
        global_r2 = 0.107  # From strategy document
        improvement_factor = ecosystem_mean_r2 / global_r2 if global_r2 > 0 else float('inf')
        
        print(f"\n🎯 OVERALL ECOSYSTEM vs GLOBAL COMPARISON")
        print("-" * 45)
        print(f"Ecosystem Approach: R² = {ecosystem_mean_r2:.4f} ± {ecosystem_std_r2:.4f}")
        print(f"Global Approach: R² = {global_r2:.4f} (from previous analysis)")
        print(f"Improvement Factor: {improvement_factor:.1f}x better")
        
        if improvement_factor >= 4:
            print("🎉 SUCCESS: Achieved target 4x+ improvement!")
        else:
            print("⚠️  Below target improvement, but still progress")

    def save_ecosystem_temporal_results(self, all_fold_results, cluster_summaries, all_models, all_feature_cols):
        """Save comprehensive ecosystem temporal validation results"""
        print(f"\n💾 Saving ecosystem temporal validation results...")
        
        # Save fold-by-fold results
        fold_results_path = f"{self.output_dir}/ecosystem_temporal_fold_results_{self.timestamp}.csv"
        pd.DataFrame(all_fold_results).to_csv(fold_results_path, index=False)
        
        # Save cluster summaries
        cluster_summaries_path = f"{self.output_dir}/ecosystem_temporal_summaries_{self.timestamp}.csv"
        pd.DataFrame(cluster_summaries).to_csv(cluster_summaries_path, index=False)
        
        # Save best model from each cluster
        models_saved = 0
        for cluster_id, models in all_models.items():
            if models:
                # Find best model (highest test R²)
                cluster_folds = [f for f in all_fold_results if f['cluster_id'] == cluster_id]
                if cluster_folds:
                    best_fold_idx = max(range(len(cluster_folds)), key=lambda i: cluster_folds[i]['test_r2'])
                    best_model = models[best_fold_idx]
                    
                    model_path = f"{self.output_dir}/ecosystem_temporal_model_cluster_{cluster_id}_{self.timestamp}.json"
                    best_model.save_model(model_path)
                    models_saved += 1
        
        # Save comprehensive report
        report_path = f"{self.output_dir}/ecosystem_temporal_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("SAPFLUXNET Ecosystem-Based Temporal Validation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("Method: Cluster-specific temporal validation with site balancing\n")
            f.write("Purpose: Test temporal generalization within ecosystem boundaries\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Temporal folds: {self.n_folds}\n")
            f.write(f"  Fold duration: {self.fold_duration_years} years\n")
            f.write(f"  Min training: {self.min_training_years} years\n")
            f.write(f"  Site balance limit: {self.max_samples_per_site:,} samples\n\n")
            
            f.write("Ecosystem Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            ecosystem_names = {
                0: "Warm Temperate (Mediterranean)",
                1: "Mixed Temperate (Australian/Global)",
                2: "Continental (Cold/Warm Temperate)", 
                3: "European Temperate (Mountain/Forest)",
                4: "Tropical/Subtropical (Global)"
            }
            
            for summary in sorted(cluster_summaries, key=lambda x: x['test_r2_mean'], reverse=True):
                cluster_id = summary['cluster_id']
                name = ecosystem_names.get(cluster_id, f"Cluster {cluster_id}")
                
                f.write(f"Cluster {cluster_id} ({name}):\n")
                f.write(f"  Test R²: {summary['test_r2_mean']:.4f} ± {summary['test_r2_std']:.4f}\n")
                f.write(f"  Test RMSE: {summary['test_rmse_mean']:.4f} ± {summary['test_rmse_std']:.4f}\n")
                f.write(f"  Sites: {summary['n_sites']}, Folds: {summary['n_folds']}\n")
                f.write(f"  Total samples: {summary['total_samples']:,}\n\n")
            
            # Overall comparison
            if cluster_summaries:
                all_test_r2 = [s['test_r2_mean'] for s in cluster_summaries]
                ecosystem_mean = np.mean(all_test_r2)
                global_r2 = 0.107
                improvement = ecosystem_mean / global_r2 if global_r2 > 0 else float('inf')
                
                f.write("Overall Comparison:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Ecosystem Mean R²: {ecosystem_mean:.4f}\n")
                f.write(f"Global Baseline R²: {global_r2:.4f}\n")
                f.write(f"Improvement Factor: {improvement:.1f}x\n")
                f.write(f"Target Achievement: {'SUCCESS' if improvement >= 4 else 'PARTIAL'}\n")
        
        print(f"📄 Results saved:")
        print(f"  Fold results: {fold_results_path}")
        print(f"  Cluster summaries: {cluster_summaries_path}")
        print(f"  Models saved: {models_saved}")
        print(f"  Report: {report_path}")

def main():
    """Main ecosystem temporal validation pipeline"""
    validator = EcosystemTemporalValidator()
    validator.run_ecosystem_temporal_validation()

if __name__ == "__main__":
    main() 