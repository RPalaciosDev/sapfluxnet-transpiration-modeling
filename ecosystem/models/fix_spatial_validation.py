#!/usr/bin/env python3
"""
CORRECTED Spatial Validation - Eliminates Data Leakage
Retrains models for each LOSO fold instead of using pre-trained models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime

class CorrectSpatialValidator:
    
    def validate_cluster_spatially_correct(self, cluster_id, cluster_sites, xgb_params):
        """
        CORRECTED: Perform Leave-One-Site-Out validation with proper retraining
        
        Args:
            cluster_id: ID of the cluster
            cluster_sites: List of sites in this cluster
            xgb_params: XGBoost parameters for retraining
        """
        print(f"\n{'='*60}")
        print(f"CORRECTED SPATIAL VALIDATION FOR CLUSTER {cluster_id}")
        print(f"{'='*60}")
        print(f"Sites in cluster: {len(cluster_sites)}")
        
        if len(cluster_sites) < 3:
            print(f"⚠️  Skipping cluster {cluster_id}: Need at least 3 sites for spatial validation")
            return None
        
        # Load ALL data for this cluster once
        cluster_df = self.load_cluster_data(cluster_id, cluster_sites)
        
        fold_results = []
        
        # Leave-One-Site-Out validation
        for i, test_site in enumerate(cluster_sites):
            print(f"\n--- Fold {i+1}/{len(cluster_sites)}: Test site {test_site} ---")
            
            try:
                # CRITICAL FIX: Split data by site BEFORE training
                train_sites = [s for s in cluster_sites if s != test_site]
                
                # Create training data (EXCLUDING test site)
                train_mask = cluster_df['site'].isin(train_sites)
                test_mask = cluster_df['site'] == test_site
                
                X_train = cluster_df.loc[train_mask, self.feature_cols].values
                y_train = cluster_df.loc[train_mask, 'E'].values
                X_test = cluster_df.loc[test_mask, self.feature_cols].values
                y_test = cluster_df.loc[test_mask, 'E'].values
                
                if len(X_test) == 0:
                    print(f"  ⚠️  No test data for {test_site}")
                    continue
                
                print(f"  Train sites: {len(train_sites)} ({len(X_train):,} samples)")
                print(f"  Test site: {test_site} ({len(X_test):,} samples)")
                
                # CRITICAL FIX: RETRAIN model for this fold
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
                # Train new model ONLY on training sites for this fold
                fold_model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    verbose_eval=False
                )
                
                # Make predictions with fold-specific model
                y_pred_train = fold_model.predict(dtrain)
                y_pred_test = fold_model.predict(dtest)
                
                # Calculate metrics
                fold_metrics = {
                    'cluster': cluster_id,
                    'fold': i + 1,
                    'test_site': test_site,
                    'train_sites': len(train_sites),
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
                print(f"  Test RMSE: {fold_metrics['test_rmse']:.4f}")
                
                # Clean up fold-specific model
                del fold_model, dtrain, dtest
                
            except Exception as e:
                print(f"  ❌ Error in fold {i+1}: {e}")
                continue
        
        return self._calculate_cluster_summary(cluster_id, fold_results)
    
    def _calculate_cluster_summary(self, cluster_id, fold_results):
        """Calculate summary statistics for the cluster"""
        if not fold_results:
            return None
        
        df = pd.DataFrame(fold_results)
        
        summary = {
            'cluster': cluster_id,
            'total_sites': len(df),
            'successful_folds': len(df),
            'mean_test_r2': df['test_r2'].mean(),
            'std_test_r2': df['test_r2'].std(),
            'mean_test_rmse': df['test_rmse'].mean(),
            'std_test_rmse': df['test_rmse'].std(),
            'min_test_r2': df['test_r2'].min(),
            'max_test_r2': df['test_r2'].max()
        }
        
        print(f"\n📊 Cluster {cluster_id} Summary:")
        print(f"  Successful folds: {summary['successful_folds']}")
        print(f"  Mean Test R²: {summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
        print(f"  Mean Test RMSE: {summary['mean_test_rmse']:.4f} ± {summary['std_test_rmse']:.4f}")
        
        return fold_results, summary


def main():
    """
    Example of how to run corrected spatial validation
    """
    
    # XGBoost parameters (same as used in training)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    validator = CorrectSpatialValidator()
    
    # Load cluster assignments
    cluster_assignments = validator.load_cluster_assignments()
    
    # Group sites by cluster
    sites_by_cluster = {}
    for site, cluster_id in cluster_assignments.items():
        if cluster_id not in sites_by_cluster:
            sites_by_cluster[cluster_id] = []
        sites_by_cluster[cluster_id].append(site)
    
    # Run corrected validation for each cluster
    all_results = []
    
    for cluster_id, cluster_sites in sites_by_cluster.items():
        print(f"\n🔄 Processing Cluster {cluster_id}...")
        
        result = validator.validate_cluster_spatially_correct(
            cluster_id, cluster_sites, xgb_params
        )
        
        if result is not None:
            fold_results, cluster_summary = result
            all_results.extend(fold_results)
    
    # Save corrected results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = f'corrected_spatial_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Corrected spatial validation completed!")
        print(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main() 