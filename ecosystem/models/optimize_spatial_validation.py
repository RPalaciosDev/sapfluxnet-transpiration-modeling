#!/usr/bin/env python3
"""
Optimize Spatial Validation Performance
Systematic approach to improve cluster-specific model performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
import optuna
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpatialValidationOptimizer:
    """
    Optimize cluster-specific models for better spatial generalization
    """
    
    def __init__(self, results_dir='./results/optimized_spatial_validation'):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(results_dir, exist_ok=True)
        
        # Define optimization strategies
        self.optimization_strategies = {
            'hyperparameter_tuning': True,
            'feature_selection': True,
            'ensemble_methods': True,
            'regularization_boost': True,
            'cluster_specific_tuning': True
        }
    
    def analyze_current_problems(self, fold_results_path):
        """Analyze current validation results to identify specific issues"""
        print("🔍 Analyzing current validation problems...")
        
        df = pd.read_csv(fold_results_path)
        
        # Identify problematic patterns
        problems = {
            'negative_r2_folds': df[df['test_r2'] < 0],
            'high_variance_clusters': df.groupby('cluster')['test_r2'].std().sort_values(ascending=False),
            'overfitting_folds': df[df['train_r2'] - df['test_r2'] > 0.3],
            'poor_performance_sites': df[df['test_r2'] < 0.5]
        }
        
        print(f"\n📊 Problem Analysis:")
        print(f"  Negative R² folds: {len(problems['negative_r2_folds'])}")
        print(f"  High overfitting folds: {len(problems['overfitting_folds'])}")
        print(f"  Poor performance sites: {len(problems['poor_performance_sites'])}")
        
        print(f"\n🎯 Cluster Variance Analysis:")
        for cluster_id, variance in problems['high_variance_clusters'].items():
            mean_r2 = df[df['cluster'] == cluster_id]['test_r2'].mean()
            print(f"  Cluster {cluster_id}: R² = {mean_r2:.3f} ± {variance:.3f}")
        
        return problems
    
    def optimize_hyperparameters_per_cluster(self, cluster_id, X_train, y_train, X_val, y_val):
        """Use Optuna to find optimal hyperparameters for each cluster"""
        print(f"🔧 Optimizing hyperparameters for Cluster {cluster_id}...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                'random_state': 42
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=trial.suggest_int('n_estimators', 50, 300),
                evals=[(dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            return r2_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        print(f"  Best R² for Cluster {cluster_id}: {study.best_value:.4f}")
        return study.best_params
    
    def feature_importance_based_selection(self, X, y, feature_cols, top_k=150):
        """Select top features based on XGBoost feature importance"""
        print(f"🎯 Selecting top {top_k} features based on importance...")
        
        # Train a quick model to get feature importance
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        model = xgb.train(params, dtrain, num_boost_round=100)
        importance = model.get_score(importance_type='gain')
        
        # Convert to feature names and sort
        feature_importance = []
        for key, value in importance.items():
            if key.startswith('f') and key[1:].isdigit():
                idx = int(key[1:])
                if idx < len(feature_cols):
                    feature_importance.append((feature_cols[idx], value))
        
        # Sort by importance and select top k
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        selected_features = [feat[0] for feat in feature_importance[:top_k]]
        
        print(f"  Selected {len(selected_features)} features")
        print(f"  Top 5: {selected_features[:5]}")
        
        return selected_features
    
    def create_ensemble_predictions(self, models, X_test):
        """Create ensemble predictions from multiple models"""
        predictions = []
        for model in models:
            if hasattr(model, 'predict'):
                dtest = xgb.DMatrix(X_test)
                pred = model.predict(dtest)
                predictions.append(pred)
        
        if predictions:
            # Simple average ensemble
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        return None
    
    def train_regularized_model(self, X_train, y_train, cluster_id, optimal_params=None):
        """Train model with enhanced regularization for better generalization"""
        
        if optimal_params is None:
            # Conservative defaults for better generalization
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,  # Shallower trees
                'learning_rate': 0.05,  # Lower learning rate
                'subsample': 0.8,  # More regularization
                'colsample_bytree': 0.8,
                'min_child_weight': 5,  # Higher minimum samples
                'reg_alpha': 2.0,  # L1 regularization
                'reg_lambda': 5.0,  # L2 regularization
                'gamma': 1.0,  # Minimum split loss
                'random_state': 42
            }
        else:
            params = optimal_params.copy()
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            params['random_state'] = 42
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=200,  # More rounds with lower learning rate
            verbose_eval=False
        )
        
        return model, params
    
    def improved_spatial_validation_fold(self, cluster_id, train_sites_data, test_site_data, 
                                       feature_cols, optimization_level='full'):
        """
        Run a single fold with all optimization strategies
        """
        try:
            # Prepare data
            X_train = train_sites_data[feature_cols].fillna(0).values
            y_train = train_sites_data['sap_flow'].values
            X_test = test_site_data[feature_cols].fillna(0).values  
            y_test = test_site_data['sap_flow'].values
            
            if len(X_test) == 0 or len(X_train) == 0:
                return None
            
            # Strategy 1: Feature Selection
            if optimization_level in ['full', 'features']:
                selected_features = self.feature_importance_based_selection(
                    X_train, y_train, feature_cols, top_k=150
                )
                feature_indices = [i for i, col in enumerate(feature_cols) if col in selected_features]
                X_train = X_train[:, feature_indices]
                X_test = X_test[:, feature_indices]
                selected_feature_cols = selected_features
            else:
                selected_feature_cols = feature_cols
            
            # Strategy 2: Hyperparameter Optimization (if full optimization)
            optimal_params = None
            if optimization_level == 'full' and len(X_train) > 1000:
                # Use a validation split for hyperparameter tuning
                val_split = int(0.8 * len(X_train))
                X_train_opt, X_val_opt = X_train[:val_split], X_train[val_split:]
                y_train_opt, y_val_opt = y_train[:val_split], y_train[val_split:]
                
                optimal_params = self.optimize_hyperparameters_per_cluster(
                    cluster_id, X_train_opt, y_train_opt, X_val_opt, y_val_opt
                )
            
            # Strategy 3: Train Regularized Model
            model, used_params = self.train_regularized_model(
                X_train, y_train, cluster_id, optimal_params
            )
            
            # Strategy 4: Ensemble (train multiple models with different seeds)
            ensemble_models = [model]
            if optimization_level == 'full':
                for seed in [123, 456, 789]:
                    ensemble_params = used_params.copy()
                    ensemble_params['random_state'] = seed
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    ensemble_model = xgb.train(
                        params=ensemble_params,
                        dtrain=dtrain,
                        num_boost_round=200,
                        verbose_eval=False
                    )
                    ensemble_models.append(ensemble_model)
            
            # Make predictions
            if len(ensemble_models) > 1:
                y_pred_test = self.create_ensemble_predictions(ensemble_models, X_test)
                y_pred_train = self.create_ensemble_predictions(ensemble_models, X_train)
            else:
                dtest = xgb.DMatrix(X_test)
                dtrain = xgb.DMatrix(X_train)
                y_pred_test = model.predict(dtest)
                y_pred_train = model.predict(dtrain)
            
            # Calculate metrics
            fold_metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'optimization_level': optimization_level,
                'n_features_used': len(selected_feature_cols),
                'ensemble_size': len(ensemble_models)
            }
            
            return fold_metrics
            
        except Exception as e:
            print(f"    ❌ Error in optimized fold: {e}")
            return None
    
    def run_optimization_comparison(self, original_results_path):
        """Compare different optimization strategies"""
        print("🚀 Running optimization comparison...")
        
        # Load original results for comparison
        original_df = pd.read_csv(original_results_path)
        
        optimization_levels = ['baseline', 'features', 'regularization', 'full']
        
        results = {}
        for level in optimization_levels:
            print(f"\n📊 Testing optimization level: {level}")
            # Here you would run the validation with different optimization levels
            # This is a framework - you'd need to integrate with your data loading
            
        return results

def main():
    """Main optimization workflow"""
    optimizer = SpatialValidationOptimizer()
    
    # Analyze current problems
    current_results = 'results/parquet_spatial_validation/parquet_spatial_fold_results_20250726_233215.csv'
    problems = optimizer.analyze_current_problems(current_results)
    
    print("\n🎯 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 50)
    
    print("\n1. 🔧 HYPERPARAMETER TUNING:")
    print("   - Cluster 0 needs aggressive regularization (high variance)")
    print("   - Cluster 1 is performing well, minor tuning only")
    print("   - Cluster 2 needs moderate optimization")
    
    print("\n2. 🎯 FEATURE SELECTION:")
    print("   - Reduce from 272 to ~150 most important features")
    print("   - Remove noisy/redundant features causing overfitting")
    
    print("\n3. 🛡️ REGULARIZATION STRATEGIES:")
    print("   - Increase L1/L2 regularization")
    print("   - Reduce max_depth (6 instead of 8)")
    print("   - Lower learning_rate (0.05 instead of 0.1)")
    print("   - Higher min_child_weight")
    
    print("\n4. 🎭 ENSEMBLE METHODS:")
    print("   - Train multiple models with different random seeds")
    print("   - Average predictions for more stable results")
    
    print("\n5. 📊 CLUSTER-SPECIFIC STRATEGIES:")
    print("   - Cluster 0: Focus on variance reduction")
    print("   - Cluster 1: Maintain current performance")  
    print("   - Cluster 2: Moderate regularization boost")

if __name__ == "__main__":
    main() 