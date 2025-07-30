#!/usr/bin/env python3
"""
Simple Minimal Model Training - No Clustering
Train a single XGBoost model on 6 core transpiration features
Pure test: Can fundamental physiology beat 277 engineered features?
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import argparse
import gc

warnings.filterwarnings('ignore')

class MinimalModelTrainer:
    """
    Simple trainer for minimal feature model
    No clustering - just train/test split
    """
    
    def __init__(self, data_dir='processed_minimal', site_split_file='site_split_assignment.json',
                 output_dir='minimal_model_results'):
        self.data_dir = Path(data_dir)
        self.site_split_file = site_split_file
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load train/test split
        self.load_site_split()
        
        print(f"🚀 Minimal Model Trainer")
        print(f"📁 Data: {data_dir}")
        print(f"📁 Output: {output_dir}")
        print(f"🎯 Strategy: Single model, 6 core features, simple train/test")
        print(f"📊 Train sites: {len(self.train_sites)}")
        print(f"📊 Test sites: {len(self.test_sites)}")
    
    def load_site_split(self):
        """Load train/test site split"""
        print(f"📂 Loading site split: {self.site_split_file}")
        
        with open(self.site_split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_sites = set(split_data['train_sites'])
        self.test_sites = set(split_data['test_sites'])
        
        print(f"✅ Split loaded: {len(self.train_sites)} train + {len(self.test_sites)} test")
    
    def load_data(self):
        """Load all minimal parquet files and split by train/test sites"""
        print(f"\n📊 Loading minimal data...")
        
        train_data = []
        test_data = []
        
        parquet_files = list(self.data_dir.glob('*_minimal.parquet'))
        
        for parquet_file in parquet_files:
            site_name = parquet_file.stem.replace('_minimal', '')
            
            try:
                df = pd.read_parquet(parquet_file)
                
                if len(df) == 0:
                    continue
                
                # Check if we have target column
                if 'sap_flow' not in df.columns:
                    print(f"  ⚠️  {site_name}: No sap_flow column")
                    continue
                
                # Split by train/test
                if site_name in self.train_sites:
                    train_data.append(df)
                    print(f"  ✅ Train: {site_name} ({len(df):,} rows)")
                elif site_name in self.test_sites:
                    test_data.append(df)
                    print(f"  ✅ Test: {site_name} ({len(df):,} rows)")
                else:
                    print(f"  ⚠️  {site_name}: Not in train/test split")
                
            except Exception as e:
                print(f"  ❌ {site_name}: Error loading - {e}")
                continue
        
        # Combine data
        if train_data:
            train_df = pd.concat(train_data, ignore_index=True)
            print(f"\n📈 Train data: {len(train_df):,} rows from {len(train_data)} sites")
        else:
            raise ValueError("No training data loaded!")
        
        if test_data:
            test_df = pd.concat(test_data, ignore_index=True)
            print(f"📈 Test data: {len(test_df):,} rows from {len(test_data)} sites")
        else:
            raise ValueError("No test data loaded!")
        
        return train_df, test_df
    
    def prepare_features(self, train_df, test_df):
        """Prepare features for training"""
        print(f"\n🔧 Preparing features...")
        
        # Expected minimal features
        feature_cols = [
            'ta_max_72h', 'ta_mean_24h', 'vpd_mean_24h', 
            'sw_in_mean_24h', 'sw_in', 'vpd'
        ]
        
        target_col = 'sap_flow'
        
        # Check available features
        available_features = [f for f in feature_cols if f in train_df.columns]
        missing_features = set(feature_cols) - set(available_features)
        
        print(f"✅ Available features: {available_features}")
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        # Prepare training data
        X_train = train_df[available_features].copy()
        y_train = train_df[target_col].copy()
        
        # Prepare test data
        X_test = test_df[available_features].copy()
        y_test = test_df[target_col].copy()
        
        # Remove rows with missing target
        train_mask = ~y_train.isna()
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        test_mask = ~y_test.isna()
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # Handle missing features (fill with median from training data)
        for feature in available_features:
            if X_train[feature].isna().any():
                fill_value = X_train[feature].median()
                X_train[feature].fillna(fill_value, inplace=True)
                X_test[feature].fillna(fill_value, inplace=True)
                print(f"  🔧 Filled {feature} with median: {fill_value:.3f}")
        
        print(f"📊 Final training data: {len(X_train):,} rows, {len(available_features)} features")
        print(f"📊 Final test data: {len(X_test):,} rows, {len(available_features)} features")
        
        return X_train, y_train, X_test, y_test, available_features
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print(f"\n🤖 Training XGBoost model...")
        
        # XGBoost parameters - simple and robust
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',  # Use GPU if available
            'gpu_id': 0,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )
        
        print(f"✅ Model trained with {model.n_estimators} trees")
        
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, features):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating model...")
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"🎯 TRAINING PERFORMANCE:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print(f"🎯 TEST PERFORMANCE:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔥 FEATURE IMPORTANCE:")
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save results
        results = {
            'timestamp': self.timestamp,
            'model_type': 'minimal_xgboost',
            'features_used': features,
            'n_features': len(features),
            'train_sites': len(self.train_sites),
            'test_sites': len(self.test_sites),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training_performance': {
                'rmse': float(train_rmse),
                'mae': float(train_mae),
                'r2': float(train_r2)
            },
            'test_performance': {
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'r2': float(test_r2)
            },
            'feature_importance': importance_df.to_dict('records')
        }
        
        # Save model
        model_file = self.output_dir / f'minimal_model_{self.timestamp}.json'
        model.save_model(str(model_file))
        
        # Save results
        results_file = self.output_dir / f'minimal_results_{self.timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  Model: {model_file}")
        print(f"  Results: {results_file}")
        
        return results
    
    def run_training(self):
        """Run complete training pipeline"""
        print(f"🚀 MINIMAL MODEL TRAINING")
        print(f"=" * 60)
        print(f"Started: {datetime.now()}")
        print(f"Strategy: 6 core features, single model, simple train/test")
        
        try:
            # Load data
            train_df, test_df = self.load_data()
            
            # Prepare features
            X_train, y_train, X_test, y_test, features = self.prepare_features(train_df, test_df)
            
            # Train model
            model = self.train_model(X_train, y_train, X_test, y_test)
            
            # Evaluate
            results = self.evaluate_model(model, X_train, y_train, X_test, y_test, features)
            
            print(f"\n🎉 MINIMAL TRAINING COMPLETE!")
            print(f"🎯 Test R²: {results['test_performance']['r2']:.4f}")
            print(f"📊 This is your baseline for minimal features!")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Minimal Model Training")
    parser.add_argument('--data-dir', default='processed_minimal',
                        help="Directory with minimal parquet files")
    parser.add_argument('--site-split-file', default='site_split_assignment.json',
                        help="Site split JSON file")
    parser.add_argument('--output-dir', default='minimal_model_results',
                        help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 MINIMAL MODEL TRAINING")
    print("=" * 50)
    print(f"Data: {args.data_dir}")
    print(f"Split: {args.site_split_file}")
    print(f"Output: {args.output_dir}")
    print(f"Started: {datetime.now()}")
    
    trainer = MinimalModelTrainer(
        data_dir=args.data_dir,
        site_split_file=args.site_split_file,
        output_dir=args.output_dir
    )
    
    results = trainer.run_training()
    
    print(f"\nFinished: {datetime.now()}")

if __name__ == "__main__":
    main()