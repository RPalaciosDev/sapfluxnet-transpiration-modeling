#!/usr/bin/env python3
"""
Temporal Validation with Daily Averages for SAPFLUXNET Data
K-FOLD TEMPORAL VALIDATION using daily averaged data
Realistic temporal patterns with manageable data size
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import warnings
import gc
import json
from pathlib import Path

warnings.filterwarnings('ignore')

def load_daily_averages(daily_data_file):
    """Load daily averages data"""
    print(f"Loading daily averages from {daily_data_file}...")
    
    df = pd.read_parquet(daily_data_file)
    
    print(f"Loaded {len(df):,} daily records from {df['site'].nunique()} sites")
    print(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def create_temporal_features(df):
    """Create additional temporal features from daily data"""
    print("Creating temporal features...")
    
    # Basic temporal features (already added in daily averages script)
    # year, month, day_of_year, season
    
    # Additional temporal features
    df['day_of_week'] = df['TIMESTAMP'].dt.dayofweek
    df['week_of_year'] = df['TIMESTAMP'].dt.isocalendar().week
    df['quarter'] = df['TIMESTAMP'].dt.quarter
    
    # Cyclical encoding for seasonal patterns
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lagged features (previous day values)
    lag_features = ['sap_flow', 'ta', 'rh', 'sw_in', 'ppfd_in', 'vpd', 'ext_rad', 'ws', 'swc_shallow']
    available_lag_features = [f for f in lag_features if f in df.columns]
    
    for feature in available_lag_features:
        df[f'{feature}_lag_1d'] = df.groupby('site')[feature].shift(1)
        df[f'{feature}_lag_7d'] = df.groupby('site')[feature].shift(7)
    
    # Rolling averages (7-day and 30-day windows)
    for feature in available_lag_features:
        df[f'{feature}_rolling_7d'] = df.groupby('site')[feature].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{feature}_rolling_30d'] = df.groupby('site')[feature].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
    
    print(f"Added temporal features. Total features: {len(df.columns)}")
    return df

def create_temporal_splits(df, n_folds=5):
    """Create temporal splits for k-fold validation"""
    print(f"Creating {n_folds}-fold temporal splits...")
    
    # Sort by site and timestamp
    df = df.sort_values(['site', 'TIMESTAMP']).reset_index(drop=True)
    
    splits = []
    
    for fold in range(n_folds):
        print(f"  Creating fold {fold + 1}/{n_folds}")
        
        # For temporal validation, we want to predict future data
        # Each fold uses earlier data for training, later data for testing
        
        # Calculate split points based on time
        total_days = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).days
        fold_size_days = total_days // n_folds
        
        # Create temporal split
        if fold == 0:
            # First fold: train on first 20% of time, test on next 20%
            train_end_date = df['TIMESTAMP'].min() + pd.Timedelta(days=int(0.2 * total_days))
            test_start_date = train_end_date
            test_end_date = df['TIMESTAMP'].min() + pd.Timedelta(days=int(0.4 * total_days))
        else:
            # Other folds: train on data up to fold, test on fold
            fold_start_date = df['TIMESTAMP'].min() + pd.Timedelta(days=fold * fold_size_days)
            fold_end_date = df['TIMESTAMP'].min() + pd.Timedelta(days=(fold + 1) * fold_size_days)
            
            train_end_date = fold_start_date
            test_start_date = fold_start_date
            test_end_date = fold_end_date
        
        train_mask = df['TIMESTAMP'] < train_end_date
        test_mask = (df['TIMESTAMP'] >= test_start_date) & (df['TIMESTAMP'] < test_end_date)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        print(f"    Train period: {train_data['TIMESTAMP'].min()} to {train_data['TIMESTAMP'].max()}")
        print(f"    Test period: {test_data['TIMESTAMP'].min()} to {test_data['TIMESTAMP'].max()}")
        print(f"    Train: {len(train_data):,} records")
        print(f"    Test: {len(test_data):,} records")
        
        splits.append((train_data, test_data))
    
    return splits

def get_feature_columns(df):
    """Get feature columns for modeling"""
    # Exclude non-feature columns
    exclude_cols = [
        'site', 'date', 'TIMESTAMP', 'year', 'month', 'day_of_year', 'season',
        'day_of_week', 'week_of_year', 'quarter'
    ]
    
    # Get all columns except target and excluded
    target_col = 'sap_flow'
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
    
    print(f"Using {len(feature_cols)} features for modeling")
    return feature_cols, target_col

def train_temporal_model(train_data, test_data, feature_cols, target_col, fold_idx):
    """Train XGBoost model for temporal validation"""
    print(f"Training temporal model for fold {fold_idx + 1}...")
    
    # Prepare data
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data[target_col]
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data[target_col]
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'feature_importance': feature_importance,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'train_period': f"{train_data['TIMESTAMP'].min()} to {train_data['TIMESTAMP'].max()}",
        'test_period': f"{test_data['TIMESTAMP'].min()} to {test_data['TIMESTAMP'].max()}"
    }

def save_temporal_results(results, feature_cols, output_dir='temporal_validation_daily_results'):
    """Save temporal validation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average metrics
    avg_metrics = {
        'train_r2_mean': np.mean([r['train_r2'] for r in results]),
        'train_r2_std': np.std([r['train_r2'] for r in results]),
        'test_r2_mean': np.mean([r['test_r2'] for r in results]),
        'test_r2_std': np.std([r['test_r2'] for r in results]),
        'train_rmse_mean': np.mean([r['train_rmse'] for r in results]),
        'train_rmse_std': np.std([r['train_rmse'] for r in results]),
        'test_rmse_mean': np.mean([r['test_rmse'] for r in results]),
        'test_rmse_std': np.std([r['test_rmse'] for r in results]),
        'train_mae_mean': np.mean([r['train_mae'] for r in results]),
        'train_mae_std': np.std([r['train_mae'] for r in results]),
        'test_mae_mean': np.mean([r['test_mae'] for r in results]),
        'test_mae_std': np.std([r['test_mae'] for r in results])
    }
    
    # Save detailed results
    results_data = []
    for i, result in enumerate(results):
        results_data.append({
            'fold': i + 1,
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'train_rmse': result['train_rmse'],
            'test_rmse': result['test_rmse'],
            'train_mae': result['train_mae'],
            'test_mae': result['test_mae'],
            'train_samples': result['train_samples'],
            'test_samples': result['test_samples'],
            'train_period': result['train_period'],
            'test_period': result['test_period']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, 'temporal_fold_results.csv'), index=False)
    
    # Save summary
    with open(os.path.join(output_dir, 'temporal_summary.txt'), 'w') as f:
        f.write("Temporal Validation Results (Daily Averages)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("Average Performance:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Train R²: {avg_metrics['train_r2_mean']:.4f} ± {avg_metrics['train_r2_std']:.4f}\n")
        f.write(f"Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}\n")
        f.write(f"Train RMSE: {avg_metrics['train_rmse_mean']:.4f} ± {avg_metrics['train_rmse_std']:.4f}\n")
        f.write(f"Test RMSE: {avg_metrics['test_rmse_mean']:.4f} ± {avg_metrics['test_rmse_std']:.4f}\n")
        f.write(f"Train MAE: {avg_metrics['train_mae_mean']:.4f} ± {avg_metrics['train_mae_std']:.4f}\n")
        f.write(f"Test MAE: {avg_metrics['test_mae_mean']:.4f} ± {avg_metrics['test_mae_std']:.4f}\n\n")
        
        f.write("Individual Fold Results:\n")
        f.write("-" * 25 + "\n")
        for i, result in enumerate(results):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Train R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}\n")
            f.write(f"  Train RMSE: {result['train_rmse']:.4f}, Test RMSE: {result['test_rmse']:.4f}\n")
            f.write(f"  Train period: {result['train_period']}\n")
            f.write(f"  Test period: {result['test_period']}\n")
            f.write(f"  Train samples: {result['train_samples']:,}, Test samples: {result['test_samples']:,}\n\n")
    
    # Save feature importance
    feature_importance_data = []
    for i, result in enumerate(results):
        for j, importance in enumerate(result['feature_importance']):
            feature_importance_data.append({
                'fold': i + 1,
                'feature_index': j,
                'feature_name': feature_cols[j],
                'importance': importance
            })
    
    importance_df = pd.DataFrame(feature_importance_data)
    importance_df.to_csv(os.path.join(output_dir, 'temporal_feature_importance.csv'), index=False)
    
    print(f"Results saved to {output_dir}/")
    return avg_metrics

def main():
    """Main temporal validation pipeline"""
    print("SAPFLUXNET Temporal Validation (Daily Averages)")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print("Approach: K-fold temporal validation using daily averaged data")
    
    # Load daily averages
    daily_data_file = 'daily_averages/sapfluxnet_daily_averages.parquet'
    
    if not os.path.exists(daily_data_file):
        print(f"❌ Daily averages file not found: {daily_data_file}")
        print("Please run create_daily_averages.py first")
        return
    
    df = load_daily_averages(daily_data_file)
    
    # Create temporal features
    df = create_temporal_features(df)
    
    # Get feature columns
    feature_cols, target_col = get_feature_columns(df)
    
    # Create temporal splits
    splits = create_temporal_splits(df, n_folds=5)
    
    # Train models
    results = []
    for i, (train_data, test_data) in enumerate(splits):
        result = train_temporal_model(train_data, test_data, feature_cols, target_col, i)
        results.append(result)
    
    # Save results
    avg_metrics = save_temporal_results(results, feature_cols)
    
    print(f"\nTemporal validation completed at: {datetime.now()}")
    print(f"Average Test R²: {avg_metrics['test_r2_mean']:.4f} ± {avg_metrics['test_r2_std']:.4f}")

if __name__ == "__main__":
    main() 