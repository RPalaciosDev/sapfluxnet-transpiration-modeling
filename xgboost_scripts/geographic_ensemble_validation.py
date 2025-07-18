#!/usr/bin/env python3
"""
Geographic Ensemble Validation for SAPFLUXNET Data
Learn from random model success and apply ensemble methods for better spatial generalization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import os
from datetime import datetime
import warnings
import gc
import json
from pathlib import Path

warnings.filterwarnings('ignore')

def load_site_metadata(parquet_dir):
    """Load site metadata to understand geographic patterns"""
    print("Loading site metadata for geographic analysis...")
    
    # Get unique sites and their characteristics
    site_info = {}
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    for parquet_file in parquet_files[:5]:  # Sample first 5 files
        file_path = os.path.join(parquet_dir, parquet_file)
        df_sample = pd.read_parquet(file_path).head(1000)
        
        if 'site' in df_sample.columns:
            for site in df_sample['site'].unique():
                if site not in site_info:
                    site_data = df_sample[df_sample['site'] == site].iloc[0]
                    site_info[site] = {
                        'latitude': site_data.get('latitude', 0),
                        'longitude': site_data.get('longitude', 0),
                        'elevation': site_data.get('elevation', 0),
                        'climate_zone': site_data.get('climate_zone_code', 0),
                        'biome': site_data.get('biome_code', 0),
                        'sample_count': len(df_sample[df_sample['site'] == site])
                    }
    
    return pd.DataFrame.from_dict(site_info, orient='index')

def create_geographic_clusters(site_metadata, n_clusters=5):
    """Create geographic clusters for ensemble approach"""
    print(f"Creating {n_clusters} geographic clusters...")
    
    # Use latitude, longitude, elevation for clustering
    features = ['latitude', 'longitude', 'elevation']
    X = site_metadata[features].fillna(0)
    
    # Normalize features
    X_norm = (X - X.mean()) / X.std()
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_norm)
    
    site_metadata['cluster'] = clusters
    
    print("Geographic clusters created:")
    for i in range(n_clusters):
        cluster_sites = site_metadata[site_metadata['cluster'] == i]
        print(f"  Cluster {i}: {len(cluster_sites)} sites")
        print(f"    Lat range: {cluster_sites['latitude'].min():.2f} to {cluster_sites['latitude'].max():.2f}")
        print(f"    Lon range: {cluster_sites['longitude'].min():.2f} to {cluster_sites['longitude'].max():.2f}")
    
    return site_metadata

def create_ensemble_splits(df, site_metadata, n_splits=5):
    """Create multiple train/test splits for ensemble"""
    print("Creating ensemble splits...")
    
    splits = []
    
    # Split 1: Random split (baseline)
    print("  Split 1: Random split")
    random_idx = np.random.permutation(len(df))
    split_point = int(0.8 * len(df))
    train_idx = random_idx[:split_point]
    test_idx = random_idx[split_point:]
    splits.append(('random', train_idx, test_idx))
    
    # Split 2-5: Geographic cluster splits
    for cluster_id in range(4):
        print(f"  Split {cluster_id + 2}: Geographic cluster {cluster_id}")
        cluster_sites = site_metadata[site_metadata['cluster'] == cluster_id].index
        cluster_mask = df['site'].isin(cluster_sites)
        
        if cluster_mask.sum() > 1000:  # Only if enough data
            cluster_df = df[cluster_mask]
            cluster_idx = cluster_df.index
            split_point = int(0.8 * len(cluster_idx))
            train_idx = cluster_idx[:split_point]
            test_idx = cluster_idx[split_point:]
            splits.append((f'cluster_{cluster_id}', train_idx, test_idx))
    
    return splits

def train_ensemble_models(df, splits, feature_cols, target_col):
    """Train multiple models for ensemble"""
    print("Training ensemble models...")
    
    models = {}
    predictions = {}
    
    for split_name, train_idx, test_idx in splits:
        print(f"  Training {split_name} model...")
        
        # Prepare data
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
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
        
        print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"    Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        
        models[split_name] = model
        predictions[split_name] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    
    return models, predictions

def create_stacking_ensemble(df, splits, feature_cols, target_col):
    """Create a stacking ensemble"""
    print("Creating stacking ensemble...")
    
    # Prepare base models
    base_models = []
    for split_name, train_idx, test_idx in splits[:3]:  # Use first 3 splits
        train_data = df.iloc[train_idx]
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        base_models.append((split_name, model))
    
    # Create stacking ensemble
    meta_model = Ridge(alpha=1.0)
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3
    )
    
    # Train on full dataset
    X = df[feature_cols]
    y = df[target_col]
    stacking.fit(X, y)
    
    return stacking

def evaluate_ensemble_performance(models, predictions, df, feature_cols, target_col):
    """Evaluate ensemble performance"""
    print("Evaluating ensemble performance...")
    
    # Calculate ensemble predictions (simple average)
    all_test_preds = []
    all_test_actuals = []
    
    for split_name, preds in predictions.items():
        if 'test_pred' in preds:
            # Get test data for this split
            test_mask = np.zeros(len(df), dtype=bool)
            # This is simplified - in practice you'd track test indices properly
            all_test_preds.append(preds['test_pred'])
            # all_test_actuals.append(y_test)  # Would need to track actual test data
    
    # For now, just report individual model performance
    print("\nIndividual Model Performance:")
    print("=" * 50)
    
    for split_name, preds in predictions.items():
        print(f"{split_name:15} | Train R²: {preds['train_r2']:8.4f} | Test R²: {preds['test_r2']:8.4f}")
        print(f"{'':15} | Train RMSE: {preds['train_rmse']:8.4f} | Test RMSE: {preds['test_rmse']:8.4f}")
    
    return predictions

def save_ensemble_results(models, predictions, output_dir='ensemble_results'):
    """Save ensemble results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model performance summary
    results = []
    for split_name, preds in predictions.items():
        results.append({
            'split_name': split_name,
            'train_r2': preds['train_r2'],
            'test_r2': preds['test_r2'],
            'train_rmse': preds['train_rmse'],
            'test_rmse': preds['test_rmse']
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'ensemble_performance.csv'), index=False)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'ensemble_summary.txt'), 'w') as f:
        f.write("Geographic Ensemble Validation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("Individual Model Performance:\n")
        f.write("-" * 30 + "\n")
        for split_name, preds in predictions.items():
            f.write(f"{split_name}:\n")
            f.write(f"  Train R²: {preds['train_r2']:.4f}\n")
            f.write(f"  Test R²: {preds['test_r2']:.4f}\n")
            f.write(f"  Train RMSE: {preds['train_rmse']:.4f}\n")
            f.write(f"  Test RMSE: {preds['test_rmse']:.4f}\n\n")
    
    print(f"Results saved to {output_dir}/")

def main():
    """Main ensemble validation pipeline"""
    print("SAPFLUXNET Geographic Ensemble Validation")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print("Approach: Learn from random model success using ensemble methods")
    
    # Load data
    parquet_dir = '../processed_parquet'
    
    # Load site metadata
    site_metadata = load_site_metadata(parquet_dir)
    
    if len(site_metadata) == 0:
        print("No site metadata found. Using simplified approach...")
        # Fallback to random splits only
        return
    
    # Create geographic clusters
    site_metadata = create_geographic_clusters(site_metadata, n_clusters=5)
    
    # Load main dataset (sample for testing)
    print("Loading main dataset...")
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    # Load first file as sample
    sample_file = os.path.join(parquet_dir, parquet_files[0])
    df = pd.read_parquet(sample_file).head(50000)  # Sample for testing
    
    # Define features (universal environmental features only)
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0']
    target_col = 'sap_flow'
    
    # Universal environmental features only
    universal_features = [
        'ta', 'rh', 'sw_in', 'ppfd_in', 'vpd', 'ext_rad', 'ws',
        'swc_shallow', 'precip',
        # Lagged features
        'ta_lag_1h', 'ta_lag_3h', 'ta_lag_6h', 'ta_lag_12h', 'ta_lag_24h',
        'rh_lag_1h', 'rh_lag_3h', 'rh_lag_6h', 'rh_lag_12h', 'rh_lag_24h',
        'sw_in_lag_1h', 'sw_in_lag_3h', 'sw_in_lag_6h', 'sw_in_lag_12h', 'sw_in_lag_24h',
        'vpd_lag_1h', 'vpd_lag_3h', 'vpd_lag_6h', 'vpd_lag_12h', 'vpd_lag_24h',
        'ws_lag_1h', 'ws_lag_3h', 'ws_lag_6h', 'ws_lag_12h', 'ws_lag_24h',
        'swc_shallow_lag_1h', 'swc_shallow_lag_3h', 'swc_shallow_lag_6h', 
        'swc_shallow_lag_12h', 'swc_shallow_lag_24h',
        'precip_lag_1h', 'precip_lag_3h', 'precip_lag_6h', 'precip_lag_12h', 'precip_lag_24h',
        'ppfd_in_lag_1h', 'ppfd_in_lag_3h', 'ppfd_in_lag_6h', 'ppfd_in_lag_12h', 'ppfd_in_lag_24h'
    ]
    
    # Filter to available features
    available_features = [f for f in universal_features if f in df.columns]
    print(f"Using {len(available_features)} universal environmental features")
    
    # Create ensemble splits
    splits = create_ensemble_splits(df, site_metadata, n_splits=5)
    
    # Train ensemble models
    models, predictions = train_ensemble_models(df, splits, available_features, target_col)
    
    # Evaluate performance
    evaluate_ensemble_performance(models, predictions, df, available_features, target_col)
    
    # Save results
    save_ensemble_results(models, predictions)
    
    print(f"\nEnsemble validation completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 