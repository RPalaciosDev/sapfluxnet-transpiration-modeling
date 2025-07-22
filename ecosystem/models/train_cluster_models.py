import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import joblib

PARQUET_DIR = '../../processed_parquet'
RESULTS_DIR = './results/cluster_models'
CLUSTER_COL = 'ecosystem_cluster'
TARGET_COL = 'sap_flow'  # Change if your target column is different
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load all parquet files and concatenate
dataframes = []
for parquet_file in glob.glob(os.path.join(PARQUET_DIR, '*.parquet')):
    df = pd.read_parquet(parquet_file)
    if CLUSTER_COL in df.columns:
        dataframes.append(df)
    else:
        print(f"[WARN] {parquet_file} missing {CLUSTER_COL}, skipping.")

if not dataframes:
    raise RuntimeError("No parquet files with cluster labels found.")

all_data = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(all_data)} rows from {len(dataframes)} parquet files.")

# 2. Train a model for each cluster
clusters = sorted(all_data[CLUSTER_COL].dropna().unique())
print(f"Found clusters: {clusters}")

metrics_list = []
for cluster in clusters:
    cluster_data = all_data[all_data[CLUSTER_COL] == cluster].copy()
    print(f"\n=== Training model for cluster {cluster} ({len(cluster_data)} rows) ===")
    
    # Exclude non-feature columns
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', CLUSTER_COL]
    feature_cols = [col for col in cluster_data.columns if col not in exclude_cols + [TARGET_COL] and not col.endswith('_flags') and not col.endswith('_md')]
    
    # Drop rows with missing target
    cluster_data = cluster_data.dropna(subset=[TARGET_COL])
    
    X = cluster_data[feature_cols].values
    y = cluster_data[TARGET_COL].values
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # XGBoost parameters (simple, can be tuned)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'verbosity': 1,
        'nthread': -1
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=15, verbose_eval=10)
    
    # Evaluate
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    
    print(f"Cluster {cluster} Results:")
    print(f"  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, MAE: {train_mae:.4f}")
    print(f"  Val   RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, MAE: {val_mae:.4f}")
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, f'xgb_model_cluster_{cluster}.json')
    model.save_model(model_path)
    print(f"  Model saved: {model_path}")
    
    # Save metrics
    metrics_list.append({
        'cluster': cluster,
        'n_rows': len(cluster_data),
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'model_path': model_path,
        'feature_count': len(feature_cols)
    })
    
    # Save feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    importance_path = os.path.join(RESULTS_DIR, f'feature_importance_cluster_{cluster}.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  Feature importance saved: {importance_path}")

# Save all metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_path = os.path.join(RESULTS_DIR, 'cluster_model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"\nAll cluster model metrics saved: {metrics_path}")
print("Done.")
