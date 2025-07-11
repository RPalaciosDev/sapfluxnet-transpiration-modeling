import dask.dataframe as dd
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data_dask(data_dir='comprehensive_processed'):
    """Load all processed site data using Dask"""
    print("Loading processed site data with Dask...")
    
    # Use Dask to read all parquet files at once
    ddf = dd.read_parquet(f"{data_dir}/*_comprehensive.parquet")
    
    print(f"Dask DataFrame created with {ddf.npartitions} partitions")
    print(f"Columns: {list(ddf.columns)}")
    
    # Get basic info about the dataset
    total_rows = len(ddf)
    print(f"Total rows: {total_rows:,}")
    
    return ddf

def prepare_features_dask(ddf):
    """Prepare features and target variable using Dask operations"""
    print("\nPreparing features and target with Dask...")
    
    # Remove non-feature columns
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id']
    
    # Find target variable
    if 'sap_flow' not in ddf.columns:
        raise ValueError("Target variable 'sap_flow' not found!")
    
    # Get numeric columns for features
    numeric_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols + ['sap_flow']]
    
    print(f"Target variable: sap_flow")
    print(f"Features: {len(feature_cols)} numeric features")
    
    # Remove rows with missing target
    ddf_clean = ddf.dropna(subset=['sap_flow'])
    print(f"Cleaned dataset: {len(ddf_clean):,} rows")
    
    return ddf_clean, feature_cols

def hybrid_temporal_split_dask(ddf, feature_cols, train_ratio=0.8):
    """Hybrid site + temporal split using simpler approach"""
    print(f"\nCreating hybrid temporal split (train_ratio={train_ratio})...")
    
    # Convert to pandas for easier manipulation
    print("Converting to pandas for splitting...")
    df = ddf.compute()
    
    train_data_list = []
    test_data_list = []
    
    # Sort data by timestamp first
    if 'TIMESTAMP' in df.columns:
        df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    # Split each site separately
    for site in df['site'].unique():
        site_data = df[df['site'] == site].copy()
        
        # Sort by timestamp within site
        if 'TIMESTAMP' in site_data.columns:
            site_data = site_data.sort_values('TIMESTAMP').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(site_data) * train_ratio)
        
        # Split the data
        train_data_list.append(site_data.iloc[:split_idx])
        test_data_list.append(site_data.iloc[split_idx:])
        
        print(f"  {site}: {len(site_data.iloc[:split_idx]):,} train, {len(site_data.iloc[split_idx:]):,} test")
    
    # Combine all train and test data
    train_data = pd.concat(train_data_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)
    
    print(f"\nFinal split:")
    print(f"  Train: {len(train_data):,} rows ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} rows ({len(test_data)/len(df)*100:.1f}%)")
    
    # Convert back to Dask DataFrames
    train_ddf = dd.from_pandas(train_data, npartitions=4)
    test_ddf = dd.from_pandas(test_data, npartitions=4)
    
    return train_ddf, test_ddf

def train_dask_xgboost(train_ddf, test_ddf, feature_cols):
    """Train XGBoost model using modern xgboost.dask"""
    print("\nTraining XGBoost model with Dask...")
    
    # Prepare data (keep as Dask DataFrames)
    X_train = train_ddf[feature_cols]
    y_train = train_ddf['sap_flow']
    X_test = test_ddf[feature_cols]
    y_test = test_ddf['sap_flow']
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'  # Better for distributed training
    }
    
    print(f"XGBoost parameters: {params}")
    print("Starting distributed training...")
    
    # Train model with modern xgboost.dask
    model = xgb.dask.DaskXGBRegressor(**params)
    
    # Fit the model (this handles distributed training automatically)
    model.fit(X_train, y_train)
    
    print("Training completed!")
    
    # Make predictions (compute results)
    print("Making predictions...")
    y_pred_train = model.predict(X_train).compute()
    y_pred_test = model.predict(X_test).compute()
    
    # Get actual values for metrics
    y_train_actual = y_train.compute()
    y_test_actual = y_test.compute()
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
        'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
        'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
        'train_r2': r2_score(y_train_actual, y_pred_train),
        'test_r2': r2_score(y_test_actual, y_pred_test)
    }
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model, metrics, y_pred_train, y_pred_test, y_train_actual, y_test_actual

def get_feature_importance(model, feature_cols):
    """Get and display feature importance"""
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def save_model_and_results(model, metrics, feature_importance, output_dir='dask_xgboost_models'):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_dask_xgboost_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature importance
    feature_path = f"{output_dir}/sapfluxnet_dask_features_{timestamp}.csv"
    feature_importance.to_csv(feature_path, index=False)
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_dask_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Dask-XGBoost Model Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Training completed: {datetime.now()}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\nModel and results saved:")
    print(f"  Model: {model_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def plot_results(y_train, y_pred_train, y_test, y_pred_test, output_dir='dask_xgboost_models'):
    """Plot model results"""
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted (Test)
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 0].set_xlabel('Actual Sap Flow')
    axes[0, 0].set_ylabel('Predicted Sap Flow')
    axes[0, 0].set_title('Test Set: Actual vs Predicted')
    axes[0, 0].grid(True)
    
    # 2. Residuals
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Sap Flow')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True)
    
    # 3. Training vs Test Performance
    axes[1, 0].scatter(y_train, y_pred_train, alpha=0.5, label='Train', color='blue')
    axes[1, 0].scatter(y_test, y_pred_test, alpha=0.5, label='Test', color='orange')
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 0].set_xlabel('Actual Sap Flow')
    axes[1, 0].set_ylabel('Predicted Sap Flow')
    axes[1, 0].set_title('Train vs Test Performance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Performance comparison
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    metrics = ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE']
    values = [train_r2, test_r2, train_rmse, test_rmse]
    
    bars = axes[1, 1].bar(metrics, values)
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].set_title('Model Performance Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Color code bars
    for i, bar in enumerate(bars):
        if 'Test' in metrics[i]:
            bar.set_color('orange')
        else:
            bar.set_color('blue')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f"{output_dir}/sapfluxnet_dask_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved: {plot_path}")
    
    plt.show()

def main():
    """Main training pipeline using Dask-XGBoost"""
    print("SAPFLUXNET Dask-XGBoost Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load data with Dask
        ddf = load_data_dask('comprehensive_processed')
        
        # Step 2: Prepare features with Dask
        ddf_clean, feature_cols = prepare_features_dask(ddf)
        
     