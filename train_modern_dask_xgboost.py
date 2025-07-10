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

def load_data_dask(data_dir='test_parquet_export'):
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
    
    # Get numeric columns for features (handle missing columns gracefully)
    numeric_cols = ddf.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols + ['sap_flow']]
    
    print(f"Target variable: sap_flow")
    print(f"Features: {len(feature_cols)} numeric features")
    
    # Remove rows with missing target
    ddf_clean = ddf.dropna(subset=['sap_flow'])
    
    # Get row count without computing the full dataset
    print(f"Cleaned dataset: {len(ddf_clean):,} rows")
    
    return ddf_clean, feature_cols

def hybrid_temporal_split_dask(ddf, feature_cols, train_ratio=0.8):
    """Hybrid site + temporal split using Dask operations"""
    print(f"\nCreating hybrid temporal split (train_ratio={train_ratio})...")
    
    # Sort by timestamp and site
    if 'TIMESTAMP' in ddf.columns:
        ddf = ddf.sort_values(['site', 'TIMESTAMP'])
    
    # Create a function to split each site
    def split_site_data(group):
        group = group.sort_values('TIMESTAMP')
        split_idx = int(len(group) * train_ratio)
        group['split'] = 'train'
        group.iloc[split_idx:, group.columns.get_loc('split')] = 'test'
        return group
    
    # Apply the split function to each site
    ddf_with_split = ddf.map_partitions(
        lambda pdf: pdf.groupby('site', group_keys=False).apply(split_site_data)
    )
    
    # Split into train and test
    train_ddf = ddf_with_split[ddf_with_split['split'] == 'train'].drop('split', axis=1)
    test_ddf = ddf_with_split[ddf_with_split['split'] == 'test'].drop('split', axis=1)
    
    print(f"Split completed:")
    print(f"  Train: {len(train_ddf):,} rows")
    print(f"  Test:  {len(test_ddf):,} rows")
    
    return train_ddf, test_ddf

def train_modern_dask_xgboost(train_ddf, test_ddf, feature_cols):
    """Train XGBoost model using modern XGBoost Dask integration"""
    print("\nTraining XGBoost model with modern Dask integration...")
    
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
    
    # Convert to Dask arrays for XGBoost
    X_train_array = X_train.to_dask_array(lengths=True)
    y_train_array = y_train.to_dask_array(lengths=True)
    X_test_array = X_test.to_dask_array(lengths=True)
    y_test_array = y_test.to_dask_array(lengths=True)
    
    # Train model with modern XGBoost Dask integration
    dtrain = xgb.dask.DaskDMatrix(X_train_array, y_train_array)
    dtest = xgb.dask.DaskDMatrix(X_test_array, y_test_array)
    
    # Train the model
    output = xgb.dask.train(
        client=None,  # Use default client
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    model = output['booster']
    print("Training completed!")
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = xgb.dask.predict(client=None, model=model, data=dtrain)
    y_pred_test = xgb.dask.predict(client=None, model=model, data=dtest)
    
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
        'importance': model.get_score(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Most Important Features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def save_model_and_results(model, metrics, feature_importance, output_dir='modern_dask_xgboost_models'):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_modern_dask_xgboost_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_path = f"{output_dir}/sapfluxnet_modern_dask_features_{timestamp}.csv"
    feature_importance.to_csv(feature_path, index=False)
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_modern_dask_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Modern Dask-XGBoost Model Results\n")
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

def plot_results(y_train, y_pred_train, y_test, y_pred_test, output_dir='modern_dask_xgboost_models'):
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
    plot_path = f"{output_dir}/sapfluxnet_modern_dask_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved: {plot_path}")
    
    plt.show()

def main():
    """Main training pipeline using modern XGBoost Dask integration"""
    print("SAPFLUXNET Modern Dask-XGBoost Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load data with Dask
        ddf = load_data_dask('test_parquet_export')
        
        # Step 2: Prepare features with Dask
        ddf_clean, feature_cols = prepare_features_dask(ddf)
        
        # Step 3: Split data with Dask
        train_ddf, test_ddf = hybrid_temporal_split_dask(ddf_clean, feature_cols, train_ratio=0.8)
        
        # Step 4: Train model with modern Dask-XGBoost
        model, metrics, y_pred_train, y_pred_test, y_train_actual, y_test_actual = train_modern_dask_xgboost(
            train_ddf, test_ddf, feature_cols
        )
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 6: Save model and results
        model_path = save_model_and_results(model, metrics, feature_importance)
        
        # Step 7: Plot results
        plot_results(y_train_actual, y_pred_train, y_test_actual, y_pred_test)
        
        print(f"\nTraining completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 