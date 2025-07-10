import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir='test_parquet_export'):
    """Load all processed site data"""
    print("Loading processed site data...")
    
    all_data = []
    sites_loaded = 0
    total_rows = 0
    
    # Get all parquet files
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('_comprehensive.parquet')]
    
    # Process files in smaller batches to avoid memory issues
    batch_size = 10  # Process 10 files at a time
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i+batch_size]
        batch_data = []
        
        for file in batch_files:
            # Extract site name
            site = file.replace('_comprehensive.parquet', '')
            file_path = os.path.join(data_dir, file)
            
            try:
                site_data = pd.read_parquet(file_path)
                
                if len(site_data) > 0:
                    batch_data.append(site_data)
                    sites_loaded += 1
                    total_rows += len(site_data)
                    print(f"  {site}: {len(site_data):,} rows")
            except Exception as e:
                print(f"  {site}: Error loading {file} - {str(e)}")
        
        # Combine batch data
        if batch_data:
            batch_combined = pd.concat(batch_data, ignore_index=True)
            all_data.append(batch_combined)
            print(f"  Batch {i//batch_size + 1}: {len(batch_combined):,} rows combined")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Combine all batches
    print(f"\nCombining {len(all_data)} batches...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_data):,} rows, {len(combined_data.columns)} columns")
    print(f"Sites loaded: {sites_loaded}")
    
    return combined_data

def prepare_features(data):
    """Prepare features and target variable"""
    print("\nPreparing features and target...")
    
    # Remove non-feature columns
    exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id']
    
    # Find target variable
    if 'sap_flow' not in data.columns:
        raise ValueError("Target variable 'sap_flow' not found!")
    
    # Prepare features (only numeric columns)
    feature_cols = []
    for col in data.columns:
        if col not in exclude_cols + ['sap_flow']:
            if pd.api.types.is_numeric_dtype(data[col]):
                feature_cols.append(col)
            else:
                print(f"  Dropping non-numeric column: {col}")
    
    print(f"Target variable: sap_flow")
    print(f"Features: {len(feature_cols)} numeric features")
    
    # Remove rows with missing target
    data_clean = data.dropna(subset=['sap_flow'])
    print(f"Cleaned dataset: {len(data_clean):,} rows")
    
    return data_clean, feature_cols

def hybrid_temporal_split(data, feature_cols, train_ratio=0.8):
    """Hybrid site + temporal split: 80% early data from each site -> train, 20% later data -> test"""
    print(f"\nCreating hybrid temporal split (train_ratio={train_ratio})...")
    
    train_data_list = []
    test_data_list = []
    
    # Sort data by timestamp first
    if 'TIMESTAMP' in data.columns:
        data = data.sort_values('TIMESTAMP').reset_index(drop=True)
    
    # Split each site separately
    for site in data['site'].unique():
        site_data = data[data['site'] == site].copy()
        
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
    print(f"  Train: {len(train_data):,} rows ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Test:  {len(test_data):,} rows ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, test_data

def train_xgboost(train_data, test_data, feature_cols):
    """Train XGBoost model"""
    print("\nTraining XGBoost model...")
    
    # Prepare data
    X_train = train_data[feature_cols]
    y_train = train_data['sap_flow']
    X_test = test_data[feature_cols]
    y_test = test_data['sap_flow']
    
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
        'n_jobs': -1
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Train model
    model = xgb.XGBRegressor(**params)
    
    # Early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=0
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model, metrics, y_pred_train, y_pred_test

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

def save_model_and_results(model, metrics, feature_importance, output_dir='xgboost_models'):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_xgboost_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature importance
    feature_path = f"{output_dir}/sapfluxnet_features_{timestamp}.csv"
    feature_importance.to_csv(feature_path, index=False)
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET XGBoost Model Results\n")
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

def plot_results(y_train, y_pred_train, y_test, y_pred_test, output_dir='xgboost_models'):
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
    plot_path = f"{output_dir}/sapfluxnet_results_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Results plot saved: {plot_path}")
    
    plt.show()

def main():
    """Main training pipeline"""
    print("SAPFLUXNET XGBoost Training Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Load data
        data = load_data('test_parquet_export')
        
        # Step 2: Prepare features
        data, feature_cols = prepare_features(data)
        
        # Step 3: Split data
        train_data, test_data = hybrid_temporal_split(data, feature_cols, train_ratio=0.8)
        
        # Step 4: Train model
        model, metrics, y_pred_train, y_pred_test = train_xgboost(train_data, test_data, feature_cols)
        
        # Step 5: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 6: Save model and results
        model_path = save_model_and_results(model, metrics, feature_importance)
        
        # Step 7: Plot results
        plot_results(train_data['sap_flow'], y_pred_train, test_data['sap_flow'], y_pred_test)
        
        print(f"\nTraining completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 