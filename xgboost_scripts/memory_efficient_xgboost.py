"""
Memory-Efficient XGBoost Training for SAPFLUXNET Data
Falls back to sampling when memory is insufficient
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def log_memory_usage(step_name):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"   Total: {memory.total / (1024**3):.1f}GB")

def load_and_sample_data(data_dir, sample_size=500000):
    """Load and sample data to fit in memory"""
    print(f"Loading data from {data_dir} with sampling...")
    log_memory_usage("Before data loading")
    
    # Get all parquet files
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    all_data = []
    total_rows = 0
    
    for file in parquet_files:
        file_path = os.path.join(data_dir, file)
        try:
            # Load each file
            df = pd.read_parquet(file_path)
            
            # Sample if too large
            if len(df) > 50000:  # Sample large files
                df = df.sample(n=50000, random_state=42)
            
            all_data.append(df)
            total_rows += len(df)
            print(f"  Loaded {file}: {len(df):,} rows")
            
            # Check if we have enough data
            if total_rows >= sample_size:
                break
                
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Combine all data
    print("Combining data...")
    data = pd.concat(all_data, ignore_index=True)
    
    # Final sampling if still too large
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        print(f"🔧 Sampled to {len(data):,} rows for memory efficiency")
    
    log_memory_usage("After data loading")
    return data

def prepare_features_simple(data):
    """Prepare features with simple approach"""
    print("Preparing features...")
    
    # Define columns to exclude
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0'
    ]
    
    # Find target column
    target_col = 'sap_flow'
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Select feature columns (keep only essential ones)
    feature_cols = [col for col in data.columns 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    # Keep only most important features
    priority_features = [col for col in feature_cols if any(base in col for base in 
                       ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc', 'ppfd', 
                        'hour', 'day_of_year', 'month', 'year', 'lag_1h', 'lag_3h', 
                        'rolling_3h', 'rolling_6h', 'site_lat', 'site_lon'])]
    
    feature_cols = priority_features[:30]  # Limit to 30 features
    
    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)} columns")
    print(f"First 10 features: {feature_cols[:10]}")
    
    return feature_cols, target_col

def random_split_pandas(data, feature_cols, target_col, train_ratio=0.8):
    """Random split using pandas"""
    print(f"Creating random split (train_ratio={train_ratio})...")
    
    # Remove missing values
    data_clean = data.dropna(subset=[target_col])
    
    # Random split
    train_data = data_clean.sample(frac=train_ratio, random_state=42)
    test_data = data_clean.drop(train_data.index)
    
    # Prepare features and target
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data[target_col]
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data[target_col]
    
    print(f"Train: {len(X_train):,} rows")
    print(f"Test: {len(X_test):,} rows")
    
    return X_train, X_test, y_train, y_test

def train_memory_efficient_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with memory-efficient settings"""
    print("Training memory-efficient XGBoost...")
    log_memory_usage("Before training")
    
    # Memory-efficient parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 64,  # Reduced for memory
        'verbosity': 0,
        'nthread': 1  # Single thread for memory efficiency
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,  # Reduced rounds
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    log_memory_usage("After training")
    
    # Make predictions
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"\nMemory-Efficient Model Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    
    return model, metrics

def main():
    """Main memory-efficient training pipeline"""
    print("SAPFLUXNET Memory-Efficient XGBoost Training")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    if available_memory < 4.0:
        print("⚠️  Low memory detected - using ultra-conservative settings")
        sample_size = 100000
    else:
        sample_size = 500000
    
    try:
        # Load and sample data
        data = load_and_sample_data('test_parquet_export', sample_size)
        
        # Prepare features
        feature_cols, target_col = prepare_features_simple(data)
        
        # Create split
        X_train, X_test, y_train, y_test = random_split_pandas(
            data, feature_cols, target_col
        )
        
        # Train model
        model, metrics = train_memory_efficient_xgboost(
            X_train, X_test, y_train, y_test
        )
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"memory_efficient_model_{timestamp}.json"
        model.save_model(model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"💡 This model uses sampling for memory efficiency")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 