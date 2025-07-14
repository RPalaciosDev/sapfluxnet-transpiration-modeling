"""
Pandas-Only Random Split XGBoost Training for SAPFLUXNET Data
BASELINE MODEL - Uses random 80/20 split for baseline comparison
Implements traditional random cross-validation as performance baseline
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import warnings
import gc
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import glob
warnings.filterwarnings('ignore')

def force_garbage_collection(context=""):
    """Force garbage collection and print memory info"""
    if context:
        print(f"🧹 Garbage collection: {context}")
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory info
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    used_percent = memory_info.percent
    
    if context:
        print(f"   Collected {collected} objects, {available_gb:.1f}GB available ({used_percent:.1f}% used)")
    
    return collected

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def load_data_pandas(data_dir):
    """Load data using pandas with memory optimization"""
    print(f"Loading data from {data_dir} using Pandas...")
    
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Try to load parquet files first
    try:
        parquet_files = glob.glob(f"{data_dir}/*.parquet")
        if parquet_files:
            print(f"Found {len(parquet_files)} parquet files")
            
            # Load all parquet files
            dfs = []
            for i, file in enumerate(parquet_files):
                print(f"Loading {os.path.basename(file)} ({i+1}/{len(parquet_files)})")
                df = pd.read_parquet(file)
                dfs.append(df)
                
                # Garbage collect every few files
                if (i + 1) % 5 == 0:
                    force_garbage_collection(f"After loading {i+1} files")
            
            # Concatenate all dataframes
            print("Concatenating all dataframes...")
            df = pd.concat(dfs, ignore_index=True)
            del dfs  # Free memory
            
            print(f"Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
            force_garbage_collection("After parquet loading")
            
            return df
            
    except Exception as e:
        print(f"Error loading parquet files: {e}")
    
    # Fallback to CSV files
    try:
        csv_files = glob.glob(f"{data_dir}/*.csv")
        if csv_files:
            print(f"Found {len(csv_files)} CSV files")
            
            # Load all CSV files
            dfs = []
            for i, file in enumerate(csv_files):
                print(f"Loading {os.path.basename(file)} ({i+1}/{len(csv_files)})")
                df = pd.read_csv(file)
                dfs.append(df)
                
                # Garbage collect every few files
                if (i + 1) % 5 == 0:
                    force_garbage_collection(f"After loading {i+1} files")
            
            # Concatenate all dataframes
            print("Concatenating all dataframes...")
            df = pd.concat(dfs, ignore_index=True)
            del dfs  # Free memory
            
            print(f"Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
            force_garbage_collection("After CSV loading")
            
            return df
            
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        raise
    
    raise ValueError(f"No data files found in {data_dir}")

def prepare_features_pandas(df):
    """Prepare features using pandas"""
    print("Preparing features...")
    
    # Define columns to exclude
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0'
    ]
    
    # Find target column
    target_col = 'sap_flow'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Select feature columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)} columns")
    print(f"First 10 features: {feature_cols[:10]}")
    
    # Handle categorical columns by converting to numeric
    categorical_cols = df[feature_cols].select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        print(f"Converting {len(categorical_cols)} categorical columns to numeric...")
        for col in categorical_cols:
            if col in df.columns:
                # Convert categorical to numeric codes
                df[col] = pd.Categorical(df[col]).codes.astype('float64')
        
        force_garbage_collection("After categorical conversion")
    
    # Final garbage collection
    force_garbage_collection("After feature preparation")
    
    return df, feature_cols, target_col

def random_split_pandas(df, feature_cols, target_col, train_ratio=0.8, random_state=42):
    """
    Random split method using pandas and sklearn
    """
    print(f"Creating random split (train_ratio={train_ratio})...")
    print("Using sklearn train_test_split for random baseline")
    
    # Force garbage collection before intensive operation
    force_garbage_collection("Before random split preparation")
    
    # Select only needed columns
    needed_cols = [target_col] + feature_cols
    if 'site' in df.columns:
        needed_cols.append('site')
    if 'TIMESTAMP' in df.columns:
        needed_cols.append('TIMESTAMP')
    
    print(f"Selecting {len(needed_cols)} columns for splitting...")
    df_subset = df[needed_cols].copy()
    
    # Force garbage collection after column selection
    force_garbage_collection("After column selection")
    
    # Remove rows with missing target
    print("Removing rows with missing target values...")
    initial_rows = len(df_subset)
    df_clean = df_subset.dropna(subset=[target_col])
    final_rows = len(df_clean)
    
    print(f"Removed {initial_rows - final_rows:,} rows with missing target")
    print(f"Final dataset: {final_rows:,} rows")
    
    # Force garbage collection after cleaning
    force_garbage_collection("After data cleaning")
    
    # Use sklearn's train_test_split for efficient random splitting
    print("Performing random split using sklearn...")
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Include metadata columns for later use
    metadata_cols = []
    if 'site' in df_clean.columns:
        metadata_cols.append('site')
    if 'TIMESTAMP' in df_clean.columns:
        metadata_cols.append('TIMESTAMP')
    
    if metadata_cols:
        metadata = df_clean[metadata_cols]
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, 
            test_size=1-train_ratio, 
            random_state=random_state,
            shuffle=True
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=1-train_ratio, 
            random_state=random_state,
            shuffle=True
        )
        meta_train = meta_test = None
    
    # Force garbage collection after split
    force_garbage_collection("After random split")
    
    print(f"\nRandom split completed:")
    print(f"  Train: {len(X_train):,} rows ({len(X_train)/len(df_clean)*100:.1f}%) - Random sample from all data")
    print(f"  Test:  {len(X_test):,} rows ({len(X_test)/len(df_clean)*100:.1f}%) - Random sample from all data")
    print(f"  Total: {len(df_clean):,} rows")
    
    print("✅ Random split completed successfully")
    return X_train, X_test, y_train, y_test, meta_train, meta_test

def train_random_xgboost_pandas(X_train, X_test, y_train, y_test, feature_cols, target_col):
    """Train XGBoost model with random split validation using pandas"""
    print("Training XGBoost with random split validation...")
    
    # Force garbage collection before training
    force_garbage_collection("Before XGBoost training")
    
    # Fill missing values
    print("Filling missing values...")
    X_train_filled = X_train.fillna(0)
    X_test_filled = X_test.fillna(0)
    
    print("Missing values filled successfully")
    force_garbage_collection("After filling missing values")
    
    # Conservative XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,          # Slightly deeper for pandas version
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'max_bin': 256,          # More bins for pandas version
        'verbosity': 0,
        'n_jobs': -1  # Use all available cores
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Create XGBoost model
    model = xgb.XGBRegressor(**params)
    
    # Train model with early stopping
    print("Starting random split training...")
    model.fit(
        X_train_filled, y_train,
        eval_set=[(X_train_filled, y_train), (X_test_filled, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=15,
        verbose=False
    )
    
    print("Random split training completed!")
    force_garbage_collection("After model training")
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(X_train_filled)
    y_pred_test = model.predict(X_test_filled)
    
    force_garbage_collection("After predictions")
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"\nRandom Split Baseline Model Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f} (Random sample)")
    print(f"  Test R²: {metrics['test_r2']:.4f} (Random sample)")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Train samples: {len(y_train):,}")
    print(f"  Test samples: {len(y_test):,}")
    
    # Clean up intermediate variables
    del X_train_filled, X_test_filled
    force_garbage_collection("After training cleanup")
    
    return model, metrics, y_train, y_test, y_pred_train, y_pred_test

def get_feature_importance_pandas(model, feature_cols):
    """Get feature importance from pandas XGBoost model"""
    try:
        # Get feature importance
        importance_scores = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': feature_cols, 'importance': [0.0] * len(feature_cols)})

def create_visualizations_pandas(model, metrics, feature_importance, y_test, y_pred_test, 
                                meta_test, output_dir, timestamp):
    """Create comprehensive visualizations for the random baseline model"""
    print("Generating visualizations...")
    
    # Create plots directory
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance metrics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # R² comparison
    r2_data = [metrics['train_r2'], metrics['test_r2']]
    ax1.bar(['Train', 'Test'], r2_data, color=['blue', 'orange'], alpha=0.7)
    ax1.set_title('Random Baseline - R² Score')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(r2_data):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    rmse_data = [metrics['train_rmse'], metrics['test_rmse']]
    ax2.bar(['Train', 'Test'], rmse_data, color=['blue', 'orange'], alpha=0.7)
    ax2.set_title('Random Baseline - RMSE')
    ax2.set_ylabel('RMSE')
    for i, v in enumerate(rmse_data):
        ax2.text(i, v + max(rmse_data)*0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # MAE comparison
    mae_data = [metrics['train_mae'], metrics['test_mae']]
    ax3.bar(['Train', 'Test'], mae_data, color=['blue', 'orange'], alpha=0.7)
    ax3.set_title('Random Baseline - MAE')
    ax3.set_ylabel('MAE')
    for i, v in enumerate(mae_data):
        ax3.text(i, v + max(mae_data)*0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Summary text
    ax4.text(0.1, 0.8, "Random Baseline Model (Pandas)", fontsize=14, weight='bold')
    ax4.text(0.1, 0.6, f"Test R²: {metrics['test_r2']:.4f}", fontsize=12)
    ax4.text(0.1, 0.4, f"Test RMSE: {metrics['test_rmse']:.4f}", fontsize=12)
    ax4.text(0.1, 0.2, f"Test MAE: {metrics['test_mae']:.4f}", fontsize=12)
    ax4.text(0.1, 0.0, f"Purpose: Baseline comparison", fontsize=10, style='italic')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    performance_path = f"{plots_dir}/performance_random_pandas_{timestamp}.png"
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=20, color='blue')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Sap Flow')
    plt.ylabel('Predicted Sap Flow')
    plt.title('Random Baseline (Pandas) - Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add R² annotation
    plt.text(0.05, 0.95, f'R² = {metrics["test_r2"]:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    scatter_path = f"{plots_dir}/scatter_random_pandas_{timestamp}.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance plot
    top_features = feature_importance.head(15)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('Random Baseline (Pandas) - Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    importance_path = f"{plots_dir}/feature_importance_random_pandas_{timestamp}.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Residual analysis
    residuals = y_pred_test - y_test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred_test, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Random Baseline (Pandas) - Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residual distribution
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Random Baseline (Pandas) - Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    residual_path = f"{plots_dir}/residuals_random_pandas_{timestamp}.png"
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Save predictions with metadata
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred_test,
        'residuals': residuals
    })
    
    # Add site information if available
    if meta_test is not None:
        for col in meta_test.columns:
            predictions_df[col] = meta_test[col].values
    
    predictions_path = f"{output_dir}/predictions_random_pandas_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    
    print(f"✅ Visualizations created:")
    print(f"  Performance: {performance_path}")
    print(f"  Scatter plot: {scatter_path}")
    print(f"  Feature importance: {importance_path}")
    print(f"  Residual analysis: {residual_path}")
    print(f"  Predictions: {predictions_path}")
    
    return {
        'performance': performance_path,
        'scatter': scatter_path,
        'importance': importance_path,
        'residuals': residual_path,
        'predictions': predictions_path
    }

def save_random_baseline_results_pandas(model, metrics, feature_importance, feature_cols, 
                                       output_dir='pandas_random_baseline_models'):
    """Save random baseline model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_random_pandas_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_random_pandas_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_random_pandas_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in random baseline training (Pandas version):\n")
        f.write("Method: Random 80/20 split\n")
        f.write("Split: Random sample (train) vs Random sample (test)\n")
        f.write("Purpose: Baseline comparison for temporal validation methods\n")
        f.write("Implementation: Pure Pandas (no Dask)\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_random_pandas_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET Random Baseline Training Results (Pandas)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: Random 80/20 split (baseline)\n")
        f.write("Approach: Traditional random cross-validation\n")
        f.write("Purpose: Baseline comparison for temporal validation methods\n")
        f.write("Implementation: Pure Pandas (no Dask dependencies)\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- Random 80/20 split across all data\n")
        f.write("- No temporal considerations\n")
        f.write("- May include data leakage (future → past)\n")
        f.write("- Provides upper bound on performance\n")
        f.write("- Baseline for comparison with temporal methods\n")
        f.write("- Traditional ML validation approach\n")
        f.write("- Uses sklearn train_test_split for efficiency\n")
        
        f.write("\nImplementation Notes:\n")
        f.write("-" * 20 + "\n")
        f.write("- Pure Pandas implementation (no Dask)\n")
        f.write("- Should be faster and more memory efficient\n")
        f.write("- Uses sklearn for random splitting\n")
        f.write("- Better suited for single-machine processing\n")
        
        f.write("\nInterpretation:\n")
        f.write("-" * 15 + "\n")
        f.write("- This represents the 'best case' performance\n")
        f.write("- Temporal validation should be lower than this\n")
        f.write("- Large gap indicates temporal dependencies\n")
        f.write("- Small gap indicates good temporal generalization\n")
    
    print(f"\nRandom baseline model results saved:")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main random baseline training pipeline using Pandas"""
    print("SAPFLUXNET Pandas Random Baseline XGBoost Training")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("Method: Random 80/20 split (baseline)")
    print("Implementation: Pure Pandas (no Dask)")
    print("Purpose: Baseline comparison for temporal validation methods")
    
    # Check if we're in Google Colab
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted.")
        data_dir = '/content/drive/MyDrive/comprehensive_processed'
    except ImportError:
        print("Not running in Google Colab, using local directory.")
        data_dir = 'comprehensive_processed'
    
    try:
        # Step 1: Load data
        print("\n" + "="*50)
        print("STEP 1: LOADING DATA")
        print("="*50)
        df = load_data_pandas(data_dir)
        
        # Step 2: Prepare features
        print("\n" + "="*50)
        print("STEP 2: PREPARING FEATURES")
        print("="*50)
        df_clean, feature_cols, target_col = prepare_features_pandas(df)
        
        # Step 3: Create random split
        print("\n" + "="*50)
        print("STEP 3: CREATING RANDOM BASELINE SPLIT")
        print("="*50)
        X_train, X_test, y_train, y_test, meta_train, meta_test = random_split_pandas(
            df_clean, feature_cols, target_col
        )
        
        # Clean up original dataframe
        del df, df_clean
        force_garbage_collection("After data splitting")
        
        # Step 4: Train model
        print("\n" + "="*50)
        print("STEP 4: TRAINING RANDOM BASELINE MODEL")
        print("="*50)
        model, metrics, y_train, y_test, y_pred_train, y_pred_test = train_random_xgboost_pandas(
            X_train, X_test, y_train, y_test, feature_cols, target_col
        )
        
        # Clean up training data
        del X_train, y_pred_train
        force_garbage_collection("After model training")
        
        # Step 5: Get feature importance
        print("\n" + "="*50)
        print("STEP 5: EXTRACTING FEATURE IMPORTANCE")
        print("="*50)
        feature_importance = get_feature_importance_pandas(model, feature_cols)
        
        # Step 6: Save results
        print("\n" + "="*50)
        print("STEP 6: SAVING RESULTS")
        print("="*50)
        output_dir = 'pandas_random_baseline_models'
        model_path = save_random_baseline_results_pandas(model, metrics, feature_importance, feature_cols, output_dir)
        
        # Step 7: Generate visualizations
        print("\n" + "="*50)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*50)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        visualization_paths = create_visualizations_pandas(
            model, metrics, feature_importance, y_test, y_pred_test, 
            meta_test, output_dir, timestamp
        )
        
        # Final cleanup
        del X_test, y_train, y_test, y_pred_test, meta_test
        force_garbage_collection("Final cleanup")
        
        print(f"\n✅ Random baseline training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"📊 Visualizations created in: {output_dir}/plots/")
        print(f"💡 This model uses random validation - represents upper bound performance")
        print(f"🚀 Implementation: Pure Pandas (no Dask dependencies)")
        print(f"📊 Method: Random 80/20 split baseline")
        print(f"🎯 Purpose: Comparison baseline for temporal validation methods")
        print(f"\n📈 Generated visualizations:")
        for viz_type, path in visualization_paths.items():
            print(f"  {viz_type}: {path}")
        
    except Exception as e:
        print(f"\n❌ Random baseline training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("💡 Try checking data format or reducing dataset size.")
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 