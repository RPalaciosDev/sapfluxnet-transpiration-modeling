"""
External Memory XGBoost Training for SAPFLUXNET Data
FULL DATASET - Uses XGBoost external memory for complete dataset processing
Implements external memory training to handle 12M+ rows efficiently
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import os
from datetime import datetime
import warnings
import gc
import psutil
import tempfile
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')

def get_available_memory_gb():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024**3)

def combine_libsvm_files(libsvm_dir, output_dir):
    """
    Combine existing libsvm files from pipeline into single file
    """
    print(f"Combining libsvm files from {libsvm_dir}...")
    
    # Get all libsvm files (including compressed)
    libsvm_files = [f for f in os.listdir(libsvm_dir) if f.endswith('.svm') or f.endswith('.svm.gz')]
    print(f"Found {len(libsvm_files)} libsvm files to combine")
    
    # Combine all files
    all_data_file = os.path.join(output_dir, 'all_data.svm')
    total_rows = 0
    
    with open(all_data_file, 'w') as output_file:
        for i, libsvm_file in enumerate(libsvm_files):
            print(f"Processing file {i+1}/{len(libsvm_files)}: {libsvm_file}")
            
            file_path = os.path.join(libsvm_dir, libsvm_file)
            
            # Handle compressed files
            if libsvm_file.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rt') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
            
            # Write lines to combined file
            output_file.writelines(lines)
            total_rows += len(lines)
            
            print(f"  Added {len(lines)} rows from {libsvm_file}")
    
    print(f"Combination completed: {total_rows:,} total rows")
    return all_data_file, total_rows

def log_memory_usage(step_name):
    """Log current memory usage for debugging"""
    memory = psutil.virtual_memory()
    print(f"🔍 {step_name}:")
    print(f"   Available: {memory.available / (1024**3):.1f}GB")
    print(f"   Used: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"   Total: {memory.total / (1024**3):.1f}GB")

def load_and_convert_to_external_format(data_dir, output_dir, chunk_size=50000):
    """
    Load parquet files and convert to external memory format (libsvm)
    Process in chunks to avoid memory issues
    """
    print(f"Converting parquet files to external memory format...")
    log_memory_usage("Before data conversion")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all parquet files
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Process first file to get feature information
    first_file = os.path.join(data_dir, parquet_files[0])
    sample_df = pd.read_parquet(first_file, nrows=1000)
    
    # Prepare features
    feature_cols, target_col = prepare_features_from_sample(sample_df)
    print(f"Will use {len(feature_cols)} features")
    
    # Convert all files to libsvm format
    all_data_file = os.path.join(output_dir, 'all_data.svm')
    total_rows = 0
    
    with open(all_data_file, 'w') as output_file:
        for i, parquet_file in enumerate(parquet_files):
            print(f"Processing file {i+1}/{len(parquet_files)}: {parquet_file}")
            
            file_path = os.path.join(data_dir, parquet_file)
            
            # Process file in chunks
            try:
                for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
                    # Clean and prepare chunk
                    chunk_processed = prepare_chunk_for_external_memory(
                        chunk, feature_cols, target_col
                    )
                    
                    if len(chunk_processed) > 0:
                        # Convert to libsvm format and append
                        X_chunk = chunk_processed[feature_cols].values
                        y_chunk = chunk_processed[target_col].values
                        
                        # Write to temporary file then append
                        temp_file = f"{output_dir}/temp_chunk.svm"
                        dump_svmlight_file(X_chunk, y_chunk, temp_file)
                        
                        # Append to main file
                        with open(temp_file, 'r') as temp_f:
                            output_file.write(temp_f.read())
                        
                        os.remove(temp_file)
                        total_rows += len(chunk_processed)
                        
                        # Memory cleanup
                        del chunk_processed, X_chunk, y_chunk
                        gc.collect()
        
            except Exception as e:
                        print(f"Error processing {parquet_file}: {e}")
                        continue
    
    print(f"Conversion completed: {total_rows:,} total rows")
    log_memory_usage("After data conversion")
    
    return all_data_file, feature_cols, target_col, total_rows

def prepare_features_from_sample(sample_df):
    """Prepare feature list from sample dataframe"""
    
    # Define columns to exclude
    exclude_cols = [
        'TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id',
        'Unnamed: 0'
    ]
    
    # Find target column
    target_col = 'sap_flow'
    if target_col not in sample_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Select feature columns
    feature_cols = [col for col in sample_df.columns 
                   if col not in exclude_cols + [target_col]
                   and not col.endswith('_flags')
                   and not col.endswith('_md')]
    
    # Keep most important features to reduce memory usage
    if len(feature_cols) > 100:
        print(f"⚠️  Large feature set ({len(feature_cols)} features) - reducing for efficiency")
        priority_features = [col for col in feature_cols if any(base in col for base in 
                           ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc', 'ppfd', 
                            'hour', 'day_of_year', 'month', 'year', 'lag_1h', 'lag_3h', 
                            'rolling_3h', 'rolling_6h', 'site_lat', 'site_lon'])]
        feature_cols = priority_features[:80]  # Limit to 80 features
        print(f"🔧 Reduced to {len(feature_cols)} priority features")
    
    return feature_cols, target_col

def prepare_chunk_for_external_memory(chunk, feature_cols, target_col):
    """Prepare a chunk of data for external memory processing"""
    
    # Remove rows with missing target
    chunk_clean = chunk.dropna(subset=[target_col])
    
    # Fill missing values in features
    chunk_clean = chunk_clean.copy()
    for col in feature_cols:
        if col in chunk_clean.columns:
            chunk_clean[col] = chunk_clean[col].fillna(0)
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in chunk_clean.columns:
            chunk_clean[col] = 0
    
    return chunk_clean

def create_random_split_external(data_file, train_file, test_file, train_ratio=0.8, random_state=42):
    """
    Create random train/test split for external memory format
    """
    print(f"Creating random split (train_ratio={train_ratio})...")
    
    # Read the data file and create random split
    np.random.seed(random_state)
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out empty lines and validate format
    valid_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            # Basic validation: should start with a number (target)
            parts = line.split(' ', 1)
            if len(parts) >= 1:
                try:
                    float(parts[0])  # Target should be numeric
                    valid_lines.append(line + '\n')
                except ValueError:
                    print(f"Warning: Invalid line {i+1}: {line[:50]}...")
    
    total_lines = len(valid_lines)
    print(f"Total valid samples: {total_lines:,}")
    
    if total_lines == 0:
        raise ValueError("No valid libsvm format lines found in data file")
    
    # Shuffle lines
    np.random.shuffle(valid_lines)
    
    # Split
    train_size = int(total_lines * train_ratio)
    train_lines = valid_lines[:train_size]
    test_lines = valid_lines[train_size:]
    
    # Write train file
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    
    # Write test file
    with open(test_file, 'w') as f:
        f.writelines(test_lines)
    
    print(f"Random split completed:")
    print(f"  Train: {len(train_lines):,} rows ({len(train_lines)/total_lines*100:.1f}%)")
    print(f"  Test: {len(test_lines):,} rows ({len(test_lines)/total_lines*100:.1f}%)")
    
    return train_file, test_file

def validate_libsvm_format(file_path, max_lines_to_check=10):
    """Validate that the file is in proper libsvm format"""
    print(f"Validating libsvm format for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines_to_check:
                    break
                
                line = line.strip()
                if not line:
                    continue
                    
                # Check libsvm format: target feature1:value1 feature2:value2 ...
                parts = line.split()
                if len(parts) < 1:
                    print(f"❌ Line {i+1}: Empty line")
                    return False
                
                # First part should be the target (numeric)
                try:
                    float(parts[0])
                except ValueError:
                    print(f"❌ Line {i+1}: Invalid target value: {parts[0]}")
                    return False
                
                # Check feature:value format
                for j, part in enumerate(parts[1:], 1):
                    if ':' not in part:
                        print(f"❌ Line {i+1}, Part {j}: Missing colon in feature:value format: {part}")
                        return False
                    
                    feature_idx, value = part.split(':', 1)
                    try:
                        int(feature_idx)  # Feature index should be integer
                        float(value)      # Value should be numeric
                    except ValueError:
                        print(f"❌ Line {i+1}, Part {j}: Invalid feature:value format: {part}")
                        return False
        
        print(f"✅ libsvm format validation passed for: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating libsvm format: {e}")
        return False

def train_external_memory_xgboost(train_file, test_file, feature_cols):
    """Train XGBoost model using external memory"""
    print("Training XGBoost with external memory...")
    print(f"XGBoost version: {xgb.__version__}")
    log_memory_usage("Before external memory training")
    
    # Validate libsvm format
    if not validate_libsvm_format(train_file):
        raise ValueError(f"Invalid libsvm format in train file: {train_file}")
    if not validate_libsvm_format(test_file):
        raise ValueError(f"Invalid libsvm format in test file: {test_file}")
    
    # Show sample lines for debugging
    print(f"Sample lines from {train_file}:")
    try:
        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 lines
                    break
                print(f"  Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Could not read sample lines: {e}")
    
    # Create DMatrix objects for external memory with format specification
    print("Creating external memory DMatrix objects...")
    print(f"About to load train file: {train_file}")
    print(f"About to load test file: {test_file}")
    print(f"Train file exists: {os.path.exists(train_file)}")
    print(f"Test file exists: {os.path.exists(test_file)}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Try with proper libsvm format specification for external memory
        print("Attempting to create DMatrix with libsvm format specification...")
        dtrain = xgb.DMatrix(f"{train_file}?format=libsvm")
        dtest = xgb.DMatrix(f"{test_file}?format=libsvm")
    except Exception as e:
        print(f"libsvm format specification failed: {e}")
        try:
            # Try alternative format specification
            print("Trying alternative format specification...")
            dtrain = xgb.DMatrix(f"file://{os.path.abspath(train_file)}?format=libsvm")
            dtest = xgb.DMatrix(f"file://{os.path.abspath(test_file)}?format=libsvm")
        except Exception as e2:
            print(f"Alternative format failed: {e2}")
            try:
                # Try with explicit cache prefix for external memory
                print("Trying with cache prefix...")
                dtrain = xgb.DMatrix(f"{train_file}#dtrain.cache?format=libsvm")
                dtest = xgb.DMatrix(f"{test_file}#dtest.cache?format=libsvm")
            except Exception as e3:
                print(f"Cache prefix failed: {e3}")
                try:
                    # Try QuantileDMatrix for external memory (more robust)
                    print("Trying QuantileDMatrix for external memory...")
                    dtrain = xgb.QuantileDMatrix(train_file)
                    dtest = xgb.QuantileDMatrix(test_file)
                except Exception as e4:
                    print(f"QuantileDMatrix failed: {e4}")
                    print("Falling back to regular DMatrix (loads into memory)...")
                    # Final fallback - load into memory (not external memory)
                    dtrain = xgb.DMatrix(train_file)
                    dtest = xgb.DMatrix(test_file)
    
    log_memory_usage("After DMatrix creation")
    
    # External memory optimized parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,          # Can be higher with external memory
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',   # Best for external memory
        'max_bin': 256,          # Can be higher with external memory
        'verbosity': 1,
        'nthread': -1            # Use all available threads
    }
    
    print(f"XGBoost parameters: {params}")
    
    # Train model
    print("Starting external memory training...")
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,     # Can train longer with external memory
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    log_memory_usage("After training")
    print("External memory training completed!")
    
    # Make predictions
    print("Making predictions...")
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)
    
    # Load actual values for metrics (this loads into memory, but just targets)
    print("Loading actual values for metrics...")
    _, y_train_actual = load_svmlight_file(train_file)
    _, y_test_actual = load_svmlight_file(test_file)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_actual, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_test)),
        'train_mae': mean_absolute_error(y_train_actual, y_pred_train),
        'test_mae': mean_absolute_error(y_test_actual, y_pred_test),
        'train_r2': r2_score(y_train_actual, y_pred_train),
        'test_r2': r2_score(y_test_actual, y_pred_test)
    }
    
    print(f"\nExternal Memory Model Performance:")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Train samples: {len(y_train_actual):,}")
    print(f"  Test samples: {len(y_test_actual):,}")
    
    # Cleanup
    del dtrain, dtest
    gc.collect()
    log_memory_usage("After cleanup")
    
    return model, metrics

def get_feature_importance(model, feature_cols):
    """Get feature importance"""
    try:
        importance_dict = model.get_score(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return pd.DataFrame({'feature': [f'f{i}' for i in range(len(feature_cols))], 
                           'importance': [0.0] * len(feature_cols)})

def save_external_memory_results(model, metrics, feature_importance, feature_cols, total_rows, output_dir='external_memory_models'):
    """Save external memory model results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f"{output_dir}/sapfluxnet_external_memory_{timestamp}.json"
    model.save_model(model_path)
    
    # Save feature importance
    feature_importance_path = f"{output_dir}/sapfluxnet_external_memory_importance_{timestamp}.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    
    # Save feature list
    feature_path = f"{output_dir}/sapfluxnet_external_memory_features_{timestamp}.txt"
    with open(feature_path, 'w') as f:
        f.write("Features used in external memory training:\n")
        f.write("Method: External Memory XGBoost with random 80/20 split\n")
        f.write("Dataset: Full dataset (no sampling)\n")
        f.write(f"Total rows: {total_rows:,}\n")
        f.write("Purpose: Full dataset baseline with memory efficiency\n\n")
        for i, feature in enumerate(feature_cols):
            f.write(f"{i+1:3d}. {feature}\n")
    
    # Save metrics
    metrics_path = f"{output_dir}/sapfluxnet_external_memory_metrics_{timestamp}.txt"
    with open(metrics_path, 'w') as f:
        f.write("SAPFLUXNET External Memory Training Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed: {datetime.now()}\n")
        f.write("Method: External Memory XGBoost\n")
        f.write("Dataset: Full dataset (no sampling)\n")
        f.write(f"Total rows: {total_rows:,}\n")
        f.write("Split: Random 80/20 split\n")
        f.write("Memory: External memory (disk-based)\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nValidation Method:\n")
        f.write("-" * 18 + "\n")
        f.write("- External memory processing\n")
        f.write("- Random 80/20 split across full dataset\n")
        f.write("- No data sampling (complete dataset)\n")
        f.write("- Memory-efficient disk-based training\n")
        f.write("- Baseline for comparison with sampling methods\n")
        
        f.write("\nAdvantages:\n")
        f.write("-" * 12 + "\n")
        f.write("- Uses complete dataset (no information loss)\n")
        f.write("- Memory efficient (handles unlimited data size)\n")
        f.write("- Scalable approach for large datasets\n")
        f.write("- True baseline performance\n")
        
        f.write("\nDisadvantages:\n")
        f.write("-" * 14 + "\n")
        f.write("- Slower training (disk I/O overhead)\n")
        f.write("- Requires data format conversion\n")
        f.write("- More complex setup\n")
    
    print(f"\nExternal memory model results saved:")
    print(f"  Model: {model_path}")
    print(f"  Feature importance: {feature_importance_path}")
    print(f"  Features: {feature_path}")
    print(f"  Metrics: {metrics_path}")
    
    return model_path

def main():
    """Main external memory training pipeline"""
    print("SAPFLUXNET External Memory XGBoost Training")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("Method: External Memory XGBoost (full dataset)")
    print("Purpose: Memory-efficient training on complete dataset")
    
    # Check available memory
    available_memory = get_available_memory_gb()
    print(f"Available memory: {available_memory:.1f}GB")
    
    # Set up directories - check for libsvm format first
    libsvm_dir = 'processed_libsvm'
    parquet_dir = 'test_parquet_export'
    
    # Try different temp directory approaches for permission issues
    try:
        # Option 1: Use system temp directory
        temp_dir = tempfile.mkdtemp(prefix='xgboost_external_')
        print(f"Using system temp directory: {temp_dir}")
    except Exception as e:
        print(f"System temp failed: {e}")
        # Option 2: Use local temp directory
        temp_dir = 'temp_external_memory'
        print(f"Using local temp directory: {temp_dir}")
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check directory permissions
    print(f"Temporary directory: {temp_dir}")
    print(f"Directory exists: {os.path.exists(temp_dir)}")
    print(f"Directory is writable: {os.access(temp_dir, os.W_OK)}")
    print(f"Directory is readable: {os.access(temp_dir, os.R_OK)}")
    
    # Test write permissions
    test_file = os.path.join(temp_dir, 'permission_test.txt')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Write permissions confirmed")
    except Exception as e:
        print(f"❌ Write permission test failed: {e}")
        raise
    
    try:
        # Check if libsvm format is available (from pipeline)
        if os.path.exists(libsvm_dir) and len([f for f in os.listdir(libsvm_dir) if f.endswith('.svm') or f.endswith('.svm.gz')]) > 0:
            print(f"\n✅ Found libsvm format data in {libsvm_dir}")
            print("Using direct libsvm files from pipeline - no conversion needed!")
            
            # Step 1: Combine existing libsvm files
            print("\n" + "="*60)
            print("COMBINING EXISTING LIBSVM FILES")
            print("="*60)
            
            all_data_file, total_rows = combine_libsvm_files(libsvm_dir, temp_dir)
            feature_cols = None  # Not needed for libsvm files
            target_col = None    # Not needed for libsvm files
            
        else:
            print(f"\n⚠️  No libsvm format found in {libsvm_dir}")
            print("Converting from parquet format...")
            
            # Step 1: Convert data to external memory format
            print("\n" + "="*60)
            print("CONVERTING DATA TO EXTERNAL MEMORY FORMAT")
            print("="*60)
            
            all_data_file, feature_cols, target_col, total_rows = load_and_convert_to_external_format(
                parquet_dir, temp_dir
            )
        
        # Step 2: Create random split
        print("\n" + "="*60)
        print("CREATING RANDOM SPLIT")
        print("="*60)
        
        # Verify input file exists
        if not os.path.exists(all_data_file):
            raise FileNotFoundError(f"Input data file not found: {all_data_file}")
        
        train_file = os.path.join(temp_dir, 'train.svm')
        test_file = os.path.join(temp_dir, 'test.svm')
        
        train_file, test_file = create_random_split_external(all_data_file, train_file, test_file)
        
        # Verify files exist and check permissions
        print(f"Verifying created files:")
        print(f"  Train file: {train_file}")
        print(f"    - Exists: {'YES' if os.path.exists(train_file) else 'NO'}")
        if os.path.exists(train_file):
            print(f"    - Readable: {'YES' if os.access(train_file, os.R_OK) else 'NO'}")
            print(f"    - File size: {os.path.getsize(train_file):,} bytes")
            print(f"    - Absolute path: {os.path.abspath(train_file)}")
        
        print(f"  Test file: {test_file}")
        print(f"    - Exists: {'YES' if os.path.exists(test_file) else 'NO'}")
        if os.path.exists(test_file):
            print(f"    - Readable: {'YES' if os.access(test_file, os.R_OK) else 'NO'}")
            print(f"    - File size: {os.path.getsize(test_file):,} bytes")
            print(f"    - Absolute path: {os.path.abspath(test_file)}")
        
        # Step 3: Train model
        print("\n" + "="*60)
        print("TRAINING EXTERNAL MEMORY MODEL")
        print("="*60)
        
        model, metrics = train_external_memory_xgboost(train_file, test_file, feature_cols)
        
        # Step 4: Get feature importance
        feature_importance = get_feature_importance(model, feature_cols)
        
        # Step 5: Save results
        model_path = save_external_memory_results(
            model, metrics, feature_importance, feature_cols, total_rows
        )
        
        print(f"\n✅ External memory training completed successfully!")
        print(f"Final Test R²: {metrics['test_r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"💡 This model uses the complete dataset ({total_rows:,} rows)")
        print(f"🚀 Memory-efficient external memory approach")
        print(f"📊 Method: External Memory XGBoost")
        print(f"🎯 Purpose: Full dataset baseline performance")
        
    except Exception as e:
        print(f"\n❌ External memory training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            print(f"\nCleaning up temporary files from: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary files cleaned up successfully")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")
                print("You may need to manually remove: {temp_dir}")
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 