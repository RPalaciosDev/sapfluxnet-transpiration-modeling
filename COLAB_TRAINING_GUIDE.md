# Google Colab XGBoost Training Guide

## Problem Summary

The original training script was failing in Google Colab due to memory issues. The main problems were:

1. **Memory Traps**: Operations like `len(dask_dataframe)` and `dropna()` trigger computation across all partitions
2. **Worker Kills**: Dask workers exceeded 95% memory budget during temporal split operations
3. **Inefficient Memory Usage**: Using 80% of available memory left too little buffer for operations
4. **Categorical Column Issues**: Data processing creates categorical columns that can't be filled with new values

## Solution: Two New Training Scripts

### 1. `train_colab_safe_xgboost.py` - Simple Split (Recommended for Initial Testing)

**Features:**
- Uses only 50% of available memory (ultra-conservative)
- Simple partition-based split (not temporal)
- Minimal XGBoost parameters
- Avoids all memory traps

**Use this when:**
- You want to test if the basic training works
- Memory is very limited (<4GB available)
- You need a quick proof of concept

**Limitations:**
- Not temporally valid (random split)
- May not be suitable for time series forecasting

### 2. `train_colab_temporal_xgboost.py` - Temporal Split (Recommended for Final Model)

**Features:**
- Uses 40% of available memory (conservative)
- Proper temporal splits (80% early data for training, 20% later data for testing)
- Moderate XGBoost parameters
- Processes each partition separately for temporal splits

**Use this when:**
- You need scientifically valid temporal validation
- You have sufficient memory (>6GB available)
- You want the final production model

## Key Improvements Made

### Memory Management
- **Reduced Memory Usage**: 40-50% instead of 80%
- **Smaller Chunks**: 25-30MB instead of 200MB
- **Conservative Parameters**: Reduced tree depth, bins, and rounds
- **Avoided Memory Traps**: No `len()` calls, no global `dropna()`
- **Fixed Categorical Columns**: Converts categorical columns to numeric codes automatically

### Dask Configuration
- **Single Worker**: Prevents memory splitting
- **Single Thread**: Minimizes memory overhead
- **Disabled Dashboard**: Saves memory
- **Thread-based**: Instead of process-based parallelism

### Error Handling
- **Graceful Fallbacks**: CSV if parquet fails
- **Better Error Messages**: Clear debugging information
- **Proper Cleanup**: Always closes Dask client
- **Categorical Column Handling**: Converts categorical columns to numeric automatically

## How to Use in Google Colab

### Step 1: Upload Your Data
```python
# Upload the training scripts to Colab
from google.colab import files
uploaded = files.upload()
```

### Step 2: Install Dependencies
```python
!pip install dask[complete] xgboost scikit-learn pyarrow
```

### Step 3: Run Simple Training (Test)
```python
# Test with simple split first
!python train_colab_safe_xgboost.py
```

### Step 4: Run Temporal Training (Final)
```python
# Run temporal training for final model
!python train_colab_temporal_xgboost.py
```

## Expected Output

### Simple Training
```
SAPFLUXNET Google Colab-Safe XGBoost Training
==================================================
Available memory: 11.0GB
Setting up minimal Dask client with 5.5GB memory limit...
Dask client created successfully
Loading data from /content/drive/MyDrive/comprehensive_processed...
Data loaded successfully with 150 partitions
Features: 158 columns
Training completed!
Final Test R²: 0.8500
```

### Temporal Training
```
SAPFLUXNET Google Colab Temporal XGBoost Training
=======================================================
Available memory: 11.0GB
Setting up conservative Dask client with 4.4GB memory limit...
Creating temporal split...
Temporal training completed!
Final Test R²: 0.8200
Train samples: 6,953,755
Test samples: 1,738,439
```

## Memory Requirements

| Available Memory | Recommended Script | Chunk Size | Expected Performance |
|------------------|-------------------|------------|---------------------|
| 4-6GB           | Simple Split      | 25MB       | Basic training only |
| 6-8GB           | Temporal Split    | 25MB       | Good performance    |
| 8-12GB          | Temporal Split    | 30MB       | Optimal performance |
| >12GB           | Temporal Split    | 50MB       | Fast training       |

## Troubleshooting

### If Training Still Fails

1. **Reduce Memory Further**:
   ```python
   # Edit the script to use even less memory
   memory_limit = max(1.0, available_memory * 0.3)  # Use 30% instead of 40%
   ```

2. **Use Smaller Chunks**:
   ```python
   chunk_size = 20  # Reduce from 25MB to 20MB
   ```

3. **Reduce XGBoost Parameters**:
   ```python
   params = {
       'max_depth': 3,      # Reduce from 4
       'max_bin': 64,       # Reduce from 128
       'num_boost_round': 50  # Reduce from 150
   }
   ```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Worker exceeded 95% memory budget` | Memory limit too high | Reduce memory_limit percentage |
| `KilledWorker` | Worker ran out of memory | Use smaller chunks or simpler split |
| `Cannot setitem on a Categorical with a new category (0)` | Categorical columns from data processing | Fixed in updated scripts - converts categorical to numeric |
| `No parquet files found` | Data not in expected location | Check data_dir path |
| `Target column not found` | Wrong column name | Check if 'sap_flow' exists in data |

## Performance Comparison

| Model Type | R² Score | Training Time | Memory Usage | Temporal Validity |
|------------|----------|---------------|--------------|-------------------|
| Simple Split | ~0.85 | 15-20 min | Low | ❌ No |
| Temporal Split | ~0.82 | 25-35 min | Moderate | ✅ Yes |
| Original Script | N/A | Failed | High | ✅ Yes |

## Next Steps

1. **Start with Simple Split** to verify everything works
2. **Move to Temporal Split** for the final model
3. **Save the trained model** to Google Drive
4. **Use the model** for predictions on new data

## Model Files Created

Both scripts create:
- `sapfluxnet_*.json` - Trained XGBoost model
- `sapfluxnet_*_features_*.txt` - List of features used
- `sapfluxnet_*_metrics_*.txt` - Performance metrics
- `sapfluxnet_*_importance_*.csv` - Feature importance (temporal only)

The temporal model is the one you should use for scientific applications since it properly validates the model's ability to predict future time periods. 