# GPU Troubleshooting Guide for Ecosystem Clustering Pipeline

## Quick GPU Check

Run this first to see what's happening:

```bash
python gpu_spatial_validation.py --help
```

The script will show detailed GPU detection output when it starts.

## Common Issues and Solutions

### 1. XGBoost Not Compiled with GPU Support

**Symptom**: "XGBoost not compiled with GPU support" error

**Solution**: Install GPU-enabled XGBoost

```bash
# Uninstall CPU-only version
pip uninstall xgboost

# Install GPU version (CUDA 11.8+)
pip install xgboost[gpu]

# OR via conda (recommended)
conda install -c conda-forge xgboost-gpu
```

### 2. CUDA Toolkit Missing

**Symptom**: GPU detected but training fails with CUDA errors

**Solution**: Install CUDA toolkit

```bash
# Check CUDA version
nvidia-smi

# Install matching CUDA toolkit (example for CUDA 11.8)
conda install cudatoolkit=11.8 -c conda-forge

# OR download from NVIDIA (more complex)
```

### 3. GPU Memory Issues

**Symptom**: "CUDA out of memory" errors

**Solutions**:

```bash
# Reduce GPU memory usage
export CUDA_VISIBLE_DEVICES=0  # Use only first GPU
export XGB_GPU_MEMORY_LIMIT=8000  # Limit to 8GB

# Or modify script parameters for smaller batches
```

### 4. Driver Issues

**Symptom**: nvidia-smi fails or shows old driver

**Solution**: Update NVIDIA drivers

```bash
# Check current driver
nvidia-smi

# Update drivers (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535  # or latest stable

# Reboot after driver update
sudo reboot
```

### 5. Multiple GPU Selection

**Symptom**: Wrong GPU being used

**Solution**: Set specific GPU

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=1  # Use second GPU

# Check which GPUs are available
nvidia-smi -L
```

### 6. Force GPU Mode

If detection fails but you know GPU should work:

```bash
python gpu_spatial_validation.py --force-gpu
```

**Warning**: This bypasses safety checks - use only if you're sure GPU works.

## Environment Variables

Set these before running:

```bash
# Essential
export CUDA_VISIBLE_DEVICES=0
export XGB_GPU_MEMORY_LIMIT=8000

# Optional performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Debugging Commands

### Check CUDA Installation

```bash
# Check CUDA compiler
nvcc --version

# Check CUDA runtime
python -c "import torch; print(torch.cuda.is_available())"
```

### Check XGBoost GPU Build

```python
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")

# Try creating GPU regressor
try:
    model = xgb.XGBRegressor(tree_method='gpu_hist')
    print("✅ XGBoost GPU support confirmed")
except Exception as e:
    print(f"❌ XGBoost GPU issue: {e}")
```

### Manual GPU Test

```python
import xgboost as xgb
import numpy as np

# Create test data
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# Try GPU training
try:
    dtrain = xgb.DMatrix(X, label=y)
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    model = xgb.train(params, dtrain, num_boost_round=10)
    print("✅ Manual GPU test successful")
except Exception as e:
    print(f"❌ Manual GPU test failed: {e}")
```

## Performance Expectations

With GPU acceleration, you should see:

- **3-10x faster training** compared to CPU
- **Higher memory usage** but faster processing
- **More hyperparameter trials** in same time
- **tree_method=gpu_hist** in log output

## Getting Help

If none of these solutions work:

1. **Check your hardware**: `nvidia-smi` should show your GPU
2. **Check CUDA version**: Must match XGBoost requirements
3. **Check XGBoost build**: `pip show xgboost` should show GPU variant
4. **Try force mode**: `--force-gpu` flag as last resort
5. **Fall back to CPU**: Remove GPU flags if needed

## Hardware Requirements

**Minimum**:

- NVIDIA GPU with CUDA Compute Capability 6.0+
- 8GB+ GPU memory
- CUDA 11.0+ toolkit

**Recommended**:

- RTX 3080/4080, Tesla V100, A100, or H100
- 16GB+ GPU memory  
- CUDA 11.8+ toolkit
- Latest NVIDIA drivers
