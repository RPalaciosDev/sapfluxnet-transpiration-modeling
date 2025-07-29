# Python Environment Setup Instructions

## For Ecosystem Clustering and Spatial Validation Pipeline

### Option 1: Full Installation (Recommended)

```bash
# Install all packages including visualization and optional utilities
pip install -r requirements.txt
```

### Option 2: Minimal Installation

```bash
# Install only essential packages for core functionality
pip install -r requirements-minimal.txt
```

### Option 3: Conda Environment (Recommended for HPC)

```bash
# Create a new conda environment
conda create -n ecosystem-modeling python=3.9
conda activate ecosystem-modeling

# Install packages via conda where possible (better for HPC)
conda install pandas numpy scipy scikit-learn matplotlib seaborn
conda install -c conda-forge xgboost optuna pyarrow psutil

# Install remaining packages via pip
pip install geopy numba statsmodels tqdm python-dateutil
```

### Verification

After installation, verify everything works:

```python
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import optuna
import pyarrow.parquet as pq

print("✅ All essential packages imported successfully!")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgb.__version__}")
print(f"optuna: {optuna.__version__}")
```

### Memory Requirements

- **Minimum RAM**: 16GB
- **Recommended RAM**: 64GB+
- **For full dataset**: 128GB+ (adaptive streaming will handle lower memory)

### System Requirements

- **Python**: 3.8+ (3.9 recommended)
- **OS**: Linux (recommended for HPC), macOS, or Windows
- **Cores**: 8+ (for parallel processing)
- **Storage**: 50GB+ free space for results

### Option 4: GPU-Accelerated Setup (High Performance)

**Prerequisites:**

- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+ installed

```bash
# Create conda environment with CUDA support
conda create -n ecosystem-gpu python=3.9
conda activate ecosystem-gpu

# Install CUDA toolkit first
conda install cudatoolkit=11.8 -c conda-forge

# Install XGBoost with GPU support
conda install -c conda-forge xgboost-gpu

# Install other packages
conda install pandas numpy scipy scikit-learn matplotlib seaborn pyarrow
pip install optuna psutil tqdm

# Optional: GPU DataFrame processing (significant speedup)
conda install -c rapidsai cudf cuml
```

**Verify GPU Setup:**

```python
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
print("GPU support:", xgb.gpu.is_supported())

# Test GPU training
import numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20)
dtrain = xgb.DMatrix(X, label=y)
params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
model = xgb.train(params, dtrain, num_boost_round=10)
print("✅ GPU training successful!")
```

### Memory Recommendations by Setup

**CPU-Only:**

- Minimum: 16GB RAM
- Recommended: 64GB+ RAM
- Full dataset: 128GB+ RAM

**GPU-Accelerated:**

- System RAM: 32-64GB (reduced)
- GPU Memory: 8GB minimum, 16GB+ recommended
- **Best**: RTX 4090 (24GB), A6000 (48GB), A100 (40/80GB)

### Troubleshooting

- If XGBoost fails to install: `conda install -c conda-forge xgboost`
- If GPU XGBoost fails: `conda install -c conda-forge xgboost-gpu`
- If CUDA issues: Verify CUDA version with `nvidia-smi`
- If Optuna fails: `pip install --upgrade optuna`
- For memory issues: Ensure you have enough RAM or enable swap
- For permission issues: Use `--user` flag: `pip install --user -r requirements.txt`
