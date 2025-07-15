# Export Format Options for SAPFLUXNET Pipeline

The pipeline now supports multiple export formats for processed data, each with different benefits for machine learning workflows.

## Supported Formats

### 1. CSV (Default)

- **Extension**: `.csv`
- **Dependencies**: None (built-in)
- **Pros**: Universal compatibility, human-readable
- **Cons**: Slower I/O, larger files, no data type preservation
- **Best for**: Sharing data, compatibility with all tools

### 2. Parquet (Recommended for ML)

- **Extension**: `.parquet`
- **Dependencies**: `pip install pyarrow`
- **Pros**: Fast read/write, excellent compression, column-oriented, preserves data types
- **Cons**: Requires pyarrow library
- **Best for**: Machine learning workflows, big data processing

### 3. Feather

- **Extension**: `.feather`
- **Dependencies**: `pip install pyarrow`
- **Pros**: Fastest I/O speeds, good for temporary data
- **Cons**: Larger files than parquet
- **Best for**: Fast data exchange, temporary storage

### 4. HDF5

- **Extension**: `.h5`
- **Dependencies**: `pip install tables`
- **Pros**: Excellent compression, supports multiple datasets
- **Cons**: Slower than parquet/feather
- **Best for**: Complex data structures, maximum compression

### 5. Pickle

- **Extension**: `.pkl`
- **Dependencies**: None (built-in)
- **Pros**: Preserves Python objects exactly
- **Cons**: Python-only, security concerns
- **Best for**: Python-specific workflows

### 6. LibSVM (XGBoost External Memory)

- **Extension**: `.svm`
- **Dependencies**: `pip install scikit-learn`
- **Pros**: Optimized for XGBoost external memory, very memory efficient, sparse format
- **Cons**: Limited to numeric features, not human-readable, ML-specific
- **Best for**: XGBoost external memory training, large datasets with memory constraints

## Usage

### Command Line

```bash
# Export as Parquet (recommended for ML) - saves to processed_parquet/
python comprehensive_processing_pipeline.py --export-format parquet

# Export as Feather (fastest) - saves to processed_feather/
python comprehensive_processing_pipeline.py --export-format feather

# Export as HDF5 (best compression) - saves to processed_hdf5/
python comprehensive_processing_pipeline.py --export-format hdf5

# Export as Pickle (Python-specific) - saves to processed_pickle/
python comprehensive_processing_pipeline.py --export-format pickle

# Export as LibSVM (XGBoost external memory) - saves to processed_libsvm/
python comprehensive_processing_pipeline.py --export-format libsvm

# Default CSV export - saves to processed_csv/
python comprehensive_processing_pipeline.py --export-format csv

# Custom output directory with format suffix
python comprehensive_processing_pipeline.py --export-format parquet --output-dir my_data
# Creates: my_data_parquet/
```

### Python API

```python
from comprehensive_processing_pipeline import MemoryEfficientSAPFLUXNETProcessor

# Create processor with specific export format
processor = MemoryEfficientSAPFLUXNETProcessor(
    export_format='parquet',  # or 'feather', 'hdf5', 'pickle', 'libsvm', 'csv'
    compress_output=True      # compression works with all formats
)

# Process all sites
processor.process_all_sites()
```

## Reading Data in ML Scripts

The training script (`simple_xgboost_training.py`) automatically detects and loads data in any supported format:

```python
# The load_data function automatically handles all formats
# Note: Use format-specific directory names
data = load_data('processed_parquet')  # For parquet files
data = load_data('processed_csv')      # For CSV files
data = load_data('processed_libsvm')   # For libsvm files
```

### Manual Loading

```python
import pandas as pd
from sklearn.datasets import load_svmlight_file

# Parquet (recommended)
data = pd.read_parquet('site_comprehensive.parquet')

# Feather
data = pd.read_feather('site_comprehensive.feather')

# HDF5
data = pd.read_hdf('site_comprehensive.h5', key='data')

# Pickle
data = pd.read_pickle('site_comprehensive.pkl')

# LibSVM (for XGBoost external memory)
X, y = load_svmlight_file('site_comprehensive.svm')

# CSV
data = pd.read_csv('site_comprehensive.csv')
```

## Performance Comparison

Run the comparison script to see performance differences:

```bash
python export_format_comparison.py
```

Typical results:

- **Feather**: Fastest read/write
- **Parquet**: Best balance of speed and compression
- **HDF5**: Best compression
- **CSV**: Slowest but most compatible
- **Pickle**: Fast but Python-only

## Recommendations

### For Machine Learning

**Use Parquet** - Best overall performance, good compression, excellent pandas integration

### For Fast I/O

**Use Feather** - Fastest read/write speeds for temporary data

### For Maximum Compression

**Use HDF5** - Best compression ratios for storage

### For Universal Compatibility

**Use CSV** - Works with all tools and languages

### For Python-Only Workflows

**Use Pickle** - Preserves exact Python objects

### For XGBoost External Memory Training

**Use LibSVM** - Optimized for memory-efficient XGBoost training on large datasets

## Installation

Install required dependencies:

```bash
# For Parquet and Feather
pip install pyarrow

# For HDF5
pip install tables

# For LibSVM
pip install scikit-learn

# CSV and Pickle are built-in
```

## Directory Structure and File Naming

### Directory Structure

Format-specific directories are automatically created:

- **CSV**: `processed_csv/`
- **Parquet**: `processed_parquet/`
- **Feather**: `processed_feather/`
- **HDF5**: `processed_hdf5/`
- **Pickle**: `processed_pickle/`
- **LibSVM**: `processed_libsvm/`

### File Naming Convention

All processed files follow the pattern:

- `{directory}/{site}_comprehensive.{extension}`

Examples:

- `processed_parquet/ARG_MAZ_comprehensive.parquet`
- `processed_feather/AUS_MAR_UBW_comprehensive.feather`
- `processed_hdf5/FRA_PUE_comprehensive.h5`
- `processed_pickle/USA_UMB_CON_comprehensive.pkl`
- `processed_libsvm/ESP_CAN_comprehensive.svm`
- `processed_csv/ESP_CAN_comprehensive.csv`

## Compression

Compression is supported for all formats:

- **CSV**: gzip compression
- **Parquet**: snappy compression
- **Feather**: lz4 compression
- **HDF5**: built-in compression
- **Pickle**: gzip compression
- **LibSVM**: gzip compression

Enable compression with `--compress` flag or `compress_output=True` parameter.
