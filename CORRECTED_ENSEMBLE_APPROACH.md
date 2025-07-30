# Corrected Ensemble Approach: Pre-Split Site Assignment

## 🚨 **Problem with Previous Approach**

The original novel site ensemble testing had a **critical flaw**: it was randomly splitting sites **after** models were already trained, which created **data leakage**. Models were trained on data that included sites later used for testing.

## ✅ **Corrected Approach: Pre-Split Site Assignment**

### **Step 1: Data Pipeline with Site Splitting**

```bash
# Run data pipeline with site splitting BEFORE any clustering or training
python data_pipeline_v3.py \
    --export-format parquet \
    --test-fraction 0.2 \
    --random-seed 42 \
    --clean-mode
```

**What this does:**

- ✅ **Splits sites BEFORE processing**: 80% train, 20% test
- ✅ **Saves split assignment**: `site_split_assignment.json`
- ✅ **No data leakage**: Test sites never used in training
- ✅ **Reproducible**: Fixed random seed ensures consistent splits

### **Step 2: Performance-Based Clustering (Train Sites Only)**

```bash
# Cluster only the training sites (80% of sites)
python ecosystem/clustering/clustering_v3_outlier_filtered.py \
    --feature-set performance
```

**What this does:**

- ✅ **Uses only train sites**: No test sites in clustering
- ✅ **Performance-based features**: Groups sites by prediction characteristics
- ✅ **Creates cluster assignments**: For train sites only

### **Step 3: Train Cluster Models (Train Sites Only)**

```bash
# Train models using only train sites
python ecosystem/models/train_cluster_models.py
```

**What this does:**

- ✅ **Uses only train sites**: Models never see test data
- ✅ **Cluster-specific models**: Each cluster gets its own model
- ✅ **No data leakage**: Complete separation of train/test

### **Step 4: Novel Site Ensemble Testing (Test Sites Only)**

```bash
# Test ensemble on withheld test sites
python ecosystem/models/novel_site_ensemble_testing.py
```

**What this does:**

- ✅ **Uses only test sites**: Sites never seen during training
- ✅ **Tests all cluster models**: Each model predicts all test sites
- ✅ **Ensemble strategies**: Three different weighting approaches
- ✅ **True generalization**: Tests real-world performance

## 🔄 **Complete Pipeline Flow**

```
1. Data Pipeline (with site splitting)
   ├── Split sites: 80% train, 20% test
   ├── Save: site_split_assignment.json
   └── Process: All sites with features

2. Clustering (train sites only)
   ├── Load: train sites from split
   ├── Cluster: Performance-based grouping
   └── Save: cluster assignments

3. Model Training (train sites only)
   ├── Load: train sites + cluster assignments
   ├── Train: Cluster-specific models
   └── Save: Trained models

4. Ensemble Testing (test sites only)
   ├── Load: test sites (never seen before)
   ├── Test: All models on all test sites
   ├── Ensemble: Three weighting strategies
   └── Compare: Individual vs. ensemble performance
```

## 📊 **Key Benefits of Corrected Approach**

### **No Data Leakage**

- ✅ Test sites completely isolated from training
- ✅ True generalization performance
- ✅ Reliable ensemble evaluation

### **Proper Validation**

- ✅ Models trained on subset of data
- ✅ Tested on completely unseen sites
- ✅ Realistic performance estimates

### **Reproducible Results**

- ✅ Fixed random seed for site splitting
- ✅ Consistent train/test assignments
- ✅ Reliable comparisons across runs

## 🚀 **Usage Commands**

### **Complete Pipeline (Recommended)**

```bash
# 1. Data pipeline with site splitting
python data_pipeline_v3.py --export-format parquet --test-fraction 0.2 --random-seed 42 --clean-mode

# 2. Performance-based clustering
python ecosystem/clustering/clustering_v3_outlier_filtered.py --feature-set performance

# 3. Train cluster models
python ecosystem/models/train_cluster_models.py

# 4. Test ensemble on novel sites
python ecosystem/models/novel_site_ensemble_testing.py
```

### **Custom Parameters**

```bash
# Different test fraction
python data_pipeline_v3.py --test-fraction 0.3 --random-seed 123

# Different clustering approach
python ecosystem/clustering/clustering_v3_outlier_filtered.py --feature-set hybrid

# Custom ensemble testing
python ecosystem/models/novel_site_ensemble_testing.py --site-split-file custom_split.json
```

## 📁 **Generated Files**

### **Data Pipeline Output**

- `processed_parquet/` - Processed data files
- `site_split_assignment.json` - **Critical**: Train/test site assignments

### **Clustering Output**

- `ecosystem/evaluation/clustering_results/` - Cluster assignments (train sites only)

### **Model Training Output**

- `ecosystem/models/results/cluster_models/` - Trained models (train sites only)

### **Ensemble Testing Output**

- `ecosystem/models/results/novel_site_ensemble/` - Ensemble performance results

## 🎯 **Expected Results**

### **Proper Validation Metrics**

- **Individual models**: Performance on completely unseen sites
- **Ensemble models**: Weighted combinations of individual models
- **True generalization**: Real-world performance estimates

### **Success Criteria**

- ✅ Ensemble outperforms individual models
- ✅ Consistent improvement across weighting strategies
- ✅ Robust performance on novel sites

## 🔧 **Troubleshooting**

### **Missing Site Split File**

```bash
# Error: "No site split file found"
# Solution: Run data pipeline with --test-fraction first
python data_pipeline_v3.py --test-fraction 0.2 --export-format parquet
```

### **Inconsistent Results**

```bash
# Error: Different results on each run
# Solution: Use fixed random seed
python data_pipeline_v3.py --test-fraction 0.2 --random-seed 42
```

### **Memory Issues**

```bash
# Error: Out of memory during processing
# Solution: Reduce test fraction or use smaller chunk size
python data_pipeline_v3.py --test-fraction 0.1 --chunk-size 500
```

This corrected approach ensures **proper validation** and **no data leakage**, providing reliable ensemble performance evaluation.
