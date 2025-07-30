# Ensemble Plan Reference Document

## 🎯 **Core Ensemble Strategy**

The goal is to create an ensemble of cluster-specific models that can predict novel sites better than individual models by leveraging weighted predictions from multiple specialized models.

## 📋 **4-Step Pipeline Overview**

### **Step 1: Data Splitting (80/20 Split)**

- **Purpose**: Prevent data leakage by separating train/test before any processing
- **Implementation**:
  - 80% of sites → Training set
  - 20% of sites → Test set (completely withheld)
- **Critical**: Split happens **before** clustering or model training
- **Output**: `site_split_assignment.json` with reproducible train/test assignments

### **Step 2: Performance-Based Clustering (Train Sites Only)**

- **Purpose**: Group training sites by prediction characteristics, not ecological similarity
- **Implementation**:
  - Use **only the 80% training sites** for clustering
  - Calculate performance features: sap flow characteristics, data quality, prediction difficulty
  - Create clusters based on sites that are "predictably similar"
- **Output**: Cluster assignments for training sites only

### **Step 3: Cluster Model Training (Train Sites Only)**

- **Purpose**: Train specialized models for each cluster's prediction characteristics
- **Implementation**:
  - Train **separate XGBoost models** for each cluster
  - Use **only training sites** within each cluster
  - Each model becomes specialized for its cluster's prediction patterns
- **Output**: Multiple trained models (one per cluster)

### **Step 4: Ensemble Testing (Test Sites Only)**

- **Purpose**: Test ensemble performance on completely novel sites
- **Implementation**:
  - Take **20% withheld test sites** (never seen during training/clustering)
  - Run **all cluster models** on each test site
  - Create **weighted ensemble predictions** using multiple strategies
  - Compare ensemble vs. individual model performance
- **Output**: Ensemble performance results and best weighting strategy

## 🔧 **Key Implementation Details**

### **No Data Leakage Principle**

- Test sites are **completely isolated** from training process
- Clustering uses **only training sites**
- Models trained on **only training sites**
- Testing uses **only withheld test sites**

### **Ensemble Weighting Strategies**

1. **Confidence-Weighted**: Models with higher prediction confidence get more weight
2. **Historical Performance**: Models with better training performance get more weight  
3. **Distance-Weighted**: Models from clusters more similar to test site get more weight

### **Expected Outcome**

- **Ensemble R² > Individual Model R²**: Weighted combination should outperform single models
- **Better Generalization**: Ensemble should handle diverse novel sites better than any single cluster model
- **Robust Predictions**: Multiple specialized models provide more stable predictions

## 🎯 **Success Criteria**

### **Primary Goal**

- Ensemble predictions on test sites outperform best individual cluster model

### **Secondary Goals**

- At least one weighting strategy shows consistent improvement
- Ensemble works across different types of novel sites
- Results are reproducible with fixed random seed

## 🚨 **Critical Requirements**

### **Data Leakage Prevention**

- ✅ Sites split before any processing
- ✅ Test sites never used in clustering
- ✅ Test sites never used in model training
- ✅ Complete separation maintained throughout pipeline

### **Reproducibility**

- ✅ Fixed random seed for site splitting
- ✅ Consistent train/test assignments across runs
- ✅ Deterministic clustering and model training

### **Performance Validation**

- ✅ True novel site testing (sites never seen before)
- ✅ Multiple weighting strategies compared
- ✅ Statistical significance of ensemble improvement

## 📁 **File Structure and Outputs**

### **Generated Files**

```
processed_parquet/
├── site_split_assignment.json          # Critical: Train/test site assignments
├── [site]_comprehensive.parquet        # Processed data for all sites

ecosystem/evaluation/clustering_results/
├── advanced_site_clusters_*.csv         # Cluster assignments (train sites only)

ecosystem/models/results/cluster_models/
├── cluster_model_*.json                 # Trained models (train sites only)

ecosystem/models/results/novel_site_ensemble/
├── novel_site_ensemble_results_*.json   # Ensemble performance results
```

### **Key Commands**

```bash
# 1. Data pipeline with site splitting
python data_pipeline_v3.py --export-format parquet --test-fraction 0.2 --random-seed 42

# 2. Performance-based clustering (train sites only)
python ecosystem/clustering/clustering_v3.py --feature-set performance

# 3. Train cluster models (train sites only)  
python ecosystem/models/train_cluster_models.py

# 4. Test ensemble on novel sites (test sites only)
python ecosystem/models/novel_site_ensemble_testing.py
```

## 🎯 **Current Status**

This document serves as the definitive reference for the ensemble approach. All implementation work should align with this plan to ensure:

1. **No data leakage** between train and test sets
2. **Proper ensemble testing** on truly novel sites
3. **Reproducible results** with consistent methodology
4. **Performance improvement** through weighted ensemble predictions

Any deviations from this plan should be documented and justified to maintain the integrity of the ensemble validation approach.
