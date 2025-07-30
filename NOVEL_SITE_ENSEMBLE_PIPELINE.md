# Novel Site Ensemble Testing Pipeline

## 🎯 **Overview**

The Novel Site Ensemble Testing Pipeline tests trained cluster models on **withheld sites** using **three different weighting strategies** to create ensemble predictions. This is a critical step in the ensemble validation plan that comes **after** training cluster models but **before** cross-ecosystem validation.

## 🔄 **Pipeline Flow**

```
1. Performance-Based Clustering → 2. Train Cluster Models → 3. Novel Site Testing → 4. Cross-Ecosystem Validation
```

## 📋 **What This Pipeline Tests**

### **Core Question**

*"Can ensemble models trained on performance-based clusters predict novel sites better than individual cluster models?"*

### **Three Weighting Strategies Tested**

1. **Confidence-Weighted Ensemble**
   - Weights based on model prediction confidence
   - More confident models get higher weights
   - **Implementation**: Simplified equal weights (can be enhanced)

2. **Historical Performance-Weighted Ensemble**
   - Weights based on each model's training performance
   - Better-performing models get higher weights
   - **Implementation**: Uses R² scores from training

3. **Distance-Weighted Ensemble**
   - Weights based on similarity to cluster characteristics
   - Models from similar clusters get higher weights
   - **Implementation**: Simplified equal weights (can be enhanced)

## 🛠️ **Implementation Details**

### **Site Splitting Strategy**

- **80% of sites** → Used for training cluster models
- **20% of sites** → Withheld for novel testing
- **Random split** within each cluster to maintain cluster representation

### **Testing Process**

1. **Load trained models** for each cluster
2. **Test each model** on all withheld sites (cross-cluster testing)
3. **Calculate ensemble predictions** using three weighting strategies
4. **Compare performance** of individual models vs. ensemble approaches

### **Performance Metrics**

- **R² Score**: How well predictions match actual values
- **MSE (Mean Squared Error)**: Average prediction error
- **Individual vs. Ensemble**: Compare single model vs. weighted ensemble performance

## 🚀 **Usage**

### **Prerequisites**

1. ✅ Performance-based clustering completed
2. ✅ Cluster models trained
3. ✅ Processed parquet data available

### **Run the Pipeline**

```bash
# Test the pipeline first
python test_novel_site_ensemble.py

# Run full ensemble testing
python ecosystem/models/novel_site_ensemble_testing.py --test-fraction 0.2

# With custom parameters
python ecosystem/models/novel_site_ensemble_testing.py \
    --parquet-dir ../../processed_parquet \
    --models-dir ./results/cluster_models \
    --output-dir ./results/novel_site_ensemble \
    --test-fraction 0.2
```

## 📊 **Expected Output**

### **Results File**: `novel_site_ensemble_results_YYYYMMDD_HHMMSS.json`

```json
{
  "individual_models": {
    "0": {"mse": 0.123, "r2": 0.789},
    "1": {"mse": 0.145, "r2": 0.756},
    "2": {"mse": 0.167, "r2": 0.723}
  },
  "confidence_weighted": {"mse": 0.134, "r2": 0.778},
  "historical_weighted": {"mse": 0.129, "r2": 0.782},
  "distance_weighted": {"mse": 0.131, "r2": 0.780},
  "metadata": {
    "timestamp": "20250127_143022",
    "test_fraction": 0.2,
    "n_test_sites": 15
  }
}
```

## 🎯 **Success Criteria**

### **What We're Looking For**

1. **Ensemble outperforms individual models**: At least one weighting strategy should have better R² than the best individual model
2. **Consistent improvement**: Multiple weighting strategies should show improvement
3. **Robust performance**: Ensemble should work across different types of novel sites

### **Expected Benefits**

- **Better generalization**: Ensemble models should handle novel sites better
- **Reduced overfitting**: Multiple models reduce reliance on single cluster characteristics
- **Improved robustness**: Weighted combinations should be more stable

## 🔄 **Next Steps After This Pipeline**

### **Phase 4: Cross-Ecosystem Validation**

Once novel site testing is complete, the next step is **cross-ecosystem validation**:

```bash
# Test models trained on one cluster predicting sites from other clusters
python ecosystem/models/spatial_parquet.py --optimize-hyperparams
```

### **Phase 5: Hybrid Ensemble Implementation**

Combine the best-performing weighting strategies with cross-ecosystem learning for the final ensemble model.

## 🧪 **Testing and Validation**

### **Test Script**: `test_novel_site_ensemble.py`

- Verifies all prerequisites are met
- Runs a small-scale test (10% of sites)
- Validates the pipeline works correctly

### **Validation Checks**

- ✅ All required files exist
- ✅ Data pipeline completed
- ✅ Clustering results available
- ✅ Trained models loaded successfully
- ✅ Ensemble predictions calculated
- ✅ Results saved correctly

## 🔧 **Troubleshooting**

### **Common Issues**

1. **No clustering results**: Run performance-based clustering first
2. **No trained models**: Train cluster models first
3. **Memory issues**: Reduce test fraction or use fewer sites
4. **Missing parquet files**: Run data pipeline with clean mode

### **Performance Optimization**

- **Small test fraction**: Use `--test-fraction 0.1` for quick testing
- **Memory management**: Script handles large datasets efficiently
- **Parallel processing**: Can be enhanced for faster execution

## 📈 **Expected Results**

### **Best Case Scenario**

- Ensemble R² > Best individual model R² by 0.05-0.10
- All three weighting strategies show improvement
- Consistent performance across different site types

### **Acceptable Results**

- At least one weighting strategy outperforms individual models
- Ensemble R² > 0.7 (good prediction performance)
- Stable performance across test runs

### **Failure Indicators**

- All ensemble strategies perform worse than individual models
- Very low R² scores (< 0.5)
- High variance in performance across different test sets

## 🎯 **Key Insights This Pipeline Provides**

1. **Weighting Strategy Effectiveness**: Which approach works best for ensemble predictions
2. **Cross-Cluster Learning**: How well models can predict sites from other clusters
3. **Ensemble Robustness**: Whether combining models improves generalization
4. **Performance Baseline**: Establishes baseline for cross-ecosystem validation

This pipeline is the **critical bridge** between individual cluster models and the final ensemble approach, providing the foundation for cross-ecosystem learning and validation.
