# Visualization Guide for Cluster Models and Spatial Validation

This guide describes the comprehensive visualization suite for analyzing cluster model performance and spatial validation results.

## 📊 Feature Importance Visualizations

### Usage

```bash
python visualize_feature_importance.py <results_directory> [options]
```

**Options:**

- `--output-dir`: Custom output directory (default: `results_dir/visualizations`)
- `--top-n`: Number of top features to show in detailed plots (default: 20)

### Visualizations Created

#### 1. **Top Features Heatmap** (`feature_importance_heatmap_top{N}.png`)

- **Purpose**: Shows importance of top N features across all clusters
- **Strengths**: Easy to identify which features are consistently important across clusters
- **Use Case**: Understanding feature universality vs. cluster-specific importance

#### 2. **Cluster Comparison Plot** (`feature_importance_by_cluster_top{N}.png`)

- **Purpose**: Side-by-side bar charts showing top features for each cluster
- **Strengths**: Reveals cluster-specific feature preferences
- **Use Case**: Understanding how different ecosystems rely on different environmental variables

#### 3. **Feature Category Analysis** (`feature_importance_by_category.png`, `feature_category_distribution.png`)

- **Purpose**: Groups features by type (meteorological, soil, vegetation, etc.)
- **Strengths**: Shows which types of environmental data are most predictive
- **Use Case**: Guiding data collection priorities and understanding ecological drivers
- **Note**: Only available if mapped feature importance files exist

#### 4. **Importance Distribution Analysis** (`feature_importance_distributions.png`)

- **Purpose**: Statistical analysis of feature importance patterns
- **Includes**:
  - Histogram of importance values by cluster
  - Box plots showing distribution spread
  - Top 20 features overall
  - Number of high-importance features per cluster
- **Use Case**: Understanding the complexity and consistency of different cluster models

#### 5. **Comprehensive Report** (`feature_importance_analysis_report.txt`)

- **Purpose**: Detailed text summary with statistics and rankings
- **Includes**: Overall stats, top features, cluster-specific analysis, category breakdown

## 🗺️ Spatial Validation Visualizations

### Usage

```bash
python visualize_spatial_validation.py <results_directory> [options]
```

**Options:**

- `--output-dir`: Custom output directory (default: `results_dir/visualizations`)

### Visualizations Created

#### 1. **Performance Overview** (`spatial_validation_overview.png`)

- **Purpose**: Four-panel overview of validation performance
- **Panels**:
  - R² distribution by cluster (box plots)
  - RMSE distribution by cluster (box plots)
  - Training vs Test R² scatter (generalization analysis)
  - Sample size vs performance relationship
- **Use Case**: Quick assessment of overall model quality and generalization

#### 2. **Cluster Performance Comparison** (`cluster_performance_comparison.png`)

- **Purpose**: Mean performance with error bars for each cluster
- **Strengths**: Clear comparison of which ecosystem types are easier/harder to model
- **Risk Assessment**: Identifies problematic clusters that may need different approaches

#### 3. **Fold-by-Fold Analysis** (`fold_by_fold_analysis.png`)

- **Purpose**: Shows performance variation across all validation folds
- **Strengths**: Reveals consistency and identifies problematic sites
- **Use Case**: Understanding spatial heterogeneity within clusters

#### 4. **Site Performance Heatmap** (`site_performance_heatmap.png`)

- **Purpose**: Matrix showing R² for each test site
- **Strengths**: Identifies specific sites that are difficult to predict
- **Risk Assessment**: Highlights potential outlier sites or data quality issues

#### 5. **Performance Distribution Analysis** (`performance_distribution_analysis.png`)

- **Purpose**: Six-panel statistical analysis including:
  - Overall R² and RMSE histograms
  - Overfitting analysis (training vs test performance)
  - Sample size relationships
  - Violin plots by cluster
  - Performance consistency metrics
- **Use Case**: Deep dive into model behavior and potential issues

#### 6. **Comprehensive Report** (`spatial_validation_analysis_report.txt`)

- **Purpose**: Detailed statistical summary
- **Includes**: Overall stats, overfitting analysis, cluster-specific performance, sample size analysis

## 🎯 Recommended Visualization Workflow

### For Feature Importance Analysis

1. **Start with the heatmap** to identify universally important features
2. **Use cluster comparison plots** to understand ecosystem-specific drivers
3. **Check category analysis** (if available) to understand data type importance
4. **Review the comprehensive report** for detailed statistics

### For Spatial Validation Analysis

1. **Begin with performance overview** for general model assessment
2. **Use cluster comparison** to identify problematic ecosystem types
3. **Examine fold-by-fold analysis** for consistency evaluation
4. **Check site heatmap** for outlier identification
5. **Review distribution analysis** for detailed model behavior understanding

## 💡 Interpretation Guidelines

### Feature Importance

- **High importance across clusters**: Core environmental drivers
- **Cluster-specific importance**: Ecosystem-adapted predictors
- **Low overall importance**: Potentially redundant or noisy features

### Spatial Validation Performance

- **R² > 0.5**: Good spatial generalization
- **R² 0.2-0.5**: Moderate generalization, acceptable for complex ecological data
- **R² < 0.2**: Poor generalization, model may be overfitting
- **High variance**: Inconsistent performance, potential data quality issues

### Risk Assessment Indicators

- **Large train-test R² gap**: Overfitting risk
- **High performance variance**: Data quality or model stability issues
- **Consistently poor clusters**: May need different modeling approaches
- **Site-specific failures**: Potential outliers or missing environmental drivers

## 📁 Example Usage

```bash
# After training cluster models
python train_cluster_models_gpu.py --run-name "biome_experiment_v1"

# Visualize feature importance
python visualize_feature_importance.py "./results/cluster_models/biome_experiment_v1_20250115_143022/"

# After spatial validation
python gpu_spatial_validation.py --run-name "biome_validation_v1"

# Visualize validation results
python visualize_spatial_validation.py "./results/parquet_spatial_validation/biome_validation_v1_20250115_150033/"
```

## 🔧 Dependencies

Both visualization scripts require:

- pandas
- numpy
- matplotlib
- seaborn

Install with: `pip install pandas numpy matplotlib seaborn`

## 📋 Output Structure

```
results_directory/
├── visualizations/
│   ├── feature_importance_heatmap_top20.png
│   ├── feature_importance_by_cluster_top20.png
│   ├── feature_importance_by_category.png
│   ├── feature_category_distribution.png
│   ├── feature_importance_distributions.png
│   ├── feature_importance_analysis_report.txt
│   ├── spatial_validation_overview.png
│   ├── cluster_performance_comparison.png
│   ├── fold_by_fold_analysis.png
│   ├── site_performance_heatmap.png
│   ├── performance_distribution_analysis.png
│   └── spatial_validation_analysis_report.txt
```

This comprehensive visualization suite provides multiple perspectives on model performance, helping you understand both the ecological drivers (feature importance) and the spatial generalization capabilities (validation performance) of your cluster-based models.
