# Model Integration Guide for Sap Flow Visualizations

## 🎯 Overview

This guide explains how your trained XGBoost ecosystem models are integrated into the sap flow visualization system, creating powerful model-data comparison and prediction visualizations.

## 🤖 Model Integration Architecture

### **Trained Models Used:**

- **XGBoost cluster-specific models** (JSON format)
- **5 ecosystem clusters** with separate trained models
- **Feature importance data** for each cluster
- **Cluster assignments** for each site

### **Integration Components:**

```python
class SapFlowVisualizer:
    def __init__(self, models_dir='ecosystem/models/results/cluster_models'):
        self.models = {}              # Loaded XGBoost models by cluster
        self.feature_names = []       # Feature names from importance files
        self.cluster_assignments = {} # Site → cluster mapping
```

## 📊 New Model-Integrated Visualizations

### **1. 🤖 Model vs Observed Comparison**

- **File:** `model_vs_observed_{timestamp}.png`
- **Shows:** Scatter plots comparing model predictions to actual observations
- **Features:**
  - 6 sites with best data coverage
  - 1:1 reference line
  - R² correlation coefficients
  - Cluster identification for each site

### **2. 📈 Prediction Time Series**

- **File:** `prediction_time_series_{timestamp}.png`
- **Shows:** Time series of predictions vs observations + residuals
- **Features:**
  - Week-long time series comparison
  - Observed vs predicted overlay
  - Residual analysis plot
  - Representative site selection

### **3. 🌌 3D Model Predictions**

- **File:** `3d_model_predictions_{timestamp}.html`
- **Shows:** Interactive 3D plot of predictions in environmental space
- **Features:**
  - Temperature × VPD × Predicted Sap Flow
  - Color-coded by prediction error
  - Multiple sites combined
  - Interactive rotation and zoom

### **4. 📊 3D Feature Importance**

- **File:** `3d_feature_importance_{timestamp}.html`
- **Shows:** Feature importance patterns across ecosystem clusters
- **Features:**
  - Cluster × Feature Rank × Importance
  - Top 20 most important features
  - Cluster-specific patterns
  - Interactive exploration

## 🚀 How to Run Model-Integrated Visualizations

### **Option 1: All Visualizations (2D + 3D + Models)**

```bash
python sap_flow_visualizations.py
```

### **Option 2: Just Model Visualizations**

```bash
python run_model_integrated_plots.py
```

### **Option 3: Just 3D Plots**

```bash
python run_3d_sap_flow_plots.py
```

## 🔧 Model Loading Process

### **1. Model Discovery**

```python
# Finds all trained models
model_files = glob.glob('ecosystem/models/results/cluster_models/xgb_model_cluster_*.json')

# Loads each model
for model_file in model_files:
    model = xgb.Booster()
    model.load_model(model_file)
    self.models[cluster_id] = model
```

### **2. Feature Name Loading**

```python
# Loads feature names from importance files
importance_files = glob.glob('feature_importance_cluster_*.csv')
self.feature_names = importance_df['feature_name'].tolist()
```

### **3. Cluster Assignment Loading**

```python
# Loads site → cluster mapping
cluster_files = glob.glob('ecosystem/evaluation/clustering_results/advanced_site_clusters_*.csv')
self.cluster_assignments = dict(zip(clusters_df['site'], clusters_df['ecosystem_cluster']))
```

## 🎯 Prediction Process

### **Smart Cluster Selection:**

```python
def predict_sap_flow(self, data, site_name=None):
    # 1. Determine cluster from site name or data
    if site_name in self.cluster_assignments:
        cluster_id = self.cluster_assignments[site_name]
    elif 'ecosystem_cluster' in data.columns:
        cluster_id = data['ecosystem_cluster'].iloc[0]
    
    # 2. Use appropriate cluster model
    model = self.models[cluster_id]
    
    # 3. Prepare features and predict
    X = data[feature_cols].fillna(0)
    dmatrix = xgb.DMatrix(X)
    predictions = model.predict(dmatrix)
```

## 📋 Requirements

### **Required Files:**

- **Trained models:** `ecosystem/models/results/cluster_models/xgb_model_cluster_*.json`
- **Feature importance:** `ecosystem/models/results/cluster_models/feature_importance_cluster_*.csv`
- **Cluster assignments:** `ecosystem/evaluation/clustering_results/advanced_site_clusters_*.csv`
- **Data:** `processed_parquet/*.parquet`

### **Python Dependencies:**

```bash
pip install xgboost plotly pandas numpy matplotlib seaborn
```

## 🎨 Visualization Outputs

### **Static Plots (PNG):**

- Model vs observed scatter plots
- Prediction time series with residuals
- High-resolution, publication-ready

### **Interactive Plots (HTML):**

- 3D model predictions in environmental space
- 3D feature importance across clusters
- Fully interactive (zoom, rotate, hover)
- Shareable web format

## 💡 Key Insights from Model Integration

### **1. Model Performance Validation**

- **Visual R² assessment** across different sites
- **Residual pattern analysis** for model diagnostics
- **Environmental space coverage** of predictions

### **2. Ecosystem-Specific Patterns**

- **Cluster-specific feature importance** differences
- **Environmental response variations** by ecosystem type
- **Prediction accuracy patterns** across climate gradients

### **3. Model Generalization**

- **Cross-site prediction quality** within clusters
- **Environmental boundary effects** on predictions
- **Temporal pattern capture** by models

## 🔬 Advanced Usage

### **Custom Model Integration:**

```python
# Initialize with custom model directory
visualizer = SapFlowVisualizer(
    data_dir='your_data_dir',
    models_dir='your_models_dir',
    output_dir='custom_output'
)

# Load specific sites
data = visualizer.load_sample_sites(n_sites=15)

# Create specific visualizations
visualizer.create_model_vs_observed_plots(data)
visualizer.create_3d_model_predictions(data)
```

### **Model Comparison:**

- Compare different model versions
- Analyze feature importance evolution
- Track prediction quality over time

## 🎉 Benefits of Model Integration

### **1. Model Validation**

- **Visual confirmation** of model performance
- **Error pattern identification** for model improvement
- **Environmental bias detection** in predictions

### **2. Scientific Insights**

- **Ecosystem-specific transpiration drivers** revealed
- **Environmental response surfaces** visualized
- **Model uncertainty patterns** understood

### **3. Operational Applications**

- **Real-time prediction visualization** capabilities
- **Model performance monitoring** tools
- **Decision support** for forest management

---

## 🚀 Next Steps

1. **Run the visualizations** to see your models in action
2. **Analyze prediction patterns** across different sites
3. **Identify model improvement opportunities** from visualizations
4. **Use insights** to refine your modeling approach

Your trained ecosystem models are now fully integrated into a comprehensive visualization system that reveals both their strengths and areas for improvement!
