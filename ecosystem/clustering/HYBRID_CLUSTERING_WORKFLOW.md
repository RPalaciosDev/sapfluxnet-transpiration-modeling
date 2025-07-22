# Hybrid Feature Set Clustering Workflow for SAPFLUXNET

## Overview

This document describes the process and rationale for using a **hybrid feature set**—combining ecologically meaningful features and XGBoost model-important features—for advanced ecosystem clustering of SAPFLUXNET sites. The goal is to achieve clusters that are both ecologically interpretable and relevant for predictive modeling.

---

## Rationale

- **Ecological features** (climate, geography, seasonality) provide interpretability and align with classic ecosystem boundaries.
- **Model-important features** (from XGBoost feature importance) ensure clusters reflect axes of variation that matter for sap flow prediction.
- **Hybrid approach** balances interpretability and predictive relevance, improving both cluster quality and downstream model transferability.

---

## Step-by-Step Workflow

### 1. **Extract Model-Important Features**

- Used the feature importance report from a random split XGBoost model (`sapfluxnet_external_memory_features.txt`).
- Selected only site-level, non-lagged, non-rolling features (e.g., climate, geography, stand structure).

### 2. **Define Hybrid Feature Set**

- **Geographic/Climate:** longitude, latitude, latitude_abs, mean_annual_temp, mean_annual_precip, aridity_index, elevation, climate_zone_code, seasonal_temp_range, seasonal_precip_range
- **Engineered/Derived:** temp_precip_ratio, seasonality_index, climate_continentality, elevation_latitude_ratio, aridity_seasonality, temp_elevation_ratio, precip_latitude_ratio
- **Stand/Forest Structure:** tree_volume_index, stand_age, n_trees, tree_density, basal_area, stand_height, tree_size_class_code, sapwood_leaf_ratio, pl_dbh, pl_age, pl_height, pl_leaf_area, pl_sapw_depth, pl_sapw_area
- **Categorical (one-hot):** climate_zone_code, biome_code, igbp_class_code, leaf_habit_code

### 3. **Update Clustering Script**

- Modified `advanced_ecosystem_clustering.py` to:
  - Accept a `--feature_set` argument (`core`, `advanced`, or `hybrid`).
  - Use the hybrid set for clustering when specified.
  - One-hot encode categorical features for clustering.
  - Handle missing values by median imputation.
  - Fix JSON serialization for cluster outputs.

### 4. **Run Clustering Pipeline**

- Ran the script with:

  ```bash
  python advanced_ecosystem_clustering.py --feature_set hybrid
  ```

- Set output directory to `./results` for organized output.

### 5. **Interpret Results**

- **Best clustering strategy:** K-means (minmax scaling), 5 clusters
- **Cluster balance ratio:** 0.700 (very good)
- **Silhouette score:** 0.293 (good)
- **Clusters:** Warm Temperate, Temperate Mixed, Arid/Semi-Arid, Tropical Humid, etc.
- **Outputs:**
  - Cluster assignments CSV
  - Ecosystem interpretations JSON
  - All strategies JSON
  - Summary report TXT

---

## Key Takeaways

- The hybrid feature set produced clusters that are both balanced and ecologically meaningful.
- This approach is recommended for ecosystem-based spatial validation and model transfer in SAPFLUXNET.
- The workflow is fully reproducible and configurable via the `--feature_set` argument.

---

## How to Reproduce

1. Place your processed parquet files in `../../processed_parquet` (relative to the script).
2. Run:

   ```bash
   python advanced_ecosystem_clustering.py --feature_set hybrid
   ```

3. Review results in the `./results` directory.

---

## Authors & Contact

- Workflow and documentation generated with assistance from AI and project contributors.
