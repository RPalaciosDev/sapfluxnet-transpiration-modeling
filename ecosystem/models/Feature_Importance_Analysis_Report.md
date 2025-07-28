# Cluster Feature Importance Analysis Report

**Analysis Date:** July 27, 2025  
**Data Source:** XGBoost cluster models trained on SAPFLUXNET ecosystem data  
**Clusters Analyzed:** 3 ecosystem clusters (0, 1, 2)

---

## Executive Summary

This analysis reveals **distinct ecosystem strategies** across three major clusters, each utilizing fundamentally different feature sets to predict sap flow. The integration of our v2 feature mapping system successfully provides interpretable insights into model behavior.

### Key Findings

- **Cluster 0**: Complex interaction-driven ecosystem (19.8% interactions)
- **Cluster 1**: Time-lag dependent ecosystem (36.8% lagged features)
- **Cluster 2**: Structure and dynamics-driven ecosystem (19.6% structural + 17.6% rolling features)

---

## Cluster Profiles

### 🎯 Cluster 0: "Interaction-Driven Complex Ecosystems"

**Total Importance:** 1,655,880 | **Features:** 269

**Top Dominant Features:**

1. `temp_soil_interaction` (Interaction) - 303,925 ⭐ **Highest single feature importance across all clusters**
2. `n_trees` (Other) - 96,825
3. `ext_rad` (Environmental) - 82,444
4. `stand_height` (Structural) - 65,919
5. `elevation` (Geographic) - 63,840

**Ecosystem Characteristics:**

- **19.8% Interaction Features** - Highest among all clusters
- Strong dependence on **complex environmental interactions**
- **Geographic factors** play significant role (elevation, latitude)
- **Environmental drivers** are crucial (10.1% of importance)

**Interpretation:** These are likely **complex, mature forest ecosystems** where multiple environmental factors interact in sophisticated ways to drive sap flow patterns.

---

### 🎯 Cluster 1: "Time-Lag Dependent Ecosystems"  

**Total Importance:** 633,591 | **Features:** 267

**Top Dominant Features:**

1. `ppfd_in_lag_2h` (Lagged) - 162,287 ⭐ **Single highest lagged feature**
2. `sw_in_lag_1h` (Lagged) - 35,726
3. `tree_volume_index` (Interaction) - 26,385
4. `sw_in_mean_3h` (Rolling) - 24,578
5. `pl_social_code` (Structural) - 19,527

**Ecosystem Characteristics:**

- **36.8% Lagged Features** - Dominant strategy
- **Low environmental dependency** (only 3.1%)
- **Light-driven responses** (PPFD and shortwave radiation lags)
- **Medium structural importance**

**Interpretation:** These are likely **younger or more responsive ecosystems** where sap flow responds with predictable delays to light and radiation changes, suggesting efficient physiological responses.

---

### 🎯 Cluster 2: "Structure & Dynamics-Driven Ecosystems"

**Total Importance:** 1,913,969 | **Features:** 271 ⭐ **Highest total importance**

**Top Dominant Features:**

1. `basal_area` (Structural) - 145,990
2. `vpd_rate_24h` (Rate of Change) - 144,447
3. `is_inside_country` (Temporal) - 126,895
4. `tree_size_class_code` (Structural) - 94,810
5. `stand_height` (Structural) - 94,145

**Ecosystem Characteristics:**

- **19.6% Structural Features** - Physical tree characteristics dominate
- **17.6% Rolling Window Features** - Temporal patterns matter
- **High rate-of-change sensitivity** (10.8% of importance)
- **Strong temporal patterns** (18.6%)

**Interpretation:** These are likely **mature, structurally-complex forests** where tree size, canopy structure, and dynamic environmental changes drive sap flow.

---

## Cross-Cluster Analysis

### Common Important Features (Top 30)

Only **2 features** appear in ALL clusters' top 30:

- `stand_height`: Critical across all ecosystem types
- `vpd`: Vapor pressure deficit universally important

### Cluster Similarities

- **Clusters 0 & 2**: Share 6 common features (structural/environmental focus)
- **Clusters 1 & 2**: Share 3 common features (minimal overlap)
- **Clusters 0 & 1**: Share 4 common features

### Feature Type Patterns

| Feature Type | Cluster 0 | Cluster 1 | Cluster 2 |
|--------------|-----------|-----------|-----------|
| Lagged | 2,238 avg | **4,859 avg** ⭐ | 3,407 avg |
| Rolling | 1,505 avg | 970 avg | **3,122 avg** ⭐ |
| Rate of Change | 3,931 avg | 521 avg | **17,198 avg** ⭐ |
| Interactions | **40,347 avg** ⭐ | 4,442 avg | 3,930 avg |

---

## Key Insights

### 1. **Ecosystem Complexity Spectrum**

The clusters represent a spectrum from simple lag-response systems (Cluster 1) to complex interaction-driven systems (Cluster 0).

### 2. **Temporal Response Strategies**

- **Cluster 1**: Direct lag responses to light
- **Cluster 2**: Dynamic rate-of-change responses  
- **Cluster 0**: Complex temporal interactions

### 3. **Environmental Dependencies**

- **Cluster 0**: High environmental interaction dependence
- **Cluster 2**: Moderate environmental sensitivity
- **Cluster 1**: Low direct environmental dependence

### 4. **Structural Importance Hierarchy**

1. **Cluster 2**: Structure dominates (mature forests)
2. **Cluster 0**: Structure + interactions (complex systems)
3. **Cluster 1**: Structure secondary to temporal patterns

---

## Technical Validation

### Feature Mapping Integration ✅

- Successfully mapped **807 total features** across all clusters
- **272 unique features** in v2 pipeline properly categorized
- **100% mapping coverage** for all cluster results

### Data Quality Metrics

- **Cluster 0**: 269/272 features (98.9% coverage)
- **Cluster 1**: 267/272 features (98.2% coverage)  
- **Cluster 2**: 271/272 features (99.6% coverage)

---

## Recommendations

### 1. **Model Interpretation**

Each cluster represents a distinct **ecosystem archetype** that should be analyzed and applied separately:

- Use **Cluster 0 models** for complex, interaction-rich environments
- Use **Cluster 1 models** for responsive, light-driven systems
- Use **Cluster 2 models** for structure-dominated mature forests

### 2. **Feature Engineering Priorities**

- **For Cluster 0**: Focus on interaction terms and complex environmental relationships
- **For Cluster 1**: Emphasize lag features and light response timing
- **For Cluster 2**: Prioritize structural metrics and rate-of-change calculations

### 3. **Data Collection Strategies**

Different ecosystem types require different monitoring emphases:

- **Complex systems**: Multi-variable environmental sensors
- **Lag-responsive systems**: High-frequency light measurements
- **Structure-driven systems**: Detailed structural surveys + dynamic monitoring

---

## Conclusions

The feature mapping integration has successfully revealed **three distinct ecosystem strategies** for sap flow regulation. This analysis demonstrates that:

1. **No single model fits all ecosystems** - different strategies require different approaches
2. **Feature importance varies dramatically** by ecosystem type
3. **Interpretable ML** is crucial for understanding ecosystem function
4. **Our feature mapping system** enables powerful model interpretation

This analysis provides a foundation for **ecosystem-specific model application** and **targeted data collection strategies** for improved sap flow prediction across diverse forest environments.

---

*Report generated using comprehensive feature importance analysis with v2 pipeline feature mapping integration.*
