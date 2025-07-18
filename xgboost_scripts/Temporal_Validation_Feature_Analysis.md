# Temporal Validation Feature Importance Analysis

**Date:** July 18, 2025  
**Model:** SAPFLUXNET Chronological Temporal Validation  
**Method:** True chronological temporal cross-validation with external memory training

## Executive Summary

This analysis examines feature importance from a temporal validation model trained on SAPFLUXNET data spanning 22 years (1996-2018). The model uses true chronological temporal splitting to assess how well features generalize across time periods, revealing both the most predictive features and the challenges of temporal generalization.

## Model Performance Overview

- **Average Test R²:** 0.107 (poor temporal generalization)
- **Best Fold R²:** 0.172 (Fold 5)
- **Temporal Coverage:** 1996-03-01 to 2018-05-17 (8,112 days)
- **Training Approach:** External memory training with chunked processing

## Top 20 Most Important Features

| Rank | Feature Index | Feature Name | Importance | Category |
|------|---------------|--------------|------------|----------|
| 1 | f117 | rh_mean_72h | 2,453,566 | Rolling_Stats |
| 2 | f19 | solar_day_cos | 982,084 | Cyclical |
| 3 | f106 | sw_in_std_72h | 905,175 | Rolling_Stats |
| 4 | f6 | swc_shallow | 875,526 | Soil |
| 5 | f119 | temp_deviation | 824,708 | Derived |
| 6 | f57 | precip_lag_12h | 802,291 | Lagged |
| 7 | f128 | biome | 786,765 | Site_Characteristics |
| 8 | f111 | rh_mean_12h | 786,395 | Rolling_Stats |
| 9 | f121 | sapwood_leaf_ratio | 750,992 | Tree_Properties |
| 10 | f67 | ppfd_in_lag_3h | 511,992 | Lagged |
| 11 | f91 | vpd_mean_48h | 462,142 | Rolling_Stats |
| 12 | f104 | sw_in_std_48h | 442,044 | Rolling_Stats |
| 13 | f138 | basal_area | 433,302 | Vegetation |
| 14 | f105 | sw_in_mean_72h | 426,430 | Rolling_Stats |
| 15 | f98 | sw_in_std_6h | 393,945 | Rolling_Stats |
| 16 | f134 | site_code | 360,606 | Site_Characteristics |
| 17 | f115 | rh_mean_48h | 358,837 | Rolling_Stats |
| 18 | f71 | ta_mean_3h | 349,954 | Rolling_Stats |
| 19 | f112 | rh_std_12h | 349,140 | Rolling_Stats |
| 20 | f125 | elevation | 341,968 | Site_Characteristics |

## Feature Category Analysis

### 1. Meteorological Features (Total: 14,519,329)

**Most Important:** Long-term humidity patterns and radiation variability

- **Top Features:** rh_mean_72h, sw_in_std_72h, precip_lag_12h
- **Insight:** 72-hour rolling statistics are crucial for sap flow prediction

### 2. Rolling Statistics (Total: 9,417,374)

**Most Important:** Multi-hour aggregations of environmental variables

- **Top Features:** rh_mean_72h, sw_in_std_72h, rh_mean_12h
- **Insight:** Temporal aggregation captures important environmental patterns

### 3. Site Characteristics (Total: 2,279,486)

**Most Important:** Geographic and ecological context

- **Top Features:** biome, site_code, elevation
- **Insight:** Site-specific factors remain important even in temporal validation

### 4. Soil Features (Total: 1,284,325)

**Most Important:** Soil moisture content

- **Top Features:** swc_shallow, swc_shallow_lag_24h
- **Insight:** Current and recent soil moisture are critical for sap flow

### 5. Tree Properties (Total: 1,070,147)

**Most Important:** Structural characteristics

- **Top Features:** sapwood_leaf_ratio, tree_size_factor
- **Insight:** Tree architecture influences water transport capacity

## Key Insights

### Ecological Significance

1. **Long-term moisture patterns** (72-hour humidity means) are the strongest predictors
2. **Radiation variability** (72-hour standard deviations) captures important environmental stress patterns
3. **Soil moisture** remains critical even across long temporal periods
4. **Site-specific factors** (biome, elevation) maintain importance over time

### Temporal Generalization Challenges

1. **Poor test performance** (R² = 0.107) indicates significant temporal drift
2. **22-year span** may include climate change effects
3. **Site-specific temporal patterns** may be overfitting the model

### Feature Engineering Success

1. **Rolling statistics** are highly effective (9.4M total importance)
2. **Cyclical encodings** (solar_day_cos) capture important diurnal patterns
3. **Lagged features** provide valuable temporal context
4. **Derived features** (temp_deviation) add meaningful information

## Recommendations

### 1. Feature Selection

- Focus on top 20-30 features for model simplification
- Prioritize rolling statistics and site characteristics
- Consider removing features with importance < 10,000

### 2. Temporal Analysis

- Investigate temporal drift patterns in feature-target relationships
- Consider climate change impacts on model performance
- Analyze site-specific temporal stability

### 3. Model Improvements

- Implement feature importance-based feature selection
- Consider ensemble methods for temporal robustness
- Explore domain adaptation techniques for temporal generalization

### 4. Data Collection

- Prioritize soil moisture measurements (swc_shallow)
- Ensure long-term humidity monitoring
- Maintain site characteristic documentation

## Technical Details

- **Validation Method:** True chronological temporal cross-validation
- **Memory Management:** External memory training with chunked processing
- **Feature Count:** 157 total features
- **Temporal Splits:** 4 folds with expanding training windows
- **Processing:** Solar timestamp-based temporal ordering

## Files Generated

- `sapfluxnet_temporal_chronological_importance_mapped_20250718_072951.csv` - Mapped feature importance
- `sapfluxnet_temporal_chronological_fold_results_20250718_072951.csv` - Fold-wise performance
- `sapfluxnet_temporal_chronological_metrics_20250718_072951.txt` - Summary metrics

---

*Analysis performed using XGBoost with external memory training on SAPFLUXNET dataset spanning 1996-2018.*
