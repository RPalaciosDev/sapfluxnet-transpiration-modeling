# Enhanced Temporal Features for SAPFLUXNET Processing

**Date:** July 18, 2025  
**Purpose:** Improve temporal feature engineering based on feature importance analysis

## Overview

Based on our temporal validation feature importance analysis, we've enhanced the temporal feature engineering in the comprehensive processing pipeline to capture more sophisticated temporal patterns that are crucial for sap flow prediction.

## Key Enhancements Made

### 1. **Extended Rolling Windows**

- **Previous:** [3, 6, 12, 24, 48, 72] hours
- **Enhanced:** [3, 6, 12, 24, 48, 72, 168, 336, 720] hours
- **Added:** 7-day (168h), 14-day (336h), and 30-day (720h) windows
- **Rationale:** 72-hour features were among the most important, suggesting longer-term patterns are crucial

### 2. **Enhanced Cyclical Features**

- **Added:** `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos`
- **Enhanced:** More granular cyclical encodings for better temporal pattern capture
- **Rationale:** `solar_day_cos` was the second most important feature

### 3. **Advanced Boolean Features**

- **Added:** `is_morning`, `is_afternoon`, `is_night`
- **Added:** `is_spring`, `is_summer`, `is_autumn`, `is_winter`
- **Added:** `hours_since_sunrise`, `hours_since_sunset`
- **Rationale:** Temporal context is important for sap flow patterns

### 4. **Rate of Change Features**

- **New Feature Type:** Rate of change for all environmental variables
- **Windows:** 1h, 3h, 6h, 12h, 24h rate of change
- **Variables:** ta, rh, vpd, sw_in, ws, precip, swc_shallow, ppfd_in
- **Rationale:** Captures how quickly environmental conditions are changing

### 5. **Cumulative Features**

- **Precipitation:** 3h, 6h, 12h, 24h, 72h, 168h cumulative sums
- **Radiation:** 3h, 6h, 12h, 24h cumulative sums
- **Rationale:** Total water and energy inputs over time periods

### 6. **Enhanced Rolling Statistics**

- **Basic:** mean, std (all windows)
- **Extended:** min, max, range (24h+ windows)
- **Advanced:** p25, p75, IQR (72h+ windows)
- **Rationale:** More comprehensive statistical summaries

### 7. **Interaction Features**

- **VPD × PPFD:** Stomatal control interaction
- **VPD × Temperature:** Thermal stress interaction
- **Temperature × Humidity:** Moisture availability
- **Soil × VPD:** Water stress index
- **PPFD × Shortwave:** Light efficiency
- **Temperature × Soil:** Thermal-moisture interaction
- **Wind × VPD:** Boundary layer effects
- **Radiation × Temperature:** Energy balance
- **Humidity × Soil:** Moisture retention

## Feature Processing Pipeline

### **Standard Processing (6 Stages):**

1. **Enhanced Temporal Features** - Cyclical, boolean, seasonal
2. **Advanced Rolling Features** - Extended windows with statistics
3. **Lagged Features** - Adaptive lag creation
4. **Rate of Change Features** - Environmental change rates
5. **Cumulative Features** - Accumulated inputs
6. **Interaction Features** - Cross-variable interactions

### **Streaming Processing (Optimized):**

- Enhanced temporal features
- Basic rolling features
- Basic interaction features
- Metadata features

## Expected Impact

### **Based on Feature Importance Analysis:**

1. **Longer rolling windows** should capture the patterns that made 72-hour features so important
2. **Enhanced cyclical features** should improve on the success of `solar_day_cos`
3. **Rate of change features** should capture temporal dynamics
4. **Cumulative features** should capture accumulated environmental inputs
5. **Interaction features** should capture complex environmental relationships

### **Predicted Improvements:**

- **Better temporal generalization** across different time periods
- **More robust feature set** for capturing environmental patterns
- **Enhanced model interpretability** through interaction features
- **Improved performance** on temporal validation

## Memory Considerations

### **Enhanced Features Added:**

- **~50 new rolling features** (longer windows)
- **~40 new rate of change features**
- **~15 new cumulative features**
- **~10 new interaction features**
- **~15 new temporal features**

### **Total New Features:** ~130 additional features

- **Memory impact:** ~130KB per 1000 rows
- **Processing time:** ~20-30% increase
- **Storage:** ~15% increase in file sizes

## Configuration Options

### **Feature Settings:**

```python
FEATURE_SETTINGS = {
    'advanced_temporal_features': True,  # Enable enhanced temporal features
    'interaction_features': True,        # Enable interaction features
    'rate_of_change_features': True,     # Enable rate of change features
    'cumulative_features': True,         # Enable cumulative features
    'rolling_windows': [3, 6, 12, 24, 48, 72, 168, 336, 720]
}
```

### **Disable Features (if needed):**

- Set individual feature flags to `False` to disable specific feature types
- Useful for memory-constrained environments
- Allows selective feature engineering

## Next Steps

1. **Process a subset of sites** with enhanced features
2. **Compare performance** with previous feature set
3. **Analyze new feature importance** patterns
4. **Optimize feature selection** based on new results
5. **Consider ensemble approaches** with different feature subsets

## Files Modified

- `comprehensive_processing_pipeline.py` - Main processing pipeline
- `ProcessingConfig` class - Configuration settings
- Feature creation functions - Enhanced temporal engineering

---

*This enhancement addresses the key findings from our temporal validation analysis, particularly the importance of long-term environmental patterns and cyclical temporal features.*
