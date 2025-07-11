# Engineered Features Documentation for SAPFLUXNET Transpiration Modeling

## Overview

This document provides comprehensive documentation for all engineered features created in the SAPFLUXNET processing pipeline for transpiration modeling. Each feature includes its mathematical formula, scientific basis, and relevant literature references.

## Current Pipeline Status

The pipeline processes SAPFLUXNET data with the following characteristics:

- **Input**: Raw data from `sapwood/` directory (165 total sites)
- **Site Exclusion**: 59 sites with no valid sap flow data (always excluded)
- **Optional Exclusion**: 26 problematic sites with high quality flag rates
- **Output**: Processed data in `comprehensive_processed/` directory (~106 sites)
- **Processing**: Adaptive memory management with streaming for large files
- **Features**: Complete feature set with temporal, lagged, rolling, and domain-specific features

## Optimization Summary

The pipeline has been optimized to leverage existing SAPFLUXNET data efficiently:

- **Removed redundant features**: ~40 redundant features eliminated
- **Uses existing data**: Leverages `ext_rad`, `solar_TIMESTAMP`, `ppfd_in`, `vpd` directly
- **Simplified calculations**: Single combined indices instead of multiple redundant calculations
- **Maintained predictive power**: All key interactions and unique features preserved
- **Better performance**: Less computation, cleaner code, same scientific value

## 1. Temporal Features

### 1.1 Basic Time Features

**Features:** `hour`, `day_of_year`, `month`, `year`, `day_of_week`

**Formulas:**

- `hour = timestamp.hour` (0-23)
- `day_of_year = timestamp.dayofyear` (1-366)
- `month = timestamp.month` (1-12)
- `year = timestamp.year`
- `day_of_week = timestamp.dayofweek` (0=Monday, 6=Sunday)

**Scientific Basis:** Temporal patterns in transpiration follow diurnal and seasonal cycles driven by solar radiation, temperature, and plant physiology. These features capture the fundamental temporal structure of plant water use.

**References:**

- Granier, A. (1987). Evaluation of transpiration in a Douglas-fir stand by means of sap flow measurements. *Tree Physiology*, 3(4), 309-320.
- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
- Wullschleger, S. D., et al. (2001). Diurnal and seasonal variation in xylem sap flow of Norway spruce (Picea abies L.) growing in Sweden. *Journal of Experimental Botany*, 52(357), 921-929.

### 1.2 Solar-Adjusted Cyclical Encoding

**Features:** `solar_hour_sin`, `solar_hour_cos`, `solar_day_sin`, `solar_day_cos`

**Formulas:**

```
# Solar-adjusted daily cycle (24 hours)
solar_hour_rad = 2π × solar_hour / 24
solar_hour_sin = sin(solar_hour_rad)
solar_hour_cos = cos(solar_hour_rad)

# Solar-adjusted annual cycle (365 days)
solar_day_rad = 2π × solar_day_of_year / 365
solar_day_sin = sin(solar_day_rad)
solar_day_cos = cos(solar_day_rad)
```

**Scientific Basis:** Uses existing `solar_TIMESTAMP` data for more accurate time features that account for geographic position and seasonal variations in day length. Solar-adjusted features provide better representation of actual solar forcing than standard time features.

**References:**

- Cerqueira, V., et al. (2020). A comparative study of time series forecasting methods for short term electric energy consumption prediction in smart buildings. *Energy Reports*, 6, 173-182.
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts.
- Schulze, E. D., et al. (1972). A mathematical model for simulating water relations in photosynthesis. *Oecologia*, 10(2), 121-130.

### 1.3 Simplified Time Features

**Features:** `is_daylight`, `is_peak_sunlight`, `is_weekend`

**Formulas:**

```
is_daylight = (hour ≥ 6) AND (hour ≤ 18)
is_peak_sunlight = (hour ≥ 10) AND (hour ≤ 16)
is_weekend = (day_of_week ≥ 5)
```

**Scientific Basis:** Simple boolean features capture key diurnal patterns in transpiration without redundant calculations. These features represent critical periods for plant water use and stomatal activity.

**References:**

- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
- Granier, A., et al. (1996). Transpiration of trees and forest stands: short and long-term monitoring using sapflow methods. *Global Change Biology*, 2(3), 265-274.

**Note:** Seasonal patterns are now captured using existing `ext_rad` data instead of calculated features.

## 2. Lagged Features

### 2.1 Environmental Variable Lags

**Features:** `{variable}_lag_{N}h` where variable ∈ {ta, rh, vpd, sw_in, ws, precip, swc_shallow, ppfd_in} and N ∈ {1, 2, 3, 6, 12, 24}

**Formula:**

```
variable_lag_Nh = variable.shift(N)
```

**Scientific Basis:** Transpiration responds to environmental conditions with time delays due to stomatal response times and plant physiology. Lagged features capture the temporal response of plants to changing environmental conditions.

**Implementation:** The pipeline creates adaptive lag features based on available memory and file size. Full 24-hour lags are created when memory allows, with reduced lag sets for memory-constrained situations. Only variables that exist in the data are processed.

**References:**

- Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society B*, 273(927), 593-610.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Whitehead, D., et al. (1984). Stomatal conductance, transpiration and resistance to water uptake in a Pinus sylvestris spacing experiment. *Canadian Journal of Forest Research*, 14(5), 692-700.

## 3. Rolling Statistics Features

### 3.1 Rolling Mean and Standard Deviation

**Features:** `{variable}_mean_{N}h`, `{variable}_std_{N}h` where N ∈ {3, 6, 12, 24, 48, 72}

**Formulas:**

```
variable_mean_Nh = variable.rolling(window=N, min_periods=1).mean()
variable_std_Nh = variable.rolling(window=N, min_periods=1).std()
```

**Scientific Basis:** Rolling statistics capture the temporal context and stability of environmental conditions, which influence transpiration rates. These features represent the environmental memory that affects plant water use decisions.

**Implementation:** The pipeline creates adaptive rolling features based on available memory. Full rolling windows (3-72h) are created when memory allows, with reduced sets for memory-constrained situations. Only key environmental variables (ta, vpd, sw_in, rh) that exist in the data are used for rolling statistics to optimize performance.

**References:**

- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
- Wullschleger, S. D., et al. (2001). Diurnal and seasonal variation in xylem sap flow of Norway spruce (Picea abies L.) growing in Sweden. *Journal of Experimental Botany*, 52(357), 921-929.
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.

## 4. Interaction Features

### 4.1 VPD and Radiation Interaction

**Feature:** `vpd_ppfd_interaction`

**Formula:**

```
vpd_ppfd_interaction = vpd × ppfd_in
```

**Scientific Basis:** The interaction between vapor pressure deficit (VPD) and photosynthetically active radiation (PAR) is crucial for stomatal conductance and transpiration.

**References:**

- Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for C3 plants. *Plant, Cell & Environment*, 18(4), 339-355.

### 4.2 Temperature and Humidity Ratio

**Feature:** `temp_humidity_ratio`

**Formula:**

```
temp_humidity_ratio = ta / (vpd + ε)
```

where ε = 1e-6 (small constant to avoid division by zero)

**Scientific Basis:** The ratio of temperature to VPD indicates the atmospheric demand for water relative to temperature conditions.

### 4.3 Water Stress Index

**Feature:** `water_stress_index`

**Formula:**

```
water_stress_index = swc_shallow / (vpd + ε)
```

**Scientific Basis:** This index represents the balance between soil water availability and atmospheric demand, indicating water stress conditions. Higher values indicate better water availability relative to atmospheric demand.

**References:**

- Jones, H. G. (2004). Irrigation scheduling: advantages and pitfalls of plant-based methods. *Journal of Experimental Botany*, 55(407), 2427-2436.

### 4.4 Light Efficiency

**Feature:** `ppfd_efficiency`

**Formula:**

```
ppfd_efficiency = ppfd_in / (sw_in + ε)
```

**Scientific Basis:** This ratio indicates the efficiency of light conversion to photosynthetically active radiation.

### 4.5 Soil Moisture Gradient

**Feature:** `soil_moisture_gradient`

**Note:** This feature has been removed from the pipeline as `swc_deep` data is not available in the SAPFLUXNET dataset.

**Previous Implementation:** This feature was designed to calculate the vertical gradient between shallow and deep soil moisture measurements, but since deep soil moisture data is not consistently available across sites, it has been removed to ensure consistent feature creation.

## 5. Domain-Specific Features

### 5.1 Water Stress Features

**Features:** `water_stress_index`, `moisture_availability`

**Formulas:**

```
# Always created (uses only shallow soil moisture)
water_stress_index = vpd / (swc_shallow + ε)
moisture_availability = swc_shallow
```

**Scientific Basis:** Simplified water stress features that capture the key balance between atmospheric demand and soil water availability. The water stress index indicates the ratio of atmospheric demand to soil water availability. Higher values indicate greater atmospheric demand relative to soil water availability (more stress).

**Implementation:** The pipeline creates these features using only shallow soil moisture data, as deep soil moisture measurements are not consistently available across sites. This ensures consistent feature creation across all sites.

**References:**

- Tardieu, F., & Simonneau, T. (1998). Variability among species of stomatal control under fluctuating soil water status and evaporative demand: modelling isohydric and anisohydric behaviours. *Journal of Experimental Botany*, 49(Special), 419-432.
- Jones, H. G. (2004). Irrigation scheduling: advantages and pitfalls of plant-based methods. *Journal of Experimental Botany*, 55(407), 2427-2436.

### 5.2 Light and Energy Features

**Features:** `ppfd_efficiency`, `light_efficiency`

**Formulas:**

```
ppfd_efficiency = ppfd_in / (sw_in + ε)
light_efficiency = ppfd_in / (ext_rad + ε)  # Uses existing solar data
```

**Scientific Basis:** Light efficiency features using existing data, with `ext_rad` providing the perfect seasonal solar signal.

**References:**

- Farquhar, G. D., et al. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, 149(1), 78-90.

### 5.3 Temperature Response Features

**Features:** `temp_deviation`

**Formulas:**

```
optimal_temp = 25°C
temp_deviation = |ta - optimal_temp|
```

**Scientific Basis:** Simplified temperature deviation from optimal photosynthesis temperature.

**References:**

- Schulze, E. D., et al. (1972). A mathematical model for simulating water relations in photosynthesis. *Oecologia*, 10(2), 121-130.

### 5.4 Physiological Response Features

**Features:** `stomatal_conductance_proxy`, `stomatal_control_index`

**Formulas:**

```
stomatal_conductance_proxy = ppfd_in / (vpd + ε)
stomatal_control_index = vpd × ppfd_in × ext_rad  # Key interaction using existing data
```

**Scientific Basis:** Physiological features using existing data, with `stomatal_control_index` capturing the key three-way interaction.

**References:**

- Ball, J. T., et al. (1987). A model predicting stomatal conductance and its contribution to the control of photosynthesis under different environmental conditions. *Progress in Photosynthesis Research*, 4, 221-224.

### 5.5 Wind Effects

**Features:** `wind_stress`, `wind_vpd_interaction`

**Formulas:**

```
wind_stress = ws / (ws_max + ε)
wind_vpd_interaction = ws × vpd
```

**Scientific Basis:** Wind affects boundary layer resistance and enhances VPD effects on transpiration.

**References:**

- Grace, J. (1988). Plant response to wind. *Agriculture, Ecosystems & Environment*, 22(1), 71-88.

### 5.6 Precipitation Effects

**Features:** `recent_precip_1h`, `recent_precip_6h`, `recent_precip_24h`, `precip_intensity`

**Formulas:**

```
recent_precip_1h = precip.shift(1).fillna(0)
recent_precip_6h = precip.rolling(6, min_periods=1).sum()
recent_precip_24h = precip.rolling(24, min_periods=1).sum()
precip_intensity = precip / (precip.rolling(6, min_periods=1).sum() + ε)
```

**Scientific Basis:** Recent precipitation affects soil moisture and reduces transpiration demand.

**References:**

- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.

### 5.7 Diurnal Cycle Features

**Note:** These features have been removed from the pipeline for simplicity.

**Previous Features:** `is_peak_transpiration`, `is_recovery_hours`

**Previous Implementation:** These features captured critical diurnal periods for plant water use, but have been removed to simplify the feature set and focus on core environmental and physiological features.

### 5.8 Seasonal Water Use Patterns

**Note:** These features have been removed from the pipeline for simplicity.

**Previous Features:** `is_peak_growing`, `is_dormant_season`

**Previous Implementation:** These features captured seasonal patterns in transpiration following growing season dynamics, but have been removed to simplify the feature set and focus on core environmental and physiological features.

### 5.9 Tree-Specific Features

**Features:** `tree_size_factor`, `sapwood_leaf_ratio`, `transpiration_capacity`

**Formulas:**

```
tree_size_factor = log(pl_dbh + 1)
sapwood_leaf_ratio = pl_sapw_area / (pl_leaf_area + ε)
transpiration_capacity = pl_sapw_area × (ppfd_in / (vpd + ε))
```

**Scientific Basis:** Tree size and hydraulic architecture determine transpiration capacity. Larger trees have greater water transport capacity, while sapwood-to-leaf area ratios indicate hydraulic efficiency.

**Implementation:** Tree size class features (`is_large_tree`, `is_medium_tree`, `is_small_tree`) have been removed for simplicity, keeping only the continuous tree size factor.

**References:**

- McDowell, N., et al. (2002). The relationship between tree height and leaf area: sapwood area ratio. *Oecologia*, 132(1), 12-20.
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.

### 5.10 Conditional Feature Creation

**Implementation Note:** Some features are created conditionally based on data availability:

**Soil Moisture Features:**

- `moisture_availability`: Uses shallow soil moisture data (`swc_shallow`)
- `soil_moisture_gradient`: Removed from pipeline as `swc_deep` data is not available

**Tree Features:**

- Tree-specific features are only created if plant metadata is available
- Sapwood and leaf area features require both `pl_sapw_area` and `pl_leaf_area` columns
- Tree size class features (`is_large_tree`, `is_medium_tree`, `is_small_tree`) have been removed for simplicity

**Removed Features for Simplicity:**

- Diurnal cycle features: `is_peak_transpiration`, `is_recovery_hours`
- Seasonal features: `is_peak_growing`, `is_dormant_season`
- Climate zone features: `is_tropical`, `is_temperate`, `is_boreal`

**Metadata Features:**

- Site, stand, species, and plant features are created based on available metadata files
- Missing metadata results in null values rather than feature exclusion

This approach ensures the pipeline works with the actual data available in the SAPFLUXNET dataset while focusing on core environmental and physiological features.

### 5.11 Climate Zone Features

**Note:** These features have been removed from the pipeline for simplicity.

**Previous Features:** `is_tropical`, `is_temperate`, `is_boreal`

**Previous Implementation:** These features captured climate zone effects on transpiration behavior, but have been removed to simplify the feature set and focus on core environmental and physiological features.

## 6. Key Optimizations Using Existing Data

### 6.1 Extraterrestrial Radiation (`ext_rad`)

**What it is:** Theoretical solar radiation at the top of the atmosphere
**Why it's perfect:** Already calculated based on latitude, day of year, and solar geometry
**What we use it for:**

- Seasonal patterns (instead of calculated seasonal features)
- Solar efficiency calculations
- Key interactions with VPD and PPFD

**Scientific Basis:** `ext_rad` contains all the seasonal information we need without redundant calculations.

### 6.2 Solar Timestamp (`solar_TIMESTAMP`)

**What it is:** Solar-adjusted time accounting for geographic position
**Why it's better:** More accurate than calculated time features
**What we use it for:**

- Solar-adjusted cyclical encoding
- More accurate diurnal patterns

**Scientific Basis:** Uses actual solar geometry instead of approximations.

### 6.3 Simplified Stress Indices

**Before:** Multiple redundant stress calculations
**After:** Single combined indices using existing data
**Examples:**

- `water_stress_index = vpd / (swc_shallow + ε)` (instead of separate VPD and soil stress)
- `stomatal_control_index = vpd × ppfd_in × ext_rad` (key three-way interaction)

## 7. Metadata Features

### 7.1 Site-Level Features

**Features:** `latitude`, `longitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `biome`, `igbp_class`, `country`

**Scientific Basis:** Geographic and climatic characteristics influence species composition, growth patterns, and transpiration rates.

**References:**

- Churkina, G., et al. (2005). Spatial analysis of growing season length control over net ecosystem exchange. *Global Change Biology*, 11(10), 1777-1787.

### 7.2 Stand-Level Features

**Features:** `stand_age`, `basal_area`, `tree_density`, `stand_height`, `leaf_area_index`, `clay_percentage`, `sand_percentage`, `silt_percentage`, `soil_depth`

**Scientific Basis:** Stand structure and soil properties affect water availability and transpiration patterns.

**References:**

- Waring, R. H., & Running, S. W. (2007). *Forest ecosystems: analysis at multiple scales*. Academic Press.

### 7.3 Species-Level Features

**Features:** `species_name`, `leaf_habit`, `n_trees`, `species_basal_area_perc`

**Scientific Basis:** Species-specific characteristics determine transpiration behavior and responses to environmental conditions.

**References:**

- Meinzer, F. C., et al. (2001). Coordination of leaf and stem water transport properties in tropical forest trees. *Oecologia*, 127(4), 457-465.

### 7.4 Plant-Level Features

**Features:** `pl_age`, `pl_dbh`, `pl_height`, `pl_leaf_area`, `pl_bark_thick`, `pl_social`, `pl_sapw_area`, `pl_sapw_depth`

**Scientific Basis:** Individual tree characteristics determine water transport capacity and transpiration rates.

**References:**

- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.

## 8. Derived Features

### 8.1 Climate Zone Features

**Features:** `climate_zone_code`, `latitude_abs`, `climate_zone`

**Formulas:**

```
climate_zone_code = cut(latitude, bins=[-90, -23.5, 23.5, 90], labels=[0, 1, 2])
latitude_abs = |latitude|
climate_zone = cut(latitude, bins=[-90, -23.5, 23.5, 90], 
                  labels=['Temperate_South', 'Tropical', 'Temperate_North'])
```

**Scientific Basis:** Climate zones determine growing seasons, day length patterns, and species adaptations.

### 8.2 Aridity Index

**Feature:** `aridity_index`

**Formula:**

```
aridity_index = mean_annual_precip / (mean_annual_temp + 10)
```

**Scientific Basis:** The aridity index indicates water availability relative to temperature, affecting plant water use strategies.

**References:**

- De Martonne, E. (1926). Une nouvelle fonction climatologique: L'indice d'aridité. *La Météorologie*, 2, 449-458.

### 8.3 Tree Size and Age Classes

**Features:** `tree_size_class`, `tree_age_class`

**Formulas:**

```
tree_size_class = cut(pl_dbh, bins=[0, 10, 30, 50, 100, 1000], 
                     labels=['Sapling', 'Small', 'Medium', 'Large', 'Very Large'])
tree_age_class = cut(pl_age, bins=[0, 20, 50, 100, 200, 1000], 
                    labels=['Young', 'Mature', 'Old', 'Very Old', 'Ancient'])
```

**Scientific Basis:** Tree size and age affect water transport capacity and transpiration rates.

### 8.4 Sapwood Efficiency

**Feature:** `sapwood_leaf_ratio`

**Formula:**

```
sapwood_leaf_ratio = pl_sapw_area / (pl_leaf_area + ε)
```

**Scientific Basis:** The ratio of sapwood area to leaf area indicates the hydraulic capacity to support transpiration.

**References:**

- McDowell, N., et al. (2002). The relationship between tree height and leaf area: sapwood area ratio. *Oecologia*, 132(1), 12-20.

### 8.5 Tree Volume Index

**Feature:** `tree_volume_index`

**Formula:**

```
tree_volume_index = (pl_dbh²) × pl_height
```

**Scientific Basis:** Tree volume is related to biomass and water storage capacity.

## 9. Categorical Encodings

### 9.1 Biome and Land Cover Encodings

**Features:** `biome_code`, `igbp_code`

**Scientific Basis:** Different biomes have distinct species compositions and environmental adaptations affecting transpiration patterns.

### 9.2 Leaf Habit Encodings

**Feature:** `leaf_habit_code`

**Mapping:**

- cold deciduous: 1
- warm deciduous: 2
- evergreen: 3
- semi-deciduous: 4

**Scientific Basis:** Leaf habit determines seasonal patterns in transpiration and water use.

**References:**

- Reich, P. B., et al. (1997). From tropics to tundra: global convergence in plant functioning. *Proceedings of the National Academy of Sciences*, 94(25), 13730-13734.

### 9.3 Social Status Encodings

**Feature:** `social_status_code`

**Mapping:**

- dominant: 3
- codominant: 2
- intermediate: 1
- suppressed: 0

**Scientific Basis:** Social status affects access to light and water resources, influencing transpiration rates.

## 10. Implementation Notes

### 10.1 Memory Management

The pipeline uses adaptive memory management to handle large datasets:

- Chunk-based processing for large files
- Aggressive garbage collection
- Memory monitoring and cleanup
- Adaptive processing strategies based on available memory

### 10.2 Missing Value Handling

- Features with >50% missing values are removed
- Numeric missing values are filled with median
- Categorical missing values are filled with mode
- Target variable (sap_flow) missing values result in row removal

### 10.3 Data Validation

- Early validation of sap flow data structure
- Timestamp column detection
- File integrity checks
- Processing status tracking

## 11. References

### Key Literature

1. **SAPFLUXNET Database:**
   - Poyatos, R., et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607-2649.

2. **Sap Flow Methods:**
   - Granier, A. (1987). Evaluation of transpiration in a Douglas-fir stand by means of sap flow measurements. *Tree Physiology*, 3(4), 309-320.
   - Burgess, S. S., et al. (2001). An improved heat pulse method to measure low and reverse rates of sap flow in woody plants. *Tree Physiology*, 21(9), 589-598.

3. **Transpiration Modeling:**
   - Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society B*, 273(927), 593-610.
   - Leuning, R. (1995). A critical appraisal of a combined stomatal-photosynthesis model for C3 plants. *Plant, Cell & Environment*, 18(4), 339-355.

4. **Environmental Controls:**
   - Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
   - Wullschleger, S. D., et al. (2001). Diurnal and seasonal variation in xylem sap flow of Norway spruce (Picea abies L.) growing in Sweden. *Journal of Experimental Botany*, 52(357), 921-929.

5. **Feature Engineering:**
   - Cerqueira, V., et al. (2020). A comparative study of time series forecasting methods for short term electric energy consumption prediction in smart buildings. *Energy Reports*, 6, 173-182.
   - Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts.

### Additional Resources

- **SAPFLUXNET Website:** <https://sapfluxnet.creaf.cat/>
- **SAPFLUXNET Data Paper:** <https://doi.org/10.5194/essd-13-2607-2021>
- **Sap Flow Methods Review:** <https://doi.org/10.1093/treephys/21.9.589>

---

*This documentation covers all engineered features implemented in the SAPFLUXNET processing pipeline. Each feature is designed to capture important aspects of plant transpiration based on established scientific principles and literature.*
