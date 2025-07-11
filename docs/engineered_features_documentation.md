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
- **Features**: Streamlined feature set focused on core environmental and physiological features

## Feature Creation Pipeline

The pipeline creates features in six stages:

1. **Temporal Features**: Basic time features and solar-adjusted cyclical encoding
2. **Lagged Features**: Environmental variables with 1-24 hour lags
3. **Rolling Features**: Moving averages and standard deviations (3-72 hour windows)
4. **Interaction Features**: Currently disabled (all commented out)
5. **Domain-Specific Features**: Temperature deviation and tree-specific features
6. **Metadata Features**: Site, stand, species, and plant characteristics

## 1. Temporal Features

### 1.1 Basic Time Features

**Features Created:** `hour`, `day_of_year`, `month`, `year`, `day_of_week`

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

**Features Created:** `solar_hour`, `solar_day_of_year`, `solar_hour_sin`, `solar_hour_cos`, `solar_day_sin`, `solar_day_cos`

**Formulas:**
```
# Extract solar time components
solar_hour = solar_TIMESTAMP.hour
solar_day_of_year = solar_TIMESTAMP.dayofyear

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
- Campbell, G. S., & Norman, J. M. (1998). *An introduction to environmental biophysics*. Springer.

### 1.3 Boolean Time Features

**Features Created:** `is_daylight`, `is_peak_sunlight`, `is_weekend`

**Formulas:**
```
is_daylight = (hour >= 6) AND (hour <= 18)
is_peak_sunlight = (hour >= 10) AND (hour <= 16)
is_weekend = (day_of_week >= 5)
```

**Scientific Basis:** Simple boolean features capture key diurnal patterns in transpiration without redundant calculations. These features represent critical periods for plant water use and stomatal activity.

**References:**
- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
- Granier, A., et al. (1996). Transpiration of trees and forest stands: short and long-term monitoring using sapflow methods. *Global Change Biology*, 2(3), 265-274.

## 2. Lagged Features

### 2.1 Environmental Variable Lags

**Features Created:** `{variable}_lag_{N}h` where variable ∈ {ta, rh, vpd, sw_in, ws, precip, swc_shallow, ppfd_in} and N ∈ {1, 2, 3, 6, 12, 24}

**Formula:**
```
variable_lag_Nh = variable.shift(N)
```

**Scientific Basis:** Transpiration responds to environmental conditions with time delays due to stomatal response times, hydraulic conductance, and plant water storage. Lagged features capture the temporal response of plants to changing environmental conditions.

**Implementation:** The pipeline creates adaptive lag features based on available memory and file size. Full 24-hour lags are created when memory allows, with reduced lag sets for memory-constrained situations.

**References:**
- Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society B*, 273(927), 593-610.
- Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Whitehead, D., et al. (1984). Stomatal conductance, transpiration and resistance to water uptake in a Pinus sylvestris spacing experiment. *Canadian Journal of Forest Research*, 14(5), 692-700.

## 3. Rolling Statistics Features

### 3.1 Rolling Mean and Standard Deviation

**Features Created:** `{variable}_mean_{N}h`, `{variable}_std_{N}h` where variable ∈ {ta, vpd, sw_in, rh} and N ∈ {3, 6, 12, 24, 48, 72}

**Formulas:**
```
variable_mean_Nh = variable.rolling(window=N, min_periods=1).mean()
variable_std_Nh = variable.rolling(window=N, min_periods=1).std()
```

**Scientific Basis:** Rolling statistics capture the temporal context and stability of environmental conditions, which influence transpiration rates. These features represent the environmental memory that affects plant water use decisions and stomatal behavior.

**Implementation:** The pipeline creates adaptive rolling features based on available memory. Full rolling windows (3-72h) are created when memory allows, with reduced sets for memory-constrained situations.

**References:**
- Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
- Wullschleger, S. D., et al. (2001). Diurnal and seasonal variation in xylem sap flow of Norway spruce (Picea abies L.) growing in Sweden. *Journal of Experimental Botany*, 52(357), 921-929.
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.

## 4. Interaction Features

**Status:** Currently disabled in the pipeline. All interaction features are commented out and not created.

**Rationale:** Interaction features can be computed during training phase to reduce preprocessing complexity and memory usage. The pipeline focuses on core environmental and physiological features.

## 5. Domain-Specific Features

### 5.1 Temperature Response Features

**Features Created:** `temp_deviation`

**Formula:**
```
temp_deviation = |ta - 25|
```

**Scientific Basis:** Temperature deviation from optimal photosynthesis temperature (25°C) affects stomatal conductance and transpiration rates. This feature captures the non-linear relationship between temperature and plant physiological processes.

**References:**
- Farquhar, G. D., et al. (1980). A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species. *Planta*, 149(1), 78-90.
- Schulze, E. D., et al. (1972). A mathematical model for simulating water relations in photosynthesis. *Oecologia*, 10(2), 121-130.

### 5.2 Tree-Specific Features

**Features Created:** `tree_size_factor`, `sapwood_leaf_ratio`, `transpiration_capacity`

**Formulas:**
```
tree_size_factor = log(pl_dbh + 1)
sapwood_leaf_ratio = pl_sapw_area / (pl_leaf_area + 1e-6)
transpiration_capacity = pl_sapw_area × (ppfd_in / (vpd + 1e-6))
```

**Scientific Basis:** Tree size and hydraulic architecture determine transpiration capacity. Larger trees have greater water transport capacity, while sapwood-to-leaf area ratios indicate hydraulic efficiency. Transpiration capacity combines hydraulic capacity with environmental driving forces.

**References:**
- McDowell, N., et al. (2002). The relationship between tree height and leaf area: sapwood area ratio. *Oecologia*, 132(1), 12-20.
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.
- Tyree, M. T., & Ewers, F. W. (1991). The hydraulic architecture of trees and other woody plants. *New Phytologist*, 119(3), 345-370.

## 6. Metadata Features

### 6.1 Site-Level Features

**Features Created:** `latitude`, `longitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `biome`, `igbp_class`, `country`, `site_code`, `site_name`, `is_inside_country`

**Scientific Basis:** Geographic and climatic characteristics influence species composition, growth patterns, and transpiration rates. These features provide environmental context for transpiration modeling.

**References:**
- Churkina, G., et al. (2005). Spatial analysis of growing season length control over net ecosystem exchange. *Global Change Biology*, 11(10), 1777-1787.
- Peel, M. C., et al. (2007). Updated world map of the Köppen-Geiger climate classification. *Hydrology and Earth System Sciences*, 11(5), 1633-1644.

### 6.2 Stand-Level Features

**Features Created:** `stand_age`, `basal_area`, `tree_density`, `stand_height`, `leaf_area_index`, `clay_percentage`, `sand_percentage`, `silt_percentage`, `soil_depth`, `soil_texture`, `terrain`, `growth_condition`

**Scientific Basis:** Stand structure and soil properties affect water availability, root distribution, and transpiration patterns. These features capture the forest ecosystem context that influences water use.

**References:**
- Waring, R. H., & Running, S. W. (2007). *Forest ecosystems: analysis at multiple scales*. Academic Press.
- Bréda, N., et al. (2006). Temperate forest trees and stands under severe drought: a review of ecophysiological responses, adaptation processes and long-term consequences. *Annals of Forest Science*, 63(6), 625-644.

### 6.3 Species-Level Features

**Features Created:** `species_name`, `leaf_habit`, `n_trees`

**Scientific Basis:** Species-specific characteristics determine transpiration behavior and responses to environmental conditions. Leaf habit particularly affects seasonal patterns and drought responses.

**References:**
- Meinzer, F. C., et al. (2001). Coordination of leaf and stem water transport properties in tropical forest trees. *Oecologia*, 127(4), 457-465.
- Reich, P. B., et al. (1997). From tropics to tundra: global convergence in plant functioning. *Proceedings of the National Academy of Sciences*, 94(25), 13730-13734.

### 6.4 Plant-Level Features

**Features Created:** `pl_age`, `pl_dbh`, `pl_height`, `pl_leaf_area`, `pl_bark_thick`, `pl_social`, `pl_species`, `pl_sapw_area`, `pl_sapw_depth`

**Scientific Basis:** Individual tree characteristics determine water transport capacity and transpiration rates. These features capture tree-level variation in hydraulic architecture and competitive status.

**References:**
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.
- Meinzer, F. C., et al. (2001). Coordination of leaf and stem water transport properties in tropical forest trees. *Oecologia*, 127(4), 457-465.

### 6.5 Environmental Metadata Features

**Features Created:** `measurement_timestep`, `timezone`

**Scientific Basis:** Measurement protocol and temporal context affect data quality and temporal patterns in transpiration measurements.

**References:**
- Poyatos, R., et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607-2649.

## 7. Derived Features

### 7.1 Climate Zone Features

**Features Created:** `climate_zone_code`, `latitude_abs`, `climate_zone`

**Formulas:**
```
climate_zone_code = cut(latitude, bins=[-90, -23.5, 23.5, 90], labels=[0, 1, 2])
latitude_abs = |latitude|
climate_zone = cut(latitude, bins=[-90, -23.5, 23.5, 90], 
                  labels=['Temperate_South', 'Tropical', 'Temperate_North'])
```

**Scientific Basis:** Climate zones determine growing seasons, day length patterns, and species adaptations. These features capture large-scale climatic controls on transpiration.

**References:**
- Peel, M. C., et al. (2007). Updated world map of the Köppen-Geiger climate classification. *Hydrology and Earth System Sciences*, 11(5), 1633-1644.
- Woodward, F. I., et al. (2004). Global climate and the distribution of plant biomes. *Philosophical Transactions of the Royal Society B*, 359(1450), 1465-1476.

### 7.2 Aridity Index

**Features Created:** `aridity_index`

**Formula:**
```
aridity_index = mean_annual_precip / (mean_annual_temp + 10)
```

**Scientific Basis:** The aridity index indicates water availability relative to temperature, affecting plant water use strategies and drought adaptations.

**References:**
- De Martonne, E. (1926). Une nouvelle fonction climatologique: L'indice d'aridité. *La Météorologie*, 2, 449-458.
- Trabucco, A., & Zomer, R. J. (2009). Global aridity index (global-aridity) and global potential evapo-transpiration (global-PET) geospatial database. *CGIAR Consortium for Spatial Information*.

### 7.3 Leaf Habit Encoding

**Features Created:** `leaf_habit_code`

**Mapping:**
- cold deciduous: 1
- warm deciduous: 2
- evergreen: 3
- semi-deciduous: 4

**Scientific Basis:** Leaf habit determines seasonal patterns in transpiration, water use efficiency, and drought tolerance strategies.

**References:**
- Reich, P. B., et al. (1997). From tropics to tundra: global convergence in plant functioning. *Proceedings of the National Academy of Sciences*, 94(25), 13730-13734.
- Givnish, T. J. (2002). Adaptive significance of evergreen vs. deciduous leaves: solving the triple paradox. *Silva Fennica*, 36(3), 703-743.

### 7.4 Biome and Land Cover Encodings

**Features Created:** `biome_code`, `igbp_code`

**Scientific Basis:** Different biomes and land cover types have distinct species compositions, environmental conditions, and transpiration patterns.

**References:**
- Olson, D. M., et al. (2001). Terrestrial ecoregions of the world: a new map of life on Earth. *BioScience*, 51(11), 933-938.
- Friedl, M. A., et al. (2010). MODIS Collection 5 global land cover: algorithm refinements and characterization of new datasets. *Remote Sensing of Environment*, 114(1), 168-182.

### 7.5 Tree Size and Age Classes

**Features Created:** `tree_size_class`, `tree_age_class`

**Formulas:**
```
tree_size_class = cut(pl_dbh, bins=[0, 10, 30, 50, 100, 1000], 
                     labels=['Sapling', 'Small', 'Medium', 'Large', 'Very Large'])
tree_age_class = cut(pl_age, bins=[0, 20, 50, 100, 200, 1000], 
                    labels=['Young', 'Mature', 'Old', 'Very Old', 'Ancient'])
```

**Scientific Basis:** Tree size and age affect hydraulic architecture, water transport capacity, and transpiration rates through ontogenetic changes in plant structure.

**References:**
- Ryan, M. G., & Yoder, B. J. (1997). Hydraulic limits to tree height and tree growth. *BioScience*, 47(4), 235-242.
- Phillips, N., et al. (2002). Canopy and hydraulic conductance in young, mature and old Douglas-fir trees. *Tree Physiology*, 22(2-3), 205-211.

### 7.6 Social Status Encoding

**Features Created:** `social_status_code`

**Mapping:**
- dominant: 3
- codominant: 2
- intermediate: 1
- suppressed: 0

**Scientific Basis:** Social status affects access to light and water resources, influencing transpiration rates and competitive interactions.

**References:**
- Oliver, C. D., & Larson, B. C. (1996). *Forest stand dynamics*. John Wiley & Sons.
- Pretzsch, H. (2009). *Forest dynamics, growth and yield*. Springer.

### 7.7 Tree Volume Index

**Features Created:** `tree_volume_index`

**Formula:**
```
tree_volume_index = (pl_dbh²) × pl_height
```

**Scientific Basis:** Tree volume is related to biomass, water storage capacity, and hydraulic conductance, affecting transpiration patterns.

**References:**
- West, G. B., et al. (1999). A general model for the structure and allometry of plant vascular systems. *Nature*, 400(6745), 664-667.
- Enquist, B. J., et al. (1999). Allometric scaling of production and life-history variation in vascular plants. *Nature*, 401(6756), 907-911.

### 7.8 Timezone and Measurement Features

**Features Created:** `timezone_offset`, `measurement_frequency`

**Formulas:**
```
timezone_offset = extract timezone offset from timezone string
measurement_frequency = 60 / measurement_timestep
```

**Scientific Basis:** Temporal context and measurement frequency affect data quality and temporal resolution of transpiration patterns.

**References:**
- Poyatos, R., et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607-2649.

## 8. Categorical Encodings

### 8.1 Comprehensive Categorical Encoding

The pipeline includes comprehensive categorical encoding for the following variables:

**Encoded Variables:**
- `biome` → `biome_code`
- `igbp_class` → `igbp_code`
- `country` → `country_code`
- `soil_texture` → `soil_texture_code`
- `aspect` → `aspect_code`
- `terrain` → `terrain_code`
- `growth_condition` → `growth_condition_code`
- `leaf_habit` → `leaf_habit_code`
- `pl_social` → `social_status_code`
- `climate_zone` → `climate_zone_code`
- `tree_size_class` → `tree_size_class_code`
- `tree_age_class` → `tree_age_class_code`

**Scientific Basis:** Categorical encoding converts text-based categorical variables into numeric representations suitable for machine learning while preserving the categorical nature of the data.

**References:**
- Potdar, K., et al. (2017). A comparative study of categorical variable encoding techniques for neural network classifiers. *International Journal of Computer Applications*, 175(4), 7-9.

## 9. Features Explicitly Removed

The following features are explicitly removed from the final schema to ensure consistency and avoid problematic columns:

**Removed Features:**
- `pl_name` - High cardinality, inconsistent across sites
- `swc_deep` - Not available in SAPFLUXNET dataset
- `netrad` - Inconsistent across sites
- `seasonal_leaf_area` - Inconsistent across sites
- All interaction features - Commented out to reduce complexity
- All redundant features - Can be computed during training

**Scientific Rationale:** These features were removed to create a clean, consistent dataset that works reliably across all sites while maintaining scientific validity.

## 10. Implementation Notes

### 10.1 Adaptive Feature Creation

The pipeline uses adaptive feature creation based on:
- Available memory
- File size
- System capabilities

**Memory-Constrained Situations:**
- Reduced lag windows (1-12 hours instead of 1-24 hours)
- Reduced rolling windows (3-24 hours instead of 3-72 hours)
- Streaming processing for large files

### 10.2 Missing Value Handling

- Missing environmental variables result in NaN values for derived features
- Missing metadata results in NaN values rather than feature exclusion
- Target variable (sap_flow) missing values result in row removal

### 10.3 Consistent Schema

The pipeline ensures consistent schema across all sites by:
- Adding missing columns with NaN values
- Removing problematic columns
- Standardizing data types
- Handling XGBoost compatibility issues

## 11. Key References

### Core Literature

1. **SAPFLUXNET Database:**
   - Poyatos, R., et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607-2649.

2. **Sap Flow Methods:**
   - Granier, A. (1987). Evaluation of transpiration in a Douglas-fir stand by means of sap flow measurements. *Tree Physiology*, 3(4), 309-320.
   - Burgess, S. S., et al. (2001). An improved heat pulse method to measure low and reverse rates of sap flow in woody plants. *Tree Physiology*, 21(9), 589-598.

3. **Transpiration Modeling:**
   - Jarvis, P. G. (1976). The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field. *Philosophical Transactions of the Royal Society B*, 273(927), 593-610.
   - Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.

4. **Environmental Controls:**
   - Oren, R., et al. (1999). Survey of sap flow in forest trees using heat pulse velocity. *Agricultural and Forest Meteorology*, 95(4), 225-244.
   - Wullschleger, S. D., et al. (2001). Diurnal and seasonal variation in xylem sap flow of Norway spruce (Picea abies L.) growing in Sweden. *Journal of Experimental Botany*, 52(357), 921-929.

5. **Plant Hydraulics:**
   - Tyree, M. T., & Ewers, F. W. (1991). The hydraulic architecture of trees and other woody plants. *New Phytologist*, 119(3), 345-370.
   - McDowell, N., et al. (2002). The relationship between tree height and leaf area: sapwood area ratio. *Oecologia*, 132(1), 12-20.

### Additional Resources

- **SAPFLUXNET Website:** <https://sapfluxnet.creaf.cat/>
- **SAPFLUXNET Data Paper:** <https://doi.org/10.5194/essd-13-2607-2021>
- **Sap Flow Methods Review:** <https://doi.org/10.1093/treephys/21.9.589>

---

*This documentation covers all engineered features actually implemented in the SAPFLUXNET processing pipeline. Each feature is designed to capture important aspects of plant transpiration based on established scientific principles and literature.*
