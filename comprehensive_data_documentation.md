# Comprehensive SAPFLUXNET Data Documentation

This document describes all columns in the comprehensive processed dataset, including original data, engineered features, and metadata-derived features.

## Data Sources

The dataset combines multiple data sources:

- **Environmental data**: Meteorological and soil measurements
- **Sap flow data**: Tree transpiration measurements  
- **Site metadata**: Geographic and climate information
- **Stand metadata**: Forest stand characteristics
- **Species metadata**: Tree species information
- **Environmental metadata**: Measurement protocols and sensor details
- **Plant metadata**: Individual tree characteristics and sensor specifications
- **Data quality flags**: Quality indicators for filtering

## Current Pipeline Status

The pipeline processes SAPFLUXNET data with the following characteristics:

- **Input**: Raw data from `sapwood/` directory (165 total sites)
- **Site Exclusion**: 59 sites with no valid sap flow data (always excluded)
- **Optional Exclusion**: 26 problematic sites with high quality flag rates
- **Output**: Processed data in `comprehensive_processed/` directory (~106 sites)
- **Processing**: Adaptive memory management with streaming for large files
- **Features**: Streamlined feature set focused on core environmental and physiological features

## Optimization Summary

The pipeline has been optimized to leverage existing SAPFLUXNET data efficiently:

- **Removed redundant features**: ~40 redundant features eliminated
- **Uses existing data**: Leverages `ext_rad`, `solar_TIMESTAMP`, `ppfd_in`, `vpd` directly
- **Simplified calculations**: Single combined indices instead of multiple redundant calculations
- **Maintained predictive power**: All key interactions and unique features preserved
- **Better performance**: Less computation, cleaner code, same scientific value

## Current Pipeline Status

The pipeline processes SAPFLUXNET data with the following characteristics:

- **Input**: Raw data from `sapwood/` directory (165 total sites)
- **Site Exclusion**: 59 sites with no valid sap flow data (always excluded)
- **Optional Exclusion**: 26 problematic sites with high quality flag rates
- **Output**: Processed data in `comprehensive_processed/` directory (~106 sites)
- **Processing**: Adaptive memory management with streaming for large files
- **Features**: Streamlined feature set focused on core environmental and physiological features

## Optimization Summary

The pipeline has been optimized to leverage existing SAPFLUXNET data efficiently:

- **Removed redundant features**: ~40 redundant features eliminated
- **Uses existing data**: Leverages `ext_rad`, `solar_TIMESTAMP`, `ppfd_in`, `vpd` directly
- **Simplified calculations**: Single combined indices instead of multiple redundant calculations
- **Maintained predictive power**: All key interactions and unique features preserved
- **Better performance**: Less computation, cleaner code, same scientific value

## Column Categories

### 1. Original Data Columns

#### Environmental Variables

- `ta`: Air temperature (°C)
- `rh`: Relative humidity (%)
- `vpd`: Vapor pressure deficit (kPa)
- `sw_in`: Incoming shortwave radiation (W m⁻²)
- `ws`: Wind speed (m s⁻¹)
- `precip`: Precipitation (mm)
- `swc_shallow`: Shallow soil water content (m³ m⁻³)
- `ppfd_in`: Photosynthetic photon flux density (μmol m⁻² s⁻¹)
- `ext_rad`: Extraterrestrial radiation (W m⁻²)

**Note**: `swc_deep` (deep soil water content) is not consistently available across sites and has been removed from the pipeline.

#### Sap Flow Data

- `sap_flow`: Sap flow rate (cm³ cm⁻² h⁻¹ or g h⁻¹)
- `plant_id`: Individual tree identifier
- `site`: Site identifier

### 2. Temporal Features

#### Basic Time Features

- `hour`: Hour of day (0-23)
- `day_of_year`: Day of year (1-365)
- `month`: Month (1-12)
- `year`: Year
- `day_of_week`: Day of week (0=Monday, 6=Sunday)

#### Solar-Adjusted Cyclical Encodings

- `solar_hour_sin`, `solar_hour_cos`: Solar-adjusted cyclical encoding of hour (24-hour cycle)
- `solar_day_sin`, `solar_day_cos`: Solar-adjusted cyclical encoding of day of year (365-day cycle)

**Scientific Basis**: Uses existing `solar_TIMESTAMP` data for more accurate time features that account for geographic position and seasonal variations in day length.

#### Boolean Time Features

- `is_daylight`: 1 if hour is 6-18, 0 otherwise
- `is_peak_sunlight`: 1 if hour is 10-16, 0 otherwise
- `is_weekend`: 1 if day is weekend, 0 otherwise

**Note**: Seasonal patterns are now captured using existing `ext_rad` data instead of calculated features.

### 3. Lagged Features

Lagged environmental variables (1, 2, 3, 6, 12, 24 hours):

- `ta_lag_1h`, `ta_lag_2h`, `ta_lag_3h`, `ta_lag_6h`, `ta_lag_12h`, `ta_lag_24h`
- `rh_lag_1h`, `rh_lag_2h`, `rh_lag_3h`, `rh_lag_6h`, `rh_lag_12h`, `rh_lag_24h`
- `vpd_lag_1h`, `vpd_lag_2h`, `vpd_lag_3h`, `vpd_lag_6h`, `vpd_lag_12h`, `vpd_lag_24h`
- `sw_in_lag_1h`, `sw_in_lag_2h`, `sw_in_lag_3h`, `sw_in_lag_6h`, `sw_in_lag_12h`, `sw_in_lag_24h`
- `ws_lag_1h`, `ws_lag_2h`, `ws_lag_3h`, `ws_lag_6h`, `ws_lag_12h`, `ws_lag_24h`
- `precip_lag_1h`, `precip_lag_2h`, `precip_lag_3h`, `precip_lag_6h`, `precip_lag_12h`, `precip_lag_24h`
- `swc_shallow_lag_1h`, `swc_shallow_lag_2h`, `swc_shallow_lag_3h`, `swc_shallow_lag_6h`, `swc_shallow_lag_12h`, `swc_shallow_lag_24h`
- `ppfd_in_lag_1h`, `ppfd_in_lag_2h`, `ppfd_in_lag_3h`, `ppfd_in_lag_6h`, `ppfd_in_lag_12h`, `ppfd_in_lag_24h`

**Implementation**: The pipeline creates adaptive lag features based on available memory and file size. Full 24-hour lags are created when memory allows, with reduced lag sets for memory-constrained situations.

### 4. Rolling Features

Rolling statistics (3, 6, 12, 24, 48, 72 hours) for key environmental variables:

- `ta_mean_3h`, `ta_mean_6h`, `ta_mean_12h`, `ta_mean_24h`, `ta_mean_48h`, `ta_mean_72h`
- `ta_std_3h`, `ta_std_6h`, `ta_std_12h`, `ta_std_24h`, `ta_std_48h`, `ta_std_72h`
- `vpd_mean_3h`, `vpd_mean_6h`, `vpd_mean_12h`, `vpd_mean_24h`, `vpd_mean_48h`, `vpd_mean_72h`
- `vpd_std_3h`, `vpd_std_6h`, `vpd_std_12h`, `vpd_std_24h`, `vpd_std_48h`, `vpd_std_72h`
- `sw_in_mean_3h`, `sw_in_mean_6h`, `sw_in_mean_12h`, `sw_in_mean_24h`, `sw_in_mean_48h`, `sw_in_mean_72h`
- `sw_in_std_3h`, `sw_in_std_6h`, `sw_in_std_12h`, `sw_in_std_24h`, `sw_in_std_48h`, `sw_in_std_72h`
- `rh_mean_3h`, `rh_mean_6h`, `rh_mean_12h`, `rh_mean_24h`, `rh_mean_48h`, `rh_mean_72h`
- `rh_std_3h`, `rh_std_6h`, `rh_std_12h`, `rh_std_24h`, `rh_std_48h`, `rh_std_72h`

**Implementation**: The pipeline creates adaptive rolling features based on available memory. Full rolling windows (3-72h) are created when memory allows, with reduced sets for memory-constrained situations.

### 5. Interaction Features

#### Environmental Interactions

- `vpd_ppfd_interaction`: VPD × PPFD (water stress × light availability)
- `temp_humidity_ratio`: Temperature / VPD (thermal-hydraulic ratio)
- `water_stress_index`: Shallow soil moisture / VPD (water stress index)
- `light_efficiency`: PPFD / Shortwave radiation (light use efficiency)

**Scientific Basis**: These interactions capture key physiological responses in transpiration, particularly the balance between atmospheric demand and environmental conditions.

### 6. Domain-Specific Features

#### Water Stress Features

- `water_stress_index`: Shallow soil moisture / VPD (water stress index)
- `moisture_availability`: Shallow soil moisture (moisture availability indicator)

**Implementation**: The pipeline creates these features using only shallow soil moisture data, as deep soil moisture measurements are not consistently available across sites.

#### Light and Energy Features

- `ppfd_efficiency`: PPFD / Shortwave radiation (light use efficiency)
- `light_efficiency`: PPFD / Extraterrestrial radiation (solar efficiency)

**Scientific Basis**: Uses existing `ext_rad` data for the perfect seasonal solar signal.

#### Temperature Features

- `temp_deviation`: Absolute deviation from optimal temperature (25°C)

#### Physiological Features

- `stomatal_conductance_proxy`: PPFD / VPD (stomatal control proxy)
- `stomatal_control_index`: VPD × PPFD × Extraterrestrial radiation (key three-way interaction)

**Scientific Basis**: Physiological features using existing data, with `stomatal_control_index` capturing the key three-way interaction.

#### Wind Effects

- `wind_stress`: Wind speed / Maximum wind speed (normalized)
- `wind_vpd_interaction`: Wind speed × VPD (wind-enhanced VPD effects)

#### Precipitation Effects

- `recent_precip_1h`: Precipitation lagged by 1 hour
- `recent_precip_6h`: 6-hour rolling sum of precipitation
- `recent_precip_24h`: 24-hour rolling sum of precipitation
- `precip_intensity`: Precipitation / 6-hour rolling sum (intensity)

#### Tree-Specific Features

- `tree_size_factor`: Log(DBH + 1) (tree size effect)
- `sapwood_leaf_ratio`: Sapwood area / Leaf area (hydraulic efficiency)
- `transpiration_capacity`: Sapwood area × (PPFD / VPD) (transpiration potential)

**Implementation**: Tree size class features (`is_large_tree`, `is_medium_tree`, `is_small_tree`) have been removed for simplicity, keeping only the continuous tree size factor.

### 7. Site Metadata Features

#### Geographic Features

- `latitude`: Site latitude (degrees)
- `longitude`: Site longitude (degrees)
- `elevation`: Site elevation (m above sea level)

#### Climate Features

- `mean_annual_temp`: Mean annual temperature (°C)
- `mean_annual_precip`: Mean annual precipitation (mm)
- `aridity_index`: Precipitation / (Temperature + 10) (simplified aridity)

#### Land Cover Features

- `biome`: Biome classification
- `biome_code`: Encoded biome (1-15)
- `igbp_class`: IGBP land cover classification
- `igbp_code`: Encoded IGBP class (1-16)
- `country`: Country name

#### Derived Geographic Features

- `climate_zone`: Climate zone based on latitude (Temperate_South/Tropical/Temperate_North)
- `climate_zone_code`: Numeric climate zone (0=Temperate_South, 1=Tropical, 2=Temperate_North)
- `latitude_abs`: Absolute latitude value

### 8. Stand Metadata Features

#### Stand Characteristics

- `stand_age`: Stand age (years)
- `basal_area`: Stand basal area (m² ha⁻¹)
- `tree_density`: Tree density (trees ha⁻¹)
- `stand_height`: Stand height (m)
- `leaf_area_index`: Leaf area index (m² m⁻²)

#### Soil Characteristics

- `clay_percentage`: Soil clay content (%)
- `sand_percentage`: Soil sand content (%)
- `silt_percentage`: Soil silt content (%)
- `soil_depth`: Soil depth (cm)
- `soil_texture`: USDA soil texture classification

#### Terrain and Management

- `aspect`: Terrain aspect (degrees)
- `terrain`: Terrain type
- `growth_condition`: Growth condition classification

### 9. Species Metadata Features

#### Species Characteristics

- `species_name`: Tree species name
- `leaf_habit`: Leaf habit (evergreen/deciduous)
- `leaf_habit_code`: Encoded leaf habit (1-4)
- `n_trees`: Number of trees of this species
- `species_basal_area_perc`: Species basal area percentage

### 10. Environmental Metadata Features

#### Measurement Protocol

- `measurement_timestep`: Measurement timestep (minutes)
- `measurement_frequency`: Measurements per hour
- `timezone`: Timezone information
- `timezone_offset`: Timezone offset (hours)
- `daylight_time`: Daylight saving time indicator

#### Sensor Specifications

- `swc_shallow_depth`: Shallow soil moisture sensor depth (cm)

#### Measurement Context

- `plant_water_potential_time`: Plant water potential measurement time
- `seasonal_leaf_area`: Seasonal leaf area variation
- `env_remarks`: Environmental measurement remarks

### 11. Plant Metadata Features (Individual Tree Level)

#### Tree Characteristics

- `pl_age`: Individual tree age (years)
- `tree_age_class`: Age class (Young/Mature/Old/Very Old/Ancient)
- `pl_dbh`: Diameter at breast height (cm)
- `tree_size_class`: Size class (Sapling/Small/Medium/Large/Very Large)
- `pl_height`: Tree height (m)
- `pl_leaf_area`: Individual tree leaf area (m²)
- `pl_bark_thick`: Bark thickness (mm)
- `pl_social`: Social status (dominant/codominant/intermediate/suppressed)
- `social_status_code`: Encoded social status (0-3)
- `pl_species`: Individual tree species name

#### Sapwood Characteristics

- `pl_sapw_area`: Sapwood area (cm²)
- `pl_sapw_depth`: Sapwood depth (cm)
- `sapwood_leaf_ratio`: Sapwood area / Leaf area ratio

#### Derived Tree Features

- `tree_volume_index`: DBH² × Height (tree volume proxy)

## Removed Features for Simplicity

The following features have been removed from the pipeline to create a cleaner, more focused feature set:

### Diurnal Cycle Features

- `is_peak_transpiration`: Removed - redundant with `is_peak_sunlight`
- `is_recovery_hours`: Removed - not essential for core modeling

### Seasonal Features

- `is_peak_growing`: Removed - seasonal patterns captured by `ext_rad`
- `is_dormant_season`: Removed - seasonal patterns captured by `ext_rad`

### Climate Zone Features

- `is_tropical`: Removed - redundant with `climate_zone` and `climate_zone_code`
- `is_temperate`: Removed - redundant with `climate_zone` and `climate_zone_code`
- `is_boreal`: Removed - redundant with `climate_zone` and `climate_zone_code`

### Tree Size Class Features

- `is_large_tree`: Removed - redundant with `tree_size_factor`
- `is_medium_tree`: Removed - redundant with `tree_size_factor`
- `is_small_tree`: Removed - redundant with `tree_size_factor`

### Soil Moisture Gradient Features

- `soil_moisture_gradient`: Removed - `swc_deep` data not consistently available

## Data Quality Features

### Quality Filtering

The dataset includes quality filtering based on flag files:

- **Environmental flags**: Removes rows with OUT_WARN and RANGE_WARN for key variables
- **Sap flow flags**: Removes rows with OUT_WARN, RANGE_WARN, or MISSING flags
- **Quality indicators**: Binary flags for correction applications

### Site Exclusion

The pipeline automatically excludes:

- **59 sites with no valid sap flow data**: These sites lack any valid sap flow measurements
- **26 problematic sites with high quality flag rates**: Sites with >10% quality flag rates (optional exclusion)

## Feature Engineering Methodology

### Optimization Approach

The pipeline has been optimized to leverage existing SAPFLUXNET data efficiently:

- **Uses existing seasonal data**: `ext_rad` (extraterrestrial radiation) instead of calculated seasonal features
- **Uses existing time data**: `solar_TIMESTAMP` instead of calculated time features  
- **Simplified stress indices**: Single combined indices instead of multiple redundant calculations
- **Reduced redundancy**: ~40 redundant features removed while maintaining predictive power
- **Better performance**: Less computation, cleaner code, same scientific value

### Key Optimizations Using Existing Data

#### Extraterrestrial Radiation (`ext_rad`)

- **What it is**: Theoretical solar radiation at the top of the atmosphere
- **Why it's perfect**: Already calculated based on latitude, day of year, and solar geometry
- **What we use it for**: Seasonal patterns, solar efficiency calculations, key interactions with VPD and PPFD

#### Solar Timestamp (`solar_TIMESTAMP`)

- **What it is**: Solar-adjusted time accounting for geographic position
- **Why it's better**: More accurate than calculated time features
- **What we use it for**: Solar-adjusted cyclical encoding, more accurate diurnal patterns

#### Simplified Stress Indices

- **Before**: Multiple redundant stress calculations
- **After**: Single combined indices using existing data
- **Examples**: `water_stress_index = swc_shallow / (vpd + ε)`, `stomatal_control_index = vpd × ppfd_in × ext_rad`

### Conditional Feature Creation

**Implementation Note**: Some features are created conditionally based on data availability:

**Soil Moisture Features**:

- `moisture_availability`: Uses shallow soil moisture data (`swc_shallow`)
- `soil_moisture_gradient`: Removed from pipeline as `swc_deep` data is not available

**Tree Features**:

- Tree-specific features are only created if plant metadata is available
- Sapwood and leaf area features require both `pl_sapw_area` and `pl_leaf_area` columns

**Metadata Features**:

- Site, stand, species, and plant features are created based on available metadata files
- Missing metadata results in null values rather than feature exclusion

## Data Quality Assurance

### Quality Filtering Process

1. **Flag file loading**: Load environmental and sap flow quality flags
2. **Environmental filtering**: Remove rows with quality warnings for key variables
3. **Sap flow filtering**: Remove rows with quality issues for specific trees
4. **Missing data removal**: Remove rows with missing sap flow data

### Quality Indicators

- **Correction flags**: Indicate whether azimuthal/radial corrections were applied
- **Sensor specifications**: Manufacturer and method information for quality assessment
- **Measurement protocols**: Timestep and timezone information for consistency

## Usage Recommendations

### For Machine Learning Models

1. **Feature selection**: Use tree-level features for individual tree modeling
2. **Transfer learning**: Use site/stand/species metadata for cross-site predictions
3. **Quality filtering**: Consider using quality indicators as features or for filtering
4. **Temporal features**: Essential for capturing diurnal and seasonal patterns

### For Data Analysis

1. **Site comparisons**: Use metadata features to group similar sites
2. **Quality assessment**: Use flag information to assess data reliability
3. **Sensor effects**: Consider sensor manufacturer and method in analysis
4. **Tree-level analysis**: Use individual tree characteristics for detailed studies

## File Structure

The processed data is saved as individual CSV files:

- `{site}_comprehensive.csv`: Complete processed dataset for each site
- Contains all original data, engineered features, and metadata
- Ready for machine learning model training
- Includes quality filtering and comprehensive feature engineering

## Expected Improvements

With the streamlined feature set, the dataset now provides:

1. **Enhanced Transfer Learning**: Rich site, stand, and species context for cross-site predictions
2. **Individual Tree Modeling**: Detailed tree characteristics for precise predictions
3. **Quality Control**: Data quality filtering and sensor specification tracking
4. **Measurement Context**: Protocol and sensor details for better model interpretation
5. **Physiological Insights**: Tree-level features that capture individual tree responses
6. **Optimized Performance**: Cleaner feature set with reduced redundancy and better computational efficiency

This streamlined dataset maintains all essential predictive features while providing better performance and cleaner code for transpiration modeling.
