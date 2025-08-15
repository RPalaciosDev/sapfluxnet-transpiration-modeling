## Clustering Feature-Sets Deep Dive

Related documents: [Clustering Pipeline Overview](./ClusteringPipelineOverview.md) · [Clustering Examples and Usage](./ClusteringExamplesUsage.md)

### Purpose

- Enumerate available clustering feature sets, what they include, dependencies on processed pipeline outputs, and expected preprocessing.

### Preliminaries

- Source: `ecosystem/clustering/feature_definitions.py` (managed by `FeatureManager`)
- Each `FeatureSet` defines:
  - `numeric_features`: numeric inputs assumed present in processed parquet
  - `categorical_features`: categorical inputs assumed already encoded (or encoded by the clustering preprocessor if needed)
- Many columns originate from the data processing pipeline (e.g., `koppen_geiger_code_encoded`, functional group codes, seasonal ranges).

### Feature-set catalog

- geographic
  - numeric: `longitude`, `latitude`
  - categorical: —
  - notes: purely spatial grouping; fast; minimal requirements

- biome
  - numeric: `longitude`, `latitude`, `mean_annual_temp`, `mean_annual_precip`, `seasonal_temp_range`, `seasonal_precip_range`, `koppen_geiger_code_encoded`
  - categorical: `biome_code`, `igbp_class_code`
  - deps: seasonal ranges, encoded Köppen, biome/IGBP codes

- climate
  - numeric: `longitude`, `latitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `seasonal_temp_range`, `seasonal_precip_range`, `koppen_geiger_code_encoded`
  - categorical: `biome_code`, `igbp_class_code`
  - deps: same as biome + elevation

- ecological
  - numeric: `longitude`, `latitude`, `elevation`, `basal_area`, `tree_density`, `leaf_area_index`, `koppen_geiger_code_encoded`
  - categorical: `species_functional_group_code`, `leaf_habit_code`, `biome_code`, `igbp_class_code`
  - deps: structural (basal_area, tree_density, LAI), species functional encodings

- comprehensive
  - numeric: `longitude`, `latitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `seasonal_temp_range`, `seasonal_precip_range`, `koppen_geiger_code_encoded`
  - categorical: `species_functional_group_code`, `leaf_habit_code`, `biome_code`, `igbp_class_code`

- performance
  - numeric: `longitude`, `latitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `mean_sap_flux`, `max_sap_flux`, `sap_flux_variability` (if available)
  - categorical: `biome_code`, `igbp_class_code`
  - deps: aggregated sap flux summaries (optional)

- environmental
  - numeric: `longitude`, `latitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `seasonal_temp_range`, `seasonal_precip_range`, `mean_ta`, `mean_rh`, `mean_vpd`, `mean_sw_in`, `mean_precip`, `mean_ws`, `koppen_geiger_code_encoded`
  - categorical: —
  - deps: environmental aggregates (optional)

- plant_functional
  - numeric: —
  - categorical: `species_functional_group_code`

- v2_core (legacy)
  - numeric: `mean_annual_temp`, `mean_annual_precip`, `aridity_index`, `latitude_abs`, `elevation`, `seasonal_temp_range`, `seasonal_precip_range`
  - categorical: —

- v2_advanced (legacy)
  - numeric: `temp_precip_ratio`, `seasonality_index`, `climate_continentality`, `elevation_latitude_ratio`, `aridity_seasonality`, `temp_elevation_ratio`, `precip_latitude_ratio`
  - categorical: —
  - deps: derived metrics must exist in parquet (only if exported by pipeline)

- v2_hybrid (legacy)
  - numeric: `longitude`, `latitude`, `latitude_abs`, `mean_annual_temp`, `mean_annual_precip`, `aridity_index`, `elevation`, `seasonal_temp_range`, `seasonal_precip_range`, `temp_precip_ratio`, `seasonality_index`, `climate_continentality`, `elevation_latitude_ratio`, `aridity_seasonality`, `temp_elevation_ratio`, `precip_latitude_ratio`
  - categorical: `climate_zone_code`, `biome_code`, `igbp_class_code`, `leaf_habit_code`, `soil_texture_code`, `species_functional_group_code`, `koppen_geiger_code_encoded`

- v3_hybrid (legacy)
  - numeric: `longitude`, `latitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `seasonal_temp_range`, `seasonal_precip_range`, `basal_area`, `tree_density`, `leaf_area_index`
  - categorical: `species_functional_group_code`, `leaf_habit_code`, `biome_code`, `igbp_class_code`

- advanced_core (legacy)
  - numeric: `mean_annual_temp`, `mean_annual_precip`, `aridity_index`, `latitude_abs`, `elevation`, `seasonal_temp_range`, `seasonal_precip_range`
  - categorical: —

- advanced_derived (legacy)
  - numeric: `temp_precip_ratio`, `seasonality_index`, `climate_continentality`, `elevation_latitude_ratio`, `aridity_seasonality`, `temp_elevation_ratio`, `precip_latitude_ratio`
  - categorical: —

- advanced_hybrid (legacy)
  - numeric (full hybrid):
    - geographic/climate/derived: see v2_hybrid
    - structure: `tree_volume_index`, `stand_age`, `n_trees`, `tree_density`, `basal_area`, `stand_height`, `tree_size_class_code`, `sapwood_leaf_ratio`, `pl_dbh`, `pl_age`, `pl_height`, `pl_leaf_area`, `pl_sapw_depth`, `pl_sapw_area`
  - categorical: `climate_zone_code`, `biome_code`, `igbp_class_code`, `leaf_habit_code`, `soil_texture_code`, `terrain_code`, `species_functional_group_code`, `koppen_geiger_code_encoded`

### Availability and dependencies

- Often rely on processed pipeline columns:
  - Climate: `mean_annual_temp`, `mean_annual_precip`, `aridity_index`, `koppen_geiger_code_encoded`
  - Seasonality: `seasonal_temp_range`, `seasonal_precip_range`
  - Encodings: `biome_code`, `igbp_class_code`, `leaf_habit_code`, `species_functional_group_code`
  - Structure (if present): `basal_area`, `leaf_area_index`, `tree_density`, tree metrics and derived indices
- If availability < minimum threshold, the preprocessor suggests alternative feature sets with higher coverage.

### Preprocessing behavior (per feature set)

- Missing handling: median (default), mean, zero, or drop
- Categorical encoding: `LabelEncoder` applied to requested categorical fields if present
- Scaling: `StandardScaler` fitted on selected numeric features prior to clustering

### Strategy considerations

- Sets with many categorical codes (after encoding) can influence scaling; ensure sufficient numeric variation when applicable.
- Climate-driven sets tend to produce geographically coherent clusters when climate gradients dominate.
- Structure-heavy sets rely on richer site metadata and may emphasize management/stand effects.
