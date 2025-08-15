## Engineered Features Deep Dive

Related documents: [Pipeline Overview](./PipelineOverview.md) · [Data Pipeline Examples and Usage](./ExamplesUsage.md)

### 1) Design goals

- Robust across sites and formats; minimize identity leakage
- Memory-aware, chunk/parallel capable; avoids loading all data at once
- Preserve missing values as `np.nan`; no arbitrary fills
- Minimal, phase-based logging aligned with reduced verbosity

### 2) Temporal features

- Source: primary timestamp via `DataLoader`
- Derived:
  - Discrete: `hour`, `day_of_year`, `month`, `year`, `day_of_week`
  - Cyclical: `hour_sin/cos`, `day_sin/cos`, `month_sin/cos`
  - Booleans: `is_daylight`, `is_peak_sunlight`, `is_weekend`, `is_morning`, `is_afternoon`, `is_night`
  - Solar-adjusted (if `solar_TIMESTAMP` exists): `solar_hour`, `solar_day_of_year`, `solar_hour_sin/cos`, `solar_day_sin/cos`

### 3) Environmental features

- Columns: `ta`, `rh`, `vpd`, `sw_in`, `ws`, `precip`, `swc_shallow`, `ppfd_in`
- Rolling statistics (windows from `ProcessingConfig.FEATURE_SETTINGS['rolling_windows']`):
  - Always: mean, std; for window ≥ 72h: min, max, range
- Lagged: lags `[1, 2, 3, 6, 12, 24]` hours
- Rate-of-change: diffs over `[1, 6, 24]` hours
- Cumulative:
  - Precip: `precip_cum_24h`, `precip_cum_72h`, `precip_cum_168h`
  - Radiation: `sw_in_cum_24h`, `sw_in_cum_72h`

### 4) Interaction features (always created)

- `vpd_ppfd_interaction = vpd * ppfd_in`
- `vpd_ta_interaction = vpd * ta`
- `temp_humidity_ratio = ta / (rh + 1e-6)`
- `water_stress_index = swc_shallow / (vpd + 1e-6)`
- `light_efficiency = ppfd_in / (sw_in + 1e-6)`
- `temp_soil_interaction = ta * swc_shallow`
- `wind_vpd_interaction = ws * vpd`
- `radiation_temp_interaction = sw_in * ta`
- `humidity_soil_interaction = rh * swc_shallow`

### 5) Metadata features

- Loaded from site/stand/species/environmental/plant metadata via `FileManager.load_metadata(site)`
- Added when available:
  - Site: `latitude`, `longitude`, `elevation`, `mean_annual_temp`, `mean_annual_precip`, `biome`, `igbp_class`, `country`, `site_code`, `site_name`, `is_inside_country`
  - Stand: `stand_age`, `basal_area`, `tree_density`, `stand_height`, `leaf_area_index`, `clay/sand/silt_percentage`, `soil_depth`, `soil_texture`, `terrain`, `growth_condition`
  - Species: `species_name`, `leaf_habit`, `n_trees`
  - Environmental: `measurement_timestep`, `timezone`
  - Plant (merged by `plant_id`/`pl_code`): `pl_age`, `pl_dbh`, `pl_height`, `pl_leaf_area`, `pl_bark_thick`, `pl_social`, `pl_species`, `pl_sapw_area`, `pl_sapw_depth`

### 6) Derived metadata features

- `latitude_abs = |latitude|`
- Köppen–Geiger: `koppen_geiger_code` (vectorized from `mean_annual_temp`, `mean_annual_precip`, `latitude`)
- `aridity_index = mean_annual_precip / (mean_annual_temp + 10)`
- Encoded categoricals (to float codes): `leaf_habit_code`, `biome_code`, `igbp_code`
- Binnings:
  - `tree_size_class` from `pl_dbh` → `Sapling|Small|Medium|Large|Very Large`
  - `tree_age_class` from `pl_age` → `Young|Mature|Old|Very Old|Ancient`
- Social status: `social_status_code` from `pl_social`
- Composites:
  - `sapwood_leaf_ratio = pl_sapw_area / (pl_leaf_area + 1e-6)`
  - `tree_volume_index = (pl_dbh^2) * pl_height`
- Timezone/measurement:
  - `timezone_offset` parsed from `timezone`
  - `measurement_frequency = 60 / measurement_timestep`

### 7) Domain-specific ecophysiological features

- Light: `ppfd_efficiency = ppfd_in / (sw_in + 1e-6)`
- Temperature deviation: `temp_deviation = |ta - median(ta)|` (median fallback 25.0 if all missing)
- Stomatal proxy: `stomatal_conductance_proxy = ppfd_in / (vpd + 1e-6)`
- Wind: `wind_stress = ws / (max(ws) + 1e-6)`, `wind_vpd_interaction = ws * vpd`
- Extraterrestrial radiation (if `ext_rad` exists):
  - `stomatal_control_index = vpd * ppfd_in * ext_rad`
  - `light_efficiency = ppfd_in / (ext_rad + 1e-6)`
- Tree-related: `tree_size_factor = log(pl_dbh + 1)`, `transpiration_capacity = pl_sapw_area * ppfd_in / (vpd + 1e-6)`

Note: `light_efficiency` is created in interactions (uses `sw_in`) and again in domain features (uses `ext_rad`) if available. The latter overwrites the former when both present. Consider renaming to `light_efficiency_sw` and `light_efficiency_ext` in a future revision.

### 8) Seasonality features (site-level; skipped in streaming)

- Monthly aggregations per `site` for `ta` and `precip`:
  - `ta_min/ta_max`, `precip_min/precip_max`
  - `seasonal_temp_range = ta_max - ta_min`
  - `seasonal_precip_range = precip_max - precip_min`
- Merged back by `site`; remaining NaNs filled with `0.0` for these two range features

### 9) Categorical encoding and leakage protection

- Identity blacklist first: drop `IDENTITY_BLACKLIST` columns
- Geography handling:
  - Drop `PURE_GEOGRAPHIC_IDENTIFIERS` (e.g., `timezone`, `country`, `region`, ...)
  - Allow climate-geographic (`biome_region`, `koppen_class`, `climate_classification`)
- Species handling:
  - `species_name` → `species_functional_group` via exact and genus-level matches from `SPECIES_FUNCTIONAL_GROUPS`; unmatched → `unknown`; drop original `species_name`
- Approved ecological categoricals encoded via `CATEGORICAL_ENCODINGS`:
  - `biome`, `igbp_class`, `soil_texture`, `aspect`, `terrain`, `growth_condition`, `leaf_habit`, `pl_social`, `tree_size_class`, `tree_age_class`, `species_functional_group`, and `koppen_geiger_code` (via factorize)
- Remaining `object` columns:
  - Preserve: `TIMESTAMP`, `solar_TIMESTAMP`, `plant_id`, `sap_flow`, `site`
  - Convert to numeric when safe; conservative encoding for low-cardinality text; drop high-cardinality identity-like text

### 10) Schema finalization and compatibility

- Drop any columns in `PROBLEMATIC_COLUMNS_TO_EXCLUDE` that slipped through
- Ensure base expected schema (core environmental, soils, metadata, plant features)
- Conditionally ensure temporal/interaction features exist per config flags
- Add XGBoost-inconsistent columns if missing: `leaf_habit_code`, `soil_depth`
- Clean invalid tokens to `np.nan`; log final column count in the "Ensuring Data Compatibility" phase

### 11) Memory, chunking, and parallelism inside FeatureEngineer

- Determine chunking from DataFrame memory, thresholds, and available memory
- If chunking enabled: split rows, process with chunk-local `FeatureEngineer`, then concatenate
- Parallelize independent groups (environmental, interaction, domain, metadata) when enabled

### 12) Logging (key phases)

- Always logs phase start/complete for:
  - Engineering Features
  - Encoding Features
  - Dropping Problematic Columns
  - Ensuring Data Compatibility
