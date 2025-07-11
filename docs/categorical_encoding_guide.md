# Categorical Encoding Guide

This document provides the complete mapping between original text values and their numeric encodings used in the comprehensive processed dataset.

## Overview

All categorical text variables have been converted to numeric codes for machine learning compatibility. This guide shows what each code represents.

---

## 1. Biome Classification

| Code | Original Value |
|------|----------------|
| 1 | Tropical and Subtropical Moist Broadleaf Forests |
| 2 | Tropical and Subtropical Dry Broadleaf Forests |
| 3 | Tropical and Subtropical Coniferous Forests |
| 4 | Temperate Broadleaf and Mixed Forests |
| 5 | Temperate Conifer Forests |
| 6 | Boreal Forests/Taiga |
| 7 | Tropical and Subtropical Grasslands, Savannas and Shrublands |
| 8 | Temperate Grasslands, Savannas and Shrublands |
| 9 | Flooded Grasslands and Savannas |
| 10 | Montane Grasslands and Shrublands |
| 11 | Tundra |
| 12 | Mediterranean Forests, Woodlands and Scrub |
| 13 | Deserts and Xeric Shrublands |
| 14 | Mangroves |
| 15 | Woodland/Shrubland |

**Column**: `biome_code`

---

## 2. IGBP Land Cover Classification

| Code | Original Value | Description |
|------|----------------|-------------|
| 1 | ENF | Evergreen Needleleaf Forests |
| 2 | EBF | Evergreen Broadleaf Forests |
| 3 | DNF | Deciduous Needleleaf Forests |
| 4 | DBF | Deciduous Broadleaf Forests |
| 5 | MF | Mixed Forests |
| 6 | CSH | Closed Shrublands |
| 7 | OSH | Open Shrublands |
| 8 | WSA | Woody Savannas |
| 9 | SAV | Savannas |
| 10 | GRA | Grasslands |
| 11 | WET | Permanent Wetlands |
| 12 | CRO | Croplands |
| 13 | URB | Urban and Built-up |
| 14 | CVM | Cropland/Natural Vegetation Mosaics |
| 15 | SNO | Snow and Ice |
| 16 | BSV | Barren or Sparsely Vegetated |

**Column**: `igbp_code`

---

## 3. Country Codes

| Code | Country |
|------|---------|
| 1 | Argentina (ARG) |
| 2 | Australia (AUS) |
| 3 | Austria (AUT) |
| 4 | Brazil (BRA) |
| 5 | Canada (CAN) |
| 6 | Switzerland (CHE) |
| 7 | China (CHN) |
| 8 | Colombia (COL) |
| 9 | Czech Republic (CZE) |
| 10 | Germany (DEU) |
| 11 | Spain (ESP) |
| 12 | Finland (FIN) |
| 13 | France (FRA) |
| 14 | United Kingdom (GBR) |
| 15 | French Guiana (GUF) |
| 16 | Japan (JPN) |
| 17 | South Korea (KOR) |
| 18 | Madagascar (MDG) |
| 19 | Portugal (PRT) |
| 20 | Russia (RUS) |
| 21 | Senegal (SEN) |
| 22 | Sweden (SWE) |
| 23 | United States (USA) |
| 24 | South Africa (ZAF) |

**Column**: `country_code`

---

## 4. Soil Texture Classification

| Code | Soil Texture |
|------|-------------|
| 1 | clay |
| 2 | clay loam |
| 3 | loam |
| 4 | loamy sand |
| 5 | sandy clay |
| 6 | sandy clay loam |
| 7 | sandy loam |
| 8 | sand |
| 9 | silt |
| 10 | silt loam |
| 11 | silty clay |
| 12 | silty clay loam |

**Column**: `soil_texture_code`

---

## 5. Terrain Aspect

| Code | Aspect |
|------|--------|
| 1 | N (North) |
| 2 | NE (Northeast) |
| 3 | E (East) |
| 4 | SE (Southeast) |
| 5 | S (South) |
| 6 | SW (Southwest) |
| 7 | W (West) |
| 8 | NW (Northwest) |

**Column**: `aspect_code`

---

## 6. Terrain Type

| Code | Terrain |
|------|---------|
| 1 | Flat |
| 2 | Gentle slope (<2 %) |
| 3 | Moderate slope (2-10 %) |
| 4 | Steep slope (>10 %) |
| 5 | Valley |
| 6 | Ridge |

**Column**: `terrain_code`

---

## 7. Growth Condition

| Code | Growth Condition |
|------|------------------|
| 1 | Naturally regenerated, managed |
| 2 | Naturally regenerated, unmanaged |
| 3 | Planted, managed |
| 4 | Planted, unmanaged |

**Column**: `growth_condition_code`

---

## 8. Leaf Habit

| Code | Leaf Habit |
|------|-----------|
| 1 | cold deciduous |
| 2 | warm deciduous |
| 3 | evergreen |
| 4 | semi-deciduous |

**Column**: `leaf_habit_code`

---

## 9. Tree Social Status

| Code | Social Status |
|------|---------------|
| 0 | suppressed |
| 1 | intermediate |
| 2 | codominant |
| 3 | dominant |

**Column**: `social_status_code`

---

## 10. Sensor Manufacturer

| Code | Manufacturer |
|------|--------------|
| 1 | ICT International |
| 2 | Dynamax |
| 3 | UP GmbH |
| 4 | Ecomatik |
| 5 | Delta-T Devices |

**Column**: `sensor_manufacturer_code`

---

## 11. Sensor Method

| Code | Method | Full Name |
|------|--------|-----------|
| 1 | HR | Heat Ratio |
| 2 | CHP | Compensation Heat Pulse |
| 3 | TDP | Thermal Dissipation Probe |
| 4 | SHB | Stem Heat Balance |
| 5 | HFD | Heat Field Deformation |

**Column**: `sensor_method_code`

---

## 12. Climate Zone

| Code | Climate Zone |
|------|--------------|
| 0 | Temperate_South |
| 1 | Tropical |
| 2 | Temperate_North |

**Column**: `climate_zone_code`

---

## 13. Tree Size Class

| Code | Size Class | DBH Range (cm) |
|------|------------|----------------|
| 0 | Sapling | 0-10 |
| 1 | Small | 10-30 |
| 2 | Medium | 30-50 |
| 3 | Large | 50-100 |
| 4 | Very Large | 100+ |

**Column**: `tree_size_class`

---

## 14. Tree Age Class

| Code | Age Class | Age Range (years) |
|------|-----------|-------------------|
| 0 | Young | 0-20 |
| 1 | Mature | 20-50 |
| 2 | Old | 50-100 |
| 3 | Very Old | 100-200 |
| 4 | Ancient | 200+ |

**Column**: `tree_age_class`

---

## 15. Sap Flow Units

| Code | Units |
|------|-------|
| 0 | cm³ cm⁻² h⁻¹ |
| 1 | g h⁻¹ |
| 2 | kg h⁻¹ |
| 3 | L h⁻¹ |

**Column**: `sap_units_code`

---

## 16. Quality Indicators

### Azimuthal Correction

| Code | Correction Status |
|------|------------------|
| 0 | No azimuthal correction |
| 1 | Corrected, measured azimuthal variation |

**Column**: `has_azimuthal_correction`

### Radial Correction

| Code | Correction Status |
|------|------------------|
| 0 | No radial correction |
| 1 | Corrected, measured radial variation |

**Column**: `has_radial_correction`

---

## 17. Time Features

### Daylight Time

| Code | Status |
|------|--------|
| 0 | FALSE |
| 1 | TRUE |

**Column**: `is_daylight`

### Peak Sunlight

| Code | Status |
|------|--------|
| 0 | Not peak sunlight hours |
| 1 | Peak sunlight hours (10-16) |

**Column**: `is_peak_sunlight`

---

## 18. Derived Features

### Tree Volume Index

- **Formula**: DBH² × Height
- **Units**: cm² × m
- **Purpose**: Proxy for tree volume

### Sapwood-Leaf Ratio

- **Formula**: Sapwood Area / Leaf Area
- **Units**: cm² / m²
- **Purpose**: Efficiency indicator

### Aridity Index

- **Formula**: Mean Annual Precipitation / (Mean Annual Temperature + 10)
- **Purpose**: Climate aridity indicator

---

## Usage Notes

1. **Missing Values**: NaN values in encoded columns indicate missing data
2. **Unknown Categories**: New categories not in the mapping will be assigned sequential codes
3. **Ordinal Relationships**: Some codes preserve ordinal relationships (e.g., social status, tree size)
4. **Nominal Categories**: Other codes are purely nominal (e.g., country, biome)

## For Machine Learning

- **Feature Selection**: Use `*_code` columns for training
- **Interpretation**: Use this guide to interpret model results
- **Transfer Learning**: Consistent encodings across sites enable cross-site predictions
- **Feature Importance**: Codes preserve meaningful relationships for model learning
