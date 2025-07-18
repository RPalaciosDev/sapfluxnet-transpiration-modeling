# QGIS Visualization Guide for Validation Results

## Available GeoJSON Files:

### spatial_validation_sapfluxnet_spatial_external_sites_20250717_203222
- **Validation Type:** spatial_validation
- **Total Sites:** 87
- **Mean R²:** -1377.873
- **Mean RMSE:** 5.128

**Styling Recommendations:**
- Use 'test_r2' field for graduated symbology
- Color ramp: RdYlGn (inverted)
- Classes: 7 (Natural Breaks)
- Symbol size: 3-5 mm

### spatial_validation_no_proxies_sapfluxnet_spatial_external_sites_20250717_214046
- **Validation Type:** spatial_validation_no_proxies
- **Total Sites:** 87
- **Mean R²:** -865.327
- **Mean RMSE:** 5.100

**Styling Recommendations:**
- Use 'test_r2' field for graduated symbology
- Color ramp: RdYlGn (inverted)
- Classes: 7 (Natural Breaks)
- Symbol size: 3-5 mm

### spatial_validation_universal_features_sapfluxnet_spatial_external_sites_20250717_225322
- **Validation Type:** spatial_validation_universal_features
- **Total Sites:** 87
- **Mean R²:** -638.786
- **Mean RMSE:** 4.954

**Styling Recommendations:**
- Use 'test_r2' field for graduated symbology
- Color ramp: RdYlGn (inverted)
- Classes: 7 (Natural Breaks)
- Symbol size: 3-5 mm

### spatial_validation_universal_features_sapfluxnet_spatial_external_sites_20250718_004135
- **Validation Type:** spatial_validation_universal_features
- **Total Sites:** 87
- **Mean R²:** -612.694
- **Mean RMSE:** 5.202

**Styling Recommendations:**
- Use 'test_r2' field for graduated symbology
- Color ramp: RdYlGn (inverted)
- Classes: 7 (Natural Breaks)
- Symbol size: 3-5 mm

## Comparison Analysis:

1. **Load multiple GeoJSON files** as separate layers
2. **Use different colors** for each validation type
3. **Compare performance patterns** across validation strategies
4. **Identify consistent problem areas** across all methods
5. **Export comparison maps** using print layouts

