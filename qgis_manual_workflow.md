# QGIS Manual Workflow for SAPFLUXNET Spatial Visualization

## Overview

This guide provides step-by-step instructions for creating professional spatial visualizations of your XGBoost model performance using QGIS.

## Prerequisites

- QGIS 3.28+ installed
- Python 3.8+ with pandas, numpy
- Your SAPFLUXNET spatial data and performance results

## Step 1: Data Preparation

### 1.1 Create Enhanced Performance Dataset

First, let's create a more comprehensive dataset that includes geographic information:

```python
import pandas as pd
import geopandas as gpd

# Load performance data
performance_df = pd.read_csv('xgboost_scripts/external_memory_models/spatial_validation/sapfluxnet_spatial_external_sites_20250717_203222.csv')

# Load sites shapefile
sites_gdf = gpd.read_file('shapefiles/sapfluxnet_sites.shp')

# Merge performance data with spatial data
# Note: You may need to adjust the join column based on your shapefile structure
merged_data = sites_gdf.merge(performance_df, left_on='SITE_ID', right_on='site', how='inner')

# Save as GeoJSON for easy QGIS import
merged_data.to_file('sapfluxnet_performance_sites.geojson', driver='GeoJSON')
```

### 1.2 Performance Analysis Summary

```python
# Basic statistics
print("Performance Summary:")
print(f"Total sites: {len(performance_df)}")
print(f"Mean R²: {performance_df['test_r2'].mean():.3f}")
print(f"Median R²: {performance_df['test_r2'].median():.3f}")
print(f"Mean RMSE: {performance_df['test_rmse'].mean():.3f}")

# Performance categories
def categorize_performance(r2):
    if r2 >= 0.7: return "Excellent"
    elif r2 >= 0.5: return "Good"
    elif r2 >= 0.3: return "Fair"
    elif r2 >= 0: return "Poor"
    else: return "Very Poor"

performance_df['category'] = performance_df['test_r2'].apply(categorize_performance)
print("\nPerformance by Category:")
print(performance_df['category'].value_counts())
```

## Step 2: QGIS Project Setup

### 2.1 Create New QGIS Project

1. Open QGIS
2. Create new project: `Project → New`
3. Save project as `sapfluxnet_spatial_analysis.qgz`

### 2.2 Add Base Layers

1. **Add OpenStreetMap as basemap:**
   - Install "QuickMapServices" plugin if not already installed
   - `Web → QuickMapServices → OSM → OSM Standard`

2. **Add study area boundary:**
   - `Layer → Add Layer → Add Vector Layer`
   - Select `shapefiles/sapfluxnet_study_area.shp`
   - Style with transparent fill, dark outline

3. **Add sites layer:**
   - `Layer → Add Layer → Add Vector Layer`
   - Select `sapfluxnet_performance_sites.geojson` (created in Step 1)

## Step 3: Create Performance Visualizations

### 3.1 R² Performance Map

1. **Select the performance sites layer**
2. **Open Layer Properties → Symbology**
3. **Choose "Graduated" renderer**
4. **Set Value to "test_r2"**
5. **Configure classes:**
   - Method: Natural Breaks (Jenks)
   - Classes: 7
   - Color ramp: RdYlGn (Red-Yellow-Green)
   - Invert color ramp (green = good performance)

6. **Set symbol size:**
   - Size: 3-8 mm (graduated by sample size)
   - Size field: "test_samples"

### 3.2 RMSE Performance Map

1. **Duplicate the performance layer**
2. **Rename to "RMSE Performance"**
3. **Change value field to "test_rmse"**
4. **Use color ramp: YlOrRd (Yellow-Orange-Red)**
   - Don't invert (red = high RMSE = poor performance)

### 3.3 Performance Category Map

1. **Create another duplicate layer**
2. **Rename to "Performance Categories"**
3. **Change to "Categorized" renderer**
4. **Set value to "category"**
5. **Use distinct colors for each category:**
   - Excellent: Dark Green (#006400)
   - Good: Green (#32CD32)
   - Fair: Yellow (#FFD700)
   - Poor: Orange (#FF8C00)
   - Very Poor: Red (#DC143C)

## Step 4: Advanced Styling

### 4.1 Label Sites

1. **Layer Properties → Labels**
2. **Enable labels**
3. **Set label field to "site"**
4. **Configure text style:**
   - Font: Arial, 8pt
   - Color: Black
   - Buffer: White, 1mm

### 4.2 Add Popups

1. **Layer Properties → Attributes Form**
2. **Configure fields to display:**
   - Site ID
   - R² value
   - RMSE value
   - Sample count
   - Performance category

### 4.3 Create Legend

1. **Layer Properties → Symbology**
2. **Add legend text:**
   - "Model Performance (R²)"
   - "Excellent: R² ≥ 0.7"
   - "Good: 0.5 ≤ R² < 0.7"
   - "Fair: 0.3 ≤ R² < 0.5"
   - "Poor: 0 ≤ R² < 0.3"
   - "Very Poor: R² < 0"

## Step 5: Create Print Layout

### 5.1 Setup Layout

1. **Project → New Print Layout**
2. **Name: "SAPFLUXNET Performance Map"**
3. **Add map item:**
   - Size: A3 landscape
   - Extent: Fit to layer extent

### 5.2 Add Map Elements

1. **Add title:**
   - Text: "SAPFLUXNET XGBoost Model Performance - Spatial Validation"
   - Font: Arial, 16pt, Bold

2. **Add legend:**
   - Include all performance layers
   - Title: "Model Performance Metrics"

3. **Add scale bar:**
   - Style: Single box
   - Units: kilometers

4. **Add north arrow:**
   - Style: Simple arrow

5. **Add text box with statistics:**

   ```
   Performance Summary:
   • Total Sites: 89
   • Mean R²: 0.234
   • Median R²: 0.426
   • Sites with R² < 0: 15
   • Sites with R² > 0.5: 28
   ```

### 5.3 Export Options

1. **Layout → Export as Image**
2. **Resolution: 300 DPI**
3. **Format: PNG or PDF**
4. **Save as: `sapfluxnet_performance_map.png`**

## Step 6: Advanced Analysis

### 6.1 Geographic Clustering

1. **Install "Heatmap" plugin**
2. **Raster → Heatmap → Heatmap**
3. **Input: Performance sites layer**
4. **Radius: 1000 meters**
5. **Output: Performance heatmap**

### 6.2 Elevation Analysis

If you have elevation data:

1. **Add DEM layer**
2. **Extract elevation values to sites**
3. **Create scatter plot: Elevation vs R²**

### 6.3 Climate Zone Analysis

If you have climate data:

1. **Add climate zones layer**
2. **Join with performance data**
3. **Calculate mean performance by climate zone**

## Step 7: Interactive Web Map

### 7.1 Export to Web

1. **Install "qgis2web" plugin**
2. **Web → qgis2web**
3. **Configure options:**
   - Format: Leaflet
   - Extent: Fit to layers
   - Popups: Enabled
   - Legend: Enabled
4. **Export to folder: `web_map/`**

## Troubleshooting

### Common Issues

1. **Layer not displaying:**
   - Check coordinate reference system (CRS)
   - Ensure layers use same CRS (EPSG:4326 recommended)

2. **Performance data not joining:**
   - Verify site ID field names match
   - Check for extra spaces or special characters

3. **Colors not showing correctly:**
   - Ensure field contains numeric data
   - Check for NULL values

### Performance Tips

1. **Use spatial indexes** for large datasets
2. **Simplify geometries** for faster rendering
3. **Use appropriate zoom levels** for detail

## Next Steps

1. **Create temporal analysis** - performance over time
2. **Add environmental variables** - climate, soil, vegetation
3. **Perform spatial autocorrelation** analysis
4. **Create uncertainty maps** - confidence intervals
5. **Develop interactive dashboard** with QGIS Web Client

## Resources

- [QGIS Documentation](https://docs.qgis.org/)
- [SAPFLUXNET Data Portal](https://sapfluxnet.creaf.cat/)
- [Spatial Analysis with QGIS](https://www.qgistutorials.com/en/)
