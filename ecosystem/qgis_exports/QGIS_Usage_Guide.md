# QGIS Usage Guide for SAPFLUXNET Analysis

## 📁 Files Overview

### Vector Layers (Points & Polygons)

- **`sapflux_sites_20250724_014920.geojson`** - 87 site locations with attributes
- **`cluster_boundaries_20250724_014920.shp`** - 5 ecosystem cluster boundaries

### Raster Layer (Continuous Surface)

- **`performance_interpolation_20250724_014920.tif`** - Model performance interpolation

## 🎨 Styling Recommendations

### 1. Site Points (sapflux_sites_*.geojson)

**Attributes Available:**

- `site` - Site name
- `eco_clust` - Ecosystem cluster (0-4)
- `temp_c` - Mean annual temperature (°C)
- `precip_mm` - Mean annual precipitation (mm)
- `elevation` - Elevation (m)
- `spatial_r2` - Spatial validation R² score
- `spatial_rm` - Spatial validation RMSE
- `fold_count` - Number of successful validation folds

**Styling Options:**

- **Color by Cluster:** Use `eco_clust` field with categorical colors
- **Size by Performance:** Use `spatial_r2` field for point size
- **Color by Climate:** Use `temp_c` or `precip_mm` with color ramp

### 2. Cluster Boundaries (cluster_boundaries_*.shp)

**Attributes Available:**

- `cluster_id` - Cluster ID (0-4)
- `site_count` - Number of sites in cluster
- `avg_temp` - Average temperature for cluster
- `avg_precip` - Average precipitation for cluster
- `avg_r2` - Average R² performance for cluster

**Styling Options:**

- **Fill by Performance:** Use `avg_r2` with color ramp
- **Transparency:** Set to 50-70% to see underlying layers
- **Outline:** Use contrasting colors for cluster boundaries

### 3. Performance Raster (performance_interpolation_*.tif)

**Data:** Continuous R² values (0-1)

**Styling Options:**

- **Color Ramp:** Use "Spectral" or "RdYlGn" (red=low, green=high)
- **Transparency:** Set to 60-80% to overlay with other layers
- **Min/Max Values:** Stretch to actual data range

## 📊 Analysis Workflows

### Workflow 1: Cluster Analysis

1. **Load cluster boundaries** (polygons)
2. **Style by average performance** (`avg_r2`)
3. **Add site points** on top
4. **Color sites by cluster** to verify groupings
5. **Add labels** showing cluster statistics

### Workflow 2: Performance Mapping

1. **Load performance raster** as base layer
2. **Style with performance color ramp**
3. **Add site points** colored by actual performance
4. **Compare interpolated vs actual** values
5. **Identify performance hotspots**

### Workflow 3: Climate-Performance Relationship

1. **Load site points**
2. **Create multiple maps:**
   - Sites colored by temperature
   - Sites colored by precipitation
   - Sites colored by performance
3. **Use layout manager** for multi-panel comparison

## 🔧 QGIS Import Steps

### Method 1: Drag & Drop

1. Open QGIS
2. Open file explorer to `ecosystem/qgis_exports/`
3. Drag files directly into QGIS map canvas

### Method 2: Layer Menu

1. **Vector:** `Layer` → `Add Layer` → `Add Vector Layer`
2. **Raster:** `Layer` → `Add Layer` → `Add Raster Layer`
3. Browse to files and add

### Method 3: Browser Panel

1. In QGIS Browser panel, navigate to folder
2. Double-click files to add to map

## 🎯 Key Insights to Explore

1. **Cluster Separation:** Do cluster boundaries make ecological sense?
2. **Performance Patterns:** Are there geographic patterns in model performance?
3. **Climate Relationships:** How does performance relate to temperature/precipitation?
4. **Outliers:** Which sites have unexpectedly high/low performance?
5. **Spatial Autocorrelation:** Are nearby sites more similar in performance?

## 📈 Advanced Analysis Ideas

1. **Create buffer zones** around high-performance sites
2. **Calculate cluster centroids** and distances
3. **Overlay with environmental data** (if available)
4. **Export styled maps** for presentations
5. **Create animated maps** showing temporal patterns (if time series data available)

## 💡 Tips

- **Layer Order:** Raster → Polygons → Points (bottom to top)
- **Coordinate System:** All data is in WGS84 (EPSG:4326)
- **Attribute Table:** Right-click layer → "Open Attribute Table" to see all data
- **Identify Tool:** Click on features to see attribute values
- **Export Maps:** `Project` → `Import/Export` → `Export Map to Image`

## 🔍 Troubleshooting

- **Files not loading:** Check file paths are correct
- **Missing attributes:** Use GeoJSON instead of Shapefile for full attributes
- **Raster not displaying:** Check if raster has valid data range
- **Projection issues:** All files should be in WGS84, reproject if needed
