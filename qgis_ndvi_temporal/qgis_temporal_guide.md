# QGIS Temporal NDVI Visualization Guide

## Overview
This guide provides step-by-step instructions for creating temporal NDVI visualizations in QGIS.

## Prerequisites
- QGIS 3.28+ installed
- QuickMapServices plugin installed
- Temporal Controller plugin enabled

## Step 1: Load Temporal Data

### 1.1 Load Monthly NDVI Data
1. Open QGIS
2. Layer → Add Layer → Add Vector Layer
3. Select: `qgis_ndvi_temporal/ndvi_monthly.geojson`
4. Right-click layer → Properties → Symbology
5. Choose "Graduated" renderer
6. Set Value to "ndvi_mean"
7. Use color ramp: "RdYlGn" (Red-Yellow-Green)
8. Set 7 classes with natural breaks

### 1.2 Load Seasonal Data
1. Add: `qgis_ndvi_temporal/ndvi_seasonal.geojson`
2. Style by "ndvi_mean" with seasonal color scheme

### 1.3 Load Annual Data
1. Add: `qgis_ndvi_temporal/ndvi_annual.geojson`
2. Style by "ndvi_mean" with annual trends

## Step 2: Enable Temporal Controller

### 2.1 Setup Temporal Controller
1. View → Panels → Temporal Controller
2. In Temporal Controller panel:
   - Set "Temporal Range" to cover your data period
   - Set "Frame Duration" to 1 month
   - Enable "Loop" for continuous playback

### 2.2 Configure Temporal Properties
1. Right-click monthly layer → Properties → Temporal
2. Enable "Temporal"
3. Set "Temporal Field" to "date_str"
4. Set "Temporal Range" to match your data

## Step 3: Create Temporal Animations

### 3.1 Basic Animation
1. In Temporal Controller, click "Play"
2. Watch NDVI changes over time
3. Use "Step Forward/Backward" for manual control

### 3.2 Export Animation
1. Project → New Print Layout
2. Add Map to layout
3. In Temporal Controller, set export settings:
   - Frame rate: 1 fps
   - Output directory: choose location
   - Format: PNG or MP4
4. Click "Export Animation"

## Step 4: Advanced Temporal Analysis

### 4.1 Seasonal Patterns
1. Load seasonal data
2. Filter by specific seasons
3. Compare seasonal NDVI patterns
4. Create seasonal difference maps

### 4.2 Trend Analysis
1. Load annual data
2. Calculate NDVI trends over time
3. Create trend maps showing increasing/decreasing vegetation

### 4.3 Multi-temporal Comparison
1. Load multiple time periods
2. Use "Blend Modes" for comparison
3. Create before/after visualizations

## Step 5: Interactive Features

### 5.1 Time Slider
1. Add temporal slider widget
2. Connect to temporal controller
3. Allow user interaction

### 5.2 Popup Information
1. Configure popups to show:
   - NDVI value
   - Date
   - Site information
   - Cloud cover
   - Number of observations

### 5.3 Filtering
1. Add attribute table filters
2. Filter by date range
3. Filter by NDVI value range
4. Filter by site

## Step 6: Export and Sharing

### 6.1 Static Maps
1. Create print layouts for key time periods
2. Export as PDF or high-resolution PNG
3. Include legends and temporal information

### 6.2 Web Maps
1. Use QGIS2Web plugin
2. Export as interactive web map
3. Include temporal controls

### 6.3 Animations
1. Export as GIF or MP4
2. Include temporal scale
3. Add explanatory text

## Tips and Best Practices

### Color Schemes
- Use intuitive color schemes (green = high vegetation)
- Consider colorblind-friendly palettes
- Maintain consistency across time periods

### Temporal Resolution
- Choose appropriate temporal resolution for your analysis
- Monthly data good for seasonal patterns
- Annual data good for long-term trends

### Performance
- Use spatial indexing for large datasets
- Consider data aggregation for better performance
- Use appropriate symbology for web export

### Quality Control
- Filter out high cloud cover observations
- Check for data gaps in time series
- Validate NDVI values (should be -1 to 1)

## Example Workflows

### Workflow 1: Seasonal Vegetation Patterns
1. Load seasonal NDVI data
2. Create seasonal difference maps
3. Identify areas of seasonal change
4. Export seasonal comparison

### Workflow 2: Long-term Vegetation Trends
1. Load annual NDVI data
2. Calculate trend statistics
3. Create trend maps
4. Identify areas of vegetation change

### Workflow 3: Site-specific Time Series
1. Focus on specific sites
2. Create detailed time series plots
3. Analyze site-specific patterns
4. Compare with environmental factors

## Troubleshooting

### Common Issues
- Temporal controller not working: Check temporal field format
- Slow performance: Use data aggregation
- Missing data: Check for gaps in time series
- Color issues: Verify NDVI value ranges

### Performance Optimization
- Use spatial indexes
- Aggregate data appropriately
- Simplify symbology for large datasets
- Use appropriate projection

## Resources
- QGIS Temporal Controller documentation
- NDVI interpretation guides
- Temporal analysis best practices
- QGIS animation tutorials
