#!/usr/bin/env python3
"""
QGIS Temporal NDVI Visualization Workflow
=========================================

This script creates comprehensive temporal visualizations of NDVI data using QGIS.
Features include:
- Time-series maps with temporal slider
- Seasonal NDVI patterns
- Interactive temporal analysis
- Export-ready animations
- Multi-temporal comparison

Author: AI Assistant
Date: 2025-01-17
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class QGISNDVITemporalWorkflow:
    """QGIS workflow for temporal NDVI visualization."""
    
    def __init__(self):
        self.ndvi_data = None
        self.sites_gdf = None
        self.output_dir = Path("qgis_ndvi_temporal")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load NDVI data and prepare for QGIS visualization."""
        print("Loading NDVI data...")
        
        # Load NDVI data
        ndvi_df = pd.read_csv('SAPFLUXNET_Landsat_NDVI_AllSites.csv')
        ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
        
        # Load sites shapefile
        sites_gdf = gpd.read_file('shapefiles/sapfluxnet_sites.shp')
        
        # Merge NDVI data with spatial data
        merged_data = []
        
        for site in sites_gdf['site'].unique():
            site_ndvi = ndvi_df[ndvi_df['site'] == site]
            site_geom = sites_gdf[sites_gdf['site'] == sites_gdf['site']].iloc[0]
            
            for _, row in site_ndvi.iterrows():
                merged_data.append({
                    'site': site,
                    'date': row['date'],
                    'ndvi': row['ndvi'],
                    'cloud_cover': row['cloud_cover'],
                    'collection': row['collection'],
                    'geometry': site_geom.geometry,
                    'latitude': site_geom.latitude,
                    'longitude': site_geom.longitude
                })
        
        # Create GeoDataFrame
        self.ndvi_gdf = gpd.GeoDataFrame(merged_data, crs=sites_gdf.crs)
        
        print(f"Prepared {len(self.ndvi_gdf)} NDVI observations for {self.ndvi_gdf['site'].nunique()} sites")
        
    def create_temporal_datasets(self):
        """Create different temporal datasets for QGIS visualization."""
        print("Creating temporal datasets...")
        
        # 1. Monthly aggregated data
        monthly_data = self.ndvi_gdf.copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        
        monthly_agg = monthly_data.groupby(['site', 'year_month', 'geometry', 'latitude', 'longitude']).agg({
            'ndvi': ['mean', 'std', 'count'],
            'cloud_cover': 'mean'
        }).reset_index()
        
        monthly_agg.columns = ['site', 'year_month', 'geometry', 'latitude', 'longitude', 
                              'ndvi_mean', 'ndvi_std', 'ndvi_count', 'cloud_cover_mean']
        
        # Convert to GeoDataFrame
        monthly_gdf = gpd.GeoDataFrame(monthly_agg, crs=self.ndvi_gdf.crs)
        monthly_gdf['date_str'] = monthly_gdf['year_month'].astype(str)
        
        # 2. Seasonal data
        seasonal_data = self.ndvi_gdf.copy()
        seasonal_data['season'] = seasonal_data['date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        seasonal_data['year'] = seasonal_data['date'].dt.year
        
        seasonal_agg = seasonal_data.groupby(['site', 'year', 'season', 'geometry', 'latitude', 'longitude']).agg({
            'ndvi': ['mean', 'std', 'count'],
            'cloud_cover': 'mean'
        }).reset_index()
        
        seasonal_agg.columns = ['site', 'year', 'season', 'geometry', 'latitude', 'longitude',
                               'ndvi_mean', 'ndvi_std', 'ndvi_count', 'cloud_cover_mean']
        
        seasonal_gdf = gpd.GeoDataFrame(seasonal_agg, crs=self.ndvi_gdf.crs)
        
        # 3. Annual data
        annual_data = self.ndvi_gdf.copy()
        annual_data['year'] = annual_data['date'].dt.year
        
        annual_agg = annual_data.groupby(['site', 'year', 'geometry', 'latitude', 'longitude']).agg({
            'ndvi': ['mean', 'std', 'count', 'min', 'max'],
            'cloud_cover': 'mean'
        }).reset_index()
        
        annual_agg.columns = ['site', 'year', 'geometry', 'latitude', 'longitude',
                             'ndvi_mean', 'ndvi_std', 'ndvi_count', 'ndvi_min', 'ndvi_max', 'cloud_cover_mean']
        
        annual_gdf = gpd.GeoDataFrame(annual_agg, crs=self.ndvi_gdf.crs)
        
        # Save datasets
        monthly_gdf.to_file(self.output_dir / "ndvi_monthly.geojson", driver='GeoJSON')
        seasonal_gdf.to_file(self.output_dir / "ndvi_seasonal.geojson", driver='GeoJSON')
        annual_gdf.to_file(self.output_dir / "ndvi_annual.geojson", driver='GeoJSON')
        
        # Save individual time periods for temporal slider
        self.create_temporal_layers(monthly_gdf)
        
        print(f"Created temporal datasets:")
        print(f"  - Monthly: {len(monthly_gdf)} records")
        print(f"  - Seasonal: {len(seasonal_gdf)} records")
        print(f"  - Annual: {len(annual_gdf)} records")
        
    def create_temporal_layers(self, monthly_gdf):
        """Create individual layers for each time period."""
        print("Creating temporal layers...")
        
        temporal_dir = self.output_dir / "temporal_layers"
        temporal_dir.mkdir(exist_ok=True)
        
        # Group by date and create individual files
        for date_str in monthly_gdf['date_str'].unique():
            period_data = monthly_gdf[monthly_gdf['date_str'] == date_str]
            
            # Clean filename
            clean_date = date_str.replace('-', '_')
            filename = f"ndvi_{clean_date}.geojson"
            
            period_data.to_file(temporal_dir / filename, driver='GeoJSON')
        
        print(f"Created {len(monthly_gdf['date_str'].unique())} temporal layers")
        
    def create_qgis_project_file(self):
        """Create a QGIS project file with temporal visualization setup."""
        print("Creating QGIS project file...")
        
        project_content = f"""<!DOCTYPE qgis>
<qgis version="3.28.0-Firenze" simplifyAlgorithm="0" simplifyMaxScale="1" simplifyDrawingHints="1" simplifyLocal="1" readOnly="0" styleCategories="AllStyleCategories" labelsEnabled="0" symbologyReferenceScale="-1" mapUnitsMMPerMapUnit="1000" minScale="100000000" maxScale="0" simplifyDrawingTol="1">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal>
    <fixedRange>
      <start>{self.ndvi_gdf['date'].min().strftime('%Y-%m-%d')}</start>
      <end>{self.ndvi_gdf['date'].max().strftime('%Y-%m-%d')}</end>
    </fixedRange>
  </temporal>
  <renderer-v2 type="categorizedSymbol" forceraster="0" enableorderby="0" attr="ndvi_mean" symbollevels="0">
    <categories>
      <category symbol="0" value="-0.5" label="-0.5 to -0.1" render="true"/>
      <category symbol="1" value="0.0" label="0.0 to 0.1" render="true"/>
      <category symbol="2" value="0.1" label="0.1 to 0.2" render="true"/>
      <category symbol="3" value="0.2" label="0.2 to 0.3" render="true"/>
      <category symbol="4" value="0.3" label="0.3 to 0.4" render="true"/>
      <category symbol="5" value="0.4" label="0.4 to 0.5" render="true"/>
      <category symbol="6" value="0.5" label="0.5 to 1.0" render="true"/>
    </categories>
    <symbols>
      <symbol type="marker" name="0" alpha="1" force_rhr="0" clip_to_extent="1">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" name="name" value=""/>
            <Option name="properties"/>
            <Option type="QString" name="type" value="collection"/>
          </Option>
        </data_defined_properties>
        <layer class="SimpleMarker" locked="0" enabled="1" pass="0">
          <Option type="Map">
            <Option type="QString" name="angle" value="0"/>
            <Option type="QString" name="cap_style" value="square"/>
            <Option type="QString" name="color" value="0,0,255,255"/>
            <Option type="QString" name="horizontal_anchor_point" value="1"/>
            <Option type="QString" name="joinstyle" value="bevel"/>
            <Option type="QString" name="name" value="circle"/>
            <Option type="QString" name="offset" value="0,0"/>
            <Option type="QString" name="offset_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="offset_unit" value="MM"/>
            <Option type="QString" name="outline_color" value="35,35,35,255"/>
            <Option type="QString" name="outline_style" value="solid"/>
            <Option type="QString" name="outline_width" value="0"/>
            <Option type="QString" name="outline_width_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
            <Option type="QString" name="scale_method" value="diameter"/>
            <Option type="QString" name="size" value="2"/>
            <Option type="QString" name="size_map_unit_scale" value="3x:0,0,0,0,0,0"/>
            <Option type="QString" name="size_unit" value="MM"/>
            <Option type="QString" name="vertical_anchor_point" value="1"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>0</layerGeometryType>
</qgis>"""
        
        with open(self.output_dir / "ndvi_temporal.qgz", 'w', encoding='utf-8') as f:
            f.write(project_content)
            
    def create_visualization_guide(self):
        """Create a comprehensive guide for QGIS temporal visualization."""
        guide_content = """# QGIS Temporal NDVI Visualization Guide

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
"""
        
        with open(self.output_dir / "qgis_temporal_guide.md", 'w', encoding='utf-8') as f:
            f.write(guide_content)
            
    def create_summary_statistics(self):
        """Create summary statistics for the temporal data."""
        print("Creating summary statistics...")
        
        stats = {
            'total_observations': len(self.ndvi_gdf),
            'unique_sites': self.ndvi_gdf['site'].nunique(),
            'date_range': {
                'start': self.ndvi_gdf['date'].min().strftime('%Y-%m-%d'),
                'end': self.ndvi_gdf['date'].max().strftime('%Y-%m-%d')
            },
            'ndvi_statistics': {
                'mean': float(self.ndvi_gdf['ndvi'].mean()),
                'median': float(self.ndvi_gdf['ndvi'].median()),
                'std': float(self.ndvi_gdf['ndvi'].std()),
                'min': float(self.ndvi_gdf['ndvi'].min()),
                'max': float(self.ndvi_gdf['ndvi'].max())
            },
            'temporal_coverage': {
                'years': self.ndvi_gdf['date'].dt.year.nunique(),
                'months': self.ndvi_gdf['date'].dt.to_period('M').nunique(),
                'seasons': self.ndvi_gdf['date'].dt.month.map({
                    12: 'Winter', 1: 'Winter', 2: 'Winter',
                    3: 'Spring', 4: 'Spring', 5: 'Spring',
                    6: 'Summer', 7: 'Summer', 8: 'Summer',
                    9: 'Fall', 10: 'Fall', 11: 'Fall'
                }).nunique()
            },
            'quality_metrics': {
                'mean_cloud_cover': float(self.ndvi_gdf['cloud_cover'].mean()),
                'negative_ndvi_count': int((self.ndvi_gdf['ndvi'] < 0).sum()),
                'high_cloud_cover_count': int((self.ndvi_gdf['cloud_cover'] > 50).sum())
            }
        }
        
        with open(self.output_dir / "temporal_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            
        # Create readable summary
        summary_content = f"""# NDVI Temporal Data Summary

## Overview
- **Total Observations**: {stats['total_observations']:,}
- **Unique Sites**: {stats['unique_sites']}
- **Date Range**: {stats['date_range']['start']} to {stats['date_range']['end']}
- **Temporal Coverage**: {stats['temporal_coverage']['years']} years, {stats['temporal_coverage']['months']} months

## NDVI Statistics
- **Mean**: {stats['ndvi_statistics']['mean']:.3f}
- **Median**: {stats['ndvi_statistics']['median']:.3f}
- **Standard Deviation**: {stats['ndvi_statistics']['std']:.3f}
- **Range**: {stats['ndvi_statistics']['min']:.3f} to {stats['ndvi_statistics']['max']:.3f}

## Data Quality
- **Mean Cloud Cover**: {stats['quality_metrics']['mean_cloud_cover']:.1f}%
- **Negative NDVI Values**: {stats['quality_metrics']['negative_ndvi_count']} ({stats['quality_metrics']['negative_ndvi_count']/stats['total_observations']*100:.1f}%)
- **High Cloud Cover (>50%)**: {stats['quality_metrics']['high_cloud_cover_count']} ({stats['quality_metrics']['high_cloud_cover_count']/stats['total_observations']*100:.1f}%)

## Temporal Distribution
- **Years with Data**: {stats['temporal_coverage']['years']}
- **Months with Data**: {stats['temporal_coverage']['months']}
- **Seasons Covered**: {stats['temporal_coverage']['seasons']}

## Files Created
- `ndvi_monthly.geojson`: Monthly aggregated NDVI data
- `ndvi_seasonal.geojson`: Seasonal aggregated NDVI data  
- `ndvi_annual.geojson`: Annual aggregated NDVI data
- `temporal_layers/`: Individual time period layers
- `qgis_temporal_guide.md`: Complete QGIS workflow guide
- `temporal_statistics.json`: Detailed statistics

## Next Steps
1. Open QGIS and load the monthly data
2. Follow the temporal visualization guide
3. Create animations and time-series maps
4. Analyze seasonal and long-term patterns
"""
        
        with open(self.output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(summary_content)
            
    def run_complete_workflow(self):
        """Run the complete temporal NDVI workflow."""
        print("🌍 Starting QGIS Temporal NDVI Workflow")
        print("=" * 50)
        
        self.load_and_prepare_data()
        self.create_temporal_datasets()
        self.create_qgis_project_file()
        self.create_visualization_guide()
        self.create_summary_statistics()
        
        print("\n✅ QGIS Temporal NDVI Workflow Complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print("\n🎯 Next Steps:")
        print("1. Open QGIS")
        print("2. Load: qgis_ndvi_temporal/ndvi_monthly.geojson")
        print("3. Follow: qgis_ndvi_temporal/qgis_temporal_guide.md")
        print("4. Create temporal animations and time-series maps")
        print("5. Analyze seasonal and long-term NDVI patterns")

if __name__ == "__main__":
    workflow = QGISNDVITemporalWorkflow()
    workflow.run_complete_workflow() 