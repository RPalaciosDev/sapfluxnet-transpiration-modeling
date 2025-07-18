#!/usr/bin/env python3
"""
QGIS Spatial Visualization Script for SAPFLUXNET Model Performance
==================================================================

This script creates comprehensive spatial visualizations of XGBoost model
performance across different sites using QGIS.

Features:
- Site performance maps (R² and RMSE)
- Performance clustering analysis
- Interactive legends and styling
- Export-ready maps
- Statistical summaries by region

Author: AI Assistant
Date: 2025-01-17
"""

import os
import sys
import pandas as pd
import numpy as np
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsFields, QgsVectorFileWriter, QgsSymbol, QgsRendererCategory,
    QgsCategorizedSymbolRenderer, QgsGraduatedSymbolRenderer,
    QgsColorRamp, QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer,
    QgsPalLayerSettings, QgsTextFormat, QgsTextBufferSettings,
    QgsLayoutItemMap, QgsLayoutItemLegend, QgsLayoutItemLabel,
    QgsPrintLayout, QgsLayoutExporter, QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor, QFont

class SAPFLUXNETSpatialVisualizer:
    """Main class for creating spatial visualizations of SAPFLUXNET model performance."""
    
    def __init__(self, project_path):
        """Initialize the visualizer with project path."""
        self.project_path = project_path
        self.project = QgsProject.instance()
        self.performance_data = None
        self.sites_layer = None
        self.study_area_layer = None
        
    def load_performance_data(self, csv_path):
        """Load the spatial validation performance data."""
        print("Loading performance data...")
        self.performance_data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.performance_data)} site performance records")
        
        # Basic statistics
        print("\nPerformance Summary:")
        print(f"Mean R²: {self.performance_data['test_r2'].mean():.3f}")
        print(f"Median R²: {self.performance_data['test_r2'].median():.3f}")
        print(f"Mean RMSE: {self.performance_data['test_rmse'].mean():.3f}")
        print(f"Median RMSE: {self.performance_data['test_rmse'].median():.3f}")
        
        # Identify problematic sites
        poor_performance = self.performance_data[self.performance_data['test_r2'] < 0]
        print(f"\nSites with negative R²: {len(poor_performance)}")
        if len(poor_performance) > 0:
            print("Problematic sites:")
            for _, row in poor_performance.head(10).iterrows():
                print(f"  {row['site']}: R²={row['test_r2']:.3f}, RMSE={row['test_rmse']:.3f}")
    
    def load_spatial_data(self):
        """Load the spatial data layers."""
        print("\nLoading spatial data...")
        
        # Load sites shapefile
        sites_path = os.path.join(self.project_path, "shapefiles", "sapfluxnet_sites.shp")
        if os.path.exists(sites_path):
            self.sites_layer = QgsVectorLayer(sites_path, "SAPFLUXNET Sites", "ogr")
            if self.sites_layer.isValid():
                self.project.addMapLayer(self.sites_layer)
                print(f"Loaded sites layer: {self.sites_layer.featureCount()} features")
            else:
                print("Error: Could not load sites layer")
                return False
        else:
            print(f"Error: Sites shapefile not found at {sites_path}")
            return False
        
        # Load study area
        study_area_path = os.path.join(self.project_path, "shapefiles", "sapfluxnet_study_area.shp")
        if os.path.exists(study_area_path):
            self.study_area_layer = QgsVectorLayer(study_area_path, "Study Area", "ogr")
            if self.study_area_layer.isValid():
                self.project.addMapLayer(self.study_area_layer)
                print("Loaded study area layer")
            else:
                print("Warning: Could not load study area layer")
        
        return True
    
    def create_performance_layer(self):
        """Create a new layer with performance data joined to spatial data."""
        print("\nCreating performance layer...")
        
        # Create a new memory layer
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "Model Performance", "memory")
        provider = layer.dataProvider()
        
        # Add fields
        fields = QgsFields()
        fields.append(QgsField("site", QVariant.String))
        fields.append(QgsField("test_r2", QVariant.Double))
        fields.append(QgsField("test_rmse", QVariant.Double))
        fields.append(QgsField("test_samples", QVariant.Int))
        fields.append(QgsField("performance_category", QVariant.String))
        provider.addAttributes(fields)
        layer.updateFields()
        
        # Get site coordinates from the sites layer
        site_coords = {}
        for feature in self.sites_layer.getFeatures():
            geom = feature.geometry()
            if geom:
                point = geom.asPoint()
                site_name = feature.attribute("SITE_ID") if "SITE_ID" in [f.name() for f in feature.fields()] else feature.attribute(0)
                site_coords[site_name] = (point.x(), point.y())
        
        # Create features with performance data
        features = []
        for _, row in self.performance_data.iterrows():
            site = row['site']
            if site in site_coords:
                x, y = site_coords[site]
                
                # Create feature
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                
                # Set attributes
                feat.setAttributes([
                    site,
                    row['test_r2'],
                    row['test_rmse'],
                    row['test_samples'],
                    self._categorize_performance(row['test_r2'])
                ])
                
                features.append(feat)
        
        # Add features to layer
        provider.addFeatures(features)
        layer.updateExtents()
        
        # Add layer to project
        self.project.addMapLayer(layer)
        self.performance_layer = layer
        
        print(f"Created performance layer with {len(features)} features")
        return layer
    
    def _categorize_performance(self, r2):
        """Categorize performance based on R² value."""
        if r2 >= 0.7:
            return "Excellent"
        elif r2 >= 0.5:
            return "Good"
        elif r2 >= 0.3:
            return "Fair"
        elif r2 >= 0:
            return "Poor"
        else:
            return "Very Poor"
    
    def style_performance_layer(self, metric='test_r2'):
        """Apply graduated styling to the performance layer."""
        print(f"\nStyling performance layer with {metric}...")
        
        # Create graduated renderer
        renderer = QgsGraduatedSymbolRenderer()
        renderer.setClassAttribute(metric)
        
        # Define color ramp
        if metric == 'test_r2':
            # Green to red for R² (good to bad)
            colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
            ranges = [
                (-200, -10, "Very Poor (< -10)"),
                (-10, 0, "Poor (-10 to 0)"),
                (0, 0.2, "Fair (0-0.2)"),
                (0.2, 0.4, "Moderate (0.2-0.4)"),
                (0.4, 0.6, "Good (0.4-0.6)"),
                (0.6, 0.8, "Very Good (0.6-0.8)"),
                (0.8, 1.0, "Excellent (0.8-1.0)")
            ]
        else:
            # Red to green for RMSE (bad to good)
            colors = ['#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', '#fdae61', '#f46d43', '#d73027']
            ranges = [
                (0, 2, "Excellent (0-2)"),
                (2, 4, "Very Good (2-4)"),
                (4, 6, "Good (4-6)"),
                (6, 8, "Moderate (6-8)"),
                (8, 10, "Fair (8-10)"),
                (10, 15, "Poor (10-15)"),
                (15, 50, "Very Poor (>15)")
            ]
        
        # Create symbol categories
        categories = []
        for i, (min_val, max_val, label) in enumerate(ranges):
            symbol = QgsSymbol.defaultSymbol(self.performance_layer.geometryType())
            symbol.setColor(QColor(colors[i % len(colors)]))
            symbol.setSize(3)
            
            category = QgsRendererCategory(min_val, symbol, label)
            category.setUpperValue(max_val)
            categories.append(category)
        
        renderer.addCategories(categories)
        self.performance_layer.setRenderer(renderer)
        self.performance_layer.triggerRepaint()
        
        print("Applied graduated styling")
    
    def create_performance_summary(self):
        """Create a summary of performance by geographic regions."""
        print("\nCreating performance summary...")
        
        # Add region information if available
        if hasattr(self, 'performance_layer'):
            # Group by performance category
            categories = {}
            for feature in self.performance_layer.getFeatures():
                category = feature.attribute("performance_category")
                r2 = feature.attribute("test_r2")
                rmse = feature.attribute("test_rmse")
                
                if category not in categories:
                    categories[category] = {'count': 0, 'r2_values': [], 'rmse_values': []}
                
                categories[category]['count'] += 1
                categories[category]['r2_values'].append(r2)
                categories[category]['rmse_values'].append(rmse)
            
            print("\nPerformance Summary by Category:")
            print("=" * 50)
            for category, data in sorted(categories.items()):
                avg_r2 = np.mean(data['r2_values'])
                avg_rmse = np.mean(data['rmse_values'])
                print(f"{category}: {data['count']} sites, Avg R²: {avg_r2:.3f}, Avg RMSE: {avg_rmse:.3f}")
    
    def export_map(self, output_path, title="SAPFLUXNET Model Performance"):
        """Export the map to an image file."""
        print(f"\nExporting map to {output_path}...")
        
        # Create layout
        layout = QgsPrintLayout(self.project)
        layout.initializeDefaults()
        layout.setName("SAPFLUXNET Performance Map")
        
        # Add map item
        map_item = QgsLayoutItemMap(layout)
        map_item.setRect(20, 20, 200, 200)
        layout.addLayoutItem(map_item)
        
        # Set map extent to show all sites
        if hasattr(self, 'performance_layer'):
            map_item.setExtent(self.performance_layer.extent())
        
        # Add title
        title_item = QgsLayoutItemLabel(layout)
        title_item.setText(title)
        title_item.setFont(QFont("Arial", 16, QFont.Bold))
        title_item.setRect(20, 5, 200, 15)
        layout.addLayoutItem(title_item)
        
        # Add legend
        legend_item = QgsLayoutItemLegend(layout)
        legend_item.setTitle("Model Performance (R²)")
        legend_item.setRect(230, 20, 80, 200)
        layout.addLayoutItem(legend_item)
        
        # Export
        exporter = QgsLayoutExporter(layout)
        result = exporter.exportToImage(output_path, QgsLayoutExporter.ImageExportSettings())
        
        if result[0] == QgsLayoutExporter.Success:
            print(f"Map exported successfully to {output_path}")
        else:
            print(f"Error exporting map: {result[1]}")
    
    def create_interactive_visualization(self):
        """Create an interactive visualization with popups and tooltips."""
        print("\nCreating interactive visualization...")
        
        if hasattr(self, 'performance_layer'):
            # Enable layer editing for popups
            self.performance_layer.setDisplayExpression("'Site: ' || \"site\" || '\nR²: ' || round(\"test_r2\", 3) || '\nRMSE: ' || round(\"test_rmse\", 3)")
            
            # Set up popup configuration
            popup_config = self.performance_layer.attributeTableConfig()
            popup_config.setDisplayExpression("\"site\"")
            self.performance_layer.setAttributeTableConfig(popup_config)
            
            print("Interactive features enabled - click on sites to see details")

def main():
    """Main function to run the spatial visualization."""
    # Set up paths
    project_path = os.getcwd()
    performance_csv = os.path.join(project_path, "xgboost_scripts", "external_memory_models", 
                                  "spatial_validation", "sapfluxnet_spatial_external_sites_20250717_203222.csv")
    
    # Initialize visualizer
    visualizer = SAPFLUXNETSpatialVisualizer(project_path)
    
    # Load data
    if not os.path.exists(performance_csv):
        print(f"Error: Performance CSV not found at {performance_csv}")
        return
    
    visualizer.load_performance_data(performance_csv)
    
    if not visualizer.load_spatial_data():
        print("Error: Could not load spatial data")
        return
    
    # Create visualizations
    visualizer.create_performance_layer()
    visualizer.style_performance_layer('test_r2')
    visualizer.create_performance_summary()
    visualizer.create_interactive_visualization()
    
    # Export map
    output_path = os.path.join(project_path, "sapfluxnet_performance_map.png")
    visualizer.export_map(output_path)
    
    print("\n" + "="*60)
    print("QGIS SPATIAL VISUALIZATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Open QGIS and load the project")
    print("2. The performance layer will be styled with R² values")
    print("3. Use the layer panel to switch between R² and RMSE visualization")
    print("4. Export high-quality maps using QGIS print composer")
    print("5. Consider creating additional visualizations:")
    print("   - Performance by elevation")
    print("   - Performance by climate zone")
    print("   - Temporal performance patterns")

if __name__ == "__main__":
    main() 