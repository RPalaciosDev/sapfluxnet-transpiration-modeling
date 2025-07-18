#!/usr/bin/env python3
"""
Data Preparation Script for QGIS Spatial Visualization
======================================================

This script prepares the SAPFLUXNET performance data for QGIS visualization
by merging spatial data with model performance metrics.

Author: AI Assistant
Date: 2025-01-17
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path

def load_and_analyze_performance_data(csv_path):
    """Load and analyze the performance data."""
    print("Loading performance data...")
    performance_df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(performance_df)} site performance records")
    
    # Basic statistics
    print("\nPerformance Summary:")
    print(f"Mean R²: {performance_df['test_r2'].mean():.3f}")
    print(f"Median R²: {performance_df['test_r2'].median():.3f}")
    print(f"Mean RMSE: {performance_df['test_rmse'].mean():.3f}")
    print(f"Median RMSE: {performance_df['test_rmse'].median():.3f}")
    
    # Performance categories
    def categorize_performance(r2):
        if r2 >= 0.7: return "Excellent"
        elif r2 >= 0.5: return "Good"
        elif r2 >= 0.3: return "Fair"
        elif r2 >= 0: return "Poor"
        else: return "Very Poor"
    
    performance_df['category'] = performance_df['test_r2'].apply(categorize_performance)
    
    print("\nPerformance by Category:")
    category_counts = performance_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} sites")
    
    # Identify problematic sites
    poor_performance = performance_df[performance_df['test_r2'] < 0]
    print(f"\nSites with negative R²: {len(poor_performance)}")
    if len(poor_performance) > 0:
        print("Top problematic sites:")
        for _, row in poor_performance.head(5).iterrows():
            print(f"  {row['site']}: R²={row['test_r2']:.3f}, RMSE={row['test_rmse']:.3f}")
    
    return performance_df

def load_spatial_data(shapefile_path):
    """Load the spatial data."""
    print(f"\nLoading spatial data from {shapefile_path}...")
    
    if not os.path.exists(shapefile_path):
        print(f"Error: Shapefile not found at {shapefile_path}")
        return None
    
    try:
        sites_gdf = gpd.read_file(shapefile_path)
        print(f"Loaded {len(sites_gdf)} spatial features")
        print(f"CRS: {sites_gdf.crs}")
        print(f"Columns: {list(sites_gdf.columns)}")
        
        # Show first few records
        print("\nFirst few spatial records:")
        print(sites_gdf.head())
        
        return sites_gdf
    except Exception as e:
        print(f"Error loading spatial data: {e}")
        return None

def merge_data(performance_df, sites_gdf):
    """Merge performance data with spatial data."""
    print("\nMerging performance data with spatial data...")
    
    # Try to identify the site ID column in the spatial data
    possible_site_columns = ['SITE_ID', 'site', 'Site', 'SITE', 'site_id', 'Site_ID']
    site_column = None
    
    for col in possible_site_columns:
        if col in sites_gdf.columns:
            site_column = col
            break
    
    if site_column is None:
        print("Warning: Could not find site ID column. Available columns:")
        print(list(sites_gdf.columns))
        print("\nTrying to use first column as site ID...")
        site_column = sites_gdf.columns[0]
    
    print(f"Using '{site_column}' as site identifier")
    
    # Clean site names for better matching
    performance_df['site_clean'] = performance_df['site'].str.strip()
    sites_gdf['site_clean'] = sites_gdf[site_column].astype(str).str.strip()
    
    # Merge data
    merged_data = sites_gdf.merge(
        performance_df, 
        left_on='site_clean', 
        right_on='site_clean', 
        how='inner'
    )
    
    print(f"Successfully merged {len(merged_data)} records")
    
    # Check for unmatched sites
    unmatched_performance = performance_df[~performance_df['site_clean'].isin(sites_gdf['site_clean'])]
    if len(unmatched_performance) > 0:
        print(f"\nWarning: {len(unmatched_performance)} performance records could not be matched:")
        for site in unmatched_performance['site'].head(10):
            print(f"  {site}")
    
    return merged_data

def create_enhanced_dataset(merged_data):
    """Create an enhanced dataset with additional derived fields."""
    print("\nCreating enhanced dataset...")
    
    # Add derived fields
    merged_data['r2_normalized'] = (merged_data['test_r2'] - merged_data['test_r2'].min()) / \
                                  (merged_data['test_r2'].max() - merged_data['test_r2'].min())
    
    merged_data['rmse_normalized'] = (merged_data['test_rmse'] - merged_data['test_rmse'].min()) / \
                                    (merged_data['test_rmse'].max() - merged_data['test_rmse'].min())
    
    # Add performance score (combined metric)
    merged_data['performance_score'] = merged_data['r2_normalized'] * (1 - merged_data['rmse_normalized'])
    
    # Add size category based on sample count
    merged_data['size_category'] = pd.cut(
        merged_data['test_samples'], 
        bins=[0, 1000, 2000, 3000, float('inf')], 
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    
    print("Enhanced dataset created with additional fields:")
    print("  - r2_normalized: Normalized R² values (0-1)")
    print("  - rmse_normalized: Normalized RMSE values (0-1)")
    print("  - performance_score: Combined performance metric")
    print("  - size_category: Site size based on sample count")
    
    return merged_data

def export_formats(merged_data, output_dir):
    """Export data in multiple formats for QGIS."""
    print(f"\nExporting data to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export as GeoJSON (recommended for QGIS)
    geojson_path = os.path.join(output_dir, "sapfluxnet_performance_sites.geojson")
    merged_data.to_file(geojson_path, driver='GeoJSON')
    print(f"✓ GeoJSON exported: {geojson_path}")
    
    # Export as Shapefile
    shapefile_path = os.path.join(output_dir, "sapfluxnet_performance_sites.shp")
    merged_data.to_file(shapefile_path, driver='ESRI Shapefile')
    print(f"✓ Shapefile exported: {shapefile_path}")
    
    # Export as CSV with coordinates
    csv_path = os.path.join(output_dir, "sapfluxnet_performance_sites.csv")
    merged_data.to_csv(csv_path, index=False)
    print(f"✓ CSV exported: {csv_path}")
    
    # Create summary statistics
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAPFLUXNET Model Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Sites: {len(merged_data)}\n")
        f.write(f"Mean R²: {merged_data['test_r2'].mean():.3f}\n")
        f.write(f"Median R²: {merged_data['test_r2'].median():.3f}\n")
        f.write(f"Mean RMSE: {merged_data['test_rmse'].mean():.3f}\n")
        f.write(f"Median RMSE: {merged_data['test_rmse'].median():.3f}\n\n")
        
        f.write("Performance by Category:\n")
        for category, count in merged_data['category'].value_counts().items():
            f.write(f"  {category}: {count} sites\n")
        
        f.write("\nGeographic Coverage:\n")
        f.write(f"  Latitude: {merged_data.geometry.y.min():.2f}° to {merged_data.geometry.y.max():.2f}°\n")
        f.write(f"  Longitude: {merged_data.geometry.x.min():.2f}° to {merged_data.geometry.x.max():.2f}°\n")
    
    print(f"✓ Summary exported: {summary_path}")
    
    return {
        'geojson': geojson_path,
        'shapefile': shapefile_path,
        'csv': csv_path,
        'summary': summary_path
    }

def main():
    """Main function to prepare data for QGIS visualization."""
    print("SAPFLUXNET QGIS Data Preparation")
    print("=" * 40)
    
    # Set up paths
    project_path = Path.cwd()
    performance_csv = project_path / "xgboost_scripts" / "external_memory_models" / "spatial_validation" / "sapfluxnet_spatial_external_sites_20250717_203222.csv"
    shapefile_path = project_path / "shapefiles" / "sapfluxnet_sites.shp"
    output_dir = project_path / "qgis_data"
    
    # Check if performance data exists
    if not performance_csv.exists():
        print(f"Error: Performance CSV not found at {performance_csv}")
        return
    
    # Load and analyze performance data
    performance_df = load_and_analyze_performance_data(performance_csv)
    
    # Load spatial data
    sites_gdf = load_spatial_data(shapefile_path)
    if sites_gdf is None:
        print("Error: Could not load spatial data")
        return
    
    # Merge data
    merged_data = merge_data(performance_df, sites_gdf)
    if merged_data is None or len(merged_data) == 0:
        print("Error: No data could be merged")
        return
    
    # Create enhanced dataset
    enhanced_data = create_enhanced_dataset(merged_data)
    
    # Export data
    export_paths = export_formats(enhanced_data, output_dir)
    
    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETE")
    print("=" * 50)
    print("\nNext steps for QGIS:")
    print("1. Open QGIS")
    print("2. Add the GeoJSON file: qgis_data/sapfluxnet_performance_sites.geojson")
    print("3. Style the layer using the 'test_r2' or 'test_rmse' fields")
    print("4. Use the 'category' field for categorical styling")
    print("5. Follow the manual workflow guide: qgis_manual_workflow.md")
    
    print(f"\nExported files:")
    for format_name, path in export_paths.items():
        print(f"  {format_name.upper()}: {path}")

if __name__ == "__main__":
    main() 