"""
Advanced 3D Visualizations and QGIS Data Export for SAPFLUXNET Results
Creates interactive 3D plots and GIS-compatible datasets from clustering and model results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import os
import glob
import json
from datetime import datetime
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.transform import from_bounds
import pyproj
from geopy.distance import geodesic

warnings.filterwarnings('ignore')

class Advanced3DQGISVisualizer:
    """Creates advanced 3D visualizations and QGIS-compatible datasets"""
    
    def __init__(self, results_base_dir='../'):
        self.results_base_dir = results_base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Output directories
        self.output_3d_dir = os.path.join(results_base_dir, 'visualization', '3d_plots')
        self.output_qgis_dir = os.path.join(results_base_dir, 'qgis_exports')
        
        os.makedirs(self.output_3d_dir, exist_ok=True)
        os.makedirs(self.output_qgis_dir, exist_ok=True)
        
        print(f"🎨 Advanced 3D & QGIS Visualizer initialized")
        print(f"📁 3D outputs: {self.output_3d_dir}")
        print(f"🗺️  QGIS outputs: {self.output_qgis_dir}")
    
    def load_comprehensive_data(self):
        """Load all available results data"""
        print("\n📊 Loading comprehensive results data...")
        
        data = {}
        
        # 1. Load clustering results
        cluster_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'evaluation/clustering_results/advanced_site_clusters_*.csv'
        )))
        if cluster_files:
            data['clusters'] = pd.read_csv(cluster_files[-1])
            print(f"✅ Loaded clustering data: {len(data['clusters'])} sites")
        
        # 2. Load model performance results
        model_metrics_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'models/results/cluster_models/cluster_model_metrics_*.csv'
        )))
        if model_metrics_files:
            data['model_metrics'] = pd.read_csv(model_metrics_files[-1])
            print(f"✅ Loaded model metrics: {len(data['model_metrics'])} clusters")
        
        # 3. Load spatial validation results
        spatial_summaries_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'models/results/cluster_spatial_validation/cluster_spatial_summaries_*.csv'
        )))
        if spatial_summaries_files:
            data['spatial_validation'] = pd.read_csv(spatial_summaries_files[-1])
            print(f"✅ Loaded spatial validation: {len(data['spatial_validation'])} clusters")
        
        # 4. Load site-level validation results
        spatial_folds_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'models/results/cluster_spatial_validation/cluster_spatial_fold_results_*.csv'
        )))
        if spatial_folds_files:
            data['site_performance'] = pd.read_csv(spatial_folds_files[-1])
            print(f"✅ Loaded site performance: {len(data['site_performance'])} site-folds")
        
        # 5. Load sample parquet data for geographic/climate info
        parquet_dir = os.path.join(self.results_base_dir, 'processed_parquet')
        if os.path.exists(parquet_dir):
            parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')][:10]  # Sample first 10
            site_data = []
            
            for pfile in parquet_files:
                try:
                    df_sample = pd.read_parquet(os.path.join(parquet_dir, pfile))
                    if len(df_sample) > 0:
                        # Get site-level info (should be same for all rows)
                        site_info = {
                            'site': df_sample['site'].iloc[0] if 'site' in df_sample.columns else pfile.replace('_comprehensive.parquet', ''),
                            'latitude': df_sample['latitude'].iloc[0] if 'latitude' in df_sample.columns else None,
                            'longitude': df_sample['longitude'].iloc[0] if 'longitude' in df_sample.columns else None,
                            'elevation': df_sample['elevation'].iloc[0] if 'elevation' in df_sample.columns else None,
                            'mean_annual_temp': df_sample['mean_annual_temp'].iloc[0] if 'mean_annual_temp' in df_sample.columns else None,
                            'mean_annual_precip': df_sample['mean_annual_precip'].iloc[0] if 'mean_annual_precip' in df_sample.columns else None,
                            'ecosystem_cluster': df_sample['ecosystem_cluster'].iloc[0] if 'ecosystem_cluster' in df_sample.columns else None,
                            'avg_sap_flow': df_sample['sap_flow'].mean() if 'sap_flow' in df_sample.columns else None
                        }
                        site_data.append(site_info)
                except Exception as e:
                    print(f"    ⚠️  Error loading {pfile}: {e}")
                    continue
            
            if site_data:
                data['site_info'] = pd.DataFrame(site_data)
                print(f"✅ Loaded site geographic data: {len(data['site_info'])} sites")
        
        return data
    
    def create_merged_dataset(self, data):
        """Create a comprehensive merged dataset for visualization"""
        print("\n🔗 Creating merged dataset...")
        
        # Start with clustering data
        if 'clusters' not in data:
            raise ValueError("No clustering data available!")
        
        merged = data['clusters'].copy()
        merged = merged.rename(columns={'cluster': 'ecosystem_cluster'})
        
        # Merge with site geographic info
        if 'site_info' in data:
            merged = merged.merge(
                data['site_info'], 
                on='site', 
                how='left', 
                suffixes=('', '_geo')
            )
            print(f"  ✅ Merged with geographic data")
        
        # Add cluster-level model metrics
        if 'model_metrics' in data:
            cluster_metrics = data['model_metrics'].rename(columns={'cluster': 'ecosystem_cluster'})
            merged = merged.merge(
                cluster_metrics[['ecosystem_cluster', 'test_r2', 'test_rmse', 'feature_count']], 
                on='ecosystem_cluster', 
                how='left'
            )
            print(f"  ✅ Merged with model metrics")
        
        # Add cluster-level spatial validation
        if 'spatial_validation' in data:
            spatial_metrics = data['spatial_validation'].rename(columns={'cluster': 'ecosystem_cluster'})
            merged = merged.merge(
                spatial_metrics[['ecosystem_cluster', 'mean_test_r2', 'mean_test_rmse', 'successful_folds']], 
                on='ecosystem_cluster', 
                how='left',
                suffixes=('', '_spatial')
            )
            print(f"  ✅ Merged with spatial validation")
        
        # Add site-level performance (average across folds per site)
        if 'site_performance' in data:
            site_perf = data['site_performance'].groupby('test_site').agg({
                'test_r2': 'mean',
                'test_rmse': 'mean',
                'test_samples': 'sum'
            }).reset_index()
            site_perf = site_perf.rename(columns={'test_site': 'site'})
            
            merged = merged.merge(
                site_perf, 
                on='site', 
                how='left',
                suffixes=('', '_site_perf')
            )
            print(f"  ✅ Merged with site-level performance")
        
        # Clean and filter
        merged = merged.dropna(subset=['latitude', 'longitude'])
        
        print(f"  📊 Final merged dataset: {len(merged)} sites with {len(merged.columns)} attributes")
        
        return merged
    
    def create_3d_geographic_performance(self, merged_data):
        """Create 3D plot of geographic locations with performance as height"""
        print("\n🌍 Creating 3D Geographic Performance plot...")
        
        # Filter data with valid coordinates and performance
        plot_data = merged_data.dropna(subset=['latitude', 'longitude', 'mean_test_r2']).copy()
        
        if len(plot_data) == 0:
            print("  ⚠️  No valid data for 3D geographic plot")
            return None
        
        # Create interactive Plotly 3D scatter
        fig = go.Figure(data=go.Scatter3d(
            x=plot_data['longitude'],
            y=plot_data['latitude'],
            z=plot_data['mean_test_r2'],
            mode='markers',
            marker=dict(
                size=8,
                color=plot_data['ecosystem_cluster'],
                colorscale='Viridis',
                colorbar=dict(title="Ecosystem Cluster"),
                line=dict(width=1, color='black')
            ),
            text=[f"Site: {row['site']}<br>"
                  f"Cluster: {row['ecosystem_cluster']}<br>"
                  f"R²: {row['mean_test_r2']:.3f}<br>"
                  f"RMSE: {row['mean_test_rmse']:.3f}<br>"
                  f"Lat: {row['latitude']:.2f}<br>"
                  f"Lon: {row['longitude']:.2f}"
                  for _, row in plot_data.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Geographic Distribution of Model Performance',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Spatial Validation R²',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_3d_dir, f'geographic_performance_3d_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig
    
    def create_3d_climate_space(self, merged_data):
        """Create 3D plot in climate space (temp, precip, elevation)"""
        print("\n🌡️  Creating 3D Climate Space plot...")
        
        # Filter data with valid climate data
        climate_cols = ['mean_annual_temp', 'mean_annual_precip', 'elevation']
        plot_data = merged_data.dropna(subset=climate_cols).copy()
        
        if len(plot_data) == 0:
            print("  ⚠️  No valid climate data for 3D plot")
            return None
        
        # Handle marker sizes - use performance if available, otherwise use constant size
        if 'mean_test_r2' in plot_data.columns:
            # Fill NaN values with median and ensure positive values
            r2_values = plot_data['mean_test_r2'].fillna(plot_data['mean_test_r2'].median())
            r2_values = np.maximum(r2_values, 0.1)  # Ensure minimum size
            marker_sizes = r2_values * 15
        else:
            marker_sizes = 8  # Constant size if no performance data
        
        # Create interactive Plotly 3D scatter
        fig = go.Figure(data=go.Scatter3d(
            x=plot_data['mean_annual_temp'],
            y=plot_data['mean_annual_precip'],
            z=plot_data['elevation'],
            mode='markers',
            marker=dict(
                size=marker_sizes,  # Size by performance (cleaned)
                color=plot_data['ecosystem_cluster'],
                colorscale='Viridis',
                colorbar=dict(title="Ecosystem Cluster"),
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[f"Site: {row['site']}<br>"
                  f"Cluster: {row['ecosystem_cluster']}<br>"
                  f"R²: {row.get('mean_test_r2', 'N/A') if pd.notna(row.get('mean_test_r2')) else 'N/A'}<br>"
                  f"Temp: {row['mean_annual_temp']:.1f}°C<br>"
                  f"Precip: {row['mean_annual_precip']:.0f}mm<br>"
                  f"Elevation: {row['elevation']:.0f}m"
                  for _, row in plot_data.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Climate Space Distribution (Size = Model Performance)',
            scene=dict(
                xaxis_title='Mean Annual Temperature (°C)',
                yaxis_title='Mean Annual Precipitation (mm)',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_3d_dir, f'climate_space_3d_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig
    
    def create_3d_feature_space_pca(self, merged_data):
        """Create 3D PCA plot of feature space"""
        print("\n🔬 Creating 3D Feature Space (PCA) plot...")
        
        # Get numerical features for PCA
        feature_cols = [col for col in merged_data.columns 
                       if merged_data[col].dtype in [np.float64, np.int64]
                       and col not in ['site', 'ecosystem_cluster', 'latitude', 'longitude']]
        
        plot_data = merged_data.dropna(subset=feature_cols).copy()
        
        if len(plot_data) < 10 or len(feature_cols) < 3:
            print(f"  ⚠️  Insufficient data for PCA (sites: {len(plot_data)}, features: {len(feature_cols)})")
            return None
        
        # Perform PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(plot_data[feature_cols])
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features_scaled)
        
        explained_var = pca.explained_variance_ratio_
        
        # Create 3D PCA plot
        fig = go.Figure(data=go.Scatter3d(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            z=pca_result[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=plot_data['ecosystem_cluster'],
                colorscale='Viridis',
                colorbar=dict(title="Ecosystem Cluster"),
                line=dict(width=1, color='black')
            ),
            text=[f"Site: {row['site']}<br>"
                  f"Cluster: {row['ecosystem_cluster']}<br>"
                  f"R²: {row.get('mean_test_r2', 'N/A')}"
                  for _, row in plot_data.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D Feature Space (PCA) - Clusters in {len(feature_cols)} Dimensional Space',
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.1%} variance)',
                yaxis_title=f'PC2 ({explained_var[1]:.1%} variance)',
                zaxis_title=f'PC3 ({explained_var[2]:.1%} variance)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8)
                )
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_3d_dir, f'feature_space_pca_3d_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig
    
    def create_performance_surface_3d(self, merged_data):
        """Create 3D surface plot of performance over climate space"""
        print("\n📈 Creating 3D Performance Surface plot...")
        
        # Filter data with valid climate and performance data
        required_cols = ['mean_annual_temp', 'mean_annual_precip', 'mean_test_r2']
        plot_data = merged_data.dropna(subset=required_cols).copy()
        
        if len(plot_data) < 10:
            print("  ⚠️  Insufficient data for surface plot")
            return None
        
        # Create grid for interpolation
        temp_range = np.linspace(
            plot_data['mean_annual_temp'].min(), 
            plot_data['mean_annual_temp'].max(), 
            30
        )
        precip_range = np.linspace(
            plot_data['mean_annual_precip'].min(), 
            plot_data['mean_annual_precip'].max(), 
            30
        )
        
        temp_grid, precip_grid = np.meshgrid(temp_range, precip_range)
        
        # Interpolate performance values
        performance_grid = griddata(
            (plot_data['mean_annual_temp'], plot_data['mean_annual_precip']),
            plot_data['mean_test_r2'],
            (temp_grid, precip_grid),
            method='cubic',
            fill_value=0
        )
        
        # Create surface plot
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            x=temp_grid,
            y=precip_grid,
            z=performance_grid,
            colorscale='Viridis',
            name='Performance Surface',
            opacity=0.8
        ))
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=plot_data['mean_annual_temp'],
            y=plot_data['mean_annual_precip'],
            z=plot_data['mean_test_r2'] + 0.01,  # Slightly above surface
            mode='markers',
            marker=dict(
                size=6,
                color=plot_data['ecosystem_cluster'],
                colorscale='Plotly3',
                line=dict(width=2, color='black')
            ),
            name='Actual Sites',
            text=[f"Site: {row['site']}<br>Cluster: {row['ecosystem_cluster']}" 
                  for _, row in plot_data.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Performance Surface over Climate Space',
            scene=dict(
                xaxis_title='Mean Annual Temperature (°C)',
                yaxis_title='Mean Annual Precipitation (mm)',
                zaxis_title='Model Performance (R²)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_3d_dir, f'performance_surface_3d_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig
    
    def create_qgis_site_points(self, merged_data):
        """Create QGIS-compatible point shapefile with all site attributes"""
        print("\n🗺️  Creating QGIS site points shapefile...")
        
        # Filter data with valid coordinates
        gis_data = merged_data.dropna(subset=['latitude', 'longitude']).copy()
        
        if len(gis_data) == 0:
            print("  ⚠️  No valid geographic data for QGIS export")
            return None
        
        # Create point geometries
        geometry = [Point(lon, lat) for lon, lat in zip(gis_data['longitude'], gis_data['latitude'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(gis_data, geometry=geometry, crs='EPSG:4326')
        
        # Clean column names for shapefile compatibility (10 char limit)
        column_mapping = {
            'ecosystem_cluster': 'eco_clust',
            'mean_annual_temp': 'temp_c',
            'mean_annual_precip': 'precip_mm',
            'test_r2': 'model_r2',
            'test_rmse': 'model_rmse',
            'mean_test_r2': 'spatial_r2',
            'mean_test_rmse': 'spatial_rm',
            'successful_folds': 'fold_count',
            'feature_count': 'feat_count',
            'avg_sap_flow': 'avg_sap'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in gdf.columns:
                gdf = gdf.rename(columns={old_name: new_name})
        
        # Keep only essential columns (shapefiles have limits)
        essential_cols = ['site', 'eco_clust', 'temp_c', 'precip_mm', 'elevation', 
                         'spatial_r2', 'spatial_rm', 'fold_count', 'geometry']
        
        # Filter to existing columns
        available_cols = [col for col in essential_cols if col in gdf.columns]
        gdf_clean = gdf[available_cols].copy()
        
        # Save shapefile
        output_path = os.path.join(self.output_qgis_dir, f'sapflux_sites_{self.timestamp}.shp')
        gdf_clean.to_file(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        # Also save as GeoJSON for better attribute support
        geojson_path = os.path.join(self.output_qgis_dir, f'sapflux_sites_{self.timestamp}.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"  ✅ Saved: {geojson_path}")
        
        return gdf_clean
    
    def create_qgis_cluster_polygons(self, merged_data):
        """Create cluster boundary polygons for QGIS"""
        print("\n🔷 Creating QGIS cluster boundary polygons...")
        
        gis_data = merged_data.dropna(subset=['latitude', 'longitude', 'ecosystem_cluster']).copy()
        
        if len(gis_data) < 6:  # Need at least 3 points per cluster minimum
            print("  ⚠️  Insufficient data for cluster polygons")
            return None
        
        cluster_polygons = []
        
        for cluster_id in sorted(gis_data['ecosystem_cluster'].unique()):
            cluster_sites = gis_data[gis_data['ecosystem_cluster'] == cluster_id]
            
            if len(cluster_sites) < 3:  # Need at least 3 points for a polygon
                print(f"    ⚠️  Cluster {cluster_id}: Only {len(cluster_sites)} sites, skipping polygon")
                continue
            
            try:
                # Create convex hull
                points = cluster_sites[['longitude', 'latitude']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                # Create polygon
                polygon = Polygon(hull_points)
                
                # Calculate cluster statistics
                cluster_stats = {
                    'cluster_id': int(cluster_id),
                    'site_count': len(cluster_sites),
                    'avg_temp': cluster_sites['mean_annual_temp'].mean() if 'mean_annual_temp' in cluster_sites.columns else None,
                    'avg_precip': cluster_sites['mean_annual_precip'].mean() if 'mean_annual_precip' in cluster_sites.columns else None,
                    'avg_r2': cluster_sites['mean_test_r2'].mean() if 'mean_test_r2' in cluster_sites.columns else None,
                    'geometry': polygon
                }
                
                cluster_polygons.append(cluster_stats)
                print(f"    ✅ Cluster {cluster_id}: {len(cluster_sites)} sites -> polygon")
                
            except Exception as e:
                print(f"    ❌ Error creating polygon for cluster {cluster_id}: {e}")
                continue
        
        if not cluster_polygons:
            print("  ⚠️  No cluster polygons created")
            return None
        
        # Create GeoDataFrame
        gdf_polygons = gpd.GeoDataFrame(cluster_polygons, crs='EPSG:4326')
        
        # Save shapefile
        output_path = os.path.join(self.output_qgis_dir, f'cluster_boundaries_{self.timestamp}.shp')
        gdf_polygons.to_file(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return gdf_polygons
    
    def create_performance_interpolation_raster(self, merged_data):
        """Create interpolated performance raster for QGIS"""
        print("\n🌐 Creating performance interpolation raster...")
        
        # Filter data with valid coordinates and performance
        raster_data = merged_data.dropna(subset=['latitude', 'longitude', 'mean_test_r2']).copy()
        
        if len(raster_data) < 10:
            print("  ⚠️  Insufficient data for raster interpolation")
            return None
        
        # Define raster bounds and resolution
        min_lon, max_lon = raster_data['longitude'].min(), raster_data['longitude'].max()
        min_lat, max_lat = raster_data['latitude'].min(), raster_data['latitude'].max()
        
        # Add buffer
        lon_buffer = (max_lon - min_lon) * 0.1
        lat_buffer = (max_lat - min_lat) * 0.1
        
        min_lon -= lon_buffer
        max_lon += lon_buffer
        min_lat -= lat_buffer
        max_lat += lat_buffer
        
        # Create grid
        resolution = 100  # 100x100 grid
        lon_grid = np.linspace(min_lon, max_lon, resolution)
        lat_grid = np.linspace(min_lat, max_lat, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Interpolate performance values
        performance_raster = griddata(
            (raster_data['longitude'], raster_data['latitude']),
            raster_data['mean_test_r2'],
            (lon_mesh, lat_mesh),
            method='cubic',
            fill_value=np.nan
        )
        
        # Create raster file
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, resolution, resolution)
        
        output_path = os.path.join(self.output_qgis_dir, f'performance_interpolation_{self.timestamp}.tif')
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=resolution,
            width=resolution,
            count=1,
            dtype=performance_raster.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(performance_raster, 1)
        
        print(f"  ✅ Saved: {output_path}")
        
        return output_path
    
    def generate_qgis_project_file(self):
        """Generate a QGIS project file (.qgs) that loads all created layers"""
        print("\n📋 Creating QGIS project file...")
        
        # Find all created files
        shapefiles = glob.glob(os.path.join(self.output_qgis_dir, '*.shp'))
        geojson_files = glob.glob(os.path.join(self.output_qgis_dir, '*.geojson'))
        raster_files = glob.glob(os.path.join(self.output_qgis_dir, '*.tif'))
        
        project_info = {
            'project_name': f'SAPFLUXNET_Analysis_{self.timestamp}',
            'created': datetime.now().isoformat(),
            'layers': {
                'vector': shapefiles + geojson_files,
                'raster': raster_files
            },
            'description': 'SAPFLUXNET ecosystem clustering and model performance analysis'
        }
        
        # Save project info as JSON (QGIS can import this)
        project_path = os.path.join(self.output_qgis_dir, f'sapflux_project_{self.timestamp}.json')
        with open(project_path, 'w') as f:
            json.dump(project_info, f, indent=2)
        
        print(f"  ✅ Project info saved: {project_path}")
        print(f"  💡 Import these layers manually in QGIS:")
        for layer_type, files in project_info['layers'].items():
            for file in files:
                print(f"    - {os.path.basename(file)} ({layer_type})")
        
        return project_info
    
    def run_comprehensive_visualization(self):
        """Run all visualization and export tasks"""
        print("🚀 Starting Comprehensive 3D & QGIS Visualization")
        print("=" * 60)
        
        try:
            # Load all data
            raw_data = self.load_comprehensive_data()
            merged_data = self.create_merged_dataset(raw_data)
            
            print(f"\n📊 Dataset Summary:")
            print(f"  Sites: {len(merged_data)}")
            print(f"  Clusters: {merged_data['ecosystem_cluster'].nunique() if 'ecosystem_cluster' in merged_data.columns else 'N/A'}")
            print(f"  Attributes: {len(merged_data.columns)}")
            
            # Create 3D visualizations
            print(f"\n🎨 Creating 3D Visualizations...")
            self.create_3d_geographic_performance(merged_data)
            self.create_3d_climate_space(merged_data)
            self.create_3d_feature_space_pca(merged_data)
            self.create_performance_surface_3d(merged_data)
            
            # Create QGIS exports
            print(f"\n🗺️  Creating QGIS Exports...")
            self.create_qgis_site_points(merged_data)
            self.create_qgis_cluster_polygons(merged_data)
            self.create_performance_interpolation_raster(merged_data)
            self.generate_qgis_project_file()
            
            print(f"\n🎉 Comprehensive visualization completed!")
            print(f"📁 3D plots available in: {self.output_3d_dir}")
            print(f"🗺️  QGIS data available in: {self.output_qgis_dir}")
            
            # Summary of outputs
            print(f"\n📋 Generated Files:")
            print(f"  3D Plots: {len(glob.glob(os.path.join(self.output_3d_dir, '*.html')))} interactive HTML files")
            print(f"  QGIS Layers: {len(glob.glob(os.path.join(self.output_qgis_dir, '*.shp')))} shapefiles")
            print(f"  GeoJSON: {len(glob.glob(os.path.join(self.output_qgis_dir, '*.geojson')))} files")
            print(f"  Rasters: {len(glob.glob(os.path.join(self.output_qgis_dir, '*.tif')))} files")
            
            return merged_data
            
        except Exception as e:
            print(f"\n❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced 3D and QGIS Visualization")
    parser.add_argument('--results-dir', default='../',
                       help="Base results directory")
    
    args = parser.parse_args()
    
    try:
        visualizer = Advanced3DQGISVisualizer(args.results_dir)
        merged_data = visualizer.run_comprehensive_visualization()
        
        print(f"\n✨ All visualizations complete!")
        print(f"💡 Next steps:")
        print(f"   - Open the HTML files in your browser for interactive 3D plots")
        print(f"   - Import the shapefiles/GeoJSON into QGIS for spatial analysis")
        print(f"   - Use the raster files for continuous surface analysis")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise

if __name__ == "__main__":
    main() 