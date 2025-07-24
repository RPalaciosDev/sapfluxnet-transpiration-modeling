"""
Simple 3D Visualizations for SAPFLUXNET Results
Using only matplotlib and standard libraries (no plotly/geopandas dependencies)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import glob
import json
from datetime import datetime
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class Simple3DVisualizer:
    """Creates 3D visualizations using matplotlib only"""
    
    def __init__(self, results_base_dir='../'):
        self.results_base_dir = results_base_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Output directory
        self.output_dir = os.path.join(results_base_dir, 'visualization', 'simple_3d_plots')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🎨 Simple 3D Visualizer initialized")
        print(f"📁 Output: {self.output_dir}")
    
    def load_data(self):
        """Load and merge available data"""
        print("\n📊 Loading data...")
        
        data = {}
        
        # Load clustering results
        cluster_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'evaluation/clustering_results/advanced_site_clusters_*.csv'
        )))
        if cluster_files:
            data['clusters'] = pd.read_csv(cluster_files[-1])
            print(f"✅ Clustering: {len(data['clusters'])} sites")
        
        # Load spatial validation results
        spatial_summaries_files = sorted(glob.glob(os.path.join(
            self.results_base_dir, 'models/results/cluster_spatial_validation/cluster_spatial_summaries_*.csv'
        )))
        if spatial_summaries_files:
            data['spatial_validation'] = pd.read_csv(spatial_summaries_files[-1])
            print(f"✅ Spatial validation: {len(data['spatial_validation'])} clusters")
        
        # Sample parquet files for geographic data
        parquet_dir = os.path.join(self.results_base_dir, 'processed_parquet')
        if os.path.exists(parquet_dir):
            parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')][:20]
            site_data = []
            
            for pfile in parquet_files:
                try:
                    df_sample = pd.read_parquet(os.path.join(parquet_dir, pfile))
                    if len(df_sample) > 0:
                        site_info = {
                            'site': df_sample['site'].iloc[0] if 'site' in df_sample.columns else pfile.replace('_comprehensive.parquet', ''),
                            'latitude': df_sample['latitude'].iloc[0] if 'latitude' in df_sample.columns else None,
                            'longitude': df_sample['longitude'].iloc[0] if 'longitude' in df_sample.columns else None,
                            'elevation': df_sample['elevation'].iloc[0] if 'elevation' in df_sample.columns else None,
                            'mean_annual_temp': df_sample['mean_annual_temp'].iloc[0] if 'mean_annual_temp' in df_sample.columns else None,
                            'mean_annual_precip': df_sample['mean_annual_precip'].iloc[0] if 'mean_annual_precip' in df_sample.columns else None,
                            'ecosystem_cluster': df_sample['ecosystem_cluster'].iloc[0] if 'ecosystem_cluster' in df_sample.columns else None,
                        }
                        site_data.append(site_info)
                except Exception as e:
                    continue
            
            if site_data:
                data['site_info'] = pd.DataFrame(site_data)
                print(f"✅ Site info: {len(data['site_info'])} sites")
        
        return self.merge_data(data)
    
    def merge_data(self, data):
        """Merge all data sources"""
        if 'clusters' not in data:
            raise ValueError("No clustering data found!")
        
        merged = data['clusters'].copy()
        merged = merged.rename(columns={'cluster': 'ecosystem_cluster'})
        
        # Merge with site info
        if 'site_info' in data:
            merged = merged.merge(data['site_info'], on='site', how='left', suffixes=('', '_geo'))
        
        # Merge with spatial validation
        if 'spatial_validation' in data:
            spatial_metrics = data['spatial_validation'].rename(columns={'cluster': 'ecosystem_cluster'})
            merged = merged.merge(
                spatial_metrics[['ecosystem_cluster', 'mean_test_r2', 'mean_test_rmse']], 
                on='ecosystem_cluster', 
                how='left'
            )
        
        # Clean data
        merged = merged.dropna(subset=['latitude', 'longitude'])
        
        print(f"  📊 Merged dataset: {len(merged)} sites, {len(merged.columns)} columns")
        return merged
    
    def create_3d_geographic_plot(self, data):
        """3D plot: Longitude, Latitude, Performance"""
        print("\n🌍 Creating 3D Geographic Performance plot...")
        
        plot_data = data.dropna(subset=['latitude', 'longitude', 'mean_test_r2']).copy()
        
        if len(plot_data) == 0:
            print("  ⚠️  No valid data")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            plot_data['longitude'], 
            plot_data['latitude'], 
            plot_data['mean_test_r2'],
            c=plot_data['ecosystem_cluster'], 
            cmap='viridis', 
            s=60, 
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Spatial Validation R²')
        ax.set_title('3D Geographic Distribution of Model Performance')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Ecosystem Cluster')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'geographic_3d_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_3d_climate_plot(self, data):
        """3D plot: Temperature, Precipitation, Elevation"""
        print("\n🌡️ Creating 3D Climate Space plot...")
        
        climate_cols = ['mean_annual_temp', 'mean_annual_precip', 'elevation']
        plot_data = data.dropna(subset=climate_cols).copy()
        
        if len(plot_data) == 0:
            print("  ⚠️  No valid climate data")
            return
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Size by performance if available
        sizes = plot_data['mean_test_r2'] * 200 if 'mean_test_r2' in plot_data.columns else 60
        
        scatter = ax.scatter(
            plot_data['mean_annual_temp'], 
            plot_data['mean_annual_precip'], 
            plot_data['elevation'],
            c=plot_data['ecosystem_cluster'], 
            cmap='viridis', 
            s=sizes, 
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Mean Annual Temperature (°C)')
        ax.set_ylabel('Mean Annual Precipitation (mm)')
        ax.set_zlabel('Elevation (m)')
        ax.set_title('3D Climate Space Distribution\n(Size = Model Performance)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Ecosystem Cluster')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'climate_space_3d_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_3d_pca_plot(self, data):
        """3D PCA plot of feature space"""
        print("\n🔬 Creating 3D PCA Feature Space plot...")
        
        # Get numerical features
        feature_cols = [col for col in data.columns 
                       if data[col].dtype in [np.float64, np.int64]
                       and col not in ['site', 'ecosystem_cluster', 'latitude', 'longitude']]
        
        plot_data = data.dropna(subset=feature_cols).copy()
        
        if len(plot_data) < 10 or len(feature_cols) < 3:
            print(f"  ⚠️  Insufficient data for PCA")
            return
        
        # Perform PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(plot_data[feature_cols])
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features_scaled)
        
        explained_var = pca.explained_variance_ratio_
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            pca_result[:, 0], 
            pca_result[:, 1], 
            pca_result[:, 2],
            c=plot_data['ecosystem_cluster'], 
            cmap='viridis', 
            s=60, 
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%} variance)')
        ax.set_title(f'3D Feature Space (PCA)\nClusters in {len(feature_cols)}-Dimensional Space')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Ecosystem Cluster')
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'pca_3d_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_multiple_view_plot(self, data):
        """Create multiple 3D views in subplots"""
        print("\n📊 Creating multiple view summary plot...")
        
        # Filter for complete data
        complete_data = data.dropna(subset=[
            'latitude', 'longitude', 'mean_annual_temp', 'mean_annual_precip', 'elevation', 'mean_test_r2'
        ]).copy()
        
        if len(complete_data) < 5:
            print("  ⚠️  Insufficient complete data")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Geographic + Performance
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(
            complete_data['longitude'], complete_data['latitude'], complete_data['mean_test_r2'],
            c=complete_data['ecosystem_cluster'], cmap='viridis', s=50, alpha=0.7
        )
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_zlabel('R²')
        ax1.set_title('Geographic Performance')
        
        # Plot 2: Climate Space
        ax2 = fig.add_subplot(222, projection='3d')
        scatter2 = ax2.scatter(
            complete_data['mean_annual_temp'], complete_data['mean_annual_precip'], complete_data['elevation'],
            c=complete_data['ecosystem_cluster'], cmap='viridis', s=50, alpha=0.7
        )
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Precipitation (mm)')
        ax2.set_zlabel('Elevation (m)')
        ax2.set_title('Climate Space')
        
        # Plot 3: Performance vs Climate
        ax3 = fig.add_subplot(223, projection='3d')
        scatter3 = ax3.scatter(
            complete_data['mean_annual_temp'], complete_data['mean_annual_precip'], complete_data['mean_test_r2'],
            c=complete_data['ecosystem_cluster'], cmap='viridis', s=50, alpha=0.7
        )
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Precipitation (mm)')
        ax3.set_zlabel('R²')
        ax3.set_title('Performance vs Climate')
        
        # Plot 4: Elevation vs Performance
        ax4 = fig.add_subplot(224, projection='3d')
        scatter4 = ax4.scatter(
            complete_data['elevation'], complete_data['mean_test_r2'], 
            complete_data['ecosystem_cluster'],  # Use cluster as Z for variety
            c=complete_data['ecosystem_cluster'], cmap='viridis', s=50, alpha=0.7
        )
        ax4.set_xlabel('Elevation (m)')
        ax4.set_ylabel('R²')
        ax4.set_zlabel('Cluster ID')
        ax4.set_title('Elevation vs Performance')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'multiple_views_3d_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def export_csv_for_external_tools(self, data):
        """Export clean CSV for external tools (QGIS, R, etc.)"""
        print("\n📄 Exporting CSV for external tools...")
        
        # Clean data for export
        export_data = data.copy()
        
        # Remove problematic columns
        cols_to_remove = [col for col in export_data.columns if 
                         col.endswith('_geo') or col.startswith('Unnamed')]
        export_data = export_data.drop(columns=cols_to_remove, errors='ignore')
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, f'sapflux_data_export_{self.timestamp}.csv')
        export_data.to_csv(csv_path, index=False)
        
        print(f"  ✅ CSV exported: {csv_path}")
        print(f"  📊 Columns: {len(export_data.columns)}, Rows: {len(export_data)}")
        print(f"  💡 Import this CSV into QGIS as delimited text layer using lat/lon coordinates")
        
        return csv_path
    
    def run_simple_visualization(self):
        """Run all simple visualizations"""
        print("🚀 Starting Simple 3D Visualization")
        print("=" * 50)
        
        try:
            # Load data
            data = self.load_data()
            
            print(f"\n📊 Dataset Summary:")
            print(f"  Sites: {len(data)}")
            print(f"  Clusters: {data['ecosystem_cluster'].nunique() if 'ecosystem_cluster' in data.columns else 'N/A'}")
            
            # Create visualizations
            self.create_3d_geographic_plot(data)
            self.create_3d_climate_plot(data)
            self.create_3d_pca_plot(data)
            self.create_multiple_view_plot(data)
            
            # Export data
            self.export_csv_for_external_tools(data)
            
            print(f"\n🎉 Simple visualization completed!")
            print(f"📁 All files saved to: {self.output_dir}")
            
            # List outputs
            png_files = glob.glob(os.path.join(self.output_dir, '*.png'))
            csv_files = glob.glob(os.path.join(self.output_dir, '*.csv'))
            
            print(f"\n📋 Generated Files:")
            print(f"  3D Plots: {len(png_files)} PNG files")
            print(f"  Data Export: {len(csv_files)} CSV files")
            
            return data
            
        except Exception as e:
            print(f"\n❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple 3D Visualization")
    parser.add_argument('--results-dir', default='../',
                       help="Base results directory")
    
    args = parser.parse_args()
    
    try:
        visualizer = Simple3DVisualizer(args.results_dir)
        data = visualizer.run_simple_visualization()
        
        print(f"\n✨ Visualization complete!")
        print(f"💡 Next steps:")
        print(f"   - View the PNG files for static 3D plots")
        print(f"   - Import the CSV into QGIS as a delimited text layer")
        print(f"   - Use the lat/lon columns for point coordinates")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        raise

if __name__ == "__main__":
    main() 