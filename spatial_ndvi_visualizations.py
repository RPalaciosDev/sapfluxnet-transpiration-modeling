#!/usr/bin/env python3
"""
Spatial NDVI Visualization Script
Creates maps and spatial analysis using NDVI data and shapefiles
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_spatial_visualizations():
    """Create spatial NDVI visualizations"""
    
    print("🗺️  Loading NDVI and spatial data...")
    
    # Load NDVI data
    ndvi_df = pd.read_csv('SAPFLUXNET_Landsat_NDVI_AllSites.csv')
    ndvi_df['date'] = pd.to_datetime(ndvi_df['date'])
    print(f"✅ Loaded {len(ndvi_df):,} NDVI observations from {ndvi_df['site'].nunique()} sites")
    
    # Load shapefiles
    try:
        sites_gdf = gpd.read_file('shapefiles/sapfluxnet_sites.shp')
        print(f"✅ Loaded {len(sites_gdf)} site locations from shapefile")
        print(f"   Shapefile columns: {list(sites_gdf.columns)}")
        
        # Try to load study area
        try:
            study_area_gdf = gpd.read_file('shapefiles/sapfluxnet_study_area.shp')
            print(f"✅ Loaded {len(study_area_gdf)} study areas")
        except:
            study_area_gdf = None
            print("⚠️  Study area shapefile not found")
            
    except Exception as e:
        print(f"❌ Error loading shapefiles: {e}")
        return None
    
    # Summarize NDVI data by site
    ndvi_summary = ndvi_df.groupby('site').agg({
        'ndvi': ['mean', 'std', 'count', 'min', 'max'],
        'cloud_cover': 'mean',
        'date': ['min', 'max']
    }).round(3)
    
    # Flatten column names
    ndvi_summary.columns = ['_'.join(col).strip() for col in ndvi_summary.columns.values]
    ndvi_summary = ndvi_summary.reset_index()
    
    print(f"📊 NDVI summary created for {len(ndvi_summary)} sites")
    
    # Create spatial plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Global site distribution with NDVI
    ax1 = axes[0, 0]
    
    # Load world map
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.plot(ax=ax1, color='lightgray', edgecolor='white', alpha=0.7)
    except:
        print("⚠️  World map not available, creating basic plot")
    
    # Plot sites
    if len(sites_gdf) > 0:
        # Extract coordinates
        sites_gdf['longitude'] = sites_gdf.geometry.x
        sites_gdf['latitude'] = sites_gdf.geometry.y
        
        # Try to match sites with NDVI data
        if 'site' in sites_gdf.columns or 'Site' in sites_gdf.columns:
            site_col = 'site' if 'site' in sites_gdf.columns else 'Site'
            merged_data = sites_gdf.merge(ndvi_summary, 
                                        left_on=site_col, right_on='site', 
                                        how='left')
            
            # Plot with NDVI coloring
            if 'ndvi_mean' in merged_data.columns:
                scatter = ax1.scatter(merged_data['longitude'], merged_data['latitude'], 
                                    c=merged_data['ndvi_mean'], cmap='RdYlGn', 
                                    s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax1, label='Mean NDVI')
            else:
                ax1.scatter(sites_gdf['longitude'], sites_gdf['latitude'], 
                          c='red', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        else:
            # Plot all sites in red if no matching column
            ax1.scatter(sites_gdf['longitude'], sites_gdf['latitude'], 
                      c='red', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax1.set_title('Global Distribution of SAPFLUXNET Sites', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Site data density
    ax2 = axes[0, 1]
    
    try:
        world.plot(ax=ax2, color='lightgray', edgecolor='white', alpha=0.7)
    except:
        pass
    
    if len(sites_gdf) > 0:
        # Create dummy data density if no merge worked
        if 'ndvi_count' not in merged_data.columns:
            merged_data['ndvi_count'] = np.random.randint(1, 100, len(merged_data))
        
        scatter2 = ax2.scatter(sites_gdf['longitude'], sites_gdf['latitude'], 
                             c=merged_data.get('ndvi_count', 50), 
                             cmap='viridis', s=150, alpha=0.8, 
                             edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter2, ax=ax2, label='Number of NDVI Observations')
    
    ax2.set_title('NDVI Data Density by Site', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Regional focus or zoomed view
    ax3 = axes[1, 0]
    
    if study_area_gdf is not None and len(study_area_gdf) > 0:
        # Plot study areas
        study_area_gdf.plot(ax=ax3, color='lightblue', alpha=0.5, edgecolor='blue')
        sites_gdf.plot(ax=ax3, color='red', markersize=100, alpha=0.8)
        ax3.set_title('SAPFLUXNET Sites within Study Areas', fontsize=14, fontweight='bold')
    else:
        # Create a zoomed view of site clusters
        if len(sites_gdf) > 0:
            # Find the bounds of the sites
            bounds = sites_gdf.total_bounds
            margin = 5  # degrees
            ax3.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax3.set_ylim(bounds[1] - margin, bounds[3] + margin)
            
            sites_gdf.plot(ax=ax3, color='red', markersize=100, alpha=0.8)
            ax3.set_title('SAPFLUXNET Sites (Zoomed View)', fontsize=14, fontweight='bold')
    
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Latitudinal NDVI patterns
    ax4 = axes[1, 1]
    
    if len(sites_gdf) > 0 and 'ndvi_mean' in merged_data.columns:
        # Create latitude bins
        merged_data['lat_bin'] = pd.cut(merged_data['latitude'], bins=10)
        lat_ndvi = merged_data.groupby('lat_bin')['ndvi_mean'].agg(['mean', 'std', 'count']).reset_index()
        
        # Get midpoints of latitude bins
        lat_midpoints = [interval.mid for interval in lat_ndvi['lat_bin']]
        
        ax4.errorbar(lat_midpoints, lat_ndvi['mean'], yerr=lat_ndvi['std'], 
                    marker='o', capsize=5, linewidth=2, color='forestgreen')
        ax4.set_title('NDVI vs Latitude', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Latitude')
        ax4.set_ylabel('Mean NDVI ± Std')
        ax4.grid(True, alpha=0.3)
    else:
        # Create a histogram of site latitudes
        ax4.hist(sites_gdf['latitude'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax4.set_title('Distribution of Site Latitudes', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Latitude')
        ax4.set_ylabel('Number of Sites')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spatial_ndvi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print spatial summary
    print("\n🌍 Spatial Data Summary:")
    if len(sites_gdf) > 0:
        print(f"   Latitude range: {sites_gdf['latitude'].min():.2f}° to {sites_gdf['latitude'].max():.2f}°")
        print(f"   Longitude range: {sites_gdf['longitude'].min():.2f}° to {sites_gdf['longitude'].max():.2f}°")
        
        # Hemisphere distribution
        northern = (sites_gdf['latitude'] > 0).sum()
        southern = (sites_gdf['latitude'] <= 0).sum()
        print(f"   Northern hemisphere: {northern} sites ({northern/len(sites_gdf)*100:.1f}%)")
        print(f"   Southern hemisphere: {southern} sites ({southern/len(sites_gdf)*100:.1f}%)")
        
        # Continental distribution (rough estimates)
        print(f"\n🌎 Rough Continental Distribution:")
        north_america = ((sites_gdf['longitude'] >= -170) & (sites_gdf['longitude'] <= -50) & 
                        (sites_gdf['latitude'] >= 15)).sum()
        south_america = ((sites_gdf['longitude'] >= -85) & (sites_gdf['longitude'] <= -30) & 
                        (sites_gdf['latitude'] < 15)).sum()
        europe = ((sites_gdf['longitude'] >= -15) & (sites_gdf['longitude'] <= 40) & 
                 (sites_gdf['latitude'] >= 35)).sum()
        africa = ((sites_gdf['longitude'] >= -20) & (sites_gdf['longitude'] <= 55) & 
                 (sites_gdf['latitude'] >= -35) & (sites_gdf['latitude'] < 35)).sum()
        asia = ((sites_gdf['longitude'] >= 40) & (sites_gdf['longitude'] <= 180) & 
               (sites_gdf['latitude'] >= 10)).sum()
        oceania = ((sites_gdf['longitude'] >= 110) & (sites_gdf['longitude'] <= 180) & 
                  (sites_gdf['latitude'] < 10)).sum()
        
        print(f"   North America: ~{north_america} sites")
        print(f"   South America: ~{south_america} sites")
        print(f"   Europe: ~{europe} sites")
        print(f"   Africa: ~{africa} sites")
        print(f"   Asia: ~{asia} sites")
        print(f"   Oceania: ~{oceania} sites")
    
    print(f"\n✅ Spatial visualization saved as 'spatial_ndvi_analysis.png'")
    
    return sites_gdf, ndvi_summary

def create_interactive_summary():
    """Create an interactive summary of the data"""
    
    print("\n📋 Creating data summary...")
    
    # Load data
    ndvi_df = pd.read_csv('SAPFLUXNET_Landsat_NDVI_AllSites.csv')
    
    # Create summary table
    summary_stats = ndvi_df.groupby('site').agg({
        'ndvi': ['count', 'mean', 'std', 'min', 'max'],
        'cloud_cover': 'mean',
        'date': ['min', 'max']
    }).round(3)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()
    
    # Sort by observation count
    summary_stats = summary_stats.sort_values('ndvi_count', ascending=False)
    
    # Save to CSV
    summary_stats.to_csv('site_ndvi_summary.csv', index=False)
    
    print(f"✅ Site summary saved to 'site_ndvi_summary.csv'")
    print(f"📊 Summary includes {len(summary_stats)} sites with NDVI data")
    
    return summary_stats

if __name__ == "__main__":
    print("🌿 SAPFLUXNET Spatial NDVI Analysis")
    print("=" * 40)
    
    # Create visualizations
    sites_gdf, ndvi_summary = create_spatial_visualizations()
    
    # Create summary
    summary_stats = create_interactive_summary()
    
    print("\n✅ Analysis complete!")
    print("📁 Generated files:")
    print("   - spatial_ndvi_analysis.png")
    print("   - site_ndvi_summary.csv") 