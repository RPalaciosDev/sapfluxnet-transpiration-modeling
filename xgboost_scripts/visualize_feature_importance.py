"""
Feature Importance Visualization for SAPFLUXNET XGBoost Models
Analyzes and visualizes feature importance from external memory training results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
from datetime import datetime

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def find_latest_importance_file(results_dir='xgboost_scripts/external_memory_models/random_split'):
    """Find the most recent feature importance file"""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all importance files
    importance_files = [f for f in os.listdir(results_dir) if f.endswith('_importance.csv')]
    
    if not importance_files:
        raise FileNotFoundError(f"No importance files found in {results_dir}")
    
    # Get the most recent file
    latest_file = max(importance_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
    return os.path.join(results_dir, latest_file)

def load_feature_importance(file_path):
    """Load feature importance data"""
    print(f"Loading feature importance from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} features")
    
    # Show sample of data
    print("\nSample of feature importance data:")
    print(df.head())
    
    return df

def create_environmental_heatmap(df, output_dir):
    """Create heatmap of environmental variable importance across time windows"""
    
    # Define environmental variables and their time windows
    env_vars = ['ta', 'rh', 'vpd', 'sw_in', 'ppfd_in', 'ws', 'precip', 'swc_shallow']
    time_windows = ['', '_lag_1h', '_lag_2h', '_lag_3h', '_lag_6h', '_lag_12h', '_lag_24h', 
                   '_mean_3h', '_mean_6h', '_mean_12h', '_mean_24h', '_mean_48h', '_mean_72h',
                   '_std_3h', '_std_6h', '_std_12h', '_std_24h', '_std_48h', '_std_72h']
    
    # Create matrix for heatmap
    heatmap_data = []
    row_labels = []
    
    for var in env_vars:
        for window in time_windows:
            feature_name = var + window
            # Find this feature in our data
            feature_data = df[df['feature_name'] == feature_name]
            if len(feature_data) > 0:
                importance = feature_data.iloc[0]['importance']
                heatmap_data.append(importance)
                row_labels.append(feature_name)
            else:
                heatmap_data.append(0)
                row_labels.append(feature_name)
    
    # Reshape data for heatmap
    n_vars = len(env_vars)
    n_windows = len(time_windows)
    heatmap_matrix = np.array(heatmap_data).reshape(n_vars, n_windows)
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Use log scale for better visualization
    heatmap_matrix_log = np.log1p(heatmap_matrix)  # log(1+x) to handle zeros
    
    im = ax.imshow(heatmap_matrix_log, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(n_windows))
    ax.set_xticklabels(time_windows, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(env_vars, fontsize=10)
    
    ax.set_xlabel('Time Window', fontsize=12)
    ax.set_ylabel('Environmental Variable', fontsize=12)
    ax.set_title('Environmental Variable Importance Heatmap\n(Log Scale)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log(Importance + 1)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/environmental_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_geographic_features_chart(df, output_dir):
    """Create bar chart of geographic feature importance"""
    
    # Define geographic features
    geo_features = ['longitude', 'latitude', 'latitude_abs', 'elevation', 'climate_zone_code', 
                   'timezone_offset', 'aridity_index', 'mean_annual_temp', 'mean_annual_precip']
    
    # Filter for geographic features
    geo_data = df[df['feature_name'].isin(geo_features)].copy()
    geo_data = geo_data.sort_values('importance', ascending=True)
    
    if len(geo_data) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bars = ax.barh(range(len(geo_data)), geo_data['importance'], color='forestgreen')
    ax.set_yticks(range(len(geo_data)))
    ax.set_yticklabels(geo_data['feature_name'], fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Geographic Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, geo_data['importance'])):
        ax.text(value + max(geo_data['importance']) * 0.01, i, f'{value:.0f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/geographic_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_site_characteristics_chart(df, output_dir):
    """Create bar chart of site characteristics importance"""
    
    # Define site characteristics
    site_features = ['tree_volume_index', 'stand_age', 'n_trees', 'tree_density', 'basal_area', 
                    'stand_height', 'pl_dbh', 'pl_age', 'pl_height', 'pl_leaf_area', 
                    'pl_sapw_depth', 'pl_sapw_area', 'pl_bark_thick', 'sapwood_leaf_ratio',
                    'tree_size_class_code', 'tree_age_class_code', 'species_name_code']
    
    # Filter for site features
    site_data = df[df['feature_name'].isin(site_features)].copy()
    site_data = site_data.sort_values('importance', ascending=True)
    
    if len(site_data) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    bars = ax.barh(range(len(site_data)), site_data['importance'], color='saddlebrown')
    ax.set_yticks(range(len(site_data)))
    ax.set_yticklabels(site_data['feature_name'], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Site Characteristics Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, site_data['importance'])):
        ax.text(value + max(site_data['importance']) * 0.01, i, f'{value:.0f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/site_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_solar_radiation_chart(df, output_dir):
    """Create bar chart of solar radiation feature family"""
    
    # Define solar radiation features
    solar_features = ['sw_in', 'ppfd_in', 'ext_rad', 'sw_in_lag_1h', 'sw_in_lag_2h', 'sw_in_lag_3h',
                     'sw_in_lag_6h', 'sw_in_lag_12h', 'sw_in_lag_24h', 'ppfd_in_lag_1h', 
                     'ppfd_in_lag_2h', 'ppfd_in_lag_3h', 'ppfd_in_lag_6h', 'ppfd_in_lag_12h', 
                     'ppfd_in_lag_24h', 'sw_in_mean_3h', 'sw_in_mean_6h', 'sw_in_mean_12h',
                     'sw_in_mean_24h', 'sw_in_mean_48h', 'sw_in_mean_72h', 'sw_in_std_3h',
                     'sw_in_std_6h', 'sw_in_std_12h', 'sw_in_std_24h', 'sw_in_std_48h', 'sw_in_std_72h']
    
    # Filter for solar features
    solar_data = df[df['feature_name'].isin(solar_features)].copy()
    solar_data = solar_data.sort_values('importance', ascending=True)
    
    if len(solar_data) == 0:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    bars = ax.barh(range(len(solar_data)), solar_data['importance'], color='gold')
    ax.set_yticks(range(len(solar_data)))
    ax.set_yticklabels(solar_data['feature_name'], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Solar Radiation Feature Family Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, solar_data['importance'])):
        ax.text(value + max(solar_data['importance']) * 0.01, i, f'{value:.0f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/solar_radiation_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_reduction_chart(df, output_dir):
    """Create feature reduction analysis chart"""
    
    # Calculate cumulative importance
    cumulative_importance = df['importance'].cumsum()
    total_importance = cumulative_importance.iloc[-1]
    cumulative_pct = (cumulative_importance / total_importance) * 100
    
    # Find thresholds
    thresholds = [50, 60, 70, 80, 85, 90, 95, 98, 99]
    feature_counts = []
    
    for threshold in thresholds:
        count = np.where(cumulative_pct >= threshold)[0]
        feature_counts.append(len(count) + 1 if len(count) > 0 else len(df))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bars = ax.bar(thresholds, feature_counts, color='lightcoral', alpha=0.7)
    ax.set_xlabel('Cumulative Importance (%)', fontsize=12)
    ax.set_ylabel('Number of Features Required', fontsize=12)
    ax.set_title('Feature Reduction Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_counts) * 0.01,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_reduction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_lag_feature_chart(df, output_dir):
    """Create lag feature analysis chart"""
    
    # Define lag periods
    lag_periods = ['1h', '2h', '3h', '6h', '12h', '24h']
    
    # Calculate total importance for each lag period
    lag_importance = {}
    
    for lag in lag_periods:
        lag_features = df[df['feature_name'].str.contains(f'_lag_{lag}', na=False)]
        total_importance = lag_features['importance'].sum()
        lag_importance[lag] = total_importance
    
    # Create bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    periods = list(lag_importance.keys())
    importance_values = list(lag_importance.values())
    
    bars = ax.bar(periods, importance_values, color='skyblue', alpha=0.7)
    ax.set_xlabel('Lag Period', fontsize=12)
    ax.set_ylabel('Total Feature Importance (Gain)', fontsize=12)
    ax.set_title('Lag Feature Importance by Time Period', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, importance_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values) * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lag_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def categorize_features(df):
    """Categorize features into different types for better analysis"""
    
    def get_feature_category(feature_name):
        """Determine feature category based on name"""
        feature_name = feature_name.lower()
        
        # Environmental variables
        if any(var in feature_name for var in ['ta', 'temp', 'air_temp']):
            return 'Air Temperature'
        elif any(var in feature_name for var in ['rh', 'humidity']):
            return 'Humidity'
        elif any(var in feature_name for var in ['vpd', 'vapor']):
            return 'VPD'
        elif any(var in feature_name for var in ['sw_in', 'solar', 'radiation']):
            return 'Solar Radiation'
        elif any(var in feature_name for var in ['ws', 'wind']):
            return 'Wind Speed'
        elif any(var in feature_name for var in ['precip', 'rain']):
            return 'Precipitation'
        elif any(var in feature_name for var in ['swc', 'soil']):
            return 'Soil Moisture'
        elif any(var in feature_name for var in ['ppfd', 'par']):
            return 'Light'
        
        # Temporal features
        elif any(var in feature_name for var in ['hour', 'time']):
            return 'Hour'
        elif any(var in feature_name for var in ['day_of_year', 'doy']):
            return 'Day of Year'
        elif any(var in feature_name for var in ['month']):
            return 'Month'
        elif any(var in feature_name for var in ['year']):
            return 'Year'
        
        # Lag features
        elif any(var in feature_name for var in ['lag_', 'prev_']):
            return 'Lag Features'
        
        # Rolling window features
        elif any(var in feature_name for var in ['rolling_', 'mean_', 'std_', 'min_', 'max_']):
            return 'Rolling Window'
        
        # Site characteristics
        elif any(var in feature_name for var in ['site_lat', 'site_lon', 'elevation']):
            return 'Site Location'
        
        # Default category
        else:
            return 'Other'
    
    # Add category column
    df['category'] = df['feature_name'].apply(get_feature_category)
    
    return df

def create_feature_importance_plots(df, output_dir='feature_importance_plots'):
    """Create comprehensive feature importance visualizations"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top 20 Features Bar Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    top_20 = df.head(20)
    bars = ax.barh(range(len(top_20)), top_20['importance'], color='steelblue')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature_name'], fontsize=10)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Top 20 Most Important Features for Sap Flow Prediction', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_20['importance'])):
        ax.text(value + max(top_20['importance']) * 0.01, i, f'{value:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_20_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Environmental vs Temporal Heatmap
    create_environmental_heatmap(df, output_dir)
    
    # 3. Geographic Features Bar Chart
    create_geographic_features_chart(df, output_dir)
    
    # 4. Site Characteristics Rankings
    create_site_characteristics_chart(df, output_dir)
    
    # 5. Solar Radiation Feature Family
    create_solar_radiation_chart(df, output_dir)
    
    # 6. Feature Reduction Analysis
    create_feature_reduction_chart(df, output_dir)
    
    # 7. Lag Feature Analysis
    create_lag_feature_chart(df, output_dir)
    
    # Calculate some basic stats for the summary
    category_summary = df.groupby('category')['importance'].agg(['sum', 'mean', 'count']).reset_index()
    category_summary = category_summary.sort_values('sum', ascending=False)
    
    # Calculate cumulative importance for summary
    cumulative_importance = df['importance'].cumsum()
    total_importance = cumulative_importance.iloc[-1]
    cumulative_pct = (cumulative_importance / total_importance) * 100
    
    # Find features needed for 80% and 95% importance
    features_80 = np.where(cumulative_pct >= 80)[0][0] + 1
    features_95 = np.where(cumulative_pct >= 95)[0][0] + 1
    
    return category_summary, features_80, features_95

def create_summary_report(df, category_summary, features_80, features_95, output_dir='feature_importance_plots'):
    """Create a text summary report"""
    
    report_path = f'{output_dir}/feature_importance_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("SAPFLUXNET Feature Importance Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis completed: {datetime.now()}\n")
        f.write(f"Total features analyzed: {len(df)}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 15 + "\n")
        f.write(f"• {features_80} features explain 80% of model importance\n")
        f.write(f"• {features_95} features explain 95% of model importance\n")
        f.write(f"• Most important feature: {df.iloc[0]['feature_name']} (importance: {df.iloc[0]['importance']:.4f})\n\n")
        
        f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
        f.write("-" * 35 + "\n")
        for i, row in df.head(10).iterrows():
            f.write(f"{i+1:2d}. {row['feature_name']:<30} {row['importance']:.4f} ({row['category']})\n")
        
        f.write("\nFEATURE CATEGORY RANKINGS:\n")
        f.write("-" * 27 + "\n")
        for i, row in category_summary.iterrows():
            f.write(f"{i+1:2d}. {row['category']:<20} Total: {row['sum']:.3f}, "
                   f"Avg: {row['mean']:.4f}, Count: {row['count']}\n")
        
        f.write("\nINSIGHTS FOR MODEL INTERPRETATION:\n")
        f.write("-" * 35 + "\n")
        
        # Environmental insights
        env_categories = ['Air Temperature', 'Humidity', 'VPD', 'Solar Radiation', 'Wind Speed', 
                         'Precipitation', 'Soil Moisture', 'Light']
        env_importance = category_summary[category_summary['category'].isin(env_categories)]['sum'].sum()
        total_importance = category_summary['sum'].sum()
        
        f.write(f"• Environmental variables account for {env_importance/total_importance*100:.1f}% of total importance\n")
        
        # Temporal insights
        temporal_categories = ['Hour', 'Day of Year', 'Month', 'Year']
        temporal_importance = category_summary[category_summary['category'].isin(temporal_categories)]['sum'].sum()
        f.write(f"• Temporal features account for {temporal_importance/total_importance*100:.1f}% of total importance\n")
        
        # Lag and rolling window insights
        lag_rolling_categories = ['Lag Features', 'Rolling Window']
        lag_rolling_importance = category_summary[category_summary['category'].isin(lag_rolling_categories)]['sum'].sum()
        f.write(f"• Lag and rolling window features account for {lag_rolling_importance/total_importance*100:.1f}% of total importance\n")
        
        f.write(f"\nThis analysis helps understand which environmental and temporal factors\n")
        f.write(f"are most critical for predicting sap flow in the SAPFLUXNET dataset.\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    """Main function to run feature importance analysis"""
    
    print("SAPFLUXNET Feature Importance Visualization")
    print("=" * 50)
    
    try:
        # Find and load the latest importance file
        importance_file = find_latest_importance_file()
        df = load_feature_importance(importance_file)
        
        # Categorize features
        df = categorize_features(df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        category_summary, features_80, features_95 = create_feature_importance_plots(df)
        
        # Create summary report
        print("Creating summary report...")
        create_summary_report(df, category_summary, features_80, features_95)
        
        print("\n✅ Feature importance analysis completed!")
        print("📊 Visualizations saved to: feature_importance_plots/")
        print("📝 Summary report: feature_importance_plots/feature_importance_summary.txt")
        
        print(f"\nQuick Summary:")
        print(f"  • Total features: {len(df)}")
        print(f"  • Features for 80% importance: {features_80}")
        print(f"  • Features for 95% importance: {features_95}")
        print(f"  • Most important: {df.iloc[0]['feature_name']}")
        print(f"  • Top category: {category_summary.iloc[0]['category']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 