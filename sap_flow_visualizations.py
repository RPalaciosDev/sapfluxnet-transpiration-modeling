#!/usr/bin/env python3
"""
Comprehensive Sap Flow Visualization Script for SAPFLUXNET Data
Creates multiple types of visualizations to analyze transpiration patterns
Now includes integration with trained XGBoost ecosystem models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 3D plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️  Plotly not available - 3D plots will be skipped")
    PLOTLY_AVAILABLE = False

# Model integration imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - model integration will be skipped")
    XGBOOST_AVAILABLE = False

class SapFlowVisualizer:
    """Comprehensive sap flow visualization toolkit with model integration"""
    
    def __init__(self, data_dir='processed_parquet', output_dir='sap_flow_plots', 
                 models_dir='ecosystem/models/results/cluster_models'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Model integration
        self.models = {}
        self.feature_names = []
        self.cluster_assignments = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🌊 Sap Flow Visualizer initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🤖 Models directory: {models_dir}")
        
        if PLOTLY_AVAILABLE:
            print("🎯 3D plotting enabled with Plotly")
        else:
            print("⚠️  3D plotting disabled - install plotly for 3D visualizations")
            
        if XGBOOST_AVAILABLE:
            print("🤖 XGBoost model integration enabled")
            self.load_trained_models()
        else:
            print("⚠️  XGBoost not available - model integration disabled")

    def load_trained_models(self):
        """Load trained XGBoost cluster models"""
        if not XGBOOST_AVAILABLE:
            return
            
        print(f"\n🤖 Loading trained cluster models...")
        
        # Find model files
        model_files = glob.glob(os.path.join(self.models_dir, 'xgb_model_cluster_*.json'))
        
        if not model_files:
            print(f"❌ No trained models found in {self.models_dir}")
            return
        
        # Load models
        for model_file in model_files:
            try:
                # Extract cluster ID from filename
                filename = os.path.basename(model_file)
                parts = filename.replace('.json', '').split('_')
                cluster_id = None
                
                for i, part in enumerate(parts):
                    if part == 'cluster' and i + 1 < len(parts):
                        try:
                            cluster_id = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                
                if cluster_id is not None:
                    # Try different loading methods for compatibility
                    model = xgb.Booster()
                    try:
                        model.load_model(model_file)
                        self.models[cluster_id] = model
                        print(f"  ✅ Loaded model for cluster {cluster_id}")
                    except Exception as load_error:
                        # Try alternative loading method for older XGBoost versions
                        print(f"  ⚠️  Standard loading failed for cluster {cluster_id}: {str(load_error)[:100]}...")
                        try:
                            # For very old XGBoost versions, try loading as binary
                            with open(model_file, 'rb') as f:
                                model_data = f.read()
                            # Skip models that can't be loaded due to version incompatibility
                            print(f"  ❌ Model format incompatible for cluster {cluster_id} (XGBoost version mismatch)")
                        except:
                            print(f"  ❌ Could not load model for cluster {cluster_id}")
                    
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        
        # Load feature names from importance files
        self.load_feature_names()
        
        # Load cluster assignments
        self.load_cluster_assignments()
        
        print(f"📊 Loaded {len(self.models)} cluster models")
        
        if len(self.models) == 0:
            print(f"⚠️  WARNING: No models loaded due to XGBoost version incompatibility")
            print(f"   Current XGBoost version: {xgb.__version__}")
            print(f"   Models may have been saved with a newer XGBoost version")
            print(f"   Visualization will continue without model predictions")

    def load_feature_names(self):
        """Load feature names from feature importance files"""
        importance_files = glob.glob(os.path.join(self.models_dir, 'feature_importance_cluster_*.csv'))
        
        if importance_files:
            # Use the first available importance file to get feature names
            try:
                importance_df = pd.read_csv(importance_files[0])
                if 'feature_name' in importance_df.columns:
                    self.feature_names = importance_df['feature_name'].tolist()
                    print(f"  ✅ Loaded {len(self.feature_names)} feature names")
            except Exception as e:
                print(f"  ⚠️  Could not load feature names: {e}")

    def load_cluster_assignments(self):
        """Load cluster assignments for sites"""
        cluster_files = glob.glob('ecosystem/evaluation/clustering_results/advanced_site_clusters_*.csv')
        
        if cluster_files:
            try:
                latest_file = max(cluster_files, key=os.path.getctime)
                clusters_df = pd.read_csv(latest_file)
                
                # Check for different possible column names
                cluster_col = None
                if 'ecosystem_cluster' in clusters_df.columns:
                    cluster_col = 'ecosystem_cluster'
                elif 'cluster' in clusters_df.columns:
                    cluster_col = 'cluster'
                else:
                    print(f"  ⚠️  No cluster column found. Available columns: {list(clusters_df.columns)}")
                    return
                
                self.cluster_assignments = dict(zip(clusters_df['site'], clusters_df[cluster_col]))
                print(f"  ✅ Loaded cluster assignments for {len(self.cluster_assignments)} sites using column '{cluster_col}'")
            except Exception as e:
                print(f"  ⚠️  Could not load cluster assignments: {e}")

    def predict_sap_flow(self, data, site_name=None):
        """Make sap flow predictions using trained models"""
        if not self.models or not XGBOOST_AVAILABLE:
            return None
        
        # Determine cluster for prediction
        cluster_id = None
        if site_name and site_name in self.cluster_assignments:
            cluster_id = self.cluster_assignments[site_name]
        elif 'ecosystem_cluster' in data.columns:
            cluster_id = data['ecosystem_cluster'].iloc[0]
        
        if cluster_id is None or cluster_id not in self.models:
            print(f"  ⚠️  No model available for cluster {cluster_id}")
            return None
        
        try:
            # Prepare features (this is simplified - you may need to adjust based on your feature engineering)
            # Exclude known non-feature columns and object columns
            excluded_cols = ['sap_flow', 'site', 'TIMESTAMP', 'ecosystem_cluster', 'plant_id', 'solar_TIMESTAMP']
            feature_cols = [col for col in data.columns 
                          if col not in excluded_cols and data[col].dtype in ['int64', 'float64', 'bool', 'category']]
            
            if len(feature_cols) == 0:
                print("  ❌ No features available for prediction")
                return None
            
            # Create feature matrix
            X = data[feature_cols].fillna(0)  # Simple fillna - you may want more sophisticated handling
            
            # Convert to DMatrix
            dmatrix = xgb.DMatrix(X)
            
            # Make predictions
            predictions = self.models[cluster_id].predict(dmatrix)
            
            return predictions
            
        except Exception as e:
            print(f"  ❌ Error making predictions: {e}")
            return None

    def load_sample_sites(self, n_sites=10):
        """Load a sample of sites for visualization"""
        print(f"\n📊 Loading sample of {n_sites} sites...")
        
        # Get available parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        
        if not parquet_files:
            print(f"❌ No parquet files found in {self.data_dir}")
            return None
        
        # Sample files
        sample_files = parquet_files[:n_sites]
        
        combined_data = []
        for file_path in sample_files:
            try:
                site_name = os.path.basename(file_path).replace('_comprehensive.parquet', '')
                df = pd.read_parquet(file_path)
                
                # Add site identifier
                df['site'] = site_name
                
                # Convert timestamp
                if 'TIMESTAMP' in df.columns:
                    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                
                # Check for required columns
                required_cols = ['sap_flow', 'ta', 'vpd']
                if all(col in df.columns for col in required_cols):
                    combined_data.append(df)
                    print(f"  ✅ {site_name}: {len(df):,} records")
                else:
                    print(f"  ⚠️  {site_name}: Missing required columns")
                    
            except Exception as e:
                print(f"  ❌ Error loading {file_path}: {e}")
                continue
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            print(f"📊 Combined dataset: {len(result):,} records from {len(combined_data)} sites")
            return result
        else:
            print("❌ No valid data loaded")
            return None

    def create_daily_patterns(self, data):
        """Create daily sap flow pattern visualizations"""
        print("\n🌅 Creating daily pattern visualizations...")
        
        if 'TIMESTAMP' not in data.columns:
            print("❌ TIMESTAMP column not found")
            return
        
        # Extract hour from timestamp
        data['hour'] = data['TIMESTAMP'].dt.hour
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Daily Sap Flow Patterns', fontsize=16, fontweight='bold')
        
        # 1. Average daily pattern across all sites
        daily_pattern = data.groupby('hour')['sap_flow'].agg(['mean', 'std']).reset_index()
        
        axes[0,0].plot(daily_pattern['hour'], daily_pattern['mean'], 'b-', linewidth=2, label='Mean')
        axes[0,0].fill_between(daily_pattern['hour'], 
                              daily_pattern['mean'] - daily_pattern['std'],
                              daily_pattern['mean'] + daily_pattern['std'], 
                              alpha=0.3, color='blue', label='±1 SD')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Sap Flow Rate')
        axes[0,0].set_title('Average Daily Pattern (All Sites)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Daily patterns by site
        site_samples = data['site'].unique()[:5]  # Show top 5 sites
        for site in site_samples:
            site_data = data[data['site'] == site]
            site_pattern = site_data.groupby('hour')['sap_flow'].mean()
            axes[0,1].plot(site_pattern.index, site_pattern.values, 
                          marker='o', markersize=3, label=site, alpha=0.8)
        
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Sap Flow Rate')
        axes[0,1].set_title('Daily Patterns by Site')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Heatmap of hourly patterns
        if len(data) > 1000:  # Only if we have enough data
            # Create pivot table for heatmap
            data['date'] = data['TIMESTAMP'].dt.date
            sample_dates = sorted(data['date'].unique())[:30]  # Sample 30 days
            heatmap_data = data[data['date'].isin(sample_dates)]
            
            pivot_data = heatmap_data.pivot_table(
                values='sap_flow', 
                index='date', 
                columns='hour', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, ax=axes[1,0], cmap='YlOrRd', 
                       cbar_kws={'label': 'Sap Flow Rate'})
            axes[1,0].set_title('Daily Pattern Heatmap (30 days)')
            axes[1,0].set_xlabel('Hour of Day')
            axes[1,0].set_ylabel('Date')
        
        # 4. Box plot of hourly distributions
        sample_hours = [6, 9, 12, 15, 18, 21]  # Sample key hours
        hourly_data = []
        hour_labels = []
        
        for hour in sample_hours:
            hour_sap_flow = data[data['hour'] == hour]['sap_flow'].dropna()
            if len(hour_sap_flow) > 10:
                hourly_data.append(hour_sap_flow)
                hour_labels.append(f"{hour}:00")
        
        if hourly_data:
            axes[1,1].boxplot(hourly_data, labels=hour_labels)
            axes[1,1].set_xlabel('Hour of Day')
            axes[1,1].set_ylabel('Sap Flow Rate')
            axes[1,1].set_title('Hourly Distribution')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'daily_patterns_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_environmental_responses(self, data):
        """Create environmental response visualizations"""
        print("\n🌡️ Creating environmental response visualizations...")
        
        # Check for required columns
        env_vars = ['ta', 'vpd', 'sw_in', 'ppfd_in']
        available_vars = [var for var in env_vars if var in data.columns]
        
        if not available_vars:
            print("❌ No environmental variables found")
            return
        
        n_vars = len(available_vars)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sap Flow Environmental Responses', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i, var in enumerate(available_vars[:4]):
            if i >= 4:
                break
                
            # Filter out extreme values for better visualization
            var_data = data[[var, 'sap_flow']].dropna()
            var_q99 = var_data[var].quantile(0.99)
            var_q01 = var_data[var].quantile(0.01)
            sap_q99 = var_data['sap_flow'].quantile(0.99)
            
            filtered_data = var_data[
                (var_data[var] >= var_q01) & 
                (var_data[var] <= var_q99) & 
                (var_data['sap_flow'] <= sap_q99) &
                (var_data['sap_flow'] >= 0)
            ]
            
            if len(filtered_data) < 100:
                continue
            
            # Create scatter plot with trend line
            if len(filtered_data) > 5000:  # Sample for performance
                sample_data = filtered_data.sample(5000)
            else:
                sample_data = filtered_data
            
            axes[i].scatter(sample_data[var], sample_data['sap_flow'], 
                           alpha=0.5, s=1, color='blue')
            
            # Add trend line
            try:
                z = np.polyfit(sample_data[var], sample_data['sap_flow'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(sample_data[var].min(), sample_data[var].max(), 100)
                axes[i].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                
                # Calculate R²
                correlation = np.corrcoef(sample_data[var], sample_data['sap_flow'])[0,1]
                axes[i].text(0.05, 0.95, f'r = {correlation:.3f}', 
                           transform=axes[i].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            except:
                pass
            
            # Labels
            var_labels = {
                'ta': 'Air Temperature (°C)',
                'vpd': 'Vapor Pressure Deficit (kPa)',
                'sw_in': 'Solar Radiation (W/m²)',
                'ppfd_in': 'PPFD (μmol/m²/s)'
            }
            
            axes[i].set_xlabel(var_labels.get(var, var))
            axes[i].set_ylabel('Sap Flow Rate')
            axes[i].set_title(f'Sap Flow vs {var_labels.get(var, var)}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_vars), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'environmental_responses_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_seasonal_analysis(self, data):
        """Create seasonal analysis visualizations"""
        print("\n🍂 Creating seasonal analysis...")
        
        if 'TIMESTAMP' not in data.columns:
            print("❌ TIMESTAMP column not found")
            return
        
        # Extract seasonal information
        data['month'] = data['TIMESTAMP'].dt.month
        data['day_of_year'] = data['TIMESTAMP'].dt.dayofyear
        
        # Define seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        data['season'] = data['month'].apply(get_season)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seasonal Sap Flow Analysis', fontsize=16, fontweight='bold')
        
        # 1. Monthly averages
        monthly_avg = data.groupby('month')['sap_flow'].agg(['mean', 'std']).reset_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        axes[0,0].bar(monthly_avg['month'], monthly_avg['mean'], 
                     yerr=monthly_avg['std'], capsize=5, alpha=0.8, color='green')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Average Sap Flow Rate')
        axes[0,0].set_title('Monthly Average Sap Flow')
        axes[0,0].set_xticks(range(1, 13))
        axes[0,0].set_xticklabels(month_names, rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Seasonal box plots
        seasonal_data = [data[data['season'] == season]['sap_flow'].dropna() 
                        for season in ['Spring', 'Summer', 'Fall', 'Winter']]
        seasonal_data = [season_data for season_data in seasonal_data if len(season_data) > 0]
        
        if seasonal_data:
            axes[0,1].boxplot(seasonal_data, labels=['Spring', 'Summer', 'Fall', 'Winter'])
            axes[0,1].set_ylabel('Sap Flow Rate')
            axes[0,1].set_title('Seasonal Distribution')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Day of year pattern
        if len(data) > 1000:
            daily_pattern = data.groupby('day_of_year')['sap_flow'].mean()
            axes[1,0].plot(daily_pattern.index, daily_pattern.values, 'b-', alpha=0.8)
            axes[1,0].set_xlabel('Day of Year')
            axes[1,0].set_ylabel('Average Sap Flow Rate')
            axes[1,0].set_title('Annual Pattern')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Temperature vs sap flow by season
        if 'ta' in data.columns:
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
            colors = ['green', 'red', 'orange', 'blue']
            
            for season, color in zip(seasons, colors):
                season_data = data[data['season'] == season]
                if len(season_data) > 100:
                    sample_size = min(1000, len(season_data))
                    sample = season_data.sample(sample_size)
                    axes[1,1].scatter(sample['ta'], sample['sap_flow'], 
                                    alpha=0.6, s=10, color=color, label=season)
            
            axes[1,1].set_xlabel('Air Temperature (°C)')
            axes[1,1].set_ylabel('Sap Flow Rate')
            axes[1,1].set_title('Temperature Response by Season')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'seasonal_analysis_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_site_comparison(self, data):
        """Create site comparison visualizations"""
        print("\n🌍 Creating site comparison analysis...")
        
        # Get sites with sufficient data
        site_counts = data['site'].value_counts()
        good_sites = site_counts[site_counts > 1000].index[:8]  # Top 8 sites
        
        if len(good_sites) < 2:
            print("❌ Not enough sites with sufficient data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Site Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average sap flow by site
        site_means = data[data['site'].isin(good_sites)].groupby('site')['sap_flow'].agg(['mean', 'std'])
        
        axes[0,0].bar(range(len(site_means)), site_means['mean'], 
                     yerr=site_means['std'], capsize=5, alpha=0.8)
        axes[0,0].set_xlabel('Site')
        axes[0,0].set_ylabel('Average Sap Flow Rate')
        axes[0,0].set_title('Average Sap Flow by Site')
        axes[0,0].set_xticks(range(len(site_means)))
        axes[0,0].set_xticklabels(site_means.index, rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Daily patterns by site
        for site in good_sites[:5]:  # Show top 5
            site_data = data[data['site'] == site]
            if 'TIMESTAMP' in site_data.columns:
                site_data['hour'] = site_data['TIMESTAMP'].dt.hour
                hourly_pattern = site_data.groupby('hour')['sap_flow'].mean()
                axes[0,1].plot(hourly_pattern.index, hourly_pattern.values, 
                              marker='o', markersize=3, label=site, alpha=0.8)
        
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Average Sap Flow Rate')
        axes[0,1].set_title('Daily Patterns by Site')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribution comparison
        site_data_list = []
        site_labels = []
        for site in good_sites[:6]:
            site_sap_flow = data[data['site'] == site]['sap_flow'].dropna()
            if len(site_sap_flow) > 100:
                site_data_list.append(site_sap_flow)
                site_labels.append(site)
        
        if site_data_list:
            axes[1,0].boxplot(site_data_list, labels=site_labels)
            axes[1,0].set_ylabel('Sap Flow Rate')
            axes[1,0].set_title('Distribution by Site')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Environmental response comparison
        if 'vpd' in data.columns:
            for site in good_sites[:4]:
                site_data = data[data['site'] == site]
                if len(site_data) > 200:
                    # Bin VPD data for cleaner visualization
                    site_data_clean = site_data[
                        (site_data['vpd'] > 0) & 
                        (site_data['vpd'] < site_data['vpd'].quantile(0.95)) &
                        (site_data['sap_flow'] > 0) &
                        (site_data['sap_flow'] < site_data['sap_flow'].quantile(0.95))
                    ]
                    
                    if len(site_data_clean) > 100:
                        # Create bins for smoother visualization
                        vpd_bins = pd.cut(site_data_clean['vpd'], bins=20)
                        binned_data = site_data_clean.groupby(vpd_bins)['sap_flow'].mean()
                        bin_centers = [interval.mid for interval in binned_data.index]
                        
                        axes[1,1].plot(bin_centers, binned_data.values, 
                                      marker='o', markersize=4, label=site, alpha=0.8)
        
            axes[1,1].set_xlabel('Vapor Pressure Deficit (kPa)')
            axes[1,1].set_ylabel('Average Sap Flow Rate')
            axes[1,1].set_title('VPD Response by Site')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'site_comparison_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    def create_summary_statistics(self, data):
        """Create summary statistics table"""
        print("\n📊 Creating summary statistics...")
        
        # Overall statistics
        stats = {
            'Total Records': f"{len(data):,}",
            'Number of Sites': data['site'].nunique(),
            'Date Range': f"{data['TIMESTAMP'].min()} to {data['TIMESTAMP'].max()}" if 'TIMESTAMP' in data.columns else "N/A",
            'Mean Sap Flow': f"{data['sap_flow'].mean():.3f}",
            'Median Sap Flow': f"{data['sap_flow'].median():.3f}",
            'Max Sap Flow': f"{data['sap_flow'].max():.3f}",
            'Min Sap Flow': f"{data['sap_flow'].min():.3f}",
            'Std Sap Flow': f"{data['sap_flow'].std():.3f}"
        }
        
        # Site-level statistics
        site_stats = data.groupby('site').agg({
            'sap_flow': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, f'summary_statistics_{self.timestamp}.txt')
        with open(stats_path, 'w') as f:
            f.write("SAPFLUXNET Sap Flow Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nSite-Level Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(site_stats.to_string())
        
        print(f"  ✅ Saved: {stats_path}")
        
        # Print summary to console
        print(f"\n📋 Data Summary:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def create_3d_environmental_space(self, data):
        """Create 3D plot in environmental space (temp, VPD, radiation)"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly required for 3D plots")
            return
            
        print("\n🌌 Creating 3D Environmental Space plot...")
        
        # Check for required columns
        required_cols = ['ta', 'vpd', 'sap_flow']
        if not all(col in data.columns for col in required_cols):
            print(f"❌ Missing required columns: {required_cols}")
            return
        
        # Filter and sample data for performance
        plot_data = data[required_cols + (['sw_in'] if 'sw_in' in data.columns else [])].dropna()
        
        # Remove outliers
        for col in ['ta', 'vpd', 'sap_flow']:
            q99 = plot_data[col].quantile(0.99)
            q01 = plot_data[col].quantile(0.01)
            plot_data = plot_data[(plot_data[col] >= q01) & (plot_data[col] <= q99)]
        
        # Sample for performance if too large
        if len(plot_data) > 10000:
            plot_data = plot_data.sample(10000)
        
        # Use solar radiation if available, otherwise use a derived metric
        if 'sw_in' in plot_data.columns:
            z_var = 'sw_in'
            z_label = 'Solar Radiation (W/m²)'
        else:
            # Create a synthetic third dimension using temperature * VPD
            plot_data['temp_vpd'] = plot_data['ta'] * plot_data['vpd']
            z_var = 'temp_vpd'
            z_label = 'Temperature × VPD'
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=plot_data['ta'],
            y=plot_data['vpd'],
            z=plot_data[z_var],
            mode='markers',
            marker=dict(
                size=3,
                color=plot_data['sap_flow'],
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Sap Flow Rate"),
                line=dict(width=0.5, color='black')
            ),
            text=[f"Temp: {temp:.1f}°C<br>VPD: {vpd:.2f} kPa<br>Sap Flow: {sf:.3f}"
                  for temp, vpd, sf in zip(plot_data['ta'], plot_data['vpd'], plot_data['sap_flow'])],
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))
        
        fig.update_layout(
            title='3D Environmental Space - Sap Flow Response',
            scene=dict(
                xaxis_title='Air Temperature (°C)',
                yaxis_title='Vapor Pressure Deficit (kPa)',
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_environmental_space_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_3d_temporal_surface(self, data):
        """Create 3D temporal surface showing sap flow patterns over time"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly required for 3D plots")
            return
            
        print("\n⏰ Creating 3D Temporal Surface plot...")
        
        if 'TIMESTAMP' not in data.columns:
            print("❌ TIMESTAMP column required")
            return
        
        # Extract time components
        data['hour'] = data['TIMESTAMP'].dt.hour
        data['day_of_year'] = data['TIMESTAMP'].dt.dayofyear
        
        # Create hourly and daily aggregations
        temporal_grid = data.groupby(['day_of_year', 'hour'])['sap_flow'].mean().reset_index()
        
        # Sample days for better visualization (every 5th day)
        sample_days = sorted(temporal_grid['day_of_year'].unique())[::5]
        temporal_grid = temporal_grid[temporal_grid['day_of_year'].isin(sample_days)]
        
        # Create pivot table for surface
        surface_data = temporal_grid.pivot(index='day_of_year', columns='hour', values='sap_flow')
        
        # Create 3D surface plot
        fig = go.Figure(data=go.Surface(
            z=surface_data.values,
            x=surface_data.columns,  # Hours
            y=surface_data.index,    # Days
            colorscale='Viridis',
            colorbar=dict(title="Sap Flow Rate")
        ))
        
        fig.update_layout(
            title='3D Temporal Surface - Daily and Seasonal Patterns',
            scene=dict(
                xaxis_title='Hour of Day',
                yaxis_title='Day of Year',
                zaxis_title='Sap Flow Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_temporal_surface_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_3d_site_comparison(self, data):
        """Create 3D plot comparing sites in environmental space"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly required for 3D plots")
            return
            
        print("\n🌍 Creating 3D Site Comparison plot...")
        
        # Get sites with sufficient data
        site_counts = data['site'].value_counts()
        good_sites = site_counts[site_counts > 500].index[:8]  # Top 8 sites
        
        if len(good_sites) < 3:
            print("❌ Need at least 3 sites with sufficient data")
            return
        
        # Filter data for good sites
        site_data = data[data['site'].isin(good_sites)]
        
        # Aggregate by site
        site_summary = site_data.groupby('site').agg({
            'ta': 'mean',
            'vpd': 'mean', 
            'sap_flow': 'mean',
            'TIMESTAMP': 'count'  # Number of records
        }).reset_index()
        site_summary.rename(columns={'TIMESTAMP': 'record_count'}, inplace=True)
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=site_summary['ta'],
            y=site_summary['vpd'],
            z=site_summary['sap_flow'],
            mode='markers+text',
            marker=dict(
                size=site_summary['record_count'] / 100,  # Size by data availability
                color=site_summary['sap_flow'],
                colorscale='RdYlBu_r',
                opacity=0.8,
                colorbar=dict(title="Mean Sap Flow"),
                line=dict(width=2, color='black')
            ),
            text=site_summary['site'],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>" +
                         "Temperature: %{x:.1f}°C<br>" +
                         "VPD: %{y:.2f} kPa<br>" +
                         "Sap Flow: %{z:.3f}<br>" +
                         "Records: %{marker.size:.0f}k<extra></extra>"
        ))
        
        fig.update_layout(
            title='3D Site Comparison - Environmental Conditions vs Sap Flow',
            scene=dict(
                xaxis_title='Mean Temperature (°C)',
                yaxis_title='Mean VPD (kPa)',
                zaxis_title='Mean Sap Flow Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_site_comparison_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_3d_response_surface(self, data):
        """Create 3D response surface for sap flow vs temperature and VPD"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly required for 3D plots")
            return
            
        print("\n📈 Creating 3D Response Surface plot...")
        
        required_cols = ['ta', 'vpd', 'sap_flow']
        if not all(col in data.columns for col in required_cols):
            print(f"❌ Missing required columns: {required_cols}")
            return
        
        # Filter and clean data
        clean_data = data[required_cols].dropna()
        
        # Remove extreme outliers
        for col in required_cols:
            q99 = clean_data[col].quantile(0.99)
            q01 = clean_data[col].quantile(0.01)
            clean_data = clean_data[(clean_data[col] >= q01) & (clean_data[col] <= q99)]
        
        # Create bins for temperature and VPD
        temp_bins = pd.cut(clean_data['ta'], bins=15)
        vpd_bins = pd.cut(clean_data['vpd'], bins=15)
        
        # Calculate mean sap flow for each bin combination
        binned_data = clean_data.groupby([temp_bins, vpd_bins])['sap_flow'].mean().reset_index()
        
        # Extract bin centers
        binned_data['temp_center'] = binned_data['ta'].apply(lambda x: x.mid)
        binned_data['vpd_center'] = binned_data['vpd'].apply(lambda x: x.mid)
        
        # Create pivot table for surface
        surface_data = binned_data.pivot(index='temp_center', columns='vpd_center', values='sap_flow')
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add surface
        fig.add_trace(go.Surface(
            z=surface_data.values,
            x=surface_data.columns,
            y=surface_data.index,
            colorscale='Viridis',
            name='Response Surface',
            opacity=0.8,
            colorbar=dict(title="Sap Flow Rate")
        ))
        
        # Add scatter points (sample)
        sample_data = clean_data.sample(min(1000, len(clean_data)))
        fig.add_trace(go.Scatter3d(
            x=sample_data['vpd'],
            y=sample_data['ta'],
            z=sample_data['sap_flow'],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.6
            ),
            name='Data Points'
        ))
        
        fig.update_layout(
            title='3D Response Surface - Sap Flow vs Temperature and VPD',
            scene=dict(
                xaxis_title='Vapor Pressure Deficit (kPa)',
                yaxis_title='Air Temperature (°C)',
                zaxis_title='Sap Flow Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_response_surface_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_all_3d_plots(self, data):
        """Create all 3D visualizations"""
        if not PLOTLY_AVAILABLE:
            print("❌ Plotly not available - skipping 3D plots")
            print("💡 Install plotly with: pip install plotly")
            return
        
        print("\n🎯 Creating 3D Visualizations...")
        print("=" * 40)
        
        try:
            self.create_3d_environmental_space(data)
            self.create_3d_temporal_surface(data)
            self.create_3d_site_comparison(data)
            self.create_3d_response_surface(data)
            
            print(f"\n✅ All 3D plots completed!")
            
        except Exception as e:
            print(f"❌ Error creating 3D plots: {e}")
            raise

    def create_model_vs_observed_plots(self, data):
        """Create model vs observed sap flow comparison plots"""
        if not self.models or not XGBOOST_AVAILABLE:
            print("❌ No trained models available for comparison")
            return
            
        print("\n🤖 Creating model vs observed comparison plots...")
        
        # Get sites with sufficient data and known clusters
        site_counts = data['site'].value_counts()
        good_sites = []
        
        for site in site_counts[site_counts > 1000].index[:6]:  # Top 6 sites
            if site in self.cluster_assignments:
                good_sites.append(site)
        
        if len(good_sites) < 2:
            print("❌ Not enough sites with cluster assignments")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model vs Observed Sap Flow Comparison', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, site in enumerate(good_sites[:6]):
            if idx >= 6:
                break
                
            site_data = data[data['site'] == site].copy()
            
            # Make predictions
            predictions = self.predict_sap_flow(site_data, site)
            
            if predictions is not None:
                # Sample data for visualization
                if len(site_data) > 2000:
                    sample_indices = np.random.choice(len(site_data), 2000, replace=False)
                    site_data_sample = site_data.iloc[sample_indices]
                    predictions_sample = predictions[sample_indices]
                else:
                    site_data_sample = site_data
                    predictions_sample = predictions
                
                observed = site_data_sample['sap_flow']
                
                # Create scatter plot
                axes[idx].scatter(observed, predictions_sample, alpha=0.5, s=10, color='blue')
                
                # Add 1:1 line
                min_val = min(observed.min(), predictions_sample.min())
                max_val = max(observed.max(), predictions_sample.max())
                axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # Calculate R²
                r2 = np.corrcoef(observed, predictions_sample)[0, 1] ** 2
                axes[idx].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[idx].transAxes, 
                              fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                # Labels and title
                cluster_id = self.cluster_assignments[site]
                axes[idx].set_xlabel('Observed Sap Flow')
                axes[idx].set_ylabel('Predicted Sap Flow')
                axes[idx].set_title(f'{site} (Cluster {cluster_id})')
                axes[idx].grid(True, alpha=0.3)
            else:
                axes[idx].text(0.5, 0.5, 'No predictions\navailable', 
                              transform=axes[idx].transAxes, ha='center', va='center')
                axes[idx].set_title(f'{site} (No Model)')
        
        # Hide unused subplots
        for idx in range(len(good_sites), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'model_vs_observed_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")

    def create_prediction_time_series(self, data):
        """Create time series plots showing predictions vs observations"""
        if not self.models or not XGBOOST_AVAILABLE:
            print("❌ No trained models available for time series")
            return
            
        print("\n📈 Creating prediction time series plots...")
        
        # Get a representative site
        site_counts = data['site'].value_counts()
        target_site = None
        
        for site in site_counts[site_counts > 2000].index:
            if site in self.cluster_assignments:
                target_site = site
                break
        
        if not target_site:
            print("❌ No suitable site found for time series")
            return
        
        site_data = data[data['site'] == target_site].copy()
        
        # Sort by timestamp
        if 'TIMESTAMP' in site_data.columns:
            site_data = site_data.sort_values('TIMESTAMP')
        
        # Make predictions
        predictions = self.predict_sap_flow(site_data, target_site)
        
        if predictions is None:
            print("❌ Could not generate predictions")
            return
        
        # Sample a week of data for visualization
        if len(site_data) > 168:  # More than a week of hourly data
            start_idx = len(site_data) // 2  # Start from middle
            end_idx = start_idx + 168  # One week
            site_data_sample = site_data.iloc[start_idx:end_idx]
            predictions_sample = predictions[start_idx:end_idx]
        else:
            site_data_sample = site_data
            predictions_sample = predictions
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Prediction Time Series - {target_site}', fontsize=16, fontweight='bold')
        
        # Time series comparison
        if 'TIMESTAMP' in site_data_sample.columns:
            x_axis = site_data_sample['TIMESTAMP']
        else:
            x_axis = range(len(site_data_sample))
        
        ax1.plot(x_axis, site_data_sample['sap_flow'], 'b-', label='Observed', alpha=0.8, linewidth=1.5)
        ax1.plot(x_axis, predictions_sample, 'r-', label='Predicted', alpha=0.8, linewidth=1.5)
        ax1.set_ylabel('Sap Flow Rate')
        ax1.set_title('Time Series Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = site_data_sample['sap_flow'] - predictions_sample
        ax2.plot(x_axis, residuals, 'g-', alpha=0.6, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time' if 'TIMESTAMP' in site_data_sample.columns else 'Sample')
        ax2.set_ylabel('Residuals (Obs - Pred)')
        ax2.set_title('Prediction Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'prediction_time_series_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")

    def create_3d_model_predictions(self, data):
        """Create 3D plot showing model predictions in environmental space"""
        if not PLOTLY_AVAILABLE or not self.models or not XGBOOST_AVAILABLE:
            print("❌ Plotly or models not available for 3D predictions")
            return
            
        print("\n🌌 Creating 3D model predictions plot...")
        
        # Get sites with cluster assignments
        sites_with_clusters = [site for site in data['site'].unique() 
                              if site in self.cluster_assignments][:5]  # Top 5 sites
        
        if len(sites_with_clusters) < 3:
            print("❌ Not enough sites with cluster assignments")
            return
        
        plot_data_list = []
        
        for site in sites_with_clusters:
            site_data = data[data['site'] == site].copy()
            
            # Sample data for performance
            if len(site_data) > 500:
                site_data = site_data.sample(500)
            
            # Make predictions
            predictions = self.predict_sap_flow(site_data, site)
            
            if predictions is not None:
                site_data['predicted_sap_flow'] = predictions
                site_data['prediction_error'] = site_data['sap_flow'] - predictions
                plot_data_list.append(site_data)
        
        if not plot_data_list:
            print("❌ No prediction data available")
            return
        
        plot_data = pd.concat(plot_data_list, ignore_index=True)
        
        # Filter for required columns
        required_cols = ['ta', 'vpd', 'predicted_sap_flow']
        if not all(col in plot_data.columns for col in required_cols):
            print(f"❌ Missing required columns for 3D plot")
            return
        
        # Remove outliers
        for col in required_cols:
            q99 = plot_data[col].quantile(0.99)
            q01 = plot_data[col].quantile(0.01)
            plot_data = plot_data[(plot_data[col] >= q01) & (plot_data[col] <= q99)]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=plot_data['ta'],
            y=plot_data['vpd'],
            z=plot_data['predicted_sap_flow'],
            mode='markers',
            marker=dict(
                size=4,
                color=plot_data['prediction_error'],
                colorscale='RdBu',
                opacity=0.7,
                colorbar=dict(title="Prediction Error"),
                line=dict(width=0.5, color='black')
            ),
            text=[f"Site: {site}<br>Temp: {temp:.1f}°C<br>VPD: {vpd:.2f} kPa<br>Predicted: {pred:.3f}<br>Error: {err:.3f}"
                  for site, temp, vpd, pred, err in zip(plot_data['site'], plot_data['ta'], 
                                                       plot_data['vpd'], plot_data['predicted_sap_flow'],
                                                       plot_data['prediction_error'])],
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))
        
        fig.update_layout(
            title='3D Model Predictions - Environmental Space',
            scene=dict(
                xaxis_title='Air Temperature (°C)',
                yaxis_title='Vapor Pressure Deficit (kPa)',
                zaxis_title='Predicted Sap Flow Rate',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_model_predictions_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_model_feature_importance_3d(self):
        """Create 3D visualization of feature importance across clusters"""
        if not PLOTLY_AVAILABLE or not self.models:
            print("❌ Plotly or models not available for feature importance")
            return
            
        print("\n📊 Creating 3D feature importance visualization...")
        
        # Load feature importance data
        importance_files = glob.glob(os.path.join(self.models_dir, 'feature_importance_cluster_*.csv'))
        
        if len(importance_files) < 2:
            print("❌ Need at least 2 clusters for 3D comparison")
            return
        
        importance_data = []
        
        for file_path in importance_files:
            try:
                # Extract cluster ID
                filename = os.path.basename(file_path)
                cluster_id = None
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    if part == 'cluster' and i + 1 < len(parts):
                        try:
                            cluster_id = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                
                if cluster_id is not None:
                    df = pd.read_csv(file_path)
                    df['cluster'] = cluster_id
                    importance_data.append(df)
                    
            except Exception as e:
                print(f"  ⚠️  Error loading {filename}: {e}")
        
        if len(importance_data) < 2:
            print("❌ Could not load enough importance data")
            return
        
        # Combine all importance data
        all_importance = pd.concat(importance_data, ignore_index=True)
        
        # Get top features across all clusters
        top_features = (all_importance.groupby('feature_name')['importance']
                       .mean().sort_values(ascending=False).head(20).index.tolist())
        
        # Filter to top features
        plot_data = all_importance[all_importance['feature_name'].isin(top_features)]
        
        # Create 3D surface/scatter
        fig = go.Figure()
        
        # Add scatter points for each cluster
        clusters = sorted(plot_data['cluster'].unique())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, cluster in enumerate(clusters):
            cluster_data = plot_data[plot_data['cluster'] == cluster]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_data['cluster'],
                y=list(range(len(cluster_data))),
                z=cluster_data['importance'],
                mode='markers+text',
                marker=dict(
                    size=6,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                text=cluster_data['feature_name'],
                textposition="middle right",
                name=f'Cluster {cluster}',
                hovertemplate="<b>%{text}</b><br>" +
                             "Cluster: %{x}<br>" +
                             "Importance: %{z:.0f}<extra></extra>"
            ))
        
        fig.update_layout(
            title='3D Feature Importance Across Clusters',
            scene=dict(
                xaxis_title='Ecosystem Cluster',
                yaxis_title='Feature Rank',
                zaxis_title='Feature Importance',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        # Save interactive plot
        output_path = os.path.join(self.output_dir, f'3d_feature_importance_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"  ✅ Saved: {output_path}")
        
        return fig

    def create_all_model_visualizations(self, data):
        """Create all model-integrated visualizations"""
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available - skipping model visualizations")
            return
            
        if not self.models:
            print("❌ No trained models available - skipping model visualizations")
            print("💡 This could be due to XGBoost version incompatibility")
            print("💡 Consider updating XGBoost: pip install --upgrade xgboost")
            return
        
        print("\n🤖 Creating Model-Integrated Visualizations...")
        print("=" * 50)
        
        try:
            self.create_model_vs_observed_plots(data)
            self.create_prediction_time_series(data)
            
            if PLOTLY_AVAILABLE:
                self.create_3d_model_predictions(data)
                self.create_model_feature_importance_3d()
            
            print(f"\n✅ All model visualizations completed!")
            
        except Exception as e:
            print(f"❌ Error creating model visualizations: {e}")
            raise

    def run_all_visualizations(self):
        """Run all sap flow visualizations"""
        print("🌊 SAPFLUXNET Sap Flow Comprehensive Analysis")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        
        # Load data
        data = self.load_sample_sites(n_sites=15)  # Load 15 sites for analysis
        
        if data is None:
            print("❌ No data loaded, exiting...")
            return
        
        # Filter out invalid sap flow values
        initial_len = len(data)
        data = data[
            (data['sap_flow'].notna()) & 
            (data['sap_flow'] >= 0) & 
            (data['sap_flow'] < data['sap_flow'].quantile(0.999))  # Remove extreme outliers
        ]
        print(f"🧹 Filtered data: {len(data):,} records ({initial_len - len(data):,} removed)")
        
        # Create all visualizations
        try:
            self.create_summary_statistics(data)
            self.create_daily_patterns(data)
            self.create_environmental_responses(data)
            self.create_seasonal_analysis(data)
            self.create_site_comparison(data)
            self.create_all_3d_plots(data)  # Add 3D visualizations
            self.create_all_model_visualizations(data) # Add model-integrated visualizations
            
            print(f"\n✅ All visualizations completed successfully!")
            print(f"📁 Check output directory: {self.output_dir}")
            
            # List generated files
            output_files = glob.glob(os.path.join(self.output_dir, f'*{self.timestamp}*'))
            print(f"\n📄 Generated {len(output_files)} files:")
            for file in sorted(output_files):
                file_type = "📊 2D Plot" if file.endswith('.png') else "🌐 3D Interactive" if file.endswith('.html') else "📋 Summary"
                print(f"  - {file_type}: {os.path.basename(file)}")
                
        except Exception as e:
            print(f"❌ Error in visualization creation: {e}")
            raise

def main():
    """Main function"""
    print("🌊 SAPFLUXNET Sap Flow Visualization Suite")
    print("=" * 50)
    
    # Check for data directory
    data_dirs = ['processed_parquet', '../processed_parquet', 'ecosystem/processed_parquet']
    data_dir = None
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path) and glob.glob(os.path.join(dir_path, "*.parquet")):
            data_dir = dir_path
            break
    
    if not data_dir:
        print("❌ No parquet data directory found!")
        print("Expected locations: processed_parquet/, ../processed_parquet/, ecosystem/processed_parquet/")
        return
    
    # Initialize and run visualizer
    visualizer = SapFlowVisualizer(data_dir=data_dir)
    visualizer.run_all_visualizations()
    
    print(f"\n🎉 Sap flow analysis complete!")
    print(f"💡 Generated visualizations show:")
    print(f"   - Daily transpiration patterns")
    print(f"   - Environmental response relationships")
    print(f"   - Seasonal variation analysis")
    print(f"   - Site-to-site comparisons")

if __name__ == "__main__":
    main() 