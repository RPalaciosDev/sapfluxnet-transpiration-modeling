"""
Comprehensive Results Visualization for SAPFLUXNET Ecosystem Models

This script creates visualizations for:
1. Clustering results - cluster composition, geographic distribution, feature separation
2. Model training performance - training metrics, feature importance across clusters  
3. Spatial validation results - within-cluster generalization, outlier identification

Generates publication-ready figures summarizing the entire ecosystem modeling pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for publication-ready figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ComprehensiveResultsVisualizer:
    """
    Creates comprehensive visualizations of clustering, training, and validation results
    """
    
    def __init__(self, 
                 clustering_results_dir='../evaluation/clustering_results',
                 model_results_dir='../models/results',
                 output_dir='./visualization_outputs'):
        
        self.clustering_results_dir = clustering_results_dir
        self.model_results_dir = model_results_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Comprehensive Results Visualizer initialized")
        print(f"📁 Output directory: {output_dir}")
        
    def load_all_results(self):
        """Load all available results data"""
        print("\n📂 Loading all results data...")
        
        results = {}
        
        # 1. Load clustering results
        try:
            cluster_files = sorted(glob.glob(os.path.join(self.clustering_results_dir, 'advanced_site_clusters_*.csv')))
            if cluster_files:
                latest_cluster_file = cluster_files[-1]
                results['clustering'] = pd.read_csv(latest_cluster_file)
                print(f"✅ Loaded clustering data: {os.path.basename(latest_cluster_file)}")
            
            # Load ecosystem interpretations
            interpretation_files = sorted(glob.glob(os.path.join(self.clustering_results_dir, 'advanced_ecosystem_interpretations_*.json')))
            if interpretation_files:
                latest_interpretation_file = interpretation_files[-1]
                with open(latest_interpretation_file, 'r') as f:
                    results['ecosystem_interpretations'] = json.load(f)
                print(f"✅ Loaded ecosystem interpretations: {os.path.basename(latest_interpretation_file)}")
                
        except Exception as e:
            print(f"⚠️  Could not load clustering results: {e}")
        
        # 2. Load model training results
        try:
            model_metrics_files = sorted(glob.glob(os.path.join(self.model_results_dir, 'cluster_models/cluster_model_metrics_*.csv')))
            if model_metrics_files:
                latest_metrics_file = model_metrics_files[-1]
                results['model_metrics'] = pd.read_csv(latest_metrics_file)
                print(f"✅ Loaded model metrics: {os.path.basename(latest_metrics_file)}")
            
            # Load feature importance for each cluster
            feature_importance_files = glob.glob(os.path.join(self.model_results_dir, 'cluster_models/feature_importance_cluster_*.csv'))
            results['feature_importance'] = {}
            for file in feature_importance_files:
                cluster_id = int(file.split('_cluster_')[1].split('_')[0])
                results['feature_importance'][cluster_id] = pd.read_csv(file)
            print(f"✅ Loaded feature importance for {len(results['feature_importance'])} clusters")
            
        except Exception as e:
            print(f"⚠️  Could not load model training results: {e}")
        
        # 3. Load spatial validation results
        try:
            spatial_results_files = sorted(glob.glob(os.path.join(self.model_results_dir, 'cluster_spatial_validation/cluster_spatial_fold_results_*.csv')))
            if spatial_results_files:
                latest_spatial_file = spatial_results_files[-1]
                results['spatial_validation'] = pd.read_csv(latest_spatial_file)
                print(f"✅ Loaded spatial validation: {os.path.basename(latest_spatial_file)}")
            
            # Load spatial validation summary
            spatial_summary_files = sorted(glob.glob(os.path.join(self.model_results_dir, 'cluster_spatial_validation/cluster_spatial_summaries_*.csv')))
            if spatial_summary_files:
                latest_summary_file = spatial_summary_files[-1]
                results['spatial_summary'] = pd.read_csv(latest_summary_file)
                print(f"✅ Loaded spatial validation summary: {os.path.basename(latest_summary_file)}")
                
        except Exception as e:
            print(f"⚠️  Could not load spatial validation results: {e}")
        
        return results
    
    def create_clustering_visualizations(self, results):
        """Create visualizations for clustering results"""
        print("\n🎨 Creating clustering visualizations...")
        
        if 'clustering' not in results:
            print("⚠️  No clustering data available")
            return
        
        clustering_df = results['clustering']
        
        # 1. Cluster composition pie chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SAPFLUXNET Ecosystem Clustering Analysis', fontsize=16, fontweight='bold')
        
        # Cluster size distribution
        cluster_counts = clustering_df['cluster'].value_counts().sort_index()
        colors = sns.color_palette("husl", len(cluster_counts))
        
        axes[0,0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,0].set_title('Cluster Size Distribution\n(Number of Sites)')
        
        # 2. Geographic distribution if we have lat/lon
        if 'latitude' in clustering_df.columns and 'longitude' in clustering_df.columns:
            scatter = axes[0,1].scatter(clustering_df['longitude'], clustering_df['latitude'], 
                                      c=clustering_df['cluster'], cmap='tab10', s=50, alpha=0.7)
            axes[0,1].set_xlabel('Longitude')
            axes[0,1].set_ylabel('Latitude')
            axes[0,1].set_title('Geographic Distribution of Clusters')
            plt.colorbar(scatter, ax=axes[0,1], label='Cluster ID')
        else:
            axes[0,1].text(0.5, 0.5, 'Geographic data\nnot available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Geographic Distribution')
        
        # 3. Climate characteristics by cluster
        if 'mean_annual_temp' in clustering_df.columns and 'mean_annual_precip' in clustering_df.columns:
            for cluster_id in sorted(clustering_df['cluster'].unique()):
                cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
                axes[1,0].scatter(cluster_data['mean_annual_temp'], cluster_data['mean_annual_precip'], 
                                label=f'Cluster {cluster_id}', alpha=0.7, s=50)
            axes[1,0].set_xlabel('Mean Annual Temperature (°C)')
            axes[1,0].set_ylabel('Mean Annual Precipitation (mm)')
            axes[1,0].set_title('Climate Space Distribution')
            axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'Climate data\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Climate Space Distribution')
        
        # 4. Cluster balance metrics
        if 'ecosystem_interpretations' in results:
            interpretations = results['ecosystem_interpretations']
            cluster_names = []
            cluster_sizes = []
            
            for cluster_id, info in interpretations.items():
                cluster_names.append(f"C{cluster_id}: {info['ecosystem_type'][:15]}")
                cluster_sizes.append(info['n_sites'])
            
            axes[1,1].barh(cluster_names, cluster_sizes, color=colors[:len(cluster_names)])
            axes[1,1].set_xlabel('Number of Sites')
            axes[1,1].set_title('Cluster Sizes with Ecosystem Types')
        else:
            axes[1,1].bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
            axes[1,1].set_xlabel('Cluster ID')
            axes[1,1].set_ylabel('Number of Sites')
            axes[1,1].set_title('Cluster Sizes')
            axes[1,1].set_xticks(range(len(cluster_counts)))
        
        plt.tight_layout()
        clustering_fig_path = os.path.join(self.output_dir, f'clustering_analysis_{self.timestamp}.png')
        plt.savefig(clustering_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved clustering analysis: {clustering_fig_path}")
        
    def create_model_performance_visualizations(self, results):
        """Create visualizations for model training performance"""
        print("\n🎨 Creating model performance visualizations...")
        
        if 'model_metrics' not in results:
            print("⚠️  No model metrics data available")
            return
        
        model_metrics = results['model_metrics']
        
        # Main performance comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SAPFLUXNET Cluster Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training vs Test R² by cluster
        x_pos = np.arange(len(model_metrics))
        width = 0.35
        
        axes[0,0].bar(x_pos - width/2, model_metrics['train_r2'], width, 
                     label='Training R²', alpha=0.8, color='skyblue')
        axes[0,0].bar(x_pos + width/2, model_metrics['test_r2'], width, 
                     label='Test R²', alpha=0.8, color='lightcoral')
        
        axes[0,0].set_xlabel('Cluster ID')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Training vs Test Performance by Cluster')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([f'C{i}' for i in model_metrics['cluster']])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Add performance labels
        for i, (train_r2, test_r2) in enumerate(zip(model_metrics['train_r2'], model_metrics['test_r2'])):
            axes[0,0].text(i, max(train_r2, test_r2) + 0.01, f'{test_r2:.3f}', 
                          ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE by cluster
        axes[0,1].bar(x_pos, model_metrics['test_rmse'], color='lightgreen', alpha=0.8)
        axes[0,1].set_xlabel('Cluster ID')
        axes[0,1].set_ylabel('Test RMSE')
        axes[0,1].set_title('Test RMSE by Cluster')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f'C{i}' for i in model_metrics['cluster']])
        axes[0,1].grid(True, alpha=0.3)
        
        # Add RMSE labels
        for i, rmse in enumerate(model_metrics['test_rmse']):
            axes[0,1].text(i, rmse + 0.05, f'{rmse:.2f}', ha='center', va='bottom')
        
        # 3. Dataset size vs performance
        if 'total_samples' in model_metrics.columns:
            scatter = axes[1,0].scatter(model_metrics['total_samples'], model_metrics['test_r2'], 
                                      c=model_metrics['cluster'], cmap='tab10', s=100, alpha=0.7)
            axes[1,0].set_xlabel('Total Training Samples')
            axes[1,0].set_ylabel('Test R²')
            axes[1,0].set_title('Dataset Size vs Performance')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add cluster labels
            for i, row in model_metrics.iterrows():
                axes[1,0].annotate(f'C{row["cluster"]}', 
                                 (row['total_samples'], row['test_r2']), 
                                 xytext=(5, 5), textcoords='offset points')
        else:
            axes[1,0].text(0.5, 0.5, 'Sample size data\nnot available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Dataset Size vs Performance')
        
        # 4. Performance summary statistics
        axes[1,1].axis('off')
        summary_text = f"""Performance Summary:
        
Average Test R²: {model_metrics['test_r2'].mean():.4f} ± {model_metrics['test_r2'].std():.4f}
Best Cluster: C{model_metrics.loc[model_metrics['test_r2'].idxmax(), 'cluster']} (R² = {model_metrics['test_r2'].max():.4f})
Worst Cluster: C{model_metrics.loc[model_metrics['test_r2'].idxmin(), 'cluster']} (R² = {model_metrics['test_r2'].min():.4f})

Average Test RMSE: {model_metrics['test_rmse'].mean():.4f} ± {model_metrics['test_rmse'].std():.4f}
        
Total Clusters: {len(model_metrics)}
Performance Range: {model_metrics['test_r2'].max() - model_metrics['test_r2'].min():.4f}"""
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=11, verticalalignment='top', 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[1,1].set_title('Performance Summary')
        
        plt.tight_layout()
        model_perf_path = os.path.join(self.output_dir, f'model_performance_{self.timestamp}.png')
        plt.savefig(model_perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved model performance: {model_perf_path}")
        
    def create_feature_importance_visualizations(self, results):
        """Create visualizations for feature importance across clusters"""
        print("\n🎨 Creating feature importance visualizations...")
        
        if 'feature_importance' not in results:
            print("⚠️  No feature importance data available")
            return
        
        feature_importance = results['feature_importance']
        
        # Create comprehensive feature importance comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Importance Analysis Across Clusters', fontsize=16, fontweight='bold')
        
        # Individual cluster feature importance (top 10 for each)
        cluster_ids = sorted(feature_importance.keys())[:5]  # Show up to 5 clusters
        
        for i, cluster_id in enumerate(cluster_ids):
            if i >= 5:  # Max 5 clusters to fit in subplot
                break
                
            row = i // 3
            col = i % 3
            
            if row < 2 and col < 3:  # Make sure we don't exceed subplot grid
                fi_data = feature_importance[cluster_id].head(10)
                
                y_pos = np.arange(len(fi_data))
                axes[row, col].barh(y_pos, fi_data['importance'], alpha=0.8)
                axes[row, col].set_yticks(y_pos)
                axes[row, col].set_yticklabels(fi_data['feature_name'], fontsize=8)
                axes[row, col].set_xlabel('Importance')
                axes[row, col].set_title(f'Cluster {cluster_id} - Top Features')
                axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(cluster_ids), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        feature_imp_path = os.path.join(self.output_dir, f'feature_importance_by_cluster_{self.timestamp}.png')
        plt.savefig(feature_imp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved feature importance by cluster: {feature_imp_path}")
        
        # Universal features analysis
        self.create_universal_features_analysis(feature_importance)
    
    def create_universal_features_analysis(self, feature_importance):
        """Analyze features that are important across multiple clusters"""
        print("  🔍 Analyzing universal features...")
        
        # Collect all features and their importance across clusters
        all_features = {}
        
        for cluster_id, fi_df in feature_importance.items():
            for _, row in fi_df.iterrows():
                feature = row['feature_name']
                importance = row['importance']
                
                if feature not in all_features:
                    all_features[feature] = {'clusters': [], 'importances': [], 'total_importance': 0}
                
                all_features[feature]['clusters'].append(cluster_id)
                all_features[feature]['importances'].append(importance)
                all_features[feature]['total_importance'] += importance
        
        # Find universal features (present in multiple clusters)
        universal_features = []
        for feature, data in all_features.items():
            if len(data['clusters']) >= 3:  # Present in at least 3 clusters
                        universal_features.append({
                'feature_name': feature,
                'n_clusters': len(data['clusters']),
                    'avg_importance': np.mean(data['importances']),
                    'total_importance': data['total_importance'],
                    'clusters': data['clusters']
                })
        
        # Sort by number of clusters and average importance
        universal_features.sort(key=lambda x: (x['n_clusters'], x['avg_importance']), reverse=True)
        
        if universal_features:
            # Create universal features visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Universal Features: Important Across Multiple Clusters', fontsize=14, fontweight='bold')
            
            # Top universal features by cluster count
            top_universal = universal_features[:15]
            feature_names = [f['feature_name'][:20] + '...' if len(f['feature_name']) > 20 else f['feature_name'] for f in top_universal]
            cluster_counts = [f['n_clusters'] for f in top_universal]
            avg_importances = [f['avg_importance'] for f in top_universal]
            
            y_pos = np.arange(len(feature_names))
            
            # Subplot 1: Number of clusters each feature appears in
            bars1 = ax1.barh(y_pos, cluster_counts, alpha=0.8, color='lightblue')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(feature_names, fontsize=9)
            ax1.set_xlabel('Number of Clusters')
            ax1.set_title('Feature Universality\n(Number of Clusters)')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for i, count in enumerate(cluster_counts):
                ax1.text(count + 0.05, i, str(count), va='center', fontweight='bold')
            
            # Subplot 2: Average importance across clusters
            bars2 = ax2.barh(y_pos, avg_importances, alpha=0.8, color='lightcoral')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names, fontsize=9)
            ax2.set_xlabel('Average Importance')
            ax2.set_title('Average Feature Importance\n(Across Clusters)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            universal_features_path = os.path.join(self.output_dir, f'universal_features_{self.timestamp}.png')
            plt.savefig(universal_features_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved universal features analysis: {universal_features_path}")
    
    def create_spatial_validation_visualizations(self, results):
        """Create visualizations for spatial validation results"""
        print("\n🎨 Creating spatial validation visualizations...")
        
        if 'spatial_validation' not in results:
            print("⚠️  No spatial validation data available")
            return
        
        spatial_df = results['spatial_validation']
        
        # Main spatial validation analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SAPFLUXNET Spatial Validation Analysis (Within-Cluster LOSO)', fontsize=16, fontweight='bold')
        
        # 1. R² distribution by cluster
        cluster_ids = sorted(spatial_df['cluster'].unique())
        r2_by_cluster = []
        cluster_labels = []
        
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            r2_by_cluster.append(cluster_data['test_r2'].values)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        bp1 = axes[0,0].boxplot(r2_by_cluster, labels=cluster_labels, patch_artist=True)
        axes[0,0].set_ylabel('Test R²')
        axes[0,0].set_title('Spatial Validation R² by Cluster')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R² = 0')
        
        # Color the boxplots
        colors = sns.color_palette("husl", len(cluster_ids))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 2. RMSE distribution by cluster
        rmse_by_cluster = []
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            rmse_by_cluster.append(cluster_data['test_rmse'].values)
        
        bp2 = axes[0,1].boxplot(rmse_by_cluster, labels=cluster_labels, patch_artist=True)
        axes[0,1].set_ylabel('Test RMSE')
        axes[0,1].set_title('Spatial Validation RMSE by Cluster')
        axes[0,1].grid(True, alpha=0.3)
        
        # Color the boxplots
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 3. Scatter plot: R² vs site (to identify outliers)
        scatter_colors = [colors[cluster_ids.index(cluster)] for cluster in spatial_df['cluster']]
        scatter = axes[1,0].scatter(range(len(spatial_df)), spatial_df['test_r2'], 
                                  c=scatter_colors, alpha=0.7, s=50)
        axes[1,0].set_xlabel('Site Index')
        axes[1,0].set_ylabel('Test R²')
        axes[1,0].set_title('Site-Level Spatial Validation Performance')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Identify and label extreme outliers
        outliers = spatial_df[spatial_df['test_r2'] < -5]  # Very poor performance
        for _, outlier in outliers.iterrows():
            site_idx = spatial_df.index[spatial_df['test_site'] == outlier['test_site']].tolist()[0]
            axes[1,0].annotate(outlier['test_site'][:8], 
                             (site_idx, outlier['test_r2']), 
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8)
        
        # 4. Performance summary statistics
        axes[1,1].axis('off')
        
        # Calculate summary statistics
        overall_mean_r2 = spatial_df['test_r2'].mean()
        overall_std_r2 = spatial_df['test_r2'].std()
        best_site = spatial_df.loc[spatial_df['test_r2'].idxmax()]
        worst_site = spatial_df.loc[spatial_df['test_r2'].idxmin()]
        
        # Cluster-level summaries
        cluster_summaries = []
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            cluster_mean = cluster_data['test_r2'].mean()
            cluster_summaries.append(f"Cluster {cluster_id}: {cluster_mean:.3f}")
        
        summary_text = f"""Spatial Validation Summary:
        
Overall Performance:
  Mean R²: {overall_mean_r2:.4f} ± {overall_std_r2:.4f}
  Total Sites: {len(spatial_df)}
  Successful Folds: {len(spatial_df[spatial_df['test_r2'] > -10])}

Best Site: {best_site['test_site']} (R² = {best_site['test_r2']:.3f})
Worst Site: {worst_site['test_site']} (R² = {worst_site['test_r2']:.3f})

Cluster Performance:
""" + "\n".join(cluster_summaries)
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=10, verticalalignment='top', 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[1,1].set_title('Spatial Validation Summary')
        
        plt.tight_layout()
        spatial_val_path = os.path.join(self.output_dir, f'spatial_validation_{self.timestamp}.png')
        plt.savefig(spatial_val_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved spatial validation analysis: {spatial_val_path}")
        
        # Create detailed outlier analysis
        self.create_outlier_analysis_visualization(spatial_df)
    
    def create_outlier_analysis_visualization(self, spatial_df):
        """Create detailed analysis of outlier sites"""
        print("  🔍 Creating outlier analysis...")
        
        # Identify outliers using multiple criteria
        outliers = spatial_df[
            (spatial_df['test_r2'] < -2) |  # Very poor R²
            (spatial_df['test_rmse'] > spatial_df['test_rmse'].quantile(0.95))  # Very high RMSE
        ].copy()
        
        if len(outliers) == 0:
            print("  ✅ No significant outliers found")
            return
        
        # Create outlier-focused visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Outlier Analysis: {len(outliers)} Problematic Sites Identified', fontsize=14, fontweight='bold')
        
        # 1. Outlier sites by cluster
        outlier_clusters = outliers['cluster'].value_counts().sort_index()
        axes[0,0].bar(outlier_clusters.index, outlier_clusters.values, alpha=0.8, color='red')
        axes[0,0].set_xlabel('Cluster ID')
        axes[0,0].set_ylabel('Number of Outlier Sites')
        axes[0,0].set_title('Outlier Distribution by Cluster')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. R² vs RMSE for outliers
        scatter = axes[0,1].scatter(outliers['test_r2'], outliers['test_rmse'], 
                                  c=outliers['cluster'], cmap='tab10', s=80, alpha=0.7)
        axes[0,1].set_xlabel('Test R²')
        axes[0,1].set_ylabel('Test RMSE')
        axes[0,1].set_title('Outlier Performance Scatter')
        axes[0,1].grid(True, alpha=0.3)
        
        # Annotate worst outliers
        worst_outliers = outliers.nsmallest(3, 'test_r2')
        for _, outlier in worst_outliers.iterrows():
            axes[0,1].annotate(outlier['test_site'][:8], 
                             (outlier['test_r2'], outlier['test_rmse']), 
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8)
        
        # 3. Outlier R² distribution
        axes[1,0].hist(outliers['test_r2'], bins=15, alpha=0.7, color='red', edgecolor='black')
        axes[1,0].set_xlabel('Test R²')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Outlier R² Distribution')
        axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='R² = 0')
        axes[1,0].axvline(x=outliers['test_r2'].mean(), color='blue', linestyle='-', alpha=0.7, label='Mean')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Detailed outlier list
        axes[1,1].axis('off')
        
        # Top 10 worst outliers
        worst_10 = outliers.nsmallest(10, 'test_r2')
        outlier_text = "Top 10 Worst Performing Sites:\n\n"
        
        for i, (_, outlier) in enumerate(worst_10.iterrows(), 1):
            outlier_text += f"{i:2d}. {outlier['test_site']:<15} "
            outlier_text += f"(C{outlier['cluster']}) "
            outlier_text += f"R² = {outlier['test_r2']:6.3f}\n"
        
        if 'COL_MAC_SAF_RAD' in outliers['test_site'].values:
            col_mac_data = outliers[outliers['test_site'] == 'COL_MAC_SAF_RAD'].iloc[0]
            outlier_text += f"\n🚨 COL_MAC_SAF_RAD Analysis:\n"
            outlier_text += f"   Cluster: {col_mac_data['cluster']}\n"
            outlier_text += f"   R²: {col_mac_data['test_r2']:.3f}\n"
            outlier_text += f"   RMSE: {col_mac_data['test_rmse']:.3f}\n"
            outlier_text += f"   Status: EXTREME OUTLIER"
        
        axes[1,1].text(0.05, 0.95, outlier_text, transform=axes[1,1].transAxes, 
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.8))
        axes[1,1].set_title('Worst Performing Sites')
        
        plt.tight_layout()
        outlier_analysis_path = os.path.join(self.output_dir, f'outlier_analysis_{self.timestamp}.png')
        plt.savefig(outlier_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved outlier analysis: {outlier_analysis_path}")
    
    def create_comprehensive_summary(self, results):
        """Create a comprehensive summary dashboard"""
        print("\n🎨 Creating comprehensive summary dashboard...")
        
        # Create a large summary figure
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('SAPFLUXNET Ecosystem Modeling: Comprehensive Results Summary', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Create a complex grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Ecosystem clustering overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'clustering' in results:
            cluster_counts = results['clustering']['cluster'].value_counts().sort_index()
            colors = sns.color_palette("husl", len(cluster_counts))
            ax1.pie(cluster_counts.values, labels=[f'C{i}' for i in cluster_counts.index], 
                   autopct='%1.0f', colors=colors, startangle=90)
            ax1.set_title('Ecosystem Clusters\n(87 Sites)', fontweight='bold')
        
        # 2. Model performance summary (top center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'model_metrics' in results:
            model_metrics = results['model_metrics']
            x_pos = np.arange(len(model_metrics))
            ax2.bar(x_pos, model_metrics['test_r2'], alpha=0.8, color='lightcoral')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Test R²')
            ax2.set_title('Model Performance\nby Cluster', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'C{i}' for i in model_metrics['cluster']])
            ax2.grid(True, alpha=0.3)
        
        # 3. Spatial validation overview (top center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'spatial_validation' in results:
            spatial_df = results['spatial_validation']
            cluster_ids = sorted(spatial_df['cluster'].unique())
            r2_means = []
            for cluster_id in cluster_ids:
                cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
                r2_means.append(cluster_data['test_r2'].mean())
            
            colors = sns.color_palette("husl", len(cluster_ids))
            bars = ax3.bar(range(len(cluster_ids)), r2_means, color=colors, alpha=0.8)
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Mean Spatial R²')
            ax3.set_title('Spatial Validation\nPerformance', fontweight='bold')
            ax3.set_xticks(range(len(cluster_ids)))
            ax3.set_xticklabels([f'C{i}' for i in cluster_ids])
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Key metrics summary (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        summary_stats = []
        if 'model_metrics' in results:
            model_metrics = results['model_metrics']
            summary_stats.append(f"Training Performance:")
            summary_stats.append(f"  Mean R²: {model_metrics['test_r2'].mean():.3f}")
            summary_stats.append(f"  Best: {model_metrics['test_r2'].max():.3f}")
            summary_stats.append(f"  Worst: {model_metrics['test_r2'].min():.3f}")
        
        if 'spatial_validation' in results:
            spatial_df = results['spatial_validation']
            summary_stats.append(f"\nSpatial Validation:")
            summary_stats.append(f"  Mean R²: {spatial_df['test_r2'].mean():.3f}")
            summary_stats.append(f"  Success Rate: {len(spatial_df[spatial_df['test_r2'] > 0]) / len(spatial_df) * 100:.1f}%")
            summary_stats.append(f"  Total Sites: {len(spatial_df)}")
        
        if 'clustering' in results:
            clustering_df = results['clustering']
            summary_stats.append(f"\nClustering:")
            summary_stats.append(f"  Total Sites: {len(clustering_df)}")
            summary_stats.append(f"  Clusters: {clustering_df['cluster'].nunique()}")
            
            if 'ecosystem_interpretations' in results:
                interpretations = results['ecosystem_interpretations']
                summary_stats.append(f"  Ecosystems: {len(interpretations)}")
        
        summary_text = '\n'.join(summary_stats)
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        ax4.set_title('Key Metrics', fontweight='bold')
        
        # 5. Feature importance heatmap (middle section)
        if 'feature_importance' in results:
            ax5 = fig.add_subplot(gs[1, :])
            
            # Create a heatmap of top features across clusters
            feature_importance = results['feature_importance']
            
            # Get top 20 most important features across all clusters
            all_features = {}
            for cluster_id, fi_df in feature_importance.items():
                for _, row in fi_df.head(20).iterrows():
                    feature = row['feature_name']
                    importance = row['importance']
                    if feature not in all_features:
                        all_features[feature] = {}
                    all_features[feature][cluster_id] = importance
            
            # Create matrix for heatmap
            cluster_ids = sorted(feature_importance.keys())
            top_features = sorted(all_features.keys(), 
                                key=lambda x: sum(all_features[x].values()), 
                                reverse=True)[:15]
            
            heatmap_data = []
            for feature in top_features:
                row = []
                for cluster_id in cluster_ids:
                    importance = all_features[feature].get(cluster_id, 0)
                    row.append(importance)
                heatmap_data.append(row)
            
            # Plot heatmap
            heatmap_array = np.array(heatmap_data)
            # Normalize by row for better visualization
            heatmap_normalized = heatmap_array / (heatmap_array.max(axis=1, keepdims=True) + 1e-8)
            
            im = ax5.imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto')
            ax5.set_xticks(range(len(cluster_ids)))
            ax5.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
            ax5.set_yticks(range(len(top_features)))
            ax5.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features], fontsize=9)
            ax5.set_title('Feature Importance Across Clusters (Normalized)', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
            cbar.set_label('Normalized Importance')
        
        # 6. Bottom section: Ecosystem interpretations
        if 'ecosystem_interpretations' in results:
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')
            
            interpretations = results['ecosystem_interpretations']
            ecosystem_text = "Identified Ecosystem Types:\n\n"
            
            for cluster_id, info in interpretations.items():
                ecosystem_text += f"Cluster {cluster_id}: {info['ecosystem_type']} ({info['n_sites']} sites)\n"
                # Show example sites
                example_sites = info['sites'][:3]
                ecosystem_text += f"  Examples: {', '.join(example_sites)}\n"
                
                # Show climate zones if available
                if 'climate_zones' in info and info['climate_zones']:
                    main_climate = max(info['climate_zones'].items(), key=lambda x: x[1])
                    ecosystem_text += f"  Dominant climate: {main_climate[0]}\n"
                
                ecosystem_text += "\n"
            
            ax6.text(0.02, 0.98, ecosystem_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='sans-serif',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.2))
            ax6.set_title('Ecosystem Interpretations', fontweight='bold', pad=20)
        
        # Save the comprehensive summary
        comprehensive_path = os.path.join(self.output_dir, f'comprehensive_summary_{self.timestamp}.png')
        plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved comprehensive summary: {comprehensive_path}")
    
    def run_all_visualizations(self):
        """Run all visualization analyses"""
        print("🎨 SAPFLUXNET Comprehensive Results Visualization")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Create comprehensive visualizations of clustering, training, and validation results")
        
        try:
            # Load all results
            results = self.load_all_results()
            
            if not results:
                print("❌ No results data found")
                return
            
            # Create all visualizations
            print(f"\n📊 Creating visualizations...")
            
            self.create_clustering_visualizations(results)
            self.create_model_performance_visualizations(results)
            self.create_feature_importance_visualizations(results)
            self.create_spatial_validation_visualizations(results)
            self.create_comprehensive_summary(results)
            
            print(f"\n✅ All visualizations completed!")
            print(f"📁 Check output directory: {self.output_dir}")
            
            # List generated files
            output_files = glob.glob(os.path.join(self.output_dir, f'*{self.timestamp}*'))
            print(f"\n📄 Generated {len(output_files)} visualization files:")
            for file in sorted(output_files):
                print(f"  - {os.path.basename(file)}")
                
        except Exception as e:
            print(f"❌ Error in visualization creation: {e}")
            raise

def main():
    visualizer = ComprehensiveResultsVisualizer()
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main() 