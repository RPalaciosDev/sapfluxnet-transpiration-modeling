"""
Matplotlib-Only Results Visualization for SAPFLUXNET Ecosystem Models

This script creates comprehensive visualizations using only matplotlib, with no pie charts.
Focuses on analytical chart types: bar plots, scatter plots, box plots, heatmaps, and histograms.

Generates publication-ready figures summarizing clustering, training, and validation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import glob
import os
import json
from datetime import datetime
import warnings
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Set style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class MatplotlibResultsVisualizer:
    """
    Creates comprehensive matplotlib-only visualizations (no pie charts)
    """
    
    def __init__(self, 
                 clustering_results_dir='../evaluation/clustering_results',
                 model_results_dir='../models/results',
                 output_dir='./matplotlib_outputs'):
        
        self.clustering_results_dir = clustering_results_dir
        self.model_results_dir = model_results_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define color scheme
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))
        self.cluster_colors = {i: self.colors[i % len(self.colors)] for i in range(10)}
        
        print(f"📊 Matplotlib Results Visualizer initialized")
        print(f"📁 Output directory: {output_dir}")
        
    def load_all_results(self):
        """Load all available results data"""
        print("\n📂 Loading all results data...")
        
        results = {}
        
        # Load clustering results
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
        
        # Load model training results
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
        
        # Load spatial validation results
        try:
            spatial_results_files = sorted(glob.glob(os.path.join(self.model_results_dir, 'cluster_spatial_validation/cluster_spatial_fold_results_*.csv')))
            if spatial_results_files:
                latest_spatial_file = spatial_results_files[-1]
                results['spatial_validation'] = pd.read_csv(latest_spatial_file)
                print(f"✅ Loaded spatial validation: {os.path.basename(latest_spatial_file)}")
                
        except Exception as e:
            print(f"⚠️  Could not load spatial validation results: {e}")
        
        return results
    
    def create_clustering_analysis(self, results):
        """Create clustering analysis without pie charts"""
        print("\n🎨 Creating clustering analysis...")
        
        if 'clustering' not in results:
            print("⚠️  No clustering data available")
            return
        
        clustering_df = results['clustering']
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('SAPFLUXNET Ecosystem Clustering Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cluster size bar chart (instead of pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        cluster_counts = clustering_df['cluster'].value_counts().sort_index()
        colors = [self.cluster_colors[i] for i in cluster_counts.index]
        
        bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Sites')
        ax1.set_title('Cluster Size Distribution', fontweight='bold')
        ax1.set_xticks(range(len(cluster_counts)))
        ax1.set_xticklabels([f'C{i}' for i in cluster_counts.index])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Geographic scatter plot
        ax2 = fig.add_subplot(gs[0, 1])
        if 'latitude' in clustering_df.columns and 'longitude' in clustering_df.columns:
            for cluster_id in sorted(clustering_df['cluster'].unique()):
                cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
                ax2.scatter(cluster_data['longitude'], cluster_data['latitude'], 
                          c=[self.cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', 
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Longitude (°)')
            ax2.set_ylabel('Latitude (°)')
            ax2.set_title('Geographic Distribution', fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Geographic data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Geographic Distribution', fontweight='bold')
        
        # 3. Climate space analysis
        ax3 = fig.add_subplot(gs[0, 2])
        if 'mean_annual_temp' in clustering_df.columns and 'mean_annual_precip' in clustering_df.columns:
            for cluster_id in sorted(clustering_df['cluster'].unique()):
                cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
                ax3.scatter(cluster_data['mean_annual_temp'], cluster_data['mean_annual_precip'], 
                          c=[self.cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', 
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax3.set_xlabel('Mean Annual Temperature (°C)')
            ax3.set_ylabel('Mean Annual Precipitation (mm)')
            ax3.set_title('Climate Space Distribution', fontweight='bold')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Climate data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Climate Space Distribution', fontweight='bold')
        
        # 4. Elevation analysis
        ax4 = fig.add_subplot(gs[1, :])
        if 'elevation' in clustering_df.columns:
            elevation_data = []
            cluster_labels = []
            
            for cluster_id in sorted(clustering_df['cluster'].unique()):
                cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
                elevation_data.append(cluster_data['elevation'].values)
                cluster_labels.append(f'Cluster {cluster_id}')
            
            bp = ax4.boxplot(elevation_data, labels=cluster_labels, patch_artist=True, 
                           showmeans=True, meanline=True)
            
            # Color the boxplots
            for patch, cluster_id in zip(bp['boxes'], sorted(clustering_df['cluster'].unique())):
                patch.set_facecolor(self.cluster_colors[cluster_id])
                patch.set_alpha(0.7)
            
            ax4.set_ylabel('Elevation (m)')
            ax4.set_title('Elevation Distribution by Cluster', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Elevation data not available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Elevation Distribution by Cluster', fontweight='bold')
        
        # 5. Ecosystem types bar chart
        ax5 = fig.add_subplot(gs[2, :])
        if 'ecosystem_interpretations' in results:
            interpretations = results['ecosystem_interpretations']
            cluster_names = []
            cluster_sizes = []
            colors_for_eco = []
            
            for cluster_id, info in interpretations.items():
                cluster_id_int = int(cluster_id)
                ecosystem_name = info['ecosystem_type']
                # Truncate long names
                if len(ecosystem_name) > 20:
                    ecosystem_name = ecosystem_name[:17] + "..."
                
                cluster_names.append(f"C{cluster_id}: {ecosystem_name}")
                cluster_sizes.append(info['n_sites'])
                colors_for_eco.append(self.cluster_colors[cluster_id_int])
            
            bars = ax5.barh(cluster_names, cluster_sizes, color=colors_for_eco, alpha=0.8, edgecolor='black')
            ax5.set_xlabel('Number of Sites')
            ax5.set_title('Ecosystem Types and Cluster Sizes', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, size in zip(bars, cluster_sizes):
                ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        str(size), ha='left', va='center', fontweight='bold')
        else:
            bars = ax5.bar(range(len(cluster_counts)), cluster_counts.values, 
                          color=[self.cluster_colors[i] for i in cluster_counts.index], 
                          alpha=0.8, edgecolor='black')
            ax5.set_xlabel('Cluster ID')
            ax5.set_ylabel('Number of Sites')
            ax5.set_title('Cluster Sizes', fontweight='bold')
            ax5.set_xticks(range(len(cluster_counts)))
            ax5.set_xticklabels([f'C{i}' for i in cluster_counts.index])
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        clustering_path = os.path.join(self.output_dir, f'clustering_analysis_matplotlib_{self.timestamp}.png')
        plt.savefig(clustering_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved clustering analysis: {clustering_path}")
    
    def create_model_performance_analysis(self, results):
        """Create model performance analysis"""
        print("\n🎨 Creating model performance analysis...")
        
        if 'model_metrics' not in results:
            print("⚠️  No model metrics data available")
            return
        
        model_metrics = results['model_metrics']
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('SAPFLUXNET Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Training vs Test R² comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(model_metrics))
        width = 0.35
        
        colors_train = [self.cluster_colors[i] for i in model_metrics['cluster']]
        colors_test = colors_train.copy()
        
        bars1 = ax1.bar(x_pos - width/2, model_metrics['train_r2'], width, 
                       label='Training R²', alpha=0.6, color=colors_train, edgecolor='black')
        bars2 = ax1.bar(x_pos + width/2, model_metrics['test_r2'], width, 
                       label='Test R²', alpha=0.9, color=colors_test, edgecolor='black')
        
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Training vs Test Performance', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'C{i}' for i in model_metrics['cluster']])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (train_r2, test_r2) in enumerate(zip(model_metrics['train_r2'], model_metrics['test_r2'])):
            ax1.text(i, max(train_r2, test_r2) + 0.01, f'{test_r2:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. RMSE by cluster
        ax2 = fig.add_subplot(gs[0, 1])
        colors_rmse = [self.cluster_colors[i] for i in model_metrics['cluster']]
        bars = ax2.bar(x_pos, model_metrics['test_rmse'], color=colors_rmse, 
                      alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Test RMSE')
        ax2.set_title('Test RMSE by Cluster', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'C{i}' for i in model_metrics['cluster']])
        ax2.grid(True, alpha=0.3)
        
        # Add RMSE labels
        for bar, rmse in zip(bars, model_metrics['test_rmse']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 3. Performance vs dataset size
        ax3 = fig.add_subplot(gs[0, 2])
        if 'total_samples' in model_metrics.columns:
            for i, row in model_metrics.iterrows():
                cluster_id = row['cluster']
                ax3.scatter(row['total_samples'], row['test_r2'], 
                          c=[self.cluster_colors[cluster_id]], s=150, alpha=0.8, 
                          edgecolors='black', linewidth=1)
                ax3.annotate(f'C{cluster_id}', (row['total_samples'], row['test_r2']), 
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax3.set_xlabel('Total Training Samples')
            ax3.set_ylabel('Test R²')
            ax3.set_title('Dataset Size vs Performance', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(model_metrics) > 2:
                z = np.polyfit(model_metrics['total_samples'], model_metrics['test_r2'], 1)
                p = np.poly1d(z)
                ax3.plot(model_metrics['total_samples'], p(model_metrics['total_samples']), 
                        "r--", alpha=0.8, linewidth=2, label=f'Trend: R²={z[0]:.2e}×samples+{z[1]:.3f}')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Sample size data\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Dataset Size vs Performance', fontweight='bold')
        
        # 4. R² distribution histogram
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(model_metrics['test_r2'], bins=10, alpha=0.7, color='skyblue', 
                edgecolor='black', linewidth=1)
        ax4.axvline(model_metrics['test_r2'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {model_metrics["test_r2"].mean():.3f}')
        ax4.axvline(model_metrics['test_r2'].median(), color='orange', linestyle='-', 
                   linewidth=2, label=f'Median: {model_metrics["test_r2"].median():.3f}')
        ax4.set_xlabel('Test R²')
        ax4.set_ylabel('Frequency')
        ax4.set_title('R² Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. RMSE distribution histogram
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(model_metrics['test_rmse'], bins=10, alpha=0.7, color='lightcoral', 
                edgecolor='black', linewidth=1)
        ax5.axvline(model_metrics['test_rmse'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {model_metrics["test_rmse"].mean():.3f}')
        ax5.axvline(model_metrics['test_rmse'].median(), color='orange', linestyle='-', 
                   linewidth=2, label=f'Median: {model_metrics["test_rmse"].median():.3f}')
        ax5.set_xlabel('Test RMSE')
        ax5.set_ylabel('Frequency')
        ax5.set_title('RMSE Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance summary table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create performance summary
        summary_data = []
        for _, row in model_metrics.iterrows():
            cluster_id = row['cluster']
            test_r2 = row['test_r2']
            test_rmse = row['test_rmse']
            
            if test_r2 > 0.95:
                status = '🔥 Excellent'
            elif test_r2 > 0.90:
                status = '✅ Very Good'
            elif test_r2 > 0.85:
                status = '👍 Good'
            elif test_r2 > 0.70:
                status = '⚠️ Fair'
            else:
                status = '❌ Poor'
            
            summary_data.append([f'C{cluster_id}', f'{test_r2:.4f}', f'{test_rmse:.3f}', status])
        
        # Create table
        table = ax6.table(cellText=summary_data,
                         colLabels=['Cluster', 'Test R²', 'Test RMSE', 'Status'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15, 0.2, 0.2, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by performance
        for i, row in enumerate(summary_data, 1):
            r2_val = float(row[1])
            if r2_val > 0.95:
                color = '#d4edda'  # Light green
            elif r2_val > 0.90:
                color = '#cce5ff'  # Light blue
            elif r2_val > 0.85:
                color = '#fff3cd'  # Light yellow
            else:
                color = '#f8d7da'  # Light red
            
            for j in range(4):
                table[(i, j)].set_facecolor(color)
        
        ax6.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        model_perf_path = os.path.join(self.output_dir, f'model_performance_matplotlib_{self.timestamp}.png')
        plt.savefig(model_perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved model performance analysis: {model_perf_path}")
    
    def create_feature_importance_analysis(self, results):
        """Create feature importance analysis"""
        print("\n🎨 Creating feature importance analysis...")
        
        if 'feature_importance' not in results:
            print("⚠️  No feature importance data available")
            return
        
        feature_importance = results['feature_importance']
        
        # Create individual cluster feature importance plots
        n_clusters = len(feature_importance)
        fig = plt.figure(figsize=(20, 4 * ((n_clusters + 1) // 2)))
        gs = gridspec.GridSpec((n_clusters + 1) // 2, 2, hspace=0.4, wspace=0.3)
        fig.suptitle('Feature Importance Analysis by Cluster', fontsize=16, fontweight='bold')
        
        cluster_ids = sorted(feature_importance.keys())
        
        for i, cluster_id in enumerate(cluster_ids):
            row = i // 2
            col = i % 2
            
            ax = fig.add_subplot(gs[row, col])
            
            fi_data = feature_importance[cluster_id].head(15)  # Top 15 features
            
            # Truncate long feature names
            feature_names = []
            for name in fi_data['feature_name']:
                if len(name) > 25:
                    feature_names.append(name[:22] + '...')
                else:
                    feature_names.append(name)
            
            y_pos = np.arange(len(fi_data))
            bars = ax.barh(y_pos, fi_data['importance'], alpha=0.8, 
                          color=self.cluster_colors[cluster_id], edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names, fontsize=9)
            ax.set_xlabel('Importance')
            ax.set_title(f'Cluster {cluster_id} - Top 15 Features', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add importance values to bars
            for bar, importance in zip(bars, fi_data['importance']):
                ax.text(bar.get_width() + max(fi_data['importance']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{importance:.0f}', ha='left', va='center', fontsize=8)
        
        # Hide empty subplot if odd number of clusters
        if n_clusters % 2 == 1:
            fig.add_subplot(gs[-1, -1]).axis('off')
        
        plt.tight_layout()
        feature_imp_path = os.path.join(self.output_dir, f'feature_importance_clusters_matplotlib_{self.timestamp}.png')
        plt.savefig(feature_imp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved feature importance by cluster: {feature_imp_path}")
        
        # Create universal features analysis
        self.create_universal_features_heatmap(feature_importance)
    
    def create_universal_features_heatmap(self, feature_importance):
        """Create heatmap of universal features across clusters"""
        print("  🔍 Creating universal features heatmap...")
        
        # Collect top features from each cluster
        all_features = set()
        for cluster_id, fi_df in feature_importance.items():
            top_features = fi_df.head(20)['feature_name'].values
            all_features.update(top_features)
        
        # Create importance matrix
        feature_matrix = []
        feature_names = sorted(list(all_features))
        cluster_ids = sorted(feature_importance.keys())
        
        for feature in feature_names:
            row = []
            for cluster_id in cluster_ids:
                fi_df = feature_importance[cluster_id]
                feature_row = fi_df[fi_df['feature_name'] == feature]
                if len(feature_row) > 0:
                    importance = feature_row['importance'].iloc[0]
                else:
                    importance = 0
                row.append(importance)
            feature_matrix.append(row)
        
        # Convert to numpy array and normalize
        feature_matrix = np.array(feature_matrix)
        
        # Only keep features that appear in at least 2 clusters
        feature_counts = (feature_matrix > 0).sum(axis=1)
        universal_mask = feature_counts >= 2
        
        if universal_mask.sum() == 0:
            print("  ⚠️  No universal features found")
            return
        
        universal_features = [feature_names[i] for i in range(len(feature_names)) if universal_mask[i]]
        universal_matrix = feature_matrix[universal_mask]
        
        # Normalize by row for better visualization
        row_max = universal_matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1  # Avoid division by zero
        universal_matrix_norm = universal_matrix / row_max
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(universal_features) * 0.3)))
        
        # Truncate feature names for display
        display_names = []
        for name in universal_features:
            if len(name) > 30:
                display_names.append(name[:27] + '...')
            else:
                display_names.append(name)
        
        im = ax.imshow(universal_matrix_norm, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
        ax.set_yticks(range(len(universal_features)))
        ax.set_yticklabels(display_names, fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Importance', rotation=270, labelpad=20)
        
        # Add text annotations for high importance values
        for i in range(len(universal_features)):
            for j in range(len(cluster_ids)):
                if universal_matrix_norm[i, j] > 0.5:  # Only annotate high values
                    text = ax.text(j, i, f'{universal_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="white" if universal_matrix_norm[i, j] > 0.7 else "black",
                                 fontsize=8, fontweight='bold')
        
        ax.set_title('Universal Features: Importance Across Clusters', fontweight='bold', pad=20)
        ax.set_xlabel('Clusters', fontweight='bold')
        ax.set_ylabel('Features', fontweight='bold')
        
        plt.tight_layout()
        universal_path = os.path.join(self.output_dir, f'universal_features_heatmap_matplotlib_{self.timestamp}.png')
        plt.savefig(universal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved universal features heatmap: {universal_path}")
    
    def create_spatial_validation_analysis(self, results):
        """Create spatial validation analysis"""
        print("\n🎨 Creating spatial validation analysis...")
        
        if 'spatial_validation' not in results:
            print("⚠️  No spatial validation data available")
            return
        
        spatial_df = results['spatial_validation']
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)
        fig.suptitle('SAPFLUXNET Spatial Validation Analysis', fontsize=16, fontweight='bold')
        
        cluster_ids = sorted(spatial_df['cluster'].unique())
        
        # 1. R² box plots by cluster
        ax1 = fig.add_subplot(gs[0, 0])
        r2_by_cluster = []
        cluster_labels = []
        
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            r2_by_cluster.append(cluster_data['test_r2'].values)
            cluster_labels.append(f'C{cluster_id}')
        
        bp1 = ax1.boxplot(r2_by_cluster, labels=cluster_labels, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Color box plots
        for patch, cluster_id in zip(bp1['boxes'], cluster_ids):
            patch.set_facecolor(self.cluster_colors[cluster_id])
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Test R²')
        ax1.set_title('Spatial Validation R² by Cluster', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='R² = 0')
        ax1.legend()
        
        # 2. RMSE box plots by cluster
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_by_cluster = []
        
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            rmse_by_cluster.append(cluster_data['test_rmse'].values)
        
        bp2 = ax2.boxplot(rmse_by_cluster, labels=cluster_labels, patch_artist=True,
                         showmeans=True, meanline=True)
        
        # Color box plots
        for patch, cluster_id in zip(bp2['boxes'], cluster_ids):
            patch.set_facecolor(self.cluster_colors[cluster_id])
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Test RMSE')
        ax2.set_title('Spatial Validation RMSE by Cluster', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Site performance scatter plot
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create x positions for sites grouped by cluster
        x_positions = []
        x_labels = []
        current_x = 0
        
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id].sort_values('test_r2')
            cluster_size = len(cluster_data)
            
            cluster_x_pos = np.arange(current_x, current_x + cluster_size)
            x_positions.extend(cluster_x_pos)
            
            # Plot points for this cluster
            ax3.scatter(cluster_x_pos, cluster_data['test_r2'], 
                       c=[self.cluster_colors[cluster_id]] * cluster_size, 
                       s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                       label=f'Cluster {cluster_id}')
            
            # Add cluster separator
            if cluster_id != cluster_ids[-1]:  # Don't add separator after last cluster
                ax3.axvline(x=current_x + cluster_size - 0.5, color='gray', 
                           linestyle='-', alpha=0.5, linewidth=1)
            
            # Add cluster label
            ax3.text(current_x + cluster_size/2, ax3.get_ylim()[1] * 0.9, 
                    f'C{cluster_id}', ha='center', va='center', 
                    fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=self.cluster_colors[cluster_id], alpha=0.7))
            
            current_x += cluster_size
        
        ax3.set_xlabel('Sites (grouped by cluster)')
        ax3.set_ylabel('Test R²')
        ax3.set_title('Site-Level Spatial Validation Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Identify and label extreme outliers
        outliers = spatial_df[spatial_df['test_r2'] < -5]
        for _, outlier in outliers.iterrows():
            site_idx = spatial_df.index[spatial_df['test_site'] == outlier['test_site']].tolist()[0]
            ax3.annotate(outlier['test_site'][:8], 
                        (site_idx, outlier['test_r2']), 
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8, alpha=0.8, color='red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
        
        # 4. Performance statistics by cluster
        ax4 = fig.add_subplot(gs[2, 0])
        
        cluster_stats = []
        for cluster_id in cluster_ids:
            cluster_data = spatial_df[spatial_df['cluster'] == cluster_id]
            mean_r2 = cluster_data['test_r2'].mean()
            std_r2 = cluster_data['test_r2'].std()
            success_rate = len(cluster_data[cluster_data['test_r2'] > 0]) / len(cluster_data) * 100
            
            cluster_stats.append([mean_r2, std_r2, success_rate])
        
        cluster_stats = np.array(cluster_stats)
        
        # Bar plot of mean R² with error bars
        bars = ax4.bar(range(len(cluster_ids)), cluster_stats[:, 0], 
                      yerr=cluster_stats[:, 1], capsize=5,
                      color=[self.cluster_colors[i] for i in cluster_ids], 
                      alpha=0.8, edgecolor='black')
        
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Mean Test R²')
        ax4.set_title('Mean Performance by Cluster', fontweight='bold')
        ax4.set_xticks(range(len(cluster_ids)))
        ax4.set_xticklabels([f'C{i}' for i in cluster_ids])
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, mean_r2 in zip(bars, cluster_stats[:, 0]):
            ax4.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.05 if bar.get_height() > 0 else -0.1), 
                    f'{mean_r2:.3f}', ha='center', 
                    va='bottom' if bar.get_height() > 0 else 'top', 
                    fontweight='bold', fontsize=9)
        
        # 5. Success rates
        ax5 = fig.add_subplot(gs[2, 1])
        
        success_rates = cluster_stats[:, 2]
        bars = ax5.bar(range(len(cluster_ids)), success_rates,
                      color=[self.cluster_colors[i] for i in cluster_ids], 
                      alpha=0.8, edgecolor='black')
        
        ax5.set_xlabel('Cluster ID')
        ax5.set_ylabel('Success Rate (%)')
        ax5.set_title('Success Rate by Cluster (R² > 0)', fontweight='bold')
        ax5.set_xticks(range(len(cluster_ids)))
        ax5.set_xticklabels([f'C{i}' for i in cluster_ids])
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        spatial_val_path = os.path.join(self.output_dir, f'spatial_validation_matplotlib_{self.timestamp}.png')
        plt.savefig(spatial_val_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved spatial validation analysis: {spatial_val_path}")
        
        # Create detailed outlier analysis
        self.create_outlier_detailed_analysis(spatial_df)
    
    def create_outlier_detailed_analysis(self, spatial_df):
        """Create detailed outlier analysis"""
        print("  🔍 Creating detailed outlier analysis...")
        
        # Identify outliers
        q1 = spatial_df['test_r2'].quantile(0.25)
        q3 = spatial_df['test_r2'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = spatial_df[
            (spatial_df['test_r2'] < lower_bound) | 
            (spatial_df['test_r2'] > upper_bound) |
            (spatial_df['test_r2'] < -2)  # Additional criterion for poor performance
        ].copy()
        
        if len(outliers) == 0:
            print("  ✅ No significant outliers found")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)
        fig.suptitle(f'Outlier Analysis: {len(outliers)} Problematic Sites', fontsize=16, fontweight='bold')
        
        # 1. Outliers by cluster
        ax1 = fig.add_subplot(gs[0, 0])
        outlier_counts = outliers['cluster'].value_counts().sort_index()
        
        if len(outlier_counts) > 0:
            colors = [self.cluster_colors[i] for i in outlier_counts.index]
            bars = ax1.bar(range(len(outlier_counts)), outlier_counts.values, 
                          color=colors, alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Number of Outliers')
            ax1.set_title('Outlier Distribution by Cluster', fontweight='bold')
            ax1.set_xticks(range(len(outlier_counts)))
            ax1.set_xticklabels([f'C{i}' for i in outlier_counts.index])
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, outlier_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. R² vs RMSE scatter for outliers
        ax2 = fig.add_subplot(gs[0, 1])
        for cluster_id in outliers['cluster'].unique():
            cluster_outliers = outliers[outliers['cluster'] == cluster_id]
            ax2.scatter(cluster_outliers['test_r2'], cluster_outliers['test_rmse'], 
                       c=[self.cluster_colors[cluster_id]], s=80, alpha=0.7, 
                       edgecolors='black', linewidth=0.5, label=f'Cluster {cluster_id}')
        
        ax2.set_xlabel('Test R²')
        ax2.set_ylabel('Test RMSE')
        ax2.set_title('Outlier Performance Scatter', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Annotate worst outliers
        worst_outliers = outliers.nsmallest(5, 'test_r2')
        for _, outlier in worst_outliers.iterrows():
            ax2.annotate(outlier['test_site'][:8], 
                        (outlier['test_r2'], outlier['test_rmse']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, fontweight='bold')
        
        # 3. Outlier R² histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(outliers['test_r2'], bins=15, alpha=0.7, color='red', 
                edgecolor='black', linewidth=1)
        ax3.axvline(outliers['test_r2'].mean(), color='blue', linestyle='-', 
                   linewidth=2, label=f'Mean: {outliers["test_r2"].mean():.3f}')
        ax3.axvline(outliers['test_r2'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {outliers["test_r2"].median():.3f}')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.7, 
                   linewidth=2, label='R² = 0')
        ax3.set_xlabel('Test R²')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Outlier R² Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Top 10 worst sites table
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        worst_10 = outliers.nsmallest(10, 'test_r2')
        
        table_data = []
        for i, (_, outlier) in enumerate(worst_10.iterrows(), 1):
            status_icon = "🚨" if outlier['test_r2'] < -10 else "⚠️" if outlier['test_r2'] < 0 else "❓"
            table_data.append([
                f"{i:2d}",
                outlier['test_site'][:15],
                f"C{outlier['cluster']}",
                f"{outlier['test_r2']:8.4f}",
                f"{outlier['test_rmse']:6.3f}",
                f"{outlier['train_samples']:,}",
                f"{outlier['test_samples']:,}",
                status_icon
            ])
        
        # Create table
        table = ax4.table(cellText=table_data,
                         colLabels=['Rank', 'Site', 'Cluster', 'Test R²', 'Test RMSE', 
                                   'Train Samples', 'Test Samples', 'Status'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.08, 0.2, 0.1, 0.15, 0.15, 0.12, 0.12, 0.08])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Color the header
        for i in range(8):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by performance
        for i, row_data in enumerate(table_data, 1):
            r2_val = float(row_data[3])
            if r2_val < -10:
                color = '#ffcccc'  # Light red
            elif r2_val < 0:
                color = '#ffe6cc'  # Light orange
            else:
                color = '#fff3cd'  # Light yellow
            
            for j in range(8):
                table[(i, j)].set_facecolor(color)
        
        # Add special highlighting for COL_MAC_SAF_RAD
        for i, row_data in enumerate(table_data, 1):
            if 'COL_MAC' in row_data[1]:
                for j in range(8):
                    table[(i, j)].set_facecolor('#ff9999')  # Bright red
                    table[(i, j)].set_text_props(weight='bold')
        
        ax4.set_title('Top 10 Worst Performing Sites', fontweight='bold', pad=30, fontsize=14)
        
        plt.tight_layout()
        outlier_path = os.path.join(self.output_dir, f'outlier_analysis_matplotlib_{self.timestamp}.png')
        plt.savefig(outlier_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved outlier analysis: {outlier_path}")
    
    def run_all_visualizations(self):
        """Run all matplotlib visualizations"""
        print("📊 SAPFLUXNET Matplotlib Results Visualization")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Create comprehensive matplotlib-only visualizations (no pie charts)")
        
        try:
            # Load all results
            results = self.load_all_results()
            
            if not results:
                print("❌ No results data found")
                return
            
            # Create all visualizations
            print(f"\n🎨 Creating matplotlib visualizations...")
            
            self.create_clustering_analysis(results)
            self.create_model_performance_analysis(results)
            self.create_feature_importance_analysis(results)
            self.create_spatial_validation_analysis(results)
            
            print(f"\n✅ All matplotlib visualizations completed!")
            print(f"📁 Check output directory: {self.output_dir}")
            
            # List generated files
            output_files = glob.glob(os.path.join(self.output_dir, f'*matplotlib_{self.timestamp}*'))
            print(f"\n📄 Generated {len(output_files)} visualization files:")
            for file in sorted(output_files):
                print(f"  - {os.path.basename(file)}")
                
        except Exception as e:
            print(f"❌ Error in visualization creation: {e}")
            raise

def main():
    visualizer = MatplotlibResultsVisualizer()
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main() 