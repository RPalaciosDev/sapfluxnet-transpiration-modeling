#!/usr/bin/env python3
"""
Comprehensive Feature Importance Visualization
Creates 2D and 3D visualizations for cluster model feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceVisualizer:
    def __init__(self, results_dir='results/cluster_models'):
        self.results_dir = Path(results_dir)
        self.timestamp = '20250726_112236'  # Latest run timestamp
        self.output_dir = Path('feature_importance_visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load all cluster feature importance data"""
        self.cluster_data = {}
        
        for cluster_id in [0, 1, 2]:
            file_path = self.results_dir / f'feature_importance_cluster_{cluster_id}_{self.timestamp}_mapped.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                self.cluster_data[cluster_id] = df
                print(f"✅ Loaded cluster {cluster_id}: {len(df)} features")
            else:
                print(f"❌ File not found: {file_path}")
        
        return len(self.cluster_data) > 0
    
    def create_2d_visualizations(self):
        """Create comprehensive 2D visualizations"""
        print("🎨 Creating 2D visualizations...")
        
        # 1. Top Features Comparison Across Clusters
        self._plot_top_features_comparison()
        
        # 2. Category Distribution by Cluster
        self._plot_category_distributions()
        
        # 3. Feature Importance Heatmap
        self._plot_importance_heatmap()
        
        # 4. Category Importance Comparison
        self._plot_category_importance()
        
        # 5. Distribution Analysis
        self._plot_importance_distributions()
        
    def create_3d_visualizations(self):
        """Create interactive 3D visualizations"""
        print("🌟 Creating 3D visualizations...")
        
        # 1. 3D Feature Space
        self._plot_3d_feature_space()
        
        # 2. 3D Category Analysis
        self._plot_3d_category_analysis()
        
        # 3. 3D Cluster Comparison
        self._plot_3d_cluster_comparison()
        
    def _plot_top_features_comparison(self):
        """Compare top features across clusters"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Top 15 Features by Cluster', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (cluster_id, df) in enumerate(self.cluster_data.items()):
            top_features = df.head(15)
            
            bars = axes[i].barh(range(len(top_features)), top_features['importance_score'], 
                               color=colors[i], alpha=0.8)
            
            # Add feature names
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels([f"{row['feature_name'][:20]}..." if len(row['feature_name']) > 20 
                                   else row['feature_name'] for _, row in top_features.iterrows()], 
                                  fontsize=10)
            
            axes[i].set_xlabel('Importance Score', fontsize=12)
            axes[i].set_title(f'Cluster {cluster_id}\n({self._get_cluster_strategy(cluster_id)})', 
                            fontsize=14, fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                width = bar.get_width()
                axes[i].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:,.0f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_features_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_category_distributions(self):
        """Plot feature category distributions"""
        # Prepare data
        category_data = []
        for cluster_id, df in self.cluster_data.items():
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                category_data.append({
                    'cluster': f'Cluster {cluster_id}',
                    'category': category,
                    'count': count,
                    'percentage': (count / len(df)) * 100
                })
        
        category_df = pd.DataFrame(category_data)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Stacked bar chart - counts
        pivot_counts = category_df.pivot(index='category', columns='cluster', values='count').fillna(0)
        pivot_counts.plot(kind='bar', stacked=True, ax=ax1, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax1.set_title('Feature Count by Category and Cluster', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Category', fontsize=12)
        ax1.set_ylabel('Number of Features', fontsize=12)
        ax1.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        
        # Percentage comparison
        pivot_pct = category_df.pivot(index='category', columns='cluster', values='percentage').fillna(0)
        pivot_pct.plot(kind='bar', ax=ax2, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax2.set_title('Feature Percentage by Category and Cluster', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Category', fontsize=12)
        ax2.set_ylabel('Percentage of Features (%)', fontsize=12)
        ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_importance_heatmap(self):
        """Create heatmap of top features across clusters"""
        # Get top 20 features from each cluster
        top_features = set()
        for df in self.cluster_data.values():
            top_features.update(df.head(20)['feature_name'].tolist())
        
        # Create matrix
        heatmap_data = []
        for feature in top_features:
            row = {'feature': feature}
            for cluster_id, df in self.cluster_data.items():
                feature_row = df[df['feature_name'] == feature]
                if not feature_row.empty:
                    row[f'Cluster_{cluster_id}'] = feature_row['importance_score'].iloc[0]
                else:
                    row[f'Cluster_{cluster_id}'] = 0
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data).set_index('feature')
        
        # Create heatmap
        plt.figure(figsize=(12, 16))
        sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Importance Score'})
        plt.title('Feature Importance Heatmap Across Clusters', fontsize=16, fontweight='bold')
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_category_importance(self):
        """Plot total importance by category"""
        category_importance = []
        
        for cluster_id, df in self.cluster_data.items():
            category_sums = df.groupby('category')['importance_score'].sum().reset_index()
            for _, row in category_sums.iterrows():
                category_importance.append({
                    'cluster': f'Cluster {cluster_id}',
                    'category': row['category'],
                    'total_importance': row['importance_score']
                })
        
        cat_df = pd.DataFrame(category_importance)
        
        plt.figure(figsize=(16, 10))
        
        # Create grouped bar chart
        categories = cat_df['category'].unique()
        x = np.arange(len(categories))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, cluster in enumerate(['Cluster 0', 'Cluster 1', 'Cluster 2']):
            cluster_data = cat_df[cat_df['cluster'] == cluster]
            values = [cluster_data[cluster_data['category'] == cat]['total_importance'].values[0] 
                     if len(cluster_data[cluster_data['category'] == cat]) > 0 else 0 
                     for cat in categories]
            
            bars = plt.bar(x + i*width, values, width, label=cluster, color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Feature Category', fontsize=12)
        plt.ylabel('Total Importance Score', fontsize=12)
        plt.title('Total Feature Importance by Category and Cluster', fontsize=16, fontweight='bold')
        plt.xticks(x + width, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_importance_distributions(self):
        """Plot importance score distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Distributions', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Overall distribution
        for i, (cluster_id, df) in enumerate(self.cluster_data.items()):
            axes[0, 0].hist(df['importance_score'], bins=50, alpha=0.7, 
                          label=f'Cluster {cluster_id}', color=colors[i])
        axes[0, 0].set_xlabel('Importance Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. Box plot by cluster
        all_data = []
        all_clusters = []
        for cluster_id, df in self.cluster_data.items():
            all_data.extend(df['importance_score'].tolist())
            all_clusters.extend([f'Cluster {cluster_id}'] * len(df))
        
        box_df = pd.DataFrame({'importance': all_data, 'cluster': all_clusters})
        sns.boxplot(data=box_df, x='cluster', y='importance', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution by Cluster')
        axes[0, 1].set_yscale('log')
        
        # 3. Top 10 features violin plot
        top_features_data = []
        for cluster_id, df in self.cluster_data.items():
            top_10 = df.head(10)
            for _, row in top_10.iterrows():
                top_features_data.append({
                    'cluster': f'Cluster {cluster_id}',
                    'importance': row['importance_score']
                })
        
        top_df = pd.DataFrame(top_features_data)
        sns.violinplot(data=top_df, x='cluster', y='importance', ax=axes[1, 0])
        axes[1, 0].set_title('Top 10 Features Distribution')
        
        # 4. Cumulative importance
        for i, (cluster_id, df) in enumerate(self.cluster_data.items()):
            sorted_importance = df['importance_score'].sort_values(ascending=False)
            cumulative_pct = (sorted_importance.cumsum() / sorted_importance.sum()) * 100
            axes[1, 1].plot(range(1, len(cumulative_pct) + 1), cumulative_pct, 
                          label=f'Cluster {cluster_id}', color=colors[i], linewidth=2)
        
        axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('Cumulative Importance (%)')
        axes[1, 1].set_title('Cumulative Importance')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'importance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_3d_feature_space(self):
        """Create 3D scatter plot of feature importance space"""
        # Prepare data for 3D plot
        plot_data = []
        for cluster_id, df in self.cluster_data.items():
            top_30 = df.head(30)  # Top 30 features for readability
            for _, row in top_30.iterrows():
                plot_data.append({
                    'cluster': f'Cluster {cluster_id}',  # Keep as string for discrete handling
                    'cluster_id': cluster_id,  # Keep numeric for coloring
                    'feature_name': row['feature_name'],
                    'category': row['category'],
                    'importance': row['importance_score'],
                    'rank': row.name + 1,
                    'log_importance': np.log10(row['importance_score'] + 1)
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create 3D scatter plot with proper discrete handling
        fig = px.scatter_3d(
            plot_df,
            x='category',  # Use category instead of cluster for x-axis
            y='rank', 
            z='log_importance',
            color='cluster',  # Use cluster as color (discrete)
            size='importance',
            hover_data=['feature_name', 'importance', 'cluster'],
            title='3D Feature Importance Space<br><sub>X=Category, Y=Rank, Z=Log(Importance), Color=Cluster</sub>',
            labels={
                'category': 'Feature Category',
                'rank': 'Feature Rank (within cluster)',
                'log_importance': 'Log(Importance Score)',
                'cluster': 'Cluster'
            },
            color_discrete_map={
                'Cluster 0': '#FF6B6B',
                'Cluster 1': '#4ECDC4', 
                'Cluster 2': '#45B7D1'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature Category',
                yaxis_title='Feature Rank (within cluster)',
                zaxis_title='Log(Importance Score)'
            ),
            width=1000,
            height=800
        )
        
        # Save as HTML
        fig.write_html(self.output_dir / '3d_feature_space.html')
        print(f"📊 3D Feature Space saved to: {self.output_dir / '3d_feature_space.html'}")
        
    def _plot_3d_category_analysis(self):
        """Create 3D analysis of category importance"""
        # Aggregate by category
        category_data = []
        for cluster_id, df in self.cluster_data.items():
            category_stats = df.groupby('category').agg({
                'importance_score': ['sum', 'mean', 'count']
            }).round(2)
            
            category_stats.columns = ['total_importance', 'mean_importance', 'feature_count']
            category_stats = category_stats.reset_index()
            
            for _, row in category_stats.iterrows():
                category_data.append({
                    'cluster': f'Cluster {cluster_id}',  # Keep as string
                    'cluster_id': cluster_id,
                    'category': row['category'],
                    'total_importance': row['total_importance'],
                    'mean_importance': row['mean_importance'],
                    'feature_count': row['feature_count']
                })
        
        cat_df = pd.DataFrame(category_data)
        
        # Create 3D scatter plot with category on x-axis instead of cluster
        fig = px.scatter_3d(
            cat_df,
            x='category',  # Use category as x-axis (discrete)
            y='feature_count',
            z='total_importance',
            color='cluster',  # Use cluster for coloring (discrete)
            size='mean_importance',
            hover_data=['mean_importance', 'cluster'],
            title='3D Category Analysis<br><sub>X=Category, Y=Feature Count, Z=Total Importance, Size=Mean Importance</sub>',
            labels={
                'category': 'Feature Category',
                'feature_count': 'Number of Features',
                'total_importance': 'Total Importance Score',
                'cluster': 'Cluster'
            },
            color_discrete_map={
                'Cluster 0': '#FF6B6B',
                'Cluster 1': '#4ECDC4', 
                'Cluster 2': '#45B7D1'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Feature Category',
                yaxis_title='Number of Features',
                zaxis_title='Total Importance Score'
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(self.output_dir / '3d_category_analysis.html')
        print(f"📊 3D Category Analysis saved to: {self.output_dir / '3d_category_analysis.html'}")
        
    def _plot_3d_cluster_comparison(self):
        """Create 3D comparison of clusters using separate traces"""
        # Create comparison data
        comparison_data = []
        
        for cluster_id, df in self.cluster_data.items():
            # Get top features by category
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                if len(cat_df) > 0:
                    top_feature = cat_df.iloc[0]
                    comparison_data.append({
                        'cluster': cluster_id,
                        'cluster_name': f'Cluster {cluster_id}',
                        'category': category,
                        'top_feature': top_feature['feature_name'],
                        'max_importance': top_feature['importance_score'],
                        'total_importance': cat_df['importance_score'].sum(),
                        'feature_count': len(cat_df)
                    })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create interactive 3D plot with separate traces for each cluster
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        cluster_names = ['Interaction-Driven', 'Time-Lag Dependent', 'Structure & Dynamics']
        
        # Create a separate subplot for each cluster to avoid fractional cluster IDs
        for i, cluster_id in enumerate([0, 1, 2]):
            cluster_data = comp_df[comp_df['cluster'] == cluster_id]
            
            # Use category positions for x-axis (discrete)
            categories = cluster_data['category'].unique()
            category_positions = {cat: j for j, cat in enumerate(categories)}
            x_positions = [category_positions[cat] for cat in cluster_data['category']]
            
            fig.add_trace(go.Scatter3d(
                x=x_positions,
                y=cluster_data['total_importance'],
                z=cluster_data['max_importance'],
                mode='markers+text',
                marker=dict(
                    size=np.sqrt(cluster_data['max_importance']) / 50,  # Scale for visibility
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=cluster_data['category'],
                textposition='top center',
                name=f'Cluster {cluster_id}: {cluster_names[i]}',
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'Cluster: ' + cluster_data['cluster_name'].iloc[0] + '<br>' +
                    'Feature Count: ' + cluster_data['feature_count'].astype(str) + '<br>' +
                    'Total Importance: %{y:,.0f}<br>' +
                    'Max Importance: %{z:,.0f}<br>' +
                    '<extra></extra>'
                ),
                customdata=cluster_data[['feature_count', 'cluster_name']]
            ))
        
        # Update x-axis to show category names
        all_categories = sorted(comp_df['category'].unique())
        category_tickvals = list(range(len(all_categories)))
        
        fig.update_layout(
            title='3D Cluster Strategy Comparison<br><sub>X=Category Position, Y=Total Importance, Z=Max Feature Importance</sub>',
            scene=dict(
                xaxis=dict(
                    title='Feature Categories',
                    tickvals=category_tickvals,
                    ticktext=all_categories,
                    type='category'
                ),
                yaxis_title='Total Category Importance',
                zaxis_title='Highest Feature Importance',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html(self.output_dir / '3d_cluster_comparison.html')
        print(f"📊 3D Cluster Comparison saved to: {self.output_dir / '3d_cluster_comparison.html'}")
        
    def _get_cluster_strategy(self, cluster_id):
        """Get cluster strategy description"""
        strategies = {
            0: "Interaction-Driven",
            1: "Time-Lag Dependent", 
            2: "Structure & Dynamics"
        }
        return strategies.get(cluster_id, "Unknown")
        
    def create_summary_report(self):
        """Create a summary report of the analysis"""
        report_path = self.output_dir / 'visualization_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("FEATURE IMPORTANCE VISUALIZATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: Cluster models from {self.timestamp}\n\n")
            
            f.write("GENERATED VISUALIZATIONS:\n")
            f.write("-" * 25 + "\n")
            f.write("2D Visualizations:\n")
            f.write("  • top_features_comparison.png - Top 15 features per cluster\n")
            f.write("  • category_distributions.png - Feature category analysis\n")
            f.write("  • importance_heatmap.png - Cross-cluster feature heatmap\n")
            f.write("  • category_importance.png - Total importance by category\n")
            f.write("  • importance_distributions.png - Statistical distributions\n\n")
            
            f.write("3D Interactive Visualizations:\n")
            f.write("  • 3d_feature_space.html - Feature importance space\n")
            f.write("  • 3d_category_analysis.html - Category-level analysis\n")
            f.write("  • 3d_cluster_comparison.html - Cluster strategy comparison\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 12 + "\n")
            for cluster_id, df in self.cluster_data.items():
                strategy = self._get_cluster_strategy(cluster_id)
                top_feature = df.iloc[0]
                f.write(f"Cluster {cluster_id} ({strategy}):\n")
                f.write(f"  • Top feature: {top_feature['feature_name']} ({top_feature['importance_score']:,.0f})\n")
                f.write(f"  • Total features: {len(df)}\n")
                f.write(f"  • Most common category: {df['category'].mode()[0]}\n\n")
        
        print(f"📄 Summary report saved to: {report_path}")

def main():
    """Main execution function"""
    print("🎨 Starting Feature Importance Visualization...")
    
    visualizer = FeatureImportanceVisualizer()
    
    if not visualizer.load_data():
        print("❌ Failed to load data. Check file paths.")
        return
    
    print(f"📁 Output directory: {visualizer.output_dir}")
    
    # Create visualizations
    visualizer.create_2d_visualizations()
    visualizer.create_3d_visualizations()
    visualizer.create_summary_report()
    
    print("✅ All visualizations completed successfully!")
    print(f"📂 Check the '{visualizer.output_dir}' directory for all outputs")

if __name__ == "__main__":
    main() 