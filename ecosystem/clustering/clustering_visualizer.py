"""
Clustering Visualization Module

Provides comprehensive visualization capabilities for the flexible clustering pipeline,
including feature space plots, cluster validation, strategy comparison, and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')


class ClusteringVisualizer:
    """
    Comprehensive visualization toolkit for clustering analysis.
    
    Provides static and interactive visualizations for cluster validation,
    feature analysis, strategy comparison, and geographic mapping.
    """
    
    def __init__(self, output_dir: str = 'clustering_visualizations', 
                 style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
            style: Matplotlib style for static plots
            figsize: Default figure size for static plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib
        plt.style.use(style)
        self.figsize = figsize
        
        # Configure seaborn
        sns.set_palette("husl")
        
        # Storage for visualization data
        self.clustering_data = {}
        self.feature_data = {}
        self.strategy_results = {}
        
        print(f"🎨 ClusteringVisualizer initialized")
        print(f"📁 Output directory: {output_dir}")
    
    def set_clustering_data(self, clustering_df: pd.DataFrame, features: List[str], 
                           cluster_labels: np.ndarray, strategy_info: Dict[str, Any]):
        """
        Set the main clustering data for visualization.
        
        Args:
            clustering_df: DataFrame with processed clustering data
            features: List of feature names used for clustering
            cluster_labels: Cluster assignments for each site
            strategy_info: Information about the clustering strategy used
        """
        self.clustering_data = {
            'df': clustering_df.copy(),
            'features': features,
            'labels': cluster_labels,
            'strategy': strategy_info,
            'n_clusters': len(np.unique(cluster_labels)),
            'n_sites': len(clustering_df)
        }
        
        # Add cluster labels to dataframe
        self.clustering_data['df']['cluster'] = cluster_labels
        
        print(f"✅ Clustering data set: {self.clustering_data['n_sites']} sites, "
              f"{len(features)} features, {self.clustering_data['n_clusters']} clusters")
    
    def set_strategy_comparison(self, strategies: List[Dict[str, Any]]):
        """
        Set strategy comparison data for visualization.
        
        Args:
            strategies: List of strategy results from clustering evaluation
        """
        self.strategy_results = {
            'strategies': strategies,
            'n_strategies': len(strategies)
        }
        
        print(f"✅ Strategy comparison data set: {len(strategies)} strategies")
    
    def plot_feature_space_2d(self, method: str = 'pca', save: bool = True, 
                             show: bool = True) -> str:
        """
        Plot 2D feature space visualization using dimensionality reduction.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'tsne')
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        df = self.clustering_data['df']
        features = self.clustering_data['features']
        
        # Prepare feature matrix
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X_scaled)
            explained_var = reducer.explained_variance_ratio_
            xlabel = f'PC1 ({explained_var[0]:.1%} variance)'
            ylabel = f'PC2 ({explained_var[1]:.1%} variance)'
            title = f'PCA Feature Space - {self.clustering_data["strategy"]["name"]}'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
            X_2d = reducer.fit_transform(X_scaled)
            xlabel = 't-SNE Component 1'
            ylabel = 't-SNE Component 2'
            title = f't-SNE Feature Space - {self.clustering_data["strategy"]["name"]}'
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot points colored by cluster
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                           c=df['cluster'], 
                           cmap='tab10',
                           s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add site labels for small datasets
        if len(df) <= 50:
            for i, site in enumerate(df['site']):
                ax.annotate(site, (X_2d[i, 0], X_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Customize plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        silhouette = self.clustering_data['strategy']['silhouette']
        balance = self.clustering_data['strategy']['balance_ratio']
        stats_text = f'Silhouette: {silhouette:.3f}\nBalance: {balance:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'feature_space_{method}_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved feature space plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filepath) if save else ""
    
    def plot_feature_space_3d_interactive(self, method: str = 'pca', save: bool = True) -> str:
        """
        Create interactive 3D feature space visualization.
        
        Args:
            method: Dimensionality reduction method ('pca' or 'tsne')
            save: Whether to save the plot
            
        Returns:
            Path to saved HTML file
        """
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        df = self.clustering_data['df']
        features = self.clustering_data['features']
        
        # Prepare feature matrix
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=3)
            X_3d = reducer.fit_transform(X_scaled)
            explained_var = reducer.explained_variance_ratio_
            labels = [f'PC{i+1} ({explained_var[i]:.1%})' for i in range(3)]
            title = f'3D PCA Feature Space - {self.clustering_data["strategy"]["name"]}'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(df)-1))
            X_3d = reducer.fit_transform(X_scaled)
            labels = [f't-SNE {i+1}' for i in range(3)]
            title = f'3D t-SNE Feature Space - {self.clustering_data["strategy"]["name"]}'
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # Create interactive 3D plot
        fig = go.Figure(data=go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1], 
            z=X_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=df['cluster'],
                colorscale='Plotly3',
                showscale=True,
                colorbar=dict(title="Cluster"),
                line=dict(width=1, color='black')
            ),
            text=df['site'],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{labels[0]}: %{{x:.2f}}<br>' +
                         f'{labels[1]}: %{{y:.2f}}<br>' +
                         f'{labels[2]}: %{{z:.2f}}<br>' +
                         'Cluster: %{marker.color}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2]
            ),
            width=800,
            height=600
        )
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'feature_space_3d_{method}_{timestamp}.html'
            filepath = self.output_dir / filename
            fig.write_html(filepath)
            print(f"💾 Saved 3D feature space plot: {filepath}")
            return str(filepath)
        
        return ""
    
    def plot_silhouette_analysis(self, save: bool = True, show: bool = True) -> str:
        """
        Create detailed silhouette analysis plot.
        
        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        df = self.clustering_data['df']
        features = self.clustering_data['features']
        labels = self.clustering_data['labels']
        
        # Prepare feature matrix
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X_scaled, labels)
        sample_silhouette_values = silhouette_samples(X_scaled, labels)
        
        # Create silhouette plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Silhouette plot
        y_lower = 10
        for i in range(self.clustering_data['n_clusters']):
            # Aggregate silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(i)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster label')
        ax1.set_title('Silhouette Plot for Individual Clusters')
        
        # Add vertical line for average silhouette score
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {silhouette_avg:.3f}')
        ax1.legend()
        
        # Plot 2: Cluster visualization in 2D
        # Use PCA for 2D representation
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        colors = plt.cm.tab10(labels)
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], marker='o', s=50, c=colors, alpha=0.7, edgecolors='black')
        
        # Mark cluster centers
        centers_2d = []
        for i in range(self.clustering_data['n_clusters']):
            cluster_center = X_2d[labels == i].mean(axis=0)
            centers_2d.append(cluster_center)
            ax2.scatter(cluster_center[0], cluster_center[1], marker='x', s=200, 
                       c='red', linewidths=3, label=f'Center {i}' if i == 0 else "")
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.set_title('Clusters in PCA Space')
        if self.clustering_data['n_clusters'] == 1:
            ax2.legend(['Center'])
        
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'silhouette_analysis_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved silhouette analysis: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filepath) if save else ""
    
    def plot_strategy_comparison(self, save: bool = True, show: bool = True) -> str:
        """
        Create comprehensive strategy comparison visualization.
        
        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.strategy_results:
            raise ValueError("No strategy results set. Call set_strategy_comparison() first.")
        
        strategies = self.strategy_results['strategies']
        
        # Convert strategies to DataFrame for easier plotting
        strategy_df = pd.DataFrame([{
            'name': s['name'],
            'algorithm': s['algorithm'],
            'n_clusters': s['n_clusters'],
            'silhouette': s['silhouette'],
            'balance_ratio': s['balance_ratio'],
            'final_score': s.get('final_score', s['silhouette'] * 1.5 + s['balance_ratio'])
        } for s in strategies])
        
        # Create multi-panel comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Silhouette vs Balance colored by algorithm
        algorithms = strategy_df['algorithm'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        algorithm_colors = dict(zip(algorithms, colors))
        
        for alg in algorithms:
            alg_data = strategy_df[strategy_df['algorithm'] == alg]
            ax1.scatter(alg_data['silhouette'], alg_data['balance_ratio'], 
                       c=[algorithm_colors[alg]], label=alg, s=60, alpha=0.7, edgecolors='black')
        
        ax1.set_xlabel('Silhouette Score')
        ax1.set_ylabel('Balance Ratio')
        ax1.set_title('Strategy Performance: Silhouette vs Balance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final scores by number of clusters
        for alg in algorithms:
            alg_data = strategy_df[strategy_df['algorithm'] == alg]
            ax2.plot(alg_data['n_clusters'], alg_data['final_score'], 
                    'o-', label=alg, color=algorithm_colors[alg], linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Final Score')
        ax2.set_title('Final Scores by Cluster Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Silhouette scores by algorithm (box plot)
        algorithm_silhouettes = [strategy_df[strategy_df['algorithm'] == alg]['silhouette'].values 
                               for alg in algorithms]
        box_plot = ax3.boxplot(algorithm_silhouettes, labels=algorithms, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Silhouette Score Distribution by Algorithm')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Top strategies ranking
        top_strategies = strategy_df.nlargest(10, 'final_score')
        y_pos = np.arange(len(top_strategies))
        
        bars = ax4.barh(y_pos, top_strategies['final_score'], 
                       color=[algorithm_colors[alg] for alg in top_strategies['algorithm']])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{row['algorithm']} ({row['n_clusters']})" 
                           for _, row in top_strategies.iterrows()])
        ax4.set_xlabel('Final Score')
        ax4.set_title('Top 10 Strategies')
        ax4.grid(True, alpha=0.3)
        
        # Highlight best strategy
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(3)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'strategy_comparison_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved strategy comparison: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filepath) if save else ""
    
    def plot_geographic_clusters(self, save: bool = True, show: bool = True) -> str:
        """
        Create geographic visualization of clusters (if lat/lon available).
        
        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        df = self.clustering_data['df']
        
        # Check if geographic coordinates are available
        if 'longitude' not in df.columns or 'latitude' not in df.columns:
            print("⚠️  Geographic coordinates not available for mapping")
            return ""
        
        # Create geographic plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot points colored by cluster
        scatter = ax.scatter(df['longitude'], df['latitude'], 
                           c=df['cluster'], 
                           cmap='tab10',
                           s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add site labels for small datasets
        if len(df) <= 30:
            for _, row in df.iterrows():
                ax.annotate(row['site'], (row['longitude'], row['latitude']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Geographic Distribution of Clusters - {self.clustering_data["strategy"]["name"]}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add cluster statistics
        cluster_counts = df['cluster'].value_counts().sort_index()
        stats_text = '\n'.join([f'Cluster {i}: {count} sites' 
                               for i, count in cluster_counts.items()])
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'geographic_clusters_{timestamp}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved geographic clusters plot: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(filepath) if save else ""
    
    def create_interactive_dashboard(self, save: bool = True) -> str:
        """
        Create an interactive dashboard with multiple visualizations.
        
        Args:
            save: Whether to save the dashboard
            
        Returns:
            Path to saved HTML file
        """
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        df = self.clustering_data['df']
        features = self.clustering_data['features']
        
        # Prepare data
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA for dashboard
        pca = PCA(n_components=min(3, len(features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        if 'longitude' in df.columns and 'latitude' in df.columns:
            subplot_titles = ['PCA Feature Space', 'Geographic Distribution', 
                            'Feature Correlations', 'Cluster Statistics']
            specs = [[{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "bar"}]]
        else:
            subplot_titles = ['PCA Feature Space', 'Feature Distributions',
                            'Feature Correlations', 'Cluster Statistics']
            specs = [[{"type": "scatter"}, {"type": "histogram"}],
                    [{"type": "heatmap"}, {"type": "bar"}]]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            specs=specs
        )
        
        # Plot 1: PCA Feature Space
        fig.add_trace(
            go.Scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                mode='markers',
                marker=dict(
                    color=df['cluster'],
                    colorscale='Plotly3',
                    size=10,
                    line=dict(width=1, color='black')
                ),
                text=df['site'],
                name='Sites',
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Geographic or Feature Distribution
        if 'longitude' in df.columns and 'latitude' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['longitude'],
                    y=df['latitude'],
                    mode='markers',
                    marker=dict(
                        color=df['cluster'],
                        colorscale='Plotly3',
                        size=12,
                        line=dict(width=1, color='black')
                    ),
                    text=df['site'],
                    name='Geographic',
                    hovertemplate='<b>%{text}</b><br>Lon: %{x:.2f}<br>Lat: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
        else:
            # Show distribution of first feature by cluster
            for cluster in sorted(df['cluster'].unique()):
                cluster_data = df[df['cluster'] == cluster]
                fig.add_trace(
                    go.Histogram(
                        x=cluster_data[features[0]],
                        name=f'Cluster {cluster}',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Feature Correlation Matrix
        feature_corr = df[features].corr()
        fig.add_trace(
            go.Heatmap(
                z=feature_corr.values,
                x=feature_corr.columns,
                y=feature_corr.columns,
                colorscale='RdBu',
                zmid=0,
                name='Correlation'
            ),
            row=2, col=1
        )
        
        # Plot 4: Cluster Statistics
        cluster_counts = df['cluster'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                name='Site Count',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Clustering Dashboard - {self.clustering_data["strategy"]["name"]}',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', row=1, col=1)
        fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', row=1, col=1)
        
        if 'longitude' in df.columns and 'latitude' in df.columns:
            fig.update_xaxes(title_text='Longitude', row=1, col=2)
            fig.update_yaxes(title_text='Latitude', row=1, col=2)
        else:
            fig.update_xaxes(title_text=features[0], row=1, col=2)
            fig.update_yaxes(title_text='Frequency', row=1, col=2)
        
        fig.update_xaxes(title_text='Features', row=2, col=1)
        fig.update_yaxes(title_text='Features', row=2, col=1)
        fig.update_xaxes(title_text='Cluster', row=2, col=2)
        fig.update_yaxes(title_text='Number of Sites', row=2, col=2)
        
        # Save dashboard
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'clustering_dashboard_{timestamp}.html'
            filepath = self.output_dir / filename
            fig.write_html(filepath)
            print(f"💾 Saved interactive dashboard: {filepath}")
            return str(filepath)
        
        return ""
    
    def generate_visualization_report(self, include_3d: bool = True, 
                                    include_dashboard: bool = True) -> Dict[str, str]:
        """
        Generate a comprehensive set of visualizations.
        
        Args:
            include_3d: Whether to generate 3D interactive plots
            include_dashboard: Whether to generate interactive dashboard
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        print(f"\n🎨 Generating comprehensive visualization report...")
        
        if not self.clustering_data:
            raise ValueError("No clustering data set. Call set_clustering_data() first.")
        
        visualizations = {}
        
        # Generate all visualizations
        try:
            # 2D Feature spaces
            visualizations['pca_2d'] = self.plot_feature_space_2d(method='pca', show=False)
            visualizations['tsne_2d'] = self.plot_feature_space_2d(method='tsne', show=False)
            
            # Silhouette analysis
            visualizations['silhouette'] = self.plot_silhouette_analysis(show=False)
            
            # Geographic plot (if possible)
            geo_plot = self.plot_geographic_clusters(show=False)
            if geo_plot:
                visualizations['geographic'] = geo_plot
            
            # Strategy comparison (if available)
            if self.strategy_results:
                visualizations['strategy_comparison'] = self.plot_strategy_comparison(show=False)
            
            # 3D interactive plots
            if include_3d:
                visualizations['pca_3d'] = self.plot_feature_space_3d_interactive(method='pca')
                visualizations['tsne_3d'] = self.plot_feature_space_3d_interactive(method='tsne')
            
            # Interactive dashboard
            if include_dashboard:
                visualizations['dashboard'] = self.create_interactive_dashboard()
            
            print(f"\n✅ Generated {len(visualizations)} visualizations:")
            for name, path in visualizations.items():
                print(f"  📊 {name}: {Path(path).name}")
            
            # Create summary report
            self._create_summary_report(visualizations)
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            raise
        
        return visualizations
    
    def _create_summary_report(self, visualizations: Dict[str, str]):
        """Create a summary HTML report linking all visualizations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clustering Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .visualization {{ margin: 20px 0; }}
                .stats {{ display: flex; gap: 20px; }}
                .stat-box {{ background: #3498db; color: white; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🧬 Clustering Visualization Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>📊 Clustering Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <strong>Sites:</strong> {self.clustering_data['n_sites']}
                    </div>
                    <div class="stat-box">
                        <strong>Clusters:</strong> {self.clustering_data['n_clusters']}
                    </div>
                    <div class="stat-box">
                        <strong>Features:</strong> {len(self.clustering_data['features'])}
                    </div>
                    <div class="stat-box">
                        <strong>Algorithm:</strong> {self.clustering_data['strategy']['name']}
                    </div>
                    <div class="stat-box">
                        <strong>Silhouette:</strong> {self.clustering_data['strategy']['silhouette']:.3f}
                    </div>
                </div>
                
                <h3>Features Used:</h3>
                <p>{', '.join(self.clustering_data['features'])}</p>
            </div>
            
            <h2>🎨 Visualizations</h2>
        """
        
        for name, path in visualizations.items():
            filename = Path(path).name
            if filename.endswith('.html'):
                html_content += f"""
                <div class="visualization">
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <p><a href="{filename}" target="_blank">Open Interactive Visualization</a></p>
                </div>
                """
            else:
                html_content += f"""
                <div class="visualization">
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <img src="{filename}" style="max-width: 100%; height: auto;" alt="{name}">
                </div>
                """
        
        html_content += """
            </body>
            </html>
        """
        
        # Save summary report
        report_file = self.output_dir / f'clustering_report_{timestamp}.html'
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Summary report: {report_file}")


if __name__ == "__main__":
    # Demo the visualizer
    print("🎨 CLUSTERING VISUALIZER DEMO")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'site': [f'site_{i:02d}' for i in range(50)],
        'longitude': np.random.uniform(-125, -100, 50),
        'latitude': np.random.uniform(40, 50, 50),
        'elevation': np.random.uniform(100, 2000, 50),
        'mean_annual_temp': np.random.normal(10, 5, 50),
        'mean_annual_precip': np.random.uniform(300, 1500, 50)
    })
    
    # Create sample clusters
    features = ['longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip']
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(sample_data[features])
    
    # Demo visualizer
    visualizer = ClusteringVisualizer()
    
    strategy_info = {
        'name': 'K-means (3 clusters)',
        'algorithm': 'K-means',
        'silhouette': 0.543,
        'balance_ratio': 0.667
    }
    
    visualizer.set_clustering_data(sample_data, features, labels, strategy_info)
    
    # Generate sample visualizations
    print("🎯 Generating sample visualizations...")
    visualizer.plot_feature_space_2d(method='pca', show=False)
    visualizer.plot_silhouette_analysis(show=False)
    visualizer.plot_geographic_clusters(show=False)
    visualizer.create_interactive_dashboard()
    
    print("✅ Demo complete! Check the clustering_visualizations/ directory")