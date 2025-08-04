#!/usr/bin/env python3
"""
Flexible Ecosystem Clusterer - Modular version of clustering_v3.py

This modular clusterer separates concerns and provides easy feature selection
capabilities while maintaining the same core functionality as the original.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import argparse

from .feature_definitions import FeatureManager, FeatureSet
from .data_preprocessor import ClusteringDataPreprocessor
from .clustering_visualizer import ClusteringVisualizer

warnings.filterwarnings('ignore')


class FlexibleEcosystemClusterer:
    """
    Flexible ecosystem clusterer with modular feature selection.
    
    Provides the same clustering capabilities as the original but with
    easy feature set switching and better separation of concerns.
    """
    
    def __init__(self, data_dir='../../processed_parquet', 
                 output_dir='../evaluation/clustering_results', 
                 feature_set_name='comprehensive',
                 site_split_file=None,
                 verbose=True):
        """
        Initialize the flexible clusterer.
        
        Args:
            data_dir: Directory containing processed parquet files
            output_dir: Directory to save clustering results
            feature_set_name: Name of the feature set to use for clustering
            site_split_file: JSON file with train/test site split (optional)
            verbose: Whether to print detailed progress information
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.feature_set_name = feature_set_name
        self.site_split_file = site_split_file
        self.verbose = verbose
        
        # Initialize modular components
        self.feature_manager = FeatureManager()
        self.preprocessor = ClusteringDataPreprocessor(self.feature_manager, verbose=verbose)
        self.visualizer = ClusteringVisualizer(
            output_dir=os.path.join(output_dir, 'visualizations'),
            figsize=(12, 8)
        )
        
        # Site split data
        self.train_sites = None
        self.test_sites = None
        if site_split_file:
            self.load_site_split(site_split_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        self.clustering_results = {}
        self.processed_features = []
        
        self.log(f"🌍 Flexible Ecosystem Clusterer initialized")
        self.log(f"📁 Data directory: {data_dir}")
        self.log(f"📁 Output directory: {output_dir}")
        self.log(f"🧬 Feature set: {feature_set_name}")
        
        # Show feature set details
        try:
            feature_set = self.feature_manager.get_feature_set(feature_set_name)
            self.log(f"🎯 Features: {feature_set.feature_count} total "
                    f"({len(feature_set.numeric_features)} numeric + "
                    f"{len(feature_set.categorical_features)} categorical)")
        except ValueError as e:
            available_sets = list(self.feature_manager.list_feature_sets().keys())
            self.log(f"❌ Invalid feature set: {e}")
            self.log(f"💡 Available feature sets: {available_sets}")
            raise
        
        if site_split_file:
            self.log(f"🎯 Using site split: {site_split_file}")
            self.log(f"📊 Train sites: {len(self.train_sites) if self.train_sites else 0}")
            self.log(f"📊 Test sites: {len(self.test_sites) if self.test_sites else 0}")
        else:
            self.log(f"⚠️  No site split provided - clustering ALL sites")
    
    def log(self, message: str, indent: int = 0):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{message}")
    
    def load_site_split(self, site_split_file: str):
        """Load train/test site split from JSON file"""
        self.log(f"📂 Loading site split from: {site_split_file}")
        
        if not os.path.exists(site_split_file):
            raise FileNotFoundError(f"Site split file not found: {site_split_file}")
        
        with open(site_split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_sites = set(split_data['train_sites'])
        self.test_sites = set(split_data['test_sites'])
        
        self.log(f"✅ Loaded site split:")
        self.log(f"  📊 Train sites: {len(self.train_sites)}", 1)
        self.log(f"  📊 Test sites: {len(self.test_sites)}", 1)
        self.log(f"  📅 Split created: {split_data['metadata']['timestamp']}", 1)
        self.log(f"  🎲 Random seed: {split_data['metadata']['random_seed']}", 1)
    
    def load_site_data(self) -> pd.DataFrame:
        """Load and combine site data from processed parquet files"""
        self.log("\n📊 Loading site data...")
        
        site_data = []
        parquet_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.parquet')])
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        for parquet_file in parquet_files:
            site_name = parquet_file.replace('_comprehensive.parquet', '')
            file_path = os.path.join(self.data_dir, parquet_file)
            
            try:
                # Load a sample of the data to get site-level features
                df_sample = pd.read_parquet(file_path)
                
                # Take first row for site-level features (same for all rows of a site)
                site_features = df_sample.iloc[0].to_dict()
                
                # Add site identifier
                site_features['site'] = site_name
                
                site_data.append(site_features)
                self.log(f"  ✅ Loaded {site_name}: {len(df_sample)} rows", 1)
                
            except Exception as e:
                self.log(f"  ❌ Error loading {site_name}: {e}", 1)
                continue
        
        # Create DataFrame
        site_df = pd.DataFrame(site_data)
        
        if self.train_sites is not None:
            train_count = len([s for s in site_df['site'] if s in self.train_sites])
            test_count = len([s for s in site_df['site'] if s in self.test_sites])
            self.log(f"\n📈 Loaded {len(site_df)} sites total ({train_count} train + {test_count} test)")
            self.log(f"🎯 Clustering ALL sites for better representation (no data leakage)")
            self.log(f"🔒 Train/test split will be used only for model training")
        else:
            self.log(f"\n📈 Loaded {len(site_df)} sites with {len(site_df.columns)} features")
        
        return site_df
    
    def prepare_clustering_data(self, site_df: pd.DataFrame, 
                              missing_strategy: str = 'median',
                              min_availability: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for clustering using the selected feature set.
        
        Args:
            site_df: DataFrame with site-level data
            missing_strategy: How to handle missing values ('median', 'mean', 'drop', 'zero')
            min_availability: Minimum feature availability ratio to proceed
            
        Returns:
            Tuple of (processed_dataframe, list_of_final_features)
        """
        # Use the modular preprocessor
        clustering_df, features = self.preprocessor.prepare_clustering_data(
            site_df, 
            self.feature_set_name, 
            handle_missing=missing_strategy,
            min_availability=min_availability
        )
        
        self.processed_features = features
        return clustering_df, features
    
    def evaluate_clustering_strategies(self, clustering_df: pd.DataFrame, 
                                     features: List[str],
                                     cluster_range: List[int] = None) -> List[Dict[str, any]]:
        """
        Evaluate different clustering strategies.
        
        Args:
            clustering_df: DataFrame with processed data
            features: List of feature column names to use
            cluster_range: List of cluster numbers to try (default: [3,4,5,6,7,8,9,10])
            
        Returns:
            List of strategy results
        """
        if cluster_range is None:
            cluster_range = [3, 4, 5, 6, 7, 8, 9, 10]
        
        self.log(f"\n🎯 Evaluating clustering strategies...")
        self.log(f"📊 Using {len(features)} features: {features}")
        self.log(f"🎲 Testing cluster counts: {cluster_range}")
        
        # Standardize features using the preprocessor
        X_scaled = self.preprocessor.standardize_features(clustering_df, features, fit_scaler=True)
        
        strategies = []
        
        # Strategy 1: K-means
        self.log(f"\n  🎲 Strategy 1: K-means")
        for n_clusters in cluster_range:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'algorithm': 'K-means',
                    'name': f'K-means ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels,
                    'model': kmeans
                })
                
                self.log(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}", 2)
                
            except Exception as e:
                self.log(f"    ❌ Error with {n_clusters} clusters: {e}", 2)
                continue
        
        # Strategy 2: Agglomerative Clustering
        self.log(f"\n  🎲 Strategy 2: Agglomerative Clustering")
        for n_clusters in cluster_range:
            try:
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                labels = agg.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'algorithm': 'Agglomerative',
                    'name': f'Agglomerative ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels,
                    'model': agg
                })
                
                self.log(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}", 2)
                
            except Exception as e:
                self.log(f"    ❌ Error with {n_clusters} clusters: {e}", 2)
                continue
        
        # Strategy 3: Gaussian Mixture Models
        self.log(f"\n  🎲 Strategy 3: Gaussian Mixture Models")
        for n_clusters in cluster_range:
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = gmm.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'algorithm': 'GMM',
                    'name': f'GMM ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels,
                    'model': gmm
                })
                
                self.log(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}", 2)
                
            except Exception as e:
                self.log(f"    ❌ Error with {n_clusters} clusters: {e}", 2)
                continue
        
        self.log(f"\n📊 Evaluated {len(strategies)} clustering strategies")
        return strategies
    
    def select_best_strategy(self, strategies: List[Dict[str, any]], 
                           min_balance_ratio: float = 0.15,
                           silhouette_weight: float = 1.5,
                           balance_weight: float = 1.0) -> Dict[str, any]:
        """
        Select the best clustering strategy using configurable scoring.
        
        Args:
            strategies: List of strategy results
            min_balance_ratio: Minimum balance ratio to consider
            silhouette_weight: Weight for silhouette score in final scoring
            balance_weight: Weight for balance ratio in final scoring
            
        Returns:
            Best strategy dictionary
        """
        self.log(f"\n🏆 Selecting best clustering strategy...")
        
        if not strategies:
            raise ValueError("No successful clustering strategies found")
        
        # Filter strategies with reasonable balance
        balanced_strategies = [s for s in strategies if s['balance_ratio'] >= min_balance_ratio]
        
        if not balanced_strategies:
            self.log(f"  ⚠️  No strategies with good balance (≥{min_balance_ratio:.2f}), using best available")
            balanced_strategies = sorted(strategies, key=lambda x: x['balance_ratio'], reverse=True)[:10]
        
        # Score each strategy
        for strategy in balanced_strategies:
            strategy['eco_score'] = strategy['silhouette'] * silhouette_weight
            strategy['balance_score'] = strategy['balance_ratio'] * balance_weight
            strategy['final_score'] = strategy['eco_score'] + strategy['balance_score']
        
        # Sort by final score
        balanced_strategies.sort(key=lambda x: x['final_score'], reverse=True)
        
        self.log(f"  📊 Top 5 strategies (silhouette_weight={silhouette_weight}, balance_weight={balance_weight}):")
        for i, strategy in enumerate(balanced_strategies[:5]):
            self.log(f"    {i+1}. {strategy['name']}: "
                    f"silhouette={strategy['silhouette']:.3f}, "
                    f"balance={strategy['balance_ratio']:.3f}, "
                    f"final_score={strategy['final_score']:.3f}", 2)
        
        return balanced_strategies[0]
    
    def save_clustering_results(self, clustering_df: pd.DataFrame, 
                              best_strategy: Dict[str, any]) -> str:
        """Save clustering results and metadata"""
        self.log(f"\n💾 Saving clustering results...")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'site': clustering_df['site'],
            'cluster': best_strategy['labels']
        })
        
        # Save cluster assignments
        output_file = os.path.join(self.output_dir, f'flexible_site_clusters_{self.timestamp}.csv')
        results_df.to_csv(output_file, index=False)
        self.log(f"  ✅ Cluster assignments: {output_file}")
        
        # Print cluster summary
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        self.log(f"\n📊 Cluster Summary:")
        for cluster_id, count in cluster_counts.items():
            self.log(f"  Cluster {cluster_id}: {count} sites", 1)
        
        # Save comprehensive strategy details
        strategy_file = os.path.join(self.output_dir, f'flexible_clustering_strategy_{self.timestamp}.json')
        
        # Get feature set details
        feature_set = self.feature_manager.get_feature_set(self.feature_set_name)
        
        strategy_info = {
            # Strategy results
            'selected_strategy': best_strategy['name'],
            'algorithm': best_strategy['algorithm'],
            'n_clusters': best_strategy['n_clusters'],
            'silhouette_score': best_strategy['silhouette'],
            'balance_ratio': best_strategy['balance_ratio'],
            'final_score': best_strategy['final_score'],
            
            # Feature configuration
            'feature_set_name': self.feature_set_name,
            'feature_set_description': feature_set.description,
            'features_used': self.processed_features,
            'total_features_used': len(self.processed_features),
            'numeric_features_requested': feature_set.numeric_features,
            'categorical_features_requested': feature_set.categorical_features,
            
            # Data configuration
            'site_split_file': self.site_split_file,
            'train_only_clustering': False,  # Clustering ALL sites
            'all_sites_clustering': True,
            'train_sites_count': len(self.train_sites) if self.train_sites else None,
            'test_sites_count': len(self.test_sites) if self.test_sites else None,
            'total_sites_clustered': len(results_df),
            
            # Preprocessing details
            'preprocessing_summary': self.preprocessor.get_preprocessing_summary(),
            
            # Metadata
            'timestamp': self.timestamp,
            'data_directory': self.data_dir,
            'flexible_clusterer_version': '1.0'
        }
        
        with open(strategy_file, 'w') as f:
            json.dump(strategy_info, f, indent=2, default=str)
        
        self.log(f"  ✅ Strategy details: {strategy_file}")
        
        # Save preprocessing artifacts
        self.preprocessor.save_preprocessing_artifacts(self.output_dir, self.timestamp)
        
        # Store results
        self.clustering_results = {
            'output_file': output_file,
            'strategy_file': strategy_file,
            'best_strategy': best_strategy,
            'cluster_counts': cluster_counts,
            'clustering_df': clustering_df,
            'features': self.processed_features
        }
        
        return output_file
    
    def run_clustering(self, missing_strategy: str = 'median',
                      min_availability: float = 0.5,
                      cluster_range: List[int] = None) -> Optional[str]:
        """
        Run the complete flexible clustering pipeline.
        
        Args:
            missing_strategy: How to handle missing values
            min_availability: Minimum feature availability ratio
            cluster_range: List of cluster numbers to try
            
        Returns:
            Path to output file if successful, None if failed
        """
        self.log(f"\n🚀 FLEXIBLE ECOSYSTEM CLUSTERING")
        self.log(f"{'='*60}")
        self.log(f"Started at: {datetime.now()}")
        self.log(f"STRATEGY: Modular clustering with flexible feature selection")
        self.log(f"FEATURE SET: {self.feature_set_name}")
        
        # Show feature set details
        feature_set = self.feature_manager.get_feature_set(self.feature_set_name)
        self.log(f"FEATURES: {feature_set.description}")
        self.log(f"APPROACH: Cluster all sites for better representation (no data leakage)")
        
        try:
            # Load site data
            site_df = self.load_site_data()
            
            # Prepare features using modular preprocessor
            clustering_df, features = self.prepare_clustering_data(
                site_df, missing_strategy, min_availability
            )
            
            # Evaluate clustering strategies
            strategies = self.evaluate_clustering_strategies(clustering_df, features, cluster_range)
            
            # Store all strategies for visualization comparisons
            self._all_strategies = strategies
            
            # Select best strategy
            best_strategy = self.select_best_strategy(strategies)
            
            # Save results
            output_file = self.save_clustering_results(clustering_df, best_strategy)
            
            self.log(f"\n🎉 Clustering completed successfully!")
            self.log(f"📁 Results saved to: {output_file}")
            
            if self.train_sites is not None:
                train_clustered = len([s for s in clustering_df['site'] if s in self.train_sites])
                test_clustered = len([s for s in clustering_df['site'] if s in self.test_sites])
                self.log(f"🎯 All-sites clustering: {len(clustering_df)} sites total")
                self.log(f"  📊 Training sites: {train_clustered}")
                self.log(f"  📊 Test sites: {test_clustered}")
                self.log(f"🔒 Train/test split preserved for model training (no data leakage)")
            else:
                self.log(f"✅ Clustered {len(clustering_df)} sites (no train/test split)")
            
            return output_file
            
        except Exception as e:
            self.log(f"\n❌ Clustering failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise
    
    def list_available_feature_sets(self):
        """Print all available feature sets"""
        self.log(f"\n🧬 AVAILABLE FEATURE SETS:")
        feature_sets = self.feature_manager.list_feature_sets()
        
        for name, description in feature_sets.items():
            marker = "✅" if name == self.feature_set_name else "📊"
            self.log(f"{marker} {name}")
            self.log(f"   {description}", 1)
    
    def switch_feature_set(self, new_feature_set_name: str):
        """Switch to a different feature set"""
        try:
            # Validate the new feature set exists
            new_feature_set = self.feature_manager.get_feature_set(new_feature_set_name)
            
            old_name = self.feature_set_name
            self.feature_set_name = new_feature_set_name
            
            self.log(f"🔄 Switched feature set: {old_name} → {new_feature_set_name}")
            self.log(f"📊 New feature set: {new_feature_set.description}")
            self.log(f"🎯 Features: {new_feature_set.feature_count} total "
                    f"({len(new_feature_set.numeric_features)} numeric + "
                    f"{len(new_feature_set.categorical_features)} categorical)")
            
        except ValueError as e:
            self.log(f"❌ Cannot switch feature set: {e}")
            available_sets = list(self.feature_manager.list_feature_sets().keys())
            self.log(f"💡 Available feature sets: {available_sets}")
            raise
    
    def visualize_clustering(self, strategies: List[Dict[str, Any]] = None, 
                           include_3d: bool = True, include_dashboard: bool = True,
                           show_plots: bool = False) -> Dict[str, str]:
        """
        Generate comprehensive visualizations of the clustering results.
        
        Args:
            strategies: Optional list of all strategies for comparison plots
            include_3d: Whether to generate 3D interactive plots
            include_dashboard: Whether to generate interactive dashboard
            show_plots: Whether to display plots (useful for Jupyter notebooks)
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run clustering first.")
        
        self.log(f"\n🎨 Generating clustering visualizations...")
        
        # Set up visualizer with clustering data
        clustering_df = self.clustering_results['clustering_df']
        features = self.clustering_results['features'] 
        best_strategy = self.clustering_results['best_strategy']
        
        self.visualizer.set_clustering_data(
            clustering_df=clustering_df,
            features=features,
            cluster_labels=best_strategy['labels'],
            strategy_info=best_strategy
        )
        
        # Set strategy comparison data if available
        if strategies:
            self.visualizer.set_strategy_comparison(strategies)
        
        # Generate comprehensive visualization report
        visualizations = self.visualizer.generate_visualization_report(
            include_3d=include_3d,
            include_dashboard=include_dashboard
        )
        
        self.log(f"✅ Generated {len(visualizations)} visualizations")
        for name, path in visualizations.items():
            self.log(f"  📊 {name}: {os.path.basename(path)}", 1)
        
        return visualizations
    
    def quick_visualize(self, method: str = 'pca', show: bool = True) -> str:
        """
        Generate a quick visualization of the clustering results.
        
        Args:
            method: Visualization method ('pca', 'tsne', 'geographic', 'silhouette')
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run clustering first.")
        
        # Set up visualizer with clustering data
        clustering_df = self.clustering_results['clustering_df']
        features = self.clustering_results['features']
        best_strategy = self.clustering_results['best_strategy']
        
        self.visualizer.set_clustering_data(
            clustering_df=clustering_df,
            features=features,
            cluster_labels=best_strategy['labels'],
            strategy_info=best_strategy
        )
        
        # Generate requested visualization
        if method.lower() == 'pca':
            return self.visualizer.plot_feature_space_2d(method='pca', show=show)
        elif method.lower() == 'tsne':
            return self.visualizer.plot_feature_space_2d(method='tsne', show=show)
        elif method.lower() == 'geographic':
            return self.visualizer.plot_geographic_clusters(show=show)
        elif method.lower() == 'silhouette':
            return self.visualizer.plot_silhouette_analysis(show=show)
        else:
            raise ValueError(f"Unknown visualization method: {method}. "
                           f"Available: 'pca', 'tsne', 'geographic', 'silhouette'")
    
    def create_interactive_dashboard(self) -> str:
        """
        Create an interactive dashboard for the clustering results.
        
        Returns:
            Path to saved HTML dashboard file
        """
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run clustering first.")
        
        # Set up visualizer with clustering data
        clustering_df = self.clustering_results['clustering_df']
        features = self.clustering_results['features']
        best_strategy = self.clustering_results['best_strategy']
        
        self.visualizer.set_clustering_data(
            clustering_df=clustering_df,
            features=features,
            cluster_labels=best_strategy['labels'],
            strategy_info=best_strategy
        )
        
        return self.visualizer.create_interactive_dashboard()


if __name__ == "__main__":
    # Demo the flexible clusterer
    print("🧬 FLEXIBLE ECOSYSTEM CLUSTERER DEMO")
    print("=" * 50)
    
    # List available feature sets
    from .feature_definitions import list_available_feature_sets
    list_available_feature_sets()
    
    print("\n🎯 Example usage:")
    print("clusterer = FlexibleEcosystemClusterer(feature_set_name='climate')")
    print("clusterer.run_clustering()")
    print("\n# Switch feature sets easily:")
    print("clusterer.switch_feature_set('geographic')")
    print("clusterer.run_clustering()")