#!/usr/bin/env python3
"""
Advanced Ecosystem Clustering for SAPFLUXNET Spatial Validation
Uses ecological features from v2 pipeline with site split functionality
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import argparse

warnings.filterwarnings('ignore')

class AdvancedEcosystemClusterer:
    """
    Advanced ecosystem clustering using ecological features with site split support
    """
    
    def __init__(self, data_dir='../../processed_parquet', 
                 output_dir='../evaluation/clustering_results', 
                 feature_set='hybrid',
                 site_split_file=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.feature_set = feature_set
        self.site_split_file = site_split_file
        
        # Load site split for train-only clustering
        self.train_sites = None
        self.test_sites = None
        if site_split_file:
            self.load_site_split(site_split_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Geographic features for minimal clustering (available in all datasets)
        self.geographic_numeric = [
            'longitude', 'latitude', 'elevation'
        ]
        
        # No categorical features for minimal approach
        self.geographic_categorical = []
        
        print(f"🌍 Advanced Ecosystem Clusterer initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Feature set: {feature_set}")
        print(f"🌍 Geographic numeric: {len(self.geographic_numeric)} | Geographic categorical: {len(self.geographic_categorical)}")
        
        if site_split_file:
            print(f"🎯 Using site split: {site_split_file}")
            print(f"📊 Train sites: {len(self.train_sites) if self.train_sites else 0}")
            print(f"📊 Test sites: {len(self.test_sites) if self.test_sites else 0}")
        else:
            print(f"⚠️  No site split provided - clustering ALL sites")
    
    def load_site_split(self, site_split_file):
        """Load train/test site split from JSON file"""
        print(f"📂 Loading site split from: {site_split_file}")
        
        if not os.path.exists(site_split_file):
            raise FileNotFoundError(f"Site split file not found: {site_split_file}")
        
        with open(site_split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_sites = set(split_data['train_sites'])
        self.test_sites = set(split_data['test_sites'])
        
        print(f"✅ Loaded site split:")
        print(f"  📊 Train sites: {len(self.train_sites)}")
        print(f"  📊 Test sites: {len(self.test_sites)}")
        print(f"  📅 Split created: {split_data['metadata']['timestamp']}")
        print(f"  🎲 Random seed: {split_data['metadata']['random_seed']}")
    
    def load_site_data(self):
        """Load and combine site data from processed parquet files"""
        print("\n📊 Loading site data...")
        
        site_data = []
        parquet_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.parquet')])
        
        for parquet_file in parquet_files:
            site_name = parquet_file.replace('_comprehensive.parquet', '')
            
            # Load ALL sites for clustering (no filtering by train/test split)
            # This gives better cluster representation without data leakage
            # since we only use ecological metadata features
            
            file_path = os.path.join(self.data_dir, parquet_file)
            
            try:
                # Load a sample of the data to get site-level features
                df_sample = pd.read_parquet(file_path)
                
                # Take first row for site-level features (same for all rows of a site)
                site_features = df_sample.iloc[0].to_dict()
                
                # Add site identifier
                site_features['site'] = site_name
                
                site_data.append(site_features)
                print(f"  ✅ Loaded {site_name}: {len(df_sample)} rows")
                
            except Exception as e:
                print(f"  ❌ Error loading {site_name}: {e}")
                continue
        
        # Create DataFrame
        site_df = pd.DataFrame(site_data)
        
        if self.train_sites is not None:
            train_count = len([s for s in site_df['site'] if s in self.train_sites])
            test_count = len([s for s in site_df['site'] if s in self.test_sites])
            print(f"\n📈 Loaded {len(site_df)} sites total ({train_count} train + {test_count} test)")
            print(f"🎯 Clustering ALL sites for better representation (no data leakage)")
            print(f"🔒 Train/test split will be used only for model training")
        else:
            print(f"\n📈 Loaded {len(site_df)} sites with {len(site_df.columns)} features")
        
        return site_df
    
    def prepare_clustering_data(self, site_df):
        """Prepare data for clustering with geographic features only"""
        print("\n🔧 Preparing clustering data...")
        
        # Use geographic features only (minimal approach)
        clustering_features = self.geographic_numeric.copy()
        
        # Check which features are available
        available_numeric = [f for f in self.geographic_numeric if f in site_df.columns]
        available_categorical = [f for f in self.geographic_categorical if f in site_df.columns]
        
        missing_numeric = set(self.geographic_numeric) - set(available_numeric)
        missing_categorical = set(self.geographic_categorical) - set(available_categorical)
        
        print(f"✅ Available numeric features: {len(available_numeric)}/{len(self.geographic_numeric)}")
        print(f"✅ Available categorical features: {len(available_categorical)}/{len(self.geographic_categorical)}")
        
        if missing_numeric:
            print(f"⚠️  Missing numeric features: {missing_numeric}")
        if missing_categorical:
            print(f"⚠️  Missing categorical features: {missing_categorical}")
        
        # Select available features
        clustering_df = site_df[['site'] + available_numeric].copy()
        
        # Handle missing values in numeric features
        for feature in available_numeric:
            if clustering_df[feature].isnull().sum() > 0:
                median_val = clustering_df[feature].median()
                clustering_df[feature].fillna(median_val, inplace=True)
                print(f"  🔧 Filled {feature} with median: {median_val:.2f}")
        
        # Add categorical features (encoded)
        for cat_feature in available_categorical:
            if cat_feature in site_df.columns:
                # Convert to string and encode
                cat_data = site_df[cat_feature].astype(str)
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(cat_data)
                    clustering_df[f'{cat_feature}_encoded'] = encoded
                    clustering_features.append(f'{cat_feature}_encoded')
                    print(f"  ✅ Encoded {cat_feature}")
                except Exception as e:
                    print(f"  ⚠️  Could not encode {cat_feature}: {e}")
        
        # Store available features
        self.available_features = [f for f in clustering_features if f in clustering_df.columns]
        
        print(f"\n✅ Prepared clustering data: {len(clustering_df)} sites, {len(self.available_features)} features")
        print(f"📊 Features: {self.available_features}")
        
        return clustering_df
    
    def evaluate_clustering_strategies(self, clustering_df):
        """Evaluate different clustering strategies"""
        print(f"\n🎯 Evaluating clustering strategies...")
        
        X = clustering_df[self.available_features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        strategies = []
        
        # Strategy 1: K-means
        print(f"  🎲 Strategy 1: K-means")
        for n_clusters in [3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'name': f'K-means ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels
                })
                
                print(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with {n_clusters} clusters: {e}")
                continue
        
        # Strategy 2: Agglomerative Clustering
        print(f"  🎲 Strategy 2: Agglomerative Clustering")
        for n_clusters in [3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                labels = agg.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'name': f'Agglomerative ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels
                })
                
                print(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with {n_clusters} clusters: {e}")
                continue
        
        # Strategy 3: Gaussian Mixture Models
        print(f"  🎲 Strategy 3: Gaussian Mixture Models")
        for n_clusters in [3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = gmm.fit_predict(X_scaled)
                
                silhouette = silhouette_score(X_scaled, labels)
                
                # Calculate balance ratio
                unique_labels, counts = np.unique(labels, return_counts=True)
                balance_ratio = min(counts) / max(counts)
                
                strategies.append({
                    'name': f'GMM ({n_clusters} clusters)',
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'balance_ratio': balance_ratio,
                    'labels': labels
                })
                
                print(f"    {n_clusters} clusters: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with {n_clusters} clusters: {e}")
                continue
        
        return strategies
    
    def select_best_strategy(self, strategies):
        """Select the best clustering strategy"""
        print(f"\n🏆 Selecting best clustering strategy...")
        
        if not strategies:
            raise ValueError("No successful clustering strategies found")
        
        # Filter strategies with reasonable balance (at least 0.15 for better balance)
        balanced_strategies = [s for s in strategies if s['balance_ratio'] >= 0.15]
        
        if not balanced_strategies:
            print(f"  ⚠️  No strategies with good balance (>0.15), using best available")
            balanced_strategies = sorted(strategies, key=lambda x: x['balance_ratio'], reverse=True)[:10]
        
        # Score each strategy - balance ecological coherence and cluster balance
        for strategy in balanced_strategies:
            # Balanced scoring: both silhouette and balance matter
            strategy['eco_score'] = strategy['silhouette'] * 1.5  # Ecological coherence
            strategy['balance_score'] = strategy['balance_ratio'] * 1.0  # Balance
            strategy['final_score'] = strategy['eco_score'] + strategy['balance_score']
        
        # Sort by final score
        balanced_strategies.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"  📊 Top 5 strategies (balanced scoring):")
        for i, strategy in enumerate(balanced_strategies[:5]):
            print(f"    {i+1}. {strategy['name']}: eco_score={strategy['eco_score']:.3f}, "
                  f"balance_score={strategy['balance_score']:.3f}, clusters={strategy['n_clusters']}, "
                  f"final_score={strategy['final_score']:.3f}")
        
        return balanced_strategies[0]
    
    def save_clustering_results(self, clustering_df, best_strategy):
        """Save clustering results"""
        print(f"\n💾 Saving clustering results...")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'site': clustering_df['site'],
            'cluster': best_strategy['labels']
        })
        
        # Save cluster assignments
        output_file = os.path.join(self.output_dir, f'advanced_site_clusters_{self.timestamp}.csv')
        results_df.to_csv(output_file, index=False)
        print(f"  ✅ Cluster assignments: {output_file}")
        
        # Print cluster summary
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        print(f"\n📊 Cluster Summary:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} sites")
        
        # Save strategy details
        strategy_file = os.path.join(self.output_dir, f'clustering_strategy_{self.timestamp}.json')
        strategy_info = {
            'selected_strategy': best_strategy['name'],
            'n_clusters': best_strategy['n_clusters'],
            'silhouette_score': best_strategy['silhouette'],
            'balance_ratio': best_strategy['balance_ratio'],
            'final_score': best_strategy['final_score'],
            'feature_set': self.feature_set,
            'features_used': self.available_features,
            'site_split_file': self.site_split_file,
            'train_only_clustering': False,  # Now clustering ALL sites
            'all_sites_clustering': True,    # New flag
            'train_sites_count': len(self.train_sites) if self.train_sites else None,
            'test_sites_count': len(self.test_sites) if self.test_sites else None,
            'total_sites_clustered': len(results_df),
            'timestamp': self.timestamp
        }
        
        with open(strategy_file, 'w') as f:
            json.dump(strategy_info, f, indent=2)
        
        print(f"  ✅ Strategy details: {strategy_file}")
        
        return output_file
    
    def run_clustering(self):
        """Run the complete clustering pipeline"""
        print(f"🚀 ADVANCED ECOSYSTEM CLUSTERING")
        print(f"{'='*60}")
        print(f"Started at: {datetime.now()}")
        print(f"STRATEGY: ALL-SITES clustering with geographic features only")
        print(f"FEATURES: Longitude, latitude, elevation (minimal approach)")
        print(f"APPROACH: Simple geographic clustering for minimal feature experiment")
        
        try:
            # Load site data (ALL sites for better clustering representation)
            site_df = self.load_site_data()
            
            # Prepare features
            clustering_df = self.prepare_clustering_data(site_df)
            
            # Evaluate clustering strategies
            strategies = self.evaluate_clustering_strategies(clustering_df)
            
            # Select best strategy
            best_strategy = self.select_best_strategy(strategies)
            
            # Save results
            output_file = self.save_clustering_results(clustering_df, best_strategy)
            
            print(f"\n🎉 Clustering completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            
            if self.train_sites is not None:
                train_clustered = len([s for s in clustering_df['site'] if s in self.train_sites])
                test_clustered = len([s for s in clustering_df['site'] if s in self.test_sites])
                print(f"🎯 All-sites clustering: {len(clustering_df)} sites total")
                print(f"  📊 Training sites: {train_clustered}")
                print(f"  📊 Test sites: {test_clustered}")
                print(f"🔒 Train/test split preserved for model training (no data leakage)")
            else:
                print(f"✅ Clustered {len(clustering_df)} sites (no train/test split)")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function for ecosystem clustering"""
    parser = argparse.ArgumentParser(description="Advanced Ecosystem Clustering for Ensemble Validation")
    parser.add_argument('--data-dir', default='../../processed_parquet',
                        help="Directory containing processed parquet files")
    parser.add_argument('--output-dir', default='../evaluation/clustering_results',
                        help="Directory to save clustering results")
    parser.add_argument('--feature-set', choices=['geographic'], default='geographic',
                        help="Feature set: 'geographic' (longitude, latitude, elevation only)")
    parser.add_argument('--site-split-file', 
                        help="JSON file with train/test site split (only train sites will be clustered)")
    
    args = parser.parse_args()
    
    print("🚀 ADVANCED ECOSYSTEM CLUSTERING")
    print("=" * 50)
    print(f"Feature set: {args.feature_set}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Site split file: {args.site_split_file if args.site_split_file else 'None (clustering all sites)'}")
    print(f"Started at: {datetime.now()}")
    
    try:
        clusterer = AdvancedEcosystemClusterer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            site_split_file=args.site_split_file
        )
        
        output_file = clusterer.run_clustering()
        
        if output_file:
            print(f"\n✅ Ecosystem clustering completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            if args.site_split_file:
                print(f"🎯 Ready for cluster model training")
                print(f"  📊 All sites clustered for better representation")
                print(f"  🔒 Train/test split preserved for model training")
                print(f"  ✅ No data leakage (only geographic coordinates used)")
            else:
                print(f"🎯 Ready for model training (all sites)")
        else:
            print(f"\n❌ Clustering failed")
            
    except Exception as e:
        print(f"\n❌ Clustering failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main()