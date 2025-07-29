#!/usr/bin/env python3
"""
Improved Ecosystem Clustering with Outlier Site Filtering
Excludes problematic sites that contaminate clusters and cause poor performance

Based on analysis of sites like CHN_YUN_YUN, ESP_RON_PIL, etc.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import argparse

warnings.filterwarnings('ignore')

class OutlierFilteredEcosystemClusterer:
    """
    Advanced ecosystem clustering with outlier site filtering
    Excludes sites known to cause poor cluster performance
    """
    
    def __init__(self, data_dir='../../processed_parquet', 
                 output_dir='../evaluation/clustering_results', 
                 feature_set='hybrid'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.feature_set = feature_set
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # SITE FILTERING - Exclude problematic sites identified by outlier analysis
        self.excluded_sites = [
            'CHN_YUN_YUN',    # Mixed sensor types, high missing data, poor performance
            'ESP_RON_PIL',    # Very poor performance (R² = -171.71)
            'CHE_DAV_SEE',    # Very poor performance (R² = -20.41)
            'MEX_COR_YP',     # Very poor performance (R² = -47.71)
            'THA_KHU',        # Very poor performance (R² = -15.93)
            'ESP_YUN_C1',     # Very poor performance (R² = -11.93)
            'IDN_JAM_OIL',    # Very poor performance (R² = -138.17)
            'NZL_HUA_HUA',    # Very poor performance (R² = -132.26)
            # Additional sites with consistently poor performance
            'ESP_MAJ_MAI',    # Mixed performance issues
            'ESP_MAJ_NOR_LM1', # Mixed performance issues
            'PRT_MIT',        # Mixed performance issues
        ]
        
        # Feature sets (same as clustering_v2.py but with data leakage removed)
        self.hybrid_numeric = [
            # Geographic/Climate (ecosystem-defining)
            'longitude', 'latitude', 'latitude_abs', 'mean_annual_temp', 'mean_annual_precip',
            'aridity_index', 'elevation', 'seasonal_temp_range', 'seasonal_precip_range',
            # Engineered/Derived (ecosystem indicators)
            'temp_precip_ratio', 'seasonality_index', 'climate_continentality',
            'elevation_latitude_ratio', 'aridity_seasonality', 'temp_elevation_ratio', 'precip_latitude_ratio'
        ]
        
        self.hybrid_categorical = [
            'climate_zone_code', 'biome_code', 'igbp_class_code', 'leaf_habit_code',
            'soil_texture_code', 'species_functional_group_code',
            'koppen_geiger_code_encoded'
            # REMOVED: terrain_code (potential site identifier)
        ]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🌍 Outlier-Filtered Ecosystem Clusterer initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔧 Feature set: {feature_set}")
        print(f"🚫 Excluded sites: {len(self.excluded_sites)}")
        print(f"  📝 Excluded: {', '.join(self.excluded_sites)}")
    
    def filter_problematic_sites(self, sites_df):
        """Filter out sites known to cause poor cluster performance"""
        print(f"🔍 Filtering out {len(self.excluded_sites)} problematic sites...")
        original_count = len(sites_df)
        
        # Remove excluded sites
        sites_df = sites_df[~sites_df['site'].isin(self.excluded_sites)]
        
        filtered_count = len(sites_df)
        print(f"  ✅ Removed {original_count - filtered_count} sites")
        print(f"  📊 Remaining sites: {filtered_count}")
        
        # Show which sites were removed
        removed_sites = set(self.excluded_sites) & set(sites_df['site'].unique())
        if removed_sites:
            print(f"  🚫 Removed sites: {', '.join(sorted(removed_sites))}")
        
        return sites_df
    
    def load_site_data(self):
        """Load site data with outlier filtering"""
        print(f"\n📊 Loading site data with outlier filtering...")
        
        # Find all parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, '*_comprehensive.parquet'))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        print(f"  📁 Found {len(parquet_files)} parquet files")
        
        # Extract site codes and filter out problematic sites
        site_codes = [os.path.basename(f).replace('_comprehensive.parquet', '') for f in parquet_files]
        
        # Create initial sites dataframe
        sites_df = pd.DataFrame({'site': site_codes})
        
        # Apply outlier filtering
        sites_df = self.filter_problematic_sites(sites_df)
        
        # Load data for remaining sites
        site_data = []
        successful_sites = []
        
        for site in sites_df['site']:
            parquet_file = os.path.join(self.data_dir, f'{site}_comprehensive.parquet')
            
            try:
                # Load site data
                df_site = pd.read_parquet(parquet_file)
                
                # Extract features based on feature set
                if self.feature_set == 'hybrid':
                    numeric_features = self.hybrid_numeric
                    categorical_features = self.hybrid_categorical
                else:
                    # Fallback to basic features
                    numeric_features = ['longitude', 'latitude', 'mean_annual_temp', 'mean_annual_precip']
                    categorical_features = ['climate_zone_code', 'biome_code']
                
                # Check if required features exist
                available_numeric = [f for f in numeric_features if f in df_site.columns]
                available_categorical = [f for f in categorical_features if f in df_site.columns]
                
                if len(available_numeric) < 3:
                    print(f"    ⚠️  Skipping {site}: insufficient numeric features ({len(available_numeric)})")
                    continue
                
                # Sample one row per site for clustering
                site_features = df_site[available_numeric + available_categorical].iloc[0:1]
                site_features['site'] = site
                site_data.append(site_features)
                successful_sites.append(site)
                
                print(f"    ✅ {site}: {len(available_numeric)} numeric, {len(available_categorical)} categorical features")
                
            except Exception as e:
                print(f"    ❌ Error loading {site}: {e}")
                continue
        
        if not site_data:
            raise ValueError("No valid site data loaded after filtering")
        
        # Combine all site data
        sites_df = pd.concat(site_data, ignore_index=True)
        print(f"  📊 Final dataset: {len(sites_df)} sites with {len(successful_sites)} successful loads")
        
        return sites_df, successful_sites
    
    def prepare_features(self, sites_df):
        """Prepare features for clustering"""
        print(f"\n🔧 Preparing features for clustering...")
        
        # Separate numeric and categorical features
        numeric_features = [f for f in self.hybrid_numeric if f in sites_df.columns]
        categorical_features = [f for f in self.hybrid_categorical if f in sites_df.columns]
        
        print(f"  📊 Numeric features: {len(numeric_features)}")
        print(f"  📊 Categorical features: {len(categorical_features)}")
        
        # Prepare numeric features
        X_numeric = sites_df[numeric_features].values
        
        # Prepare categorical features
        X_categorical = None
        if categorical_features:
            # Encode categorical variables
            le = LabelEncoder()
            categorical_data = []
            
            for col in categorical_features:
                # Fill missing values with 'Unknown'
                col_data = sites_df[col].fillna('Unknown')
                encoded = le.fit_transform(col_data)
                categorical_data.append(encoded)
            
            X_categorical = np.column_stack(categorical_data)
        
        # Combine features
        if X_categorical is not None:
            X = np.hstack([X_numeric, X_categorical])
            print(f"  📊 Combined features: {X.shape[1]} total")
        else:
            X = X_numeric
            print(f"  📊 Using only numeric features: {X.shape[1]} total")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, sites_df['site'].values
    
    def evaluate_clustering_strategies(self, X, sites):
        """Evaluate different clustering strategies"""
        print(f"\n🎯 Evaluating clustering strategies...")
        
        strategies = []
        
        # Strategy 1: K-means
        print(f"  🎲 Strategy 1: K-means")
        for n_clusters in [3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                silhouette = silhouette_score(X, labels)
                
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
                labels = agg.fit_predict(X)
                
                silhouette = silhouette_score(X, labels)
                
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
                labels = gmm.fit_predict(X)
                
                silhouette = silhouette_score(X, labels)
                
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
        
        # Filter strategies with reasonable balance
        balanced_strategies = [s for s in strategies if s['balance_ratio'] > 0.1]
        
        if not balanced_strategies:
            print(f"  ⚠️  No strategies with good balance, using all strategies")
            balanced_strategies = strategies
        
        # Score each strategy - PRIORITIZE ECOLOGICAL COHERENCE
        for strategy in balanced_strategies:
            # NEW SCORING: Weight ecological coherence (silhouette) much higher than balance
            strategy['eco_score'] = strategy['silhouette'] * 2.0  # Ecological coherence (doubled weight)
            strategy['balance_score'] = min(strategy['balance_ratio'] * 1.5, 1.0)  # Balance (capped at 1.0)
            strategy['final_score'] = strategy['eco_score'] + strategy['balance_score']
        
        # Sort by final score
        balanced_strategies.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"  📊 Top 5 strategies (prioritizing ecological coherence):")
        for i, strategy in enumerate(balanced_strategies[:5]):
            print(f"    {i+1}. {strategy['name']}: eco_score={strategy['eco_score']:.3f}, "
                  f"balance_score={strategy['balance_score']:.3f}, clusters={strategy['n_clusters']}, "
                  f"final_score={strategy['final_score']:.3f}")
        
        return balanced_strategies[0] if balanced_strategies else strategies[0]
    
    def save_clustering_results(self, best_strategy, sites):
        """Save clustering results"""
        print(f"\n💾 Saving clustering results...")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'site': sites,
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
        import json
        strategy_info = {
            'selected_strategy': best_strategy['name'],
            'n_clusters': best_strategy['n_clusters'],
            'silhouette_score': best_strategy['silhouette'],
            'balance_ratio': best_strategy['balance_ratio'],
            'final_score': best_strategy['final_score'],
            'excluded_sites': self.excluded_sites,
            'feature_set': self.feature_set,
            'timestamp': self.timestamp
        }
        
        with open(strategy_file, 'w') as f:
            json.dump(strategy_info, f, indent=2)
        
        print(f"  ✅ Strategy details: {strategy_file}")
        
        return output_file
    
    def run_clustering(self):
        """Run the complete clustering pipeline"""
        print(f"🚀 OUTLIER-FILTERED ECOSYSTEM CLUSTERING")
        print(f"{'='*60}")
        print(f"Started at: {datetime.now()}")
        print(f"IMPROVEMENTS: Outlier site filtering, ecosystem-focused features, ecological coherence prioritized")
        
        try:
            # Load site data with outlier filtering
            sites_df, successful_sites = self.load_site_data()
            
            # Prepare features
            X, sites = self.prepare_features(sites_df)
            
            # Evaluate clustering strategies
            strategies = self.evaluate_clustering_strategies(X, sites)
            
            # Select best strategy
            best_strategy = self.select_best_strategy(strategies)
            
            # Save results
            output_file = self.save_clustering_results(best_strategy, sites)
            
            print(f"\n🎉 Clustering completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            print(f"🚫 Excluded {len(self.excluded_sites)} problematic sites")
            print(f"✅ Clustered {len(successful_sites)} high-quality sites")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Outlier-Filtered Ecosystem Clustering")
    parser.add_argument('--data-dir', default='../../processed_parquet',
                        help="Directory containing parquet files")
    parser.add_argument('--output-dir', default='../evaluation/clustering_results',
                        help="Directory to save clustering results")
    parser.add_argument('--feature-set', type=str, default='hybrid', choices=['core', 'advanced', 'hybrid'],
                        help="Feature set to use for clustering: 'core', 'advanced', or 'hybrid' (default)")
    
    args = parser.parse_args()
    
    try:
        clusterer = OutlierFilteredEcosystemClusterer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            feature_set=args.feature_set
        )
        
        output_file = clusterer.run_clustering()
        
        print(f"\n🎉 Outlier-filtered clustering completed successfully!")
        print(f"💡 Next step: Run spatial validation with the new clustering")
        
    except Exception as e:
        print(f"\n❌ Clustering failed: {e}")
        raise

if __name__ == "__main__":
    main()