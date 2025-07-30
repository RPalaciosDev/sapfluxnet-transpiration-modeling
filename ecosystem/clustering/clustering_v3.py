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
import json
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import argparse

warnings.filterwarnings('ignore')

class PerformanceBasedEcosystemClusterer:
    """
    Performance-based ecosystem clusterer for ensemble validation
    Clusters sites based on their prediction performance characteristics
    """
    
    def __init__(self, data_dir='../../processed_parquet', 
                 output_dir='../evaluation/clustering_results', 
                 feature_set='performance',
                 site_split_file=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.feature_set = feature_set
        self.site_split_file = site_split_file
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load site split for train-only clustering
        self.train_sites = None
        self.test_sites = None
        if site_split_file:
            self.load_site_split(site_split_file)
        
        # Performance-based feature sets (NEW - aligned with ensemble plan)
        self.performance_numeric = [
            'mean_sap_flow', 'sap_flow_std', 'sap_flow_cv',  # Sap flow characteristics
            'temporal_coverage_days', 'data_completeness',   # Data quality
            'environmental_range_temp', 'environmental_range_vpd',  # Environmental range
            'prediction_difficulty_score', 'outlier_ratio'   # Prediction difficulty
        ]
        
        self.performance_categorical = [
            'climate_zone_code', 'ecosystem_type', 'species_functional_group_code'
        ]
        
        # Hybrid set combines performance + ecological features
        self.hybrid_numeric = self.performance_numeric + [
            'longitude', 'latitude', 'elevation', 'mean_annual_temp', 'mean_annual_precip'
        ]
        
        self.hybrid_categorical = self.performance_categorical + [
            'koppen_geiger_code_encoded', 'continent_code', 'biome_code'
        ]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 Performance-Based Ecosystem Clusterer initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Feature set: {feature_set}")
        print(f"📊 Performance features: {len(self.performance_numeric)} numeric, {len(self.performance_categorical)} categorical")
        print(f"🔗 Hybrid features: {len(self.hybrid_numeric)} numeric, {len(self.hybrid_categorical)} categorical")
        
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
    
    def calculate_performance_features(self, sites_df):
        """Calculate performance-based features for clustering"""
        print("\n🔧 Calculating performance-based features...")
        
        performance_features = []
        
        for site in sites_df['site'].unique():
            site_data = sites_df[sites_df['site'] == site]
            
            # Calculate sap flow characteristics
            sap_flow_data = site_data['sap_flow'].dropna()
            if len(sap_flow_data) == 0:
                continue
                
            mean_sap_flow = sap_flow_data.mean()
            sap_flow_std = sap_flow_data.std()
            sap_flow_cv = sap_flow_std / mean_sap_flow if mean_sap_flow > 0 else 0
            
            # Calculate temporal coverage
            temporal_coverage_days = len(site_data) / 24  # Assuming hourly data
            data_completeness = len(sap_flow_data) / len(site_data)
            
            # Calculate environmental range
            temp_range = site_data['air_temperature'].max() - site_data['air_temperature'].min() if 'air_temperature' in site_data.columns else 0
            vpd_range = site_data['vapor_pressure_deficit'].max() - site_data['vapor_pressure_deficit'].min() if 'vapor_pressure_deficit' in site_data.columns else 0
            
            # Calculate prediction difficulty (based on variability and outliers)
            outlier_ratio = len(sap_flow_data[sap_flow_data > sap_flow_data.quantile(0.99)]) / len(sap_flow_data)
            prediction_difficulty_score = (sap_flow_cv * 0.4 + outlier_ratio * 0.3 + (1 - data_completeness) * 0.3)
            
            performance_features.append({
                'site': site,
                'mean_sap_flow': mean_sap_flow,
                'sap_flow_std': sap_flow_std,
                'sap_flow_cv': sap_flow_cv,
                'temporal_coverage_days': temporal_coverage_days,
                'data_completeness': data_completeness,
                'environmental_range_temp': temp_range,
                'environmental_range_vpd': vpd_range,
                'prediction_difficulty_score': prediction_difficulty_score,
                'outlier_ratio': outlier_ratio
            })
        
        return pd.DataFrame(performance_features)

    def load_site_data(self):
        """Load site data - only training sites if site split provided"""
        print(f"\n📊 Loading site data for clustering...")
        
        # Find all parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, '*_comprehensive.parquet'))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        print(f"  📁 Found {len(parquet_files)} parquet files")
        
        # Extract site codes
        site_codes = [os.path.basename(f).replace('_comprehensive.parquet', '') for f in parquet_files]
        
        # Filter to training sites only if site split is provided
        if self.train_sites is not None:
            original_count = len(site_codes)
            site_codes = [site for site in site_codes if site in self.train_sites]
            print(f"  🎯 Filtered to training sites only: {len(site_codes)}/{original_count} sites")
            print(f"     (Excluding {original_count - len(site_codes)} test sites from clustering)")
        else:
            print(f"  ⚠️  No site split - considering all {len(site_codes)} sites")
        
        # Create initial sites dataframe
        sites_df = pd.DataFrame({'site': site_codes})
        
        # Load data for all sites
        site_data = []
        successful_sites = []
        
        for site in sites_df['site']:
            parquet_file = os.path.join(self.data_dir, f'{site}_comprehensive.parquet')
            
            try:
                # Load site data
                df_site = pd.read_parquet(parquet_file)
                
                # Extract features based on feature set
                if self.feature_set == 'performance':
                    numeric_features = self.performance_numeric
                    categorical_features = self.performance_categorical
                elif self.feature_set == 'hybrid':
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
        """Prepare features for clustering with performance-based features"""
        print("\n🔧 Preparing features for clustering...")
        
        # Calculate performance features if needed
        if self.feature_set == 'performance':
            print("📊 Calculating performance-based features...")
            performance_df = self.calculate_performance_features(sites_df)
            
            # Merge performance features with site metadata
            sites_df = sites_df.merge(performance_df, on='site', how='left')
            
            # Use performance features for clustering
            numeric_features = self.performance_numeric
            categorical_features = self.performance_categorical
        elif self.feature_set == 'hybrid':
            # Calculate performance features and combine with ecological
            print("📊 Calculating hybrid features (performance + ecological)...")
            performance_df = self.calculate_performance_features(sites_df)
            sites_df = sites_df.merge(performance_df, on='site', how='left')
            
            numeric_features = self.hybrid_numeric
            categorical_features = self.hybrid_categorical
        else:
            # Use existing features (fallback)
            numeric_features = self.hybrid_numeric
            categorical_features = self.hybrid_categorical
        
        print(f"📊 Numeric features: {len(numeric_features)}")
        print(f"📊 Categorical features: {len(categorical_features)}")
        
        # Check which features are available
        available_numeric = [f for f in numeric_features if f in sites_df.columns]
        available_categorical = [f for f in categorical_features if f in sites_df.columns]
        
        missing_numeric = set(numeric_features) - set(available_numeric)
        missing_categorical = set(categorical_features) - set(available_categorical)
        
        if missing_numeric:
            print(f"⚠️  Missing numeric features: {missing_numeric}")
        if missing_categorical:
            print(f"⚠️  Missing categorical features: {missing_categorical}")
        
        # Prepare numeric features
        X_numeric = sites_df[available_numeric].copy()
        
        # Handle missing values in numeric features
        for col in X_numeric.columns:
            if X_numeric[col].isnull().any():
                median_val = X_numeric[col].median()
                X_numeric[col] = X_numeric[col].fillna(median_val)
                print(f"  🔧 Filled missing values in {col} with median: {median_val:.4f}")
        
        # Prepare categorical features
        X_categorical = pd.DataFrame()
        if available_categorical:
            for col in available_categorical:
                col_data = sites_df[col].astype(str)  # Convert to string for encoding
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(col_data)
                    X_categorical[f'{col}_encoded'] = encoded
                except Exception as e:
                    print(f"  ⚠️  Could not encode {col}: {e}")
                    # Fill with zeros if encoding fails
                    X_categorical[f'{col}_encoded'] = 0
        
        # Combine features
        X_combined = pd.concat([X_numeric, X_categorical], axis=1)
        
        # Final check for any remaining NaN values
        if X_combined.isnull().any().any():
            print("  🔧 Final cleanup: filling remaining NaN values with 0")
            X_combined = X_combined.fillna(0)
        
        # Convert to numpy array and ensure no NaN values
        X = np.nan_to_num(X_combined.values)
        sites = sites_df['site'].values
        
        print(f"✅ Prepared {X.shape[1]} features for {len(sites)} sites")
        print(f"📊 Feature matrix shape: {X.shape}")
        
        return X, sites
    
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
        
        if not strategies:
            print(f"  ❌ No successful clustering strategies found!")
            print(f"  💡 This usually means all strategies failed due to data issues")
            raise ValueError("No successful clustering strategies found. Check data quality and missing values.")
        
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
        
        return balanced_strategies[0]
    
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
            'feature_set': self.feature_set,
            'site_split_file': self.site_split_file,
            'train_only_clustering': self.train_sites is not None,
            'train_sites_count': len(self.train_sites) if self.train_sites else None,
            'test_sites_count': len(self.test_sites) if self.test_sites else None,
            'timestamp': self.timestamp
        }
        
        with open(strategy_file, 'w') as f:
            json.dump(strategy_info, f, indent=2)
        
        print(f"  ✅ Strategy details: {strategy_file}")
        
        return output_file
    
    def run_clustering(self):
        """Run the complete clustering pipeline"""
        print(f"🚀 PERFORMANCE-BASED ECOSYSTEM CLUSTERING")
        print(f"{'='*60}")
        print(f"Started at: {datetime.now()}")
        print(f"STRATEGY: Let clustering algorithm decide which sites work well together")
        print(f"FEATURES: Ecosystem-focused features, ecological coherence prioritized")
        
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
            
            if self.train_sites is not None:
                print(f"🎯 Train-only clustering: {len(successful_sites)} training sites clustered")
                print(f"🔒 Test sites ({len(self.test_sites)}) withheld for ensemble validation")
            else:
                print(f"✅ Clustered {len(successful_sites)} sites (no train/test split)")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function for performance-based ecosystem clustering"""
    parser = argparse.ArgumentParser(description="Performance-Based Ecosystem Clustering for Ensemble Validation")
    parser.add_argument('--data-dir', default='../../processed_parquet',
                        help="Directory containing processed parquet files")
    parser.add_argument('--output-dir', default='../evaluation/clustering_results',
                        help="Directory to save clustering results")
    parser.add_argument('--feature-set', choices=['performance', 'hybrid'], default='performance',
                        help="Feature set: 'performance' (performance-based only), 'hybrid' (performance + ecological)")
    parser.add_argument('--site-split-file', 
                        help="JSON file with train/test site split (only train sites will be clustered)")
    
    args = parser.parse_args()
    
    print("🚀 PERFORMANCE-BASED ECOSYSTEM CLUSTERING")
    print("=" * 50)
    print(f"Feature set: {args.feature_set}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Site split file: {args.site_split_file if args.site_split_file else 'None (clustering all sites)'}")
    print(f"Started at: {datetime.now()}")
    
    try:
        clusterer = PerformanceBasedEcosystemClusterer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            site_split_file=args.site_split_file
        )
        
        output_file = clusterer.run_clustering()
        
        if output_file:
            print(f"\n✅ Performance-based clustering completed successfully!")
            print(f"📁 Results saved to: {output_file}")
            if args.site_split_file:
                print(f"🎯 Ready for cluster model training (train sites only)")
                print(f"🔒 Test sites withheld for ensemble validation")
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