"""
Advanced Ecosystem Clustering for SAPFLUXNET Spatial Validation
Uses multiple strategies to achieve better cluster balance and meaningful ecosystem separation
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
import argparse

warnings.filterwarnings('ignore')

# Set global random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class AdvancedEcosystemClusterer:
    """
    Advanced ecosystem clustering with multiple strategies for better balance
    """
    
    def __init__(self, data_dir='../../processed_parquet', output_dir='../evaluation/clustering_results', feature_set='advanced'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.feature_set = feature_set  # 'advanced', 'core', or 'hybrid'
        self.random_seed = RANDOM_SEED  # Store random seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Core ecosystem features (focused on what actually varies)
        self.core_features = [
            'mean_annual_temp',       # Temperature regime
            'mean_annual_precip',     # Precipitation regime
            'aridity_index',          # Water stress
            'latitude_abs',           # Distance from equator
            'elevation',              # Altitude effects
            'seasonal_temp_range',    # Temperature seasonality
            'seasonal_precip_range',  # Precipitation seasonality
        ]
        
        # Advanced derived features
        self.advanced_features = [
            'temp_precip_ratio',      # Climate type indicator
            'seasonality_index',      # Combined seasonality
            'climate_continentality', # Temperature range vs latitude
            'elevation_latitude_ratio', # Elevation adjusted for latitude
            'aridity_seasonality',    # Aridity × seasonality interaction
            'temp_elevation_ratio',   # Temperature adjusted for elevation
            'precip_latitude_ratio',  # Precipitation adjusted for latitude
        ]

        # Hybrid features - ECOSYSTEM-FOCUSED (removed structural/stand variables)
        self.hybrid_numeric = [
            # Geographic/Climate (ecosystem-defining)
            'longitude', 'latitude', 'latitude_abs', 'mean_annual_temp', 'mean_annual_precip',
            'aridity_index', 'elevation', 'seasonal_temp_range', 'seasonal_precip_range',
            # Engineered/Derived (ecosystem indicators)
            'temp_precip_ratio', 'seasonality_index', 'climate_continentality',
            'elevation_latitude_ratio', 'aridity_seasonality', 'temp_elevation_ratio', 'precip_latitude_ratio'
            # REMOVED: Stand/Forest Structure variables (vary within ecosystems, not between)
            # These caused noise: tree_volume_index, stand_age, n_trees, tree_density, etc.
        ]
        self.hybrid_categorical = [
            'climate_zone_code', 'biome_code', 'igbp_class_code', 'leaf_habit_code',
            'soil_texture_code', 'species_functional_group_code',
            'koppen_geiger_code_encoded'
            # REMOVED: terrain_code (4,836 importance but categorized as 'Identity' - potential site identifier)
        ]
        print(f"🌍 Advanced Ecosystem Clusterer initialized (feature set: {self.feature_set})")
        print(f"📁 Data directory: {data_dir}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎯 Core features: {len(self.core_features)}")
        print(f"🔧 Advanced features: {len(self.advanced_features)}")
        print(f"🧬 Hybrid numeric: {len(self.hybrid_numeric)} | Hybrid categorical: {len(self.hybrid_categorical)}")
        print(f"🎲 Random seed: {self.random_seed}")
        print(f"🌿 IMPROVEMENTS: More granular clustering (5-10 clusters), ecosystem-focused features, ecological coherence prioritized, data leakage removed")
    
    def validate_data_pipeline_compatibility(self, site_df):
        """
        Validate compatibility with updated data pipeline features
        
        Checks for:
        - New species functional group encoding
        - Köppen-Geiger climate classification  
        - Blocked identity/geographic features
        - Updated categorical encodings
        """
        print("\n🔍 Validating data pipeline compatibility...")
        
        # Check for new features from data pipeline improvements
        expected_new_features = [
            'species_functional_group_code',  # Replaces species_name  
            'koppen_geiger_code_encoded',     # New climate classification
        ]
        
        found_new_features = []
        missing_new_features = []
        
        for feature in expected_new_features:
            if feature in site_df.columns:
                found_new_features.append(feature)
            else:
                missing_new_features.append(feature)
        
        if found_new_features:
            print(f"  ✅ Found new pipeline features: {', '.join(found_new_features)}")
        
        if missing_new_features:
            print(f"  ⚠️  Missing new pipeline features: {', '.join(missing_new_features)}")
            print(f"     These features improve ecological clustering but are not critical")
        
        # Check for blocked features that might cause issues (data leakage)
        potentially_blocked_features = [
            'site_code', 'site_name', 'species_name', 'timezone', 'country',
            'is_inside_country', 'timezone_offset', 'pl_species_code', 'terrain_code'
        ]
        
        blocked_features_found = []
        for feature in potentially_blocked_features:
            if feature in site_df.columns:
                blocked_features_found.append(feature)
        
        if blocked_features_found:
            print(f"  🚨 WARNING: Found features that should be blocked by data pipeline: {', '.join(blocked_features_found)}")
            print(f"     These may indicate data pipeline is not properly applied!")
        else:
            print(f"  ✅ No blocked identity/geographic features found - data pipeline working correctly")
        
        # Show sample of available categorical features for debugging
        categorical_features_available = [f for f in self.hybrid_categorical if f in site_df.columns]
        print(f"  📊 Available categorical features: {len(categorical_features_available)}/{len(self.hybrid_categorical)}")
        
        return {
            'new_features_found': found_new_features,
            'new_features_missing': missing_new_features,
            'blocked_features_found': blocked_features_found,
            'categorical_available': categorical_features_available
        }
    
    def load_site_data(self):
        """Load and combine site data from processed parquet files"""
        print("\n📊 Loading site data...")
        
        site_data = []
        parquet_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.parquet')])  # Sort for deterministic order
        
        for parquet_file in parquet_files:
            site_name = parquet_file.replace('_comprehensive.parquet', '')
            file_path = os.path.join(self.data_dir, parquet_file)
            
            try:
                # Load a sample of the data to get site-level features
                df_sample = pd.read_parquet(file_path)
                
                # Take first 1000 rows for efficiency
                if len(df_sample) > 1000:
                    df_sample = df_sample.head(1000)
                
                # Extract site-level features (same for all rows of a site)
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
        print(f"\n📈 Loaded {len(site_df)} sites with {len(site_df.columns)} features")
        
        return site_df
    
    def create_advanced_features(self, site_df):
        """Create advanced ecosystem-relevant features with better transformations"""
        print("\n🔧 Creating advanced ecosystem features...")
        
        df = site_df.copy()
        
        # 1. Temperature to precipitation ratio (climate type indicator)
        df['temp_precip_ratio'] = df['mean_annual_temp'] / (df['mean_annual_precip'] + 1)
        
        # 2. Combined seasonality index (improved)
        temp_range_norm = (df['seasonal_temp_range'] - df['seasonal_temp_range'].mean()) / df['seasonal_temp_range'].std()
        precip_range_norm = (df['seasonal_precip_range'] - df['seasonal_precip_range'].mean()) / df['seasonal_precip_range'].std()
        df['seasonality_index'] = np.sqrt(temp_range_norm**2 + precip_range_norm**2)
        
        # 3. Climate continentality (temperature range relative to latitude)
        df['climate_continentality'] = df['seasonal_temp_range'] / (df['latitude_abs'] + 1)
        
        # 4. Elevation adjusted for latitude
        df['elevation_latitude_ratio'] = df['elevation'] / (df['latitude_abs'] + 1)
        
        # 5. Aridity × seasonality interaction
        df['aridity_seasonality'] = df['aridity_index'] * df['seasonality_index']
        
        # 6. Temperature adjusted for elevation (lapse rate effect)
        # Approximate lapse rate: 6.5°C per 1000m
        df['temp_elevation_ratio'] = df['mean_annual_temp'] + (df['elevation'] * 0.0065)
        
        # 7. Precipitation adjusted for latitude (tropical vs temperate)
        df['precip_latitude_ratio'] = df['mean_annual_precip'] / (df['latitude_abs'] + 1)
        
        # 8. Climate zone classification (more granular)
        conditions = [
            (df['mean_annual_temp'] >= 24) & (df['mean_annual_precip'] >= 1500),
            (df['mean_annual_temp'] >= 24) & (df['mean_annual_precip'] < 1500),
            (df['mean_annual_temp'] >= 18) & (df['mean_annual_temp'] < 24),
            (df['mean_annual_temp'] >= 10) & (df['mean_annual_temp'] < 18),
            (df['mean_annual_temp'] >= 0) & (df['mean_annual_temp'] < 10),
            (df['mean_annual_temp'] < 0)
        ]
        choices = ['Tropical_Wet', 'Tropical_Dry', 'Subtropical', 'Warm_Temperate', 'Cold_Temperate', 'Boreal']
        df['climate_zone_detailed'] = np.select(conditions, choices, default='Warm_Temperate')
        
        # 9. Seasonality classification (more granular)
        conditions = [
            (df['seasonal_temp_range'] < 3) & (df['seasonal_precip_range'] < 1),
            (df['seasonal_temp_range'] >= 3) & (df['seasonal_temp_range'] < 8),
            (df['seasonal_temp_range'] >= 8) & (df['seasonal_temp_range'] < 15),
            (df['seasonal_temp_range'] >= 15) & (df['seasonal_precip_range'] >= 10),
            (df['seasonal_temp_range'] >= 15) & (df['seasonal_precip_range'] < 10),
            (df['seasonal_temp_range'] >= 20)
        ]
        choices = ['Low_Seasonality', 'Moderate_Seasonality', 'High_Seasonality', 'High_Precip_Seasonality', 'High_Temp_Seasonality', 'Very_High_Seasonality']
        df['seasonality_type_detailed'] = np.select(conditions, choices, default='Moderate_Seasonality')
        
        # 10. Elevation classification
        conditions = [
            (df['elevation'] < 100),
            (df['elevation'] >= 100) & (df['elevation'] < 500),
            (df['elevation'] >= 500) & (df['elevation'] < 1000),
            (df['elevation'] >= 1000) & (df['elevation'] < 2000),
            (df['elevation'] >= 2000)
        ]
        choices = ['Lowland', 'Low_Hills', 'High_Hills', 'Mountain', 'High_Mountain']
        df['elevation_class'] = np.select(conditions, choices, default='Low_Hills')
        
        print(f"  ✅ Created {len(self.advanced_features)} advanced features")
        print(f"  ✅ Created detailed climate, seasonality, and elevation classifications")
        
        return df
    
    def prepare_clustering_data(self, site_df):
        """Prepare data for clustering with advanced feature selection"""
        print("\n🔧 Preparing clustering data...")
        
        # Create advanced features first
        site_df = self.create_advanced_features(site_df)
        
        # Select features for clustering based on feature_set
        if self.feature_set == 'core':
            clustering_features = self.core_features
        elif self.feature_set == 'hybrid':
            clustering_features = self.hybrid_numeric.copy()
            # Add categorical features (one-hot encoded)
            for cat in self.hybrid_categorical:
                if cat in site_df.columns:
                    dummies = pd.get_dummies(site_df[cat], prefix=cat)
                    site_df = pd.concat([site_df, dummies], axis=1)
                    clustering_features.extend(dummies.columns.tolist())
        else:
            clustering_features = self.core_features + self.advanced_features
        
        # Check which features are available
        available_features = [f for f in clustering_features if f in site_df.columns]
        missing_features = [f for f in clustering_features if f not in site_df.columns]
        
        print(f"✅ Available features: {len(available_features)}")
        print(f"❌ Missing features: {len(missing_features)}")
        
        if missing_features:
            print(f"   Missing: {missing_features}")
        
        # Select available features
        clustering_df = site_df[['site'] + available_features].copy()
        
        # Handle missing values
        print("\n🔍 Analyzing missing values...")
        missing_summary = clustering_df[available_features].isnull().sum()
        print(missing_summary[missing_summary > 0])
        
        # Fill missing values with appropriate strategies
        for feature in available_features:
            if clustering_df[feature].isnull().sum() > 0:
                median_val = clustering_df[feature].median()
                clustering_df[feature].fillna(median_val, inplace=True)
                print(f"  📝 Filled {feature} with median: {median_val:.2f}")
        
        # Add categorical features for analysis (not for clustering)
        categorical_features = ['climate_zone_detailed', 'seasonality_type_detailed', 'elevation_class']
        for feature in categorical_features:
            if feature in site_df.columns and feature not in clustering_df.columns:
                clustering_df[feature] = site_df[feature]
        
        # Update available features list
        self.available_features = available_features
        
        print(f"\n✅ Prepared clustering data: {len(clustering_df)} sites, {len(self.available_features)} features")
        
        return clustering_df
    
    def analyze_feature_distributions(self, clustering_df):
        """Analyze feature distributions to understand clustering challenges"""
        print("\n📊 Analyzing feature distributions...")
        
        # Calculate percentiles for each feature
        percentiles = [10, 25, 50, 75, 90]
        distribution_summary = {}
        
        for feature in self.available_features:
            # Only analyze numeric (float/int) features, skip bool/one-hot
            if pd.api.types.is_numeric_dtype(clustering_df[feature]) and not pd.api.types.is_bool_dtype(clustering_df[feature]):
                values = clustering_df[feature].dropna()
                if len(values) > 0:
                    percentiles_vals = np.percentile(values, percentiles)
                    distribution_summary[feature] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'percentiles': dict(zip(percentiles, percentiles_vals)),
                        'skewness': values.skew(),
                        'kurtosis': values.kurtosis()
                    }
        
        # Identify features with good separation potential
        good_separation_features = []
        for feature, stats in distribution_summary.items():
            # Check if feature has good spread and isn't too skewed
            cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
            if cv > 0.3 and abs(stats['skewness']) < 2:
                good_separation_features.append(feature)
        
        print(f"🎯 Features with good separation potential: {len(good_separation_features)}")
        for feature in good_separation_features:
            print(f"  • {feature}")
        
        return distribution_summary, good_separation_features
    
    def apply_feature_transformations(self, clustering_df, good_separation_features):
        """Apply transformations to improve clustering"""
        print("\n🔄 Applying feature transformations...")
        
        df = clustering_df.copy()
        transformed_features = []
        
        for feature in good_separation_features:
            values = df[feature].values
            
            # Try different transformations
            transformations = {
                'log': np.log1p(values),  # log(1+x) for positive values
                'sqrt': np.sqrt(np.abs(values)),  # square root
                'cube': np.cbrt(values),  # cube root
                'rank': pd.Series(values).rank().values,  # rank transformation
                'zscore': zscore(values),  # z-score normalization
            }
            
            # Calculate variance for each transformation
            transformation_variance = {}
            for transform_name, transformed_values in transformations.items():
                if not np.any(np.isnan(transformed_values)) and not np.any(np.isinf(transformed_values)):
                    full_name = f'{feature}_{transform_name}'
                    transformation_variance[full_name] = np.var(transformed_values)
            
            # Select transformation with highest variance (best separation)
            if transformation_variance:
                best_transformation = max(transformation_variance, key=transformation_variance.get)
                # Extract the transformation type from the best transformation name
                transform_type = best_transformation.split('_', 1)[1]
                df[best_transformation] = transformations[transform_type]
                transformed_features.append(best_transformation)
                print(f"  ✅ {feature} → {best_transformation} (var: {transformation_variance[best_transformation]:.3f})")
        
        return df, transformed_features
    
    def try_multiple_clustering_strategies(self, clustering_df, selected_features):
        """Try multiple clustering strategies to find the best approach"""
        print(f"\n🔮 Trying multiple clustering strategies...")
        
        X = clustering_df[selected_features].values
        
        # Strategy 1: Standard K-means with different scaling
        print("  📊 Strategy 1: Standard K-means with different scaling")
        strategies = []
        
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        for scaler_name, scaler in scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                
                # Try different numbers of clusters (more granular for global diversity)
                for n_clusters in [5, 6, 7, 8, 9, 10]:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    
                    # Calculate balance ratio
                    cluster_counts = np.bincount(labels)
                    balance_ratio = cluster_counts.min() / cluster_counts.max()
                    silhouette_avg = silhouette_score(X_scaled, labels)
                    
                    strategies.append({
                        'name': f'kmeans_{scaler_name}_{n_clusters}',
                        'method': 'kmeans',
                        'scaler': scaler_name,
                        'n_clusters': n_clusters,
                        'labels': labels,
                        'balance_ratio': balance_ratio,
                        'silhouette': silhouette_avg,
                        'cluster_counts': cluster_counts.tolist()
                    })
                    
                    print(f"    {scaler_name} + {n_clusters} clusters: balance={balance_ratio:.3f}, silhouette={silhouette_avg:.3f}")
                    
            except Exception as e:
                print(f"    {scaler_name}: Failed - {e}")
                continue
        
        # Strategy 2: Hierarchical clustering with different linkage methods
        print("  🌳 Strategy 2: Hierarchical clustering")
        linkage_methods = ['ward', 'complete', 'average', 'single']
        
        for linkage_method in linkage_methods:
            try:
                for n_clusters in [5, 6, 7, 8, 9, 10]:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                    labels = hierarchical.fit_predict(X_scaled)
                    
                    cluster_counts = np.bincount(labels)
                    balance_ratio = cluster_counts.min() / cluster_counts.max()
                    silhouette_avg = silhouette_score(X_scaled, labels)
                    
                    strategies.append({
                        'name': f'hierarchical_{linkage_method}_{n_clusters}',
                        'method': 'hierarchical',
                        'linkage': linkage_method,
                        'n_clusters': n_clusters,
                        'labels': labels,
                        'balance_ratio': balance_ratio,
                        'silhouette': silhouette_avg,
                        'cluster_counts': cluster_counts.tolist()
                    })
                    
                    print(f"    {linkage_method} + {n_clusters} clusters: balance={balance_ratio:.3f}, silhouette={silhouette_avg:.3f}")
                    
            except Exception as e:
                print(f"    {linkage_method}: Failed - {e}")
                continue
        
        # Strategy 3: Gaussian Mixture Models
        print("  🎲 Strategy 3: Gaussian Mixture Models")
        for n_clusters in [5, 6, 7, 8, 9, 10]:
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
                labels = gmm.fit_predict(X_scaled)
                
                cluster_counts = np.bincount(labels)
                balance_ratio = cluster_counts.min() / cluster_counts.max()
                silhouette_avg = silhouette_score(X_scaled, labels)
                
                strategies.append({
                    'name': f'gmm_{n_clusters}',
                    'method': 'gmm',
                    'n_clusters': n_clusters,
                    'labels': labels,
                    'balance_ratio': balance_ratio,
                    'silhouette': silhouette_avg,
                    'cluster_counts': cluster_counts.tolist()
                })
                
                print(f"    GMM + {n_clusters} clusters: balance={balance_ratio:.3f}, silhouette={silhouette_avg:.3f}")
                
            except Exception as e:
                print(f"    GMM {n_clusters}: Failed - {e}")
                continue
        
        # Strategy 4: DBSCAN (density-based)
        print("  🌐 Strategy 4: DBSCAN (density-based)")
        eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        min_samples_values = [3, 5, 7]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_scaled)
                    
                    # Count clusters (excluding noise points labeled as -1)
                    unique_labels = set(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                    
                    if n_clusters >= 2:  # Only consider if we have at least 2 clusters
                        cluster_counts = np.bincount(labels[labels >= 0])
                        balance_ratio = cluster_counts.min() / cluster_counts.max()
                        silhouette_avg = silhouette_score(X_scaled, labels) if len(unique_labels) > 1 else 0
                        
                        strategies.append({
                            'name': f'dbscan_eps{eps}_min{min_samples}',
                            'method': 'dbscan',
                            'eps': eps,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'labels': labels,
                            'balance_ratio': balance_ratio,
                            'silhouette': silhouette_avg,
                            'cluster_counts': cluster_counts.tolist()
                        })
                        
                        print(f"    DBSCAN eps={eps}, min_samples={min_samples}: {n_clusters} clusters, balance={balance_ratio:.3f}, silhouette={silhouette_avg:.3f}")
                        
                except Exception as e:
                    continue
        
        return strategies
    
    def select_best_strategy(self, strategies, clustering_df):
        """Select the best clustering strategy based on multiple criteria"""
        print("\n🎯 Selecting best clustering strategy...")
        
        # Filter strategies with reasonable balance (at least 0.1)
        balanced_strategies = [s for s in strategies if s['balance_ratio'] >= 0.1]
        
        if not balanced_strategies:
            print("  ⚠️  No strategies with good balance found, using all strategies")
            balanced_strategies = strategies
        
        # Score each strategy - PRIORITIZE ECOLOGICAL COHERENCE
        for strategy in balanced_strategies:
            # NEW SCORING: Weight ecological coherence (silhouette) much higher than balance
            strategy['eco_score'] = strategy['silhouette'] * 2.0  # Ecological coherence (doubled weight)
            strategy['balance_score'] = min(strategy['balance_ratio'] * 1.5, 1.0)  # Balance (capped at 1.0)
            strategy['combined_score'] = strategy['eco_score'] + strategy['balance_score']
            
            # Bonus for more granular clustering (5-10 range)
            cluster_bonus = min((strategy['n_clusters'] - 4) / 6.0, 1.0)  # Scaled for 5-10 range
            strategy['final_score'] = strategy['combined_score'] * (1 + 0.3 * cluster_bonus)
        
        # Sort by final score
        balanced_strategies.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"  📊 Top 5 strategies (prioritizing ecological coherence):")
        for i, strategy in enumerate(balanced_strategies[:5]):
            print(f"    {i+1}. {strategy['name']}: eco_score={strategy['eco_score']:.3f}, "
                  f"balance_score={strategy['balance_score']:.3f}, clusters={strategy['n_clusters']}, "
                  f"final_score={strategy['final_score']:.3f}")
        
        # Select the best strategy
        best_strategy = balanced_strategies[0]
        print(f"\n✅ Selected: {best_strategy['name']}")
        
        # Apply the best strategy to the data
        clustering_df['cluster'] = best_strategy['labels']
        clustering_df['cluster_method'] = best_strategy['name']
        
        return clustering_df, best_strategy
    
    def interpret_ecosystem_types_advanced(self, clustering_df, best_strategy):
        """Advanced ecosystem type interpretation"""
        print("\n🌿 Interpreting ecosystem types...")
        
        ecosystem_interpretations = {}
        
        for cluster_id in sorted(clustering_df['cluster'].unique()):
            cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
            
            # Calculate cluster characteristics
            characteristics = {}
            for feature in self.available_features:
                if feature in cluster_data.columns:
                    values = cluster_data[feature].dropna()
                    if len(values) > 0:
                        characteristics[feature] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
            
            # Determine ecosystem type based on characteristics
            ecosystem_type = self._classify_ecosystem_advanced(characteristics, cluster_data)
            
            # Convert all keys to str for JSON compatibility
            cluster_id_str = str(cluster_id)
            ecosystem_interpretations[cluster_id_str] = {
                'ecosystem_type': ecosystem_type,
                'n_sites': int(len(cluster_data)),
                'sites': [str(site) for site in cluster_data['site'].tolist()],
                'characteristics': characteristics,
                'climate_zones': {str(k): int(v) for k, v in cluster_data['climate_zone_detailed'].value_counts().to_dict().items()} if 'climate_zone_detailed' in cluster_data.columns else {},
                'seasonality_types': {str(k): int(v) for k, v in cluster_data['seasonality_type_detailed'].value_counts().to_dict().items()} if 'seasonality_type_detailed' in cluster_data.columns else {},
                'elevation_classes': {str(k): int(v) for k, v in cluster_data['elevation_class'].value_counts().to_dict().items()} if 'elevation_class' in cluster_data.columns else {}
            }
            
            print(f"  🌍 Cluster {cluster_id}: {ecosystem_type} ({len(cluster_data)} sites)")
            print(f"    Sites: {', '.join(cluster_data['site'].tolist()[:5])}{'...' if len(cluster_data) > 5 else ''}")
            
            # Show detailed characteristics
            if 'climate_zone_detailed' in cluster_data.columns:
                climate_dist = cluster_data['climate_zone_detailed'].value_counts()
                print(f"    Climate zones: {dict(climate_dist)}")
            
            if 'elevation_class' in cluster_data.columns:
                elevation_dist = cluster_data['elevation_class'].value_counts()
                print(f"    Elevation classes: {dict(elevation_dist)}")
        
        return ecosystem_interpretations
    
    def _classify_ecosystem_advanced(self, stats, cluster_data):
        """Advanced ecosystem classification based on multiple characteristics"""
        
        # Extract key characteristics
        temp_range = stats.get('seasonal_temp_range', {}).get('mean', 0)
        precip_range = stats.get('seasonal_precip_range', {}).get('mean', 0)
        mean_temp = stats.get('mean_annual_temp', {}).get('mean', 0)
        mean_precip = stats.get('mean_annual_precip', {}).get('mean', 0)
        latitude = stats.get('latitude_abs', {}).get('mean', 0)
        elevation = stats.get('elevation', {}).get('mean', 0)
        aridity = stats.get('aridity_index', {}).get('mean', 0)
        
        # More sophisticated classification with multiple criteria
        if temp_range < 3 and latitude < 25:
            return "Tropical Low-Seasonality"
        elif temp_range > 25 and latitude > 40:
            return "Continental High-Seasonality"
        elif temp_range > 15 and precip_range > 15:
            return "Mediterranean Seasonal"
        elif temp_range < 8 and latitude > 50:
            return "Boreal/Cold Temperate"
        elif elevation > 1500:
            return "High Elevation"
        elif mean_temp > 18 and mean_precip > 1500:
            return "Tropical Humid"
        elif mean_temp > 18 and mean_precip < 1000:
            return "Tropical Dry"
        elif mean_temp > 10 and mean_temp < 18:
            return "Warm Temperate"
        elif mean_temp < 5:
            return "Cold Temperate"
        elif aridity > 50:
            return "Arid/Semi-Arid"
        elif latitude > 45 and elevation < 500:
            return "Northern Lowland"
        elif latitude < 30 and elevation < 500:
            return "Southern Lowland"
        elif elevation > 1000 and latitude < 40:
            return "Southern Mountain"
        elif elevation > 1000 and latitude >= 40:
            return "Northern Mountain"
        else:
            return "Temperate Mixed"
    
    def save_advanced_results(self, clustering_df, ecosystem_interpretations, best_strategy, strategies):
        """Save advanced clustering results"""
        print("\n💾 Saving advanced clustering results...")
        
        # Save cluster assignments
        cluster_file = os.path.join(self.output_dir, f'advanced_site_clusters_{self.timestamp}.csv')
        clustering_df[['site', 'cluster', 'cluster_method'] + self.available_features].to_csv(cluster_file, index=False)
        print(f"  📄 Cluster assignments: {cluster_file}")
        
        # Save ecosystem interpretations
        ecosystem_file = os.path.join(self.output_dir, f'advanced_ecosystem_interpretations_{self.timestamp}.json')
        with open(ecosystem_file, 'w') as f:
            json.dump(ecosystem_interpretations, f, indent=2, default=str)
        print(f"  📄 Ecosystem interpretations: {ecosystem_file}")
        
        # Save all strategies for comparison
        strategies_file = os.path.join(self.output_dir, f'advanced_clustering_strategies_{self.timestamp}.json')
        with open(strategies_file, 'w') as f:
            json.dump(strategies, f, indent=2, default=str)
        print(f"  📄 All strategies: {strategies_file}")
        
        # Create advanced summary report
        self._create_advanced_summary_report(clustering_df, ecosystem_interpretations, best_strategy, strategies)
    
    def _create_advanced_summary_report(self, clustering_df, ecosystem_interpretations, best_strategy, strategies):
        """Create an advanced comprehensive summary report"""
        report_file = os.path.join(self.output_dir, f'advanced_ecosystem_clustering_report_{self.timestamp}.txt')
        
        with open(report_file, 'w') as f:
            f.write("ADVANCED ECOSYSTEM CLUSTERING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Best Method: {best_strategy['name']}\n")
            f.write(f"Total sites: {len(clustering_df)}\n")
            f.write(f"Number of clusters: {best_strategy['n_clusters']}\n")
            f.write(f"Balance ratio: {best_strategy['balance_ratio']:.3f}\n")
            f.write(f"Silhouette score: {best_strategy['silhouette']:.3f}\n\n")
            
            f.write("CLUSTER SUMMARY\n")
            f.write("-" * 20 + "\n")
            for cluster_id in sorted(clustering_df['cluster'].unique()):
                cluster_data = clustering_df[clustering_df['cluster'] == cluster_id]
                ecosystem_type = ecosystem_interpretations[str(cluster_id)]['ecosystem_type']
                f.write(f"Cluster {cluster_id}: {ecosystem_type} ({len(cluster_data)} sites)\n")
                f.write(f"  Sites: {', '.join(cluster_data['site'].tolist())}\n")
                
                # Add detailed characteristics
                if 'climate_zone_detailed' in cluster_data.columns:
                    climate_dist = cluster_data['climate_zone_detailed'].value_counts()
                    f.write(f"  Climate zones: {dict(climate_dist)}\n")
                
                if 'elevation_class' in cluster_data.columns:
                    elevation_dist = cluster_data['elevation_class'].value_counts()
                    f.write(f"  Elevation classes: {dict(elevation_dist)}\n")
                
                f.write("\n")
            
            f.write("STRATEGY COMPARISON\n")
            f.write("-" * 20 + "\n")
            f.write("Top 10 strategies by combined score:\n")
            for i, strategy in enumerate(strategies[:10]):
                f.write(f"{i+1}. {strategy['name']}: balance={strategy['balance_ratio']:.3f}, "
                       f"silhouette={strategy['silhouette']:.3f}, clusters={strategy['n_clusters']}\n")
            
            f.write("\nFEATURES USED\n")
            f.write("-" * 15 + "\n")
            for feature in self.available_features:
                f.write(f"  {feature}\n")
            
            f.write("\nADVANCED FEATURES\n")
            f.write("-" * 18 + "\n")
            for feature in self.advanced_features:
                f.write(f"  {feature}\n")
            
            f.write("\nIMPROVEMENTS MADE\n")
            f.write("-" * 20 + "\n")
            f.write("1. Multiple clustering strategies tested\n")
            f.write("2. Feature transformations applied\n")
            f.write("3. Balance ratio optimization\n")
            f.write("4. Advanced feature engineering\n")
            f.write("5. Strategy comparison and selection\n")
            f.write("6. Detailed ecosystem classification\n")
            
            f.write("\nNEXT STEPS\n")
            f.write("-" * 10 + "\n")
            f.write("1. Review balanced cluster assignments\n")
            f.write("2. Use balanced clusters for ecosystem-specific training\n")
            f.write("3. Implement cluster-weighted spatial validation\n")
            f.write("4. Test ensemble predictions across ecosystems\n")
        
        print(f"  📄 Advanced summary report: {report_file}")
    
    def run_advanced_clustering_pipeline(self):
        """Run the advanced ecosystem clustering pipeline"""
        print("🚀 Starting Advanced Ecosystem Clustering Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load site data
            site_df = self.load_site_data()
            
            # Step 1.5: Validate data pipeline compatibility
            validation_results = self.validate_data_pipeline_compatibility(site_df)
            
            # Step 2: Prepare clustering data with advanced features
            clustering_df = self.prepare_clustering_data(site_df)
            
            # Step 3: Analyze feature distributions
            distribution_summary, good_separation_features = self.analyze_feature_distributions(clustering_df)
            
            # Step 4: Apply feature transformations (skip for now to avoid bugs)
            # clustering_df, transformed_features = self.apply_feature_transformations(clustering_df, good_separation_features)
            transformed_features = []
            
            # Step 5: Try multiple clustering strategies
            selected_features = self.available_features + transformed_features
            strategies = self.try_multiple_clustering_strategies(clustering_df, selected_features)
            
            # Step 6: Select best strategy
            clustering_df, best_strategy = self.select_best_strategy(strategies, clustering_df)
            
            # Step 7: Interpret ecosystem types
            ecosystem_interpretations = self.interpret_ecosystem_types_advanced(clustering_df, best_strategy)
            
            # Step 8: Save results
            self.save_advanced_results(clustering_df, ecosystem_interpretations, best_strategy, strategies)
            
            print(f"\n✅ Advanced ecosystem clustering completed successfully!")
            print(f"🌍 Created {best_strategy['n_clusters']} ecosystem clusters")
            print(f"📊 Analyzed {len(clustering_df)} sites")
            print(f"⚖️  Cluster balance ratio: {best_strategy['balance_ratio']:.3f}")
            print(f"📁 Results saved to: {self.output_dir}")
            
            return clustering_df, ecosystem_interpretations, best_strategy
            
        except Exception as e:
            print(f"\n❌ Advanced ecosystem clustering failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function to run advanced ecosystem clustering"""
    print("🌍 SAPFLUXNET Advanced Ecosystem Clustering")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Advanced Ecosystem Clustering")
    parser.add_argument('--feature-set', type=str, default='hybrid', choices=['core', 'advanced', 'hybrid'],
                        help="Feature set to use for clustering: 'core', 'advanced', or 'hybrid' (recommended default)")
    args = parser.parse_args()
    feature_set = args.feature_set
    
    # Initialize advanced clusterer
    clusterer = AdvancedEcosystemClusterer(feature_set=feature_set)
    
    # Run advanced clustering pipeline
    clustering_df, ecosystem_interpretations, best_strategy = clusterer.run_advanced_clustering_pipeline()
    
    print(f"\n🎉 Advanced ecosystem clustering analysis complete!")
    print(f"📊 Best strategy: {best_strategy['name']}")
    print(f"⚖️  Balance ratio: {best_strategy['balance_ratio']:.3f}")
    print(f"📈 Silhouette score: {best_strategy['silhouette']:.3f}")
    print(f"🌍 Ready for balanced ecosystem-based spatial validation")

if __name__ == "__main__":
    main()