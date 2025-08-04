"""
Data preprocessing utilities for ecosystem clustering.

Handles feature selection, data cleaning, encoding, and validation
for clustering workflows using the flexible feature management system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .feature_definitions import FeatureManager, FeatureSet


class ClusteringDataPreprocessor:
    """
    Preprocesses data for clustering using flexible feature selection.
    
    Handles missing values, categorical encoding, feature validation,
    and data standardization for clustering algorithms.
    """
    
    def __init__(self, feature_manager: FeatureManager = None, verbose: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            feature_manager: FeatureManager instance (creates new if None)
            verbose: Whether to print detailed progress information
        """
        self.feature_manager = feature_manager or FeatureManager()
        self.verbose = verbose
        self.label_encoders = {}  # Store encoders for reproducibility
        self.scaler = None
        self.processed_features = []
        
    def log(self, message: str, indent: int = 0):
        """Log message if verbose mode is enabled"""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{message}")
    
    def analyze_data_compatibility(self, site_df: pd.DataFrame, 
                                 feature_set_name: str) -> Dict[str, any]:
        """
        Analyze how well a feature set matches the available data.
        
        Args:
            site_df: DataFrame with site-level data
            feature_set_name: Name of the feature set to analyze
            
        Returns:
            Dictionary with compatibility analysis results
        """
        feature_set = self.feature_manager.get_feature_set(feature_set_name)
        available_columns = list(site_df.columns)
        
        validation = self.feature_manager.validate_features_availability(
            feature_set, available_columns
        )
        
        if self.verbose:
            self._print_compatibility_report(validation)
        
        return validation
    
    def _print_compatibility_report(self, validation: Dict[str, any]):
        """Print a detailed compatibility report"""
        self.log(f"\n🔍 FEATURE COMPATIBILITY ANALYSIS")
        self.log(f"📊 Feature set: {validation['feature_set_name']}")
        self.log(f"🎯 Availability: {validation['total_available']}/{validation['total_requested']} "
                f"({validation['availability_ratio']:.1%})")
        
        # Numeric features
        numeric = validation['numeric']
        if numeric['requested']:
            self.log(f"\n🧮 Numeric features:")
            self.log(f"  ✅ Available ({numeric['count_available']}): {numeric['available']}", 1)
            if numeric['missing']:
                self.log(f"  ❌ Missing ({numeric['count_missing']}): {numeric['missing']}", 1)
        
        # Categorical features
        categorical = validation['categorical']
        if categorical['requested']:
            self.log(f"\n🏷️  Categorical features:")
            self.log(f"  ✅ Available ({categorical['count_available']}): {categorical['available']}", 1)
            if categorical['missing']:
                self.log(f"  ❌ Missing ({categorical['count_missing']}): {categorical['missing']}", 1)
    
    def suggest_best_feature_sets(self, site_df: pd.DataFrame, 
                                 min_availability: float = 0.7) -> List[Dict[str, any]]:
        """
        Suggest the best feature sets for the available data.
        
        Args:
            site_df: DataFrame with site-level data
            min_availability: Minimum availability ratio to consider
            
        Returns:
            List of recommended feature sets with details
        """
        available_columns = list(site_df.columns)
        recommendations = self.feature_manager.suggest_alternative_feature_sets(
            available_columns, min_availability
        )
        
        if self.verbose and recommendations:
            self.log(f"\n💡 RECOMMENDED FEATURE SETS (≥{min_availability:.0%} availability):")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                self.log(f"  {i}. {rec['name']}: {rec['availability_ratio']:.1%} "
                        f"({rec['available_features']} features)")
                self.log(f"     {rec['description']}", 1)
        
        return recommendations
    
    def prepare_clustering_data(self, site_df: pd.DataFrame, 
                              feature_set_name: str,
                              handle_missing: str = 'median',
                              min_availability: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for clustering using the specified feature set.
        
        Args:
            site_df: DataFrame with site-level data (must include 'site' column)
            feature_set_name: Name of the feature set to use
            handle_missing: How to handle missing values ('median', 'mean', 'drop', 'zero')
            min_availability: Minimum feature availability ratio to proceed
            
        Returns:
            Tuple of (processed_dataframe, list_of_final_features)
            
        Raises:
            ValueError: If feature availability is too low or data issues
        """
        self.log(f"\n🔧 PREPARING CLUSTERING DATA")
        self.log(f"📊 Feature set: {feature_set_name}")
        self.log(f"💾 Sites: {len(site_df)}")
        
        # Validate feature compatibility
        validation = self.analyze_data_compatibility(site_df, feature_set_name)
        
        if validation['availability_ratio'] < min_availability:
            self.log(f"❌ Feature availability {validation['availability_ratio']:.1%} "
                    f"< minimum {min_availability:.1%}")
            
            # Suggest alternatives
            recommendations = self.suggest_best_feature_sets(site_df, min_availability)
            if recommendations:
                recommended_names = [r['name'] for r in recommendations[:3]]
                raise ValueError(f"Feature set '{feature_set_name}' has insufficient availability. "
                               f"Try these instead: {recommended_names}")
            else:
                raise ValueError(f"No feature sets meet minimum availability of {min_availability:.1%}")
        
        # Get the feature set
        feature_set = self.feature_manager.get_feature_set(feature_set_name)
        
        # Start with site column + available numeric features
        available_numeric = validation['numeric']['available']
        available_categorical = validation['categorical']['available']
        
        if not available_numeric and not available_categorical:
            raise ValueError(f"No features available for clustering with feature set '{feature_set_name}'")
        
        # Create base clustering dataframe
        clustering_df = site_df[['site'] + available_numeric].copy()
        
        self.log(f"✅ Selected {len(available_numeric)} numeric features")
        
        # Handle missing values in numeric features
        if available_numeric:
            clustering_df = self._handle_missing_values(clustering_df, available_numeric, handle_missing)
        
        # Process categorical features
        categorical_features_encoded = []
        if available_categorical:
            clustering_df, categorical_features_encoded = self._encode_categorical_features(
                clustering_df, site_df, available_categorical
            )
        
        # Compile final feature list
        final_features = available_numeric + categorical_features_encoded
        self.processed_features = final_features
        
        self.log(f"\n✅ PREPROCESSING COMPLETE")
        self.log(f"🧮 Numeric features: {len(available_numeric)}")
        self.log(f"🏷️  Categorical features: {len(categorical_features_encoded)}")
        self.log(f"🎯 Total features: {len(final_features)}")
        self.log(f"📊 Final features: {final_features}")
        
        return clustering_df, final_features
    
    def _handle_missing_values(self, clustering_df: pd.DataFrame, 
                              numeric_features: List[str], 
                              handle_missing: str) -> pd.DataFrame:
        """Handle missing values in numeric features"""
        self.log(f"\n🔧 Handling missing values (strategy: {handle_missing})")
        
        missing_counts = {}
        for feature in numeric_features:
            missing_count = clustering_df[feature].isnull().sum()
            if missing_count > 0:
                missing_counts[feature] = missing_count
                
                if handle_missing == 'median':
                    fill_value = clustering_df[feature].median()
                elif handle_missing == 'mean':
                    fill_value = clustering_df[feature].mean()
                elif handle_missing == 'zero':
                    fill_value = 0
                elif handle_missing == 'drop':
                    # Will handle this after the loop
                    continue
                else:
                    raise ValueError(f"Unknown missing value strategy: {handle_missing}")
                
                clustering_df[feature].fillna(fill_value, inplace=True)
                self.log(f"  🔧 {feature}: filled {missing_count} missing values with {fill_value:.2f}", 1)
        
        # Handle 'drop' strategy
        if handle_missing == 'drop' and missing_counts:
            initial_rows = len(clustering_df)
            clustering_df = clustering_df.dropna(subset=numeric_features)
            dropped_rows = initial_rows - len(clustering_df)
            if dropped_rows > 0:
                self.log(f"  🗑️  Dropped {dropped_rows} rows with missing values", 1)
        
        if missing_counts:
            total_missing = sum(missing_counts.values())
            self.log(f"  📊 Total missing values handled: {total_missing}")
        else:
            self.log(f"  ✅ No missing values found")
        
        return clustering_df
    
    def _encode_categorical_features(self, clustering_df: pd.DataFrame, 
                                   site_df: pd.DataFrame, 
                                   categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Encode categorical features using LabelEncoder"""
        self.log(f"\n🏷️  Encoding categorical features")
        
        encoded_feature_names = []
        
        for cat_feature in categorical_features:
            try:
                # Get categorical data and convert to string
                cat_data = site_df[cat_feature].astype(str)
                
                # Create and fit encoder
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(cat_data)
                
                # Store encoder for potential future use
                self.label_encoders[cat_feature] = encoder
                
                # Add encoded feature to dataframe
                encoded_feature_name = f'{cat_feature}_encoded'
                clustering_df[encoded_feature_name] = encoded_values
                encoded_feature_names.append(encoded_feature_name)
                
                # Get unique classes for reporting
                unique_classes = encoder.classes_
                self.log(f"  ✅ {cat_feature} → {encoded_feature_name} "
                        f"({len(unique_classes)} categories)", 1)
                
                if len(unique_classes) <= 10:  # Show categories if not too many
                    self.log(f"     Categories: {list(unique_classes)}", 2)
                
            except Exception as e:
                self.log(f"  ❌ Failed to encode {cat_feature}: {str(e)}", 1)
                continue
        
        self.log(f"  📊 Successfully encoded {len(encoded_feature_names)} categorical features")
        
        return clustering_df, encoded_feature_names
    
    def standardize_features(self, clustering_df: pd.DataFrame, 
                           features: List[str], 
                           fit_scaler: bool = True) -> np.ndarray:
        """
        Standardize features for clustering (zero mean, unit variance).
        
        Args:
            clustering_df: DataFrame with features
            features: List of feature column names to standardize
            fit_scaler: Whether to fit a new scaler (True) or use existing (False)
            
        Returns:
            Standardized feature matrix as numpy array
        """
        self.log(f"\n📏 Standardizing features for clustering")
        
        # Extract feature matrix
        X = clustering_df[features].values
        
        if fit_scaler:
            # Fit new scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.log(f"  ✅ Fitted new StandardScaler on {X.shape[0]} samples, {X.shape[1]} features")
        else:
            # Use existing scaler
            if self.scaler is None:
                raise ValueError("No scaler available. Set fit_scaler=True to fit a new scaler.")
            X_scaled = self.scaler.transform(X)
            self.log(f"  ✅ Applied existing StandardScaler")
        
        # Report scaling statistics
        self.log(f"  📊 Feature means: {X_scaled.mean(axis=0).round(3).tolist()}")
        self.log(f"  📊 Feature stds: {X_scaled.std(axis=0).round(3).tolist()}")
        
        return X_scaled
    
    def get_preprocessing_summary(self) -> Dict[str, any]:
        """Get a summary of the preprocessing steps performed"""
        return {
            'processed_features': self.processed_features,
            'num_features': len(self.processed_features),
            'categorical_encoders': list(self.label_encoders.keys()),
            'scaler_fitted': self.scaler is not None,
            'preprocessing_complete': len(self.processed_features) > 0
        }
    
    def save_preprocessing_artifacts(self, output_dir: str, timestamp: str):
        """Save preprocessing artifacts (encoders, scaler) for reproducibility"""
        import pickle
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save label encoders
        if self.label_encoders:
            encoders_file = os.path.join(output_dir, f'label_encoders_{timestamp}.pkl')
            with open(encoders_file, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            self.log(f"💾 Saved label encoders to: {encoders_file}")
        
        # Save scaler
        if self.scaler:
            scaler_file = os.path.join(output_dir, f'feature_scaler_{timestamp}.pkl')
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.log(f"💾 Saved feature scaler to: {scaler_file}")
        
        # Save preprocessing summary
        summary = self.get_preprocessing_summary()
        summary_file = os.path.join(output_dir, f'preprocessing_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.log(f"💾 Saved preprocessing summary to: {summary_file}")


if __name__ == "__main__":
    # Demo the preprocessor
    print("🔧 CLUSTERING DATA PREPROCESSOR DEMO")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'site': ['site1', 'site2', 'site3', 'site4'],
        'longitude': [-120.5, -118.2, -122.1, -119.8],
        'latitude': [45.2, 46.1, 44.8, 45.9], 
        'elevation': [1200, 800, 1500, 950],
        'mean_annual_temp': [8.5, 12.1, 6.8, 10.2],
        'biome_code': ['temperate_forest', 'mixed_forest', 'montane', 'temperate_forest']
    })
    
    # Demo preprocessing
    preprocessor = ClusteringDataPreprocessor(verbose=True)
    
    # Analyze compatibility
    preprocessor.analyze_data_compatibility(sample_data, 'climate')
    
    # Prepare data
    processed_df, features = preprocessor.prepare_clustering_data(sample_data, 'climate')
    
    # Standardize for clustering
    X_scaled = preprocessor.standardize_features(processed_df, features)
    
    print(f"\n✅ Demo complete! Processed {X_scaled.shape[0]} sites with {X_scaled.shape[1]} features")