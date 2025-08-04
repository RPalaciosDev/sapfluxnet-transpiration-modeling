"""
Feature definitions and selection utilities for ecosystem clustering.

This module provides flexible feature selection capabilities, allowing easy
experimentation with different feature combinations for clustering.
"""

from typing import Dict, List, Set, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class FeatureSet:
    """Represents a complete feature set for clustering"""
    name: str
    numeric_features: List[str]
    categorical_features: List[str]
    description: str
    
    @property
    def all_features(self) -> List[str]:
        """Get all features (numeric + categorical)"""
        return self.numeric_features + self.categorical_features
    
    @property
    def feature_count(self) -> int:
        """Get total feature count"""
        return len(self.all_features)


class FeatureManager:
    """
    Manages feature definitions and selection for ecosystem clustering.
    
    Provides predefined feature sets and utilities for custom feature selection.
    """
    
    def __init__(self):
        """Initialize with predefined feature sets"""
        self.feature_sets = self._initialize_feature_sets()
        self.custom_sets = {}
    
    def _initialize_feature_sets(self) -> Dict[str, FeatureSet]:
        """Initialize predefined feature sets based on processed parquet data"""
        
        return {
            # Geographic-only clustering (minimal)
            'geographic': FeatureSet(
                name='geographic',
                numeric_features=[
                    'longitude', 'latitude', 'elevation'
                ],
                categorical_features=[],
                description='Basic geographic coordinates and elevation only'
            ),
            
            # Climate-focused clustering
            'climate': FeatureSet(
                name='climate', 
                numeric_features=[
                    'longitude', 'latitude', 'elevation',
                    'mean_annual_temp', 'mean_annual_precip',
                    'seasonal_temp_range', 'seasonal_precip_range'
                ],
                categorical_features=[
                    'biome_code', 'igbp_class_code'
                ],
                description='Geographic + climate variables and biome classifications'
            ),
            
            # Ecological characteristics 
            'ecological': FeatureSet(
                name='ecological',
                numeric_features=[
                    'longitude', 'latitude', 'elevation',
                    'basal_area', 'tree_density', 'leaf_area_index'
                ],
                categorical_features=[
                    'species_functional_group_code', 'leaf_habit_code',
                    'biome_code', 'igbp_class_code'
                ],
                description='Geographic + stand characteristics and species traits'
            ),
            
            # Comprehensive (original hybrid approach)
            'comprehensive': FeatureSet(
                name='comprehensive',
                numeric_features=[
                    # Geographic/Climate features
                    'longitude', 'latitude', 'elevation', 
                    'mean_annual_temp', 'mean_annual_precip',
                    # Seasonal features
                    'seasonal_temp_range', 'seasonal_precip_range',
                    # Stand characteristics
                    'basal_area', 'tree_density', 'leaf_area_index'
                ],
                categorical_features=[
                    'species_functional_group_code', 'leaf_habit_code',
                    'biome_code', 'igbp_class_code'
                ],
                description='All available ecological and climate features'
            ),
            
            # Performance-focused (if available in processed data)
            'performance': FeatureSet(
                name='performance',
                numeric_features=[
                    'longitude', 'latitude', 'elevation',
                    'mean_annual_temp', 'mean_annual_precip',
                    # Add any aggregated performance metrics from your processed data
                    'mean_sap_flux', 'max_sap_flux', 'sap_flux_variability'  # if available
                ],
                categorical_features=[
                    'biome_code', 'igbp_class_code'
                ],
                description='Geographic + climate + aggregated performance metrics'
            ),
            
            # Environmental drivers only
            'environmental': FeatureSet(
                name='environmental',
                numeric_features=[
                    'longitude', 'latitude', 'elevation',
                    'mean_annual_temp', 'mean_annual_precip',
                    'seasonal_temp_range', 'seasonal_precip_range',
                    # Environmental features from processed data
                    'mean_ta', 'mean_rh', 'mean_vpd', 'mean_sw_in',  # if available
                    'mean_precip', 'mean_ws'  # if available
                ],
                categorical_features=[],
                description='Pure environmental/climate variables without biological traits'
            )
        }
    
    def get_feature_set(self, name: str) -> FeatureSet:
        """
        Get a feature set by name.
        
        Args:
            name: Name of the feature set
            
        Returns:
            FeatureSet object
            
        Raises:
            ValueError: If feature set name not found
        """
        if name in self.feature_sets:
            return self.feature_sets[name]
        elif name in self.custom_sets:
            return self.custom_sets[name]
        else:
            available = list(self.feature_sets.keys()) + list(self.custom_sets.keys())
            raise ValueError(f"Feature set '{name}' not found. Available: {available}")
    
    def list_feature_sets(self) -> Dict[str, str]:
        """
        List all available feature sets with descriptions.
        
        Returns:
            Dictionary mapping feature set names to descriptions
        """
        descriptions = {}
        
        # Predefined sets
        for name, feature_set in self.feature_sets.items():
            descriptions[name] = feature_set.description
            
        # Custom sets
        for name, feature_set in self.custom_sets.items():
            descriptions[f"{name} (custom)"] = feature_set.description
            
        return descriptions
    
    def create_custom_feature_set(self, name: str, numeric_features: List[str], 
                                 categorical_features: List[str] = None,
                                 description: str = None) -> FeatureSet:
        """
        Create a custom feature set.
        
        Args:
            name: Name for the custom feature set
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names (optional)
            description: Description of the feature set (optional)
            
        Returns:
            Created FeatureSet object
        """
        if categorical_features is None:
            categorical_features = []
            
        if description is None:
            description = f"Custom feature set with {len(numeric_features)} numeric and {len(categorical_features)} categorical features"
        
        feature_set = FeatureSet(
            name=name,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            description=description
        )
        
        self.custom_sets[name] = feature_set
        return feature_set
    
    def validate_features_availability(self, feature_set: FeatureSet, 
                                     available_columns: List[str]) -> Dict[str, any]:
        """
        Validate which features from a feature set are available in the data.
        
        Args:
            feature_set: FeatureSet to validate
            available_columns: List of available column names in the data
            
        Returns:
            Dictionary with validation results
        """
        available_columns_set = set(available_columns)
        
        # Check numeric features
        available_numeric = [f for f in feature_set.numeric_features if f in available_columns_set]
        missing_numeric = [f for f in feature_set.numeric_features if f not in available_columns_set]
        
        # Check categorical features  
        available_categorical = [f for f in feature_set.categorical_features if f in available_columns_set]
        missing_categorical = [f for f in feature_set.categorical_features if f not in available_columns_set]
        
        return {
            'feature_set_name': feature_set.name,
            'total_requested': feature_set.feature_count,
            'total_available': len(available_numeric) + len(available_categorical),
            'availability_ratio': (len(available_numeric) + len(available_categorical)) / feature_set.feature_count if feature_set.feature_count > 0 else 0,
            'numeric': {
                'requested': feature_set.numeric_features,
                'available': available_numeric,
                'missing': missing_numeric,
                'count_available': len(available_numeric),
                'count_missing': len(missing_numeric)
            },
            'categorical': {
                'requested': feature_set.categorical_features,
                'available': available_categorical,
                'missing': missing_categorical,
                'count_available': len(available_categorical),
                'count_missing': len(missing_categorical)
            }
        }
    
    def suggest_alternative_feature_sets(self, available_columns: List[str], 
                                       min_availability_ratio: float = 0.7) -> List[str]:
        """
        Suggest feature sets that have good availability in the data.
        
        Args:
            available_columns: List of available column names
            min_availability_ratio: Minimum ratio of features that must be available
            
        Returns:
            List of recommended feature set names
        """
        recommendations = []
        
        for name, feature_set in self.feature_sets.items():
            validation = self.validate_features_availability(feature_set, available_columns)
            
            if validation['availability_ratio'] >= min_availability_ratio:
                recommendations.append({
                    'name': name,
                    'availability_ratio': validation['availability_ratio'],
                    'available_features': validation['total_available'],
                    'description': feature_set.description
                })
        
        # Sort by availability ratio (descending)
        recommendations.sort(key=lambda x: x['availability_ratio'], reverse=True)
        
        return recommendations
    
    def print_feature_set_summary(self, feature_set_name: str):
        """Print a detailed summary of a feature set"""
        feature_set = self.get_feature_set(feature_set_name)
        
        print(f"\n📊 FEATURE SET: {feature_set.name.upper()}")
        print(f"📝 Description: {feature_set.description}")
        print(f"🔢 Total features: {feature_set.feature_count}")
        
        if feature_set.numeric_features:
            print(f"\n🧮 Numeric features ({len(feature_set.numeric_features)}):")
            for i, feature in enumerate(feature_set.numeric_features, 1):
                print(f"  {i:2d}. {feature}")
        
        if feature_set.categorical_features:
            print(f"\n🏷️  Categorical features ({len(feature_set.categorical_features)}):")
            for i, feature in enumerate(feature_set.categorical_features, 1):
                print(f"  {i:2d}. {feature}")
    
    def export_feature_set_to_json(self, feature_set_name: str, output_file: str):
        """Export a feature set to JSON for sharing/reproduction"""
        import json
        
        feature_set = self.get_feature_set(feature_set_name)
        
        export_data = {
            'name': feature_set.name,
            'description': feature_set.description,
            'numeric_features': feature_set.numeric_features,
            'categorical_features': feature_set.categorical_features,
            'feature_count': feature_set.feature_count,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Feature set '{feature_set_name}' exported to: {output_file}")


# Convenience function for quick access
def get_feature_manager() -> FeatureManager:
    """Get a configured FeatureManager instance"""
    return FeatureManager()


# Quick access to predefined feature sets
def list_available_feature_sets() -> None:
    """Print all available predefined feature sets"""
    manager = get_feature_manager()
    feature_sets = manager.list_feature_sets()
    
    print("🎯 AVAILABLE FEATURE SETS:")
    print("=" * 50)
    
    for name, description in feature_sets.items():
        print(f"📊 {name}")
        print(f"   {description}")
        print()


if __name__ == "__main__":
    # Demo the feature manager
    print("🧬 ECOSYSTEM CLUSTERING FEATURE MANAGER")
    print("=" * 50)
    
    # Show all available feature sets
    list_available_feature_sets()
    
    # Demo feature set details
    manager = get_feature_manager()
    
    for feature_set_name in ['geographic', 'climate', 'comprehensive']:
        manager.print_feature_set_summary(feature_set_name)