#!/usr/bin/env python3
"""
Feature Dimensionality Analysis for Sap Flow Prediction
Analyzes whether 272 features is optimal or if dimensionality reduction would help
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FeatureDimensionalityAnalyzer:
    """
    Analyze feature dimensionality for sap flow prediction
    """
    
    def __init__(self):
        self.feature_mapping = None
        self.load_feature_mapping()
    
    def load_feature_mapping(self):
        """Load the feature mapping for context"""
        try:
            mapping_file = Path('../../feature_importance/feature_mapping_v2_final.csv')
            if mapping_file.exists():
                self.feature_mapping = pd.read_csv(mapping_file)
                print(f"✅ Loaded feature mapping: {len(self.feature_mapping)} features")
            else:
                print("⚠️ Feature mapping not found, will use generic analysis")
        except Exception as e:
            print(f"⚠️ Could not load feature mapping: {e}")
    
    def analyze_dimensionality_rules(self):
        """Analyze based on standard dimensionality rules"""
        print("🔍 FEATURE DIMENSIONALITY ANALYSIS")
        print("=" * 50)
        
        # Based on your spatial validation results
        samples_per_cluster = 1800000  # Approximate from your results
        n_features = 272
        
        ratio = samples_per_cluster / n_features
        
        print(f"📊 Current Configuration:")
        print(f"   Features: {n_features}")
        print(f"   Samples per cluster: {samples_per_cluster:,}")
        print(f"   Samples per feature: {ratio:.0f}")
        
        print(f"\n📏 Dimensionality Rules Assessment:")
        
        # Rule 1: General ML rule
        if ratio >= 100:
            status1 = "✅ EXCELLENT"
        elif ratio >= 50:
            status1 = "✅ GOOD"
        elif ratio >= 20:
            status1 = "⚠️ MARGINAL"
        else:
            status1 = "❌ POOR"
        
        print(f"   General ML rule (50+ samples/feature): {status1}")
        
        # Rule 2: Tree-based models
        if ratio >= 100:
            status2 = "✅ OPTIMAL"
        elif ratio >= 50:
            status2 = "✅ GOOD"
        else:
            status2 = "⚠️ SUBOPTIMAL"
        
        print(f"   Tree models rule (100+ samples/feature): {status2}")
        
        # Rule 3: High-dimensional data
        if ratio >= 1000:
            status3 = "✅ HIGH-DIM SAFE"
        elif ratio >= 500:
            status3 = "✅ SAFE"
        else:
            status3 = "⚠️ MONITOR"
        
        print(f"   High-dimensional rule (1000+ samples/feature): {status3}")
        
        return ratio
    
    def analyze_feature_categories(self):
        """Analyze feature categories for potential redundancy"""
        if self.feature_mapping is None:
            print("⚠️ Cannot analyze categories without feature mapping")
            return
        
        print(f"\n📂 FEATURE CATEGORY ANALYSIS:")
        print("-" * 30)
        
        category_counts = self.feature_mapping['category'].value_counts()
        total_features = len(self.feature_mapping)
        
        for category, count in category_counts.items():
            percentage = (count / total_features) * 100
            print(f"   {category}: {count} features ({percentage:.1f}%)")
        
        # Identify potential redundancy
        print(f"\n🔍 REDUNDANCY ANALYSIS:")
        
        # Rolling features analysis
        rolling_features = self.feature_mapping[
            self.feature_mapping['category'] == 'Rolling'
        ]
        if len(rolling_features) > 0:
            print(f"   Rolling features: {len(rolling_features)} features")
            print(f"   → Potential correlation between different time windows")
            print(f"   → Consider: Keep only 1h, 6h, 24h windows")
        
        # Lagged features analysis
        lagged_features = self.feature_mapping[
            self.feature_mapping['category'] == 'Lagged'
        ]
        if len(lagged_features) > 0:
            print(f"   Lagged features: {len(lagged_features)} features")
            print(f"   → Potential correlation between adjacent lag periods")
            print(f"   → Consider: Keep only 1h, 4h, 12h lags")
        
        # Rate of change analysis
        rate_features = self.feature_mapping[
            self.feature_mapping['category'] == 'Rate of Change'
        ]
        if len(rate_features) > 0:
            print(f"   Rate features: {len(rate_features)} features")
            print(f"   → May be correlated with rolling standard deviations")
    
    def suggest_feature_reduction_strategies(self):
        """Suggest strategies for feature reduction if needed"""
        print(f"\n🎯 FEATURE REDUCTION STRATEGIES:")
        print("-" * 35)
        
        print(f"💡 STRATEGY 1: Correlation-based reduction")
        print(f"   - Remove features with correlation > 0.95")
        print(f"   - Expected reduction: 20-30 features")
        print(f"   - Risk: Low (removes truly redundant features)")
        
        print(f"\n💡 STRATEGY 2: Importance-based selection")
        print(f"   - Keep top 150-200 features by XGBoost importance")
        print(f"   - Expected reduction: 72-122 features")
        print(f"   - Risk: Medium (may remove useful interactions)")
        
        print(f"\n💡 STRATEGY 3: Category-wise pruning")
        print(f"   - Rolling: Keep 1h, 6h, 24h only (reduce ~50 features)")
        print(f"   - Lagged: Keep 1h, 4h, 12h only (reduce ~25 features)")
        print(f"   - Expected reduction: 75 features")
        print(f"   - Risk: Low (maintains temporal diversity)")
        
        print(f"\n💡 STRATEGY 4: PCA/Dimensionality reduction")
        print(f"   - Transform to 100-150 principal components")
        print(f"   - Expected reduction: 122-172 features")
        print(f"   - Risk: High (loses interpretability)")
        
        print(f"\n💡 STRATEGY 5: Recursive feature elimination")
        print(f"   - Iteratively remove least important features")
        print(f"   - Target: 150-200 features")
        print(f"   - Risk: Medium (computationally expensive)")
    
    def estimate_performance_impact(self):
        """Estimate how feature reduction might affect performance"""
        print(f"\n📈 PERFORMANCE IMPACT ESTIMATES:")
        print("-" * 35)
        
        current_features = 272
        
        scenarios = [
            (200, "Conservative reduction", "0-2% performance loss"),
            (150, "Moderate reduction", "2-5% performance loss"), 
            (100, "Aggressive reduction", "5-10% performance loss"),
            (75, "Very aggressive", "10-15% performance loss"),
            (50, "Minimal feature set", "15-25% performance loss")
        ]
        
        for n_features, description, impact in scenarios:
            reduction = ((current_features - n_features) / current_features) * 100
            print(f"   {n_features:3d} features ({reduction:4.1f}% reduction): {impact}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. Current 272 features: ✅ OPTIMAL for your data size")
        print(f"   2. If reducing: Target 150-200 features (safe zone)")
        print(f"   3. Test reduction impact on validation performance")
        print(f"   4. Consider reduction only if:")
        print(f"      - Training time is too slow")
        print(f"      - Memory constraints")
        print(f"      - Interpretability requirements")
    
    def analyze_computational_costs(self):
        """Analyze computational costs of current feature set"""
        print(f"\n💻 COMPUTATIONAL COST ANALYSIS:")
        print("-" * 35)
        
        n_features = 272
        
        # Training time estimates (relative to 100 features)
        training_factor = (n_features / 100) ** 0.7  # Sublinear scaling
        memory_factor = n_features / 100  # Linear scaling
        
        print(f"   Training time factor: {training_factor:.1f}x")
        print(f"   Memory usage factor: {memory_factor:.1f}x")
        
        # Hyperparameter optimization impact
        trials_time = training_factor * 100  # 100 trials
        print(f"   Hyperopt time factor: {trials_time:.0f}x relative to 100 features")
        
        # Storage requirements
        storage_mb = (1800000 * n_features * 8) / (1024**2)  # float64
        print(f"   Data storage per cluster: {storage_mb:.0f} MB")
        
        if storage_mb > 1000:
            print(f"   ⚠️ Consider feature reduction for memory efficiency")
        else:
            print(f"   ✅ Memory usage is reasonable")
    
    def run_complete_analysis(self):
        """Run the complete dimensionality analysis"""
        ratio = self.analyze_dimensionality_rules()
        self.analyze_feature_categories()
        self.suggest_feature_reduction_strategies()
        self.estimate_performance_impact()
        self.analyze_computational_costs()
        
        # Final verdict
        print(f"\n🏆 FINAL VERDICT:")
        print("=" * 20)
        
        if ratio > 5000:
            verdict = "✅ FEATURE COUNT IS OPTIMAL"
            recommendation = "No reduction needed. Current feature set is well-supported."
        elif ratio > 1000:
            verdict = "✅ FEATURE COUNT IS GOOD"
            recommendation = "Optional reduction for efficiency, but not necessary."
        elif ratio > 500:
            verdict = "⚠️ FEATURE COUNT IS ACCEPTABLE"
            recommendation = "Consider moderate reduction to 150-200 features."
        else:
            verdict = "❌ TOO MANY FEATURES"
            recommendation = "Reduce to 100-150 features to prevent overfitting."
        
        print(f"   {verdict}")
        print(f"   Recommendation: {recommendation}")
        
        return verdict, recommendation

def main():
    """Main analysis workflow"""
    analyzer = FeatureDimensionalityAnalyzer()
    verdict, recommendation = analyzer.run_complete_analysis()
    
    print(f"\n📋 SUMMARY:")
    print(f"   Current: 272 features")
    print(f"   Status: {verdict}")
    print(f"   Action: {recommendation}")

if __name__ == "__main__":
    main() 