#!/usr/bin/env python3
"""
Comprehensive Analysis of Cluster Feature Importance Patterns
Analyzes the mapped feature importance files to identify key patterns across clusters
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_cluster_data():
    """Load all three cluster feature importance files"""
    
    results_dir = Path('results/cluster_models')
    
    # Find the latest feature importance files
    cluster_files = {
        0: results_dir / 'feature_importance_cluster_0_20250726_112236_mapped.csv',
        1: results_dir / 'feature_importance_cluster_1_20250726_112236_mapped.csv', 
        2: results_dir / 'feature_importance_cluster_2_20250726_112236_mapped.csv'
    }
    
    clusters = {}
    for cluster_id, file_path in cluster_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            clusters[cluster_id] = df
            print(f"✅ Loaded Cluster {cluster_id}: {len(df)} features")
        else:
            print(f"❌ Missing: {file_path}")
    
    return clusters

def analyze_category_patterns(clusters):
    """Analyze feature category patterns across clusters"""
    
    print("\n" + "="*80)
    print("📊 CATEGORY ANALYSIS ACROSS CLUSTERS")
    print("="*80)
    
    category_analysis = {}
    
    for cluster_id, df in clusters.items():
        # Calculate total importance by category
        category_totals = df.groupby('category')['importance_score'].agg(['sum', 'count', 'mean']).round(2)
        category_totals = category_totals.sort_values('sum', ascending=False)
        
        print(f"\n🏷️  CLUSTER {cluster_id} - Category Importance:")
        print(f"{'Category':<20} {'Total':<12} {'Count':<8} {'Avg':<10}")
        print("-" * 50)
        
        for category, row in category_totals.head(10).iterrows():
            print(f"{category:<20} {row['sum']:<12,.0f} {row['count']:<8} {row['mean']:<10,.0f}")
        
        category_analysis[cluster_id] = category_totals
    
    return category_analysis

def analyze_top_features(clusters, top_n=20):
    """Analyze top N features for each cluster"""
    
    print("\n" + "="*80)
    print(f"🏆 TOP {top_n} FEATURES ANALYSIS")
    print("="*80)
    
    for cluster_id, df in clusters.items():
        print(f"\n🎯 CLUSTER {cluster_id} Top Features:")
        top_features = df.head(top_n)
        
        print(f"{'Rank':<4} {'Feature':<25} {'Category':<15} {'Importance':<12}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            feat_name = row['feature_name'][:24]  # Truncate long names
            print(f"{i:<4} {feat_name:<25} {row['category']:<15} {row['importance_score']:<12,.0f}")

def find_common_important_features(clusters, top_n=50):
    """Find features that are important across multiple clusters"""
    
    print("\n" + "="*80)
    print("🔄 CROSS-CLUSTER FEATURE ANALYSIS")
    print("="*80)
    
    # Get top N features from each cluster
    all_top_features = {}
    for cluster_id, df in clusters.items():
        top_features = set(df.head(top_n)['feature_name'].tolist())
        all_top_features[cluster_id] = top_features
    
    # Find intersections
    if len(clusters) >= 2:
        cluster_ids = list(clusters.keys())
        
        # Features in multiple clusters
        for i, cluster_a in enumerate(cluster_ids):
            for cluster_b in cluster_ids[i+1:]:
                common = all_top_features[cluster_a] & all_top_features[cluster_b]
                print(f"\n🤝 Common top-{top_n} features between Cluster {cluster_a} & {cluster_b}: {len(common)}")
                if common:
                    for feat in sorted(common):
                        # Get importance scores from both clusters
                        imp_a = clusters[cluster_a][clusters[cluster_a]['feature_name'] == feat]['importance_score'].iloc[0]
                        imp_b = clusters[cluster_b][clusters[cluster_b]['feature_name'] == feat]['importance_score'].iloc[0]
                        print(f"  • {feat}: C{cluster_a}={imp_a:.0f}, C{cluster_b}={imp_b:.0f}")
        
        # Features common to all clusters
        if len(clusters) == 3:
            all_common = all_top_features[0] & all_top_features[1] & all_top_features[2]
            print(f"\n🎯 Features in ALL top-{top_n}: {len(all_common)}")
            if all_common:
                for feat in sorted(all_common):
                    imp_0 = clusters[0][clusters[0]['feature_name'] == feat]['importance_score'].iloc[0]
                    imp_1 = clusters[1][clusters[1]['feature_name'] == feat]['importance_score'].iloc[0] 
                    imp_2 = clusters[2][clusters[2]['feature_name'] == feat]['importance_score'].iloc[0]
                    print(f"  • {feat}: C0={imp_0:.0f}, C1={imp_1:.0f}, C2={imp_2:.0f}")

def analyze_feature_types(clusters):
    """Analyze different types of features (lagged, rolling, etc.)"""
    
    print("\n" + "="*80)
    print("🔍 FEATURE TYPE PATTERNS")
    print("="*80)
    
    for cluster_id, df in clusters.items():
        print(f"\n📈 CLUSTER {cluster_id} Feature Type Analysis:")
        
        # Analyze feature name patterns
        feature_patterns = {
            'Lagged Features': df[df['feature_name'].str.contains('_lag_', na=False)],
            'Rolling Window': df[df['feature_name'].str.contains('_std_|_mean_|_min_|_max_|_range_', na=False)],
            'Rate of Change': df[df['feature_name'].str.contains('_rate_', na=False)],
            'Cumulative': df[df['feature_name'].str.contains('_cum_', na=False)],
            'Interactions': df[df['feature_name'].str.contains('_interaction|_ratio', na=False)],
            'Temporal Indicators': df[df['feature_name'].str.contains('is_|hour|day|month|year', na=False)],
        }
        
        for pattern_name, pattern_df in feature_patterns.items():
            if len(pattern_df) > 0:
                total_importance = pattern_df['importance_score'].sum()
                avg_importance = pattern_df['importance_score'].mean()
                print(f"  {pattern_name:<20}: {len(pattern_df):>3} features, Total: {total_importance:>10,.0f}, Avg: {avg_importance:>8,.0f}")

def generate_cluster_insights(clusters):
    """Generate key insights about cluster differences"""
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS & CLUSTER CHARACTERISTICS")
    print("="*80)
    
    for cluster_id, df in clusters.items():
        print(f"\n🎯 CLUSTER {cluster_id} PROFILE:")
        
        # Get top 5 features
        top_5 = df.head(5)
        print(f"  🥇 Dominant Features:")
        for _, row in top_5.iterrows():
            print(f"    • {row['feature_name']} ({row['category']}) - {row['importance_score']:,.0f}")
        
        # Category dominance
        top_categories = df.groupby('category')['importance_score'].sum().nlargest(3)
        print(f"  📊 Top Categories:")
        for cat, importance in top_categories.items():
            pct = (importance / df['importance_score'].sum()) * 100
            print(f"    • {cat}: {importance:,.0f} ({pct:.1f}%)")
        
        # Feature type insights
        total_importance = df['importance_score'].sum()
        
        if cluster_id == 0:
            interaction_importance = df[df['category'] == 'Interaction']['importance_score'].sum()
            interaction_pct = (interaction_importance / total_importance) * 100
            print(f"  🧬 Interaction Features: {interaction_pct:.1f}% of total importance")
        
        environmental_importance = df[df['category'] == 'Environmental']['importance_score'].sum()
        env_pct = (environmental_importance / total_importance) * 100
        print(f"  🌍 Environmental Features: {env_pct:.1f}% of total importance")

def main():
    """Main analysis function"""
    
    print("🔍 COMPREHENSIVE CLUSTER FEATURE IMPORTANCE ANALYSIS")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    clusters = load_cluster_data()
    
    if not clusters:
        print("❌ No cluster data found!")
        return
    
    # Run all analyses
    category_analysis = analyze_category_patterns(clusters)
    analyze_top_features(clusters, top_n=15)
    find_common_important_features(clusters, top_n=30)
    analyze_feature_types(clusters)
    generate_cluster_insights(clusters)
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    for cluster_id, df in clusters.items():
        total_features = len(df)
        total_importance = df['importance_score'].sum()
        max_importance = df['importance_score'].max()
        print(f"Cluster {cluster_id}: {total_features} features, Total importance: {total_importance:,.0f}, Max: {max_importance:,.0f}")
    
    print("\n✅ Analysis complete! Check the detailed patterns above.")

if __name__ == "__main__":
    main() 