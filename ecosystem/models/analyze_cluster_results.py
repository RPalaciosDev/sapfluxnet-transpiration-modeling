"""
Analyze Cluster Model Results
Comprehensive analysis of cluster-specific XGBoost model performance and feature patterns
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import argparse

class ClusterResultsAnalyzer:
    """
    Analyze and summarize cluster model training results
    """
    
    def __init__(self, results_dir='./results/cluster_models'):
        self.results_dir = results_dir
        self.timestamp = None
        self.metrics_df = None
        self.feature_importance_data = {}
        self.cluster_metadata = {}
        
    def load_latest_results(self):
        """Load the most recent cluster training results"""
        print("🔍 Loading latest cluster model results...")
        
        # Find latest metrics file
        metrics_files = glob.glob(os.path.join(self.results_dir, 'cluster_model_metrics_*.csv'))
        if not metrics_files:
            raise FileNotFoundError(f"No metrics files found in {self.results_dir}")
        
        latest_metrics = max(metrics_files, key=os.path.getctime)
        self.timestamp = latest_metrics.split('_')[-1].replace('.csv', '')
        
        print(f"📊 Loading results from: {os.path.basename(latest_metrics)}")
        
        # Load metrics
        self.metrics_df = pd.read_csv(latest_metrics)
        
        # Load feature importance for each cluster
        for _, row in self.metrics_df.iterrows():
            cluster_id = int(row['cluster'])
            importance_file = os.path.join(
                self.results_dir, 
                f'feature_importance_cluster_{cluster_id}_{self.timestamp}.csv'
            )
            
            if os.path.exists(importance_file):
                importance_df = pd.read_csv(importance_file)
                self.feature_importance_data[cluster_id] = importance_df
                print(f"  ✅ Loaded feature importance for Cluster {cluster_id}")
            else:
                print(f"  ⚠️  Missing feature importance for Cluster {cluster_id}")
        
        # Load cluster metadata if available
        self._load_cluster_metadata()
        
        print(f"✅ Loaded results for {len(self.metrics_df)} clusters")
    
    def _load_cluster_metadata(self):
        """Load cluster metadata from preprocessing stage"""
        preprocessed_dir = os.path.join(self.results_dir, 'preprocessed_libsvm')
        if not os.path.exists(preprocessed_dir):
            print("📝 No preprocessed metadata found")
            return
        
        for _, row in self.metrics_df.iterrows():
            cluster_id = int(row['cluster'])
            metadata_file = os.path.join(preprocessed_dir, f'cluster_{cluster_id}_metadata.json')
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.cluster_metadata[cluster_id] = metadata
    
    def print_performance_summary(self):
        """Print overall performance summary"""
        print("\n" + "="*80)
        print("🎯 CLUSTER MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Sort by cluster ID
        sorted_metrics = self.metrics_df.sort_values('cluster')
        
        print(f"📅 Training completed: {self.timestamp}")
        print(f"🔢 Total clusters: {len(sorted_metrics)}")
        
        # Overall statistics
        avg_test_r2 = sorted_metrics['test_r2'].mean()
        avg_test_rmse = sorted_metrics['test_rmse'].mean()
        total_samples = sorted_metrics['total_rows'].sum()
        
        print(f"📊 Average Test R²: {avg_test_r2:.4f} ± {sorted_metrics['test_r2'].std():.4f}")
        print(f"📊 Average Test RMSE: {avg_test_rmse:.4f} ± {sorted_metrics['test_rmse'].std():.4f}")
        print(f"📊 Total training samples: {total_samples:,}")
        
        # Performance ranking
        best_cluster = sorted_metrics.loc[sorted_metrics['test_r2'].idxmax()]
        worst_cluster = sorted_metrics.loc[sorted_metrics['test_r2'].idxmin()]
        
        print(f"\n🏆 Best performing cluster: {int(best_cluster['cluster'])} (R² = {best_cluster['test_r2']:.4f})")
        print(f"⚠️  Most challenging cluster: {int(worst_cluster['cluster'])} (R² = {worst_cluster['test_r2']:.4f})")
        
        # Detailed table
        print(f"\n📋 DETAILED CLUSTER PERFORMANCE:")
        print("-" * 100)
        print(f"{'Cluster':<8} {'Total Rows':<12} {'Train R²':<10} {'Test R²':<10} {'Test RMSE':<12} {'Features':<10} {'Status'}")
        print("-" * 100)
        
        for _, row in sorted_metrics.iterrows():
            cluster_id = int(row['cluster'])
            status = self._get_performance_status(row['test_r2'])
            
            print(f"{cluster_id:<8} {row['total_rows']:<12,} {row['train_r2']:<10.4f} "
                  f"{row['test_r2']:<10.4f} {row['test_rmse']:<12.4f} {row['feature_count']:<10} {status}")
    
    def _get_performance_status(self, r2_score):
        """Get performance status emoji"""
        if r2_score > 0.95:
            return "🔥 Excellent"
        elif r2_score > 0.90:
            return "✅ Very Good"
        elif r2_score > 0.85:
            return "👍 Good"
        elif r2_score > 0.80:
            return "⚡ Fair"
        else:
            return "⚠️  Challenging"
    
    def analyze_feature_patterns(self, top_n=10):
        """Analyze feature importance patterns across clusters"""
        print("\n" + "="*80)
        print("🔬 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        for cluster_id in sorted(self.feature_importance_data.keys()):
            importance_df = self.feature_importance_data[cluster_id]
            cluster_metrics = self.metrics_df[self.metrics_df['cluster'] == cluster_id].iloc[0]
            
            print(f"\n📊 CLUSTER {cluster_id} - Feature Analysis")
            print("-" * 60)
            print(f"Performance: R² = {cluster_metrics['test_r2']:.4f}, "
                  f"RMSE = {cluster_metrics['test_rmse']:.4f}")
            print(f"Data size: {cluster_metrics['total_rows']:,} samples")
            
            if cluster_id in self.cluster_metadata:
                metadata = self.cluster_metadata[cluster_id]
                print(f"Sites: {len(metadata['sites'])} ({', '.join(metadata['sites'][:3])}{'...' if len(metadata['sites']) > 3 else ''})")
            
            # Top features
            top_features = importance_df.head(top_n)
            print(f"\n🔝 Top {top_n} Most Important Features:")
            
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                importance_pct = (row['importance'] / top_features['importance'].sum()) * 100
                print(f"  {i:2d}. {row['feature_name']:<30} {row['importance']:<12.1f} ({importance_pct:.1f}%)")
            
            # Feature category analysis
            self._analyze_feature_categories(importance_df, cluster_id)
    
    def _analyze_feature_categories(self, importance_df, cluster_id):
        """Analyze feature categories for a cluster"""
        # Define feature categories
        categories = {
            'Environmental': ['ta', 'rh', 'vpd', 'precip', 'sw_in', 'ppfd_in', 'ext_rad', 'ws'],
            'Temporal': ['hour', 'day', 'month', 'year', 'solar_', 'is_'],
            'Lag Features': ['_lag_', '_cum_', '_std_', '_mean_', '_min_', '_max_', '_range_'],
            'Site Characteristics': ['elevation', 'latitude', 'longitude', 'soil_', 'stand_', 'mean_annual_'],
            'Plant Traits': ['pl_', 'leaf_', 'sapwood_', 'bark_'],
            'Geographic': ['timezone', 'country', 'biome', 'climate_zone', 'igbp'],
            'Management': ['tree_density', 'basal_area', 'n_trees', 'social_', 'measurement_']
        }
        
        # Categorize features
        category_importance = {cat: 0.0 for cat in categories}
        total_importance = importance_df['importance'].sum()
        
        for _, row in importance_df.iterrows():
            feature_name = row['feature_name'].lower()
            categorized = False
            
            for category, patterns in categories.items():
                if any(pattern.lower() in feature_name for pattern in patterns):
                    category_importance[category] += row['importance']
                    categorized = True
                    break
            
            if not categorized:
                if 'Other' not in category_importance:
                    category_importance['Other'] = 0.0
                category_importance['Other'] += row['importance']
        
        # Print category analysis
        print(f"\n🏷️  Feature Category Breakdown:")
        sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        
        for category, importance in sorted_categories:
            if importance > 0:
                percentage = (importance / total_importance) * 100
                print(f"  {category:<20} {percentage:>6.1f}%")
    
    def compare_clusters(self):
        """Compare clusters and identify patterns"""
        print("\n" + "="*80)
        print("🔍 CLUSTER COMPARISON & PATTERNS")
        print("="*80)
        
        # Size vs Performance analysis
        print("\n📏 Cluster Size vs Performance Analysis:")
        print("-" * 50)
        
        sorted_by_size = self.metrics_df.sort_values('total_rows', ascending=False)
        for _, row in sorted_by_size.iterrows():
            cluster_id = int(row['cluster'])
            size_category = self._get_size_category(row['total_rows'])
            performance_category = self._get_performance_category(row['test_r2'])
            
            print(f"Cluster {cluster_id}: {size_category:<15} → {performance_category}")
        
        # Feature diversity analysis
        print(f"\n🌈 Feature Diversity Analysis:")
        print("-" * 40)
        
        for cluster_id in sorted(self.feature_importance_data.keys()):
            importance_df = self.feature_importance_data[cluster_id]
            
            # Calculate feature concentration (Gini coefficient-like measure)
            sorted_importance = importance_df['importance'].sort_values(ascending=False)
            total_importance = sorted_importance.sum()
            
            # Top 10 features concentration
            top10_concentration = sorted_importance.head(10).sum() / total_importance * 100
            
            print(f"Cluster {cluster_id}: Top 10 features = {top10_concentration:.1f}% of total importance")
    
    def _get_size_category(self, total_rows):
        """Categorize cluster by size"""
        if total_rows > 2000000:
            return "Very Large"
        elif total_rows > 1000000:
            return "Large"
        elif total_rows > 500000:
            return "Medium"
        else:
            return "Small"
    
    def _get_performance_category(self, r2_score):
        """Categorize cluster by performance"""
        if r2_score > 0.95:
            return "🔥 Excellent Performance"
        elif r2_score > 0.90:
            return "✅ Very Good Performance"
        elif r2_score > 0.85:
            return "👍 Good Performance"
        else:
            return "⚠️  Needs Improvement"
    
    def suggest_next_steps(self):
        """Suggest next steps based on results"""
        print("\n" + "="*80)
        print("🚀 RECOMMENDED NEXT STEPS")
        print("="*80)
        
        best_cluster = self.metrics_df.loc[self.metrics_df['test_r2'].idxmax()]
        worst_cluster = self.metrics_df.loc[self.metrics_df['test_r2'].idxmin()]
        
        print(f"Based on your cluster model results:")
        print(f"")
        
        # Performance-based recommendations
        if best_cluster['test_r2'] > 0.95:
            print(f"✅ EXCELLENT: Cluster {int(best_cluster['cluster'])} shows exceptional performance (R² = {best_cluster['test_r2']:.4f})")
            print(f"   → This cluster's approach could be a template for others")
        
        if worst_cluster['test_r2'] < 0.90:
            print(f"⚠️  ATTENTION: Cluster {int(worst_cluster['cluster'])} needs improvement (R² = {worst_cluster['test_r2']:.4f})")
            print(f"   → Consider feature engineering or different modeling approach")
        
        print(f"\n🎯 Immediate Action Items:")
        print(f"1. 📍 Spatial Validation: Test Leave-One-Site-Out within each cluster")
        print(f"2. 🔬 Compare to Baseline: How do these compare to original spatial validation?")
        print(f"3. 🧬 Ecosystem Interpretation: What ecological patterns do clusters represent?")
        print(f"4. 🔄 Cross-Cluster Testing: Can models generalize between similar clusters?")
        print(f"5. 📊 New Site Classification: Build pipeline for classifying new sites")
        
        print(f"\n💡 Research Questions to Explore:")
        print(f"• Do high-performing clusters share common environmental characteristics?")
        print(f"• Can we identify the ecological drivers behind each cluster?")
        print(f"• How much do geographic vs. environmental factors contribute?")
        print(f"• Are there opportunities for ensemble modeling across clusters?")
    
    def export_summary_report(self, output_file=None):
        """Export detailed summary report"""
        if output_file is None:
            output_file = f"cluster_analysis_report_{self.timestamp}.md"
        
        print(f"\n💾 Exporting detailed report to: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write(f"# Cluster Model Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training timestamp: {self.timestamp}\n\n")
            
            # Performance summary
            f.write(f"## Performance Summary\n\n")
            f.write(f"| Cluster | Total Rows | Train R² | Test R² | Test RMSE | Features |\n")
            f.write(f"|---------|------------|----------|---------|-----------|----------|\n")
            
            for _, row in self.metrics_df.sort_values('cluster').iterrows():
                f.write(f"| {int(row['cluster'])} | {row['total_rows']:,} | "
                       f"{row['train_r2']:.4f} | {row['test_r2']:.4f} | "
                       f"{row['test_rmse']:.4f} | {row['feature_count']} |\n")
            
            # Feature importance details
            f.write(f"\n## Top Features by Cluster\n\n")
            for cluster_id in sorted(self.feature_importance_data.keys()):
                importance_df = self.feature_importance_data[cluster_id]
                f.write(f"### Cluster {cluster_id}\n\n")
                
                top_features = importance_df.head(15)
                for i, (_, row) in enumerate(top_features.iterrows(), 1):
                    f.write(f"{i}. **{row['feature_name']}**: {row['importance']:.1f}\n")
                f.write(f"\n")
        
        print(f"✅ Report exported successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze Cluster Model Results")
    parser.add_argument('--results-dir', default='./results/cluster_models',
                        help="Directory containing cluster model results")
    parser.add_argument('--top-features', type=int, default=10,
                        help="Number of top features to show per cluster")
    parser.add_argument('--export-report', action='store_true',
                        help="Export detailed markdown report")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ClusterResultsAnalyzer(args.results_dir)
        
        # Load and analyze results
        analyzer.load_latest_results()
        analyzer.print_performance_summary()
        analyzer.analyze_feature_patterns(args.top_features)
        analyzer.compare_clusters()
        analyzer.suggest_next_steps()
        
        # Export report if requested
        if args.export_report:
            analyzer.export_summary_report()
        
        print(f"\n🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 