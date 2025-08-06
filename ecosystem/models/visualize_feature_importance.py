"""
Feature Importance Visualization for Cluster Models
Creates comprehensive visualizations of feature importance across different clusters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_feature_importance_data(results_dir):
    """Load all feature importance files from a results directory"""
    # Look for both basic and mapped importance files
    importance_files = glob.glob(os.path.join(results_dir, 'feature_importance_cluster_*.csv'))
    mapped_files = glob.glob(os.path.join(results_dir, 'mapped_feature_importance_cluster_*.csv'))
    
    if not importance_files:
        raise FileNotFoundError(f"No feature importance files found in {results_dir}")
    
    # Prefer mapped files if available
    files_to_use = mapped_files if mapped_files else importance_files
    
    all_importance = []
    
    for file_path in files_to_use:
        # Extract cluster ID from filename
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        cluster_id = None
        for i, part in enumerate(parts):
            if part == 'cluster' and i + 1 < len(parts):
                try:
                    cluster_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue
        
        if cluster_id is not None:
            df = pd.read_csv(file_path)
            df['cluster'] = cluster_id
            all_importance.append(df)
            print(f"✅ Loaded importance data for cluster {cluster_id}")
    
    if not all_importance:
        raise ValueError("No valid feature importance data found")
    
    combined_df = pd.concat(all_importance, ignore_index=True)
    print(f"📊 Combined importance data: {len(combined_df)} records across {len(all_importance)} clusters")
    
    return combined_df

def create_top_features_heatmap(importance_df, output_dir, top_n=20):
    """Create heatmap of top N features across clusters"""
    print(f"🎨 Creating top {top_n} features heatmap...")
    
    # Get top features overall
    feature_importance_sum = importance_df.groupby('feature_name')['importance'].sum().sort_values(ascending=False)
    top_features = feature_importance_sum.head(top_n).index.tolist()
    
    # Create pivot table for heatmap
    heatmap_data = importance_df[importance_df['feature_name'].isin(top_features)].pivot_table(
        index='feature_name', 
        columns='cluster', 
        values='importance', 
        fill_value=0
    )
    
    # Reorder by overall importance
    heatmap_data = heatmap_data.reindex(top_features)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.0f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Feature Importance (Gain)'})
    
    plt.title(f'Top {top_n} Feature Importance Across Clusters', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'feature_importance_heatmap_top{top_n}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_cluster_comparison_plot(importance_df, output_dir, top_n=15):
    """Create bar plot comparing top features across clusters"""
    print(f"🎨 Creating cluster comparison plot for top {top_n} features...")
    
    # Get top features overall
    feature_importance_sum = importance_df.groupby('feature_name')['importance'].sum().sort_values(ascending=False)
    top_features = feature_importance_sum.head(top_n).index.tolist()
    
    # Filter data for top features
    plot_data = importance_df[importance_df['feature_name'].isin(top_features)]
    
    # Create subplots for each cluster
    clusters = sorted(importance_df['cluster'].unique())
    n_clusters = len(clusters)
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 8), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = plot_data[plot_data['cluster'] == cluster_id].sort_values('importance', ascending=True)
        
        if len(cluster_data) > 0:
            axes[i].barh(cluster_data['feature_name'], cluster_data['importance'], 
                        color=plt.cm.Set3(i / n_clusters))
            axes[i].set_title(f'Cluster {cluster_id}', fontweight='bold')
            axes[i].set_xlabel('Feature Importance (Gain)')
            
            # Rotate y-axis labels for better readability
            axes[i].tick_params(axis='y', labelsize=8)
    
    plt.suptitle(f'Top {top_n} Feature Importance by Cluster', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'feature_importance_by_cluster_top{top_n}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_feature_category_analysis(importance_df, output_dir):
    """Create analysis based on feature categories (if available)"""
    print("🎨 Creating feature category analysis...")
    
    # Check if we have feature categories (from mapped importance files)
    if 'feature_category' not in importance_df.columns:
        print("  ⚠️  No feature categories found, skipping category analysis")
        return None
    
    # Aggregate importance by category and cluster
    category_importance = importance_df.groupby(['cluster', 'feature_category'])['importance'].sum().reset_index()
    
    # Create stacked bar plot
    pivot_data = category_importance.pivot(index='cluster', columns='feature_category', values='importance').fillna(0)
    
    plt.figure(figsize=(12, 8))
    pivot_data.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='Set3')
    
    plt.title('Feature Importance by Category Across Clusters', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Total Feature Importance (Gain)', fontsize=12)
    plt.legend(title='Feature Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_importance_by_category.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    
    # Also create a pie chart for overall category distribution
    plt.figure(figsize=(10, 8))
    category_totals = importance_df.groupby('feature_category')['importance'].sum().sort_values(ascending=False)
    
    plt.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%', startangle=90)
    plt.title('Overall Feature Importance Distribution by Category', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    pie_output_path = os.path.join(output_dir, 'feature_category_distribution.png')
    plt.savefig(pie_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {pie_output_path}")
    
    return [output_path, pie_output_path]

def create_importance_distribution_plot(importance_df, output_dir):
    """Create distribution plot of feature importance values"""
    print("🎨 Creating importance distribution plot...")
    
    plt.figure(figsize=(14, 10))
    
    # Create subplots: histogram, box plot, and violin plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram of importance values by cluster
    clusters = sorted(importance_df['cluster'].unique())
    for cluster_id in clusters:
        cluster_data = importance_df[importance_df['cluster'] == cluster_id]
        axes[0, 0].hist(cluster_data['importance'], alpha=0.6, label=f'Cluster {cluster_id}', bins=30)
    
    axes[0, 0].set_title('Distribution of Feature Importance Values by Cluster')
    axes[0, 0].set_xlabel('Feature Importance (Gain)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')  # Log scale for better visualization
    
    # 2. Box plot of importance by cluster
    sns.boxplot(data=importance_df, x='cluster', y='importance', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Importance Distribution by Cluster')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Feature Importance (Gain)')
    axes[0, 1].set_yscale('log')
    
    # 3. Top features across all clusters
    top_features_overall = importance_df.groupby('feature_name')['importance'].sum().nlargest(20)
    axes[1, 0].barh(range(len(top_features_overall)), top_features_overall.values)
    axes[1, 0].set_yticks(range(len(top_features_overall)))
    axes[1, 0].set_yticklabels(top_features_overall.index, fontsize=8)
    axes[1, 0].set_title('Top 20 Features (Overall Importance)')
    axes[1, 0].set_xlabel('Total Feature Importance (Gain)')
    
    # 4. Number of important features per cluster (importance > threshold)
    threshold = importance_df['importance'].quantile(0.8)  # Top 20% threshold
    important_features_per_cluster = importance_df[importance_df['importance'] > threshold].groupby('cluster').size()
    
    axes[1, 1].bar(important_features_per_cluster.index, important_features_per_cluster.values)
    axes[1, 1].set_title(f'Number of High-Importance Features per Cluster\n(Importance > {threshold:.0f})')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Number of Important Features')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_importance_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def generate_feature_importance_report(importance_df, output_dir):
    """Generate a comprehensive text report of feature importance analysis"""
    print("📝 Generating feature importance report...")
    
    report_path = os.path.join(output_dir, 'feature_importance_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total features analyzed: {importance_df['feature_name'].nunique()}\n")
        f.write(f"Total clusters: {importance_df['cluster'].nunique()}\n")
        f.write(f"Average importance per feature: {importance_df['importance'].mean():.2f}\n")
        f.write(f"Max importance: {importance_df['importance'].max():.2f}\n")
        f.write(f"Min importance: {importance_df['importance'].min():.2f}\n\n")
        
        # Top 20 features overall
        f.write("TOP 20 FEATURES (OVERALL)\n")
        f.write("-" * 25 + "\n")
        top_features = importance_df.groupby('feature_name')['importance'].sum().nlargest(20)
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            f.write(f"{i:2d}. {feature}: {importance:.2f}\n")
        f.write("\n")
        
        # Cluster-specific analysis
        f.write("CLUSTER-SPECIFIC ANALYSIS\n")
        f.write("-" * 25 + "\n")
        for cluster_id in sorted(importance_df['cluster'].unique()):
            cluster_data = importance_df[importance_df['cluster'] == cluster_id]
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Number of features: {len(cluster_data)}\n")
            f.write(f"  Average importance: {cluster_data['importance'].mean():.2f}\n")
            f.write(f"  Max importance: {cluster_data['importance'].max():.2f}\n")
            f.write(f"  Top 5 features:\n")
            
            top_cluster_features = cluster_data.nlargest(5, 'importance')
            for i, (_, row) in enumerate(top_cluster_features.iterrows(), 1):
                f.write(f"    {i}. {row['feature_name']}: {row['importance']:.2f}\n")
        
        # Feature categories analysis (if available)
        if 'feature_category' in importance_df.columns:
            f.write("\n\nFEATURE CATEGORY ANALYSIS\n")
            f.write("-" * 25 + "\n")
            category_totals = importance_df.groupby('feature_category')['importance'].sum().sort_values(ascending=False)
            total_importance = category_totals.sum()
            
            for category, importance in category_totals.items():
                percentage = (importance / total_importance) * 100
                f.write(f"{category}: {importance:.2f} ({percentage:.1f}%)\n")
    
    print(f"  💾 Saved: {report_path}")
    return report_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize Feature Importance from Cluster Models")
    parser.add_argument('results_dir', 
                        help="Directory containing feature importance CSV files")
    parser.add_argument('--output-dir', default=None,
                        help="Directory to save visualizations (default: results_dir/visualizations)")
    parser.add_argument('--top-n', type=int, default=20,
                        help="Number of top features to show in detailed plots (default: 20)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.results_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎨 Feature Importance Visualization Suite")
    print("=" * 50)
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Top N features: {args.top_n}")
    
    try:
        # Load feature importance data
        importance_df = load_feature_importance_data(args.results_dir)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # 1. Top features heatmap
        create_top_features_heatmap(importance_df, output_dir, args.top_n)
        
        # 2. Cluster comparison plot
        create_cluster_comparison_plot(importance_df, output_dir, args.top_n)
        
        # 3. Feature category analysis (if available)
        create_feature_category_analysis(importance_df, output_dir)
        
        # 4. Importance distribution plots
        create_importance_distribution_plot(importance_df, output_dir)
        
        # 5. Generate comprehensive report
        generate_feature_importance_report(importance_df, output_dir)
        
        print(f"\n✅ Feature importance visualization completed!")
        print(f"📁 All outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
