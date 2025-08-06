"""
Spatial Validation Visualization Suite
Creates comprehensive visualizations of Leave-One-Site-Out spatial validation results
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

def load_spatial_validation_data(results_dir):
    """Load spatial validation results from CSV files"""
    # Look for fold results files
    fold_files = glob.glob(os.path.join(results_dir, '*spatial_fold_results_*.csv'))
    summary_files = glob.glob(os.path.join(results_dir, '*spatial_summaries_*.csv'))
    
    if not fold_files:
        raise FileNotFoundError(f"No spatial validation fold results found in {results_dir}")
    
    # Use the most recent file
    fold_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_fold_file = fold_files[0]
    
    print(f"📊 Loading fold results from: {os.path.basename(latest_fold_file)}")
    fold_results = pd.read_csv(latest_fold_file)
    
    # Load summary data if available
    summaries = None
    if summary_files:
        summary_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_summary_file = summary_files[0]
        print(f"📊 Loading summaries from: {os.path.basename(latest_summary_file)}")
        summaries = pd.read_csv(latest_summary_file)
    
    print(f"✅ Loaded {len(fold_results)} fold results across {fold_results['cluster'].nunique()} clusters")
    
    return fold_results, summaries

def create_performance_overview(fold_results, summaries, output_dir):
    """Create overview plot of model performance across clusters"""
    print("🎨 Creating performance overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² distribution by cluster (box plot)
    sns.boxplot(data=fold_results, x='cluster', y='test_r2', ax=axes[0, 0])
    axes[0, 0].set_title('Test R² Distribution by Cluster', fontweight='bold')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Test R²')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='R²=0')
    axes[0, 0].legend()
    
    # 2. RMSE distribution by cluster
    sns.boxplot(data=fold_results, x='cluster', y='test_rmse', ax=axes[0, 1])
    axes[0, 1].set_title('Test RMSE Distribution by Cluster', fontweight='bold')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Test RMSE')
    
    # 3. Training vs Test R² scatter plot
    axes[1, 0].scatter(fold_results['train_r2'], fold_results['test_r2'], 
                      c=fold_results['cluster'], cmap='Set3', alpha=0.7)
    axes[1, 0].plot([fold_results['train_r2'].min(), fold_results['train_r2'].max()],
                   [fold_results['train_r2'].min(), fold_results['train_r2'].max()],
                   'r--', alpha=0.7, label='Perfect generalization')
    axes[1, 0].set_title('Training vs Test R² (Generalization)', fontweight='bold')
    axes[1, 0].set_xlabel('Training R²')
    axes[1, 0].set_ylabel('Test R²')
    axes[1, 0].legend()
    
    # Add colorbar for cluster identification
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='Set3'), ax=axes[1, 0])
    cbar.set_label('Cluster ID')
    
    # 4. Sample size vs performance
    axes[1, 1].scatter(fold_results['test_samples'], fold_results['test_r2'], 
                      c=fold_results['cluster'], cmap='Set3', alpha=0.7)
    axes[1, 1].set_title('Test Sample Size vs Performance', fontweight='bold')
    axes[1, 1].set_xlabel('Test Sample Size')
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'spatial_validation_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_cluster_performance_comparison(fold_results, summaries, output_dir):
    """Create detailed cluster performance comparison"""
    print("🎨 Creating cluster performance comparison...")
    
    if summaries is not None:
        # Use summary statistics
        plot_data = summaries.copy()
        r2_col, rmse_col = 'mean_test_r2', 'mean_test_rmse'
        r2_err_col, rmse_err_col = 'std_test_r2', 'std_test_rmse'
    else:
        # Calculate from fold results
        plot_data = fold_results.groupby('cluster').agg({
            'test_r2': ['mean', 'std'],
            'test_rmse': ['mean', 'std'],
            'test_samples': 'sum'
        }).round(4)
        plot_data.columns = ['mean_test_r2', 'std_test_r2', 'mean_test_rmse', 'std_test_rmse', 'total_samples']
        plot_data = plot_data.reset_index()
        r2_col, rmse_col = 'mean_test_r2', 'mean_test_rmse'
        r2_err_col, rmse_err_col = 'std_test_r2', 'std_test_rmse'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Mean R² with error bars
    axes[0].bar(plot_data['cluster'], plot_data[r2_col], 
                yerr=plot_data[r2_err_col], capsize=5, alpha=0.7,
                color=['green' if x > 0 else 'red' for x in plot_data[r2_col]])
    axes[0].set_title('Mean Test R² by Cluster (±1 std)', fontweight='bold')
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Mean Test R²')
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (cluster, r2, std) in enumerate(zip(plot_data['cluster'], plot_data[r2_col], plot_data[r2_err_col])):
        axes[0].text(i, r2 + std + 0.1, f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Mean RMSE with error bars
    axes[1].bar(plot_data['cluster'], plot_data[rmse_col], 
                yerr=plot_data[rmse_err_col], capsize=5, alpha=0.7, color='orange')
    axes[1].set_title('Mean Test RMSE by Cluster (±1 std)', fontweight='bold')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Mean Test RMSE')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (cluster, rmse, std) in enumerate(zip(plot_data['cluster'], plot_data[rmse_col], plot_data[rmse_err_col])):
        axes[1].text(i, rmse + std + 0.1, f'{rmse:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'cluster_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_fold_by_fold_analysis(fold_results, output_dir):
    """Create detailed fold-by-fold analysis for each cluster"""
    print("🎨 Creating fold-by-fold analysis...")
    
    clusters = sorted(fold_results['cluster'].unique())
    n_clusters = len(clusters)
    
    # Create a large figure with subplots for each cluster
    fig, axes = plt.subplots(n_clusters, 2, figsize=(16, 4*n_clusters))
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for i, cluster_id in enumerate(clusters):
        cluster_data = fold_results[fold_results['cluster'] == cluster_id].sort_values('fold')
        
        # Left plot: R² progression across folds
        axes[i, 0].plot(cluster_data['fold'], cluster_data['train_r2'], 
                       'o-', label='Training R²', color='blue', alpha=0.7)
        axes[i, 0].plot(cluster_data['fold'], cluster_data['test_r2'], 
                       'o-', label='Test R²', color='red', alpha=0.7)
        axes[i, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[i, 0].set_title(f'Cluster {cluster_id}: R² Across Folds', fontweight='bold')
        axes[i, 0].set_xlabel('Fold Number')
        axes[i, 0].set_ylabel('R²')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Right plot: RMSE progression across folds
        axes[i, 1].plot(cluster_data['fold'], cluster_data['train_rmse'], 
                       'o-', label='Training RMSE', color='blue', alpha=0.7)
        axes[i, 1].plot(cluster_data['fold'], cluster_data['test_rmse'], 
                       'o-', label='Test RMSE', color='red', alpha=0.7)
        axes[i, 1].set_title(f'Cluster {cluster_id}: RMSE Across Folds', fontweight='bold')
        axes[i, 1].set_xlabel('Fold Number')
        axes[i, 1].set_ylabel('RMSE')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Add site labels on x-axis if not too many
        if len(cluster_data) <= 15:
            axes[i, 0].set_xticks(cluster_data['fold'])
            axes[i, 1].set_xticks(cluster_data['fold'])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fold_by_fold_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_site_performance_heatmap(fold_results, output_dir):
    """Create heatmap showing performance for each test site"""
    print("🎨 Creating site performance heatmap...")
    
    # Create pivot table with sites as rows and metrics as columns
    heatmap_data = fold_results.pivot_table(
        index='test_site', 
        columns='cluster', 
        values='test_r2', 
        fill_value=np.nan
    )
    
    plt.figure(figsize=(max(8, len(heatmap_data.columns)*0.8), max(10, len(heatmap_data)*0.3)))
    
    # Create heatmap with custom colormap
    mask = heatmap_data.isnull()
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn',
                center=0,
                mask=mask,
                cbar_kws={'label': 'Test R²'},
                linewidths=0.5)
    
    plt.title('Site-Specific Test R² Performance by Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Test Site', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'site_performance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def create_performance_distribution_analysis(fold_results, output_dir):
    """Create detailed analysis of performance distributions"""
    print("🎨 Creating performance distribution analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall R² histogram
    axes[0, 0].hist(fold_results['test_r2'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(fold_results['test_r2'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {fold_results["test_r2"].mean():.3f}')
    axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='R²=0')
    axes[0, 0].set_title('Distribution of Test R² (All Folds)')
    axes[0, 0].set_xlabel('Test R²')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. Overall RMSE histogram
    axes[0, 1].hist(fold_results['test_rmse'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(fold_results['test_rmse'].mean(), color='red', linestyle='--',
                      label=f'Mean: {fold_results["test_rmse"].mean():.2f}')
    axes[0, 1].set_title('Distribution of Test RMSE (All Folds)')
    axes[0, 1].set_xlabel('Test RMSE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Overfitting analysis (Train R² - Test R²)
    overfitting = fold_results['train_r2'] - fold_results['test_r2']
    axes[0, 2].hist(overfitting, bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 2].axvline(overfitting.mean(), color='red', linestyle='--',
                      label=f'Mean: {overfitting.mean():.3f}')
    axes[0, 2].axvline(0, color='black', linestyle='-', alpha=0.5, label='No overfitting')
    axes[0, 2].set_title('Overfitting Analysis (Train R² - Test R²)')
    axes[0, 2].set_xlabel('Training - Test R²')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. Performance vs sample size
    axes[1, 0].scatter(fold_results['train_samples'], fold_results['test_r2'], alpha=0.6)
    axes[1, 0].set_title('Test Performance vs Training Sample Size')
    axes[1, 0].set_xlabel('Training Sample Size')
    axes[1, 0].set_ylabel('Test R²')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 5. Violin plot of R² by cluster
    sns.violinplot(data=fold_results, x='cluster', y='test_r2', ax=axes[1, 1])
    axes[1, 1].set_title('Test R² Distribution by Cluster (Violin Plot)')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 6. Performance consistency (coefficient of variation)
    cluster_stats = fold_results.groupby('cluster')['test_r2'].agg(['mean', 'std']).reset_index()
    cluster_stats['cv'] = abs(cluster_stats['std'] / cluster_stats['mean'])  # Coefficient of variation
    
    bars = axes[1, 2].bar(cluster_stats['cluster'], cluster_stats['cv'], alpha=0.7, color='purple')
    axes[1, 2].set_title('Performance Consistency by Cluster\n(Coefficient of Variation)')
    axes[1, 2].set_xlabel('Cluster ID')
    axes[1, 2].set_ylabel('Coefficient of Variation (CV)')
    
    # Add value labels on bars
    for bar, cv in zip(bars, cluster_stats['cv']):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{cv:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {output_path}")
    return output_path

def generate_spatial_validation_report(fold_results, summaries, output_dir):
    """Generate comprehensive text report of spatial validation results"""
    print("📝 Generating spatial validation report...")
    
    report_path = os.path.join(output_dir, 'spatial_validation_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("SPATIAL VALIDATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total folds: {len(fold_results)}\n")
        f.write(f"Total clusters: {fold_results['cluster'].nunique()}\n")
        f.write(f"Total test sites: {fold_results['test_site'].nunique()}\n")
        f.write(f"Average test R²: {fold_results['test_r2'].mean():.4f} ± {fold_results['test_r2'].std():.4f}\n")
        f.write(f"Average test RMSE: {fold_results['test_rmse'].mean():.4f} ± {fold_results['test_rmse'].std():.4f}\n")
        f.write(f"Percentage of folds with R² > 0: {(fold_results['test_r2'] > 0).mean()*100:.1f}%\n")
        f.write(f"Best performing fold: R² = {fold_results['test_r2'].max():.4f} (Cluster {fold_results.loc[fold_results['test_r2'].idxmax(), 'cluster']}, Site {fold_results.loc[fold_results['test_r2'].idxmax(), 'test_site']})\n")
        f.write(f"Worst performing fold: R² = {fold_results['test_r2'].min():.4f} (Cluster {fold_results.loc[fold_results['test_r2'].idxmin(), 'cluster']}, Site {fold_results.loc[fold_results['test_r2'].idxmin(), 'test_site']})\n\n")
        
        # Overfitting analysis
        overfitting = fold_results['train_r2'] - fold_results['test_r2']
        f.write("OVERFITTING ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average overfitting (Train R² - Test R²): {overfitting.mean():.4f} ± {overfitting.std():.4f}\n")
        f.write(f"Folds with negative overfitting (test > train): {(overfitting < 0).sum()} ({(overfitting < 0).mean()*100:.1f}%)\n")
        f.write(f"Severe overfitting (diff > 1.0): {(overfitting > 1.0).sum()} folds\n\n")
        
        # Cluster-specific analysis
        f.write("CLUSTER-SPECIFIC PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        cluster_stats = fold_results.groupby('cluster').agg({
            'test_r2': ['count', 'mean', 'std', 'min', 'max'],
            'test_rmse': ['mean', 'std'],
            'train_samples': 'mean',
            'test_samples': 'mean'
        }).round(4)
        
        for cluster_id in sorted(fold_results['cluster'].unique()):
            cluster_data = fold_results[fold_results['cluster'] == cluster_id]
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Number of folds: {len(cluster_data)}\n")
            f.write(f"  Test R²: {cluster_data['test_r2'].mean():.4f} ± {cluster_data['test_r2'].std():.4f}\n")
            f.write(f"  Test R² range: [{cluster_data['test_r2'].min():.4f}, {cluster_data['test_r2'].max():.4f}]\n")
            f.write(f"  Test RMSE: {cluster_data['test_rmse'].mean():.4f} ± {cluster_data['test_rmse'].std():.4f}\n")
            f.write(f"  Average training samples: {cluster_data['train_samples'].mean():.0f}\n")
            f.write(f"  Average test samples: {cluster_data['test_samples'].mean():.0f}\n")
            f.write(f"  Folds with R² > 0: {(cluster_data['test_r2'] > 0).sum()}/{len(cluster_data)} ({(cluster_data['test_r2'] > 0).mean()*100:.1f}%)\n")
            
            # Best and worst sites in cluster
            best_site = cluster_data.loc[cluster_data['test_r2'].idxmax(), 'test_site']
            worst_site = cluster_data.loc[cluster_data['test_r2'].idxmin(), 'test_site']
            f.write(f"  Best site: {best_site} (R² = {cluster_data['test_r2'].max():.4f})\n")
            f.write(f"  Worst site: {worst_site} (R² = {cluster_data['test_r2'].min():.4f})\n")
        
        # Sample size analysis
        f.write("\n\nSAMPLE SIZE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average training samples per fold: {fold_results['train_samples'].mean():.0f} ± {fold_results['train_samples'].std():.0f}\n")
        f.write(f"Average test samples per fold: {fold_results['test_samples'].mean():.0f} ± {fold_results['test_samples'].std():.0f}\n")
        f.write(f"Training sample range: [{fold_results['train_samples'].min():.0f}, {fold_results['train_samples'].max():.0f}]\n")
        f.write(f"Test sample range: [{fold_results['test_samples'].min():.0f}, {fold_results['test_samples'].max():.0f}]\n")
        
        # Correlation between sample size and performance
        train_size_corr = fold_results['train_samples'].corr(fold_results['test_r2'])
        test_size_corr = fold_results['test_samples'].corr(fold_results['test_r2'])
        f.write(f"Correlation (training size vs test R²): {train_size_corr:.3f}\n")
        f.write(f"Correlation (test size vs test R²): {test_size_corr:.3f}\n")
    
    print(f"  💾 Saved: {report_path}")
    return report_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize Spatial Validation Results")
    parser.add_argument('results_dir', 
                        help="Directory containing spatial validation CSV files")
    parser.add_argument('--output-dir', default=None,
                        help="Directory to save visualizations (default: results_dir/visualizations)")
    
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
    
    print("🎨 Spatial Validation Visualization Suite")
    print("=" * 50)
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load spatial validation data
        fold_results, summaries = load_spatial_validation_data(args.results_dir)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # 1. Performance overview
        create_performance_overview(fold_results, summaries, output_dir)
        
        # 2. Cluster performance comparison
        create_cluster_performance_comparison(fold_results, summaries, output_dir)
        
        # 3. Fold-by-fold analysis
        create_fold_by_fold_analysis(fold_results, output_dir)
        
        # 4. Site performance heatmap
        create_site_performance_heatmap(fold_results, output_dir)
        
        # 5. Performance distribution analysis
        create_performance_distribution_analysis(fold_results, output_dir)
        
        # 6. Generate comprehensive report
        generate_spatial_validation_report(fold_results, summaries, output_dir)
        
        print(f"\n✅ Spatial validation visualization completed!")
        print(f"📁 All outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
