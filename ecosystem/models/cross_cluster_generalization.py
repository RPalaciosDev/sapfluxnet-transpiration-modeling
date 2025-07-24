"""
Cross-Cluster Generalization Analysis for SAPFLUXNET Ecosystem Models

This script tests what happens when we try to generalize across clusters by:
1. Taking each cluster's trained model
2. Applying it to predict sites from all other clusters  
3. Measuring the performance drop compared to within-cluster predictions

This addresses the critical question: "What happens if we try to generalize across clusters?"
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
from datetime import datetime
import warnings
import gc
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class CrossClusterGeneralizationAnalyzer:
    """
    Analyzes cross-cluster generalization by testing each cluster model on all other clusters
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 models_dir='./results/cluster_models',
                 results_dir='./results/cross_cluster_analysis'):
        self.parquet_dir = parquet_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🔄 Cross-Cluster Generalization Analyzer initialized")
        print(f"📁 Results directory: {results_dir}")
    
    def load_cluster_assignments(self):
        """Load cluster assignments from the latest clustering results"""
        cluster_files = sorted(glob.glob('../evaluation/clustering_results/advanced_site_clusters_*.csv'))
        
        if not cluster_files:
            raise FileNotFoundError("No cluster assignment files found")
        
        latest_file = cluster_files[-1]
        print(f"📊 Loading cluster assignments from: {os.path.basename(latest_file)}")
        
        clusters_df = pd.read_csv(latest_file)
        cluster_assignments = dict(zip(clusters_df['site'], clusters_df['cluster']))
        
        print(f"✅ Loaded {len(cluster_assignments)} site assignments")
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} sites")
        
        return cluster_assignments, cluster_counts
    
    def load_cluster_models(self):
        """Load trained cluster-specific models"""
        print(f"\n🤖 Loading cluster-specific models from {self.models_dir}...")
        
        model_files = glob.glob(os.path.join(self.models_dir, 'xgb_model_cluster_*.json'))
        
        if not model_files:
            raise FileNotFoundError(f"No cluster models found in {self.models_dir}")
        
        models = {}
        for model_file in model_files:
            # Extract cluster ID from filename
            filename = os.path.basename(model_file)
            # Format: xgb_model_cluster_{cluster_id}_{timestamp}.json
            parts = filename.replace('.json', '').split('_')
            cluster_id = None
            for i, part in enumerate(parts):
                if part == 'cluster' and i + 1 < len(parts):
                    try:
                        cluster_id = int(parts[i + 1])
                        break
                    except ValueError:
                        continue
            
            if cluster_id is not None:
                model = xgb.Booster()
                model.load_model(model_file)
                models[cluster_id] = model
                print(f"  ✅ Loaded model for cluster {cluster_id}")
            else:
                print(f"  ⚠️  Could not extract cluster ID from {filename}")
        
        print(f"📊 Loaded {len(models)} cluster models")
        return models
    
    def load_cluster_sample_data(self, cluster_id, cluster_sites, sample_size=5000):
        """Load a sample of data from a cluster for testing"""
        print(f"📊 Loading sample data for cluster {cluster_id} ({len(cluster_sites)} sites)...")
        
        cluster_data = []
        total_loaded = 0
        sites_per_cluster = max(1, sample_size // len(cluster_sites))
        
        for site in cluster_sites:
            if total_loaded >= sample_size:
                break
                
            parquet_file = os.path.join(self.parquet_dir, f'{site}_comprehensive.parquet')
            
            if not os.path.exists(parquet_file):
                print(f"  ⚠️  Missing: {parquet_file}")
                continue
            
            try:
                # Load site data
                df_site = pd.read_parquet(parquet_file)
                df_site = df_site[df_site[self.cluster_col] == cluster_id]
                df_site = df_site.dropna(subset=[self.target_col])
                
                if len(df_site) == 0:
                    continue
                
                # Sample from this site
                site_sample_size = min(sites_per_cluster, len(df_site))
                df_sample = df_site.sample(n=site_sample_size, random_state=42)
                
                cluster_data.append(df_sample)
                total_loaded += len(df_sample)
                
                print(f"  ✅ {site}: {len(df_sample):,} samples")
                
            except Exception as e:
                print(f"  ❌ Error loading {site}: {e}")
                continue
        
        if not cluster_data:
            return None
        
        # Combine all data
        combined_df = pd.concat(cluster_data, ignore_index=True)
        print(f"  📊 Total loaded: {len(combined_df):,} samples from cluster {cluster_id}")
        
        return combined_df
    
    def prepare_features(self, df):
        """Prepare features for prediction (same as used in cluster training)"""
        # Exclude columns (same as in train_cluster_models.py)
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [self.target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        # Extract and clean features
        X_df = df[feature_cols].copy()
        
        # Convert boolean columns to numeric (True=1, False=0)
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype == 'object':
                # Try to convert object columns to numeric, fill non-numeric with 0
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        # Fill remaining NaN values with 0
        X = X_df.fillna(0).values
        y = df[self.target_col].values
        
        return X, y, feature_cols
    
    def test_cross_cluster_generalization(self, source_cluster_id, target_cluster_id, 
                                        source_model, target_data):
        """Test how well a source cluster model predicts target cluster data"""
        
        if target_data is None or len(target_data) == 0:
            return None
        
        try:
            # Prepare features
            X_target, y_target, feature_cols = self.prepare_features(target_data)
            
            if len(X_target) == 0:
                return None
            
            # Create DMatrix and make predictions
            dtest = xgb.DMatrix(X_target, label=y_target)
            y_pred = source_model.predict(dtest)
            
            # Calculate metrics
            metrics = {
                'source_cluster': int(source_cluster_id),
                'target_cluster': int(target_cluster_id),
                'n_samples': int(len(y_target)),
                'rmse': float(np.sqrt(mean_squared_error(y_target, y_pred))),
                'mae': float(mean_absolute_error(y_target, y_pred)),
                'r2': float(r2_score(y_target, y_pred))
            }
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ Error testing {source_cluster_id} → {target_cluster_id}: {e}")
            return None
    
    def run_cross_cluster_analysis(self):
        """Run comprehensive cross-cluster generalization analysis"""
        print("🔄 SAPFLUXNET Cross-Cluster Generalization Analysis")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Test what happens when we generalize across ecosystem boundaries")
        
        try:
            # Load cluster assignments and models
            cluster_assignments, cluster_counts = self.load_cluster_assignments()
            models = self.load_cluster_models()
            
            # Group sites by cluster
            sites_by_cluster = {}
            for site, cluster_id in cluster_assignments.items():
                if cluster_id not in sites_by_cluster:
                    sites_by_cluster[cluster_id] = []
                sites_by_cluster[cluster_id].append(site)
            
            # Load sample data for each cluster
            print("\n📊 Loading sample data for all clusters...")
            cluster_data = {}
            for cluster_id in sorted(sites_by_cluster.keys()):
                if cluster_id in models:
                    cluster_sites = sites_by_cluster[cluster_id]
                    data = self.load_cluster_sample_data(cluster_id, cluster_sites)
                    if data is not None:
                        cluster_data[cluster_id] = data
            
            # Run cross-cluster tests
            print("\n🔄 Running cross-cluster generalization tests...")
            results = []
            
            for source_cluster in sorted(models.keys()):
                if source_cluster not in cluster_data:
                    continue
                    
                source_model = models[source_cluster]
                print(f"\n--- Testing model from Cluster {source_cluster} ---")
                
                for target_cluster in sorted(cluster_data.keys()):
                    target_data = cluster_data[target_cluster]
                    
                    if source_cluster == target_cluster:
                        test_type = "Within-Cluster (Baseline)"
                    else:
                        test_type = "Cross-Cluster"
                    
                    print(f"  {test_type}: Cluster {source_cluster} → Cluster {target_cluster}")
                    
                    metrics = self.test_cross_cluster_generalization(
                        source_cluster, target_cluster, source_model, target_data
                    )
                    
                    if metrics is not None:
                        metrics['test_type'] = test_type
                        results.append(metrics)
                        
                        print(f"    R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
                        
                        # Flag catastrophic failures
                        if metrics['r2'] < -10:
                            print(f"    🚨 CATASTROPHIC FAILURE detected!")
                        elif metrics['r2'] < 0:
                            print(f"    ⚠️  Negative R² - worse than mean prediction")
            
            # Save and analyze results
            self.save_and_analyze_results(results)
            
            print(f"\n✅ Cross-cluster analysis completed!")
            print(f"📊 Tested {len(results)} model-cluster combinations")
            
        except Exception as e:
            print(f"❌ Error in cross-cluster analysis: {e}")
            raise
    
    def save_and_analyze_results(self, results):
        """Save results and create analysis summary"""
        if not results:
            print("⚠️  No results to analyze")
            return
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, f'cross_cluster_results_{self.timestamp}.csv')
        df_results.to_csv(results_file, index=False)
        print(f"📄 Detailed results saved: {results_file}")
        
        # Create summary analysis
        self.create_summary_analysis(df_results)
        
        # Create visualization
        self.create_generalization_heatmap(df_results)
    
    def create_summary_analysis(self, df_results):
        """Create summary analysis of cross-cluster generalization"""
        
        # Separate within-cluster and cross-cluster results
        within_cluster = df_results[df_results['test_type'] == 'Within-Cluster (Baseline)']
        cross_cluster = df_results[df_results['test_type'] == 'Cross-Cluster']
        
        summary = {
            'analysis_timestamp': self.timestamp,
            'total_tests': len(df_results),
            'within_cluster_performance': {
                'count': len(within_cluster),
                'mean_r2': float(within_cluster['r2'].mean()),
                'std_r2': float(within_cluster['r2'].std()),
                'mean_rmse': float(within_cluster['rmse'].mean()),
                'best_r2': float(within_cluster['r2'].max()),
                'worst_r2': float(within_cluster['r2'].min())
            },
            'cross_cluster_performance': {
                'count': len(cross_cluster),
                'mean_r2': float(cross_cluster['r2'].mean()),
                'std_r2': float(cross_cluster['r2'].std()),
                'mean_rmse': float(cross_cluster['rmse'].mean()),
                'best_r2': float(cross_cluster['r2'].max()),
                'worst_r2': float(cross_cluster['r2'].min()),
                'catastrophic_failures': int(len(cross_cluster[cross_cluster['r2'] < -10])),
                'negative_r2_count': int(len(cross_cluster[cross_cluster['r2'] < 0]))
            }
        }
        
        # Performance drop analysis
        if len(within_cluster) > 0 and len(cross_cluster) > 0:
            summary['performance_drop'] = {
                'r2_drop_mean': float(within_cluster['r2'].mean() - cross_cluster['r2'].mean()),
                'r2_drop_median': float(within_cluster['r2'].median() - cross_cluster['r2'].median()),
                'rmse_increase_mean': float(cross_cluster['rmse'].mean() - within_cluster['rmse'].mean())
            }
        
        # Best and worst cross-cluster transfers
        if len(cross_cluster) > 0:
            best_transfer = cross_cluster.loc[cross_cluster['r2'].idxmax()]
            worst_transfer = cross_cluster.loc[cross_cluster['r2'].idxmin()]
            
            summary['best_cross_cluster_transfer'] = {
                'source_cluster': int(best_transfer['source_cluster']),
                'target_cluster': int(best_transfer['target_cluster']),
                'r2': float(best_transfer['r2']),
                'rmse': float(best_transfer['rmse'])
            }
            
            summary['worst_cross_cluster_transfer'] = {
                'source_cluster': int(worst_transfer['source_cluster']),
                'target_cluster': int(worst_transfer['target_cluster']),
                'r2': float(worst_transfer['r2']),
                'rmse': float(worst_transfer['rmse'])
            }
        
        # Save summary
        summary_file = os.path.join(self.results_dir, f'cross_cluster_summary_{self.timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Summary analysis saved: {summary_file}")
        
        # Print key findings
        self.print_key_findings(summary)
    
    def print_key_findings(self, summary):
        """Print key findings from the analysis"""
        print("\n" + "="*60)
        print("🔍 CROSS-CLUSTER GENERALIZATION ANALYSIS RESULTS")
        print("="*60)
        
        within = summary['within_cluster_performance']
        cross = summary['cross_cluster_performance']
        
        print(f"\n📊 WITHIN-CLUSTER PERFORMANCE (Baseline):")
        print(f"   Mean R²: {within['mean_r2']:.4f} ± {within['std_r2']:.4f}")
        print(f"   Mean RMSE: {within['mean_rmse']:.4f}")
        print(f"   Range: R² = {within['worst_r2']:.4f} to {within['best_r2']:.4f}")
        
        print(f"\n🔄 CROSS-CLUSTER PERFORMANCE:")
        print(f"   Mean R²: {cross['mean_r2']:.4f} ± {cross['std_r2']:.4f}")
        print(f"   Mean RMSE: {cross['mean_rmse']:.4f}")
        print(f"   Range: R² = {cross['worst_r2']:.4f} to {cross['best_r2']:.4f}")
        print(f"   Catastrophic Failures (R² < -10): {cross['catastrophic_failures']}")
        print(f"   Negative R² count: {cross['negative_r2_count']}")
        
        if 'performance_drop' in summary:
            drop = summary['performance_drop']
            print(f"\n📉 PERFORMANCE DROP:")
            print(f"   R² Drop (Mean): {drop['r2_drop_mean']:.4f}")
            print(f"   R² Drop (Median): {drop['r2_drop_median']:.4f}")
            print(f"   RMSE Increase: {drop['rmse_increase_mean']:.4f}")
        
        if 'best_cross_cluster_transfer' in summary:
            best = summary['best_cross_cluster_transfer']
            worst = summary['worst_cross_cluster_transfer']
            
            print(f"\n🏆 BEST CROSS-CLUSTER TRANSFER:")
            print(f"   Cluster {best['source_cluster']} → Cluster {best['target_cluster']}: R² = {best['r2']:.4f}")
            
            print(f"\n💥 WORST CROSS-CLUSTER TRANSFER:")
            print(f"   Cluster {worst['source_cluster']} → Cluster {worst['target_cluster']}: R² = {worst['r2']:.4f}")
            
            if worst['r2'] < -10:
                print(f"   🚨 This represents CATASTROPHIC FAILURE!")
    
    def create_generalization_heatmap(self, df_results):
        """Create heatmap showing cross-cluster generalization performance"""
        try:
            # Create R² matrix
            clusters = sorted(df_results['source_cluster'].unique())
            r2_matrix = np.full((len(clusters), len(clusters)), np.nan)
            
            for _, row in df_results.iterrows():
                source_idx = clusters.index(row['source_cluster'])
                target_idx = clusters.index(row['target_cluster'])
                r2_matrix[source_idx, target_idx] = row['r2']
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            # Custom colormap: red for negative, white for 0, green for positive
            cmap = plt.cm.RdYlGn
            vmin = max(-2, np.nanmin(r2_matrix))  # Cap extremely negative values for visualization
            vmax = min(1, np.nanmax(r2_matrix))
            
            heatmap = plt.imshow(r2_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            plt.colorbar(heatmap, label='R² Score')
            
            # Labels and formatting
            plt.xticks(range(len(clusters)), [f'Cluster {c}' for c in clusters], rotation=45)
            plt.yticks(range(len(clusters)), [f'Cluster {c}' for c in clusters])
            plt.xlabel('Target Cluster (being predicted)')
            plt.ylabel('Source Cluster (model trained on)')
            plt.title('Cross-Cluster Generalization Performance\n(R² scores when applying each cluster model to other clusters)')
            
            # Add text annotations
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    if not np.isnan(r2_matrix[i, j]):
                        color = 'white' if abs(r2_matrix[i, j]) > 0.5 else 'black'
                        plt.text(j, i, f'{r2_matrix[i, j]:.3f}', 
                               ha='center', va='center', color=color, fontsize=9)
            
            plt.tight_layout()
            
            # Save figure
            heatmap_file = os.path.join(self.results_dir, f'cross_cluster_heatmap_{self.timestamp}.png')
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Generalization heatmap saved: {heatmap_file}")
            
        except Exception as e:
            print(f"⚠️  Could not create heatmap: {e}")

def main():
    analyzer = CrossClusterGeneralizationAnalyzer()
    analyzer.run_cross_cluster_analysis()

if __name__ == "__main__":
    main() 