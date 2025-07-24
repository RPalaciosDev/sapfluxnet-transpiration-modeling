"""
Outlier Detection and Ensemble Prediction for SAPFLUXNET Ecosystem Models

This script addresses the question: "What happens if we try to classify a site that 
doesn't have good representation within any of our clusters?"

Key features:
1. Outlier detection using clustering uncertainty metrics
2. Ensemble prediction that weights multiple cluster models based on site similarity
3. Uncertainty quantification for predictions
4. Special handling for problematic sites like COL_MAC_SAF_RAD
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
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

class OutlierDetectionAndEnsemble:
    """
    Detects outlier sites and provides ensemble predictions for uncertain classifications
    """
    
    def __init__(self, parquet_dir='../../processed_parquet', 
                 models_dir='./results/cluster_models',
                 results_dir='./results/outlier_analysis'):
        self.parquet_dir = parquet_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.cluster_col = 'ecosystem_cluster'
        self.target_col = 'sap_flow'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🔍 Outlier Detection and Ensemble Predictor initialized")
        print(f"📁 Results directory: {results_dir}")
    
    def load_cluster_assignments_and_features(self):
        """Load cluster assignments and clustering features"""
        cluster_files = sorted(glob.glob('../evaluation/clustering_results/advanced_site_clusters_*.csv'))
        
        if not cluster_files:
            raise FileNotFoundError("No cluster assignment files found")
        
        latest_file = cluster_files[-1]
        print(f"📊 Loading cluster data from: {os.path.basename(latest_file)}")
        
        # Load the full clustering data with features
        clusters_df = pd.read_csv(latest_file)
        
        print(f"✅ Loaded {len(clusters_df)} sites with clustering features")
        
        return clusters_df
    
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
        
        print(f"📊 Loaded {len(models)} cluster models")
        return models
    
    def calculate_clustering_uncertainty(self, clusters_df):
        """Calculate uncertainty metrics for each site's cluster assignment"""
        print("\n🔍 Calculating clustering uncertainty metrics...")
        
        # Get clustering features (exclude site, cluster, cluster_method)
        feature_cols = [col for col in clusters_df.columns 
                       if col not in ['site', 'cluster', 'cluster_method']]
        
        print(f"   Using {len(feature_cols)} clustering features")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clusters_df[feature_cols].fillna(0))
        
        # Calculate cluster centroids
        cluster_centroids = {}
        for cluster_id in clusters_df['cluster'].unique():
            cluster_mask = clusters_df['cluster'] == cluster_id
            cluster_centroids[cluster_id] = X_scaled[cluster_mask].mean(axis=0)
        
        # Calculate uncertainty metrics for each site
        uncertainty_metrics = []
        
        for idx, row in clusters_df.iterrows():
            site = row['site']
            assigned_cluster = row['cluster']
            site_features = X_scaled[idx].reshape(1, -1)
            
            # Distance to assigned cluster centroid
            assigned_centroid = cluster_centroids[assigned_cluster].reshape(1, -1)
            distance_to_assigned = cdist(site_features, assigned_centroid)[0][0]
            
            # Distance to all cluster centroids
            distances_to_all = {}
            for cluster_id, centroid in cluster_centroids.items():
                centroid_reshaped = centroid.reshape(1, -1)
                distances_to_all[cluster_id] = cdist(site_features, centroid_reshaped)[0][0]
            
            # Find closest alternative cluster
            other_clusters = [c for c in distances_to_all.keys() if c != assigned_cluster]
            if other_clusters:
                closest_alternative = min(other_clusters, key=lambda c: distances_to_all[c])
                distance_to_closest_alternative = distances_to_all[closest_alternative]
                
                # Separation ratio: how much closer to assigned vs. next best
                separation_ratio = distance_to_closest_alternative / distance_to_assigned
            else:
                closest_alternative = None
                distance_to_closest_alternative = np.inf
                separation_ratio = np.inf
            
            # Classification confidence (inverse of distance to assigned)
            confidence = 1 / (1 + distance_to_assigned)
            
            # Outlier score (relative distance to assigned cluster)
            cluster_distances = [distances_to_all[c] for c in distances_to_all.keys()]
            outlier_score = distance_to_assigned / np.mean(cluster_distances)
            
            uncertainty_metrics.append({
                'site': site,
                'assigned_cluster': assigned_cluster,
                'distance_to_assigned': distance_to_assigned,
                'closest_alternative_cluster': closest_alternative,
                'distance_to_closest_alternative': distance_to_closest_alternative,
                'separation_ratio': separation_ratio,
                'confidence': confidence,
                'outlier_score': outlier_score
            })
        
        uncertainty_df = pd.DataFrame(uncertainty_metrics)
        
        print(f"✅ Calculated uncertainty metrics for {len(uncertainty_df)} sites")
        
        return uncertainty_df, scaler, cluster_centroids
    
    def identify_outlier_sites(self, uncertainty_df, outlier_threshold=2.0, confidence_threshold=0.3):
        """Identify sites that are likely outliers or poorly classified"""
        print(f"\n🚨 Identifying outlier sites...")
        print(f"   Outlier score threshold: {outlier_threshold}")
        print(f"   Confidence threshold: {confidence_threshold}")
        
        # Multiple criteria for outlier detection
        outliers = uncertainty_df[
            (uncertainty_df['outlier_score'] > outlier_threshold) |
            (uncertainty_df['confidence'] < confidence_threshold) |
            (uncertainty_df['separation_ratio'] < 1.2)  # Very close to alternative cluster
        ].copy()
        
        print(f"🔍 Found {len(outliers)} potential outlier sites:")
        
        outliers_sorted = outliers.sort_values('outlier_score', ascending=False)
        for _, row in outliers_sorted.iterrows():
            print(f"   {row['site']}: outlier_score={row['outlier_score']:.3f}, "
                  f"confidence={row['confidence']:.3f}, "
                  f"separation_ratio={row['separation_ratio']:.3f}")
        
        return outliers
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id', 'Unnamed: 0', self.cluster_col]
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [self.target_col]
                       and not col.endswith('_flags')
                       and not col.endswith('_md')]
        
        X_df = df[feature_cols].copy()
        
        # Convert boolean columns to numeric
        for col in X_df.columns:
            if X_df[col].dtype == bool:
                X_df[col] = X_df[col].astype(int)
            elif X_df[col].dtype == 'object':
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        X = X_df.fillna(0).values
        y = df[self.target_col].values if self.target_col in df.columns else None
        
        return X, y, feature_cols
    
    def ensemble_predict(self, site_data, models, uncertainty_metrics):
        """Make ensemble predictions using multiple cluster models weighted by similarity"""
        print(f"\n🔮 Making ensemble prediction for site with uncertain classification...")
        
        # Prepare features
        X, y_true, feature_cols = self.prepare_features(site_data)
        
        if len(X) == 0:
            return None
        
        # Get weights for each cluster model based on similarity
        site_name = site_data['site'].iloc[0] if 'site' in site_data.columns else 'unknown'
        
        # Find this site's uncertainty metrics
        site_uncertainty = uncertainty_metrics[uncertainty_metrics['site'] == site_name]
        
        if len(site_uncertainty) == 0:
            print(f"  ⚠️  No uncertainty metrics found for {site_name}, using equal weights")
            weights = {cluster_id: 1.0 for cluster_id in models.keys()}
        else:
            # Calculate weights based on distance to each cluster
            site_metrics = site_uncertainty.iloc[0]
            assigned_cluster = site_metrics['assigned_cluster']
            
            weights = {}
            total_weight = 0
            
            for cluster_id in models.keys():
                if cluster_id == assigned_cluster:
                    # Higher weight for assigned cluster, but not exclusive
                    weight = 1.0 / (1.0 + site_metrics['distance_to_assigned'])
                else:
                    # Weight inversely proportional to distance (we don't have all distances stored)
                    # Use a default lower weight for non-assigned clusters
                    weight = 0.3  # Base weight for other clusters
                
                weights[cluster_id] = weight
                total_weight += weight
            
            # Normalize weights
            weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"  Cluster weights: {weights}")
        
        # Make predictions with each model
        predictions = {}
        individual_results = {}
        
        for cluster_id, model in models.items():
            try:
                dtest = xgb.DMatrix(X)
                y_pred = model.predict(dtest)
                predictions[cluster_id] = y_pred
                
                if y_true is not None:
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    individual_results[cluster_id] = {'r2': r2, 'rmse': rmse}
                    print(f"    Cluster {cluster_id}: R² = {r2:.4f}, RMSE = {rmse:.4f}, Weight = {weights[cluster_id]:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with cluster {cluster_id} model: {e}")
                continue
        
        if not predictions:
            return None
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for cluster_id, pred in predictions.items():
            ensemble_pred += weights[cluster_id] * pred
        
        # Calculate ensemble metrics
        ensemble_results = {'ensemble_prediction': ensemble_pred}
        
        if y_true is not None:
            ensemble_r2 = r2_score(y_true, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            
            ensemble_results.update({
                'ensemble_r2': ensemble_r2,
                'ensemble_rmse': ensemble_rmse,
                'individual_results': individual_results,
                'weights': weights,
                'n_samples': len(y_true)
            })
            
            print(f"  🎯 Ensemble: R² = {ensemble_r2:.4f}, RMSE = {ensemble_rmse:.4f}")
            
            # Compare to best individual model
            if individual_results:
                best_individual_r2 = max(result['r2'] for result in individual_results.values())
                improvement = ensemble_r2 - best_individual_r2
                print(f"  📈 Improvement over best individual: {improvement:+.4f}")
        
        return ensemble_results
    
    def analyze_specific_outlier(self, site_name, models, uncertainty_df):
        """Deep analysis of a specific outlier site"""
        print(f"\n🔬 Deep analysis of outlier site: {site_name}")
        
        # Load site data
        parquet_file = os.path.join(self.parquet_dir, f'{site_name}_comprehensive.parquet')
        
        if not os.path.exists(parquet_file):
            print(f"❌ Data file not found: {parquet_file}")
            return None
        
        try:
            # Load full site data
            site_data = pd.read_parquet(parquet_file)
            site_data = site_data.dropna(subset=[self.target_col])
            
            if len(site_data) == 0:
                print(f"❌ No valid data found for {site_name}")
                return None
            
            print(f"📊 Loaded {len(site_data):,} samples for {site_name}")
            
            # Get uncertainty metrics for this site
            site_uncertainty = uncertainty_df[uncertainty_df['site'] == site_name]
            if len(site_uncertainty) > 0:
                metrics = site_uncertainty.iloc[0]
                print(f"🔍 Uncertainty metrics:")
                print(f"   Assigned cluster: {metrics['assigned_cluster']}")
                print(f"   Outlier score: {metrics['outlier_score']:.3f}")
                print(f"   Confidence: {metrics['confidence']:.3f}")
                print(f"   Separation ratio: {metrics['separation_ratio']:.3f}")
                print(f"   Closest alternative: Cluster {metrics['closest_alternative_cluster']}")
            
            # Test with all cluster models
            print(f"\n🧪 Testing {site_name} with all cluster models:")
            
            results = {}
            X, y_true, feature_cols = self.prepare_features(site_data)
            
            for cluster_id, model in models.items():
                try:
                    dtest = xgb.DMatrix(X)
                    y_pred = model.predict(dtest)
                    
                    r2 = r2_score(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    results[cluster_id] = {
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'n_samples': len(y_true)
                    }
                    
                    status = "🏆 BEST" if r2 == max([r['r2'] for r in results.values()]) else ""
                    if r2 < -10:
                        status = "🚨 CATASTROPHIC"
                    elif r2 < 0:
                        status = "⚠️ NEGATIVE"
                    
                    print(f"   Cluster {cluster_id}: R² = {r2:8.4f}, RMSE = {rmse:6.4f} {status}")
                    
                except Exception as e:
                    print(f"   Cluster {cluster_id}: ❌ Error - {e}")
                    continue
            
            # Ensemble prediction
            ensemble_result = self.ensemble_predict(site_data, models, uncertainty_df)
            
            analysis_result = {
                'site': site_name,
                'n_samples': len(site_data),
                'individual_model_results': results,
                'ensemble_result': ensemble_result,
                'uncertainty_metrics': site_uncertainty.to_dict('records')[0] if len(site_uncertainty) > 0 else None
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ Error analyzing {site_name}: {e}")
            return None
    
    def run_outlier_analysis(self):
        """Run comprehensive outlier detection and ensemble analysis"""
        print("🔍 SAPFLUXNET Outlier Detection and Ensemble Analysis")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print("Purpose: Identify and handle sites that don't fit well into any cluster")
        
        try:
            # Load data
            clusters_df = self.load_cluster_assignments_and_features()
            models = self.load_cluster_models()
            
            # Calculate uncertainty metrics
            uncertainty_df, scaler, centroids = self.calculate_clustering_uncertainty(clusters_df)
            
            # Identify outliers
            outliers = self.identify_outlier_sites(uncertainty_df)
            
            # Analyze specific outliers
            print(f"\n🔬 Detailed analysis of top outliers:")
            
            outlier_analyses = {}
            
            # Always analyze COL_MAC_SAF_RAD if present
            if 'COL_MAC_SAF_RAD' in uncertainty_df['site'].values:
                print(f"\n{'='*40}")
                print(f"SPECIAL ANALYSIS: COL_MAC_SAF_RAD")
                print(f"{'='*40}")
                analysis = self.analyze_specific_outlier('COL_MAC_SAF_RAD', models, uncertainty_df)
                if analysis:
                    outlier_analyses['COL_MAC_SAF_RAD'] = analysis
            
            # Analyze top 3 other outliers
            top_outliers = outliers[outliers['site'] != 'COL_MAC_SAF_RAD'].head(3)
            
            for _, outlier_row in top_outliers.iterrows():
                site_name = outlier_row['site']
                analysis = self.analyze_specific_outlier(site_name, models, uncertainty_df)
                if analysis:
                    outlier_analyses[site_name] = analysis
            
            # Save results
            self.save_outlier_results(uncertainty_df, outliers, outlier_analyses)
            
            print(f"\n✅ Outlier analysis completed!")
            print(f"🔍 Analyzed {len(outliers)} outlier sites")
            
        except Exception as e:
            print(f"❌ Error in outlier analysis: {e}")
            raise
    
    def save_outlier_results(self, uncertainty_df, outliers, analyses):
        """Save outlier analysis results"""
        
        # Save uncertainty metrics for all sites
        uncertainty_file = os.path.join(self.results_dir, f'site_uncertainty_metrics_{self.timestamp}.csv')
        uncertainty_df.to_csv(uncertainty_file, index=False)
        print(f"📄 Site uncertainty metrics saved: {uncertainty_file}")
        
        # Save outlier sites
        outliers_file = os.path.join(self.results_dir, f'outlier_sites_{self.timestamp}.csv')
        outliers.to_csv(outliers_file, index=False)
        print(f"📄 Outlier sites saved: {outliers_file}")
        
        # Save detailed analyses
        if analyses:
            analyses_file = os.path.join(self.results_dir, f'outlier_detailed_analysis_{self.timestamp}.json')
            
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            # Clean the analyses data
            clean_analyses = {}
            for site, analysis in analyses.items():
                clean_analysis = {}
                for key, value in analysis.items():
                    if isinstance(value, dict):
                        clean_analysis[key] = {k: convert_numpy(v) for k, v in value.items()}
                    else:
                        clean_analysis[key] = convert_numpy(value)
                clean_analyses[site] = clean_analysis
            
            with open(analyses_file, 'w') as f:
                json.dump(clean_analyses, f, indent=2, default=convert_numpy)
            print(f"📄 Detailed analyses saved: {analyses_file}")
        
        # Create summary
        self.create_outlier_summary(uncertainty_df, outliers, analyses)
    
    def create_outlier_summary(self, uncertainty_df, outliers, analyses):
        """Create summary of outlier analysis"""
        
        summary = {
            'analysis_timestamp': self.timestamp,
            'total_sites': len(uncertainty_df),
            'outlier_sites': len(outliers),
            'outlier_percentage': len(outliers) / len(uncertainty_df) * 100,
            'uncertainty_statistics': {
                'mean_outlier_score': float(uncertainty_df['outlier_score'].mean()),
                'std_outlier_score': float(uncertainty_df['outlier_score'].std()),
                'mean_confidence': float(uncertainty_df['confidence'].mean()),
                'std_confidence': float(uncertainty_df['confidence'].std()),
                'worst_outlier_score': float(uncertainty_df['outlier_score'].max()),
                'best_confidence': float(uncertainty_df['confidence'].max()),
                'worst_confidence': float(uncertainty_df['confidence'].min())
            }
        }
        
        if analyses:
            summary['detailed_analyses'] = {}
            for site, analysis in analyses.items():
                if 'individual_model_results' in analysis:
                    best_r2 = max([r['r2'] for r in analysis['individual_model_results'].values()])
                    worst_r2 = min([r['r2'] for r in analysis['individual_model_results'].values()])
                    
                    site_summary = {
                        'best_individual_r2': float(best_r2),
                        'worst_individual_r2': float(worst_r2),
                        'r2_range': float(best_r2 - worst_r2)
                    }
                    
                    if 'ensemble_result' in analysis and analysis['ensemble_result']:
                        ensemble = analysis['ensemble_result']
                        if 'ensemble_r2' in ensemble:
                            site_summary['ensemble_r2'] = float(ensemble['ensemble_r2'])
                            site_summary['ensemble_improvement'] = float(ensemble['ensemble_r2'] - best_r2)
                    
                    summary['detailed_analyses'][site] = site_summary
        
        # Save summary
        summary_file = os.path.join(self.results_dir, f'outlier_analysis_summary_{self.timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Analysis summary saved: {summary_file}")
        
        # Print key findings
        self.print_outlier_findings(summary)
    
    def print_outlier_findings(self, summary):
        """Print key findings from outlier analysis"""
        print("\n" + "="*60)
        print("🔍 OUTLIER DETECTION AND ENSEMBLE ANALYSIS RESULTS")
        print("="*60)
        
        stats = summary['uncertainty_statistics']
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total sites: {summary['total_sites']}")
        print(f"   Outlier sites: {summary['outlier_sites']} ({summary['outlier_percentage']:.1f}%)")
        print(f"   Mean outlier score: {stats['mean_outlier_score']:.3f} ± {stats['std_outlier_score']:.3f}")
        print(f"   Mean confidence: {stats['mean_confidence']:.3f} ± {stats['std_confidence']:.3f}")
        
        if 'detailed_analyses' in summary:
            print(f"\n🔬 DETAILED SITE ANALYSES:")
            for site, analysis in summary['detailed_analyses'].items():
                print(f"\n   {site}:")
                print(f"     Individual model R² range: {analysis['worst_individual_r2']:.4f} to {analysis['best_individual_r2']:.4f}")
                if 'ensemble_r2' in analysis:
                    print(f"     Ensemble R²: {analysis['ensemble_r2']:.4f}")
                    print(f"     Ensemble improvement: {analysis['ensemble_improvement']:+.4f}")

def main():
    analyzer = OutlierDetectionAndEnsemble()
    analyzer.run_outlier_analysis()

if __name__ == "__main__":
    main() 