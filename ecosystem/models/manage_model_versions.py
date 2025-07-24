"""
Model Version Management Utility for Cluster-Specific Training
Handles cleanup and selection of multiple model versions per cluster
"""

import os
import glob
import json
import pandas as pd
from datetime import datetime
import argparse
import shutil

class ModelVersionManager:
    """Manages multiple versions of cluster-specific models"""
    
    def __init__(self, models_dir='./results/cluster_models'):
        self.models_dir = models_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def analyze_existing_models(self):
        """Analyze all existing model files and versions"""
        print("🔍 Analyzing existing model versions...")
        
        model_files = glob.glob(os.path.join(self.models_dir, 'xgb_model_cluster_*.json'))
        
        if not model_files:
            print("✅ No existing models found - clean slate!")
            return {}
        
        # Group models by cluster
        models_by_cluster = {}
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            # Format: xgb_model_cluster_{cluster_id}_{timestamp}.json
            parts = filename.replace('.json', '').split('_')
            
            cluster_id = None
            timestamp = None
            
            for i, part in enumerate(parts):
                if part == 'cluster' and i + 1 < len(parts):
                    try:
                        cluster_id = int(parts[i + 1])
                        if i + 2 < len(parts):
                            timestamp = parts[i + 2]
                        break
                    except ValueError:
                        continue
            
            if cluster_id is not None:
                if cluster_id not in models_by_cluster:
                    models_by_cluster[cluster_id] = []
                
                file_info = {
                    'path': model_file,
                    'filename': filename,
                    'timestamp': timestamp,
                    'size_mb': os.path.getsize(model_file) / (1024**2),
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(model_file))
                }
                
                # Try to load metrics if available
                metrics_file = model_file.replace('xgb_model_', 'cluster_model_metrics_').replace('.json', '.csv')
                if os.path.exists(metrics_file):
                    try:
                        metrics_df = pd.read_csv(metrics_file)
                        cluster_metrics = metrics_df[metrics_df['cluster'] == cluster_id]
                        if len(cluster_metrics) > 0:
                            file_info['test_r2'] = cluster_metrics.iloc[0]['test_r2']
                            file_info['test_rmse'] = cluster_metrics.iloc[0]['test_rmse']
                    except:
                        pass
                
                models_by_cluster[cluster_id].append(file_info)
        
        # Sort each cluster's models by timestamp (newest first)
        for cluster_id in models_by_cluster:
            models_by_cluster[cluster_id].sort(
                key=lambda x: x['timestamp'] if x['timestamp'] else '0', 
                reverse=True
            )
        
        return models_by_cluster
    
    def print_model_summary(self, models_by_cluster):
        """Print a summary of all model versions"""
        print(f"\n📊 MODEL VERSION SUMMARY")
        print("=" * 60)
        
        total_models = sum(len(versions) for versions in models_by_cluster.values())
        total_size_mb = sum(
            sum(model['size_mb'] for model in versions) 
            for versions in models_by_cluster.values()
        )
        
        print(f"Total model files: {total_models}")
        print(f"Total storage: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        print(f"Clusters with models: {len(models_by_cluster)}")
        
        duplicates = {k: len(v) for k, v in models_by_cluster.items() if len(v) > 1}
        if duplicates:
            print(f"⚠️  Clusters with multiple versions: {len(duplicates)}")
            for cluster_id, count in duplicates.items():
                print(f"   Cluster {cluster_id}: {count} versions")
        else:
            print("✅ No duplicate models found")
        
        print(f"\nDETAILED BREAKDOWN:")
        print("-" * 60)
        
        for cluster_id in sorted(models_by_cluster.keys()):
            versions = models_by_cluster[cluster_id]
            print(f"\n🏷️  Cluster {cluster_id} ({len(versions)} versions):")
            
            for i, model in enumerate(versions):
                status = "🆕 LATEST" if i == 0 else f"📅 #{i+1}"
                metrics_str = ""
                if 'test_r2' in model:
                    metrics_str = f" (R²={model['test_r2']:.4f}, RMSE={model['test_rmse']:.4f})"
                
                print(f"   {status} {model['filename']} - {model['size_mb']:.1f}MB{metrics_str}")
                print(f"        Modified: {model['modified_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    def clean_old_models(self, keep_latest=True, backup=True):
        """Clean up old model versions"""
        models_by_cluster = self.analyze_existing_models()
        
        if not models_by_cluster:
            print("✅ No models to clean up!")
            return
        
        # Count what will be cleaned
        total_to_remove = 0
        clusters_affected = 0
        
        for cluster_id, versions in models_by_cluster.items():
            if len(versions) > 1:
                clusters_affected += 1
                if keep_latest:
                    total_to_remove += len(versions) - 1
                else:
                    total_to_remove += len(versions)
        
        if total_to_remove == 0:
            print("✅ No duplicate models to clean up!")
            return
        
        print(f"\n🧹 CLEANUP PLAN:")
        print(f"   Clusters affected: {clusters_affected}")
        print(f"   Models to remove: {total_to_remove}")
        
        # Create backup if requested
        if backup and total_to_remove > 0:
            backup_dir = os.path.join(self.models_dir, f'backup_{self.timestamp}')
            os.makedirs(backup_dir, exist_ok=True)
            print(f"📦 Creating backup in: {backup_dir}")
        
        # Perform cleanup
        removed_count = 0
        
        for cluster_id, versions in models_by_cluster.items():
            if len(versions) <= 1:
                continue  # Skip clusters with only one model
            
            models_to_remove = versions[1:] if keep_latest else versions
            
            for model in models_to_remove:
                try:
                    # Backup if requested
                    if backup:
                        backup_path = os.path.join(backup_dir, model['filename'])
                        shutil.copy2(model['path'], backup_path)
                    
                    # Remove original
                    os.remove(model['path'])
                    
                    # Also remove associated files (metrics, importance)
                    base_name = model['filename'].replace('xgb_model_', '').replace('.json', '')
                    associated_patterns = [
                        f'feature_importance_{base_name}.csv',
                        f'cluster_model_metrics_{base_name}.csv'
                    ]
                    
                    for pattern in associated_patterns:
                        associated_file = os.path.join(self.models_dir, pattern)
                        if os.path.exists(associated_file):
                            if backup:
                                shutil.copy2(associated_file, os.path.join(backup_dir, pattern))
                            os.remove(associated_file)
                    
                    print(f"   ✅ Removed: {model['filename']}")
                    removed_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error removing {model['filename']}: {e}")
        
        print(f"\n🎉 Cleanup completed!")
        print(f"   Removed: {removed_count} model files")
        if backup:
            print(f"   Backup saved to: {backup_dir}")
    
    def select_best_models(self, metric='test_r2', higher_is_better=True):
        """Keep only the best model per cluster based on a metric"""
        models_by_cluster = self.analyze_existing_models()
        
        if not models_by_cluster:
            print("✅ No models found!")
            return
        
        print(f"\n🏆 SELECTING BEST MODELS (by {metric})...")
        
        removed_count = 0
        
        for cluster_id, versions in models_by_cluster.items():
            if len(versions) <= 1:
                continue  # Skip clusters with only one model
            
            # Find models with the required metric
            models_with_metric = [m for m in versions if metric in m]
            
            if not models_with_metric:
                print(f"   ⚠️  Cluster {cluster_id}: No models have {metric} - keeping latest")
                # Keep latest, remove others
                for model in versions[1:]:
                    try:
                        os.remove(model['path'])
                        print(f"   ✅ Removed: {model['filename']}")
                        removed_count += 1
                    except Exception as e:
                        print(f"   ❌ Error removing {model['filename']}: {e}")
                continue
            
            # Sort by metric
            models_with_metric.sort(
                key=lambda x: x[metric], 
                reverse=higher_is_better
            )
            
            best_model = models_with_metric[0]
            print(f"   🏆 Cluster {cluster_id}: Best {metric} = {best_model[metric]:.4f} ({best_model['filename']})")
            
            # Remove all others
            for model in versions:
                if model['path'] != best_model['path']:
                    try:
                        os.remove(model['path'])
                        print(f"      ✅ Removed: {model['filename']}")
                        removed_count += 1
                    except Exception as e:
                        print(f"      ❌ Error removing {model['filename']}: {e}")
        
        print(f"\n🎉 Best model selection completed!")
        print(f"   Removed: {removed_count} suboptimal models")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Version Management")
    parser.add_argument('--action', choices=['analyze', 'clean', 'select-best'], 
                       default='analyze', help="Action to perform")
    parser.add_argument('--models-dir', default='./results/cluster_models',
                       help="Directory containing cluster models")
    parser.add_argument('--keep-latest', action='store_true', default=True,
                       help="Keep latest model when cleaning (default: True)")
    parser.add_argument('--no-backup', action='store_true', 
                       help="Don't create backup when cleaning")
    parser.add_argument('--metric', default='test_r2', 
                       help="Metric to use for best model selection")
    
    args = parser.parse_args()
    
    manager = ModelVersionManager(models_dir=args.models_dir)
    
    if args.action == 'analyze':
        models_by_cluster = manager.analyze_existing_models()
        manager.print_model_summary(models_by_cluster)
        
    elif args.action == 'clean':
        manager.clean_old_models(
            keep_latest=args.keep_latest, 
            backup=not args.no_backup
        )
        
    elif args.action == 'select-best':
        manager.select_best_models(metric=args.metric)

if __name__ == "__main__":
    main() 