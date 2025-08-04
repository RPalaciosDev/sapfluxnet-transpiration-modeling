#!/usr/bin/env python3
"""
List available clustering results for use with train_cluster_models.py
"""

import os
import glob
import json
from datetime import datetime
from pathlib import Path


def list_clustering_results():
    """List all available clustering results"""
    print("🔍 AVAILABLE CLUSTERING RESULTS")
    print("=" * 60)
    
    clustering_results_dir = '../evaluation/clustering_results'
    
    if not os.path.exists(clustering_results_dir):
        print(f"❌ Clustering results directory not found: {clustering_results_dir}")
        print("   Please run clustering first using FlexibleClusteringPipeline.py")
        return
    
    # Find all clustering result directories
    result_dirs = []
    for item in os.listdir(clustering_results_dir):
        item_path = os.path.join(clustering_results_dir, item)
        if os.path.isdir(item_path):
            # Look for flexible_site_clusters_*.csv files
            csv_files = glob.glob(os.path.join(item_path, 'flexible_site_clusters_*.csv'))
            if csv_files:
                result_dirs.append((item, item_path, csv_files[0]))
    
    if not result_dirs:
        print("❌ No clustering results found")
        print("   Please run clustering first using FlexibleClusteringPipeline.py")
        return
    
    # Sort by modification time (newest first)
    result_dirs.sort(key=lambda x: os.path.getmtime(x[2]), reverse=True)
    
    print(f"📊 Found {len(result_dirs)} clustering results:\n")
    
    for i, (dir_name, dir_path, csv_file) in enumerate(result_dirs, 1):
        # Parse directory name to get feature set and timestamp
        parts = dir_name.split('_')
        if len(parts) >= 2:
            feature_set = parts[0]
            timestamp_str = '_'.join(parts[1:])
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = timestamp_str
        else:
            feature_set = dir_name
            time_str = "Unknown"
        
        # Get cluster info from CSV
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            n_sites = len(df)
            n_clusters = df['cluster'].nunique()
            cluster_dist = df['cluster'].value_counts().sort_index().to_dict()
        except:
            n_sites = "Unknown"
            n_clusters = "Unknown"
            cluster_dist = {}
        
        # Try to get additional info from strategy JSON
        strategy_files = glob.glob(os.path.join(dir_path, 'flexible_clustering_strategy_*.json'))
        strategy_info = ""
        if strategy_files:
            try:
                with open(strategy_files[0], 'r') as f:
                    strategy = json.load(f)
                    method = strategy.get('selected_strategy', 'Unknown')
                    silhouette = strategy.get('silhouette_score', 'Unknown')
                    balance = strategy.get('balance_ratio', 'Unknown')
                    strategy_info = f" | Method: {method} | Silhouette: {silhouette:.3f} | Balance: {balance:.3f}"
            except:
                pass
        
        print(f"{i:2d}. 🎯 Feature Set: {feature_set.upper()}")
        print(f"    📁 Directory: {dir_name}")
        print(f"    ⏰ Created: {time_str}")
        print(f"    📊 Sites: {n_sites} | Clusters: {n_clusters}{strategy_info}")
        if cluster_dist:
            dist_str = ", ".join([f"C{k}: {v}" for k, v in cluster_dist.items()])
            print(f"    📈 Distribution: {dist_str}")
        print(f"    📄 CSV File: {os.path.basename(csv_file)}")
        print(f"    🔗 Full Path: {csv_file}")
        print()
    
    print("💡 USAGE EXAMPLES:")
    print("=" * 20)
    
    if result_dirs:
        # Show example with most recent biome clustering
        biome_results = [r for r in result_dirs if 'biome' in r[0].lower()]
        if biome_results:
            example_file = biome_results[0][2]
            print(f"📋 Use the latest BIOME clustering:")
            print(f"   python train_cluster_models.py --cluster-file \"{example_file}\"")
            print()
        
        # Show example with most recent result
        latest_file = result_dirs[0][2]
        latest_feature_set = result_dirs[0][0].split('_')[0]
        print(f"📋 Use the latest {latest_feature_set.upper()} clustering:")
        print(f"   python train_cluster_models.py --cluster-file \"{latest_file}\"")
        print()
        
        print(f"📋 Use automatic (most recent) clustering:")
        print(f"   python train_cluster_models.py")
        print()
    
    print("🔧 OTHER OPTIONS:")
    print("=" * 15)
    print("📋 Preprocess data first (recommended workflow):")
    print("   python preprocess_cluster_data.py --cluster-csv \"path/to/clusters.csv\"")
    print("   python train_cluster_models.py --mode train")
    print()
    print("📋 Run full pipeline:")
    print("   python train_cluster_models.py --mode both --cluster-file \"path/to/clusters.csv\"")


if __name__ == "__main__":
    list_clustering_results()