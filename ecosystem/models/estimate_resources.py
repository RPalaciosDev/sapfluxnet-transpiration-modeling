#!/usr/bin/env python3
"""
Resource Estimation for SLURM Hyperparameter Optimization
Estimates memory, CPU, and time requirements before job submission
"""

import pandas as pd
import os
import glob
from pathlib import Path

def estimate_hyperopt_resources():
    """Estimate resource requirements for hyperparameter optimization"""
    
    print("🔍 Estimating resource requirements for hyperparameter optimization...")
    print("=" * 60)
    
    # Check if cluster assignments exist
    cluster_files = sorted(glob.glob('../evaluation/clustering_results/advanced_site_clusters_*.csv'))
    if not cluster_files:
        print("❌ No cluster assignment files found")
        return
    
    latest_file = cluster_files[-1]
    clusters_df = pd.read_csv(latest_file)
    cluster_counts = clusters_df['cluster'].value_counts().sort_index()
    
    print(f"📊 Cluster Analysis:")
    print(f"   Total sites: {len(clusters_df)}")
    print(f"   Number of clusters: {len(cluster_counts)}")
    for cluster_id, count in cluster_counts.items():
        print(f"   Cluster {cluster_id}: {count} sites")
    
    # Estimate data size
    parquet_dir = Path('../../processed_parquet')
    if not parquet_dir.exists():
        print(f"❌ Parquet directory not found: {parquet_dir}")
        return
    
    parquet_files = list(parquet_dir.glob('*_comprehensive.parquet'))
    print(f"\n💾 Data Analysis:")
    print(f"   Available parquet files: {len(parquet_files)}")
    
    # Sample a few files to estimate size
    total_size_mb = 0
    sample_files = parquet_files[:5] if len(parquet_files) > 5 else parquet_files
    
    for file_path in sample_files:
        try:
            file_size_mb = file_path.stat().st_size / (1024**2)
            total_size_mb += file_size_mb
            print(f"   {file_path.name}: {file_size_mb:.1f} MB")
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")
    
    if sample_files:
        avg_file_size_mb = total_size_mb / len(sample_files)
        estimated_total_size_gb = (avg_file_size_mb * len(parquet_files)) / 1024
        print(f"   Average file size: {avg_file_size_mb:.1f} MB")
        print(f"   Estimated total dataset: {estimated_total_size_gb:.1f} GB")
    
    # Resource calculations
    print(f"\n🧮 Resource Estimates:")
    
    # Memory estimation
    n_clusters = len(cluster_counts)
    samples_per_cluster = 50000  # 5 sites × 10K samples
    features_per_sample = 272
    
    # Data memory (float64 = 8 bytes)
    data_memory_mb = (samples_per_cluster * features_per_sample * 8) / (1024**2)
    
    # XGBoost + Optuna overhead per cluster
    overhead_mb = 200
    
    # Total memory per cluster
    memory_per_cluster_mb = data_memory_mb + overhead_mb
    
    # Total memory needed (assuming sequential processing)
    total_memory_mb = memory_per_cluster_mb * n_clusters
    total_memory_gb = total_memory_mb / 1024
    
    print(f"   Data per cluster: {data_memory_mb:.0f} MB")
    print(f"   Overhead per cluster: {overhead_mb} MB")
    print(f"   Memory per cluster: {memory_per_cluster_mb:.0f} MB")
    print(f"   Total memory needed: {total_memory_gb:.1f} GB")
    
    # CPU estimation
    n_trials_per_cluster = 100
    total_trials = n_trials_per_cluster * n_clusters
    
    # Time estimation (based on typical XGBoost performance)
    time_per_trial_minutes = 2  # Conservative estimate
    sequential_time_hours = (total_trials * time_per_trial_minutes) / 60
    
    print(f"\n⏱️  Time Estimates:")
    print(f"   Trials per cluster: {n_trials_per_cluster}")
    print(f"   Total trials: {total_trials}")
    print(f"   Sequential time: {sequential_time_hours:.1f} hours")
    
    # Parallel efficiency estimates
    for n_cores in [8, 16, 24, 32]:
        parallel_time_hours = sequential_time_hours / n_cores
        efficiency = min(1.0, 0.9 - (n_cores - 8) * 0.05)  # Diminishing returns
        realistic_time_hours = parallel_time_hours / efficiency
        
        print(f"   With {n_cores:2d} cores: {realistic_time_hours:.1f} hours (efficiency: {efficiency:.0%})")
    
    # SLURM recommendations
    print(f"\n🎯 SLURM Resource Recommendations:")
    print(f"=" * 40)
    
    # Conservative recommendation
    recommended_memory_gb = max(8, int(total_memory_gb * 1.5))  # 50% buffer
    recommended_cores = 16
    recommended_time_hours = max(2, int(sequential_time_hours / recommended_cores * 1.5))  # 50% buffer
    
    print(f"📋 CONSERVATIVE (Recommended):")
    print(f"   --cpus-per-task={recommended_cores}")
    print(f"   --mem={recommended_memory_gb}G")
    print(f"   --time={recommended_time_hours:02d}:00:00")
    print(f"   Expected runtime: ~{recommended_time_hours * 0.7:.1f} hours")
    
    # Aggressive recommendation
    aggressive_cores = 24
    aggressive_memory_gb = max(12, recommended_memory_gb)
    aggressive_time_hours = max(1, recommended_time_hours - 1)
    
    print(f"\n🚀 AGGRESSIVE (If resources available):")
    print(f"   --cpus-per-task={aggressive_cores}")
    print(f"   --mem={aggressive_memory_gb}G")
    print(f"   --time={aggressive_time_hours:02d}:30:00")
    print(f"   Expected runtime: ~{aggressive_time_hours * 0.6:.1f} hours")
    
    # Minimal recommendation
    minimal_cores = 8
    minimal_memory_gb = max(4, int(total_memory_gb))
    minimal_time_hours = recommended_time_hours + 2
    
    print(f"\n💰 MINIMAL (Budget-friendly):")
    print(f"   --cpus-per-task={minimal_cores}")
    print(f"   --mem={minimal_memory_gb}G")
    print(f"   --time={minimal_time_hours:02d}:00:00")
    print(f"   Expected runtime: ~{minimal_time_hours * 0.8:.1f} hours")
    
    # GPU check
    print(f"\n🎮 GPU Requirements:")
    print(f"   GPU needed: ❌ NO")
    print(f"   Reason: Dataset size ({samples_per_cluster:,} samples, {features_per_sample} features)")
    print(f"           is too small to benefit from GPU acceleration")
    print(f"   CPU XGBoost will be faster for this workload")
    
    # Final recommendations
    print(f"\n💡 Final Recommendations:")
    print(f"   1. Use CPU partition only (no GPU needed)")
    print(f"   2. Start with CONSERVATIVE settings")
    print(f"   3. Monitor first job to optimize future runs")
    print(f"   4. Consider reducing --n-trials to 50 for faster testing")
    
    # Generate SLURM script snippet
    print(f"\n📝 Copy-paste for your SLURM script:")
    print(f"#SBATCH --cpus-per-task={recommended_cores}")
    print(f"#SBATCH --mem={recommended_memory_gb}G")
    print(f"#SBATCH --time={recommended_time_hours:02d}:00:00")
    print(f"#SBATCH --partition=cpu")

if __name__ == "__main__":
    estimate_hyperopt_resources() 