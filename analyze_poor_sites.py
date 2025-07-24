import pandas as pd
import numpy as np

# Load the spatial validation results
df = pd.read_csv('results/cluster_spatial_validation/cluster_spatial_fold_results_20250722_192257.csv')

print('🔍 ANALYZING POORLY PERFORMING SITES')
print('=' * 50)

# Find sites with R² < 0.5 (poor performance)
poor_sites = df[df['test_r2'] < 0.5].copy()
poor_sites = poor_sites.sort_values('test_r2')

print(f'📊 Found {len(poor_sites)} sites with R² < 0.5:')
print()

for idx, row in poor_sites.iterrows():
    print(f'🔴 {row["test_site"]} (Cluster {row["cluster"]}):')
    print(f'   R² = {row["test_r2"]:.4f}')
    print(f'   RMSE = {row["test_rmse"]:.4f}')
    print(f'   Samples = {row["test_samples"]:,}')
    print()

# Analyze by cluster
print('📈 POOR PERFORMANCE BY CLUSTER:')
print('-' * 30)
for cluster in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster]
    poor_in_cluster = cluster_data[cluster_data['test_r2'] < 0.5]
    
    if len(poor_in_cluster) > 0:
        print(f'Cluster {cluster}: {len(poor_in_cluster)}/{len(cluster_data)} poor sites')
        for _, row in poor_in_cluster.iterrows():
            print(f'  - {row["test_site"]}: R² = {row["test_r2"]:.4f}')
    else:
        print(f'Cluster {cluster}: No poor sites (all R² ≥ 0.5)')
    print()

# Find the worst outliers (R² < 0)
extreme_outliers = df[df['test_r2'] < 0].copy()
extreme_outliers = extreme_outliers.sort_values('test_r2')

print('⚠️ EXTREME OUTLIERS (R² < 0):')
print('-' * 30)
for idx, row in extreme_outliers.iterrows():
    print(f'🚨 {row["test_site"]} (Cluster {row["cluster"]}):')
    print(f'   R² = {row["test_r2"]:.4f}')
    print(f'   Train R² = {row["train_r2"]:.4f}')
    print(f'   Samples = {row["test_samples"]:,}')
    print()

# Additional analysis: Look at sample sizes
print('📊 SAMPLE SIZE ANALYSIS:')
print('-' * 25)
small_samples = df[df['test_samples'] < 5000]
if len(small_samples) > 0:
    print(f'Sites with < 5,000 samples: {len(small_samples)}')
    for _, row in small_samples.iterrows():
        print(f'  - {row["test_site"]}: {row["test_samples"]:,} samples, R² = {row["test_r2"]:.4f}')
else:
    print('No sites with very small sample sizes')

# Train vs Test performance gap
print('\n🎯 OVERFITTING ANALYSIS (Large Train-Test R² Gap):')
print('-' * 45)
df['r2_gap'] = df['train_r2'] - df['test_r2']
large_gaps = df[df['r2_gap'] > 0.3].sort_values('r2_gap', ascending=False)

if len(large_gaps) > 0:
    print(f'Sites with train-test R² gap > 0.3: {len(large_gaps)}')
    for _, row in large_gaps.iterrows():
        print(f'  - {row["test_site"]} (Cluster {row["cluster"]}):')
        print(f'    Train R² = {row["train_r2"]:.4f}, Test R² = {row["test_r2"]:.4f}')
        print(f'    Gap = {row["r2_gap"]:.4f}')
        print()
else:
    print('No sites with large overfitting gaps') 