import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

RESULTS_DIR = '../evaluation/clustering_results'
PARQUET_DIR = '../../processed_parquet'
CLUSTER_COL = 'ecosystem_cluster'

# Find the latest cluster assignments CSV
csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'advanced_site_clusters_*.csv')))
if not csv_files:
    raise FileNotFoundError('No advanced_site_clusters_*.csv found in ./results')
latest_csv = csv_files[-1]
print(f"Using cluster assignments from: {latest_csv}")

# Read cluster assignments
clusters = pd.read_csv(latest_csv)
if 'site' not in clusters.columns or 'cluster' not in clusters.columns:
    raise ValueError('CSV must contain site and cluster columns')

# For each site, append cluster label to parquet
for _, row in clusters.iterrows():
    site = row['site']
    cluster = row['cluster']
    parquet_path = os.path.join(PARQUET_DIR, f'{site}_comprehensive.parquet')
    if not os.path.exists(parquet_path):
        print(f"[WARN] Parquet file not found for site: {site}")
        continue
    print(f"Updating {parquet_path} with {CLUSTER_COL}={cluster}")
    df = pd.read_parquet(parquet_path)
    df[CLUSTER_COL] = cluster
    df.to_parquet(parquet_path, index=False)
print("Done. All parquet files updated with ecosystem_cluster label.") 