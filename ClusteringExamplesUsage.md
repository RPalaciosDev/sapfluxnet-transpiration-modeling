## Clustering Pipeline: Examples and Usage

Related documents: [Clustering Pipeline Overview](./ClusteringPipelineOverview.md) · [Clustering Feature-Sets Deep Dive](./ClusteringFeatureSetsDeepDive.md)

### Basics

- List available feature sets

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --list-feature-sets
```

- Analyze feature compatibility only (no clustering)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --analyze-only --feature-set climate --data-dir ..\..\processed_parquet
```

- Full clustering with comprehensive features

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set comprehensive --data-dir ..\..\processed_parquet --output-dir ecosystem\evaluation\clustering_results
```

### Feature-set variations

- Climate-focused clustering

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set climate --data-dir ..\..\processed_parquet
```

- Ecological (includes stand structure and species traits)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set ecological --data-dir ..\..\processed_parquet
```

- Plant functional grouping only

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set plant_functional --data-dir ..\..\processed_parquet
```

- Legacy hybrid (v3)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set v3_hybrid --data-dir ..\..\processed_parquet
```

### Cluster counts and missing handling

- Try specific cluster counts

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set climate --clusters "3,4,5,6" --data-dir ..\..\processed_parquet
```

- Missing strategy and availability threshold

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set comprehensive --missing-strategy median --min-availability 0.6 --data-dir ..\..\processed_parquet
```

- Enforce cluster balance threshold

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set comprehensive --min-balance 0.2 --data-dir ..\..\processed_parquet
```

### Visualizations

- Generate comprehensive visualizations (2D/3D, silhouette, geographic, dashboard)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set comprehensive --visualize --data-dir ..\..\processed_parquet
```

- Quick visualization only (PCA)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set climate --quick-viz pca --data-dir ..\..\processed_parquet
```

- Skip 3D plots (faster)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set climate --visualize --no-3d --data-dir ..\..\processed_parquet
```

### Site split metadata (optional)

- Provide a train/test site split file for metadata (clustering still runs on all sites)

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set comprehensive --site-split-file path\to\site_split.json --data-dir ..\..\processed_parquet
```

### Quiet/verbose

- Quiet mode

```powershell
python ecosystem\clustering\FlexibleClusteringPipeline.py --feature-set climate --quiet --data-dir ..\..\processed_parquet
```

### Expected outputs

- Timestamped folder under `--output-dir`, e.g. `ecosystem/evaluation/clustering_results/climate_YYYYMMDD_HHMMSS/`, containing:
  - `flexible_site_clusters_{timestamp}.csv`
  - `flexible_clustering_strategy_{timestamp}.json`
  - `visualizations/` (plots, dashboard, summary HTML)
  - preprocessing artifacts (`label_encoders_{timestamp}.pkl`, `feature_scaler_{timestamp}.pkl`, `preprocessing_summary_{timestamp}.json`)

### Quick validation (PowerShell)

- Count result CSVs in latest run folder

```powershell
Get-ChildItem ecosystem\evaluation\clustering_results -Recurse -Filter flexible_site_clusters_*.csv | Measure-Object | % Count
```

- Preview cluster assignments

```powershell
Get-Content ecosystem\evaluation\clustering_results\climate_*\flexible_site_clusters_*.csv -TotalCount 5
```

- Confirm visualization summary exists

```powershell
Get-ChildItem ecosystem\evaluation\clustering_results\*\visualizations\clustering_report_*.html
```
