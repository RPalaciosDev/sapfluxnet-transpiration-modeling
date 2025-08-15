## Example Usage

Related documents: [Pipeline Overview](./PipelineOverview.md) · [Engineered Features Deep Dive](./EngineeredFeaturesDeepDive.md)

### List available feature sets

```powershell
python DataPipeline.py --list-feature-sets
```

### Quick analysis-only (no processing)

```powershell
python DataPipeline.py --analyze-only
```

- Outputs under `site_analysis_results/`: JSON, CSV summary, TXT report.

### Full processing (comprehensive features → CSV)

```powershell
python DataPipeline.py --feature-set comprehensive --export-format csv
```

- Outputs under `csv_comprehensive/` as `{SITE}.csv` (or `.csv.gz` if `--compress`).

### High-fidelity Parquet with compression

```powershell
python DataPipeline.py --feature-set comprehensive --export-format parquet --compress
```

- Outputs under `parquet_comprehensive/` as `{SITE}.parquet`.

### Temporal-only feature set (faster)

```powershell
python DataPipeline.py --feature-set temporal --export-format csv
```

### Include more sites (clean mode)

```powershell
python DataPipeline.py --clean-mode --feature-set comprehensive --export-format csv
```

- Excludes only extremely problematic sites; processes moderate/high with warnings.

### Force reprocess all valid sites

```powershell
python DataPipeline.py --force --feature-set comprehensive --export-format csv
```

### Disable quality flag filtering

```powershell
python DataPipeline.py --no-quality-flags --feature-set comprehensive
```

### Targeted overrides

- Custom lag horizon:

```powershell
python DataPipeline.py --feature-set comprehensive --max-lag-hours 24
```

- Custom rolling windows:

```powershell
python DataPipeline.py --feature-set comprehensive --rolling-windows "3,6,12,24,48,72,168,336,720"
```

- Tune chunking thresholds:

```powershell
python DataPipeline.py --chunk-size-override 1500 --memory-threshold 6 --file-size-threshold 100
```

### Streaming mode triggers (automatic)

- The orchestrator selects streaming when files are large or memory is limited. To encourage streaming in constrained environments:

```powershell
python DataPipeline.py --feature-set comprehensive --memory-threshold 6 --file-size-threshold 100
```

- In streaming, outputs append chunk-by-chunk; seasonality features are skipped by design.

### Expected outputs

- Processed data:
  - `csv_comprehensive/{SITE}.csv`
  - `parquet_comprehensive/{SITE}.parquet`
  - `csv_comprehensive/{SITE}.csv.gz` when `--compress`
  - `libsvm_comprehensive/{SITE}.svm` (+ `feature_mapping.json` once per folder)
- Site analysis artifacts:
  - `site_analysis_results/site_analysis_{timestamp}.json`
  - `site_analysis_results/site_analysis_summary_{timestamp}.csv`
  - `site_analysis_results/site_analysis_report_{timestamp}.txt`

### Quick validation checks (PowerShell)

- Count processed files:

```powershell
(Get-ChildItem csv_comprehensive -Filter *.csv*).Count
```

- Peek at columns:

```powershell
Get-Content csv_comprehensive\THA_KHU.csv -TotalCount 2
```

- Confirm libsvm feature map:

```powershell
Get-Content libsvm_comprehensive\feature_mapping.json -TotalCount 20
```

### Reproducibility tips

- Record the full command, `--feature-set`, overrides, and export format for each run.
- Keep `site_analysis_results/` alongside processed outputs.
- Prefer Parquet for long-term storage and stable schema.
