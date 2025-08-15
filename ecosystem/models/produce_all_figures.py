import os
import glob
import argparse
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse loaders/plots from the existing visualization module where possible
from .visualize_spatial_results import (
    find_run_dir,
    load_fold_results,
    ensure_outdir,
    add_clipped_columns,
    plot_boxplot_r2,
    plot_fraction_positive_r2,
    plot_hist_r2,
    plot_scatter_rmse_r2,
    plot_violin_r2,
)


def _save(fig, path_no_ext: str) -> None:
    fig.tight_layout()
    fig.savefig(path_no_ext + ".png", dpi=200)
    fig.savefig(path_no_ext + ".pdf")
    plt.close(fig)


def plot_cluster_sizes(cluster_assignments_csv: str, outdir: str) -> None:
    df = pd.read_csv(cluster_assignments_csv)
    # Expect columns: site, cluster (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    if "cluster" not in cols_lower:
        raise ValueError("cluster_assignments must contain a 'cluster' column")
    cluster_col = cols_lower["cluster"]
    sizes = df.groupby(cluster_col).size().reset_index(name="count")
    order = list(sizes.sort_values("count", ascending=False)[cluster_col])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sizes, x=cluster_col, y="count", order=order, ax=ax, color="#a6cee3")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of sites")
    ax.set_title("Cluster sizes (sites per cluster)")
    _save(fig, os.path.join(outdir, "cluster_sizes_sites"))


def plot_feature_importance_panels(models_root: str, outdir: str, top_n: int = 10) -> None:
    # Looks for CSVs like feature_importance_cluster_*.csv under models_root
    paths = sorted(glob.glob(os.path.join(models_root, "**", "feature_importance_cluster_*_*.csv"), recursive=True))
    if not paths:
        # fallback to simpler glob
        paths = sorted(glob.glob(os.path.join(models_root, "feature_importance_cluster_*.csv")))
    if not paths:
        print("No feature importance CSVs found; skipping feature importance panels.")
        return

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        # Expect columns: feature, importance (gain/weight/shap), or similar
        cols_lower = {c.lower(): c for c in df.columns}
        # Try common names
        feature_col = cols_lower.get("feature") or cols_lower.get("features") or list(df.columns)[0]
        # Take the first numeric column as importance if not explicit
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if "importance" in cols_lower:
            importance_col = cols_lower["importance"]
        elif numeric_cols:
            importance_col = numeric_cols[0]
        else:
            print(f"Could not identify importance column in {p}; skipping.")
            continue

        tmp = df[[feature_col, importance_col]].dropna()
        tmp = tmp.sort_values(importance_col, ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=tmp, y=feature_col, x=importance_col, ax=ax, color="#b2df8a")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        title_name = os.path.splitext(os.path.basename(p))[0]
        ax.set_title(f"Top {top_n} features: {title_name}")
        outbase = os.path.join(outdir, f"feature_importance_{title_name}")
        _save(fig, outbase)


def plot_r2_by_site(df_folds: pd.DataFrame, outdir: str, cluster_filter: Optional[str] = None) -> None:
    # If site column exists, visualize heterogeneity within a cluster
    site_cols = [c for c in df_folds.columns if c.lower() in ("site", "site_id", "siteid")]
    if not site_cols:
        print("No site identifier column in fold results; skipping per-site R² plot.")
        return
    site_col = site_cols[0]
    df = df_folds.copy()
    if cluster_filter is not None:
        df = df[df["cluster"] == cluster_filter]
    agg = df.groupby(site_col)["test_r2"].mean().reset_index()
    agg = agg.sort_values("test_r2", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hlines(y=np.arange(len(agg)), xmin=0, xmax=0, color="lightgray")
    sns.scatterplot(data=agg, x="test_r2", y=np.arange(len(agg)), ax=ax, s=20, color="#1f78b4")
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels(agg[site_col])
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean test R² (LOSO)")
    ax.set_ylabel("Site")
    ax.set_title("Per-site mean test R²" + (f" (cluster {cluster_filter})" if cluster_filter else ""))
    _save(fig, os.path.join(outdir, "per_site_mean_test_r2" + (f"_cluster_{cluster_filter}" if cluster_filter else "")))


def main():
    ap = argparse.ArgumentParser(description="Produce a suite of figures for the LOSO spatial validation study")
    ap.add_argument("--results-root", default="ecosystem/models/results/parquet_spatial_validation",
                    help="Run dir or parent directory containing parquet_spatial_fold_results_*.csv")
    ap.add_argument("--models-root", default="ecosystem/models/results/cluster_models",
                    help="Root directory containing feature_importance_cluster_* CSVs")
    ap.add_argument("--cluster-assignments", default="",
                    help="CSV with site-to-cluster assignments (for cluster size plot)")
    ap.add_argument("--site-metadata", default="",
                    help="CSV with columns [site, lat, lon, cluster] for site map (optional)")
    ap.add_argument("--clip-low", type=float, default=0.01, help="Lower quantile for clipping (plotting only)")
    ap.add_argument("--clip-high", type=float, default=0.99, help="Upper quantile for clipping (plotting only)")
    ap.add_argument("--scale-r2", choices=["linear", "symlog"], default="linear", help="Axis scale for R² plots")
    ap.add_argument("--r2-linthresh", type=float, default=0.1, help="Symlog linear threshold for R²")
    ap.add_argument("--scale-rmse", choices=["linear", "log"], default="linear", help="Axis scale for RMSE in scatter plot")
    ap.add_argument("--outdir", default="", help="Override output directory (default: run_dir/visualizations_all)")
    args = ap.parse_args()

    # Resolve run dir
    run_dir = find_run_dir(args.results_root)
    outdir_root = os.path.join(run_dir, "visualizations_all") if not args.outdir else args.outdir
    os.makedirs(outdir_root, exist_ok=True)
    print(f"Using run_dir: {run_dir}")
    print(f"Saving figures to: {outdir_root}")

    # Load fold results and prepare clipped columns
    df = load_fold_results(run_dir)
    df = df.dropna(subset=["test_r2", "test_rmse", "test_mae"]).copy()
    df, r2_lim, rmse_lim = add_clipped_columns(df, args.clip_low, args.clip_high)

    # 9/10/11/13 core LOSO figures using existing plotting utilities
    plot_boxplot_r2(df, outdir_root, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    plot_violin_r2(df, outdir_root, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    plot_fraction_positive_r2(df, outdir_root)
    plot_hist_r2(df, outdir_root, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    plot_scatter_rmse_r2(
        df, outdir_root, r2_lim, rmse_lim,
        scale_r2=args.scale_r2, scale_rmse=args.scale_rmse, r2_linthresh=args.r2_linthresh,
    )

    # 11: Per-site heterogeneity (if site column is present)
    plot_r2_by_site(df, outdir_root)

    # 6: Cluster sizes (if assignments provided)
    if args.cluster_assignments and os.path.isfile(args.cluster_assignments):
        plot_cluster_sizes(args.cluster_assignments, outdir_root)
    else:
        print("No cluster assignments CSV provided; skipping cluster sizes plot.")

    # 18: Feature importance panels (if available)
    if args.models_root and os.path.isdir(args.models_root):
        plot_feature_importance_panels(args.models_root, outdir_root, top_n=10)
    else:
        print("Models root not found or empty; skipping feature importance panels.")

    # Placeholders (not generated without additional inputs):
    print("Note: Site map, covariate distributions/embeddings, domain-shift metrics, residual analyses, and time-series plots \n"
          "require additional metadata (lat/lon) and/or per-fold predictions with covariates. Provide inputs and we can extend.")

    print("Done.")


if __name__ == "__main__":
    main()


