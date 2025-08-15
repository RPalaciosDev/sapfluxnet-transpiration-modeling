import os, glob, argparse
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def find_run_dir(results_root: str) -> str:
    """Resolve a run directory that contains fold results.

    Supports both legacy names (parquet_spatial_*) and new names (spatial_*).
    """
    patterns = [
        "spatial_fold_results_*.csv",
        "parquet_spatial_fold_results_*.csv",
    ]
    if os.path.isfile(results_root):
        return os.path.dirname(results_root)
    if os.path.isdir(results_root):
        # If it already contains a fold file, return it
        for pat in patterns:
            if glob.glob(os.path.join(results_root, pat)):
                return results_root
        # Else search immediate children
        candidates = []
        for d in glob.glob(os.path.join(results_root, "*")):
            if not os.path.isdir(d):
                continue
            for pat in patterns:
                if glob.glob(os.path.join(d, pat)):
                    candidates.append(d)
                    break
        if not candidates:
            raise FileNotFoundError("No run directories with spatial fold results found")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    raise FileNotFoundError(f"Path not found: {results_root}")

def load_fold_results(run_dir: str) -> pd.DataFrame:
    # Prefer new naming, fallback to legacy
    paths = sorted(glob.glob(os.path.join(run_dir, "spatial_fold_results_*.csv")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(run_dir, "parquet_spatial_fold_results_*.csv")))
    if not paths:
        raise FileNotFoundError("No spatial fold results CSV in run_dir")
    df = pd.read_csv(paths[-1])
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Expected: cluster, test_r2, test_rmse, test_mae (case-insensitive)
    for needed in ["cluster", "test_r2", "test_rmse", "test_mae"]:
        if needed not in [c.lower() for c in df.columns]:
            raise ValueError(f"Missing column in fold results: {needed}")
    # Ensure types
    df["cluster"] = df[[c for c in df.columns if c.lower()=="cluster"][0]]
    df["test_r2"] = df[[c for c in df.columns if c.lower()=="test_r2"][0]]
    df["test_rmse"] = df[[c for c in df.columns if c.lower()=="test_rmse"][0]]
    df["test_mae"] = df[[c for c in df.columns if c.lower()=="test_mae"][0]]
    return df

def ensure_outdir(run_dir: str) -> str:
    out = os.path.join(run_dir, "visualizations")
    os.makedirs(out, exist_ok=True)
    return out

def save_fig(fig, outpath_base: str):
    fig.tight_layout()
    fig.savefig(outpath_base + ".png", dpi=200)
    fig.savefig(outpath_base + ".pdf")
    plt.close(fig)

def plot_boxplot_r2(
    df: pd.DataFrame,
    outdir: str,
    r2_lim: Tuple[float, float],
    scale_r2: str = "linear",
    r2_linthresh: float = 0.1,
):
    order = sorted(df["cluster"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    ycol = "test_r2" if scale_r2 == "symlog" else "test_r2_plot"
    sns.boxplot(data=df, x="cluster", y=ycol, order=order, ax=ax, color="#8dbce7", fliersize=0)
    sns.stripplot(data=df, x="cluster", y=ycol, order=order, ax=ax, color="#1f4e79", size=2, alpha=0.5)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Test R² (LOSO)")
    ax.set_title("LOSO test R² by cluster" + (" (symlog)" if scale_r2 == "symlog" else " (clipped)"))
    if scale_r2 == "symlog":
        ax.set_yscale("symlog", linthresh=r2_linthresh)
    else:
        ax.set_ylim(r2_lim)
    save_fig(fig, os.path.join(outdir, "boxplot_test_r2_by_cluster"))

def plot_fraction_positive_r2(df: pd.DataFrame, outdir: str):
    grp = df.groupby("cluster")["test_r2"].apply(lambda s: np.mean(s > 0)).reset_index(name="frac_pos")
    order = sorted(grp["cluster"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=grp, x="cluster", y="frac_pos", order=order, ax=ax, color="#7fc97f")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction of folds with R² > 0")
    ax.set_title("Transferability share per cluster")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    save_fig(fig, os.path.join(outdir, "fraction_positive_r2_by_cluster"))

def plot_hist_r2(
    df: pd.DataFrame,
    outdir: str,
    r2_lim: Tuple[float, float],
    scale_r2: str = "linear",
    r2_linthresh: float = 0.1,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    series = df["test_r2"] if scale_r2 == "symlog" else df["test_r2"].clip(*r2_lim)
    sns.histplot(series, bins=40, ax=ax, color="#fdc086")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Test R² (LOSO)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of LOSO test R²" + (" (symlog)" if scale_r2 == "symlog" else " (clipped)"))
    if scale_r2 == "symlog":
        ax.set_xscale("symlog", linthresh=r2_linthresh)
    else:
        ax.set_xlim(r2_lim)
    save_fig(fig, os.path.join(outdir, "hist_test_r2_overall"))

def plot_scatter_rmse_r2(
    df: pd.DataFrame,
    outdir: str,
    r2_lim: Tuple[float, float],
    rmse_lim: Tuple[float, float],
    scale_r2: str = "linear",
    scale_rmse: str = "linear",
    r2_linthresh: float = 0.1,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    xcol = "test_r2" if scale_r2 == "symlog" else "test_r2_plot"
    ycol = "test_rmse" if scale_rmse == "log" else "test_rmse_plot"
    sns.scatterplot(data=df, x=xcol, y=ycol, hue="cluster", palette="tab10", s=12, ax=ax, alpha=0.7)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Test R² (LOSO)")
    ax.set_ylabel("Test RMSE")
    ax.set_title("LOSO: RMSE vs R²" + (" (symlog/log)" if (scale_r2=="symlog" or scale_rmse=="log") else " (clipped)"))
    if scale_r2 == "symlog":
        ax.set_xscale("symlog", linthresh=r2_linthresh)
    else:
        ax.set_xlim(r2_lim)
    if scale_rmse == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(rmse_lim)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig(fig, os.path.join(outdir, "scatter_rmse_vs_r2"))

def export_cluster_summary_for_latex(df: pd.DataFrame, outdir: str):
    summary = df.groupby("cluster").agg(
        mean_test_r2=("test_r2", "mean"),
        sd_test_r2=("test_r2", "std"),
        mean_rmse=("test_rmse", "mean"),
        sd_rmse=("test_rmse", "std"),
        mean_mae=("test_mae", "mean"),
        sd_mae=("test_mae", "std"),
        frac_r2_pos=("test_r2", lambda s: float(np.mean(s > 0))),
        folds=("test_r2", "size"),
    ).reset_index()
    # overall row
    overall = pd.DataFrame([{
        "cluster": "OVERALL",
        "mean_test_r2": float(df["test_r2"].mean()),
        "sd_test_r2": float(df["test_r2"].std()),
        "mean_rmse": float(df["test_rmse"].mean()),
        "sd_rmse": float(df["test_rmse"].std()),
        "mean_mae": float(df["test_mae"].mean()),
        "sd_mae": float(df["test_mae"].std()),
        "frac_r2_pos": float(np.mean(df["test_r2"] > 0)),
        "folds": int(len(df))
    }])
    out = pd.concat([summary, overall], ignore_index=True)
    out_csv = os.path.join(outdir, "cluster_validation_summary_for_latex.csv")
    out.to_csv(out_csv, index=False)

def plot_violin_r2(
    df: pd.DataFrame,
    outdir: str,
    r2_lim: Tuple[float, float],
    scale_r2: str = "linear",
    r2_linthresh: float = 0.1,
):
    order = sorted(df["cluster"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    ycol = "test_r2" if scale_r2 == "symlog" else "test_r2_plot"
    sns.violinplot(
        data=df, x="cluster", y=ycol,
        order=order, inner="box", cut=0, scale="width",
        ax=ax, color="#c7d7f0"
    )
    sns.stripplot(
        data=df, x="cluster", y=ycol,
        order=order, ax=ax, color="#2c3e50", size=2, alpha=0.5
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Test R² (LOSO)")
    ax.set_title("LOSO test R² by cluster (violin)" + (" (symlog)" if scale_r2 == "symlog" else " (clipped)"))
    if scale_r2 == "symlog":
        ax.set_yscale("symlog", linthresh=r2_linthresh)
    else:
        ax.set_ylim(r2_lim)
    save_fig(fig, os.path.join(outdir, "violin_test_r2_by_cluster"))


def bounds(series: pd.Series, lo: float, hi: float) -> Tuple[float, float]:
    ql, qh = series.quantile([lo, hi])
    return float(ql), float(qh)


def add_clipped_columns(df: pd.DataFrame, lo: float, hi: float) -> Tuple[pd.DataFrame, Tuple[float, float], Tuple[float, float]]:
    r2_lo, r2_hi = bounds(df["test_r2"], lo, hi)
    rmse_lo, rmse_hi = bounds(df["test_rmse"], lo, hi)
    df = df.copy()
    df["test_r2_plot"] = df["test_r2"].clip(r2_lo, r2_hi)
    df["test_rmse_plot"] = df["test_rmse"].clip(rmse_lo, rmse_hi)
    return df, (r2_lo, r2_hi), (rmse_lo, rmse_hi)

def main():
    ap = argparse.ArgumentParser(description="Visualize LOSO spatial validation results (clipping and axis transforms)")
    ap.add_argument("--results-root", default="ecosystem/models/results/parquet_spatial_validation",
                    help="Run dir or parent directory containing run subdirs")
    ap.add_argument("--clip-low", type=float, default=0.01, help="Lower quantile for clipping (plotting only)")
    ap.add_argument("--clip-high", type=float, default=0.99, help="Upper quantile for clipping (plotting only)")
    ap.add_argument("--scale-r2", choices=["linear", "symlog"], default="linear", help="Axis scale for R² plots")
    ap.add_argument("--r2-linthresh", type=float, default=0.1, help="Symlog linear threshold for R²")
    ap.add_argument("--scale-rmse", choices=["linear", "log"], default="linear", help="Axis scale for RMSE in scatter plot")
    args = ap.parse_args()

    run_dir = find_run_dir(args.results_root)
    outdir = ensure_outdir(run_dir)
    print(f"Using run_dir: {run_dir}")
    print(f"Saving figures to: {outdir}")

    df = load_fold_results(run_dir)
    df = df.dropna(subset=["test_r2", "test_rmse", "test_mae"]).copy()

    # Add plotting-only clipped columns and limits
    df, r2_lim, rmse_lim = add_clipped_columns(df, args.clip_low, args.clip_high)

    # plots (use transforms if requested; otherwise clipped linear)
    plot_boxplot_r2(df, outdir, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    plot_fraction_positive_r2(df, outdir)
    plot_hist_r2(df, outdir, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    plot_scatter_rmse_r2(
        df, outdir, r2_lim, rmse_lim,
        scale_r2=args.scale_r2, scale_rmse=args.scale_rmse, r2_linthresh=args.r2_linthresh
    )
    export_cluster_summary_for_latex(df, outdir)
    plot_violin_r2(df, outdir, r2_lim, scale_r2=args.scale_r2, r2_linthresh=args.r2_linthresh)
    print("Done.")

if __name__ == "__main__":
    main()