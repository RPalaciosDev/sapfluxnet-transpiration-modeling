import os
import glob
import argparse
from typing import List, Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


IDENTITY_COLS = {
    "site",
    "site_id",
    "siteid",
    "plant_id",
    "TIMESTAMP",
    "solar_TIMESTAMP",
    "Unnamed: 0",
}

EXCLUDE_SUFFIXES = ("_flags", "_md")


def find_parquet_files(processed_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.lower().endswith(".parquet"):
                paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {processed_dir}")
    paths.sort()
    return paths


def infer_feature_columns(parquet_paths: List[str], sample_limit: int = 50) -> pd.DataFrame:
    # Sample a subset of files to gather union of columns and dtypes
    sample_paths = parquet_paths[: sample_limit]
    columns_seen: Set[str] = set()
    dtype_map: dict = {}
    for p in sample_paths:
        try:
            df = pd.read_parquet(p, engine="auto")
        except Exception:
            df = pd.read_parquet(p)
        for c in df.columns:
            columns_seen.add(c)
            if c not in dtype_map:
                dtype_map[c] = str(df[c].dtype)
    cols = sorted(columns_seen)
    out = pd.DataFrame({"feature": cols, "dtype": [dtype_map[c] for c in cols]})
    return out


def apply_feature_exclusions(df_cols: pd.DataFrame) -> pd.DataFrame:
    def is_excluded(name: str) -> bool:
        lname = name.lower()
        if lname in {c.lower() for c in IDENTITY_COLS}:
            return True
        if lname in ("timestamp", "solar_timestamp"):
            return True
        for suf in EXCLUDE_SUFFIXES:
            if lname.endswith(suf):
                return True
        return False

    keep = [not is_excluded(f) for f in df_cols["feature"]]
    df = df_cols.loc[keep].copy()
    return df


def save_engineered_features_table(processed_dir: str, out_tables_dir: str) -> None:
    os.makedirs(out_tables_dir, exist_ok=True)
    parquet_paths = find_parquet_files(processed_dir)
    cols = infer_feature_columns(parquet_paths)
    features = apply_feature_exclusions(cols)
    # Simple heuristic for notes
    def note_for(name: str) -> str:
        lname = name.lower()
        if any(k in lname for k in ["lag", "rolling", "roll", "window"]):
            return "temporal lag/rolling"
        if any(k in lname for k in ["doy", "hour", "month", "season"]):
            return "temporal encoding"
        if any(k in lname for k in ["vpd", "rad", "sw", "temp", "ta", "rh", "precip"]):
            return "meteorological"
        return ""

    features["note"] = features["feature"].map(note_for)
    csv_path = os.path.join(out_tables_dir, "engineered_features.csv")
    features.to_csv(csv_path, index=False)

    # LaTeX table
    tex_path = os.path.join(out_tables_dir, "engineered_features.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{ll}\n")
        f.write("\\hline\\textbf{Feature} & \\textbf{Note} \\\\ \\hline\n")
        for _, r in features.sort_values("feature").iterrows():
            feat = str(r["feature"]).replace("_", "\\_")
            note = (r["note"] or "").replace("_", "\\_")
            f.write(f"{feat} & {note} \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n")


def summarize_clustering_schemas(clustering_dir: str, out_tables_dir: str) -> None:
    os.makedirs(out_tables_dir, exist_ok=True)
    json_paths = glob.glob(os.path.join(clustering_dir, "**", "*.json"), recursive=True)
    rows = []
    for p in sorted(json_paths):
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        # Handle dict or list-of-dicts
        items = cfg if isinstance(cfg, list) else [cfg]
        for item in items:
            if not isinstance(item, dict):
                continue
            schema_name = (
                item.get("name")
                or item.get("selected_strategy")
                or item.get("feature_set_name")
                or os.path.splitext(os.path.basename(p))[0]
            )
            k = item.get("n_clusters") or item.get("k") or item.get("num_clusters")
            feats = (
                item.get("features")
                or item.get("features_used")
                or item.get("preprocessing_summary", {}).get("processed_features")
                or []
            )
            features_used = ", ".join(feats[:8]) + ("..." if len(feats) > 8 else "")
            standardization = item.get("standardization") or item.get("scaler") or (
                "scaled" if item.get("preprocessing_summary", {}).get("scaler_fitted") else ""
            )
            rows.append({
                "schema": schema_name,
                "K": k,
                "features_used": features_used,
                "standardization": standardization,
                "source": p,
            })
    if not rows:
        print("No clustering schema JSONs found; skipping table.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_tables_dir, "clustering_schemas_summary.csv"), index=False)
    # LaTeX
    with open(os.path.join(out_tables_dir, "clustering_schemas_summary.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llll}\n\\hline Schema & K & Features (summary) & Standardization \\\\ \\hline\n")
        for _, r in df.iterrows():
            schema = str(r["schema"]).replace("_", "\\_")
            k = "" if pd.isna(r["K"]) else str(r["K"]).replace("_", "\\_")
            feats = (r["features_used"] or "").replace("_", "\\_")
            std = (r["standardization"] or "").replace("_", "\\_")
            f.write(f"{schema} & {k} & {feats} & {std} \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n")


def summarize_schema_performance(results_root: str, out_tables_dir: str, out_figures_dir: str) -> None:
    os.makedirs(out_tables_dir, exist_ok=True)
    os.makedirs(out_figures_dir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(d):
            continue
        fold_csvs = sorted(glob.glob(os.path.join(d, "parquet_spatial_fold_results_*.csv")))
        if not fold_csvs:
            continue
        df = pd.read_csv(fold_csvs[-1])
        # Normalize expected columns
        cols_lower = {c.lower(): c for c in df.columns}
        for need in ["test_r2", "test_rmse", "test_mae"]:
            if need not in cols_lower:
                raise ValueError(f"Missing {need} in {fold_csvs[-1]}")
        r2 = df[cols_lower["test_r2"]]
        rmse = df[cols_lower["test_rmse"]]
        mae = df[cols_lower["test_mae"]]
        rows.append({
            "schema": os.path.basename(d),
            "mean_test_r2": float(r2.mean()),
            "sd_test_r2": float(r2.std()),
            "mean_rmse": float(rmse.mean()),
            "sd_rmse": float(rmse.std()),
            "mean_mae": float(mae.mean()),
            "sd_mae": float(mae.std()),
            "folds": int(len(df)),
        })
    if not rows:
        print("No schema performance found under results_root; skipping.")
        return
    perf = pd.DataFrame(rows)
    # Round numeric metrics to 3 decimals for readability
    perf_rounded = perf.copy()
    for col in ["mean_test_r2", "sd_test_r2", "mean_rmse", "sd_rmse", "mean_mae", "sd_mae"]:
        if col in perf_rounded.columns:
            perf_rounded[col] = perf_rounded[col].astype(float).round(3)
    perf_rounded.to_csv(os.path.join(out_tables_dir, "schema_performance_summary.csv"), index=False)
    # LaTeX
    with open(os.path.join(out_tables_dir, "schema_performance_summary.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\hline Schema & mean R^2 & sd R^2 & mean RMSE & sd RMSE & mean MAE & sd MAE & folds \\\\ \\hline\n")
        for _, r in perf.iterrows():
            schema = str(r["schema"]).replace("_", "\\_")
            f.write(f"{schema} & {r['mean_test_r2']:.3f} & {r['sd_test_r2']:.3f} & {r['mean_rmse']:.3f} & {r['sd_rmse']:.3f} & {r['mean_mae']:.3f} & {r['sd_mae']:.3f} & {int(r['folds'])} \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n")

    # Plots
    plt.figure(figsize=(8, 4))
    order = list(perf.sort_values("mean_test_r2")["schema"])
    sns.barplot(data=perf, x="schema", y="mean_test_r2", order=order, color="#8dbce7")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean test R² (LOSO)")
    plt.title("Schema comparison: mean test R²")
    plt.tight_layout()
    plt.savefig(os.path.join(out_figures_dir, "schema_mean_test_r2.png"), dpi=200)
    plt.savefig(os.path.join(out_figures_dir, "schema_mean_test_r2.pdf"))
    plt.close()

    plt.figure(figsize=(8, 4))
    order = list(perf.sort_values("mean_rmse")["schema"])
    sns.barplot(data=perf, x="schema", y="mean_rmse", order=order, color="#fdc086")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean RMSE (LOSO)")
    plt.title("Schema comparison: mean RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_figures_dir, "schema_mean_rmse.png"), dpi=200)
    plt.savefig(os.path.join(out_figures_dir, "schema_mean_rmse.pdf"))
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generate engineered-features and schema summary tables, plus schema comparison plots")
    ap.add_argument("--processed-data", required=True, help="Directory containing processed parquet files")
    ap.add_argument("--clustering-dir", default="ecosystem/evaluation/clustering_results", help="Directory with clustering JSONs")
    ap.add_argument("--results-root", default="ecosystem/models/results/parquet_spatial_validation", help="Root with LOSO result CSVs per schema/run")
    ap.add_argument("--tables-dir", default="tables", help="Output directory for tables")
    ap.add_argument("--figures-dir", default="figures", help="Output directory for figures")
    args = ap.parse_args()

    save_engineered_features_table(args.processed_data, args.tables_dir)
    summarize_clustering_schemas(args.clustering_dir, args.tables_dir)
    summarize_schema_performance(args.results_root, args.tables_dir, args.figures_dir)
    print(f"Tables written to: {args.tables_dir}; figures to: {args.figures_dir}")


if __name__ == "__main__":
    main()


