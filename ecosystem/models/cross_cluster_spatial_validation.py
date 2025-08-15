import os
import json
import argparse
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

EXCLUDE_COLS_BASE = [
    "TIMESTAMP", "solar_TIMESTAMP", "site", "plant_id", "Unnamed: 0", "ecosystem_cluster"
]
TARGET_COL = "sap_flow"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_cluster_assignments(cluster_file: str):
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")
    df = pd.read_csv(cluster_file)
    if "site" not in df.columns or "cluster" not in df.columns:
        raise ValueError("Cluster file must have columns: site, cluster")
    assignments = dict(zip(df["site"], df["cluster"]))
    sites_by_cluster = defaultdict(list)
    for site_id, cl in assignments.items():
        sites_by_cluster[int(cl)].append(site_id)
    clusters = sorted(sites_by_cluster.keys())
    return assignments, sites_by_cluster, clusters


def infer_feature_cols(sample_parquet_file: str):
    df = pd.read_parquet(sample_parquet_file, engine="auto")
    cols = list(df.columns)
    feature_cols = [
        c for c in cols
        if c not in EXCLUDE_COLS_BASE + [TARGET_COL]
        and not c.endswith("_flags")
        and not c.endswith("_md")
    ]
    return feature_cols


def prepare_features(df: pd.DataFrame, feature_cols):
    # Coerce non-numeric and fill as per training path
    X_df = df[feature_cols].copy()
    for c in X_df.columns:
        if X_df[c].dtype == bool:
            X_df[c] = X_df[c].astype(int)
        elif X_df[c].dtype == "object":
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0)
    X = X_df.fillna(0).values
    y = df[TARGET_COL].values
    return X, y


def load_sites_parquet(sites, parquet_dir, feature_cols, cap_rows=None):
    frames = []
    for site in sites:
        path = os.path.join(parquet_dir, site)
        if not os.path.exists(path):
            log(f"  ⚠️ Missing parquet for site: {site}")
            continue
        try:
            df = pd.read_parquet(path, engine="auto")
            if TARGET_COL not in df.columns:
                continue
            df = df.dropna(subset=[TARGET_COL])
            if df.empty:
                continue
            df["site"] = site
            if cap_rows is not None and len(df) > cap_rows:
                df = df.sample(n=cap_rows, random_state=42)
            frames.append(df)
        except Exception as e:
            log(f"  ❌ Error reading {site}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_xgb(use_gpu: bool):
    import xgboost as xgb
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 10 if use_gpu else 8,
        "learning_rate": 0.15 if use_gpu else 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "gpu_hist" if use_gpu else "hist",
        **({"gpu_id": 0} if use_gpu else {"n_jobs": -1}),
        "random_state": 42,
        "verbosity": 0,
    }
    n_estimators = 200 if use_gpu else 100
    model = xgb.XGBRegressor(**params, n_estimators=n_estimators)
    return model


def detect_gpu(force_gpu: bool) -> bool:
    if force_gpu:
        return True
    try:
        import xgboost as xgb  # noqa: F401
        _ = xgb.XGBRegressor(tree_method="gpu_hist", gpu_id=0, n_estimators=1)
        return True
    except Exception:
        return False


def metrics_dict(y_true, y_pred, scope_note):
    return {
        "scope": scope_note,
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def leave_one_cluster_out(parquet_dir, cluster_file, results_dir, force_gpu=False, cap_rows=None):
    os.makedirs(results_dir, exist_ok=True)
    assignments, sites_by_cluster, clusters = load_cluster_assignments(cluster_file)
    # Sample one parquet to infer features
    some_site = list(assignments.keys())[0]
    some_file = os.path.join(parquet_dir, some_site)
    feature_cols = infer_feature_cols(some_file)

    use_gpu = detect_gpu(force_gpu)
    log(f"GPU enabled: {use_gpu}")

    per_cluster_results = []

    for held in clusters:
        log(f"\n=== LOCO: hold-out cluster {held} ===")
        test_sites = sites_by_cluster[held]
        train_sites = [s for c in clusters if c != held for s in sites_by_cluster[c]]

        test_df = load_sites_parquet(test_sites, parquet_dir, feature_cols, cap_rows)
        train_df = load_sites_parquet(train_sites, parquet_dir, feature_cols, cap_rows)

        if test_df is None or train_df is None:
            log("  ⚠️ Skipping (insufficient data)")
            continue

        X_train, y_train = prepare_features(train_df, feature_cols)
        X_test, y_test = prepare_features(test_df, feature_cols)

        model = get_xgb(use_gpu)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        res = metrics_dict(y_test, y_pred, scope_note=f"LOCO_cluster_{held}")
        res["held_cluster"] = held
        per_cluster_results.append(res)
        log(f"  Test R2={res['r2']:.4f}, RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, n={res['n']}")

    if per_cluster_results:
        df = pd.DataFrame(per_cluster_results)
        out_csv = os.path.join(results_dir, "loco_cluster_results.csv")
        df.to_csv(out_csv, index=False)
        log(f"\nSaved LOCO results: {out_csv}")
        overall = {
            "mean_r2": float(np.nanmean(df["r2"])),
            "mean_rmse": float(np.nanmean(df["rmse"])),
            "mean_mae": float(np.nanmean(df["mae"])),
            "clusters": len(df),
        }
        with open(os.path.join(results_dir, "loco_summary.json"), "w") as f:
            json.dump(overall, f, indent=2)
        log(f"Summary: {overall}")
    else:
        log("No LOCO results produced.")


def transfer_matrix(parquet_dir, cluster_file, results_dir, force_gpu=False, cap_rows=None):
    os.makedirs(results_dir, exist_ok=True)
    assignments, sites_by_cluster, clusters = load_cluster_assignments(cluster_file)
    some_site = list(assignments.keys())[0]
    some_file = os.path.join(parquet_dir, some_site)
    feature_cols = infer_feature_cols(some_file)

    use_gpu = detect_gpu(force_gpu)
    log(f"GPU enabled: {use_gpu}")

    records = []
    # train on source cluster, test on target cluster (source != target)
    for src in clusters:
        train_sites = sites_by_cluster[src]
        train_df = load_sites_parquet(train_sites, parquet_dir, feature_cols, cap_rows)
        if train_df is None:
            continue
        X_train, y_train = prepare_features(train_df, feature_cols)
        model = get_xgb(use_gpu)
        model.fit(X_train, y_train)

        for tgt in clusters:
            if tgt == src:
                continue
            test_sites = sites_by_cluster[tgt]
            test_df = load_sites_parquet(test_sites, parquet_dir, feature_cols, cap_rows)
            if test_df is None:
                continue
            X_test, y_test = prepare_features(test_df, feature_cols)
            y_pred = model.predict(X_test)
            res = metrics_dict(y_test, y_pred, scope_note="transfer")
            res["source_cluster"] = src
            res["target_cluster"] = tgt
            records.append(res)
            log(f"src {src} -> tgt {tgt}: R2={res['r2']:.4f}, RMSE={res['rmse']:.4f}, n={res['n']}")

    if records:
        df = pd.DataFrame(records)
        out_csv = os.path.join(results_dir, "transfer_matrix.csv")
        df.to_csv(out_csv, index=False)
        log(f"\nSaved transfer matrix: {out_csv}")
        try:
            r2_mat = df.pivot(index="source_cluster", columns="target_cluster", values="r2")
            r2_mat.to_csv(os.path.join(results_dir, "transfer_matrix_r2.csv"))
        except Exception:
            pass
    else:
        log("No transfer results produced.")


def main():
    ap = argparse.ArgumentParser(description="Cross-cluster spatial validation")
    ap.add_argument("--parquet-dir", default="../../processed_parquet", help="Directory of site Parquet files")
    ap.add_argument("--cluster-file", required=True, help="Path to flexible_site_clusters_*.csv")
    ap.add_argument("--results-dir", default="./results/cross_cluster_validation", help="Output directory")
    ap.add_argument("--force-gpu", action="store_true", help="Force GPU mode")
    ap.add_argument("--cap-rows", type=int, default=None, help="Optional row cap per site to limit memory")
    ap.add_argument("--mode", choices=["loco", "transfer-matrix"], default="loco", help="Validation mode")
    args = ap.parse_args()

    if args.mode == "loco":
        leave_one_cluster_out(args.parquet_dir, args.cluster_file, args.results_dir, args.force_gpu, args.cap_rows)
    else:
        transfer_matrix(args.parquet_dir, args.cluster_file, args.results_dir, args.force_gpu, args.cap_rows)


if __name__ == "__main__":
    main()


