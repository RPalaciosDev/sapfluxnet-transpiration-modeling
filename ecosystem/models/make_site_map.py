import os
import argparse
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_parquet_files(processed_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.lower().endswith(".parquet"):
                paths.append(os.path.join(root, f))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {processed_dir}")
    return paths


def try_read_columns(path: str, candidate_cols: List[str]) -> pd.DataFrame:
    # Try to read only candidate columns; fall back to full read if unsupported
    last_err: Optional[Exception] = None
    cols_to_try = list(dict.fromkeys(candidate_cols))
    try:
        return pd.read_parquet(path, columns=cols_to_try)
    except Exception as e:
        last_err = e
    try:
        df = pd.read_parquet(path)
        keep = [c for c in df.columns if c in cols_to_try or c.lower() in cols_to_try]
        if keep:
            return df[keep]
        return df
    except Exception as e2:
        raise RuntimeError(f"Failed to read {path}: {last_err} | {e2}")


def normalize_column(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    # Return the chosen column name present in df matching any alias
    for n in names:
        if n in df.columns:
            return n
        lower_matches = [c for c in df.columns if c.lower() == n.lower()]
        if lower_matches:
            return lower_matches[0]
    return None


def extract_site_lat_lon(
    path: str,
    cluster_map: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[str, float, float, Optional[str]]]:
    # Candidate column aliases
    site_aliases = ["site", "site_id", "siteid", "SITE", "SITE_ID"]
    lat_aliases = ["lat", "latitude", "LAT", "LATITUDE"]
    lon_aliases = ["lon", "longitude", "LON", "LONGITUDE"]

    # Read a small subset of columns if possible
    df = try_read_columns(path, site_aliases + lat_aliases + lon_aliases)

    site_col = normalize_column(df, site_aliases)
    lat_col = normalize_column(df, lat_aliases)
    lon_col = normalize_column(df, lon_aliases)

    # Derive site from filename if missing
    site_val: str
    if site_col and site_col in df.columns and df[site_col].notna().any():
        # Take the first non-null site id
        site_series = df[site_col].astype(str)
        site_val = site_series[site_series.notna()].iloc[0]
    else:
        site_val = os.path.splitext(os.path.basename(path))[0]

    # Compute lat/lon as median if present; else cannot use this file
    if lat_col and lon_col:
        lat = float(pd.to_numeric(df[lat_col], errors="coerce").dropna().median())
        lon = float(pd.to_numeric(df[lon_col], errors="coerce").dropna().median())
        if np.isnan(lat) or np.isnan(lon):
            return None
    else:
        return None

    # Optional cluster lookup
    cluster_val: Optional[str] = None
    if cluster_map is not None and not cluster_map.empty:
        # try exact match on original site string
        row = cluster_map[cluster_map["site"].astype(str) == str(site_val)]
        if row.empty:
            # try stem-to-stem match (ignoring extensions)
            stem = os.path.splitext(os.path.basename(path))[0]
            if "site_stem" in cluster_map.columns:
                row = cluster_map[cluster_map["site_stem"].astype(str) == stem]
            else:
                row = cluster_map[cluster_map["site"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0] == stem)]
        if not row.empty and "cluster" in row.columns:
            cluster_val = str(row.iloc[0]["cluster"])

    return site_val, lat, lon, cluster_val


def save_qgis_exports(df_sites: pd.DataFrame, out_qgis_dir: str) -> None:
    os.makedirs(out_qgis_dir, exist_ok=True)
    csv_path = os.path.join(out_qgis_dir, "site_points.csv")
    df_sites.to_csv(csv_path, index=False)
    # GeoJSON minimal
    try:
        import json
        features = []
        for _, r in df_sites.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                "properties": {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items() if k not in ("lat", "lon")},
            })
        geo = {"type": "FeatureCollection", "features": features}
        with open(os.path.join(out_qgis_dir, "site_points.geojson"), "w", encoding="utf-8") as f:
            json.dump(geo, f)
    except Exception as e:
        print(f"GeoJSON export failed: {e}")


def plot_site_scatter(df_sites: pd.DataFrame, out_fig_dir: str) -> None:
    os.makedirs(out_fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "cluster" in df_sites.columns and df_sites["cluster"].notna().any():
        sns.scatterplot(data=df_sites, x="lon", y="lat", hue="cluster", s=18, ax=ax, palette="tab10")
        ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        sns.scatterplot(data=df_sites, x="lon", y="lat", s=18, ax=ax, color="#1f77b4")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Site map (scatter; export GeoJSON for base map in QGIS)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, "site_map_by_cluster.png"), dpi=200)
    fig.savefig(os.path.join(out_fig_dir, "site_map_by_cluster.pdf"))
    plt.close(fig)


def load_cluster_map(cluster_assignments_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not cluster_assignments_csv:
        return None
    if not os.path.isfile(cluster_assignments_csv):
        return None
    df = pd.read_csv(cluster_assignments_csv)
    cols = {c.lower(): c for c in df.columns}
    # Normalize site/cluster columns and create a stem for robust matching
    site_col = cols.get("site") or cols.get("site_id") or list(df.columns)[0]
    cluster_col = cols.get("cluster") if "cluster" in cols else None
    df = df.rename(columns={site_col: "site", **({cluster_col: "cluster"} if cluster_col else {})})
    df["site_stem"] = df["site"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    return df


def main():
    ap = argparse.ArgumentParser(description="Make site map from processed parquet files; export QGIS-ready outputs")
    ap.add_argument("--processed-data", required=True, help="Directory containing processed parquet files")
    ap.add_argument("--cluster-assignments", default="", help="CSV with columns [site, cluster] (optional)")
    ap.add_argument("--out-fig", default="figures", help="Directory to save figures")
    ap.add_argument("--out-qgis", default="qgis", help="Directory to save QGIS exports (CSV/GeoJSON)")
    args = ap.parse_args()

    parquet_paths = find_parquet_files(args.processed_data)
    cluster_map = load_cluster_map(args.cluster_assignments)

    rows: List[Tuple[str, float, float, Optional[str]]] = []
    for p in parquet_paths:
        try:
            rec = extract_site_lat_lon(p, cluster_map)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        if rec is None:
            continue
        rows.append(rec)

    if not rows:
        raise RuntimeError("No site records with lat/lon found.")

    df_sites = pd.DataFrame(rows, columns=["site", "lat", "lon", "cluster"])
    # Deduplicate by site (keep first)
    df_sites = df_sites.drop_duplicates(subset=["site"])

    # Save QGIS exports
    save_qgis_exports(df_sites, args.out_qgis)

    # Plot simple scatter
    plot_site_scatter(df_sites, args.out_fig)

    print(f"Saved site scatter to: {os.path.join(args.out_fig, 'site_map_by_cluster.{png,pdf}')} ")
    print(f"Saved QGIS exports to: {args.out_qgis}")


if __name__ == "__main__":
    main()


