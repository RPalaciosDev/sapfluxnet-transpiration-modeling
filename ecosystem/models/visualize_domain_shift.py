import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def list_parquet_files(processed_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.lower().endswith('.parquet'):
                paths.append(os.path.join(root, f))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under: {processed_dir}")
    return sorted(paths)


def load_clusters(cluster_csv: str) -> pd.DataFrame:
    df = pd.read_csv(cluster_csv)
    cols = {c.lower(): c for c in df.columns}
    site_col = cols.get('site') or cols.get('site_id') or list(df.columns)[0]
    clus_col = cols.get('cluster')
    if not clus_col:
        raise ValueError("Cluster assignments CSV must include a 'cluster' column")
    df = df.rename(columns={site_col: 'site', clus_col: 'cluster'})
    # normalize to stem (filename without extension)
    df['site_stem'] = df['site'].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    return df[['site', 'site_stem', 'cluster']]


def load_cluster_features(strategy_path: str) -> List[str]:
    import json
    if not strategy_path or not os.path.isfile(strategy_path):
        return []
    try:
        with open(strategy_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    except Exception:
        return []
    items = obj if isinstance(obj, list) else [obj]
    feats: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if 'features_used' in it and isinstance(it['features_used'], list):
            feats.extend([str(x) for x in it['features_used']])
        elif 'preprocessing_summary' in it and isinstance(it['preprocessing_summary'], dict):
            feats.extend([str(x) for x in it['preprocessing_summary'].get('processed_features', [])])
        elif 'features' in it and isinstance(it['features'], list):
            feats.extend([str(x) for x in it['features']])
    # basic normalization
    feats = [c.strip() for c in feats if c and isinstance(c, str)]
    return sorted(list(dict.fromkeys(feats)))


def read_site_sample(parquet_path: str, feature_cols: List[str], sample_n: int = 5000) -> pd.DataFrame:
    # Load minimally, then downsample
    try:
        df = pd.read_parquet(parquet_path, columns=[c for c in feature_cols if c])
    except Exception:
        df = pd.read_parquet(parquet_path)
        df = df[[c for c in df.columns if c in feature_cols]]
    df = df.dropna(how='any')
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df.reset_index(drop=True)


def compute_site_feature_stats(site_to_df: Dict[str, pd.DataFrame], feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for site, df in site_to_df.items():
        if df.empty:
            continue
        means = df[feature_cols].mean()
        stds = df[feature_cols].std(ddof=0).replace(0, np.nan)
        rows.append({'site': site, **{f'mean__{c}': means[c] for c in feature_cols}, **{f'std__{c}': stds[c] for c in feature_cols}})
    return pd.DataFrame(rows)


def site_shift_score(site: str, stats_df: pd.DataFrame, feature_cols: List[str]) -> float:
    # Leave-one-out: compare site's mean to pooled mean/std of others (diagonal Mahalanobis)
    this = stats_df[stats_df['site'] == site]
    others = stats_df[stats_df['site'] != site]
    if this.empty or others.empty:
        return np.nan
    mu_site = np.array([float(this[f'mean__{c}']) for c in feature_cols])
    mu_pool = np.array([float(others[f'mean__{c}'].mean()) for c in feature_cols])
    std_pool = np.array([float(others[f'std__{c}'].mean()) for c in feature_cols])
    std_pool = np.where(std_pool == 0, np.nan, std_pool)
    z = (mu_site - mu_pool) / std_pool
    return float(np.nanmean(np.abs(z)))


def compute_cluster_shift(cluster_sites: List[str], stats_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    scores = []
    for s in cluster_sites:
        score = site_shift_score(s, stats_df, feature_cols)
        scores.append({'site': s, 'shift_score': score})
    return pd.DataFrame(scores).sort_values('shift_score', ascending=False)


def compute_cluster_local_shift_scores(site_to_cluster: Dict[str, str], stats_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Compute per-site shift relative to peers in the same cluster
    rows = []
    for cl in sorted(set(site_to_cluster.values())):
        sites = [s for s, c in site_to_cluster.items() if c == cl]
        sub = stats_df[stats_df['site'].isin(sites)].copy()
        if sub.empty or len(sub) < 2:
            continue
        for s in sites:
            this = sub[sub['site'] == s]
            others = sub[sub['site'] != s]
            if this.empty or others.empty:
                continue
            mu_site = np.array([float(this[f'mean__{c}']) for c in feature_cols])
            mu_pool = np.array([float(others[f'mean__{c}'].mean()) for c in feature_cols])
            std_pool = np.array([float(others[f'std__{c}'].mean()) for c in feature_cols])
            std_pool = np.where(std_pool == 0, np.nan, std_pool)
            z = (mu_site - mu_pool) / std_pool
            score = float(np.nanmean(np.abs(z)))
            rows.append({'site': s, 'cluster': cl, 'shift_score': score})
    return pd.DataFrame(rows)


def compute_feature_shift_by_cluster(site_to_cluster: Dict[str, str], stats_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Mean |z| per feature within each cluster
    rows = []
    for cl in sorted(set(site_to_cluster.values())):
        sites = [s for s, c in site_to_cluster.items() if c == cl]
        sub = stats_df[stats_df['site'].isin(sites)].copy()
        if sub.empty or len(sub) < 2:
            continue
        feat_vals = {}
        for f in feature_cols:
            mu_sites = np.array([float(sub.loc[sub['site'] == s, f'mean__{f}']) for s in sites]).ravel()
            mu_pool = float(sub[f'mean__{f}'].mean())
            std_pool = float(sub[f'std__{f}'].mean())
            z = np.abs((mu_sites - mu_pool) / (std_pool if std_pool != 0 else np.nan))
            feat_vals[f] = np.nanmean(z)
        rows.append({'cluster': cl, **feat_vals})
    return pd.DataFrame(rows)


def plot_shift_by_cluster_violin(scores_df: pd.DataFrame, outpath: str, clip_high: float = 0.99):
    clusters = sorted(scores_df['cluster'].astype(str).unique())
    fig_w = max(8, 1.2 * len(clusters))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    sns.violinplot(data=scores_df, x='cluster', y='shift_score', inner='box', cut=0, scale='width', ax=ax, color='#c7d7f0')
    sns.stripplot(data=scores_df, x='cluster', y='shift_score', ax=ax, color='#2c3e50', size=2, alpha=0.5)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Domain shift score (|z| mean)')
    ax.set_title('Per-site domain shift by cluster')
    # Scale y-axis to a high quantile to reduce outlier dominance
    try:
        ymax = float(scores_df['shift_score'].quantile(clip_high))
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.05)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(outpath + '.png', dpi=200)
    fig.savefig(outpath + '.pdf')
    plt.close(fig)


def plot_feature_shift_heatmap(feat_shift_df: pd.DataFrame, feature_cols: List[str], outpath: str):
    # Pretty names for common features
    pretty_map = {
        'vpd': 'Vapor pressure deficit',
        'sw_in': 'Shortwave radiation',
        'ta': 'Air temperature',
        'rh': 'Relative humidity',
        'precip': 'Precipitation',
        'ppfd_in': 'Photosynthetic photon flux density',
        'swc_shallow': 'Shallow soil water content',
        'mean_annual_temp': 'Mean annual temperature',
        'mean_annual_precip': 'Mean annual precipitation',
        'seasonal_temp_range': 'Seasonal temperature range',
        'seasonal_precip_range': 'Seasonal precipitation range',
        'koppen_geiger_code_encoded': 'Köppen–Geiger (encoded)',
        'biome_code_encoded': 'Biome (encoded)',
        'igbp_class_code_encoded': 'IGBP class (encoded)',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'ext_rad_fao56': 'Extraterrestrial radiation (FAO‑56)',
        'net_radiation': 'Net radiation',
        'pet_oudin_mm_day': 'PET (Oudin, mm/day)'
    }

    def pretty_name(c: str) -> str:
        if c in pretty_map:
            return pretty_map[c]
        # Fallback: underscores to spaces, title case with common unit hints untouched
        name = c.replace('_', ' ').strip()
        # Keep lowercase for common units/tokens
        return name.capitalize() if len(name) < 30 else name

    pretty_cols = [pretty_name(c) for c in feature_cols]
    fig_w = max(7, 0.7 * len(pretty_cols))
    fig_h = max(3.5, 0.35 * len(feat_shift_df))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    df_plot = feat_shift_df.set_index('cluster')[feature_cols]
    # Reassign columns to pretty names for display only
    df_plot.columns = pretty_cols
    sns.heatmap(df_plot, cmap='YlOrRd', annot=False, ax=ax)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Cluster')
    ax.set_title('Mean feature shift (|z|) within clusters')
    # Rotate x labels for readability
    labelsize = 9 if len(pretty_cols) <= 16 else 8
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=labelsize)
    fig.tight_layout()
    fig.savefig(outpath + '.png', dpi=200)
    fig.savefig(outpath + '.pdf')
    plt.close(fig)


def plot_shift_bar(scores_df: pd.DataFrame, outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(scores_df))))
    sns.barplot(data=scores_df, x='shift_score', y='site', ax=ax, color='#fdae6b')
    ax.set_xlabel('Domain shift score (|z| mean across features)')
    ax.set_ylabel('Site')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath + '.png', dpi=200)
    fig.savefig(outpath + '.pdf')
    plt.close(fig)


def plot_pairwise_heatmap(stats_df: pd.DataFrame, sites: List[str], feature_cols: List[str], outpath: str, title: str):
    # Diagonal std Mahalanobis distance between site means using pooled std across all sites
    pooled_std = np.array([float(stats_df[f'std__{c}'].mean()) for c in feature_cols])
    pooled_std = np.where(pooled_std == 0, 1.0, pooled_std)
    mu = {s: np.array([float(stats_df.loc[stats_df['site'] == s, f'mean__{c}']) for c in feature_cols]).ravel() for s in sites}
    n = len(sites)
    D = np.zeros((n, n))
    for i, si in enumerate(sites):
        for j, sj in enumerate(sites):
            diff = (mu[si] - mu[sj]) / pooled_std
            D[i, j] = np.nanmean(np.abs(diff))
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * n), max(5, 0.35 * n)))
    sns.heatmap(D, xticklabels=sites, yticklabels=sites, cmap='mako', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath + '.png', dpi=200)
    fig.savefig(outpath + '.pdf')
    plt.close(fig)


def plot_feature_overlays(site_df: pd.DataFrame, rest_df: pd.DataFrame, feature_cols: List[str], outdir: str, site_name: str, cluster_name: str):
    os.makedirs(outdir, exist_ok=True)
    for c in feature_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        try:
            sns.kdeplot(site_df[c].dropna(), label=f'{site_name}', ax=ax, color='#e34a33')
            sns.kdeplot(rest_df[c].dropna(), label='rest of cluster', ax=ax, color='#2b8cbe')
        except Exception:
            sns.histplot(site_df[c].dropna(), stat='density', element='step', label=f'{site_name}', ax=ax, color='#e34a33', bins=40, alpha=0.4)
            sns.histplot(rest_df[c].dropna(), stat='density', element='step', label='rest of cluster', ax=ax, color='#2b8cbe', bins=40, alpha=0.4)
        ax.set_xlabel(c)
        ax.set_ylabel('Density')
        ax.set_title(f'{c}: {site_name} vs cluster {cluster_name}')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'overlay_{c}.png'), dpi=200)
        fig.savefig(os.path.join(outdir, f'overlay_{c}.pdf'))
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Visualize domain shift across sites within clusters')
    ap.add_argument('--processed-data', required=True, help='Directory with processed parquet files')
    ap.add_argument('--cluster-assignments', required=True, help='CSV with columns [site, cluster] (site names may include extensions)')
    ap.add_argument('--features', default='vpd,sw_in,ta,rh,precip,ppfd_in,swc_shallow', help='Comma-separated feature list to compare')
    ap.add_argument('--outdir', default='figures/domain_shift', help='Output directory')
    ap.add_argument('--sample-n', type=int, default=5000, help='Rows to sample per site')
    ap.add_argument('--dashboard', action='store_true', help='Produce a single comprehensive dashboard across all sites')
    ap.add_argument('--cluster-strategy', default='', help='Optional clustering strategy JSON to include cluster features in heatmap')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = list_parquet_files(args.processed_data)
    clusters_df = load_clusters(args.cluster_assignments)
    # Map file -> site_stem
    file_to_stem = {p: os.path.splitext(os.path.basename(p))[0] for p in files}
    stem_to_file = {v: k for k, v in file_to_stem.items()}

    feature_cols = [c.strip() for c in args.features.split(',') if c.strip()]
    cluster_feature_candidates = load_cluster_features(args.cluster_strategy)

    # Build per-site samples for sites present in both data and cluster map
    site_to_df: Dict[str, pd.DataFrame] = {}
    site_to_cluster: Dict[str, str] = {}
    for _, row in clusters_df.iterrows():
        stem = row['site_stem']
        if stem not in stem_to_file:
            continue
        path = stem_to_file[stem]
        try:
            df = read_site_sample(path, feature_cols, sample_n=args.sample_n)
        except Exception:
            continue
        if df.empty:
            continue
        # keep only requested features actually present
        present = [c for c in feature_cols if c in df.columns]
        if not present:
            continue
        site_to_df[stem] = df[present].copy()
        site_to_cluster[stem] = str(row['cluster'])

    # Compute stats across all sites
    stats_df = compute_site_feature_stats(site_to_df, feature_cols)
    if stats_df.empty:
        raise RuntimeError('No site stats computed. Check feature names and processed data.')

    # Optional single-dashboard figure across all clusters
    if args.dashboard:
        all_sites = list(stats_df['site'])
        # Global pooled std and means
        pooled_std = np.array([float(stats_df[f'std__{c}'].mean()) for c in feature_cols])
        pooled_std = np.where(pooled_std == 0, 1.0, pooled_std)
        X = np.vstack([np.array([float(stats_df.loc[stats_df['site'] == s, f'mean__{c}']) for c in feature_cols]).ravel() for s in all_sites])
        # PCA embedding
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Z = pca.fit_transform((X - np.nanmean(X, axis=0)) / pooled_std)
        except Exception:
            # Fallback: use first two features if PCA unavailable
            Z = X[:, :2]
        # Compute global shift scores (vs pooled across all sites)
        mu_pool = np.nanmean(X, axis=0)
        z = np.abs((X - mu_pool) / pooled_std)
        shift_scores = np.nanmean(z, axis=1)
        # Build distance matrix
        n = len(all_sites)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.nanmean(np.abs((X[i] - X[j]) / pooled_std))
        # Ordering: by cluster, then shift score desc
        site_clusters = [site_to_cluster.get(s, 'NA') for s in all_sites]
        order = sorted(range(n), key=lambda i: (str(site_clusters[i]), -shift_scores[i]))
        all_sites_ord = [all_sites[i] for i in order]
        D_ord = D[np.ix_(order, order)]
        Z_ord = Z[order]
        clusters_ord = [site_clusters[i] for i in order]
        scores_ord = shift_scores[order]
        # Plot dashboard
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
        # Left: PCA scatter colored by cluster, size by shift score
        ax0 = fig.add_subplot(gs[0, 0])
        uniq = sorted(set(clusters_ord))
        palette = sns.color_palette('Set3', n_colors=max(3, len(uniq)))
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}
        sizes = 20 + 60 * (scores_ord - np.nanmin(scores_ord)) / (np.nanmax(scores_ord) - np.nanmin(scores_ord) + 1e-9)
        for c in uniq:
            mask = np.array([cc == c for cc in clusters_ord])
            ax0.scatter(Z_ord[mask, 0], Z_ord[mask, 1], s=sizes[mask], c=[color_map[c]], alpha=0.8, label=str(c), edgecolors='k', linewidths=0.2)
        ax0.set_xlabel('PC1')
        ax0.set_ylabel('PC2')
        ax0.set_title('Site feature means (PCA), colored by cluster; size = shift score')
        ax0.legend(title='Cluster', frameon=False, markerscale=1.2, bbox_to_anchor=(1.02, 1), loc='upper left')
        # Right: Pairwise distance heatmap ordered by cluster and shift score
        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(D_ord, xticklabels=all_sites_ord, yticklabels=all_sites_ord, cmap='mako', ax=ax1)
        ax1.set_title('Pairwise site mean-difference (std-normalized), ordered by cluster/shift')
        ax1.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'domain_shift_dashboard.png'), dpi=200)
        fig.savefig(os.path.join(args.outdir, 'domain_shift_dashboard.pdf'))

    # Iterate clusters and create visuals
    clusters = sorted(set(site_to_cluster.values()))
    for cl in clusters:
        sites = [s for s, c in site_to_cluster.items() if c == cl and s in set(stats_df['site'])]
        if len(sites) < 2:
            continue
        cl_out = os.path.join(args.outdir, f'cluster_{cl}')
        os.makedirs(cl_out, exist_ok=True)
        # Shift scores
        scores_df = compute_cluster_shift(sites, stats_df, feature_cols)
        plot_shift_bar(scores_df, os.path.join(cl_out, 'site_shift_scores'), f'Domain shift scores (cluster {cl})')
        # Pairwise heatmap
        plot_pairwise_heatmap(stats_df, sites, feature_cols, os.path.join(cl_out, 'pairwise_shift_heatmap'), f'Pairwise mean-difference (cluster {cl})')
        # Overlay for the top-shifted site
        top_site = scores_df.iloc[0]['site']
        rest_sites = [s for s in sites if s != top_site]
        site_df = site_to_df[top_site]
        rest_df = pd.concat([site_to_df[s] for s in rest_sites], ignore_index=True)
        present = [c for c in feature_cols if c in site_df.columns and c in rest_df.columns]
        plot_feature_overlays(site_df, rest_df, present, os.path.join(cl_out, f'overlay_{top_site}'), top_site, str(cl))

    # Cluster-level overview plots (one per repo)
    local_scores = compute_cluster_local_shift_scores(site_to_cluster, stats_df, feature_cols)
    if not local_scores.empty:
        plot_shift_by_cluster_violin(local_scores, os.path.join(args.outdir, 'shift_by_cluster'))
        feat_shift = compute_feature_shift_by_cluster(site_to_cluster, stats_df, feature_cols)
        if not feat_shift.empty:
            plot_feature_shift_heatmap(feat_shift, feature_cols, os.path.join(args.outdir, 'feature_shift_by_cluster'))

        # Combined heatmap including clustering features (if available in data)
        if cluster_feature_candidates:
            present_features = sorted({c for df in site_to_df.values() for c in cluster_feature_candidates + feature_cols if c in df.columns})
            if present_features:
                # Recompute stats with expanded feature set
                stats_full = compute_site_feature_stats(site_to_df, present_features)
                if not stats_full.empty:
                    # Recompute feature shift per cluster with combined features
                    feat_shift_full = compute_feature_shift_by_cluster(site_to_cluster, stats_full, present_features)
                    # Order columns: clustering features first (those actually present), then the rest
                    cl_feats_present = [c for c in cluster_feature_candidates if c in present_features]
                    others = [c for c in present_features if c not in cl_feats_present]
                    ordered = cl_feats_present + others
                    plot_feature_shift_heatmap(feat_shift_full, ordered, os.path.join(args.outdir, 'feature_shift_by_cluster_combined'))

    print(f"Saved domain shift visuals to: {args.outdir}")


if __name__ == '__main__':
    main()


