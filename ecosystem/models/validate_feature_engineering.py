import os
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce') if series is not None else pd.Series(dtype='float64')


def derive_temporal_columns(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (year, day_of_year, hour). Prefer existing columns; else derive from TIMESTAMP/solar_TIMESTAMP."""
    year = df['year'] if 'year' in df.columns else None
    doy = df['day_of_year'] if 'day_of_year' in df.columns else None
    hour = df['hour'] if 'hour' in df.columns else None

    if (year is None or doy is None or hour is None):
        ts_col = None
        if 'solar_TIMESTAMP' in df.columns:
            ts_col = pd.to_datetime(df['solar_TIMESTAMP'], errors='coerce')
        elif 'TIMESTAMP' in df.columns:
            ts_col = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        if ts_col is not None:
            if year is None:
                year = ts_col.dt.year
            if doy is None:
                doy = ts_col.dt.dayofyear
            if hour is None:
                hour = ts_col.dt.hour

    # Fallbacks if still None
    if year is None:
        year = pd.Series(np.nan, index=df.index)
    if doy is None:
        doy = pd.Series(np.nan, index=df.index)
    if hour is None:
        hour = pd.Series(np.nan, index=df.index)

    return year, doy, hour


def recompute_fao56_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute FAO-56 related support terms and radiation budget per the pipeline implementation.

    Produces columns:
      - ext_rad_fao56
      - daylight_hours
      - pet_oudin_mm_day
      - net_shortwave_radiation
      - net_longwave_radiation
      - net_radiation
    Columns are float and may contain NaN when inputs are insufficient.
    """
    out = pd.DataFrame(index=df.index)

    # Required inputs
    lat = safe_numeric(df.get('latitude', pd.Series(np.nan, index=df.index)))
    elev = safe_numeric(df.get('elevation', pd.Series(np.nan, index=df.index))).fillna(0.0)
    ta = safe_numeric(df.get('ta', pd.Series(np.nan, index=df.index)))
    rh = safe_numeric(df.get('rh', pd.Series(np.nan, index=df.index)))
    sw_in = safe_numeric(df.get('sw_in', pd.Series(np.nan, index=df.index)))
    meas_ts_min = safe_numeric(df.get('measurement_timestep', pd.Series(np.nan, index=df.index)))

    # Temporal
    year, doy, hour = derive_temporal_columns(df)

    # Daylight indicator if present; else approximate (6-18)
    if 'is_daylight' in df.columns:
        is_daylight = df['is_daylight']
        try:
            is_daylight = is_daylight.astype(float)
        except Exception:
            is_daylight = pd.to_numeric(is_daylight, errors='coerce')
    else:
        is_daylight = ((hour >= 6) & (hour <= 18)).astype(float)

    # Compute extraterrestrial radiation Ra (MJ m^-2 day^-1) and daylight hours N
    with np.errstate(invalid='ignore'):
        lat_rad = np.deg2rad(lat.astype(float))
        J = doy
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * J / 365.0)
        delta = 0.409 * np.sin(2.0 * np.pi * J / 365.0 - 1.39)
        # clip domain for arccos
        cos_omega_arg = -np.tan(lat_rad) * np.tan(delta)
        cos_omega_arg = np.clip(cos_omega_arg, -1.0, 1.0)
        omega_s = np.arccos(cos_omega_arg)
        G_sc = 0.0820  # MJ m^-2 min^-1
        Ra = (24.0 * 60.0 / np.pi) * G_sc * dr * (
            omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
        )
        N = (24.0 / np.pi) * omega_s

    out['ext_rad_fao56'] = Ra
    out['daylight_hours'] = N

    # PET Oudin (mm/day)
    try:
        lambda_mj = 2.45
        pet = (Ra / lambda_mj) * ((ta + 5.0) / 100.0)
        pet = pet.where(ta > -5.0, 0.0)
        out['pet_oudin_mm_day'] = pet
    except Exception:
        out['pet_oudin_mm_day'] = np.nan

    # Per-step net radiation (MJ m^-2 per time step)
    # Determine time step hours
    dt_hours = (meas_ts_min / 60.0).fillna(1.0)

    # Shortwave incoming per step (if measured)
    Rs_step = sw_in * (dt_hours * 3600.0) / 1e6

    # Clear-sky shortwave per step using Ra and elevation
    with np.errstate(invalid='ignore', divide='ignore'):
        Ra_day = Ra
        N_nonzero = N.replace(0, np.nan)
        Ra_per_hour = Ra_day / N_nonzero
        Ra_step = Ra_per_hour * dt_hours
        z = elev.fillna(0.0)
        Rso_step = (0.75 + 2e-5 * z) * Ra_step

        # Net shortwave
        albedo = 0.23
        Rns_step = (1.0 - albedo) * Rs_step

        # Net longwave
        T_k = ta + 273.16
        es = 0.6108 * np.exp((17.27 * ta) / (ta + 237.3))
        ea = (rh / 100.0) * es
        sigma_daily = 4.903e-9
        sigma_scaled = sigma_daily * (dt_hours / 24.0)
        cloud_term = (1.35 * (Rs_step / Rso_step.replace(0, np.nan)) - 0.35).clip(lower=0.0)
        emissivity_term = (0.34 - 0.14 * np.sqrt(ea.clip(lower=0.0)))
        Rnl_step = sigma_scaled * (T_k ** 4) * emissivity_term * cloud_term

    out['net_shortwave_radiation'] = Rns_step
    out['net_longwave_radiation'] = Rnl_step
    out['net_radiation'] = Rns_step - Rnl_step

    return out


def compare_columns(truth: pd.Series, pred: pd.Series, tol: float) -> Dict[str, float]:
    mask = truth.notna() & pred.notna()
    if mask.sum() == 0:
        return {
            'present': float(not truth.isna().all()),
            'n_compared': 0.0,
            'frac_within_tol': 0.0,
            'max_abs_err': float('nan'),
            'mean_abs_err': float('nan')
        }
    abs_err = (truth[mask] - pred[mask]).abs()
    within = (abs_err <= tol).mean()
    return {
        'present': 1.0,
        'n_compared': float(mask.sum()),
        'frac_within_tol': float(within),
        'max_abs_err': float(abs_err.max()),
        'mean_abs_err': float(abs_err.mean()),
    }


def validate_file(path: Path, sample_n: int, tol: float) -> Tuple[Dict[str, any], pd.DataFrame]:
    """Validate one parquet file; return summary dict and small detailed frame."""
    site = path.stem
    # Columns needed from file
    needed_inputs = [
        'latitude', 'elevation', 'ta', 'rh', 'sw_in', 'measurement_timestep',
        'year', 'day_of_year', 'hour', 'is_daylight', 'solar_TIMESTAMP', 'TIMESTAMP'
    ]
    compare_cols = [
        'ext_rad_fao56', 'daylight_hours', 'pet_oudin_mm_day',
        'net_shortwave_radiation', 'net_longwave_radiation', 'net_radiation'
    ]
    use_cols = list(dict.fromkeys(needed_inputs + compare_cols))  # dedupe

    try:
        df = pd.read_parquet(path, columns=[c for c in use_cols if c in pd.read_parquet(path, columns=None).columns])
    except Exception:
        # Fallback: read fully then select
        df = pd.read_parquet(path)
        df = df[[c for c in use_cols if c in df.columns]]

    if len(df) == 0:
        return ({'file': str(path), 'site': site, 'rows': 0, 'status': 'empty'}, pd.DataFrame())

    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    recomputed = recompute_fao56_terms(df)

    summary = {'file': str(path), 'site': site, 'rows': int(len(df)), 'status': 'ok'}

    # Compare available columns
    details_rows = []
    for col in compare_cols:
        if col in df.columns:
            stats = compare_columns(df[col], recomputed[col], tol)
        else:
            stats = {'present': 0.0, 'n_compared': 0.0, 'frac_within_tol': 0.0, 'max_abs_err': float('nan'), 'mean_abs_err': float('nan')}
        row = {'metric': col}
        row.update(stats)
        details_rows.append(row)
        # Add to summary with compact keys
        summary[f'{col}_present'] = bool(stats['present'])
        summary[f'{col}_within'] = float(stats['frac_within_tol']) if not math.isnan(stats['frac_within_tol']) else 0.0
        summary[f'{col}_maxerr'] = stats['max_abs_err']

    details = pd.DataFrame(details_rows)
    return summary, details


def main():
    ap = argparse.ArgumentParser(description='Validate physics-based feature engineering against parquet outputs.')
    ap.add_argument('--data-dir', default='parquet_ecological', help='Directory with processed parquet files')
    ap.add_argument('--out', default='ecosystem/evaluation/feature_validation', help='Directory to save validation outputs')
    ap.add_argument('--sample-n', type=int, default=100000, help='Rows to sample per file (0 = all)')
    ap.add_argument('--tolerance', type=float, default=1e-3, help='Absolute tolerance for numeric comparisons')
    ap.add_argument('--max-files', type=int, default=0, help='Validate at most N files (0 = all)')
    args = ap.parse_args()

    ensure_dir(args.out)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    parquet_files = sorted([p for p in data_dir.iterdir() if p.suffix == '.parquet'])
    if args.max_files and args.max_files > 0:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        print(f"ERROR: No parquet files in {data_dir}")
        return 1

    all_summaries: List[Dict[str, any]] = []
    all_details: List[pd.DataFrame] = []

    print(f"Validating {len(parquet_files)} files from {data_dir} (sample_n={args.sample_n}, tol={args.tolerance})")

    for i, fpath in enumerate(parquet_files, start=1):
        try:
            summary, details = validate_file(fpath, args.sample_n, args.tolerance)
            all_summaries.append(summary)
            if not details.empty:
                details.insert(0, 'file', str(fpath))
                all_details.append(details)
            ok_frac = np.mean([
                summary.get('ext_rad_fao56_within', 0.0),
                summary.get('daylight_hours_within', 0.0),
                summary.get('pet_oudin_mm_day_within', 0.0),
                summary.get('net_radiation_within', 0.0),
            ])
            print(f"  [{i}/{len(parquet_files)}] {fpath.name}: rows={summary['rows']} match≈{ok_frac:.2f}")
        except Exception as e:
            print(f"  [{i}/{len(parquet_files)}] {fpath.name}: ERROR {e}")

    # Save outputs
    sum_df = pd.DataFrame(all_summaries)
    det_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    sum_path = Path(args.out) / 'feature_validation_summary.csv'
    det_path = Path(args.out) / 'feature_validation_details.csv'
    sum_df.to_csv(sum_path, index=False)
    if not det_df.empty:
        det_df.to_csv(det_path, index=False)

    # Console recap
    present_cols = [c for c in ['ext_rad_fao56_present', 'daylight_hours_present', 'pet_oudin_mm_day_present', 'net_radiation_present'] if c in sum_df.columns]
    if present_cols:
        present_rates = sum_df[present_cols].mean().to_dict()
        print("\nPresence rates across files:")
        for k, v in present_rates.items():
            print(f"  {k.replace('_present','')}: {v:.2f}")

    within_cols = [c for c in ['ext_rad_fao56_within', 'daylight_hours_within', 'pet_oudin_mm_day_within', 'net_radiation_within'] if c in sum_df.columns]
    if within_cols:
        within_rates = sum_df[within_cols].replace([np.inf, -np.inf], np.nan).fillna(0).mean().to_dict()
        print("\nMean fraction within tolerance across files:")
        for k, v in within_rates.items():
            print(f"  {k.replace('_within','')}: {v:.2f}")

    print(f"\nSaved: {sum_path}")
    if not det_df.empty:
        print(f"Saved: {det_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


