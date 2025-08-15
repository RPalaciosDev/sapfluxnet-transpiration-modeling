#!/usr/bin/env python
import os
import argparse
from typing import Set
import pyarrow.parquet as pq

def gather_sites_from_data(parquet_path: str) -> Set[str]:
    sites = set()
    try:
        pf = pq.ParquetFile(parquet_path)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=['site'])
            # Convert to Python list; filter None/NaN
            for v in tbl['site'].to_pylist():
                if v is None:
                    continue
                # Filter NaN if it sneaks in as float
                if isinstance(v, float) and v != v:
                    continue
                sites.add(str(v))
    except Exception as e:
        # Fail soft: return empty so caller can fall back to filename
        pass
    return sites

def main():
    parser = argparse.ArgumentParser(description="Count unique SAPFLUXNET sites from parquet files")
    parser.add_argument('--data-dir', default='./parquet_ecological', help="Directory containing parquet files")
    parser.add_argument('--list', action='store_true', help="Print the site names")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Directory not found: {args.data_dir}")
        return 1

    parquet_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.parquet')])
    if not parquet_files:
        print("No parquet files found.")
        return 0

    # 1) Filename-derived sites (fast path)
    sites_by_filename = {os.path.splitext(f)[0] for f in parquet_files}

    # 2) Data-derived sites (robust path, streaming per row group)
    sites_by_data = set()
    for i, fname in enumerate(parquet_files, 1):
        path = os.path.join(args.data_dir, fname)
        sites_from_file = gather_sites_from_data(path)
        if sites_from_file:
            sites_by_data |= sites_from_file
        # Minimal progress
        if i % 50 == 0:
            print(f"Scanned {i} files...")

    # Choose the most reliable result
    final_sites = sites_by_data if sites_by_data else sites_by_filename

    print(f"Total parquet files: {len(parquet_files)}")
    print(f"Unique sites (by filename): {sites_by_filename and len(sites_by_filename)}")
    print(f"Unique sites (by data): {sites_by_data and len(sites_by_data)}")
    print(f"Final unique site count: {len(final_sites)}")

    if args.list:
        # Print sorted site names (first 100 to keep output manageable)
        for s in sorted(list(final_sites))[:100]:
            print(s)
        if len(final_sites) > 100:
            print(f"... and {len(final_sites) - 100} more")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())