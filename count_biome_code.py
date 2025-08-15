# save as: count_site_biome.py
import os
import glob
import pandas as pd
from collections import Counter

def normalize(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in {"", "NA", "NaN", "NULL", "None"}:
        return None
    return s

def main():
    sapwood_dir = "sapwood"
    patterns = [os.path.join(sapwood_dir, "*_site_md.csv"),
                os.path.join(sapwood_dir, "*_site_md.csv.gz")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        print("No site metadata files found (expected *_site_md.csv or *_site_md.csv.gz in 'sapwood/').")
        return

    counts = Counter()
    files_used = 0
    chunksize = 10000

    for f in files:
        try:
            for chunk in pd.read_csv(f, usecols=["si_biome"], chunksize=chunksize, on_bad_lines="skip"):
                if "si_biome" not in chunk.columns:
                    continue
                vals = chunk["si_biome"].map(normalize).dropna()
                if not vals.empty:
                    counts.update(vals)
                    files_used += 1
        except Exception as e:
            print(f"Warning: could not process {f}: {e}")

    if not counts:
        print("No valid 'si_biome' values found.")
        return

    print(f"Files contributing: {files_used}")
    print(f"Unique si_biome values: {len(counts)}\n")
    for biome, cnt in counts.most_common():
        print(f"{biome}: {cnt}")

if __name__ == "__main__":
    main()