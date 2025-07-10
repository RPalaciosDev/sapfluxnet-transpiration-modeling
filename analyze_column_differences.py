import pandas as pd
import glob
from collections import defaultdict

# Get all parquet files
files = glob.glob('test_parquet_export/*_comprehensive.parquet')
print(f'Analyzing {len(files)} parquet files...')

# Track column presence across files
column_presence = defaultdict(list)
all_columns = set()

# Check each file
for file in files:
    site_name = file.split('/')[-1].replace('_comprehensive.parquet', '')
    try:
        df = pd.read_parquet(file)
        columns = set(df.columns)
        all_columns.update(columns)
        
        # Track which columns are in this file
        for col in all_columns:
            if col in columns:
                column_presence[col].append(site_name)
            else:
                column_presence[col].append(None)
                
        print(f'{site_name}: {len(columns)} columns')
        
    except Exception as e:
        print(f'Error reading {site_name}: {e}')

print(f'\nTotal unique columns across all files: {len(all_columns)}')

# Analyze column consistency
print('\n=== COLUMN ANALYSIS ===')
print('Columns present in ALL files:')
all_present = []
for col in all_columns:
    if len(column_presence[col]) == len(files) and all(column_presence[col]):
        all_present.append(col)
        print(f'  ✓ {col}')

print(f'\nColumns present in SOME files ({len(all_columns) - len(all_present)} columns):')
some_present = []
for col in all_columns:
    if col not in all_present:
        present_count = sum(1 for x in column_presence[col] if x is not None)
        missing_count = len(files) - present_count
        some_present.append((col, present_count, missing_count))
        print(f'  ⚠ {col}: present in {present_count}/{len(files)} files ({missing_count} missing)')

# Show details for columns missing in many files
print('\n=== DETAILED ANALYSIS ===')
print('Columns missing in >50% of files:')
for col, present_count, missing_count in some_present:
    if missing_count > len(files) / 2:
        missing_sites = [site for site, has_col in zip([f.split('/')[-1].replace('_comprehensive.parquet', '') for f in files], column_presence[col]) if not has_col]
        print(f'\n{col}:')
        print(f'  Present in: {present_count} files')
        print(f'  Missing in: {missing_count} files')
        print(f'  Missing sites: {missing_sites[:10]}{"..." if len(missing_sites) > 10 else ""}')

# Show columns missing in just a few files
print('\nColumns missing in <10% of files:')
for col, present_count, missing_count in some_present:
    if missing_count <= len(files) * 0.1 and missing_count > 0:
        missing_sites = [site for site, has_col in zip([f.split('/')[-1].replace('_comprehensive.parquet', '') for f in files], column_presence[col]) if not has_col]
        print(f'\n{col}:')
        print(f'  Present in: {present_count} files')
        print(f'  Missing in: {missing_count} files')
        print(f'  Missing sites: {missing_sites}') 