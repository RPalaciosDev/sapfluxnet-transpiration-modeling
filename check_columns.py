import pandas as pd
import glob

files = glob.glob('test_parquet_export/*_comprehensive.parquet')
print('Checking columns across files...')

all_columns = set()
for file in files[:5]:
    df = pd.read_parquet(file)
    all_columns.update(df.columns)
    print(f'{file.split("/")[-1]}: {len(df.columns)} columns')

print(f'Common columns: {len(all_columns)}')
print('Sample columns:', list(all_columns)[:10])

# Check for missing columns mentioned in error
missing_cols = ['solar_TIMESTAMP', 'solar_hour', 'solar_day_of_year', 'solar_hour_sin', 
                'solar_hour_cos', 'solar_day_sin', 'solar_day_cos', 'stand_height', 
                'plant_water_potential_time_code']

print('\nMissing columns:')
for col in missing_cols:
    if col not in all_columns:
        print(f'  - {col}') 