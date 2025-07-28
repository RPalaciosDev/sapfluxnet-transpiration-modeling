#!/usr/bin/env python3
"""
Extract feature descriptions from v2 pipeline code
This script parses the v2 pipeline to get accurate feature descriptions
"""

import re
import pandas as pd

def extract_feature_descriptions_from_code():
    """Extract feature descriptions from the v2 pipeline code"""
    
    # Read the v2 pipeline code - updated path to work from feature_importance directory
    with open('../data_pipeline_v2.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Dictionary to store feature descriptions
    feature_descriptions = {}
    
    # Extract environmental variables (basic features)
    env_features = {
        'ta': 'Air temperature',
        'rh': 'Relative humidity',
        'vpd': 'Vapor pressure deficit', 
        'sw_in': 'Shortwave incoming radiation',
        'ws': 'Wind speed',
        'precip': 'Precipitation',
        'ppfd_in': 'Photosynthetic photon flux density',
        'ext_rad': 'Extraterrestrial radiation',
        'swc_shallow': 'Soil water content (shallow)'
    }
    feature_descriptions.update(env_features)
    
    # Extract temporal features from create_advanced_temporal_features
    temporal_section = re.search(r'def create_advanced_temporal_features.*?return features', code, re.DOTALL)
    if temporal_section:
        temporal_code = temporal_section.group(0)
        
        # Extract temporal features with descriptions
        temporal_features = {
            'hour': 'Hour of day',
            'day_of_year': 'Day of year', 
            'month': 'Month',
            'year': 'Year',
            'day_of_week': 'Day of week',
            'hour_sin': 'Sine of hour (cyclical)',
            'hour_cos': 'Cosine of hour (cyclical)',
            'day_sin': 'Sine of day of year (cyclical)',
            'day_cos': 'Cosine of day of year (cyclical)',
            'month_sin': 'Sine of month (cyclical)',
            'month_cos': 'Cosine of month (cyclical)',
            'solar_hour': 'Solar hour',
            'solar_day_of_year': 'Solar day of year',
            'solar_hour_sin': 'Sine of solar hour',
            'solar_hour_cos': 'Cosine of solar hour',
            'solar_day_sin': 'Sine of solar day',
            'solar_day_cos': 'Cosine of solar day',
            'is_daylight': 'Daylight indicator (6-18h)',
            'is_peak_sunlight': 'Peak sunlight indicator (10-16h)',
            'is_weekend': 'Weekend indicator',
            'is_morning': 'Morning indicator (6-12h)',
            'is_afternoon': 'Afternoon indicator (12-18h)',
            'is_night': 'Night indicator',
            'is_spring': 'Spring season indicator',
            'is_summer': 'Summer season indicator',
            'is_autumn': 'Autumn season indicator',
            'is_winter': 'Winter season indicator',
            'hours_since_sunrise': 'Hours since sunrise',
            'hours_since_sunset': 'Hours since sunset'
        }
        feature_descriptions.update(temporal_features)
    
    # Extract interaction features from create_interaction_features
    interaction_section = re.search(r'def create_interaction_features.*?return features', code, re.DOTALL)
    if interaction_section:
        interaction_code = interaction_section.group(0)
        
        # Extract interaction features with descriptions
        interaction_features = {
            'vpd_ppfd_interaction': 'VPD × PPFD interaction',
            'vpd_ta_interaction': 'VPD × Temperature interaction',
            'temp_humidity_ratio': 'Temperature/humidity ratio',
            'water_stress_index': 'Water stress index',
            'ppfd_efficiency': 'PPFD efficiency',
            'stomatal_conductance_proxy': 'Stomatal conductance proxy',
            'wind_stress': 'Wind stress',
            'wind_vpd_interaction': 'Wind × VPD interaction',
            'temp_soil_interaction': 'Temperature × soil interaction',
            'humidity_soil_interaction': 'Humidity × soil interaction',
            'radiation_temp_interaction': 'Radiation × temperature interaction',
            'stomatal_control_index': 'Stomatal control index',
            'light_efficiency': 'Light efficiency'
        }
        feature_descriptions.update(interaction_features)
    
    # Extract rate of change features
    rate_section = re.search(r'def create_rate_of_change_features.*?return features', code, re.DOTALL)
    if rate_section:
        rate_code = rate_section.group(0)
        
        # Extract rate of change features
        rate_features = {
            'ta_rate_1h': 'Temperature rate of change 1h',
            'ta_rate_6h': 'Temperature rate of change 6h',
            'ta_rate_24h': 'Temperature rate of change 24h',
            'vpd_rate_1h': 'VPD rate of change 1h',
            'vpd_rate_6h': 'VPD rate of change 6h',
            'vpd_rate_24h': 'VPD rate of change 24h',
            'sw_in_rate_1h': 'Shortwave radiation rate of change 1h',
            'sw_in_rate_6h': 'Shortwave radiation rate of change 6h',
            'sw_in_rate_24h': 'Shortwave radiation rate of change 24h',
            'rh_rate_1h': 'Relative humidity rate of change 1h',
            'rh_rate_6h': 'Relative humidity rate of change 6h',
            'rh_rate_24h': 'Relative humidity rate of change 24h'
        }
        feature_descriptions.update(rate_features)
    
    # Extract cumulative features
    cumulative_section = re.search(r'def create_cumulative_features.*?return features', code, re.DOTALL)
    if cumulative_section:
        cumulative_code = cumulative_section.group(0)
        
        # Extract cumulative features
        cumulative_features = {
            'precip_cum_24h': 'Cumulative precipitation 24h',
            'precip_cum_72h': 'Cumulative precipitation 72h',
            'precip_cum_168h': 'Cumulative precipitation 168h',
            'sw_in_cum_24h': 'Cumulative shortwave radiation 24h',
            'sw_in_cum_72h': 'Cumulative shortwave radiation 72h'
        }
        feature_descriptions.update(cumulative_features)
    
    # Extract seasonality features
    seasonality_section = re.search(r'def create_seasonality_features.*?return features', code, re.DOTALL)
    if seasonality_section:
        seasonality_code = seasonality_section.group(0)
        
        # Extract seasonality features
        seasonality_features = {
            'seasonal_temp_range': 'Seasonal temperature range',
            'seasonal_precip_range': 'Seasonal precipitation range'
        }
        feature_descriptions.update(seasonality_features)
    
    # Extract domain-specific features
    domain_section = re.search(r'def create_domain_specific_features.*?return features', code, re.DOTALL)
    if domain_section:
        domain_code = domain_section.group(0)
        
        # Extract domain-specific features
        domain_features = {
            'temp_deviation': 'Temperature deviation from optimal (25°C)',
            'tree_size_factor': 'Tree size factor (log of DBH)',
            'sapwood_leaf_ratio': 'Sapwood to leaf area ratio',
            'transpiration_capacity': 'Transpiration capacity'
        }
        feature_descriptions.update(domain_features)
    
    return feature_descriptions

def get_lagged_feature_descriptions():
    """Generate descriptions for lagged features"""
    lagged_features = {}
    
    # Variables that get lagged
    variables = ['ta', 'rh', 'vpd', 'sw_in', 'ws', 'precip', 'swc_shallow', 'ppfd_in']
    lags = ['1h', '2h', '3h', '6h', '12h', '24h']
    
    for var in variables:
        for lag in lags:
            feature_name = f'{var}_lag_{lag}'
            var_desc = {
                'ta': 'Temperature',
                'rh': 'Relative humidity',
                'vpd': 'VPD',
                'sw_in': 'Shortwave radiation',
                'ws': 'Wind speed',
                'precip': 'Precipitation',
                'swc_shallow': 'Soil water content',
                'ppfd_in': 'PPFD'
            }.get(var, var)
            lagged_features[feature_name] = f'{var_desc} lag {lag}'
    
    return lagged_features

def get_rolling_feature_descriptions():
    """Generate descriptions for rolling window features"""
    rolling_features = {}
    
    # Variables that get rolling windows
    variables = ['ta', 'vpd', 'sw_in', 'rh']
    windows = ['3h', '6h', '12h', '24h', '48h', '72h', '168h', '336h', '720h']
    stats = ['mean', 'std', 'min', 'max', 'range']
    
    for var in variables:
        for window in windows:
            for stat in stats:
                feature_name = f'{var}_{stat}_{window}'
                var_desc = {
                    'ta': 'Temperature',
                    'vpd': 'VPD',
                    'sw_in': 'Shortwave radiation',
                    'rh': 'Relative humidity'
                }.get(var, var)
                stat_desc = {
                    'mean': 'mean',
                    'std': 'standard deviation',
                    'min': 'minimum',
                    'max': 'maximum',
                    'range': 'range'
                }.get(stat, stat)
                rolling_features[feature_name] = f'{var_desc} {window}-hour {stat_desc}'
    
    return rolling_features

if __name__ == "__main__":
    print("🔍 Extracting feature descriptions from v2 pipeline code...")
    
    # Extract descriptions from code
    feature_descriptions = extract_feature_descriptions_from_code()
    
    # Add lagged features
    lagged_features = get_lagged_feature_descriptions()
    feature_descriptions.update(lagged_features)
    
    # Add rolling features
    rolling_features = get_rolling_feature_descriptions()
    feature_descriptions.update(rolling_features)
    
    print(f"📈 Found {len(feature_descriptions)} feature descriptions")
    
    # Save to file
    df_descriptions = pd.DataFrame([
        {'feature_name': name, 'description': desc}
        for name, desc in feature_descriptions.items()
    ])
    df_descriptions.to_csv('v2_feature_descriptions.csv', index=False)
    print("💾 Saved feature descriptions to: v2_feature_descriptions.csv")
    
    # Print first 20 for verification
    print("\n📋 First 20 feature descriptions:")
    for i, (name, desc) in enumerate(list(feature_descriptions.items())[:20]):
        print(f"  {name}: {desc}") 