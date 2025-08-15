"""
Data constants and configuration mappings for feature engineering.

This module contains all the static configuration data used for categorical encoding
and feature processing in the SAPFLUXNET data pipeline.
"""

# 🚨 CRITICAL: Identity features blacklist - NEVER encode these as predictive features
IDENTITY_BLACKLIST = {
    'site_code', 'site_name', 'site_id', 'site_identifier', 'site',
    'plant_name', 'tree_name', 'tree_id', 'pl_name', 'plant_id',
    # Note: 'species_name' moved to functional group processing instead of blocking
    'pl_species',  # High cardinality (160+ species) - causes overfitting
    'study_id', 'plot_id', 'station_id',
    # Administrative/metadata features that should never be predictive
    'is_inside_country', 'measurement_timestep', 'measurement_frequency', 'timezone_offset'
}

# ⚠️ Pure geographic identifiers - can hinder spatial generalization
PURE_GEOGRAPHIC_IDENTIFIERS = {
    'timezone', 'country', 'continent', 'region', 'state', 'province'
}

# ✅ Climate-based geographic features (ecological information) - allowed with caution
CLIMATE_GEOGRAPHIC_FEATURES = {
    'biome_region', 'koppen_class', 'climate_classification'
}

# ✅ Approved ecological categorical variables (safe to encode)
APPROVED_ECOLOGICAL_FEATURES = {
    'biome', 'igbp_class', 'soil_texture', 'aspect', 'terrain', 
    'growth_condition', 'leaf_habit', 'pl_social',
    'tree_size_class', 'tree_age_class', 'koppen_geiger_code'
}

# 🌿 Species functional group mapping (ecological traits instead of specific species)
# This prevents site-specific memorization while preserving ecological information
SPECIES_FUNCTIONAL_GROUPS = {
    # Needleleaf Evergreen
    'Abies': 'needleleaf_evergreen', 'Picea': 'needleleaf_evergreen', 'Pinus': 'needleleaf_evergreen',
    'Pseudotsuga': 'needleleaf_evergreen', 'Tsuga': 'needleleaf_evergreen', 'Juniperus': 'needleleaf_evergreen',
    'Cupressus': 'needleleaf_evergreen', 'Thuja': 'needleleaf_evergreen', 'Taxus': 'needleleaf_evergreen',
    
    # Needleleaf Deciduous  
    'Larix': 'needleleaf_deciduous', 'Taxodium': 'needleleaf_deciduous',
    
    # Broadleaf Evergreen
    'Quercus ilex': 'broadleaf_evergreen', 'Quercus suber': 'broadleaf_evergreen',
    'Eucalyptus': 'broadleaf_evergreen', 'Acacia': 'broadleaf_evergreen',
    'Olea': 'broadleaf_evergreen', 'Arbutus': 'broadleaf_evergreen',
    'Ilex': 'broadleaf_evergreen', 'Magnolia': 'broadleaf_evergreen',
    
    # Broadleaf Deciduous Temperate
    'Quercus': 'broadleaf_deciduous_temperate', 'Fagus': 'broadleaf_deciduous_temperate',
    'Betula': 'broadleaf_deciduous_temperate', 'Acer': 'broadleaf_deciduous_temperate',
    'Populus': 'broadleaf_deciduous_temperate', 'Tilia': 'broadleaf_deciduous_temperate',
    'Fraxinus': 'broadleaf_deciduous_temperate', 'Castanea': 'broadleaf_deciduous_temperate',
    'Juglans': 'broadleaf_deciduous_temperate', 'Platanus': 'broadleaf_deciduous_temperate',
    
    # Broadleaf Deciduous Tropical
    'Cecropia': 'broadleaf_deciduous_tropical', 'Ficus': 'broadleaf_deciduous_tropical',
    'Terminalia': 'broadleaf_deciduous_tropical', 'Bombax': 'broadleaf_deciduous_tropical',
}

# Define encoding mappings for approved ecological categorical variables
CATEGORICAL_ENCODINGS = {
    'biome': {
        'Tropical and Subtropical Moist Broadleaf Forests': 1,
        'Tropical and Subtropical Dry Broadleaf Forests': 2,
        'Tropical and Subtropical Coniferous Forests': 3,
        'Temperate Broadleaf and Mixed Forests': 4,
        'Temperate Conifer Forests': 5,
        'Boreal Forests/Taiga': 6,
        'Tropical and Subtropical Grasslands, Savannas and Shrublands': 7,
        'Temperate Grasslands, Savannas and Shrublands': 8,
        'Flooded Grasslands and Savannas': 9,
        'Montane Grasslands and Shrublands': 10,
        'Tundra': 11,
        'Mediterranean Forests, Woodlands and Scrub': 12,
        'Deserts and Xeric Shrublands': 13,
        'Mangroves': 14,
        'Woodland/Shrubland': 15
    },
    'igbp_class': {
        'ENF': 1, 'EBF': 2, 'DNF': 3, 'DBF': 4, 'MF': 5,
        'CSH': 6, 'OSH': 7, 'WSA': 8, 'SAV': 9, 'GRA': 10,
        'WET': 11, 'CRO': 12, 'URB': 13, 'CVM': 14, 'SNO': 15, 'BSV': 16
    },
    'soil_texture': {
        'clay': 1, 'clay loam': 2, 'loam': 3, 'loamy sand': 4,
        'sandy clay': 5, 'sandy clay loam': 6, 'sandy loam': 7, 'sand': 8,
        'silt': 9, 'silt loam': 10, 'silty clay': 11, 'silty clay loam': 12
    },
    'aspect': {
        'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SW': 6, 'W': 7, 'NW': 8
    },
    'terrain': {
        'Flat': 1, 'Gentle slope (<2 %)': 2, 'Moderate slope (2-10 %)': 3,
        'Steep slope (>10 %)': 4, 'Valley': 5, 'Ridge': 6
    },
    'growth_condition': {
        'Naturally regenerated, managed': 1,
        'Naturally regenerated, unmanaged': 2,
        'Planted, managed': 3,
        'Planted, unmanaged': 4
    },
    'leaf_habit': {
        'cold deciduous': 1, 'warm deciduous': 2, 'evergreen': 3, 'semi-deciduous': 4
    },
    'pl_social': {
        'dominant': 3, 'codominant': 2, 'intermediate': 1, 'suppressed': 0
    },

    'tree_size_class': {
        'Sapling': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Very Large': 4
    },
    'tree_age_class': {
        'Young': 0, 'Mature': 1, 'Old': 2, 'Very Old': 3, 'Ancient': 4
    },
    'species_functional_group': {
        'needleleaf_evergreen': 1, 'needleleaf_deciduous': 2,
        'broadleaf_evergreen': 3, 'broadleaf_deciduous_temperate': 4, 
        'broadleaf_deciduous_tropical': 5, 'unknown': 0
    }
}

# 🚫 Data processing exclusions and quality flags
PROBLEMATIC_COLUMNS_TO_EXCLUDE = [
    # Basic problematic columns
    'pl_name',           # Plant name - causes inconsistencies
    # 'swc_deep',          # Deep soil water content - allow for water_balance clustering
    'netrad',            # Net radiation - inconsistent across sites
    'seasonal_leaf_area', # Seasonal leaf area - sparse data
    
    # Additional problematic columns (comprehensive list from FeatureEngineer)
    'seasonal_leaf_area_code',
    'stand_name_code', 'stand_remarks_code', 'site_remarks_code', 'env_remarks_code',
    'moisture_availability', 'swc_shallow_depth',
    
    # Redundant features (can be computed during training)
    'stand_soil_texture_code',
    'precip_intensity', 'recent_precip_1h', 'recent_precip_6h', 
    'recent_precip_24h', 'aspect_code', 'species_basal_area_perc', 'site_paper_code',
    'daylight_time'
    
    # Note: wind_stress, ppfd_efficiency, stomatal_conductance_proxy, stomatal_control_index
    # REMOVED from problematic list - these are scientifically important features (Jan 2025)
    # Note: soil_texture_code and terrain_code are kept - these are legitimate ecological features
]

# Quality flags that indicate problematic data points
BAD_QUALITY_FLAGS = ['OUT_WARN', 'RANGE_WARN']

# File format extensions mapping
FILE_FORMAT_EXTENSIONS = {
    'csv': '.csv',
    'parquet': '.parquet',
    'feather': '.feather',
    'hdf5': '.h5',
    'pickle': '.pkl',
    'libsvm': '.svm'  # Standardized to .svm
}

# Individual mapping dictionaries for backward compatibility
BIOME_MAP = CATEGORICAL_ENCODINGS['biome']
IGBP_MAP = CATEGORICAL_ENCODINGS['igbp_class']
SOIL_TEXTURE_MAP = CATEGORICAL_ENCODINGS['soil_texture']
ASPECT_MAP = CATEGORICAL_ENCODINGS['aspect']
TERRAIN_MAP = CATEGORICAL_ENCODINGS['terrain']
GROWTH_CONDITION_MAP = CATEGORICAL_ENCODINGS['growth_condition']
LEAF_HABIT_MAP = CATEGORICAL_ENCODINGS['leaf_habit']
PL_SOCIAL_MAP = CATEGORICAL_ENCODINGS['pl_social']

TREE_SIZE_CLASS_MAP = CATEGORICAL_ENCODINGS['tree_size_class']
TREE_AGE_CLASS_MAP = CATEGORICAL_ENCODINGS['tree_age_class']
SPECIES_FUNCTIONAL_GROUP_MAP = CATEGORICAL_ENCODINGS['species_functional_group']
