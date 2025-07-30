# Core Transpiration Features Mapping
# Manually curated based on physiological importance and data availability

CORE_FEATURES = {
    "Maximum Temperature": "ta_max_72h",        # Best coverage temperature max
    "Mean Temperature": "ta_mean_24h",          # Best coverage temperature mean
    "Mean VPD": "vpd_mean_24h",                 # Vapor pressure deficit (full coverage)
    "Incoming Solar Radiation": "sw_in_mean_24h",  # Solar radiation (full coverage)
    "Potential Evapotranspiration": "sw_in",    # Use direct solar as PET proxy (limited coverage but real)
    "Soil Moisture": "swc_shallow",             # Direct soil moisture (though limited coverage)
}

# Feature list for minimal pipeline (physiologically essential):
MINIMAL_FEATURES = [
    "ta_max_72h",      # Maximum temperature (3-day window for stability)
    "ta_mean_24h",     # Mean temperature (daily average)
    "vpd_mean_24h",    # Vapor pressure deficit (daily average) - KEY DRIVER
    "sw_in_mean_24h",  # Solar radiation (daily average) - ENERGY SOURCE
    "sw_in",           # Instantaneous solar radiation - IMMEDIATE ENERGY
    "vpd",             # Instantaneous VPD - IMMEDIATE DRIVING FORCE
]

# Alternative features if primary ones are missing:
FALLBACK_FEATURES = [
    "ta",              # Instantaneous temperature
    "ta_mean_12h",     # 12-hour temperature average
    "vpd_mean_12h",    # 12-hour VPD average
    "sw_in_mean_12h",  # 12-hour solar average
    "ppfd_in",         # Photosynthetic photon flux density
]
