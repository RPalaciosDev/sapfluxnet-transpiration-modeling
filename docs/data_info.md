*   **Core Transpiration Data**:
    *   The primary component of the database is **sub-daily time series of sap flow**, which represents the water movement through the plant's xylem.
    *   Sap flow rates can be expressed at different levels: per individual plant, per unit sapwood area, or per unit leaf area. The database predominantly provides whole-plant level sap flow.
    *   These measurements cover periods from 1995 to 2018, with half of the datasets being at least 3 years long, allowing for analysis of seasonal and diel (daily) patterns.

*   **Associated Hydrometeorological Drivers**:
    *   Each dataset includes **ancillary time series of key hydrometeorological drivers**.
    *   These drivers include **air temperature** (available for all datasets), **vapour pressure deficit (VPD)** (estimated for most, only 2 out of 202 missing after QC2), **radiation variables** (shortwave, photosynthetically active, net radiation), **wind speed**, and **precipitation**.
    *   **Soil water content** is also provided, with shallow depth measurements (0-30 cm) available for 56% of datasets and deep soil layers for 27%.

*   **Metadata (Documenting Data Context and Characteristics)**:
    The database includes extensive metadata, categorized to describe various aspects of the data. This metadata is crucial for understanding the ecological and technical context of the measurements.

    *   **Site Metadata**: Provides general information about the geographical location.
        *   **Site name** (`si_name`), **country code** (`si_country`), and **contact information** (`si_contact_firstname`, `si_contact_lastname`, `si_contact_email`, `si_contact_institution`) for contributors.
        *   **Geographical coordinates** (latitude `si_lat`, longitude `si_long`), and **elevation** (`si_elev`).
        *   Links to **relevant publications** (`si_paper`), and details on **recent and historic disturbances or management events** (`si_dist_mgmt`).
        *   **Vegetation type** based on IGBP classification (`si_igbp`) and participation in **FLUXNET** (`si_flux_network`) or **DENDROGLOBAL** (`si_dendro_network`) networks.
        *   Calculated **mean annual temperature** (`si_mat`), **mean annual precipitation** (`si_map`), and **biome classification** (`si_biome`).

    *   **Stand Metadata**: Describes the characteristics of the forest stand or vegetation community where measurements were taken.
        *   **Stand name** (`st_name`) and **growth condition** (origin and management, `st_growth_condition`).
        *   Applied **stand-level treatments** (`st_treatment`), such as thinning, irrigation, throughfall exclusion, or wildfire impact.
        *   **Mean stand age** (`st_age`), **canopy height** (`st_height`), **total stem density** (`st_density`), and **total stand basal area** (`st_basal_area`).
        *   **Total maximum stand leaf area index (LAI)** (`st_lai`).
        *   **Aspect** (`st_aspect`), **terrain** (`st_terrain`), **soil total depth** (`st_soil_depth`), and **soil texture class** (`st_soil_texture`) including sand, silt, and clay percentages (`st_sand_perc`, `st_silt_perc`, `st_clay_perc`).

    *   **Species Metadata**: Provides information specific to each measured species within a dataset.
        *   **Scientific name** (`sp_name`) and the **number of trees measured** for each species (`sp_ntrees`).
        *   **Leaf habit** (`sp_leaf_habit`) (e.g., evergreen, deciduous) and the **percentage of basal area** occupied by each species over the total stand basal area (`sp_basal_area_perc`).

    *   **Plant Metadata**: Contains detailed attributes for each individual measured plant.
        *   **Plant code** (`pl_name`), **species identity** (`pl_species`), and any **experimental treatment** applied at the individual level (`pl_treatment`).
        *   **Biometric data**: **diameter at breast height (DBH)** (`pl_dbh`), **height** (`pl_height`), **age** (`pl_age`), **social status** (`pl_social`), **sapwood area** (`pl_sapw_area`), **sapwood depth** (`pl_sapw_depth`), and **bark thickness** (`pl_bark_thick`).
        *   **Leaf area** (`pl_leaf_area`).
        *   **Technical details of sap flow measurements**: **method used** (`pl_sens_meth`), **sensor manufacturer** (`pl_sens_man`), **correction methods for natural temperature gradients** (`pl_sens_cor_grad`), **zero flow determination method** (`pl_sens_cor_zero`), and whether **species-specific calibration** was used (`pl_sens_calib`).
        *   **Sap flow units** (harmonized `pl_sap_units` and original `pl_sap_units_orig`), **sensor length** (`pl_sens_length`), **installation height** (`pl_sens_hgt`), and **sub-daily timestep** (`pl_sens_timestep`).
        *   Information on **radial** (`pl_radial_int`) and **azimuthal integration** (`pl_azimut_int`) procedures to account for within-stem variation in sap flow.

    *   **Environmental Metadata**: Describes the measurement details and availability of environmental variables.
        *   **Time zone** (`env_time_zone`) and whether **daylight saving time** was applied (`env_time_daylight`).
        *   **Sub-daily timestep** of environmental measurements (`env_timestep`).
        *   **Location of sensors** for air temperature (`env_ta`), relative humidity (`env_rh`), VPD (`env_vpd`), shortwave incoming radiation (`env_sw_in`), photosynthetic photon flux density (`env_ppfd_in`), net radiation (`env_netrad`), and wind speed (`env_ws`).
        *   **Location of precipitation measurements** (`env_precip`).
        *   **Average depths for shallow and deep soil water content measures** (`env_swc_shallow_depth`, `env_swc_deep_depth`).
        *   Availability of **water potential values** (`env_plant_watpot`) and **seasonal course of leaf area data** (`env_leafarea_seasonal`).

This detailed organization of data and metadata allows SAPFLUXNET to support diverse research, from plant ecophysiology to large-scale ecohydrological studies.