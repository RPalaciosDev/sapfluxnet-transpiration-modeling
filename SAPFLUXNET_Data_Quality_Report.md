# SAPFLUXNET Data Quality Analysis Report

## Understanding Site Performance in Spatial Validation Models

**Analysis Date:** July 25, 2025  
**Focus Site:** COL_MAC_SAF_RAD (Macagual Universidad de la Amazonia)  
**Dataset:** 165 SAPFLUXNET sites  

---

## Executive Summary

Our comprehensive analysis revealed that **COL_MAC_SAF_RAD's poor performance in spatial validation is not due to environmental data quality issues but rather insufficient temporal coverage**. While the site has excellent environmental data quality with no problematic zero values or missing data issues, it contains only **13.2 days** of measurements, making it inadequate for robust spatial modeling.

### Key Findings

- ✅ **Good Data Quality**: COL_MAC_SAF_RAD has 100% environmental variable coverage with no excessive zero values
- ❌ **Insufficient Temporal Coverage**: Only 13.2 days vs. required minimum of 30+ days for spatial modeling
- 📊 **Rare Problem**: Only 3 out of 165 sites (1.8%) have insufficient temporal coverage
- 🎯 **Root Cause**: Temporal sampling inadequacy, not data quality issues

---

## Detailed Analysis Results

### 1. Environmental Data Quality Assessment

**Methodology**: Analyzed environmental data quality across all 165 SAPFLUXNET sites, focusing on:

- Zero value percentages in critical variables (temperature, humidity, VPD, PPFD, solar radiation)
- Missing data rates
- Data type consistency issues

**COL_MAC_SAF_RAD Results**:

- **Environmental Score**: 100/100 (excellent)
- **Critical Variables Present**: 5/5 (100% coverage)
- **Zero Value Issues**: None detected
- **Missing Data Rate**: <5% across all variables
- **Data Type Consistency**: No conversion issues

**Cross-Site Comparison**:

- Average environmental score: 91.4/100
- 136 sites rated as "Excellent" (≥85 score)
- COL_MAC_SAF_RAD ranks among the top sites for data quality

### 2. Temporal Coverage Analysis

**COL_MAC_SAF_RAD Temporal Characteristics**:

- **Duration**: 13.2 days (Jan 13-27, 2015)
- **Total Records**: 1,270 observations
- **Sampling Frequency**: 15 minutes (high resolution)
- **Temporal Ranking**: #3 worst out of 165 sites

**Dataset-Wide Coverage Distribution**:

- **Minimum Coverage**: 12.0 days (ARG_MAZ)
- **Maximum Coverage**: 793.0 days (USA_MOR_SF)
- **Average Coverage**: 193.1 days
- **Median Coverage**: 208.3 days

**Coverage Categories**:

- 🔴 **Very Short (<7 days)**: 0 sites (0.0%)
- 🟠 **Short (7-30 days)**: 3 sites (1.8%) - *includes COL_MAC_SAF_RAD*
- 🟡 **Moderate (30-90 days)**: 22 sites (13.3%)
- 🟢 **Good (90-365 days)**: 124 sites (75.2%)
- 🟢 **Excellent (≥365 days)**: 16 sites (9.7%)

### 3. Site Characteristics Comparison

**COL_MAC_SAF_RAD Profile**:

- **Location**: Colombia, Tropical rainforest biome
- **Species**: Theobroma cacao (Cocoa)
- **Elevation**: 360m
- **Stand Age**: 4 years
- **Environmental Conditions**: Stable tropical conditions (24.6°C average temperature)

**Similar Short-Coverage Sites**:

1. **ARG_MAZ**: 12.0 days, Argentina, Woodland/Shrubland
2. **ARG_TRE**: 13.0 days, Argentina, Woodland/Shrubland  
3. **COL_MAC_SAF_RAD**: 13.2 days, Colombia, Tropical rainforest

---

## Impact on Spatial Modeling

### Why Temporal Coverage Matters for Spatial Models

**Environmental Gradient Representation**:

- 13.2 days cannot capture seasonal environmental variability
- Limited range of weather conditions (only mid-January period)
- Insufficient temperature, humidity, and radiation gradients
- No representation of plant physiological responses across conditions

**Model Training Implications**:

- **Feature Space Coverage**: Narrow environmental feature space limits model generalization
- **Relationship Learning**: Insufficient data to learn robust environment-sap flow relationships
- **Spatial Extrapolation**: Poor representation for predicting at other locations
- **Validation Performance**: Expected poor performance in spatial validation due to limited training data representativeness

### Comparison with High-Performing Sites

**Well-Represented Sites (>365 days)**:

- USA_MOR_SF: 793 days - captures multiple seasons and environmental cycles
- FIN_HYY_SME: 435 days - represents full annual cycles including winter conditions
- DEU_STE_4P5: 417 days - captures complete seasonal transitions

**These sites provide**:

- Full seasonal environmental gradients
- Complete plant physiological response ranges
- Robust training data for spatial extrapolation
- High spatial validation performance

---

## Recommendations

### 1. Immediate Actions

**For COL_MAC_SAF_RAD Specifically**:

- ❌ **Exclude from spatial model training** due to insufficient temporal coverage
- ✅ **Use as validation/test site only** if additional data collection is not possible
- 🔄 **Prioritize for extended data collection** if site is still accessible

**For Current Spatial Models**:

- Remove the 3 sites with <30 days coverage from training datasets
- Focus training on the 140 sites with >90 days coverage
- Use short-coverage sites only for validation or sensitivity analysis

### 2. Site Selection Criteria for Future Spatial Models

**Minimum Requirements**:

- **Temporal Coverage**: ≥90 days (preferably ≥180 days)
- **Seasonal Representation**: Data spanning multiple seasons or environmental conditions
- **Environmental Variable Completeness**: ≥80% of critical variables present
- **Data Quality**: <20% missing data across critical variables

**Optimal Sites for Spatial Model Training**:

- Sites with ≥365 days coverage (16 sites available)
- Sites with 90-365 days but good seasonal coverage (124 sites available)
- Prioritize sites with diverse environmental conditions and geographic distribution

### 3. Data Collection Strategy

**For Existing Short-Coverage Sites**:

- Assess feasibility of returning to collect additional data
- Prioritize sites in underrepresented geographic regions or biomes
- Consider partnerships with local institutions for ongoing monitoring

**For New Site Establishment**:

- Plan minimum 1-year measurement campaigns
- Ensure coverage of key seasonal transitions
- Include multiple weather cycles (dry/wet seasons, temperature extremes)

---

## Technical Specifications

### Analysis Tools Used

1. **Comprehensive Site Analyzer**: Evaluated temporal coverage, environmental data quality, and site metadata
2. **Environmental Quality Analyzer**: Assessed zero values, missing data, and data type consistency
3. **Temporal Coverage Sorter**: Ranked all sites by measurement duration and identified problematic sites

### Data Quality Metrics Applied

- **Temporal Representativeness**: Duration, observation count, seasonal coverage
- **Environmental Completeness**: Critical variable presence, missing data rates
- **Data Integrity**: Zero value patterns, data type consistency
- **Modeling Suitability**: Combined score incorporating all quality dimensions

---

## Conclusions

1. **COL_MAC_SAF_RAD is not a "bad data" problem** - it's an "insufficient data" problem
2. **Environmental data quality is excellent** across most SAPFLUXNET sites when timestamp issues are ignored
3. **Temporal coverage is the critical factor** determining site suitability for spatial modeling
4. **Very few sites have inadequate coverage** - this is a manageable data curation issue
5. **Clear site selection criteria** can significantly improve spatial model performance

The analysis demonstrates that **spatial model performance issues can often be resolved through better site selection based on temporal coverage criteria rather than complex data cleaning procedures**. Focusing training datasets on the 140 sites with adequate temporal coverage (≥90 days) should substantially improve spatial validation performance.

---

**Analysis conducted using memory-efficient chunked processing to handle large datasets while maintaining accuracy and completeness.**
