# SAPFLUXNET Problematic Sites Analysis

## Overview

This document identifies sites with high quality flag counts that may need special handling or exclusion during data processing.

## Top 18 Sites (50% of All Quality Flags)

These sites account for approximately 50% of all RANGE_WARN and OUT_WARN flags across the entire SAPFLUXNET dataset:

| Rank | Site Code | Total Flags | % of All Flags | Flag Rate | Country | Notes |
|------|-----------|-------------|----------------|-----------|---------|-------|
| 1 | FRA_PUE | 454,705 | 9.1% | 9.1% | France | Highest flag count |
| 2 | CAN_TUR_P74 | 279,652 | 5.5% | 15.8% | Canada | Very high flag rate |
| 3 | CAN_TUR_P39_PRE | 223,036 | 4.4% | 9.2% | Canada | Pre-treatment data |
| 4 | CAN_TUR_P39_POS | 192,773 | 3.8% | 13.2% | Canada | Post-treatment data |
| 5 | IDN_PON_STE | 166,779 | 3.3% | 63.1% | Indonesia | Extremely high flag rate |
| 6 | FRA_HES_HE2_NON | 115,249 | 2.3% | 9.0% | France | Non-irrigated treatment |
| 7 | USA_DUK_HAR | 128,634 | 2.5% | 6.0% | USA | Harvard Forest |
| 8 | SWE_NOR_ST3 | 71,340 | 1.4% | 1.7% | Sweden | Large dataset |
| 9 | USA_UMB_GIR | 117,348 | 2.3% | 1.9% | USA | Girdled treatment |
| 10 | USA_UMB_CON | 99,461 | 2.0% | 1.6% | USA | Control treatment |
| 11 | USA_PJS_P12_AMB | 104,066 | 2.0% | 3.0% | USA | Ambient treatment |
| 12 | USA_SIL_OAK_2PR | 69,016 | 1.4% | 3.0% | USA | Oak forest |
| 13 | USA_SIL_OAK_1PR | 63,561 | 1.2% | 3.0% | USA | Oak forest |
| 14 | USA_PJS_P04_AMB | 77,137 | 1.5% | 2.2% | USA | Ambient treatment |
| 15 | USA_PJS_P08_AMB | 65,212 | 1.3% | 1.8% | USA | Ambient treatment |
| 16 | USA_SYL_HL2 | 73,712 | 1.4% | 16.0% | USA | High flag rate |
| 17 | USA_SIL_OAK_POS | 35,046 | 0.7% | 3.5% | USA | Post-treatment |
| 18 | USA_WIL_WC2 | 38,387 | 0.8% | 13.3% | USA | High flag rate |

## Sites with Extremely High Flag Rates (>50%)

| Site Code | Flag Rate | Total Flags | Country | Recommendation |
|-----------|-----------|-------------|---------|----------------|
| IDN_PON_STE | 63.1% | 166,779 | Indonesia | **EXCLUDE** - Extremely poor quality |
| ZAF_NOO_E3_IRR | 25.9% | 13,419 | South Africa | **EXCLUDE** - Very poor quality |
| GUF_GUY_GUY | 35.5% | 77,419 | French Guiana | **EXCLUDE** - Very poor quality |
| USA_NWH | 53.4% | 9,808 | USA | **EXCLUDE** - Very poor quality |
| USA_TNP | 31.6% | 15,145 | USA | **EXCLUDE** - Very poor quality |
| USA_TNY | 28.9% | 11,931 | USA | **EXCLUDE** - Very poor quality |
| USA_WVF | 16.6% | 12,192 | USA | **EXCLUDE** - Very poor quality |

## Sites with High Flag Rates (20-50%)

| Site Code | Flag Rate | Total Flags | Country | Recommendation |
|-----------|-----------|-------------|---------|----------------|
| USA_SYL_HL2 | 16.0% | 73,712 | USA | **EXCLUDE** - Poor quality |
| USA_WIL_WC2 | 13.3% | 38,387 | USA | **EXCLUDE** - Poor quality |
| CAN_TUR_P39_POS | 13.2% | 192,773 | Canada | **EXCLUDE** - Poor quality |
| CAN_TUR_P74 | 15.8% | 279,652 | Canada | **EXCLUDE** - Poor quality |
| USA_PAR_FER | 16.7% | 62,345 | USA | **EXCLUDE** - Poor quality |
| USA_TNB | 19.4% | 12,260 | USA | **EXCLUDE** - Poor quality |
| USA_TNO | 19.3% | 11,657 | USA | **EXCLUDE** - Poor quality |
| USA_UMB_GIR | 27.9% | 117,348 | USA | **EXCLUDE** - Poor quality |

## Geographic Distribution

### Countries with Most Problematic Sites

1. **USA**: 10 sites in top 18
2. **Canada**: 3 sites in top 18  
3. **France**: 2 sites in top 18
4. **Sweden**: 1 site in top 18
5. **Indonesia**: 1 site in top 18

### Regions with Quality Issues

- **North America**: Heavy concentration of problematic sites
- **Europe**: Moderate issues, mainly France and Sweden
- **Southeast Asia**: One extremely problematic site (Indonesia)
- **South America**: One problematic site (French Guiana)

## Recommendations

### For Data Processing Pipeline

1. **Exclude Extremely Problematic Sites**: Sites with >50% flag rates
2. **Exclude High Flag Rate Sites**: Sites with >20% flag rates  
3. **Apply Stricter Filtering**: For sites with 10-20% flag rates
4. **Monitor Closely**: Sites with 5-10% flag rates

### For Machine Learning

1. **Use as Validation Set**: Problematic sites can test outlier detection
2. **Separate Analysis**: Analyze these sites independently
3. **Quality Thresholds**: Set minimum quality standards for inclusion

## Implementation

The processing script should include a `PROBLEMATIC_SITES` list to automatically skip these sites during processing, with options to:

- Skip entirely (default)
- Apply aggressive filtering
- Process with warnings
- Include for validation purposes

---

*Analysis Date: 2025-01-07*  
*Total Sites Analyzed: 165*  
*Total Quality Flags: 5,090,011*
