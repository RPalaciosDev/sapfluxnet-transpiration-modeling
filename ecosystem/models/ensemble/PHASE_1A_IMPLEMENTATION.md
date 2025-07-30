# Phase 1A Implementation: Clean Foundation Data Pipeline

## 🎯 **Goal**

Create a clean foundation by modifying the v3 data pipeline to be conservative with site filtering, excluding sites with no sap flow data, high flag rates (>50%), and moderate flag rates (>20%).

## ✅ **Changes Made**

### **1. Updated Problematic Site Categories**

#### **Before (v3 original):**

- **Extremely problematic**: >50% flag rates (7 sites)
- **High problematic**: 20-50% flag rates (8 sites)
- **Moderate problematic**: 10-20% flag rates (11 sites)

#### **After (Phase 1A):**

- **Extremely problematic**: >80% flag rates (2 sites)
  - `IDN_PON_STE` (63.1% flag rate)
  - `USA_NWH` (53.4% flag rate)
- **High problematic**: 50-80% flag rates (5 sites)
  - `ZAF_NOO_E3_IRR`, `GUF_GUY_GUY`, `USA_TNP`, `USA_TNY`, `USA_WVF`
- **Moderate problematic**: 20-50% flag rates (19 sites)
  - All sites with 20-50% flag rates moved to moderate category

### **2. Added Clean Mode Option**

#### **New Command Line Argument:**

```bash
--clean-mode
```

- **Purpose**: Exclude moderate and high problematic sites (>20% flag rates) for truly clean foundation
- **Effect**: Only includes sites with low flag rates (<20%)
- **Benefit**: Creates clean foundation for gradual expansion

#### **Usage Examples:**

```bash
# Clean mode - only include sites with low flag rates (<20%)
python data_pipeline_v3.py --clean-mode --export-format parquet

# Normal mode - exclude high and extremely problematic sites  
python data_pipeline_v3.py --export-format parquet

# Include all problematic sites
python data_pipeline_v3.py --include-problematic --export-format parquet
```

### **3. Updated Filtering Logic**

#### **Clean Mode Behavior:**

- ✅ **Include**: Sites with any valid sap flow data
- ✅ **Include**: Sites with 12+ days temporal coverage
- ✅ **Include**: Sites with low flag rates (<20%)
- ❌ **Exclude**: Sites with zero valid sap flow data
- ❌ **Exclude**: Sites with >20% flag rates (moderate and high problematic)

#### **Expected Dataset Size:**

- **Current**: ~50 sites (over-aggressive filtering)
- **Target**: ~40-45 sites (clean mode - only low flag rate sites)
- **Benefit**: Clean foundation for gradual expansion

## 📊 **Site Classification Summary**

### **Excluded Sites (Clean Mode):**

- **No valid data**: Dynamic analysis determines
- **Extremely problematic**: 2 sites (>80% flag rates)
- **High problematic**: 5 sites (50-80% flag rates)
- **Moderate problematic**: 19 sites (20-50% flag rates)
- **Insufficient temporal coverage**: Dynamic analysis determines

### **Included Sites (Clean Mode):**

- **Adequate temporal coverage**: ≥90 days
- **Moderate temporal coverage**: 30-90 days  
- **Low flag rates**: <20% flag rates only

## 🔧 **Implementation Details**

### **Modified Files:**

1. **`data_pipeline_v3.py`**:
   - Updated `EXTREMELY_PROBLEMATIC_SITES` (2 sites)
   - Updated `HIGH_PROBLEMATIC_SITES` (5 sites)
   - Updated `MODERATE_PROBLEMATIC_SITES` (19 sites)
   - Added `clean_mode` parameter to `__init__`
   - Modified `should_skip_site` logic
   - Added `--clean-mode` command line argument
   - Updated summary output

### **Key Methods Modified:**

- `__init__()`: Added `clean_mode` parameter
- `should_skip_site()`: Updated filtering logic for clean mode
- `main()`: Added clean mode argument handling

## 🚀 **Next Steps**

### **Immediate Testing:**

1. **Run clean mode pipeline**:

   ```bash
   python data_pipeline_v3.py --clean-mode --export-format parquet
   ```

2. **Verify dataset size**: Should process ~40-45 sites (only low flag rate sites)

3. **Check quality**: Ensure only sites with low flag rates are included

### **Phase 1B Preparation:**

- Test clustering with clean foundation
- If performance is good, gradually add moderate problematic sites
- Monitor performance impact of each addition

## 📈 **Expected Outcomes**

### **Short Term:**

- **Clean training dataset**: Only low flag rate sites
- **High quality foundation**: Minimal data quality issues
- **Better clustering**: Clean base for performance-based clustering

### **Medium Term:**

- **Gradual expansion**: Add moderate sites if clean foundation works
- **Performance monitoring**: Track impact of each addition
- **Foundation for ensemble**: Robust base for clustering and validation

## ⚠️ **Quality Assurance**

### **Monitoring Points:**

1. **Flag rates**: Ensure all included sites have <20% flag rates
2. **Temporal coverage**: Ensure adequate data for each site
3. **Feature quality**: Monitor for any data quality issues
4. **Processing success**: Track which sites process successfully

### **Fallback Options:**

- If dataset is too small, can adjust flag rate thresholds
- Can iteratively add moderate sites based on performance
- Can adjust temporal coverage requirements if needed

## 💡 **Key Benefits**

1. **Truly conservative approach**: Start with cleanest possible foundation
2. **Quality over quantity**: Prioritize data quality over dataset size
3. **Gradual expansion**: Can add sites incrementally based on performance
4. **Flexible filtering**: Multiple modes for different needs
5. **Foundation for ensemble**: Clean base for clustering and validation

This implementation provides a truly clean foundation for Phase 1B (gradual expansion) and Phase 2 (performance-based clustering).
