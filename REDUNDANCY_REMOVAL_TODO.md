# Redundancy Removal TODO List for data_pipeline_v3.py

## Overview

This document outlines a systematic approach to remove redundancies from the `data_pipeline_v3.py` file, organized by priority and impact level.

**Estimated Impact:**

- **Lines of code reduced**: ~500-800 lines
- **File size reduction**: ~15-20%
- **Maintainability**: Significantly improved
- **Performance**: Slight improvement from reduced function calls
- **Risk**: Low to medium (mostly safe refactoring)

---

## Phase 1: Helper Function Consolidation (High Impact, Low Risk)

### 1.1 Extract `safe_get_col` Helper Method

**Status**: ⏳ Pending  
**Priority**: 🔴 High  
**Impact**: High  
**Risk**: Low  

**Current Issue**: 4 duplicate `safe_get_col` functions (lines 1406, 1456, 1483, 2101)

**Action Items**:

- [ ] Create single `_safe_get_column(self, df, col_name, default_val=0)` method
- [ ] Replace instance in `create_interaction_features()` (line 1406)
- [ ] Replace instance in `create_cumulative_features()` (line 1456)
- [ ] Replace instance in `create_advanced_rolling_features()` (line 1483)
- [ ] Replace instance in `create_domain_specific_features()` (line 2101)
- [ ] Test all feature creation methods still work correctly

**Files to Modify**:

- `create_interaction_features()`
- `create_cumulative_features()`
- `create_advanced_rolling_features()`
- `create_domain_specific_features()`

### 1.2 Consolidate Timestamp Detection

**Status**: ⏳ Pending  
**Priority**: 🔴 High  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: Repeated timestamp column detection in multiple methods

**Action Items**:

- [ ] Create `_find_timestamp_column(self, df)` method
- [ ] Cache result in instance variable to avoid repeated searches
- [ ] Update `_process_site_standard_adaptive()` to use cached method
- [ ] Update `_process_site_streaming_adaptive()` to use cached method
- [ ] Test timestamp detection still works correctly

---

## Phase 2: Remove Redundant Save Methods (Medium Impact, Low Risk)

### 2.1 Remove `save_dataframe_optimized()`

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: Just calls `save_dataframe_formatted()` (line 503)

**Action Items**:

- [ ] Find all callers of `save_dataframe_optimized()`
- [ ] Update callers to use `save_dataframe_formatted()` directly
- [ ] Delete `save_dataframe_optimized()` method
- [ ] Test saving functionality still works

### 2.2 Consolidate Streaming Save Logic

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: `save_streaming_chunk_optimized()` duplicates CSV logic

**Action Items**:

- [ ] Move CSV append logic into `save_dataframe_formatted()` with streaming parameter
- [ ] Remove `save_streaming_chunk_optimized()` method
- [ ] Update streaming processing to use consolidated method
- [ ] Test streaming save functionality

---

## Phase 3: Consolidate Processing Paths (High Impact, Medium Risk)

### 3.1 Extract Common Validation Logic

**Status**: ⏳ Pending  
**Priority**: 🔴 High  
**Impact**: High  
**Risk**: Medium  

**Current Issue**: Identical validation in both standard and streaming processing

**Action Items**:

- [ ] Create `_validate_site_files(self, site)` method
- [ ] Extract file existence checks
- [ ] Extract sap flow validation logic
- [ ] Extract metadata loading
- [ ] Update `_process_site_standard_adaptive()` to use shared validation
- [ ] Update `_process_site_streaming_adaptive()` to use shared validation
- [ ] Test both processing paths still work

**Files to Modify**:

- `_process_site_standard_adaptive()`
- `_process_site_streaming_adaptive()`

### 3.2 Consolidate Feature Creation Methods

**Status**: ⏳ Pending  
**Priority**: 🔴 High  
**Impact**: High  
**Risk**: Medium  

**Current Issue**: `create_lagged_features_adaptive()` vs lagged features in `create_all_features()`

**Action Items**:

- [ ] Merge `create_lagged_features_adaptive()` into main feature creation
- [ ] Merge `create_rolling_features_adaptive()` into main feature creation
- [ ] Add adaptive parameters to existing methods
- [ ] Remove `create_lagged_features_adaptive()` method
- [ ] Remove `create_rolling_features_adaptive()` method
- [ ] Test all feature creation still works correctly

---

## Phase 4: Memory Management Optimization (Medium Impact, Low Risk)

### 4.1 Create Memory Management Decorator

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: `check_memory_usage()` and `force_memory_cleanup()` called everywhere

**Action Items**:

- [ ] Create `@memory_managed` decorator
- [ ] Apply to methods that need memory monitoring
- [ ] Test memory management still works correctly
- [ ] Benchmark memory usage improvements

**Methods to Decorate**:

- All processing methods
- Feature creation methods
- Data loading methods

### 4.2 Cache Configuration Values

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: Repeated `ProcessingConfig.get_*()` calls

**Action Items**:

- [ ] Cache config values in `__init__()` method
- [ ] Store as instance variables for faster access
- [ ] Update all config access to use cached values
- [ ] Test configuration still works correctly

---

## Phase 5: Externalize Static Data (Low Impact, Low Risk)

### 5.1 Move Site Lists to Configuration

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: Large static site lists (lines 128-163)

**Action Items**:

- [ ] Create `problematic_sites.json` configuration file
- [ ] Load in `__init__()` method
- [ ] Remove hardcoded site lists
- [ ] Test site filtering still works correctly

### 5.2 Simplify Site Classification

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: `PROBLEMATIC_SITES` is just union of three sets

**Action Items**:

- [ ] Load directly from JSON with categories
- [ ] Remove redundant set unions
- [ ] Test site classification still works

---

## Phase 6: Error Handling Consolidation (Medium Impact, Low Risk)

### 6.1 Create Error Handling Decorator

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: Similar try-catch blocks with identical error messages

**Action Items**:

- [ ] Create `@error_handled` decorator with configurable error messages
- [ ] Apply to processing methods
- [ ] Test error handling still works correctly

### 6.2 Standardize Error Messages

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: Inconsistent error message formats

**Action Items**:

- [ ] Create centralized error message constants
- [ ] Update all error messages to use constants
- [ ] Test error reporting still works

---

## Phase 7: File Validation Consolidation (Low Impact, Low Risk)

### 7.1 Create File Validation Utility

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: File existence checks scattered throughout

**Action Items**:

- [ ] Create `_validate_file_exists(self, file_path, file_type)` method
- [ ] Replace all manual `os.path.exists()` checks
- [ ] Test file validation still works correctly

---

## Phase 8: Schema Management Optimization (Medium Impact, Low Risk)

### 8.1 Optimize Feature Standardization

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Current Issue**: `standardize_features_to_reference()` loads reference file each time

**Action Items**:

- [ ] Cache reference features in instance variable
- [ ] Load once in `__init__()` or first use
- [ ] Test feature standardization still works correctly

---

## Phase 9: Code Cleanup (Low Impact, Low Risk)

### 9.1 Remove Dead Code

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: Unused methods or parameters

**Action Items**:

- [ ] Identify unused code
- [ ] Remove unused methods
- [ ] Remove unused imports
- [ ] Test functionality still works

### 9.2 Standardize Method Signatures

**Status**: ⏳ Pending  
**Priority**: 🟢 Low  
**Impact**: Low  
**Risk**: Low  

**Current Issue**: Inconsistent parameter naming and ordering

**Action Items**:

- [ ] Standardize method signatures across similar methods
- [ ] Update all callers to use new signatures
- [ ] Test all methods still work correctly

---

## Phase 10: Testing and Validation (Critical)

### 10.1 Create Test Suite

**Status**: ⏳ Pending  
**Priority**: 🔴 High  
**Impact**: Critical  
**Risk**: Low  

**Action Items**:

- [ ] Create unit tests for each refactored method
- [ ] Ensure functionality remains identical
- [ ] Test edge cases and error conditions
- [ ] Run full pipeline test with sample data

### 10.2 Performance Testing

**Status**: ⏳ Pending  
**Priority**: 🟡 Medium  
**Impact**: Medium  
**Risk**: Low  

**Action Items**:

- [ ] Benchmark before/after performance
- [ ] Verify memory usage improvements
- [ ] Test processing speed improvements
- [ ] Document performance gains

---

## Implementation Priority Summary

### 🔴 High Priority (Do First)

1. Extract `safe_get_col` helper method
2. Remove `save_dataframe_optimized()`
3. Extract common validation logic
4. Create memory management decorator
5. Create test suite

### 🟡 Medium Priority (Do Second)

6. Consolidate feature creation methods
7. Cache configuration values
8. Create error handling decorator
9. Optimize feature standardization
10. Performance testing

### 🟢 Low Priority (Do Last)

11. Externalize static data
12. File validation consolidation
13. Standardize error messages
14. Code cleanup
15. Standardize method signatures

---

## Progress Tracking

**Overall Progress**: 0% Complete  
**Phases Completed**: 0/10  
**Tasks Completed**: 0/45  

### Phase Status

- [ ] Phase 1: Helper Function Consolidation
- [ ] Phase 2: Remove Redundant Save Methods
- [ ] Phase 3: Consolidate Processing Paths
- [ ] Phase 4: Memory Management Optimization
- [ ] Phase 5: Externalize Static Data
- [ ] Phase 6: Error Handling Consolidation
- [ ] Phase 7: File Validation Consolidation
- [ ] Phase 8: Schema Management Optimization
- [ ] Phase 9: Code Cleanup
- [ ] Phase 10: Testing and Validation

---

## Notes

- **Risk Assessment**: Most changes are low-risk refactoring with clear functionality preservation
- **Testing Strategy**: Each phase should be tested independently before moving to the next
- **Rollback Plan**: Keep original file as backup until all testing is complete
- **Documentation**: Update docstrings and comments as methods are refactored

---

*Last Updated: [Current Date]*  
*Created by: AI Assistant*  
*File: REDUNDANCY_REMOVAL_TODO.md*
