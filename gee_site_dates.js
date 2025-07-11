// Google Earth Engine Script with Site-Specific Date Filtering
// Generated: 2025-07-09 11:38:32

// Site-specific date ranges
var siteDateRanges = {
  "ARG_MAZ": ["2009-11-19", "2009-12-01"],
  "ARG_TRE": ["2009-11-18", "2009-12-01"],
  "AUS_BRI_BRI": ["2009-09-09", "2009-10-13"],
  "AUS_CAN_ST1_EUC": ["2006-06-21", "2006-08-08"],
  "AUS_CAN_ST3_ACA": ["2006-06-20", "2006-08-08"],
  "AUS_CAR_THI_00F": ["2008-09-08", "2008-12-10"],
  "AUS_CAR_THI_0PF": ["2008-09-09", "2008-12-09"],
  "AUS_CAR_THI_CON": ["2008-09-09", "2008-12-09"],
  "AUS_CAR_THI_T00": ["2008-09-10", "2008-12-09"],
  "AUS_CAR_THI_T0F": ["2008-09-10", "2008-12-10"],
  "AUS_ELL_MB_MOD": ["2009-06-02", "2012-01-16"],
  "AUS_ELL_UNB": ["2009-06-02", "2012-02-19"],
  "AUS_KAR": ["2008-04-22", "2008-10-01"],
  "AUS_MAR_HSD_HIG": ["2010-02-28", "2012-02-29"],
  "AUS_MAR_HSW_HIG": ["2010-02-28", "2012-02-29"],
  "AUS_MAR_MSD_MOD": ["2010-02-28", "2012-02-29"],
  "AUS_MAR_MSW_MOD": ["2010-02-28", "2012-02-29"],
  "AUS_MAR_UBD": ["2010-02-28", "2012-02-29"],
  "AUS_MAR_UBW": ["2010-02-28", "2012-02-29"],
  "AUS_WOM": ["2012-12-31", "2015-10-31"],
  "AUT_PAT_FOR": ["2007-04-20", "2007-10-16"],
  "AUT_PAT_KRU": ["2007-04-21", "2007-10-10"],
  "AUT_PAT_TRE": ["2007-04-21", "2007-08-20"],
  "AUT_TSC": ["2012-03-13", "2012-10-19"],
  "BRA_CAM": ["2011-03-10", "2011-11-24"],
  "BRA_SAN": ["2008-06-28", "2009-09-23"],
  "CAN_TUR_P39_PRE": ["2009-01-01", "2011-11-21"],
  "CHE_DAV_SEE": ["2009-12-31", "2010-12-31"],
  "CHE_PFY_CON": ["2013-06-06", "2014-12-31"],
  "CHE_PFY_IRR": ["2013-06-06", "2014-11-03"],
  "CHN_ARG_GWD": ["2013-04-30", "2013-09-30"],
  "CHN_ARG_GWS": ["2013-04-30", "2013-09-30"],
  "CHN_HOR_AFF": ["2015-03-26", "2015-10-19"],
  "CHN_YIN_ST1": ["2012-05-25", "2012-08-26"],
  "CHN_YIN_ST2_DRO": ["2012-05-25", "2012-08-26"],
  "CHN_YIN_ST3_DRO": ["2012-05-25", "2012-08-26"],
  "CHN_YUN_YUN": ["2010-10-15", "2011-10-16"],
  "COL_MAC_SAF_RAD": ["2015-01-14", "2015-01-26"],
  "CRI_TAM_TOW": ["2015-01-01", "2016-08-01"],
  "CZE_LIZ_LES": ["2007-06-20", "2009-10-16"],
  "ESP_MAJ_MAI": ["2015-01-29", "2018-05-17"],
  "ESP_MAJ_NOR_LM1": ["2015-02-04", "2018-05-17"],
  "ESP_RIN": ["2007-05-31", "2007-09-11"],
  "ESP_RON_PIL": ["2011-04-14", "2013-12-15"],
  "ESP_SAN_A2_45I": ["2013-06-11", "2013-10-20"],
  "ESP_SAN_A_45I": ["2014-04-20", "2014-11-16"],
  "ESP_SAN_B2_100": ["2013-06-11", "2013-10-20"],
  "ESP_SAN_B_100": ["2014-04-20", "2014-11-16"],
  "ESP_TIL_MIX": ["2010-04-30", "2013-10-03"],
  "ESP_VAL_BAR": ["2003-05-16", "2005-08-31"],
  "ESP_VAL_SOR": ["2003-06-03", "2005-08-31"],
  "ESP_YUN_C1": ["2009-06-20", "2014-11-14"],
  "ESP_YUN_C2": ["2011-04-15", "2014-11-14"],
  "ESP_YUN_T1_THI": ["2010-04-23", "2014-11-29"],
  "ESP_YUN_T3_THI": ["2009-06-21", "2014-11-23"],
  "FRA_HES_HE1_NON": ["1997-04-15", "1999-11-01"],
  "FRA_PUE": ["1999-12-31", "2012-01-05"],
  "GBR_ABE_PLO": ["2001-04-18", "2001-08-23"],
  "GBR_GUI_ST1": ["2003-05-05", "2003-10-16"],
  "GUF_GUY_GUY": ["2014-01-01", "2016-06-07"],
  "HUN_SIK": ["2014-05-22", "2015-11-18"],
  "IDN_JAM_OIL": ["2013-07-02", "2013-09-30"],
  "IDN_JAM_RUB": ["2013-06-12", "2014-01-22"],
  "IDN_PON_STE": ["2007-12-31", "2008-12-31"],
  "ITA_TOR": ["2015-03-03", "2016-12-31"],
  "MDG_SEM_TAL": ["2015-05-06", "2015-09-30"],
  "MDG_YOU_SHO": ["2015-02-28", "2015-06-14"],
  "MEX_COR_YP": ["2008-11-18", "2010-05-12"],
  "NLD_LOO": ["2012-01-01", "2015-12-06"],
  "NLD_SPE_DOU": ["2015-06-18", "2015-08-31"],
  "NZL_HUA_HUA": ["2011-07-06", "2015-09-28"],
  "PRT_MIT": ["2001-01-01", "2004-01-01"],
  "PRT_PIN": ["2007-09-30", "2008-07-31"],
  "RUS_CHE_LOW": ["2016-07-01", "2016-08-29"],
  "SEN_SOU_IRR": ["1998-08-01", "1998-09-30"],
  "SEN_SOU_POS": ["1998-10-01", "2000-01-25"],
  "SEN_SOU_PRE": ["1996-03-01", "1998-07-31"],
  "THA_KHU": ["2006-12-14", "2008-04-23"],
  "USA_CHE_ASP": ["2004-07-27", "2005-07-08"],
  "USA_CHE_MAP": ["2005-07-21", "2005-08-20"],
  "USA_HIL_HF1_PRE": ["2010-06-17", "2011-02-01"],
  "USA_HIL_HF2": ["2010-06-17", "2016-11-15"],
  "USA_INM": ["1998-04-08", "1999-11-11"],
  "USA_MOR_SF": ["2011-07-01", "2013-08-23"],
  "USA_NWH": ["2013-07-26", "2013-10-16"],
  "USA_PER_PER": ["2013-01-01", "2013-12-01"],
  "USA_SWH": ["2013-06-06", "2013-09-24"],
  "USA_TNB": ["1998-05-28", "1999-11-30"],
  "USA_TNO": ["1998-03-31", "1999-11-28"],
  "USA_TNP": ["1998-02-28", "2000-01-01"],
  "USA_TNY": ["1998-05-15", "1999-11-21"],
  "USA_WVF": ["1998-04-24", "1999-11-11"],
  "ZAF_FRA_FRA": ["2015-02-02", "2016-02-02"],
  "ZAF_NOO_E3_IRR": ["2008-05-15", "2010-07-25"],
  "ZAF_RAD": ["2015-09-30", "2016-06-30"],
  "ZAF_SOU_SOU": ["2015-09-30", "2016-06-30"],
  "ZAF_WEL_SOR": ["2013-11-16", "2014-11-11"],
};

// Import study areas
var studyAreas = ee.FeatureCollection('users/yourusername/sapfluxnet_study_area_gee');

// Function to extract data for a specific site
function extractSiteData(siteCode, geometry) {
  var dateRange = siteDateRanges[siteCode];
  if (!dateRange) {
    print('No date range found for site: ' + siteCode);
    return null;
  }
  
  // Extract Sentinel-2 NDVI
  var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(dateRange[0], dateRange[1])
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));
  
  var ndvi = sentinel2.map(function(image) {
    var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return ndvi.clip(geometry).set('site', siteCode);
  });
  
  // Calculate mean NDVI for the site
  var meanNDVI = ndvi.map(function(image) {
    var mean = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geometry,
      scale: 10,
      maxPixels: 1e9
    });
    return image.set('meanNDVI', mean.get('NDVI'));
  });
  
  return meanNDVI;
}

// Process each study area
var allSiteData = studyAreas.map(function(feature) {
  var sites = feature.get('sites').split(',').map(function(s) { return s.trim(); });
  var geometry = feature.geometry();
  
  // Process first site in the region (you can modify this logic)
  var primarySite = sites[0];
  var siteData = extractSiteData(primarySite, geometry);
  
  return siteData;
}).flatten();

// Export all site data
Export.table.toDrive({
  collection: allSiteData,
  description: 'SAPFLUXNET_NDVI_BySite_WithDates',
  fileFormat: 'CSV'
});

print('Site-specific date filtering script ready!');
print('Upload your study areas shapefile and run this script.');
