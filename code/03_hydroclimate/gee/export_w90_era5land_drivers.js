/*******************************************************
 * Wettest-90d aligned ERA5-Land drivers (using precomputed startDOY)
 * Years: 2000-2022
 *
 * Input assets (per year, global):
 *   projects/i-informatics-407307/assets/ERA5Land_startDOY/<year>
 *     - single-band: startDOY (1-based), per pixel
 *
 * For each year, aggregate ERA5-Land DAILY_AGGR variables over the 90-day window:
 *   [startDOY, startDOY+89]
 *
 * Outputs (per year, per metric = 1 global GeoTIFF):
 *   - T2m90_mean (°C)
 *   - VPD90_mean (kPa) from T2m & Td2m
 *   - SM_L1_90_mean (m3/m3)
 *   - SM_root_90_mean (mean of layers 1-3)
 *   - ET90_sum (mm) from total_evaporation_sum (sign fixed)
 *   - SWnet90_mean (W/m2) from surface_net_solar_radiation_sum (J/m2/day -> W/m2)
 *
 * Export strategy: stable mode
 *   - Do NOT specify crs/scale/crsTransform/dimensions in Export
 *   - Use region only; avoid internal reprojection bugs
 *******************************************************/

// =============================
// Parameters
// =============================
var startYear = 2018;
var endYear = 2022;

var globalRegion = ee.Geometry.Rectangle([-180, -90, 180, 90], null, false);

var outFolder = 'GEE_Exports';
var maxPixels = 1e13;
var NODATA = -9999;

// StartDOY asset root (you renamed assets as .../2000, .../2001, ...)
var STARTDOY_ROOT = 'projects/i-informatics-407307/assets/ERA5Land_startDOY/';

// ERA5-Land daily
var era5d = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');

// ERA5-Land band names
var B_T2M = 'temperature_2m';
var B_TD2M = 'dewpoint_temperature_2m';
var B_SM1 = 'volumetric_soil_water_layer_1';
var B_SM2 = 'volumetric_soil_water_layer_2';
var B_SM3 = 'volumetric_soil_water_layer_3';
var B_ETSUM = 'total_evaporation_sum';
var B_SWNET = 'surface_net_solar_radiation_sum';

// =============================
// Helper: saturation vapor pressure (kPa) from temperature in °C
// Tetens formula
// =============================
function es_kPa(Tc) {
  return Tc.expression(
    '0.6108 * exp(17.27 * T / (T + 237.3))',
    {T: Tc}
  );
}

// =============================
// Helper: build daily ERA5-Land derived bands
// =============================
function era5DailyDerived(img) {
  var tC  = img.select(B_T2M).subtract(273.15).rename('t2m_c');
  var tdC = img.select(B_TD2M).subtract(273.15).rename('td2m_c');

  var vpd = es_kPa(tC).subtract(es_kPa(tdC)).rename('vpd_kpa'); // kPa

  var sm1 = img.select(B_SM1).rename('sm_l1');
  var smRoot = img.select([B_SM1, B_SM2, B_SM3])
    .reduce(ee.Reducer.mean())
    .rename('sm_root');

  // ET: meters -> mm, flip sign so ET positive
  var et_mm = img.select(B_ETSUM).multiply(-1000).rename('et_mm');

  // Net SW: J/m2/day -> W/m2
  var swnet_wm2 = img.select(B_SWNET).divide(86400).rename('swnet_wm2');

  return ee.Image.cat([tC, vpd, sm1, smRoot, et_mm, swnet_wm2])
    .copyProperties(img, ['system:time_start']);
}

// Derived ERA5 collection (server-side)
var era5Derived = era5d
  .select([B_T2M, B_TD2M, B_SM1, B_SM2, B_SM3, B_ETSUM, B_SWNET])
  .map(era5DailyDerived);

// =============================
// Input: startDOY image per year (client-side asset ID)
// =============================
function getStartDOYImage(year) {
  // year is a client-side number from the for-loop
  var assetId = STARTDOY_ROOT + year;  // e.g., .../2000
  return ee.Image(assetId).rename('startDOY');
}

// =============================
// Core: aggregate ERA5-Land over 90-day window defined by startDOY
// =============================
function alignedDriversYear(year) {
  // year is a client-side number; use ee.Number for date ops
  var yearNum = ee.Number(year);

  var yStart = ee.Date.fromYMD(yearNum, 1, 1);
  var yEnd   = ee.Date.fromYMD(yearNum.add(1), 1, 1);
  var nDays  = yEnd.difference(yStart, 'day'); // 365/366

  // startDOY: 1-based per pixel
  var startDOY = getStartDOYImage(year).toInt16();

  // Optional safety: mask out invalid startDOY (e.g., <=0 or too late)
  // Assume non-cross-year window: startDOY <= nDays-89
  var validStart = startDOY.gte(1).and(startDOY.lte(nDays.subtract(89)));
  startDOY = startDOY.updateMask(validStart);

  // Daily images for the year, each tagged with 1-based DOY
  var daily = era5Derived.filterDate(yStart, yEnd).map(function(img) {
    var doy = ee.Number(img.date().difference(yStart, 'day')).add(1); // 1..nDays
    return img.set('doy', doy);
  });

  // For each day, per-pixel in-window mask: startDOY <= doy <= startDOY+89
  var maskedDaily = daily.map(function(img) {
    var doy = ee.Number(img.get('doy'));
    var inWin = startDOY.lte(doy).and(startDOY.add(89).gte(doy));
    return img.updateMask(inWin);
  });

  // Aggregations
  var t90      = maskedDaily.select('t2m_c').mean().rename('t90_c');
  var vpd90    = maskedDaily.select('vpd_kpa').mean().rename('vpd90_kpa');
  var sm1_90   = maskedDaily.select('sm_l1').mean().rename('sm1_90');
  var smroot90 = maskedDaily.select('sm_root').mean().rename('smroot_90');
  var et90     = maskedDaily.select('et_mm').sum().rename('et90_mm');
  var swnet90  = maskedDaily.select('swnet_wm2').mean().rename('swnet90');

  // Validity: require exactly 90 contributing days per pixel
  var cnt = maskedDaily.select('t2m_c').count();
  var fullMask = cnt.eq(90);

  return ee.Image.cat([t90, vpd90, sm1_90, smroot90, et90, swnet90])
    .updateMask(fullMask)
    .toFloat()
    .set('year', yearNum);
}

// =============================
// Export helper: single band per task (stable export mode)
// =============================
function exportBand(img, band, prefix) {
  Export.image.toDrive({
    image: img.select(band).clip(globalRegion).unmask(NODATA),
    description: prefix,
    folder: outFolder,
    fileNamePrefix: prefix,
    region: globalRegion,

    // Stable mode: no crs / scale / transform / dimensions
    maxPixels: maxPixels,
    fileFormat: 'GeoTIFF',
    formatOptions: { noData: NODATA, cloudOptimized: true }
  });
}

// =============================
// Main: create tasks
// =============================
print('Creating tasks (aligned ERA5-Land drivers using startDOY assets): ' + startYear + '-' + endYear);

for (var y = startYear; y <= endYear; y++) {
  print('Year:', y);

  // Quick asset existence check in Console
  print('startDOY asset:', y, getStartDOYImage(y));

  var out = alignedDriversYear(y);

  exportBand(out, 't90_c',      'W90_ERA5Land_TmeanC_' + y);
  exportBand(out, 'vpd90_kpa',  'W90_ERA5Land_VPDkPa_' + y);
  exportBand(out, 'sm1_90',     'W90_ERA5Land_SM_L1mean_' + y);
  exportBand(out, 'smroot_90',  'W90_ERA5Land_SM_rootMean_' + y);
  exportBand(out, 'et90_mm',    'W90_ERA5Land_ETmmSum_' + y);
  exportBand(out, 'swnet90',    'W90_ERA5Land_SWnet_Wm2mean_' + y);
}

print('Done. Run exports in the Tasks panel.');

// Optional visualization check (one year)
Map.centerObject(globalRegion, 2);
var demo = alignedDriversYear(2020);
Map.addLayer(demo.select('vpd90_kpa'), {min: 0, max: 4}, 'VPD90 (2020)');
Map.addLayer(demo.select('smroot_90'), {min: 0, max: 0.5}, 'SMroot90 (2020)');
