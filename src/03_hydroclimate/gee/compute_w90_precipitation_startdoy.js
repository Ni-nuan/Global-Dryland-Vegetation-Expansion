/************************************************************
 * ERA5-Land W90 (Wettest 90-day) - ARRAY/CUMSUM METHOD
 *
 * Outputs (per year, per tile):
 *   1) W90_ERA5Land_P90mm_YYYY_<tileName>
 *   2) W90_ERA5Land_startDOY_YYYY_<tileName>
 *
 * Data: ECMWF/ERA5_LAND/DAILY_AGGR
 * Key: avoid explicit 365 rolling windows; use array + prefix sums
 *
 * Notes:
 * - Precip is converted from meters -> mm (x1000).
 * - Exports in EPSG:4326 with fixed 0.1° grid transform.
 * - Run tile by tile; run a small year batch first (e.g. 2000 only).
 ************************************************************/

// =====================
// USER CONFIG
// =====================

// Year batch for this run (keep small for stability)
var batchStartYear = 2001;   // <-- set
var batchEndYear   = 2011;   // <-- set

// Rolling window length (days)
var windowDays = 90;

// Output folder in Google Drive
var outFolder = 'GEE_Exports';

// Choose which tile to export: 0..7 (run sequentially)
var tileIndex = 7;  // <-- change 0..7

// NoData value for exports
var NODATA = -9999;

// Max pixels
var MAX_PIXELS = 1e13;

// =====================
// GLOBAL TILES (8)
// =====================
var globalTiles = [
  {name: 'tile1_west_south',   bounds: [-180, -90,  -90,   0]},
  {name: 'tile2_center_south', bounds: [ -90, -90,    0,   0]},
  {name: 'tile3_east1_south',  bounds: [   0, -90,   90,   0]},
  {name: 'tile4_east2_south',  bounds: [  90, -90,  180,   0]},
  {name: 'tile5_west_north',   bounds: [-180,   0,  -90,  90]},
  {name: 'tile6_center_north', bounds: [ -90,   0,    0,  90]},
  {name: 'tile7_east1_north',  bounds: [   0,   0,   90,  90]},
  {name: 'tile8_east2_north',  bounds: [  90,   0,  180,  90]}
];

var currentTile = globalTiles[tileIndex];
var exportRegion = ee.Geometry.Rectangle(currentTile.bounds, null, false);

print('Tile:', currentTile.name);
print('Bounds:', currentTile.bounds);

// Optional map preview
Map.centerObject(exportRegion, 2);
Map.addLayer(exportRegion, {color: 'red'}, 'Export tile');

// =====================
// DATASET
// =====================
var era5l = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');

// Detect precip band name (some versions use *_sum)
var first = ee.Image(era5l.first());
var bnames = first.bandNames();
var pBand = ee.String(
  ee.Algorithms.If(
    bnames.contains('total_precipitation_sum'),
    'total_precipitation_sum',
    ee.Algorithms.If(bnames.contains('total_precipitation'), 'total_precipitation', 'total_precipitation_sum')
  )
);
print('Using precipitation band:', pBand);

// =====================
// CORE: compute W90 using array + cumsum + argmax
// =====================
function computeW90_ERA5Land(year, regionGeom) {
  year = ee.Number(year);

  var yStart = ee.Date.fromYMD(year, 1, 1);
  var yEnd   = ee.Date.fromYMD(year.add(1), 1, 1);

  // Daily precip in mm/day
  var daily = era5l
    .filterDate(yStart, yEnd)
    .select([pBand])
    .map(function(img) {
      var pmm = img.select([pBand]).multiply(1000).max(0).rename('pmm');
      return pmm.copyProperties(img, ['system:time_start']);
    });

  var nDays = ee.Number(daily.size()); // 365/366

  // Convert daily ImageCollection to a 2D array: [time, band]
  // With one band ('pmm'), band dimension length is 1, but dimension count is 2.
 // 1D time array (avoid 2D [time, band] pitfalls)
// Convert daily collection to array image: typically dims [time, band]
var arr2 = daily.toArray();  // 2D array band

// Force it to 1D over time by keeping only the first (and only) band-slice,
// then projecting away the band dimension.
// Here we assume band dimension is axis=1 (which is the case for toArray() with [time, band]).
var arr1 = arr2.arraySlice(1, 0, 1);     // keep band index 0 -> still 2D, band dim length 1
var arr  = arr1.arrayProject([0]);       // drop band dim, keep time axis -> 1D [time]

// Prefix sum along time axis 0
var cs = arr.arrayAccum(0, ee.Reducer.sum());  // 1D [time]

// Prepend zero
var zero = cs.arraySlice(0, 0, 1).multiply(0);
var cs0  = zero.arrayCat(cs, 0);

var W = ee.Number(windowDays);

// rolling sums: roll[i] = cs0[i+W] - cs0[i]
var rollA = cs0.arraySlice(0, W, nDays.add(1));
var rollB = cs0.arraySlice(0, 0, nDays.add(1).subtract(W));
var roll  = rollA.subtract(rollB);  // 1D [time=nDays-W+1]

// Reduce over time axis 0
var p90 = roll.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0]).rename('P90mm');
var startIdx = roll.arrayArgmax().arrayGet([0]).rename('startIdx');
var startDOY = startIdx.add(1).rename('startDOY');

  // Clip late; do not reproject here (export will define grid)
  p90 = p90.clip(regionGeom).toFloat();
  startDOY = startDOY.clip(regionGeom).toInt16();

  return {p90: p90, startDOY: startDOY};
}

// =====================
// EXPORT HELPERS
// =====================
function exportToDrive(img, desc, regionGeom) {
Export.image.toDrive({
  image: img.unmask(NODATA),
  description: desc,
  folder: outFolder,
  fileNamePrefix: desc,
  region: exportRegion,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: { noData: NODATA, cloudOptimized: true }
});
}

// =====================
// CREATE TASKS (2 per year for selected tile)
// =====================
for (var y = batchStartYear; y <= batchEndYear; y++) {
  print('Create tasks for year:', y);

  var out = computeW90_ERA5Land(y, exportRegion);

  var p90Name = 'W90_ERA5Land_P90mm_' + y + '_' + currentTile.name;
  var doyName = 'W90_ERA5Land_startDOY_' + y + '_' + currentTile.name;

  exportToDrive(out.p90, p90Name, exportRegion);
  exportToDrive(out.startDOY, doyName, exportRegion);
}

print('Tasks created:',
      'tile=' + currentTile.name,
      'years=' + batchStartYear + '-' + batchEndYear,
      'windowDays=' + windowDays);
