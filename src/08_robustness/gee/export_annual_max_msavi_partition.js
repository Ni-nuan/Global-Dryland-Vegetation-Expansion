// 研究区
var roi = ee.FeatureCollection('projects/i-informatics-407307/assets/drylands_part_21');

// 参数设置
var year = 2022;
var start = ee.Date.fromYMD(year, 1, 1);
var end = ee.Date.fromYMD(year, 12, 31);

// ===========================
// 计算 MSAVI 的函数
// ===========================
var addMSAVI = function(image) {
  // MOD13A1 / MYD13A1 反射率比例因子为 0.0001
  var red = image.select('sur_refl_b01').multiply(0.0001);
  var nir = image.select('sur_refl_b02').multiply(0.0001);

  var term = nir.multiply(2).add(1);

  var msavi = term
    .subtract(
      term.pow(2)
          .subtract(nir.subtract(red).multiply(8))
          .max(0)       // 防止个别异常像元导致 sqrt 负值
          .sqrt()
    )
    .divide(2)
    .rename('MSAVI');

  return image.addBands(msavi);
};

// ===========================
// 加载 MODIS Terra 数据集
// ===========================
var modis_terra = ee.ImageCollection('MODIS/061/MOD13A1')
  .select(['sur_refl_b01', 'sur_refl_b02', 'SummaryQA'])
  .filterDate(start, end)
  .filterBounds(roi);

// ===========================
// 加载 MODIS Aqua 数据集
// ===========================
var modis_aqua = ee.ImageCollection('MODIS/061/MYD13A1')
  .select(['sur_refl_b01', 'sur_refl_b02', 'SummaryQA'])
  .filterDate(start, end)
  .filterBounds(roi);

// ===========================
// 简单质量控制函数
// SummaryQA:
// 0 = good data
// 1 = marginal data
// 2 = snow/ice
// 3 = cloudy
// 这里保留 0 和 1
// ===========================
var maskQA = function(image) {
  var qa = image.select('SummaryQA');
  var mask = qa.lte(1);
  return image.updateMask(mask);
};

// ===========================
// 融合 Terra 与 Aqua，并计算 MSAVI
// ===========================
var modis_merged = modis_terra
  .merge(modis_aqua)
  .map(maskQA)
  .map(addMSAVI);

// ===========================
// 计算年度 MSAVI 最大值
// ===========================
var msavi_max = modis_merged
  .select('MSAVI')
  .max()
  .clip(roi);

// ===========================
// 可视化
// ===========================
var msaviVis = {
  min: 0.0,
  max: 1.0,
  palette: [
    'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163',
    '99B718', '74A901', '66A000', '529400',
    '3E8601', '207401', '056201', '004C00',
    '023B01', '012E01', '011D01', '011301'
  ],
};

Map.centerObject(roi);
Map.addLayer(msavi_max, msaviVis, 'MSAVI_' + year);

// ===========================
// 导出到 Google Drive
// ===========================
Export.image.toDrive({
  image: msavi_max,
  description: 'MSAVI_MODIS_21_' + year,
  fileNamePrefix: 'MSAVI_MODIS_21_' + year,
  folder: 'MODIS_MSAVI',
  region: roi.geometry(),
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});