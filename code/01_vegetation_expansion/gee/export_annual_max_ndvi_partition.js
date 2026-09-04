// 研究区
var roi = ee.FeatureCollection('projects/i-informatics-407307/assets/drylands_part_21'); // 替换为你的研究区

// 参数设置
var year = 2022;
var start = ee.Date.fromYMD(year, 1, 1);
var end = ee.Date.fromYMD(year, 12, 31);

// 加载 MODIS Terra + Aqua NDVI 数据集
var modis_terra = ee.ImageCollection('MODIS/061/MOD13A1')
                    .select('NDVI')
                    .filterDate(start, end)
                    .filterBounds(roi);

var modis_aqua = ee.ImageCollection('MODIS/061/MYD13A1')
                    .select('NDVI')
                    .filterDate(start, end)
                    .filterBounds(roi);

// 融合 Terra 与 Aqua
var modis_merged = modis_terra.merge(modis_aqua);

// 计算年度 NDVI 最大值（单位修正 ×0.0001）
var ndvi_max = modis_merged.max().multiply(0.0001).clip(roi);

// 可视化（可选）
var ndviVis = {
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
Map.addLayer(ndvi_max, ndviVis, 'NDVI_' + year);

// ===========================
// 导出到 Google Drive
// ===========================
Export.image.toDrive({
  image: ndvi_max,
  description: 'NDVI_MODIS_21_' + year,
  fileNamePrefix: 'NDVI_MODIS_21_' + year,
  folder: 'MODIS_NDVI',  // 👈 你在 Google Drive 中的文件夹名
  region: roi,
  scale: 500,            // MOD13A1 的分辨率为 500m
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
