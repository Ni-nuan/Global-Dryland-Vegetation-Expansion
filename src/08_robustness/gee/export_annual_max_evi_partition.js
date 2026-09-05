// ===============================
// 年最大值 EVI 提取脚本（Terra + Aqua 融合）
// ===============================

// 研究区
var roi = ee.FeatureCollection('projects/i-informatics-407307/assets/drylands_part_00'); // 替换为你的研究区

// 参数设置
var start_year = 2000;
var end_year = 2022;

// 加载 MODIS Terra + Aqua EVI 数据集
var modis_terra = ee.ImageCollection('MODIS/061/MOD13A1')
                    .select('EVI');
var modis_aqua = ee.ImageCollection('MODIS/061/MYD13A1')
                    .select('EVI');

// 融合 Terra 与 Aqua
var modis_merged = modis_terra.merge(modis_aqua);

// 逐年计算年度最大 EVI 并导出
for (var year = start_year; year <= end_year; year++) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31);

  // 筛选该年的影像
  var year_modis = modis_merged
                     .filterDate(start, end)
                     .filterBounds(roi);

  // 计算该年最大 EVI（乘以比例因子 0.0001）
  var evi_max = year_modis.max().multiply(0.0001).clip(roi)
                  .rename('EVI');

  // 可视化参数
  var eviVis = {
    min: 0.0,
    max: 1.0,
    palette: [
      'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163',
      '99B718', '74A901', '66A000', '529400',
      '3E8601', '207401', '056201', '004C00',
      '023B01', '012E01', '011D01', '011301'
    ],
  };

  // 在地图上显示
  Map.centerObject(roi);
  Map.addLayer(evi_max, eviVis, 'EVI_' + year);

  // ===========================
  // 导出到 Google Drive
  // ===========================
  Export.image.toDrive({
    image: evi_max,
    description: 'EVI_MODIS_' + year + '_00',
    fileNamePrefix: 'EVI_MODIS_' + year + '_00',
    folder: 'MODIS_EVI',  // 👈 输出文件夹名
    region: roi,
    scale: 500,           // MOD13A1 空间分辨率
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}
