# Data sources

This file records the principal external datasets used by the workflow. Detailed dataset-level provenance is provided in `docs/data_inventory.csv`.

## MODIS vegetation products

Annual vegetation-index preprocessing follows the supplied Google Earth Engine code. The NDVI workflow merges MODIS Collection 6.1 `MOD13A1` and `MYD13A1` before computing the annual pixelwise maximum. EVI robustness uses the corresponding EVI products. MSAVI robustness uses red/NIR reflectance from the same product family with `SummaryQA <= 1` before MSAVI calculation and annual maximum.

Selected downstream TIFF thresholding and EPSG:8857 reprojection were performed in ArcGIS Pro. The original production scripts for those manual GIS steps are not retained.

## ERA5-Land

Wettest-90-day hydroclimatic drivers are derived from `ECMWF/ERA5_LAND/DAILY_AGGR` in Google Earth Engine. The retained workflow computes the within-year 90-day precipitation maximum and pixel-specific startDOY, then extracts temperature, VPD, soil moisture, evapotranspiration and net shortwave radiation over the aligned window.

The later raster-to-100-km²-hexagon spreadsheet conversion code is unavailable; selected downstream processed tables are therefore treated as reproducibility inputs where necessary.

## CAMS XCO2-indexed background

- Dataset: **CAMS global inversion-optimised greenhouse gas fluxes and concentrations**
- Provider: Copernicus Atmosphere Monitoring Service / Atmosphere Data Store
- DOI: `10.24381/ed2851d2`
- Main processed input: `data/processed/background/up/CO2_annual_tif.xlsx`
- Role: spatially heterogeneous XCO2-indexed shared-background component in the main attribution analysis.

The original retrieval request and gridded-product-to-hexagon conversion script are no longer available. The surviving processed table is preserved unchanged.

## NOAA CarbonTracker

- Product family: **NOAA Global Monitoring Laboratory CarbonTracker CO2**
- Landing page: `https://gml.noaa.gov/ccgg/carbontracker/`
- Processed sensitivity input: `data/processed/background/up/NOAA_CO2_EPSG8857.xlsx`
- Role: alternative background-source sensitivity.

The surviving workbook spans 2000–2022. Because the original extraction script is unavailable, the exact release composition used for the final years cannot be reconstructed from the surviving files alone.

## Dryland extent and land cover

The dryland study mask derives from the UNEP-WCMC global dryland extent used in the analysis. Land-cover process contexts use the ESA CCI/Copernicus 300-m land-cover products at the 2000 and 2022 endpoints.

## Vegetation optical-depth validation

VOD robustness uses VODCA v2 CXKu and a merged AMSR-E/AMSR2/WindSat C-band VOD product as documented in `docs/data_inventory.csv` and the scripts under `code/08_robustness/vod/`.
