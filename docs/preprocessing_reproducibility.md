# Manual preprocessing and unrecovered conversion steps

This document defines the reproducibility boundary for preprocessing steps whose executable source code is no longer available. It records the workflow as actually used rather than reconstructing missing code.

## Vegetation-index TIFF processing

Selected TIFF operations were performed interactively in **ArcGIS Pro**. The retained project record identifies at least the following operations as part of that manual GIS stage:

- thresholding of vegetation-index rasters for the threshold-defined vegetation-cover workflow;
- reprojection of relevant TIFF products to **EPSG:8857**.

No executable production script for these ArcGIS Pro operations is retained. Historical Python provenance such as `TIF_yuzhi.py` must therefore not be presented as the canonical production implementation unless an exact source and output-equivalence record is recovered later.

## Raster/netCDF to 100-km² hexagon tables

The scripts that converted some gridded TIFF/netCDF products into the final 100-km² hexagon-aligned CSV/XLSX tables are no longer available. This affects parts of the vegetation, hydroclimate and atmospheric-background preprocessing chain.

The repository therefore does **not** claim complete raw-data-to-final-table executable reproducibility for these missing conversion stages. The scientific method description is the authoritative specification of the operation, and the released processed hexagon-year tables are the reproducibility starting point for downstream statistical analyses where available.

## What is still executable

The repository retains executable code for the recovered downstream analyses, including the W90 GEE extraction components, panel attribution, stratification and figure generation. Where a recovered script consumes a processed table created by an unrecovered preprocessing step, that dependency is stated explicitly in `docs/data_inventory.csv` and in the script configuration/header where applicable.

## Non-reconstruction policy

Missing scripts are not recreated from memory, filenames or inferred GIS defaults. This avoids introducing undocumented changes in resampling, pixel alignment, zonal aggregation, NoData handling, or projection parameters. A future recovered original script can be reintroduced only after provenance and output-equivalence checks.
