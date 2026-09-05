# Workflow

## 1. Vegetation source rasters

Use the Google Earth Engine scripts under `src/01_vegetation_expansion/gee/` for annual-maximum NDVI preprocessing. EVI and MSAVI robustness source scripts are under `src/08_robustness/gee/`.

Partitioned GeoTIFFs can be mosaicked with:

```bash
python src/00_preprocessing/mosaic_geotiffs.py --input-folder <tile_folder> --output <annual_mosaic.tif>
```

## 2. Thresholding and projection

One reusable engine is used for all recovered threshold-based vegetation configurations:

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml
```

The locked coding is `value <= threshold -> 1` and `value > threshold -> 0`; the output is reprojected with nearest-neighbour resampling to EPSG:8857 at 500 m. The main NDVI threshold is 0.20. NDVI threshold sensitivities use 0.16, 0.18, 0.22 and 0.24. The recovered MSAVI branch uses 0.14.

The executed EVI threshold value is not recoverable from the current code archive. `evi.yaml` therefore begins from existing thresholded EVI rasters and does not invent a threshold.

## 3. Vegetated fraction and hexagon trends

Use the same trend engine for the main and sensitivity configurations:

```bash
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml
```

The engine generates an axial equal-area hexagon grid, computes annual vegetated fraction as `(count - sum) / count`, and applies the tie-corrected Mann–Kendall test and Sen's slope using the actual year values. The main analysis uses 100 km²; the 75 and 125 km² sensitivity analyses change only the configuration.

## 4. Endpoint summaries and process contexts

`src/01_vegetation_expansion/calculate_three_year_endpoint_change.py` computes the 2000–2002 versus 2020–2022 three-year-window endpoint summary.

`src/02_landcover_context/build_process_context_table.py` prepares the process-context table used by downstream stratification and Figure 4 analyses.

## 5. Hydroclimate

`src/03_hydroclimate/gee/compute_w90_precipitation_startdoy.js` identifies the within-year wettest consecutive 90-day precipitation window and exports P90/startDOY.

`src/03_hydroclimate/gee/export_w90_era5land_drivers.js` aligns ERA5-Land temperature, VPD, soil moisture, evapotranspiration and net shortwave radiation to the precomputed W90 window.

The missing raster-to-hex extraction step is documented in `docs/preprocessing_reproducibility.md`.

## 6. SEM and attribution

`src/04_sem/fit_sem_hydroclimate_structure.py` runs the main hydroclimatic SEM diagnostic. `src/04_sem/run_sem_structure_sensitivity.py` runs the predefined structural sensitivity.

`src/05_attribution/run_panel_attribution.py` is the main VPD-residual NAT0 panel fixed-effects attribution engine. Diagnostics are stored under `src/05_attribution/diagnostics/`; the raw-VPD sensitivity is under `src/08_robustness/attribution/`.

## 7. Stratification and agricultural neighbourhood

`src/06_stratification/` contains aridity and process-context driver-gradient summaries.

`src/07_agricultural_neighbourhood/analyze_agricultural_neighbourhoods.py` runs the agricultural-neighbourhood contextual analysis.

## 8. Robustness and figures

Additional vegetation-index, VOD, pixel-vs-hex and attribution checks are under `src/08_robustness/`.

Publication figure scripts are separated from analytical engines under `figures/main/` and `figures/supplementary/`. Generated PNG/PDF/SVG files are written to `outputs/` and are not tracked.
