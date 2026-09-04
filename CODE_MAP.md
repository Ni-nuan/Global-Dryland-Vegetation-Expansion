# Code map and execution guide

## Recommended execution sequence

1. **Prepare vegetation and hydroclimate rasters.** Use the GEE scripts under `code/01_vegetation_expansion/gee/` and `code/03_hydroclimate/gee/`. The retained mosaic utility is under `code/00_preprocessing/`.
2. **Detect vegetation-cover expansion.** Use `code/01_vegetation_expansion/compute_hex_vegetation_trends.py` and the endpoint/diagnostic helpers in the same module.
3. **Build land-cover process contexts.** Run `code/02_landcover_context/build_process_context_table.py` once the required transition and attribution inputs are available.
4. **Run hydroclimatic SEM diagnostics.** Use `code/04_sem/fit_hydroclimate_sem.py`; the companion script without the direct SWnet-to-vegetation path is the predefined sensitivity branch.
5. **Run panel attribution.** Use `code/05_attribution/run_panel_attribution.py` for the main VPD-residual NAT0 specification.
6. **Run stratification.** Use the scripts under `code/06_stratification/` for within-aridity and process-context driver gradients.
7. **Run agricultural-neighbourhood analysis.** Use `code/07_agricultural_neighbourhood/run_agricultural_neighbourhood_analysis.py`.
8. **Run robustness analyses.** Use `code/08_robustness/` for raw-VPD, year-effect, VIF, alternative vegetation-index, VOD and pixel-vs-hex checks.
9. **Reproduce figures.** Use `code/09_figures/main/` and `code/09_figures/supplementary/` after their required intermediate outputs exist.

## Method-to-code map

| Analytical component | Primary public script(s) |
|---|---|
| Annual maximum NDVI preprocessing | `code/01_vegetation_expansion/gee/export_annual_max_ndvi_partition.js` |
| Raster mosaicking utility | `code/00_preprocessing/mosaic_geotiffs.py` |
| 100-km² vegetation fraction and MK/Sen trend classification | `code/01_vegetation_expansion/compute_hex_vegetation_trends.py` |
| Three-year endpoint change | `code/01_vegetation_expansion/calculate_three_year_endpoint_change.py` |
| Land-cover process-context table | `code/02_landcover_context/build_process_context_table.py` |
| W90 precipitation and startDOY | `code/03_hydroclimate/gee/compute_w90_precipitation_startdoy.js` |
| W90-aligned ERA5-Land drivers | `code/03_hydroclimate/gee/export_w90_era5land_drivers.js` |
| Hydroclimatic SEM | `code/04_sem/fit_hydroclimate_sem.py` |
| SEM sensitivity without direct SWnet effect | `code/04_sem/run_sem_without_direct_swnet.py` |
| Main panel attribution | `code/05_attribution/run_panel_attribution.py` |
| Aridity-driver stratification | `code/06_stratification/summarize_aridity_driver_stratification.py` |
| Process-context driver quintiles | `code/06_stratification/summarize_process_context_driver_quintiles.py` |
| Agricultural-neighbourhood comparison | `code/07_agricultural_neighbourhood/run_agricultural_neighbourhood_analysis.py` |
| Attribution diagnostics and sensitivity | `code/08_robustness/attribution/` |
| Alternative EVI/MSAVI preprocessing | `code/08_robustness/gee/` |
| Pixel-vs-hex greenness checks | `code/08_robustness/greenness_comparison/` |
| VOD validation | `code/08_robustness/vod/` |
| Main figures | `code/09_figures/main/` |
| Supplementary figures | `code/09_figures/supplementary/` |

## Version policy

The public code tree contains one retained script per current analytical implementation. Working copies, numbered drafts, `_final`, `_new`, `_fixed`, `_refined`, duplicate copies and superseded branches are not stored alongside the public scripts. Where a released script descends from a historically named file, that relationship is recorded in `docs/code_provenance.csv`.
