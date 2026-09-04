# Workflow and reproducibility boundary

## 1. Vegetation preprocessing and expansion detection

The supplied GEE workflow exports annual maximum MODIS NDVI from merged Collection 6.1 MOD13A1 and MYD13A1 collections. Selected thresholding and EPSG:8857 reprojection were performed in ArcGIS Pro. The retained Python trend engine computes annual vegetation fraction on the 100-km² grid and applies Mann–Kendall and Sen's-slope classification using the locked study rules.

## 2. Land-cover process contexts

Endpoint land-cover transitions are summarized into the six analysis contexts used downstream: Other, No transition, Bare to sparse, Bare/sparse to grass/forest, Agricultural expansion and Urban expansion. These contexts describe dominant transition settings rather than unique causal mechanisms.

## 3. Wettest-90-day hydroclimate

ERA5-Land daily precipitation defines the pixel-specific within-year wettest consecutive 90-day window. The retained GEE code exports P90/startDOY and the aligned temperature, VPD, surface/root-zone soil moisture, evapotranspiration and net shortwave radiation. The raster-to-100-km²-hexagon conversion script is unavailable; processed hexagon tables are the downstream reproduction boundary for that stage.

## 4. SEM diagnostic

`code/04_sem/fit_hydroclimate_sem.py` uses the final L1 surface-soil-moisture formulation (`SM_CHOICE = "l1"`) and the theory-guided hydroclimatic SEM. `run_sem_without_direct_swnet.py` is the predefined structural sensitivity that removes the direct SWnet-to-vegetation path while retaining the remaining formulation.

## 5. Panel attribution

`code/05_attribution/run_panel_attribution.py` is the main VPD-residual NAT0 panel fixed-effects engine. It preserves entity fixed effects, the P90 × VPD_resid interaction constructed before standardization, regression-sample z-scoring, logit response transformation and the absolute-trend natural-share decomposition. The main background panel uses the CAMS-derived XCO2 table; NOAA CarbonTracker is retained as an alternative background-source sensitivity.

## 6. Stratification

Aridity analyses form driver quintiles within each aridity class. Process-context analyses form driver quintiles globally across the eligible UP sample before crossing them with process classes. End-member contrasts use the final Q5 − Q1 convention.

## 7. Agricultural neighbourhood

The neighbourhood workflow uses agricultural-expansion UP hexagons as focal units, boundary-sharing ring 1, and ring 2 defined from neighbours of ring-1 cells while excluding the focal. The main comparison uses the combined ring1+ring2 neighbourhood after retaining non-agricultural UP neighbours. Local excess is the focal observed vegetation-fraction trend minus the median observed trend of neighbouring non-agricultural UP hexagons.

## 8. Robustness and figures

Alternative vegetation indices, raw-VPD attribution, temporal diagnostics, VIF, VOD validation and pixel-vs-hex comparisons are stored under `code/08_robustness/`. Figure scripts are stored under `code/09_figures/` and write generated products to `outputs/`.

## Missing-code policy

Missing preprocessing code is documented rather than guessed. See `preprocessing_reproducibility.md`. The public code tree contains only the latest retained implementation for each current analytical function; historical versions are not duplicated in the executable tree.
