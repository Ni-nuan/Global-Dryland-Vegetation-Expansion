# Refactored code map

This is the public **refactored code name : purpose** list. Only the current retained implementation of each analytical role is exposed in the repository; historical filenames remain only in `docs/code_provenance.csv`.

## Reusable vegetation engines

- `src/01_vegetation_expansion/threshold_rasters.py` : shared threshold + EPSG:8857 reprojection engine; NDVI threshold sensitivity and the recovered MSAVI branch are selected by YAML configuration.
- `src/01_vegetation_expansion/compute_hex_trends.py` : shared axial-hex vegetated-fraction + Mann–Kendall/Sen trend engine; the main analysis, threshold sensitivity, vegetation-index sensitivity, and 75/100/125 km² scale sensitivity reuse this same code.

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml
```

## `src/00_preprocessing/`

- `src/00_preprocessing/mosaic_geotiffs.py` : GeoTIFF mosaic / dtype harmonisation.

## `src/01_vegetation_expansion/`

- `src/01_vegetation_expansion/calculate_three_year_endpoint_change.py` : Three-year-window endpoint-change calculation for UP and sensitivity products.
- `src/01_vegetation_expansion/compute_hex_trends.py` : Hex-grid annual vegetated fraction + MK/Sen trend workflow using axial hexagons.
- `src/01_vegetation_expansion/diagnostics/check_hex_grid_vegetation_fraction.py` : Single-year axial hex vegetation-fraction extraction and map.
- `src/01_vegetation_expansion/gee/export_annual_max_ndvi_partition.js` : Google Earth Engine export of annual maximum NDVI for one dryland partition after merging MOD13A1 Terra and MYD13A1 Aqua collections.
- `src/01_vegetation_expansion/threshold_rasters.py` : Threshold vegetation-index rasters and reproject to EPSG:8857.

## `src/02_landcover_context/`

- `src/02_landcover_context/build_process_context_table.py` : Build the UP process-context table by combining land-cover transition classes, attribution outputs and hydroclimatic driver trends.

## `src/03_hydroclimate/`

- `src/03_hydroclimate/gee/compute_w90_precipitation_startdoy.js` : Google Earth Engine computation of annual per-pixel wettest-90d precipitation total (P90) and 1-based startDOY using prefix sums, exported by global tiles.
- `src/03_hydroclimate/gee/export_w90_era5land_drivers.js` : Google Earth Engine export of wettest-90d-aligned ERA5-Land temperature, VPD, soil-moisture, ET and net-shortwave drivers from precomputed per-pixel startDOY assets.

## `src/04_sem/`

- `src/04_sem/fit_sem_hydroclimate_structure.py` : SEM mechanism-validation branch with VPD_resid.
- `src/04_sem/run_sem_structure_sensitivity.py` : SEM sensitivity/toggle branch with explicit Tmean and SWnet-direct controls.

## `src/05_attribution/`

- `src/05_attribution/diagnostics/run_temporal_identifiability_diagnostics.py` : Temporal identifiability diagnostics with VPD_resid, year FE, CO2 projection and linear-time replacement.
- `src/05_attribution/diagnostics/run_vif_diagnostics.py` : VIF diagnostics for pooled and entity-demeaned panel predictors.
- `src/05_attribution/run_panel_attribution.py` : Hex-FE logit trend-budget attribution using VPD_resid and XCO2 panel.

## `src/06_stratification/`

- `src/06_stratification/summarize_aridity_driver_stratification.py` : UP aridity x hydroclimatic-driver quantile stratification and screening.
- `src/06_stratification/summarize_process_context_driver_quintiles.py` : Process-context x driver-trend quantile summaries.

## `src/07_agricultural_neighbourhood/`

- `src/07_agricultural_neighbourhood/analyze_agricultural_neighbourhoods.py` : Agricultural-neighbourhood analysis and supplementary table generation.

## `src/08_robustness/`

- `src/08_robustness/attribution/run_raw_vpd_sensitivity.py` : Hex-FE logit trend-budget attribution using raw VPD90 and XCO2 panel.
- `src/08_robustness/gee/export_annual_max_evi_partition.js` : Google Earth Engine export of annual maximum EVI for one dryland partition after merging MOD13A1 Terra and MYD13A1 Aqua collections.
- `src/08_robustness/gee/export_annual_max_msavi_partition.js` : Google Earth Engine export of annual maximum MSAVI for one dryland partition from merged MODIS Terra and Aqua red/NIR reflectance with SummaryQA masking.
- `src/08_robustness/greenness_comparison/compare_pixel_hex_trends.py` : Pixel-vs-hex trend correspondence / overlap statistics.
- `src/08_robustness/greenness_comparison/compute_pixel_ndvi_trends.py` : Pixel-level NDVI MK/Sen trends within drylands.
- `src/08_robustness/vod/download_vod_data.py` : Download VOD source files from Zenodo record 17359730.
- `src/08_robustness/vod/prepare_merged_cband_vod.py` : Prepare merged C-band VOD annual-maximum GeoTIFFs.
- `src/08_robustness/vod/prepare_vodca_cxku.py` : Prepare VODCA CXKu annual-maximum GeoTIFFs.
- `src/08_robustness/vod/validate_vod_trends.py` : VOD trend validation in fixed NDVI-derived UP/DOWN samples.

## `figures/main/`

- `figures/main/plot_fig1a_global_expansion_map.py` : Categorical global expansion/contraction map with regional zooms.
- `figures/main/plot_fig1b_endpoint_change.py` : Fig. 1b 3-year-window endpoint-change calculation and violin/box visualization.
- `figures/main/plot_fig1c_up_trajectory.py` : Figure 1c annual median UP vegetated-fraction trajectory with IQR and fitted linear trend.
- `figures/main/plot_fig1d_landcover_gross_flows.py` : Figure 1d gross inflow/outflow summary by 10 aggregated land-cover classes in UP hexagons.
- `figures/main/plot_fig1e_transition_pathway_concentration.py` : Figure 1e ranked off-diagonal land-cover transition pathways and cumulative concentration in UP hexagons.
- `figures/main/plot_fig2_natural_share_distribution.py` : Distribution of per-hex natural-share values for CLIM_ONLY and CLIM_PLUS_XCO2 under NAT0.
- `figures/main/plot_fig2_three_component_budget.py` : Three-component absolute trend-budget summary: climate, incremental XCO2-indexed component, and residual.
- `figures/main/plot_fig3_aridity_climate_share.py` : Main-text compact climate-component share responses across window, state and gate gradients by aridity class.
- `figures/main/plot_fig3_aridity_residual_share_curves.py` : Refined aridity-class residual-component share curves across within-aridity quintiles of P90, SM90_L1 and VPD_resid.
- `figures/main/plot_fig3_aridity_xco2_background_share.py` : Main-text compact XCO2-indexed background share responses across window, state and gate gradients by aridity class.
- `figures/main/plot_fig4a_process_context_summary.py` : Four-panel process-context summary of climate share, observed trend, natural share and residual trend.
- `figures/main/plot_fig4b_agricultural_neighbourhood.py` : Four-panel agricultural-neighbourhood comparison of observed trend, endpoint change, natural share and natural-dominant rate.

## `figures/supplementary/`

- `figures/supplementary/plot_agricultural_neighbourhood_local_excess_map.py` : Refined agricultural-neighbourhood spatial maps with transparent export, centroid-assisted global grouping, local-excess map, and regional native panels.
- `figures/supplementary/plot_agricultural_neighbourhood_spatial_audit.py` : Topology-safe agricultural-neighbourhood spatial audit with native global polygons, reproducible aggregated display grid, and native regional detail panels.
- `figures/supplementary/plot_aridity_p90_climate_share_facets.py` : Aridity-specific climate-component share facets across within-aridity P90 trend quintiles.
- `figures/supplementary/plot_aridity_sm90_climate_share_facets.py` : Aridity-specific climate-component share facets across within-aridity SM90_L1 trend quintiles.
- `figures/supplementary/plot_aridity_state_climate_q5_minus_q1.py` : Within-aridity SM90 moisture-state end-member contrast in climate-component share.
- `figures/supplementary/plot_aridity_vpdresid_climate_share_facets.py` : Aridity-specific climate-component share facets across within-aridity VPD_resid trend quintiles.
- `figures/supplementary/plot_aridity_window_gate_climate_q5_minus_q1.py` : Within-aridity P90 window and VPD_resid gate end-member contrasts in climate-component share.
- `figures/supplementary/plot_aridity_window_gate_residual_q5_minus_q1.py` : Within-aridity P90 window and VPD_resid gate end-member contrasts in residual-component share.
- `figures/supplementary/plot_aridity_window_gate_xco2_q5_minus_q1.py` : Within-aridity P90 window and VPD_resid gate end-member contrasts in XCO2-indexed component share.
- `figures/supplementary/plot_aridity_window_state_gate_climate_heatmap.py` : Aridity-by-trend-group heatmaps of median climate-component share for P90, SM90_L1 and VPD_resid.
- `figures/supplementary/plot_landcover_transition_alluvial.py` : Alluvial visualization of all diagonal persistence flows plus top-10 off-diagonal land-cover transitions in UP hexagons.
- `figures/supplementary/plot_probability_scale_dominance_map.py` : Spatial map of natural-dominant versus residual-dominant attribution classes.
- `figures/supplementary/plot_probability_scale_residual_map.py` : Spatial map of the residual trend component beta_res_co2.
- `figures/supplementary/plot_process_group_trajectories.py` : Annual mean vegetation-fraction trajectories by land-cover process group within UP hexagons.
- `figures/supplementary/plot_process_residual_minus_climate_trimmed_mean.py` : Six-process summary of residual-minus-climate component share using 10% trimmed mean, with median and n diagnostics.
- `figures/supplementary/plot_process_window_gate_cell_counts.py` : Process-context 5x5 P90-by-VPD_resid cell-count heatmaps supporting minimum-cell masking.
- `figures/supplementary/plot_process_window_gate_climate_q5_minus_q1.py` : Process-context end-member contrasts in climate-component share along globally defined P90 and VPD_resid trend quintiles.
- `figures/supplementary/plot_process_window_gate_residual_q5_minus_q1.py` : Process-context end-member contrasts in residual-component share along globally defined P90 and VPD_resid trend quintiles.
- `figures/supplementary/plot_process_window_gate_xco2_q5_minus_q1.py` : Process-context end-member contrasts in XCO2-indexed component share along globally defined P90 and VPD_resid trend quintiles.
- `figures/supplementary/plot_response_scale_sensitivity.py` : Bar comparison of natural share and natural-dominant fraction between logit- and fraction-scale attribution summaries.
- `figures/supplementary/plot_s16_pixel_vs_hex_comparison.py` : Publication map for pixel-level NDVI greening versus threshold-defined hexagon expansion and overlap relationship.

## `configs/vegetation/`

- `ndvi_main.yaml` : NDVI > 0.20, 100 km² main configuration.
- `ndvi_threshold_016.yaml` : NDVI threshold 0.16 sensitivity.
- `ndvi_threshold_018.yaml` : NDVI threshold 0.18 sensitivity.
- `ndvi_threshold_022.yaml` : NDVI threshold 0.22 sensitivity.
- `ndvi_threshold_024.yaml` : NDVI threshold 0.24 sensitivity.
- `ndvi_hex_75km2.yaml` : 75 km² aggregation sensitivity using the main NDVI binary masks.
- `ndvi_hex_100km2.yaml` : explicit 100 km² aggregation configuration.
- `ndvi_hex_125km2.yaml` : 125 km² aggregation sensitivity using the main NDVI binary masks.
- `msavi.yaml` : recovered MSAVI threshold 0.14 configuration.
- `evi.yaml` : EVI trend configuration starting from existing thresholded EVI rasters; the unrecovered executed threshold is not guessed.
