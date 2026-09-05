This code-related work is currently in the submission stage and can not be used casually before it is officially accepted.

# Global Dryland Vegetation Expansion

Code, configuration files, and selected processed data supporting the analysis of global dryland vegetation-cover expansion during 2000–2022.

## Repository design

The public repository follows a **single-engine + configuration** rule wherever the calculation is identical across experiments.

The clearest example is vegetation-expansion detection:

```text
src/01_vegetation_expansion/
├── threshold_rasters.py
├── compute_hex_trends.py
├── calculate_three_year_endpoint_change.py
├── diagnostics/
└── gee/

configs/vegetation/
├── ndvi_main.yaml
├── ndvi_threshold_016.yaml
├── ndvi_threshold_018.yaml
├── ndvi_threshold_022.yaml
├── ndvi_threshold_024.yaml
├── evi.yaml
├── msavi.yaml
├── ndvi_hex_75km2.yaml
├── ndvi_hex_100km2.yaml
└── ndvi_hex_125km2.yaml
```

`threshold_rasters.py` and `compute_hex_trends.py` are the reusable engines. Threshold, vegetation product, input/output paths, and hexagon area are specified in YAML rather than by duplicating scripts.

## Quick start: vegetation workflow

Run from the repository root.

Check a configuration:

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml --check-config
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml --check-config
```

Run the main NDVI workflow once local raster inputs are available:

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml
```

Change only the configuration file to run threshold or spatial-scale sensitivity analyses.

## Repository structure

```text
src/                    reusable analysis and preprocessing code
configs/                experiment/product configuration files
figures/                main and supplementary figure-generation scripts
data/
  README.md              data layout and reproduction boundary
  processed/             selected processed inputs required downstream
  sample/                lightweight examples/readme material
docs/
  workflow.md            execution sequence
  preprocessing_reproducibility.md
CODE_MAP.md              human-readable “refactored code name : purpose” map
DATA_SOURCES.md           public source datasets and provenance
requirements.txt
environment.yml
```

Generated figures and analysis outputs are intentionally excluded from version control.

## Scientific method lock

Repository restructuring does not redefine the analysis. In particular, the vegetation engine preserves the axial hexagon geometry, binary coding (`1 = below-threshold/non-vegetated`, `0 = vegetated`), `veg_frac = (count - sum) / count`, `all_touched=False`, minimum three valid annual observations, tie-corrected Mann–Kendall test, and Sen's slope. Model sample definitions, residualization, standardization order, fixed-effects specifications, attribution decomposition, and plotted statistics are likewise not altered by repository cleanup.

## Reproducibility boundary

The recovered threshold/reprojection script reproduces the locked operation used in the raster workflow, but some production TIFF processing was executed in ArcGIS Pro. The original raster/netCDF-to-100-km²-hex table extraction code for several environmental datasets is no longer available. Those gaps are documented explicitly in `docs/preprocessing_reproducibility.md`; downstream analyses use selected processed hexagon-level inputs where necessary rather than inventing missing code.

## Code map

See [CODE_MAP.md](CODE_MAP.md) for the complete **refactored code name : purpose** list and [docs/code_provenance.csv](docs/code_provenance.csv) for original filenames and SHA-256 lineage.
