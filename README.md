# Global Dryland Vegetation Expansion

Code and selected processed data supporting the analysis of global dryland vegetation-cover expansion during 2000–2022.

## Repository scope

This repository is organized to support transparent reuse of the final analysis workflow rather than to preserve every historical working file. For each scientific function, only the latest retained public script is kept in the code tree. Historical filenames and superseded versions are recorded only where needed for provenance.

The workflow covers:

1. vegetation-index preprocessing and vegetation-cover expansion detection;
2. land-cover process-context classification;
3. wettest-90-day hydroclimatic alignment;
4. SEM-based hydroclimatic pathway diagnostics;
5. panel fixed-effects attribution with an XCO2-indexed shared-background component;
6. aridity and process-context stratification;
7. agricultural-neighbourhood analyses; and
8. robustness, validation and figure reproduction.

## Repository structure

```text
code/
  00_preprocessing/               Raster preprocessing utilities
  01_vegetation_expansion/        Vegetation-cover fraction and trend detection
  02_landcover_context/           Land-cover process-context preparation
  03_hydroclimate/                W90 precipitation/startDOY and aligned ERA5-Land extraction
  04_sem/                         Hydroclimatic SEM diagnostics
  05_attribution/                 Main panel fixed-effects attribution
  06_stratification/              Aridity and process-context stratification
  07_agricultural_neighbourhood/  Agricultural-neighbourhood analysis
  08_robustness/                  Alternative formulations, VOD and pixel-level checks
  09_figures/                     Main and supplementary figure scripts

data/
  README.md                       Data-release boundary and directory guide
  processed/                      Selected processed inputs needed for reproduction
  sample/                         Optional lightweight examples

docs/
  data_inventory.csv              Dataset-level provenance and release status
  code_provenance.csv             One-row-per-retained-script provenance table
  workflow.md                     Recommended execution order and reproducibility boundary
  preprocessing_reproducibility.md

CODE_MAP.md                        Method-to-code map
DATA_SOURCES.md                    External dataset provenance
requirements.txt                   Python dependencies
environment.yml                    Conda environment specification
```

Generated figures and analysis outputs are **not versioned in the repository**. Scripts write them under `outputs/`, which is excluded by `.gitignore`.

## Reproducibility boundary

Most downstream statistical and figure workflows are scripted. Two preprocessing boundaries are intentionally documented rather than reconstructed from assumptions:

- selected vegetation-index TIFF thresholding and EPSG:8857 reprojection were performed in ArcGIS Pro;
- some raster/netCDF-to-100-km²-hexagon table conversion scripts are no longer available.

For those branches, selected processed hexagon-level inputs under `data/processed/` are the reproducibility starting point. See `docs/preprocessing_reproducibility.md`.

## Environment

The Python environment can be created with Conda:

```bash
conda env create -f environment.yml
conda activate global-dryland-vegetation-expansion
```

or with pip:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Google Earth Engine JavaScript files under `code/**/gee/` are intended for the Earth Engine Code Editor and are not executed by the local Python environment.

## Running the workflow

Run Python scripts from the repository root unless a script explicitly documents a different execution context. The recommended order and required inputs are listed in `CODE_MAP.md` and `docs/workflow.md`.

## Data sources

Source datasets, product identities and known provenance limits are summarized in `DATA_SOURCES.md`. Raw global remote-sensing and reanalysis archives are not redistributed here; only selected processed inputs needed to bridge unrecovered preprocessing stages are retained.
