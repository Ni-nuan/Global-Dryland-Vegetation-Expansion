# Data directory

This repository does not redistribute the complete raw global remote-sensing or reanalysis archives. `data/processed/` contains only selected processed inputs that are useful for reproducing downstream analyses, especially where the original raster/netCDF-to-hexagon conversion code is no longer available.

Current released processed inputs include:

- `processed/background/up/` — CAMS XCO2 and NOAA CarbonTracker background tables;
- `processed/process_context/` — merged UP process-context, attribution and hydroclimate table;
- `processed/agricultural_neighbourhood/` — per-hex agricultural-neighbourhood analysis table and spatial-audit geometry;
- `processed/vegetation/` — UP hexagon geometry used by the neighbourhood workflow.

Generated outputs belong under `outputs/` and are not versioned. Large external/raw datasets should be kept locally under user-defined storage or under `data/external/` if the script expects a repository-relative path.

See `../DATA_SOURCES.md` and `../docs/data_inventory.csv` for provenance.
