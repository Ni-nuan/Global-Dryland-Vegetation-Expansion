# Data layout

The repository includes selected processed inputs that are practical and useful for downstream reproduction. Large raw and intermediate raster/reanalysis files are not versioned.

Recommended local layout:

```text
data/
├── README.md
├── processed/            selected versioned processed inputs
├── sample/               lightweight examples/documentation
└── local/                unversioned local inputs used by YAML configs
    ├── drylands/
    └── vegetation/
```

`data/local/` is ignored by Git. The vegetation configuration files resolve these paths relative to the repository root. Users may either reproduce the raster preprocessing or place equivalent local inputs at the configured paths.

See `docs/data_inventory.csv` for the role and provenance of each processed dataset and `DATA_SOURCES.md` for upstream public products.
