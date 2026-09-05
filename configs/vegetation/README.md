# Vegetation configuration files

The vegetation workflow uses **one threshold/reprojection engine** and **one hexagon-trend engine**. Product, threshold, and spatial-scale differences are expressed only through these YAML files.

Run from the repository root:

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml
```

To inspect a configuration without reading data:

```bash
python src/01_vegetation_expansion/threshold_rasters.py --config configs/vegetation/ndvi_main.yaml --check-config
python src/01_vegetation_expansion/compute_hex_trends.py --config configs/vegetation/ndvi_main.yaml --check-config
```

Configuration roles:

- `ndvi_main.yaml`: NDVI > 0.20, 100 km² main analysis.
- `ndvi_threshold_016.yaml`, `018`, `022`, `024`: threshold sensitivity at 100 km².
- `ndvi_hex_75km2.yaml`, `100km2`, `125km2`: spatial aggregation sensitivity using the NDVI > 0.20 binary masks.
- `msavi.yaml`: recovered historical MSAVI threshold 0.14 at 100 km².
- `evi.yaml`: starts from existing thresholded EVI rasters because the executed EVI threshold value is not recoverable from the current code archive; no threshold is invented in this repository.

Relative paths resolve from the repository root. Large raster inputs are expected under `data/local/`, which is excluded from version control.
