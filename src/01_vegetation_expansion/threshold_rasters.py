# -*- coding: utf-8 -*-
"""Threshold and reproject annual vegetation-index rasters.

This is the reusable public form of the historical ``TIF_yuzhi.py`` workflow.
Scientific operations are unchanged:

    value <= threshold -> 1  (below-threshold / non-vegetated)
    value >  threshold -> 0  (vegetated)
    NoData             -> configured NoData value

The binary raster is then reprojected with nearest-neighbour resampling to the
configured target CRS and resolution. Product/threshold differences belong in
YAML configuration files under ``configs/vegetation`` rather than in duplicate
copies of this script.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if "threshold" not in config:
        raise KeyError(f"Missing 'threshold' section in {path}")
    return config


def make_output_path(input_path: str | Path, output_folder: str | Path) -> Path:
    """Add ``_measure`` suffix before ``.tif``."""
    stem = Path(input_path).stem
    if not stem.lower().endswith("_measure"):
        stem = stem + "_measure"
    return Path(output_folder) / f"{stem}.tif"


def threshold_binary(
    src_array: np.ndarray,
    src_nodata,
    threshold: float,
    scale_factor: float,
    nodata_out: int,
) -> np.ndarray:
    """Convert a continuous vegetation-index raster to the locked binary code."""
    arr = src_array.astype("float32") * scale_factor

    if src_nodata is not None:
        valid = src_array != src_nodata
    else:
        valid = np.isfinite(arr)

    out = np.full(src_array.shape, nodata_out, dtype=np.uint8)
    out[valid & (arr <= threshold)] = 1
    out[valid & (arr > threshold)] = 0
    return out


def process_one_tif(
    input_tif: str | Path,
    output_tif: str | Path,
    *,
    threshold: float,
    scale_factor: float,
    dst_crs: str,
    dst_resolution: float,
    nodata_out: int,
) -> None:
    """Threshold first, then reproject the binary raster."""
    with rasterio.open(input_tif) as src:
        if src.crs is None:
            raise ValueError(f"Input raster has no CRS: {input_tif}")

        src_array = src.read(1)
        binary = threshold_binary(
            src_array=src_array,
            src_nodata=src.nodata,
            threshold=threshold,
            scale_factor=scale_factor,
            nodata_out=nodata_out,
        )

        transform, width, height = calculate_default_transform(
            src_crs=src.crs,
            dst_crs=dst_crs,
            width=src.width,
            height=src.height,
            left=src.bounds.left,
            bottom=src.bounds.bottom,
            right=src.bounds.right,
            top=src.bounds.top,
            resolution=dst_resolution,
        )

        dst_array = np.full((height, width), nodata_out, dtype=np.uint8)
        reproject(
            source=binary,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata_out,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=nodata_out,
            resampling=Resampling.nearest,
        )

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": dst_crs,
            "transform": transform,
            "nodata": nodata_out,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(dst_array, 1)


def run(config: Dict[str, Any], check_only: bool = False) -> None:
    cfg = config["threshold"]
    enabled = bool(cfg.get("enabled", True))

    if not enabled:
        note = cfg.get("note", "Thresholding is disabled for this configuration.")
        print(f"Thresholding disabled: {note}")
        return

    required = [
        "input_folder",
        "output_folder",
        "threshold",
        "scale_factor",
        "dst_crs",
        "dst_resolution_m",
        "nodata_out",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing threshold configuration keys: {missing}")

    input_folder = _resolve_repo_path(cfg["input_folder"])
    output_folder = _resolve_repo_path(cfg["output_folder"])
    tif_pattern = str(cfg.get("tif_pattern", "*.tif"))
    threshold = float(cfg["threshold"])
    scale_factor = float(cfg["scale_factor"])
    dst_crs = str(cfg["dst_crs"])
    dst_resolution = float(cfg["dst_resolution_m"])
    nodata_out = int(cfg["nodata_out"])
    overwrite = bool(cfg.get("overwrite", False))

    print("=" * 80)
    print(f"Configuration : {config.get('name', '<unnamed>')}")
    print(f"Input folder  : {input_folder}")
    print(f"Output folder : {output_folder}")
    print(f"Threshold     : value <= {threshold} -> 1; value > {threshold} -> 0")
    print(f"Scale factor  : {scale_factor}")
    print(f"Target CRS    : {dst_crs}")
    print(f"Resolution    : {dst_resolution:g} m")
    print("=" * 80)

    if check_only:
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    tif_list = sorted(glob.glob(str(input_folder / tif_pattern)))
    if not tif_list:
        raise FileNotFoundError(f"No tif files found: {input_folder / tif_pattern}")

    for input_tif in tqdm(tif_list, desc="Processing"):
        output_tif = make_output_path(input_tif, output_folder)
        if output_tif.exists() and not overwrite:
            continue
        process_one_tif(
            input_tif,
            output_tif,
            threshold=threshold,
            scale_factor=scale_factor,
            dst_crs=dst_crs,
            dst_resolution=dst_resolution,
            nodata_out=nodata_out,
        )

    print(f"Done. Outputs saved to: {output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold and reproject vegetation-index rasters using a YAML configuration."
    )
    parser.add_argument("--config", required=True, help="Path to a vegetation YAML configuration.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and display the configuration without reading rasters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), check_only=args.check_config)


if __name__ == "__main__":
    main()
