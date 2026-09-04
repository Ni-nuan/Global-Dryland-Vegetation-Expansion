#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conventional pixel-level greenness trends versus threshold-defined vegetation-
cover expansion, using the same global-map visual system as Figure 1a.

The script creates two publication-oriented figures:

1. Pixel-versus-hex comparison
   a, conventional pixel-level MODIS NDVI trend
   b, threshold-defined hexagon expansion/contraction

2. Pixel-hex relationship map
   Shows where the two approaches agree, where only one detects change, and
   where their directions oppose each other.

Global map style is harmonised with Figure 1a:
- brown = negative vegetation change
- green = positive vegetation change
- pale neutral dryland background
- white figure background
- identical Kendall's tau range for directly comparable maps

Dependencies:
    rasterio, geopandas, shapely, pyproj, numpy, pandas, matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
)
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch

from pyproj import Transformer

from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


# =============================================================================
# 1. Paths and outputs
# =============================================================================

PIXEL_TIF = Path(
    r"MODIS_NDVI_pixel_MK_Sen_8857.tif"
)

OVERLAP_TIF = Path(
    r"MODIS_pixel_hex_overlap_8857.tif"
)

LAND_SHP = Path(
    r"world_map/ne_10m_land.shp"
)

DRY_SHP: Optional[Path] = Path(
    r"Drylands_latest_July2014/drylands_UNCCD_CBD_july2014.shp"
)

OUTPUT_DIR = Path(
    r"pixel_hex_greening_comparison_maps"
)

COMPARISON_STEM = (
    "pixel_vs_threshold_expansion_comparison"
)

RELATIONSHIP_STEM = (
    "pixel_hex_overlap_relationship"
)

EXPORT_PNG = True
EXPORT_PDF = True
EXPORT_SVG = True

# White background for stable rendering and consistency with revised Fig. 1a.
EXPORT_TRANSPARENT = False

DPI = 600


# =============================================================================
# 2. User band settings
# =============================================================================

# Leave as None for automatic detection from raster band descriptions.
# Rasterio band indices start at 1.
PIXEL_TAU_BAND: Optional[int] = None
PIXEL_P_BAND: Optional[int] = None
PIXEL_SLOPE_BAND: Optional[int] = None
PIXEL_CLASS_BAND: Optional[int] = None

# Preferred display variable for the conventional trend map:
# "tau"   = Kendall's tau for significant pixels
# "slope" = Sen slope for significant pixels
# "class" = discrete significant decrease / non-significant / increase
PIXEL_DISPLAY_MODE = "tau"

PIXEL_SIGNIFICANCE_LEVEL = 0.05


# -----------------------------------------------------------------------------
# IMPORTANT: shared Kendall's tau colour range
# -----------------------------------------------------------------------------
#
# This MUST equal the final frozen absolute tau limit used in Fig. 1a.
#
# Example only:
#   if Fig. 1a uses -0.45 to +0.45, retain 0.45 below.
#   if Fig. 1a uses -0.40 to +0.40, change this to 0.40.
#
# Do not estimate the range independently for Supplementary Fig. S16.
#
TAU_ABS_MAX = 0.45


# If PIXEL_CLASS_BAND is used and values are not a standard encoding, define:
# raw raster value -> {-1: DOWN, 0: non-significant, 1: UP}
PIXEL_CLASS_VALUE_MAP: Optional[
    Dict[int, int]
] = None


# Overlap raster settings
OVERLAP_COMBINED_BAND: Optional[int] = 1

OVERLAP_PIXEL_CLASS_BAND: Optional[
    int
] = None

OVERLAP_HEX_CLASS_BAND: Optional[
    int
] = None


# If the combined overlap codes do not follow an automatically recognised
# encoding, provide a manual map:
#
# raw code -> (pixel_class, hex_class)
#
# class values:
# -1 = DOWN
#  0 = non-significant / Hex_other
#  1 = UP
#
OVERLAP_CODE_MAP: Optional[
    Dict[int, Tuple[int, int]]
] = None


# =============================================================================
# 3. Projection and rendering resolution
# =============================================================================

PROJ = "ESRI:54030"  # Robinson

LON_MIN = -180.0
LON_MAX = 180.0

LAT_MIN = -58.0
LAT_MAX = 85.0


# Global display raster.
TARGET_WIDTH = 3600
TARGET_HEIGHT = 1700


# Nearest-neighbour is retained so significance/class boundaries
# are not spatially blended.
RASTER_RESAMPLING = Resampling.nearest


# =============================================================================
# 4. Journal-style map design
# =============================================================================

FONT_FAMILY = "Arial"


# -----------------------------------------------------------------------------
# Background: matched to revised Fig. 1a
# -----------------------------------------------------------------------------

OCEAN_COLOR = "#FFFFFF"

LAND_COLOR = "#F7F7F7"

DRY_COLOR = "#E9E8E1"

COAST_COLOR = "#666666"

FRAME_COLOR = "#666666"

TEXT_COLOR = "#222222"


# -----------------------------------------------------------------------------
# Shared brown -> cream -> green trend system
# -----------------------------------------------------------------------------

NEGATIVE_DARK = "#7F3B08"

NEGATIVE_MID = "#B7783D"

NEGATIVE_LIGHT = "#DFC39A"

ZERO_COLOR = "#F5F3EA"

POSITIVE_LIGHT = "#C8E0BE"

POSITIVE_MID = "#79B879"

POSITIVE_DARK = "#238443"


TREND_CMAP = LinearSegmentedColormap.from_list(
    "dryland_brown_green",
    [
        NEGATIVE_DARK,
        NEGATIVE_MID,
        NEGATIVE_LIGHT,
        ZERO_COLOR,
        POSITIVE_LIGHT,
        POSITIVE_MID,
        POSITIVE_DARK,
    ],
    N=256,
)


# -----------------------------------------------------------------------------
# Pixel-hex relationship colours
# -----------------------------------------------------------------------------
#
# Semantic consistency:
#
# green family = greening / expansion
# brown family = browning / contraction
# charcoal     = genuinely opposing directions
#
RELATIONSHIP_COLORS = {

    # Pixel UP + Hex UP
    1: "#238443",

    # Hex UP only
    2: "#A9D39E",

    # Pixel UP only
    3: "#68AD70",

    # Pixel DOWN + Hex DOWN
    4: "#7F3B08",

    # Hex DOWN only
    5: "#DFC39A",

    # Pixel DOWN only
    6: "#B7783D",

    # Opposing directions
    7: "#686868",
}


RELATIONSHIP_LABELS = {

    1: "Positive agreement",

    2: "Hex expansion only",

    3: "Pixel greening only",

    4: "Negative agreement",

    5: "Hex contraction only",

    6: "Pixel browning only",

    7: "Opposing directions",
}


LAND_SIMPLIFY = 4500
DRY_SIMPLIFY = 4500


# =============================================================================
# 5. Figure layout
# =============================================================================

# Comparison figure
COMPARISON_FIGSIZE = (
    12.0,
    4.85
)

COMPARISON_LEFT_AX = [
    0.015,
    0.245,
    0.475,
    0.695,
]

COMPARISON_RIGHT_AX = [
    0.510,
    0.245,
    0.475,
    0.695,
]

LEFT_CBAR_AX = [
    0.105,
    0.105,
    0.295,
    0.035,
]

RIGHT_LEGEND_AX = [
    0.565,
    0.075,
    0.365,
    0.085,
]


# Relationship figure
RELATIONSHIP_FIGSIZE = (
    10.0,
    4.85
)

RELATIONSHIP_MAP_AX = [
    0.02,
    0.245,
    0.96,
    0.71,
]

RELATIONSHIP_LEGEND_AX = [
    0.09,
    0.035,
    0.82,
    0.135,
]


# =============================================================================
# 6. Generic helpers
# =============================================================================

def configure_matplotlib() -> None:

    plt.rcParams.update({

        "font.family": FONT_FAMILY,

        "font.sans-serif": [
            FONT_FAMILY
        ],

        "font.size": 8.0,

        "axes.linewidth": 0.6,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "svg.fonttype": "none",

        "figure.facecolor": "white",

        "axes.facecolor": "white",

        "savefig.facecolor": "white",

        "text.antialiased": True,

        "lines.antialiased": True,
    })


def ensure_inputs() -> None:

    required = [
        PIXEL_TIF,
        OVERLAP_TIF,
        LAND_SHP,
    ]

    if DRY_SHP is not None:
        required.append(
            DRY_SHP
        )

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Required files were not found:\n"
            + "\n".join(missing)
        )


def save_figure(
    fig: plt.Figure,
    stem: str
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    kwargs = {

        "transparent":
            EXPORT_TRANSPARENT,

        "facecolor":
            "none"
            if EXPORT_TRANSPARENT
            else "white",

        "bbox_inches":
            "tight",

        "pad_inches":
            0.02,
    }

    if EXPORT_PNG:

        fig.savefig(
            OUTPUT_DIR / f"{stem}.png",
            dpi=DPI,
            **kwargs
        )

    if EXPORT_PDF:

        fig.savefig(
            OUTPUT_DIR / f"{stem}.pdf",
            **kwargs
        )

    if EXPORT_SVG:

        fig.savefig(
            OUTPUT_DIR / f"{stem}.svg",
            **kwargs
        )


def safe_geodataframe(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    if gdf.crs is None:

        gdf = gdf.set_crs(
            "EPSG:4326",
            allow_override=True
        )

    gdf = gdf.loc[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


def subset_to_display_bbox(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:

    geo = (
        safe_geodataframe(gdf)
        .to_crs("EPSG:4326")
    )

    subset = geo.cx[
        LON_MIN:LON_MAX,
        LAT_MIN:LAT_MAX
    ].copy()

    return subset.loc[
        subset.geometry.notna()
        & ~subset.geometry.is_empty
    ].copy()


def simplify_geometry(
    gdf: Optional[gpd.GeoDataFrame],
    tolerance: float
) -> Optional[gpd.GeoDataFrame]:

    if gdf is None or tolerance <= 0:
        return gdf

    out = gdf.copy()

    valid = out.geometry.is_valid

    if valid.any():

        out.loc[
            valid,
            "geometry"
        ] = (
            out.loc[
                valid
            ]
            .geometry
            .simplify(
                tolerance,
                preserve_topology=True
            )
        )

    return out.loc[
        out.geometry.notna()
        & ~out.geometry.is_empty
    ].copy()


def make_projection_boundary_patch(
    ax: plt.Axes,
    n: int = 500,
    facecolor: str = OCEAN_COLOR,
    edgecolor: str = FRAME_COLOR,
    linewidth: float = 0.55,
    zorder: int = 0,
) -> PathPatch:

    transformer = (
        Transformer.from_crs(
            "EPSG:4326",
            PROJ,
            always_xy=True
        )
    )

    lon_bottom = np.linspace(
        LON_MIN,
        LON_MAX,
        n
    )

    lat_bottom = np.full(
        n,
        LAT_MIN
    )

    lat_right = np.linspace(
        LAT_MIN,
        LAT_MAX,
        n
    )

    lon_right = np.full(
        n,
        LON_MAX
    )

    lon_top = np.linspace(
        LON_MAX,
        LON_MIN,
        n
    )

    lat_top = np.full(
        n,
        LAT_MAX
    )

    lat_left = np.linspace(
        LAT_MAX,
        LAT_MIN,
        n
    )

    lon_left = np.full(
        n,
        LON_MIN
    )

    lon = np.concatenate([
        lon_bottom,
        lon_right,
        lon_top,
        lon_left,
    ])

    lat = np.concatenate([
        lat_bottom,
        lat_right,
        lat_top,
        lat_left,
    ])

    x, y = transformer.transform(
        lon,
        lat
    )

    vertices = np.column_stack([
        x,
        y
    ])

    vertices = np.vstack([
        vertices,
        vertices[0]
    ])

    codes = np.full(
        len(vertices),
        MplPath.LINETO
    )

    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY

    patch = PathPatch(

        MplPath(
            vertices,
            codes
        ),

        facecolor=facecolor,

        edgecolor=edgecolor,

        linewidth=linewidth,

        zorder=zorder,

        joinstyle="round",

        capstyle="round",
    )

    ax.add_patch(
        patch
    )

    return patch


def projection_extent_from_frame(
) -> Tuple[
    float,
    float,
    float,
    float
]:

    transformer = (
        Transformer.from_crs(
            "EPSG:4326",
            PROJ,
            always_xy=True
        )
    )

    n = 1000

    lon = np.concatenate([

        np.linspace(
            LON_MIN,
            LON_MAX,
            n
        ),

        np.full(
            n,
            LON_MAX
        ),

        np.linspace(
            LON_MAX,
            LON_MIN,
            n
        ),

        np.full(
            n,
            LON_MIN
        ),
    ])

    lat = np.concatenate([

        np.full(
            n,
            LAT_MIN
        ),

        np.linspace(
            LAT_MIN,
            LAT_MAX,
            n
        ),

        np.full(
            n,
            LAT_MAX
        ),

        np.linspace(
            LAT_MAX,
            LAT_MIN,
            n
        ),
    ])

    x, y = transformer.transform(
        lon,
        lat
    )

    return (
        float(np.nanmin(x)),
        float(np.nanmin(y)),
        float(np.nanmax(x)),
        float(np.nanmax(y)),
    )


def set_axis_to_frame(
    ax: plt.Axes,
    frame: PathPatch
) -> None:

    vertices = (
        frame
        .get_path()
        .vertices
    )

    xmin = np.nanmin(
        vertices[:, 0]
    )

    xmax = np.nanmax(
        vertices[:, 0]
    )

    ymin = np.nanmin(
        vertices[:, 1]
    )

    ymax = np.nanmax(
        vertices[:, 1]
    )

    dx = xmax - xmin

    dy = ymax - ymin

    ax.set_xlim(
        xmin - 0.008 * dx,
        xmax + 0.008 * dx
    )

    ax.set_ylim(
        ymin - 0.015 * dy,
        ymax + 0.015 * dy
    )

    ax.set_axis_off()


def draw_base_map(
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    dry: Optional[gpd.GeoDataFrame],
) -> PathPatch:

    frame = (
        make_projection_boundary_patch(
            ax
        )
    )

    land.plot(

        ax=ax,

        color=LAND_COLOR,

        edgecolor="none",

        linewidth=0,

        zorder=1
    )

    if dry is not None and not dry.empty:

        dry.plot(

            ax=ax,

            color=DRY_COLOR,

            edgecolor="none",

            linewidth=0,

            alpha=1.0,

            zorder=2,
        )

    land.boundary.plot(

        ax=ax,

        color=COAST_COLOR,

        linewidth=0.20,

        alpha=0.70,

        zorder=10,
    )

    set_axis_to_frame(
        ax,
        frame
    )

    return frame


def clip_axis_artists(
    ax: plt.Axes,
    frame: PathPatch
) -> None:

    for collection in ax.collections:

        collection.set_clip_path(
            frame
        )

    for image in ax.images:

        image.set_clip_path(
            frame
        )


# =============================================================================
# 7. Raster inspection and band detection
# =============================================================================

def normalise_description(
    text: Optional[str]
) -> str:

    return (
        ""
        if text is None
        else str(text).strip().lower()
    )


def band_metadata(
    src: rasterio.io.DatasetReader
) -> pd.DataFrame:

    rows = []

    for idx in range(
        1,
        src.count + 1
    ):

        description = (
            src.descriptions[
                idx - 1
            ]
        )

        tags = src.tags(
            idx
        )

        rows.append({

            "band":
                idx,

            "description":
                description or "",

            "tags":
                "; ".join(
                    f"{k}={v}"
                    for k, v
                    in tags.items()
                ),

            "dtype":
                src.dtypes[
                    idx - 1
                ],

            "nodata":
                src.nodatavals[
                    idx - 1
                ],
        })

    return pd.DataFrame(
        rows
    )


def find_band_by_keywords(
    src: rasterio.io.DatasetReader,
    keyword_groups: Sequence[
        Sequence[str]
    ],
) -> Optional[int]:

    for idx in range(
        1,
        src.count + 1
    ):

        text = " ".join([

            normalise_description(
                src.descriptions[
                    idx - 1
                ]
            ),

            " ".join(
                f"{k} {v}"
                for k, v
                in src.tags(idx).items()
            ).lower(),
        ])

        for keywords in keyword_groups:

            if all(
                k.lower() in text
                for k in keywords
            ):
                return idx

    return None


def detect_pixel_bands(
    src: rasterio.io.DatasetReader
) -> Dict[
    str,
    Optional[int]
]:

    tau = (
        PIXEL_TAU_BAND
        or find_band_by_keywords(
            src,
            [
                ("tau",),
                ("kendall", "tau"),
                ("mk", "tau"),
            ]
        )
    )

    pvalue = (
        PIXEL_P_BAND
        or find_band_by_keywords(
            src,
            [
                ("pvalue",),
                ("p_value",),
                ("p value",),
                ("mk", "p"),
            ]
        )
    )

    slope = (
        PIXEL_SLOPE_BAND
        or find_band_by_keywords(
            src,
            [
                ("sen", "slope"),
                ("slope",),
                ("senslope",),
            ]
        )
    )

    class_band = (
        PIXEL_CLASS_BAND
        or find_band_by_keywords(
            src,
            [
                ("trend", "class"),
                ("signif", "class"),
                ("mk", "class"),
            ]
        )
    )


    if (
        src.count >= 3
        and tau is None
        and pvalue is None
        and slope is None
    ):

        tau = 1
        pvalue = 2
        slope = 3


    if (
        src.count == 1
        and class_band is None
    ):

        class_band = 1


    return {

        "tau":
            tau,

        "pvalue":
            pvalue,

        "slope":
            slope,

        "class":
            class_band,
    }


def detect_overlap_bands(
    src: rasterio.io.DatasetReader
) -> Dict[
    str,
    Optional[int]
]:

    pixel_band = (
        OVERLAP_PIXEL_CLASS_BAND
        or find_band_by_keywords(
            src,
            [
                ("pixel", "class"),
                ("pixel", "trend"),
            ]
        )
    )

    hex_band = (
        OVERLAP_HEX_CLASS_BAND
        or find_band_by_keywords(
            src,
            [
                ("hex", "class"),
                ("hexagon", "class"),
                ("hex", "trend"),
            ]
        )
    )

    combined = (
        OVERLAP_COMBINED_BAND
    )

    if (
        pixel_band is not None
        and hex_band is not None
    ):

        combined = None

    elif (
        combined is None
        and src.count == 1
    ):

        combined = 1


    return {

        "combined":
            combined,

        "pixel":
            pixel_band,

        "hex":
            hex_band,
    }


def sample_unique_values(
    src: rasterio.io.DatasetReader,
    band: int,
    sample_size: int = 800,
) -> np.ndarray:

    arr = src.read(

        band,

        out_shape=(
            min(
                sample_size,
                src.height
            ),
            min(
                sample_size,
                src.width
            ),
        ),

        resampling=
            Resampling.nearest,

        masked=True,
    )

    vals = np.asarray(
        arr.compressed()
    )

    if vals.size == 0:

        return np.array([])

    return np.unique(
        vals
    )


# =============================================================================
# 8. Raster reprojection and classification
# =============================================================================

def reproject_band_to_display(
    src: rasterio.io.DatasetReader,
    band: int,
    dst_dtype: str = "float32",
    dst_nodata: float = np.nan,
) -> np.ndarray:

    xmin, ymin, xmax, ymax = (
        projection_extent_from_frame()
    )

    dst_transform = from_bounds(

        xmin,
        ymin,
        xmax,
        ymax,

        TARGET_WIDTH,
        TARGET_HEIGHT
    )

    dst = np.full(

        (
            TARGET_HEIGHT,
            TARGET_WIDTH
        ),

        dst_nodata,

        dtype=dst_dtype
    )

    reproject(

        source=
            rasterio.band(
                src,
                band
            ),

        destination=
            dst,

        src_transform=
            src.transform,

        src_crs=
            src.crs,

        src_nodata=
            src.nodatavals[
                band - 1
            ],

        dst_transform=
            dst_transform,

        dst_crs=
            PROJ,

        dst_nodata=
            dst_nodata,

        resampling=
            RASTER_RESAMPLING,
    )

    return dst


def infer_class_mapping(
    values: Iterable[float]
) -> Dict[
    int,
    int
]:

    unique = {
        int(round(v))
        for v in values
        if np.isfinite(v)
    }

    if unique.issubset(
        {-1, 0, 1}
    ):

        return {
            -1: -1,
             0:  0,
             1:  1
        }

    if unique.issubset(
        {1, 2, 3}
    ):

        return {
            1: -1,
            2:  0,
            3:  1
        }

    raise ValueError(

        "Unable to infer the class encoding. "
        "Unique sampled values were: "
        f"{sorted(unique)}. "
        "Set PIXEL_CLASS_VALUE_MAP or the overlap code map."
    )


def apply_class_mapping(
    raw: np.ndarray,
    mapping: Dict[
        int,
        int
    ]
) -> np.ndarray:

    out = np.full(
        raw.shape,
        0,
        dtype=np.int8
    )

    finite = np.isfinite(
        raw
    )

    rounded = (
        np.rint(
            raw[finite]
        )
        .astype(
            np.int64
        )
    )

    mapped = np.zeros(
        rounded.shape,
        dtype=np.int8
    )

    for (
        source_value,
        target_value
    ) in mapping.items():

        mapped[
            rounded
            == int(source_value)
        ] = int(
            target_value
        )

    out[
        finite
    ] = mapped

    return out


def build_pixel_trend_arrays(
    src: rasterio.io.DatasetReader,
    bands: Dict[
        str,
        Optional[int]
    ],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    str
]:

    mode = (
        PIXEL_DISPLAY_MODE
        .lower()
    )

    if (
        bands["pvalue"] is not None
        and (
            bands["tau"] is not None
            or bands["slope"] is not None
        )
    ):

        pvalue = (
            reproject_band_to_display(
                src,
                bands["pvalue"]
            )
        )

        signal_band = (
            bands["tau"]
            if bands["tau"] is not None
            else bands["slope"]
        )

        signal = (
            reproject_band_to_display(
                src,
                signal_band
            )
        )

        significant = (

            np.isfinite(pvalue)

            & np.isfinite(signal)

            & (
                pvalue
                < PIXEL_SIGNIFICANCE_LEVEL
            )
        )


        pixel_class = np.zeros(
            signal.shape,
            dtype=np.int8
        )

        pixel_class[
            significant
            & (signal > 0)
        ] = 1

        pixel_class[
            significant
            & (signal < 0)
        ] = -1


        if (
            mode == "tau"
            and bands["tau"] is not None
        ):

            values = (
                reproject_band_to_display(
                    src,
                    bands["tau"]
                )
            )

            values[
                ~significant
            ] = np.nan

            return (
                values,
                pixel_class,
                "Kendall’s τ"
            )


        if (
            mode == "slope"
            and bands["slope"] is not None
        ):

            values = (
                reproject_band_to_display(
                    src,
                    bands["slope"]
                )
            )

            values[
                ~significant
            ] = np.nan

            return (
                values,
                pixel_class,
                "Sen slope"
            )


        return (
            pixel_class.astype(float),
            pixel_class,
            "Trend class"
        )


    if bands["class"] is None:

        raise ValueError(

            "Could not identify sufficient bands for the pixel trend raster. "
            "Set PIXEL_TAU_BAND and PIXEL_P_BAND, or PIXEL_CLASS_BAND."
        )


    raw = (
        reproject_band_to_display(
            src,
            bands["class"]
        )
    )


    mapping = (
        PIXEL_CLASS_VALUE_MAP
    )


    if mapping is None:

        sample = (
            sample_unique_values(
                src,
                bands["class"]
            )
        )

        mapping = (
            infer_class_mapping(
                sample
            )
        )


    pixel_class = (
        apply_class_mapping(
            raw,
            mapping
        )
    )


    return (
        pixel_class.astype(float),
        pixel_class,
        "Trend class"
    )


def infer_overlap_code_map(
    values: Iterable[float]
) -> Dict[
    int,
    Tuple[int, int]
]:

    unique = {
        int(round(v))
        for v in values
        if np.isfinite(v)
    }

    unique.discard(
        0
    )


    if (
        unique
        and unique.issubset(
            set(
                range(
                    1,
                    10
                )
            )
        )
    ):

        mapping = {}

        classes = [
            -1,
             0,
             1
        ]

        for code in range(
            1,
            10
        ):

            pixel_index = (
                (code - 1)
                // 3
            )

            hex_index = (
                (code - 1)
                % 3
            )

            mapping[
                code
            ] = (
                classes[
                    pixel_index
                ],
                classes[
                    hex_index
                ],
            )

        return mapping


    expected = {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
    }


    if (
        unique
        and unique.issubset(
            expected
        )
    ):

        classes = {
            1: -1,
            2:  0,
            3:  1
        }

        return {

            code: (
                classes[
                    code // 10
                ],
                classes[
                    code % 10
                ]
            )

            for code
            in expected
        }


    raise ValueError(

        "Unable to decode the overlap raster automatically. "
        f"Sampled codes were: {sorted(unique)}. "
        "Define OVERLAP_CODE_MAP in the USER BAND SETTINGS section."
    )


def build_overlap_classes(
    src: rasterio.io.DatasetReader,
    bands: Dict[
        str,
        Optional[int]
    ],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[
        int,
        Tuple[int, int]
    ]
]:

    if (
        bands["pixel"] is not None
        and bands["hex"] is not None
    ):

        pixel_raw = (
            reproject_band_to_display(
                src,
                bands["pixel"]
            )
        )

        hex_raw = (
            reproject_band_to_display(
                src,
                bands["hex"]
            )
        )

        pixel_map = (
            infer_class_mapping(
                sample_unique_values(
                    src,
                    bands["pixel"]
                )
            )
        )

        hex_map = (
            infer_class_mapping(
                sample_unique_values(
                    src,
                    bands["hex"]
                )
            )
        )

        return (

            apply_class_mapping(
                pixel_raw,
                pixel_map
            ),

            apply_class_mapping(
                hex_raw,
                hex_map
            ),

            {},
        )


    if bands["combined"] is None:

        raise ValueError(
            "No usable overlap band was detected."
        )


    raw = (
        reproject_band_to_display(
            src,
            bands["combined"]
        )
    )


    code_map = (
        OVERLAP_CODE_MAP
    )


    if code_map is None:

        sampled_codes = (
            sample_unique_values(
                src,
                bands["combined"]
            )
        )

        code_map = (
            infer_overlap_code_map(
                sampled_codes
            )
        )


    pixel_class = np.zeros(
        raw.shape,
        dtype=np.int8
    )

    hex_class = np.zeros(
        raw.shape,
        dtype=np.int8
    )

    finite = np.isfinite(
        raw
    )


    rounded = np.zeros(
        raw.shape,
        dtype=np.int64
    )

    rounded[
        finite
    ] = (
        np.rint(
            raw[finite]
        )
        .astype(
            np.int64
        )
    )


    for (
        code,
        (
            pixel_value,
            hex_value
        )
    ) in code_map.items():

        mask = (
            finite
            & (
                rounded
                == int(code)
            )
        )

        pixel_class[
            mask
        ] = int(
            pixel_value
        )

        hex_class[
            mask
        ] = int(
            hex_value
        )


    return (
        pixel_class,
        hex_class,
        code_map
    )


def build_relationship_class(
    pixel_class: np.ndarray,
    hex_class: np.ndarray,
) -> np.ndarray:

    relationship = np.zeros(
        pixel_class.shape,
        dtype=np.uint8
    )


    relationship[
        (pixel_class == 1)
        & (hex_class == 1)
    ] = 1


    relationship[
        (pixel_class == 0)
        & (hex_class == 1)
    ] = 2


    relationship[
        (pixel_class == 1)
        & (hex_class == 0)
    ] = 3


    relationship[
        (pixel_class == -1)
        & (hex_class == -1)
    ] = 4


    relationship[
        (pixel_class == 0)
        & (hex_class == -1)
    ] = 5


    relationship[
        (pixel_class == -1)
        & (hex_class == 0)
    ] = 6


    relationship[
        (
            (pixel_class == 1)
            & (hex_class == -1)
        )
        |
        (
            (pixel_class == -1)
            & (hex_class == 1)
        )
    ] = 7


    return relationship


# =============================================================================
# 9. Raster visualisation helpers
# =============================================================================

def class_rgba(
    classes: np.ndarray,
    negative_color: str = NEGATIVE_DARK,
    positive_color: str = POSITIVE_DARK,
    alpha: int = 255,
) -> np.ndarray:

    from matplotlib.colors import (
        to_rgba
    )

    rgba = np.zeros(
        (
            *classes.shape,
            4
        ),
        dtype=np.uint8
    )

    neg = (
        np.array(
            to_rgba(
                negative_color
            )
        )
        * 255
    )

    pos = (
        np.array(
            to_rgba(
                positive_color
            )
        )
        * 255
    )

    neg[3] = alpha
    pos[3] = alpha

    rgba[
        classes == -1
    ] = neg.astype(
        np.uint8
    )

    rgba[
        classes == 1
    ] = pos.astype(
        np.uint8
    )

    return rgba


def relationship_rgba(
    relationship: np.ndarray
) -> np.ndarray:

    from matplotlib.colors import (
        to_rgba
    )

    rgba = np.zeros(
        (
            *relationship.shape,
            4
        ),
        dtype=np.uint8
    )

    for (
        code,
        colour
    ) in RELATIONSHIP_COLORS.items():

        value = (
            np.array(
                to_rgba(
                    colour
                )
            )
            * 255
        ).astype(
            np.uint8
        )

        value[3] = 255

        rgba[
            relationship == code
        ] = value

    return rgba


def robust_symmetric_limit(
    values: np.ndarray,
    percentile: float = 98.0
) -> float:

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return 1.0

    limit = float(
        np.nanpercentile(
            np.abs(finite),
            percentile
        )
    )

    if (
        not np.isfinite(limit)
        or limit <= 0
    ):

        limit = float(
            np.nanmax(
                np.abs(finite)
            )
        )

    if (
        not np.isfinite(limit)
        or limit <= 0
    ):

        limit = 1.0

    return limit


def add_panel_label(
    ax: plt.Axes,
    label: str
) -> None:

    ax.text(

        0.012,
        0.985,

        label,

        transform=
            ax.transAxes,

        ha="left",

        va="top",

        fontsize=10.5,

        fontweight="bold",

        color=TEXT_COLOR,

        zorder=30,
    )


# =============================================================================
# 10. Figure 1: conventional trend versus threshold-defined expansion
# =============================================================================

def plot_comparison_figure(
    land: gpd.GeoDataFrame,
    dry: Optional[gpd.GeoDataFrame],
    pixel_values: np.ndarray,
    pixel_class: np.ndarray,
    pixel_label: str,
    hex_class: np.ndarray,
) -> None:

    xmin, ymin, xmax, ymax = (
        projection_extent_from_frame()
    )

    extent = [
        xmin,
        xmax,
        ymin,
        ymax
    ]


    fig = plt.figure(

        figsize=
            COMPARISON_FIGSIZE,

        facecolor=
            "white"
    )


    ax_left = fig.add_axes(
        COMPARISON_LEFT_AX,
        facecolor="white"
    )


    ax_right = fig.add_axes(
        COMPARISON_RIGHT_AX,
        facecolor="white"
    )


    frame_left = draw_base_map(
        ax_left,
        land,
        dry
    )

    frame_right = draw_base_map(
        ax_right,
        land,
        dry
    )


    # =========================================================================
    # Conventional pixel-level trend
    # =========================================================================

    if pixel_label in {
        "Kendall’s τ",
        "Sen slope"
    }:

        if (
            pixel_label
            == "Kendall’s τ"
        ):

            # IMPORTANT:
            # identical tau range to Fig. 1a
            limit = TAU_ABS_MAX

        else:

            limit = (
                robust_symmetric_limit(
                    pixel_values
                )
            )


        norm = TwoSlopeNorm(

            vmin=
                -limit,

            vcenter=
                0.0,

            vmax=
                limit
        )


        masked = (
            np.ma.masked_invalid(
                pixel_values
            )
        )


        image_left = ax_left.imshow(

            masked,

            extent=
                extent,

            origin=
                "upper",

            cmap=
                TREND_CMAP,

            norm=
                norm,

            interpolation=
                "nearest",

            zorder=
                5,
        )


        # ---------------------------------------------------------------------
        # Colour bar
        # ---------------------------------------------------------------------

        cax = fig.add_axes(
            LEFT_CBAR_AX,
            facecolor="white"
        )


        scalar = (
            plt.cm.ScalarMappable(
                cmap=TREND_CMAP,
                norm=norm
            )
        )

        scalar.set_array([])


        colour_bar = fig.colorbar(

            scalar,

            cax=cax,

            orientation=
                "horizontal",

            extend=
                "both",
        )


        colour_bar.set_label(

            pixel_label,

            fontsize=8.0,

            labelpad=3
        )


        if (
            pixel_label
            == "Kendall’s τ"
        ):

            colour_bar.set_ticks([

                -limit,

                -limit / 2,

                0.0,

                limit / 2,

                limit,
            ])


        colour_bar.ax.tick_params(

            labelsize=7.0,

            length=2.3,

            width=0.55,

            pad=2,

            colors=TEXT_COLOR
        )


        colour_bar.outline.set_linewidth(
            0.55
        )


        colour_bar.outline.set_edgecolor(
            FRAME_COLOR
        )


        # Greenness interpretation rather than
        # cover-expansion terminology.
        colour_bar.ax.text(

            0.0,
            1.55,

            "Browning",

            transform=
                colour_bar.ax.transAxes,

            ha="left",

            va="bottom",

            fontsize=7.0,

            color=
                NEGATIVE_DARK
        )


        colour_bar.ax.text(

            1.0,
            1.55,

            "Greening",

            transform=
                colour_bar.ax.transAxes,

            ha="right",

            va="bottom",

            fontsize=7.0,

            color=
                POSITIVE_DARK
        )


    else:

        image_left = ax_left.imshow(

            class_rgba(
                pixel_class
            ),

            extent=extent,

            origin="upper",

            interpolation="nearest",

            zorder=5,
        )


        legend_ax = fig.add_axes(
            LEFT_CBAR_AX,
            facecolor="white"
        )

        legend_ax.set_axis_off()


        legend_ax.legend(

            handles=[

                Patch(
                    facecolor=
                        NEGATIVE_DARK,

                    edgecolor=
                        "none",

                    label=
                        "Significant pixel browning"
                ),

                Patch(
                    facecolor=
                        POSITIVE_DARK,

                    edgecolor=
                        "none",

                    label=
                        "Significant pixel greening"
                ),
            ],

            loc="center",

            ncol=2,

            frameon=False,

            fontsize=7.5,

            columnspacing=1.5,

            handlelength=1.4,
        )


    # =========================================================================
    # Threshold-defined hexagon result
    # =========================================================================

    image_right = ax_right.imshow(

        class_rgba(
            hex_class
        ),

        extent=extent,

        origin="upper",

        interpolation="nearest",

        zorder=5,
    )


    # Coastlines above raster
    for ax in (
        ax_left,
        ax_right
    ):

        land.boundary.plot(

            ax=ax,

            color=
                COAST_COLOR,

            linewidth=
                0.20,

            alpha=
                0.70,

            zorder=
                10,
        )


    image_left.set_clip_path(
        frame_left
    )

    image_right.set_clip_path(
        frame_right
    )


    clip_axis_artists(
        ax_left,
        frame_left
    )

    clip_axis_artists(
        ax_right,
        frame_right
    )


    add_panel_label(
        ax_left,
        "a"
    )

    add_panel_label(
        ax_right,
        "b"
    )


    ax_left.set_title(

        "Conventional pixel-level NDVI trend",

        fontsize=8.8,

        pad=3,

        color=TEXT_COLOR,
    )


    ax_right.set_title(

        "Threshold-defined vegetation-cover change",

        fontsize=8.8,

        pad=3,

        color=TEXT_COLOR,
    )


    # -------------------------------------------------------------------------
    # Hexagon legend
    # -------------------------------------------------------------------------

    legend_ax = fig.add_axes(
        RIGHT_LEGEND_AX,
        facecolor="white"
    )


    legend_ax.set_axis_off()


    legend_ax.legend(

        handles=[

            Patch(

                facecolor=
                    NEGATIVE_DARK,

                edgecolor=
                    "none",

                label=
                    "Hexagon contraction (DOWN)",
            ),

            Patch(

                facecolor=
                    POSITIVE_DARK,

                edgecolor=
                    "none",

                label=
                    "Hexagon expansion (UP)",
            ),
        ],

        loc="center",

        ncol=2,

        frameon=False,

        fontsize=7.5,

        columnspacing=1.6,

        handlelength=1.5,
    )


    save_figure(
        fig,
        COMPARISON_STEM
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. Figure 2: pixel-hex relationship map
# =============================================================================

def plot_relationship_figure(
    land: gpd.GeoDataFrame,
    dry: Optional[gpd.GeoDataFrame],
    relationship: np.ndarray,
) -> None:

    xmin, ymin, xmax, ymax = (
        projection_extent_from_frame()
    )

    extent = [
        xmin,
        xmax,
        ymin,
        ymax
    ]


    fig = plt.figure(

        figsize=
            RELATIONSHIP_FIGSIZE,

        facecolor=
            "white"
    )


    ax = fig.add_axes(
        RELATIONSHIP_MAP_AX,
        facecolor="white"
    )


    frame = draw_base_map(
        ax,
        land,
        dry
    )


    image = ax.imshow(

        relationship_rgba(
            relationship
        ),

        extent=extent,

        origin="upper",

        interpolation="nearest",

        zorder=5,
    )


    image.set_clip_path(
        frame
    )


    land.boundary.plot(

        ax=ax,

        color=
            COAST_COLOR,

        linewidth=
            0.20,

        alpha=
            0.70,

        zorder=
            10,
    )


    clip_axis_artists(
        ax,
        frame
    )


    add_panel_label(
        ax,
        "c"
    )


    # -------------------------------------------------------------------------
    # Relationship legend
    # -------------------------------------------------------------------------

    legend_ax = fig.add_axes(
        RELATIONSHIP_LEGEND_AX,
        facecolor="white"
    )


    legend_ax.set_axis_off()


    handles = [

        Patch(

            facecolor=
                RELATIONSHIP_COLORS[
                    code
                ],

            edgecolor=
                "none",

            label=
                RELATIONSHIP_LABELS[
                    code
                ],
        )

        for code in range(
            1,
            8
        )
    ]


    legend_ax.legend(

        handles=handles,

        loc="center",

        ncol=4,

        frameon=False,

        fontsize=7.3,

        handlelength=1.5,

        columnspacing=1.5,

        labelspacing=0.8,

        borderaxespad=0,
    )


    save_figure(
        fig,
        RELATIONSHIP_STEM
    )


    plt.close(
        fig
    )


# =============================================================================
# 12. Reporting
# =============================================================================

def write_detection_report(
    pixel_meta: pd.DataFrame,
    overlap_meta: pd.DataFrame,
    pixel_bands: Dict[
        str,
        Optional[int]
    ],
    overlap_bands: Dict[
        str,
        Optional[int]
    ],
    overlap_code_map: Dict[
        int,
        Tuple[int, int]
    ],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    pixel_meta.to_csv(

        OUTPUT_DIR
        / "pixel_raster_band_report.csv",

        index=False,

        encoding="utf-8-sig",
    )


    overlap_meta.to_csv(

        OUTPUT_DIR
        / "overlap_raster_band_report.csv",

        index=False,

        encoding="utf-8-sig",
    )


    lines = [

        "Detected pixel raster bands",

        str(
            pixel_bands
        ),

        "",

        "Detected overlap raster bands",

        str(
            overlap_bands
        ),

        "",

        "Decoded combined overlap codes",
    ]


    if overlap_code_map:

        for code in sorted(
            overlap_code_map
        ):

            lines.append(

                f"{code}: "
                f"{overlap_code_map[code]}"
            )

    else:

        lines.append(
            "Separate pixel and hex class bands were used."
        )


    lines.extend([

        "",

        "Shared map style",

        "Negative change: brown",

        "Positive change: green",

        (
            "Kendall tau absolute colour limit: "
            f"{TAU_ABS_MAX}"
        ),
    ])


    (
        OUTPUT_DIR
        / "detection_summary.txt"
    ).write_text(

        "\n".join(
            lines
        ),

        encoding="utf-8"
    )


# =============================================================================
# 13. Main
# =============================================================================

def main() -> None:

    configure_matplotlib()

    ensure_inputs()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "Reading and preparing vector backgrounds..."
    )


    land_geo = (
        subset_to_display_bbox(
            gpd.read_file(
                LAND_SHP
            )
        )
    )


    dry_geo = None


    if DRY_SHP is not None:

        dry_geo = (
            subset_to_display_bbox(
                gpd.read_file(
                    DRY_SHP
                )
            )
        )


    land = simplify_geometry(

        land_geo.to_crs(
            PROJ
        ),

        LAND_SIMPLIFY
    )


    dry = (

        simplify_geometry(

            dry_geo.to_crs(
                PROJ
            ),

            DRY_SIMPLIFY
        )

        if dry_geo is not None

        else None
    )


    print(
        "Inspecting and reprojecting the conventional pixel trend raster..."
    )


    with rasterio.open(
        PIXEL_TIF
    ) as pixel_src:

        if pixel_src.crs is None:

            raise ValueError(
                f"{PIXEL_TIF} has no CRS."
            )


        pixel_meta = (
            band_metadata(
                pixel_src
            )
        )


        pixel_bands = (
            detect_pixel_bands(
                pixel_src
            )
        )


        (
            pixel_values,
            pixel_class,
            pixel_label

        ) = build_pixel_trend_arrays(

            pixel_src,
            pixel_bands
        )


    print(
        "Inspecting and decoding the pixel-hex overlap raster..."
    )


    with rasterio.open(
        OVERLAP_TIF
    ) as overlap_src:

        if overlap_src.crs is None:

            raise ValueError(
                f"{OVERLAP_TIF} has no CRS."
            )


        overlap_meta = (
            band_metadata(
                overlap_src
            )
        )


        overlap_bands = (
            detect_overlap_bands(
                overlap_src
            )
        )


        (
            overlap_pixel_class,
            hex_class,
            code_map

        ) = build_overlap_classes(

            overlap_src,
            overlap_bands
        )


    relationship = (
        build_relationship_class(
            overlap_pixel_class,
            hex_class
        )
    )


    print(
        "Creating the two-panel comparison map..."
    )


    plot_comparison_figure(

        land=land,

        dry=dry,

        pixel_values=
            pixel_values,

        pixel_class=
            pixel_class,

        pixel_label=
            pixel_label,

        hex_class=
            hex_class,
    )


    print(
        "Creating the pixel-hex relationship map..."
    )


    plot_relationship_figure(

        land,

        dry,

        relationship
    )


    write_detection_report(

        pixel_meta=
            pixel_meta,

        overlap_meta=
            overlap_meta,

        pixel_bands=
            pixel_bands,

        overlap_bands=
            overlap_bands,

        overlap_code_map=
            code_map,
    )


    print(
        "Finished. Outputs saved in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        f"  {COMPARISON_STEM}.png/.pdf/.svg"
    )

    print(
        f"  {RELATIONSHIP_STEM}.png/.pdf/.svg"
    )

    print(
        "  pixel_raster_band_report.csv"
    )

    print(
        "  overlap_raster_band_report.csv"
    )

    print(
        "  detection_summary.txt"
    )


if __name__ == "__main__":
    main()