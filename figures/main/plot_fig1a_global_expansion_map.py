#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1a: global dryland vegetation expansion/contraction with regional zooms.

Based on the original plot_fig1a_map_export.py.

Main revisions:
- Use the final UP/DOWN filter: P < 0.05 with concordant Kendall-tau and Sen-slope signs.
- Replace the continuous Kendall's tau colour scale in the MAIN figure with
  a high-contrast categorical display:
      negative tau = contraction (DOWN; brown)
      positive tau = expansion (UP; green)
- Add five regional zoom panels corresponding directly to the spatial patterns
  described in the Results:
      1. Western North America
      2. Dry margins of South America
      3. African dryland belt
      4. West and Central Asia
      5. Australia
- Add subtle locator boxes to the global map.
- Preserve the Robinson projection and original dryland extent.
- Export white-background PNG plus vector PDF/SVG.

NOTE
----
The plotting filter follows the final analysis definition directly: UP requires P < 0.05, tau > 0 and Sen slope > 0; DOWN requires P < 0.05, tau < 0 and Sen slope < 0. Geometry and trend estimates are not recomputed here.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import box


# =============================================================================
# User settings
# =============================================================================

# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------

HEX_SHP = Path(
    r"hex_data/NDVI_trend_hex_100.shp"
)

LAND_SHP = Path(
    r"world_map/ne_10m_land.shp"
)

DRY_SHP = Path(
    r"Drylands_latest_July2014/drylands_UNCCD_CBD_july2014.shp"
)


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

OUT_PNG = Path(
    r"Fig1_NDVI_100_global_with_regional_zooms.png"
)

OUT_PDF = Path(
    r"Fig1_NDVI_100_global_with_regional_zooms.pdf"
)

OUT_SVG = Path(
    r"Fig1_NDVI_100_global_with_regional_zooms.svg"
)


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

EXPORT_TRANSPARENT = False
DPI = 600


# -----------------------------------------------------------------------------
# Attribute fields
# -----------------------------------------------------------------------------

P_COL = "pvalue"
TAU_COL = "tau"
SLOPE_COL = "slope"

P_THRESHOLD = 0.05


# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------

PROJ = "ESRI:54030"  # Robinson

LON_MIN = -180
LON_MAX = 180

LAT_MIN = -58
LAT_MAX = 85


# =============================================================================
# Regional zoom definitions
# =============================================================================
#
# bbox format:
# (lon_min, lon_max, lat_min, lat_max)
#
# These regions correspond directly to the Results text:
#
# "...western North America, the dry margins of South America,
# the African dryland belt, West and Central Asia, and parts of Australia..."
#

REGIONS = [
    {
        "title": "Western North America",
        "bbox": (-125, -100, 25, 55),
    },
    {
        "title": "Dry margins of South America",
        "bbox": (-82, -55, -52, -10),
    },
    {
        "title": "African dryland belt",
        "bbox": (-20, 40, 5, 30),
    },
    {
        "title": "West & Central Asia",
        "bbox": (35, 90, 20, 55),
    },
    {
        "title": "Australia",
        "bbox": (110, 155, -45, -10),
    },
]


# =============================================================================
# Colour system
# =============================================================================

# -----------------------------------------------------------------------------
# Background
# -----------------------------------------------------------------------------

OCEAN_COLOR = "#FFFFFF"

# Non-dryland land
LAND_COLOR = "#F7F7F7"

# Dryland domain
DRY_COLOR = "#E9E8E1"

# Coastlines
COAST_COLOR = "#666666"

# Main-frame outline
FRAME_COLOR = "#777777"


# -----------------------------------------------------------------------------
# Trend colours
# -----------------------------------------------------------------------------
#
# High contrast is deliberate because the global map is substantially reduced
# when assembled into the final multi-panel Figure 1.
#

DOWN_COLOR = "#7F3B08"
UP_COLOR = "#238443"

ALPHA_TREND = 1.00


# -----------------------------------------------------------------------------
# Locator boxes
# -----------------------------------------------------------------------------

SHOW_LOCATOR_BOXES = True

LOCATOR_COLOR = "#626262"
LOCATOR_LINEWIDTH = 0.55
LOCATOR_ALPHA = 0.70


# =============================================================================
# Geometry simplification
# =============================================================================

# Global map
LAND_SIMPLIFY_GLOBAL = 4500
DRY_SIMPLIFY_GLOBAL = 4500
HEX_SIMPLIFY_GLOBAL = 500

# Insets retain more local detail
LAND_SIMPLIFY_INSET = 1800
DRY_SIMPLIFY_INSET = 1800

# Preserve 100-km2 hexagons in zoom panels
HEX_SIMPLIFY_INSET = 0


# =============================================================================
# Figure layout
# =============================================================================

FIGSIZE = (10.0, 5.55)


# -----------------------------------------------------------------------------
# Global map
# -----------------------------------------------------------------------------

MAP_AX = [
    0.020,   # left
    0.345,   # bottom
    0.960,   # width
    0.625,   # height
]


# -----------------------------------------------------------------------------
# Regional inset row
# -----------------------------------------------------------------------------

INSET_Y = 0.105
INSET_H = 0.175
INSET_W = 0.172

INSET_LEFTS = [
    0.025,
    0.220,
    0.415,
    0.610,
    0.805,
]


# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------

LEGEND_AX = [
    0.345,
    0.015,
    0.310,
    0.055,
]


# =============================================================================
# Typography / line style
# =============================================================================

FONT_FAMILY = "Arial"

INSET_TITLE_FONT = 7.0
LEGEND_FONT = 7.3

COAST_LINEWIDTH_GLOBAL = 0.20
COAST_LINEWIDTH_INSET = 0.35

INSET_BORDER_COLOR = "#777777"
INSET_BORDER_WIDTH = 0.55


# =============================================================================
# Helper functions
# =============================================================================

def ensure_wgs84(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Ensure GeoDataFrame uses EPSG:4326.
    """

    if gdf.crs is None:

        gdf = gdf.set_crs(
            "EPSG:4326",
            allow_override=True
        )

    return gdf.to_crs(
        "EPSG:4326"
    )


def clip_to_display_extent(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip geometries to the global display extent before projection.
    """

    gdf = ensure_wgs84(
        gdf
    )

    clip_poly = box(
        LON_MIN,
        LAT_MIN,
        LON_MAX,
        LAT_MAX
    )

    clipped = gpd.clip(
        gdf,
        clip_poly
    )

    clipped = clipped[
        clipped.geometry.notna()
        & ~clipped.geometry.is_empty
    ].copy()

    return clipped


def subset_geo_bbox(
    gdf: gpd.GeoDataFrame,
    bbox_tuple,
    buffer_deg: float = 1.5,
) -> gpd.GeoDataFrame:
    """
    Select geometries intersecting a geographic bounding box.

    A small buffer is added so that coastlines do not terminate exactly
    at the inset edge.
    """

    lon_min, lon_max, lat_min, lat_max = bbox_tuple

    sub = gdf.cx[
        lon_min - buffer_deg:
        lon_max + buffer_deg,

        lat_min - buffer_deg:
        lat_max + buffer_deg
    ].copy()

    sub = sub[
        sub.geometry.notna()
        & ~sub.geometry.is_empty
    ].copy()

    return sub


def simplify_projected(
    gdf: gpd.GeoDataFrame,
    tolerance: float
) -> gpd.GeoDataFrame:
    """
    Simplify projected geometries while preserving topology.
    """

    if tolerance <= 0:
        return gdf.copy()

    out = gdf.copy()

    out["geometry"] = (
        out.geometry.simplify(
            tolerance,
            preserve_topology=True
        )
    )

    out = out[
        out.geometry.notna()
        & ~out.geometry.is_empty
    ].copy()

    return out


def make_projection_boundary_patch(
    ax,
    proj: str = PROJ,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    n: int = 500,
    facecolor: str = OCEAN_COLOR,
    edgecolor: str = FRAME_COLOR,
    linewidth: float = 0.55,
    zorder: int = 0,
) -> PathPatch:
    """
    Build a projected Robinson-style global map frame.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        proj,
        always_xy=True
    )

    # Bottom edge
    lons_bottom = np.linspace(
        lon_min,
        lon_max,
        n
    )

    lats_bottom = np.full(
        n,
        lat_min
    )

    # Right edge
    lats_right = np.linspace(
        lat_min,
        lat_max,
        n
    )

    lons_right = np.full(
        n,
        lon_max
    )

    # Top edge
    lons_top = np.linspace(
        lon_max,
        lon_min,
        n
    )

    lats_top = np.full(
        n,
        lat_max
    )

    # Left edge
    lats_left = np.linspace(
        lat_max,
        lat_min,
        n
    )

    lons_left = np.full(
        n,
        lon_min
    )

    lon = np.concatenate([
        lons_bottom,
        lons_right,
        lons_top,
        lons_left,
    ])

    lat = np.concatenate([
        lats_bottom,
        lats_right,
        lats_top,
        lats_left,
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

    path = MplPath(
        vertices,
        codes
    )

    patch = PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        joinstyle="round",
        capstyle="round"
    )

    ax.add_patch(
        patch
    )

    return patch


def set_axis_to_patch_extent(
    ax,
    patch: PathPatch,
    pad_x_frac: float = 0.008,
    pad_y_frac: float = 0.015
) -> None:
    """
    Set global axis limits from the Robinson frame.
    """

    vertices = (
        patch
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
        xmin - dx * pad_x_frac,
        xmax + dx * pad_x_frac
    )

    ax.set_ylim(
        ymin - dy * pad_y_frac,
        ymax + dy * pad_y_frac
    )


def projected_bbox_extent(
    bbox_tuple,
    n: int = 150
):
    """
    Derive projected x/y limits from a geographic bbox.

    Sampling all four geographic edges gives more reliable bounds under
    Robinson projection than transforming only four corners.
    """

    lon_min, lon_max, lat_min, lat_max = bbox_tuple

    transformer = Transformer.from_crs(
        "EPSG:4326",
        PROJ,
        always_xy=True
    )

    # Bottom
    lon_bottom = np.linspace(
        lon_min,
        lon_max,
        n
    )

    lat_bottom = np.full(
        n,
        lat_min
    )

    # Right
    lon_right = np.full(
        n,
        lon_max
    )

    lat_right = np.linspace(
        lat_min,
        lat_max,
        n
    )

    # Top
    lon_top = np.linspace(
        lon_max,
        lon_min,
        n
    )

    lat_top = np.full(
        n,
        lat_max
    )

    # Left
    lon_left = np.full(
        n,
        lon_min
    )

    lat_left = np.linspace(
        lat_max,
        lat_min,
        n
    )

    lon = np.concatenate([
        lon_bottom,
        lon_right,
        lon_top,
        lon_left
    ])

    lat = np.concatenate([
        lat_bottom,
        lat_right,
        lat_top,
        lat_left
    ])

    x, y = transformer.transform(
        lon,
        lat
    )

    return (
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )


def set_inset_extent(
    ax,
    bbox_tuple,
    pad_fraction: float = 0.025
) -> None:
    """
    Set regional inset limits in projected coordinates.
    """

    xmin, xmax, ymin, ymax = (
        projected_bbox_extent(
            bbox_tuple
        )
    )

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(
        xmin - dx * pad_fraction,
        xmax + dx * pad_fraction
    )

    ax.set_ylim(
        ymin - dy * pad_fraction,
        ymax + dy * pad_fraction
    )


def plot_geographic_bbox(
    ax,
    bbox_tuple,
    color=LOCATOR_COLOR,
    linewidth=LOCATOR_LINEWIDTH,
    alpha=LOCATOR_ALPHA,
    zorder=20,
    n: int = 100,
) -> None:
    """
    Draw a geographically correct locator box on the Robinson map.
    """

    lon_min, lon_max, lat_min, lat_max = bbox_tuple

    transformer = Transformer.from_crs(
        "EPSG:4326",
        PROJ,
        always_xy=True
    )

    # Bottom
    lons = np.linspace(
        lon_min,
        lon_max,
        n
    )

    lats = np.full(
        n,
        lat_min
    )

    x, y = transformer.transform(
        lons,
        lats
    )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

    # Top
    lons = np.linspace(
        lon_min,
        lon_max,
        n
    )

    lats = np.full(
        n,
        lat_max
    )

    x, y = transformer.transform(
        lons,
        lats
    )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

    # Left
    lats = np.linspace(
        lat_min,
        lat_max,
        n
    )

    lons = np.full(
        n,
        lon_min
    )

    x, y = transformer.transform(
        lons,
        lats
    )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )

    # Right
    lats = np.linspace(
        lat_min,
        lat_max,
        n
    )

    lons = np.full(
        n,
        lon_max
    )

    x, y = transformer.transform(
        lons,
        lats
    )

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder
    )


def split_up_down(
    sig_gdf: gpd.GeoDataFrame
):
    """
    Split significant hexagons by the sign of Kendall's tau.
    """

    down = sig_gdf.loc[
        sig_gdf[TAU_COL] < 0
    ].copy()

    up = sig_gdf.loc[
        sig_gdf[TAU_COL] > 0
    ].copy()

    return down, up


def draw_trend_hexagons(
    ax,
    sig_gdf: gpd.GeoDataFrame,
    zorder: int = 4,
) -> None:
    """
    Plot significant contraction and expansion using high-contrast categories.
    """

    down, up = split_up_down(
        sig_gdf
    )

    # Plot DOWN first.
    if not down.empty:

        down.plot(
            ax=ax,
            color=DOWN_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=ALPHA_TREND,
            zorder=zorder
        )

    # Plot UP.
    if not up.empty:

        up.plot(
            ax=ax,
            color=UP_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=ALPHA_TREND,
            zorder=zorder + 0.1
        )


def draw_inset_border(
    ax
) -> None:
    """
    Add a subtle rectangular frame around a regional inset.
    """

    border = Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=INSET_BORDER_COLOR,
        linewidth=INSET_BORDER_WIDTH,
        zorder=50,
        clip_on=False
    )

    ax.add_patch(
        border
    )


# =============================================================================
# Main plotting
# =============================================================================

def main() -> None:

    # =========================================================================
    # Matplotlib style
    # =========================================================================

    plt.rcParams.update({

        "font.family":
            FONT_FAMILY,

        "font.sans-serif":
            [FONT_FAMILY],

        "font.size":
            7.0,

        "axes.linewidth":
            0.55,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,

        "svg.fonttype":
            "none",

        "figure.facecolor":
            "white",

        "axes.facecolor":
            "white",

        "savefig.facecolor":
            "white",

        "text.antialiased":
            True,

        "lines.antialiased":
            True,
    })


    # =========================================================================
    # Read data
    # =========================================================================

    print(
        "Reading input data..."
    )

    hex_gdf = gpd.read_file(
        HEX_SHP
    )

    land_gdf = gpd.read_file(
        LAND_SHP
    )

    dry_gdf = gpd.read_file(
        DRY_SHP
    )


    # Final UP/DOWN definition: significance and concordant Kendall/Sen signs.
    required = [P_COL, TAU_COL, SLOPE_COL]
    missing = [c for c in required if c not in hex_gdf.columns]
    if missing:
        raise KeyError(
            "Input hexagon layer is missing fields required by the final "
            f"UP/DOWN definition: {missing}"
        )

    sig_gdf = hex_gdf.loc[
        (hex_gdf[P_COL] < P_THRESHOLD)
        & (
            ((hex_gdf[TAU_COL] > 0) & (hex_gdf[SLOPE_COL] > 0))
            | ((hex_gdf[TAU_COL] < 0) & (hex_gdf[SLOPE_COL] < 0))
        )
    ].copy()


    # =========================================================================
    # Convert to geographic CRS and clip global domain
    # =========================================================================

    land_geo = clip_to_display_extent(
        land_gdf
    )

    dry_geo = clip_to_display_extent(
        dry_gdf
    )

    sig_geo = clip_to_display_extent(
        sig_gdf
    )


    # =========================================================================
    # Report significant counts
    # =========================================================================

    down_geo, up_geo = split_up_down(
        sig_geo
    )

    n_up = len(
        up_geo
    )

    n_down = len(
        down_geo
    )

    ratio = (
        n_up / n_down
        if n_down > 0
        else np.nan
    )


    print(
        f"Significant expansion hexagons: {n_up:,}"
    )

    print(
        f"Significant contraction hexagons: {n_down:,}"
    )

    print(
        f"Expansion/contraction ratio: {ratio:.2f}"
    )


    # =========================================================================
    # Prepare global projected layers
    # =========================================================================

    print(
        "Preparing global projected layers..."
    )


    land_global = simplify_projected(

        land_geo.to_crs(
            PROJ
        ),

        LAND_SIMPLIFY_GLOBAL
    )


    dry_global = simplify_projected(

        dry_geo.to_crs(
            PROJ
        ),

        DRY_SIMPLIFY_GLOBAL
    )


    sig_global = simplify_projected(

        sig_geo.to_crs(
            PROJ
        ),

        HEX_SIMPLIFY_GLOBAL
    )


    # =========================================================================
    # Create figure
    # =========================================================================

    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI,
        facecolor="white"
    )


    # =========================================================================
    # Global map
    # =========================================================================

    ax_main = fig.add_axes(
        MAP_AX
    )


    map_frame = make_projection_boundary_patch(

        ax_main,

        proj=PROJ,

        lon_min=LON_MIN,
        lon_max=LON_MAX,

        lat_min=LAT_MIN,
        lat_max=LAT_MAX,

        facecolor=OCEAN_COLOR,

        edgecolor=FRAME_COLOR,

        linewidth=0.55,

        zorder=0
    )


    # -------------------------------------------------------------------------
    # Land background
    # -------------------------------------------------------------------------

    land_global.plot(

        ax=ax_main,

        color=LAND_COLOR,

        edgecolor="none",

        linewidth=0,

        zorder=1
    )


    # -------------------------------------------------------------------------
    # Dryland domain
    # -------------------------------------------------------------------------

    dry_global.plot(

        ax=ax_main,

        color=DRY_COLOR,

        edgecolor="none",

        linewidth=0,

        alpha=1.0,

        zorder=2
    )


    # -------------------------------------------------------------------------
    # Significant UP / DOWN hexagons
    # -------------------------------------------------------------------------

    draw_trend_hexagons(

        ax=ax_main,

        sig_gdf=sig_global,

        zorder=4
    )


    # -------------------------------------------------------------------------
    # Coastlines
    # -------------------------------------------------------------------------

    land_global.boundary.plot(

        ax=ax_main,

        color=COAST_COLOR,

        linewidth=
            COAST_LINEWIDTH_GLOBAL,

        alpha=0.72,

        zorder=10
    )


    # -------------------------------------------------------------------------
    # Locator boxes
    # -------------------------------------------------------------------------

    if SHOW_LOCATOR_BOXES:

        for region in REGIONS:

            plot_geographic_bbox(

                ax=ax_main,

                bbox_tuple=
                    region["bbox"]
            )


    # -------------------------------------------------------------------------
    # Clip global map layers to Robinson frame
    # -------------------------------------------------------------------------

    for collection in ax_main.collections:

        collection.set_clip_path(
            map_frame
        )


    set_axis_to_patch_extent(
        ax_main,
        map_frame
    )


    ax_main.set_axis_off()


    # =========================================================================
    # Regional zoom panels
    # =========================================================================

    print(
        "Drawing regional zoom panels..."
    )


    for (
        inset_left,
        region
    ) in zip(
        INSET_LEFTS,
        REGIONS
    ):

        bbox_tuple = (
            region["bbox"]
        )


        ax = fig.add_axes([

            inset_left,

            INSET_Y,

            INSET_W,

            INSET_H,
        ])


        ax.set_facecolor(
            "white"
        )


        # ---------------------------------------------------------------------
        # Geographic subsets
        # ---------------------------------------------------------------------

        local_land_geo = (
            subset_geo_bbox(
                land_geo,
                bbox_tuple
            )
        )

        local_dry_geo = (
            subset_geo_bbox(
                dry_geo,
                bbox_tuple
            )
        )

        local_sig_geo = (
            subset_geo_bbox(
                sig_geo,
                bbox_tuple,
                buffer_deg=0.5
            )
        )


        # ---------------------------------------------------------------------
        # Projection
        # ---------------------------------------------------------------------

        local_land = (
            simplify_projected(

                local_land_geo.to_crs(
                    PROJ
                ),

                LAND_SIMPLIFY_INSET
            )
        )


        local_dry = (
            simplify_projected(

                local_dry_geo.to_crs(
                    PROJ
                ),

                DRY_SIMPLIFY_INSET
            )
        )


        local_sig = (
            simplify_projected(

                local_sig_geo.to_crs(
                    PROJ
                ),

                HEX_SIMPLIFY_INSET
            )
        )


        # ---------------------------------------------------------------------
        # Local land
        # ---------------------------------------------------------------------

        if not local_land.empty:

            local_land.plot(

                ax=ax,

                color=LAND_COLOR,

                edgecolor="none",

                linewidth=0,

                zorder=1
            )


        # ---------------------------------------------------------------------
        # Local dryland domain
        # ---------------------------------------------------------------------

        if not local_dry.empty:

            local_dry.plot(

                ax=ax,

                color=DRY_COLOR,

                edgecolor="none",

                linewidth=0,

                alpha=1.0,

                zorder=2
            )


        # ---------------------------------------------------------------------
        # Local significant trends
        # ---------------------------------------------------------------------

        if not local_sig.empty:

            draw_trend_hexagons(

                ax=ax,

                sig_gdf=local_sig,

                zorder=4
            )


        # ---------------------------------------------------------------------
        # Local coastline
        # ---------------------------------------------------------------------

        if not local_land.empty:

            local_land.boundary.plot(

                ax=ax,

                color=COAST_COLOR,

                linewidth=
                    COAST_LINEWIDTH_INSET,

                alpha=0.78,

                zorder=10
            )


        # ---------------------------------------------------------------------
        # Inset extent
        # ---------------------------------------------------------------------

        set_inset_extent(

            ax=ax,

            bbox_tuple=bbox_tuple,

            pad_fraction=0.015
        )


        ax.set_axis_off()


        # ---------------------------------------------------------------------
        # Inset title
        # ---------------------------------------------------------------------

        ax.set_title(

            region["title"],

            fontsize=
                INSET_TITLE_FONT,

            color=
                "#222222",

            pad=
                2.5
        )


        # ---------------------------------------------------------------------
        # Inset border
        # ---------------------------------------------------------------------

        draw_inset_border(
            ax
        )


    # =========================================================================
    # Legend
    # =========================================================================

    legend_ax = fig.add_axes(
        LEGEND_AX
    )

    legend_ax.set_axis_off()


    handles = [

        Patch(

            facecolor=
                DOWN_COLOR,

            edgecolor=
                "none",

            label=
                "Contraction (DOWN)"
        ),

        Patch(

            facecolor=
                UP_COLOR,

            edgecolor=
                "none",

            label=
                "Expansion (UP)"
        ),
    ]


    legend_ax.legend(

        handles=handles,

        loc="center",

        ncol=2,

        frameon=False,

        fontsize=
            LEGEND_FONT,

        handlelength=
            1.55,

        handleheight=
            0.85,

        columnspacing=
            2.0,

        handletextpad=
            0.55,

        borderaxespad=
            0
    )


    # =========================================================================
    # Save
    # =========================================================================

    if EXPORT_TRANSPARENT:

        fig.patch.set_alpha(
            0
        )

        for ax in fig.axes:

            ax.set_facecolor(
                "none"
            )


        save_kwargs = {

            "transparent":
                True,

            "bbox_inches":
                "tight",

            "pad_inches":
                0.01,
        }


    else:

        fig.patch.set_facecolor(
            "white"
        )

        for ax in fig.axes:

            ax.set_facecolor(
                "white"
            )


        save_kwargs = {

            "transparent":
                False,

            "facecolor":
                "white",

            "bbox_inches":
                "tight",

            "pad_inches":
                0.01,
        }


    fig.savefig(

        OUT_PNG,

        dpi=DPI,

        **save_kwargs
    )


    fig.savefig(

        OUT_PDF,

        **save_kwargs
    )


    fig.savefig(

        OUT_SVG,

        **save_kwargs
    )


    plt.close(
        fig
    )


    # =========================================================================
    # Report
    # =========================================================================

    print(
        f"Saved: {OUT_PNG}"
    )

    print(
        f"Saved: {OUT_PDF}"
    )

    print(
        f"Saved: {OUT_SVG}"
    )


if __name__ == "__main__":
    main()