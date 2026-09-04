#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2g: dominant attribution class under the main CLIM_PLUS_XCO2 framework.

Revised map style aligned with Fig. 1a:
- Robinson projected map frame rather than an artificial ellipse.
- Antarctica / far-southern high latitudes removed.
- White ocean, pale land, pale dryland extent.
- Natural-dominant and residual-dominant hexagons shown with Fig. 1 / Fig. 2 palette.
- Transparent PNG plus vector PDF/SVG outputs for manuscript assembly.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from pyproj import Transformer
from shapely.geometry import box

# =============================================================================
# User settings
# =============================================================================

ATTR_XLSX = Path(r"trend_outputs_xco2_vpd_resid/06_trend_attribution_by_hex_frac_UP.xlsx")
HEX_SHP = Path(r"hex_data/NDVI_trend_hex_100.shp")
LAND_SHP = Path(r"world_map/ne_10m_land.shp")
DRY_SHP = Path(r"Drylands_latest_July2014/drylands_UNCCD_CBD_july2014.shp")

OUT_PNG = Path(r"Fig2g_dominance_map_refined.png")
OUT_PDF = Path(r"Fig2g_dominance_map_refined.pdf")
OUT_SVG = Path(r"Fig2g_dominance_map_refined.svg")
OUT_CSV = Path(r"Fig2g_dominance_map_refined.csv")

EXPORT_TRANSPARENT = True

# Projection: ESRI:54030 = Robinson.
PROJ = "ESRI:54030"

# Display extent: remove Antarctica and far-southern high latitudes.
LON_MIN = -180
LON_MAX = 180
LAT_MIN = -58
LAT_MAX = 85

# Colours aligned with Fig. 1a / Fig. 2d-f.
OCEAN_COLOR = "#FFFFFF"
LAND_COLOR = "#F4F4F4"
DRY_COLOR = "#EDE9D8"
COAST_COLOR = "#404040"
NATURAL_COLOR = "#6BAED6"
RESIDUAL_COLOR = "#F4A582"
TEXT_COLOR = "#222222"

# Geometry simplification in projected map units.
LAND_SIMPLIFY = 4500
DRY_SIMPLIFY = 4500
HEX_SIMPLIFY = 1000

# Panel layout. Main map uses a Fig. 1a-like wide frame; legend sits outside the map.
FIGSIZE = (10.0, 4.0)
MAP_AX = [0.02, 0.14, 0.88, 0.82]
LEG_AX = [0.915, 0.36, 0.08, 0.28]

# =============================================================================
# Helper functions
# =============================================================================

def clip_to_display_extent(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip geometries to the selected lon/lat display extent before projection."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf = gdf.to_crs("EPSG:4326")
    clip_poly = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    clipped = gpd.clip(gdf, clip_poly)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    return clipped


def make_projection_boundary_patch(
    ax,
    proj: str = PROJ,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    n: int = 500,
    facecolor: str = OCEAN_COLOR,
    edgecolor: str = "black",
    linewidth: float = 0.75,
    zorder: int = 0,
) -> PathPatch:
    """Build a Robinson-style projected map-frame patch from geographic edges."""
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    lons_bottom = np.linspace(lon_min, lon_max, n)
    lats_bottom = np.full(n, lat_min)

    lats_right = np.linspace(lat_min, lat_max, n)
    lons_right = np.full(n, lon_max)

    lons_top = np.linspace(lon_max, lon_min, n)
    lats_top = np.full(n, lat_max)

    lats_left = np.linspace(lat_max, lat_min, n)
    lons_left = np.full(n, lon_min)

    lon = np.concatenate([lons_bottom, lons_right, lons_top, lons_left])
    lat = np.concatenate([lats_bottom, lats_right, lats_top, lats_left])

    x, y = transformer.transform(lon, lat)
    verts = np.column_stack([x, y])
    verts = np.vstack([verts, verts[0]])

    codes = np.full(len(verts), MplPath.LINETO)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY

    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def set_axis_to_patch_extent(ax, patch: PathPatch, pad_x_frac: float = 0.01, pad_y_frac: float = 0.02) -> None:
    """Set axis limits using the projected boundary patch extent."""
    verts = patch.get_path().vertices
    xmin, ymin = np.nanmin(verts[:, 0]), np.nanmin(verts[:, 1])
    xmax, ymax = np.nanmax(verts[:, 0]), np.nanmax(verts[:, 1])

    pad_x = (xmax - xmin) * pad_x_frac
    pad_y = (ymax - ymin) * pad_y_frac
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)


def apply_transparent_background(fig) -> None:
    """Make all axes transparent for manuscript assembly."""
    fig.patch.set_alpha(0)
    for axis in fig.axes:
        axis.set_facecolor("none")


# =============================================================================
# Main plotting
# =============================================================================

def main() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })

    # ---------------------------------------------------------------------
    # Read and prepare attribution labels
    # ---------------------------------------------------------------------
    attr = pd.read_excel(ATTR_XLSX)
    tmp = attr[["hex_id", "beta_nat_co2", "beta_res_co2"]].dropna().copy()
    tmp["dominance"] = np.where(
        tmp["beta_nat_co2"].abs() >= tmp["beta_res_co2"].abs(),
        "Natural-dominant",
        "Residual-dominant",
    )
    tmp.to_csv(OUT_CSV, index=False)

    # ---------------------------------------------------------------------
    # Read spatial layers
    # ---------------------------------------------------------------------
    hex_gdf = gpd.read_file(HEX_SHP)
    land_gdf = gpd.read_file(LAND_SHP)
    dry_gdf = gpd.read_file(DRY_SHP)

    if hex_gdf.crs is None:
        hex_gdf = hex_gdf.set_crs("EPSG:4326", allow_override=True)

    dom_gdf = hex_gdf[["hex_id", "geometry"]].merge(
        tmp[["hex_id", "dominance"]], on="hex_id", how="inner"
    )
    dom_gdf = dom_gdf.dropna(subset=["dominance"]).copy()

    # ---------------------------------------------------------------------
    # Clip before projection
    # ---------------------------------------------------------------------
    land_gdf = clip_to_display_extent(land_gdf)
    dry_gdf = clip_to_display_extent(dry_gdf)
    dom_gdf = clip_to_display_extent(dom_gdf)

    # ---------------------------------------------------------------------
    # Project to Robinson
    # ---------------------------------------------------------------------
    land_gdf = land_gdf.to_crs(PROJ)
    dry_gdf = dry_gdf.to_crs(PROJ)
    dom_gdf = dom_gdf.to_crs(PROJ)

    # ---------------------------------------------------------------------
    # Simplify geometries for clean rendering
    # ---------------------------------------------------------------------
    land_gdf["geometry"] = land_gdf.geometry.simplify(LAND_SIMPLIFY, preserve_topology=True)
    dry_gdf["geometry"] = dry_gdf.geometry.simplify(DRY_SIMPLIFY, preserve_topology=True)
    dom_gdf["geometry"] = dom_gdf.geometry.simplify(HEX_SIMPLIFY, preserve_topology=True)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax = fig.add_axes(MAP_AX)

    map_frame = make_projection_boundary_patch(
        ax,
        proj=PROJ,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        facecolor=OCEAN_COLOR,
        edgecolor="black",
        linewidth=0.75,
        zorder=0,
    )

    land_gdf.plot(ax=ax, color=LAND_COLOR, edgecolor="none", linewidth=0, zorder=1)
    dry_gdf.plot(ax=ax, color=DRY_COLOR, edgecolor="none", linewidth=0, alpha=0.92, zorder=2)

    for category, color in [
        ("Natural-dominant", NATURAL_COLOR),
        ("Residual-dominant", RESIDUAL_COLOR),
    ]:
        sub = dom_gdf.loc[dom_gdf["dominance"] == category]
        if not sub.empty:
            sub.plot(ax=ax, color=color, edgecolor="none", linewidth=0, alpha=1.0, zorder=3)

    land_gdf.boundary.plot(
        ax=ax,
        color=COAST_COLOR,
        linewidth=0.25,
        alpha=0.75,
        zorder=4,
    )

    for coll in ax.collections:
        coll.set_clip_path(map_frame)

    set_axis_to_patch_extent(ax, map_frame)
    ax.set_axis_off()

    # ---------------------------------------------------------------------
    # Compact legend
    # ---------------------------------------------------------------------
    leg_ax = fig.add_axes(LEG_AX)
    leg_ax.axis("off")
    handles = [
        Patch(facecolor=LAND_COLOR, edgecolor="#777777", linewidth=0.4, label="Non-dryland"),
        Patch(facecolor=DRY_COLOR, edgecolor="#777777", linewidth=0.4, label="Dryland"),
        Patch(facecolor=NATURAL_COLOR, edgecolor="#777777", linewidth=0.4, label="Natural-dominant"),
        Patch(facecolor=RESIDUAL_COLOR, edgecolor="#777777", linewidth=0.4, label="Residual-dominant"),
    ]
    leg_ax.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=7.6,
        handlelength=1.05,
        handleheight=0.8,
        borderaxespad=0,
        labelspacing=0.55,
        columnspacing=0.4,
    )

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    if EXPORT_TRANSPARENT:
        apply_transparent_background(fig)

    save_kwargs = dict(
        transparent=EXPORT_TRANSPARENT,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.savefig(OUT_PNG, dpi=450, **save_kwargs)
    fig.savefig(OUT_PDF, **save_kwargs)
    fig.savefig(OUT_SVG, **save_kwargs)
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_SVG}")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
