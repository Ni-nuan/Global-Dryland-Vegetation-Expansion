#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined export script for the beta_res map panel.

Updates relative to the original version:
- map frame aligned with the revised Fig. 1a / Fig. 2g style;
- Antarctica removed for a cleaner journal-style composition;
- no artificial ellipse frame;
- transparent PNG plus vector PDF and SVG outputs;
- editable text in SVG/PDF;
- robust path handling based on script location.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from pyproj import Transformer
from shapely.geometry import box


# =============================================================================
# User-defined paths
# =============================================================================
# Keep paths relative to the current working directory, consistent with the
# earlier scripts. Run the script from your project folder.
ATTR_XLSX = Path(r'trend_outputs_xco2_vpd_resid/06_trend_attribution_by_hex_frac_UP.xlsx')
HEX_SHP = Path(r'hex_data/NDVI_trend_hex_100.shp')
LAND_SHP = Path(r'world_map/ne_10m_land.shp')
DRY_SHP = Path(r'Drylands_latest_July2014/drylands_UNCCD_CBD_july2014.shp')

OUT_PNG = Path(r'supp_probability_scale_residual_map_vpd_resid.png')
OUT_PDF = Path(r'supp_probability_scale_residual_map_vpd_resid.pdf')
OUT_SVG = Path(r'supp_probability_scale_residual_map_vpd_resid.svg')
OUT_CSV = Path(r'supp_probability_scale_residual_map_vpd_resid.csv')


# =============================================================================
# Styling / layout
# =============================================================================
PROJ = 'ESRI:54030'   # Robinson
LON_MIN, LON_MAX = -180, 180
LAT_MIN, LAT_MAX = -58, 85

OCEAN_COLOR = '#FFFFFF'
LAND_COLOR = '#F4F4F4'
DRY_COLOR = '#EDE9D8'
COAST_COLOR = '#404040'
TEXT_COLOR = '#222222'

LAND_SIMPLIFY = 4500
DRY_SIMPLIFY = 4500
HEX_SIMPLIFY = 1000

FIGSIZE = (10.0, 4.0)
MAP_AX = [0.02, 0.15, 0.88, 0.80]
LEG_AX = [0.91, 0.43, 0.08, 0.16]
CBAR_AX = [0.31, 0.07, 0.28, 0.035]
DPI = 300

CMAP = LinearSegmentedColormap.from_list(
    'beta_res_div',
    ['#9E1B32', '#D99A8A', '#F7F7F7', '#9ECAE1', '#2166AC'],
    N=256,
)


# =============================================================================
# Helpers
# =============================================================================
def apply_rcparams() -> None:
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.size': 9.5,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    })


def clip_to_display_extent(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326', allow_override=True)
    gdf = gdf.to_crs('EPSG:4326')
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
    edgecolor: str = 'black',
    linewidth: float = 0.75,
    zorder: int = 0,
) -> PathPatch:
    transformer = Transformer.from_crs('EPSG:4326', proj, always_xy=True)

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
        joinstyle='round',
        capstyle='round',
    )
    ax.add_patch(patch)
    return patch


def set_axis_to_patch_extent(ax, patch: PathPatch, pad_x_frac: float = 0.01, pad_y_frac: float = 0.02) -> None:
    verts = patch.get_path().vertices
    xmin, ymin = np.nanmin(verts[:, 0]), np.nanmin(verts[:, 1])
    xmax, ymax = np.nanmax(verts[:, 0]), np.nanmax(verts[:, 1])

    pad_x = (xmax - xmin) * pad_x_frac
    pad_y = (ymax - ymin) * pad_y_frac
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    apply_rcparams()

    attr = pd.read_excel(ATTR_XLSX)
    hex_gdf = gpd.read_file(HEX_SHP)
    land_gdf = gpd.read_file(LAND_SHP)
    dry_gdf = gpd.read_file(DRY_SHP)

    if hex_gdf.crs is None:
        hex_gdf = hex_gdf.set_crs('EPSG:4326', allow_override=True)

    gdf = hex_gdf[['hex_id', 'geometry']].merge(
        attr[['hex_id', 'beta_res_co2']], on='hex_id', how='inner'
    ).dropna(subset=['beta_res_co2']).copy()

    pd.DataFrame({
        'hex_id': gdf['hex_id'].astype(str),
        'beta_res_co2': gdf['beta_res_co2']
    }).to_csv(OUT_CSV, index=False)

    land_gdf = clip_to_display_extent(land_gdf)
    dry_gdf = clip_to_display_extent(dry_gdf)
    gdf = clip_to_display_extent(gdf)

    land_gdf = land_gdf.to_crs(PROJ)
    dry_gdf = dry_gdf.to_crs(PROJ)
    gdf = gdf.to_crs(PROJ)

    land_gdf['geometry'] = land_gdf.geometry.simplify(LAND_SIMPLIFY, preserve_topology=True)
    dry_gdf['geometry'] = dry_gdf.geometry.simplify(DRY_SIMPLIFY, preserve_topology=True)
    gdf['geometry'] = gdf.geometry.simplify(HEX_SIMPLIFY, preserve_topology=True)

    vals = gdf['beta_res_co2'].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(vals), 98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor='white')
    fig.patch.set_alpha(0)

    ax = fig.add_axes(MAP_AX)
    ax.set_facecolor('none')

    map_frame = make_projection_boundary_patch(
        ax,
        proj=PROJ,
        lon_min=LON_MIN,
        lon_max=LON_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        facecolor=OCEAN_COLOR,
        edgecolor='black',
        linewidth=0.75,
        zorder=0,
    )

    land_gdf.plot(ax=ax, color=LAND_COLOR, edgecolor='none', linewidth=0, zorder=1)
    dry_gdf.plot(ax=ax, color=DRY_COLOR, edgecolor='none', linewidth=0, alpha=0.95, zorder=2)
    gdf.plot(
        ax=ax,
        column='beta_res_co2',
        cmap=CMAP,
        norm=norm,
        edgecolor='none',
        linewidth=0,
        alpha=1.0,
        zorder=3,
    )
    land_gdf.boundary.plot(ax=ax, color=COAST_COLOR, linewidth=0.25, alpha=0.75, zorder=4)

    for coll in ax.collections:
        coll.set_clip_path(map_frame)

    set_axis_to_patch_extent(ax, map_frame)
    ax.set_axis_off()

    # background-class legend
    leg_ax = fig.add_axes(LEG_AX)
    leg_ax.axis('off')
    handles = [
        Patch(facecolor=LAND_COLOR, edgecolor='#777777', linewidth=0.4, label='Non-dryland'),
        Patch(facecolor=DRY_COLOR, edgecolor='#777777', linewidth=0.4, label='Dryland'),
    ]
    leg_ax.legend(handles=handles, loc='center left', frameon=False, fontsize=7.8, handlelength=1.3)

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cax = fig.add_axes(CBAR_AX)
    cax.set_facecolor('none')
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.7)
    cb.ax.tick_params(labelsize=8.2, length=2.5, width=0.7, colors=TEXT_COLOR)
    cb.set_label(r'Residual trend component ($\beta_{res}$)', fontsize=9.5, labelpad=3)

    fig.savefig(OUT_PNG, dpi=DPI, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT_PDF, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT_SVG, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


if __name__ == '__main__':
    main()
