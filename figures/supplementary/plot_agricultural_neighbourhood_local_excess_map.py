#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agricultural-neighbourhood spatial-audit maps, refined v3.

Main refinements relative to v2
-------------------------------
1. Transparent export background (PNG/PDF/SVG).
2. Global local-excess figure now uses a centred horizontal colour bar below
   the map, with the explanatory note placed above the colour bar.
3. The local-excess map is made clearer by combining exact focal-hexagon
   polygons with a restrained centroid overlay for focal agricultural hexagons.
4. The global three-group map keeps exact polygons but adds a very subtle
   centroid overlay so agricultural and neighbour groups read slightly more
   clearly at world scale without the exaggerated "bubble map" effect.
5. Regional and global figures use transparent figure/axes backgrounds.

Expected input file and fields are the same as in v2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch
from pyproj import Transformer
from shapely.geometry import box


# =============================================================================
# 1. User settings
# =============================================================================

INPUT_GPKG = Path(r"data/processed/agricultural_neighbourhood/up_hex_spatial_audit_integrated.gpkg")
LAND_SHP = Path(r"world_map/ne_10m_land.shp")
DRY_SHP: Optional[Path] = Path(r"Drylands_latest_July2014/drylands_UNCCD_CBD_july2014.shp")
OUTPUT_DIR = Path(r"ag_spatial_audit_maps_v6")

PROJ = "ESRI:54030"  # Robinson
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -58.0, 85.0

DPI = 600
EXPORT_TRANSPARENT = True
FONT_FAMILY = "Arial"
EXPORT_PNG = True
EXPORT_PDF = True
EXPORT_SVG = True

# -----------------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------------

OCEAN_COLOR = "none"           # transparent map interior
LAND_COLOR = "#F5F5F5"
DRY_COLOR = "#EEEBDD"
COAST_COLOR = "#686868"
FRAME_COLOR = "#333333"
TEXT_COLOR = "#222222"
NOTE_COLOR = "#666666"

OTHER_UP_COLOR = "#D9D9D9"
AG_COLOR = "#D97904"
NEIGHBOR_COLOR = "#A5C9AE"

NO_NEIGHBOR_COLOR = "#BDBDBD"
LOCAL_EXCESS_CMAP = LinearSegmentedColormap.from_list(
    "local_excess_restrained",
    ["#B73B3B", "#F2B6A0", "#F7F7F7", "#A8CFE3", "#2C6AA0"],
    N=256,
)

# -----------------------------------------------------------------------------
# Visual hierarchy
# -----------------------------------------------------------------------------

OTHER_UP_ALPHA_GLOBAL = 0.18
AG_ALPHA_GLOBAL = 0.72
NEIGHBOR_ALPHA_GLOBAL = 0.34
LOCAL_EXCESS_ALPHA_GLOBAL = 0.72
NO_NEIGHBOR_ALPHA_GLOBAL = 0.65

OTHER_UP_ALPHA_REGIONAL = 0.20
AG_ALPHA_REGIONAL = 0.92
NEIGHBOR_ALPHA_REGIONAL = 0.92

FOCAL_EDGE_COLOR = "none"
FOCAL_EDGE_WIDTH = 0.0

# Global three-group overlay: slight visual assistance only.
USE_GROUP_CENTROID_OVERLAY = True
AG_GROUP_CENTROID_SIZE = 1.7
NEIGHBOR_GROUP_CENTROID_SIZE = 1.00
GROUP_CENTROID_ALPHA = 0.50

# Local-excess overlay: stronger than group map because the values are the key
# analytical message and otherwise can become too hard to see globally.
USE_LOCAL_EXCESS_CENTROIDS = True
LOCAL_EXCESS_POINT_SIZE = 3.8
LOCAL_EXCESS_POINT_EDGEWIDTH = 0.10
LOCAL_EXCESS_POINT_EDGE_COLOR = "#FFFFFF"
LOCAL_EXCESS_NO_NEIGHBOR_POINT_SIZE = 3.2

LAND_SIMPLIFY = 4500
DRY_SIMPLIFY = 4500
HEX_SIMPLIFY_GLOBAL = 500
HEX_SIMPLIFY_REGIONAL = 0

LOCAL_EXCESS_ABS_PERCENTILE = 95.0
SHOW_LOCAL_EXCESS_NOTE = True

# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------

GLOBAL_FIGSIZE = (10.8, 4.85)
GLOBAL_MAP_AX = [0.020, 0.205, 0.960, 0.765]
GLOBAL_LEGEND_AX = [0.055, 0.075, 0.890, 0.090]

LOCAL_FIGSIZE = (10.8, 5.15)
LOCAL_MAP_AX = [0.020, 0.255, 0.960, 0.715]
LOCAL_NOTE_AX = [0.25, 0.180, 0.50, 0.028]
LOCAL_CBAR_AX = [0.315, 0.090, 0.37, 0.040]
LOCAL_LEGEND_AX = [0.360, 0.132, 0.28, 0.040]

REGIONAL_FIGSIZE = (11.4, 7.35)
REGIONAL_PANEL_ASPECT = 1.55
REGIONAL_BOTTOM = 0.145
REGIONAL_TOP = 0.985
REGIONAL_LEFT = 0.035
REGIONAL_RIGHT = 0.985
REGIONAL_HSPACE = 0.22
REGIONAL_WSPACE = 0.08
REGION_LABEL_Y = -0.075

REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "Western North America": (-130.0, -95.0, 20.0, 58.0),
    "African dryland belt": (-20.0, 45.0, 3.0, 30.0),
    "West and Central Asia": (30.0, 105.0, 18.0, 58.0),
    "South American dry margins": (-82.0, -38.0, -42.0, 6.0),
}


# =============================================================================
# 2. General helpers
# =============================================================================


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 8.0,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "none",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
        }
    )


def ensure_inputs() -> None:
    required = [INPUT_GPKG, LAND_SHP]
    if DRY_SHP is not None:
        required.append(DRY_SHP)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files were not found:\n" + "\n".join(missing)
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "transparent": EXPORT_TRANSPARENT,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    if EXPORT_PNG:
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=DPI, **kwargs)
    if EXPORT_PDF:
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", **kwargs)
    if EXPORT_SVG:
        fig.savefig(OUTPUT_DIR / f"{stem}.svg", **kwargs)


def clip_to_lonlat_extent(
    gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    lon_min, lon_max, lat_min, lat_max = bounds
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS and cannot be clipped safely.")
    geo = gdf.to_crs("EPSG:4326")
    clipped = gpd.clip(geo, box(lon_min, lat_min, lon_max, lat_max))
    return clipped.loc[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()


def simplify_geometry(
    gdf: Optional[gpd.GeoDataFrame],
    tolerance: float,
) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or tolerance <= 0:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(tolerance=tolerance, preserve_topology=True)
    return out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()


def make_projection_boundary_patch(
    ax: plt.Axes,
    proj: str = PROJ,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    n: int = 500,
    facecolor: str = OCEAN_COLOR,
    edgecolor: str = FRAME_COLOR,
    linewidth: float = 0.70,
    zorder: int = 0,
) -> PathPatch:
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    lon_bottom = np.linspace(lon_min, lon_max, n)
    lat_bottom = np.full(n, lat_min)
    lat_right = np.linspace(lat_min, lat_max, n)
    lon_right = np.full(n, lon_max)
    lon_top = np.linspace(lon_max, lon_min, n)
    lat_top = np.full(n, lat_max)
    lat_left = np.linspace(lat_max, lat_min, n)
    lon_left = np.full(n, lon_min)

    lon = np.concatenate([lon_bottom, lon_right, lon_top, lon_left])
    lat = np.concatenate([lat_bottom, lat_right, lat_top, lat_left])
    x, y = transformer.transform(lon, lat)

    vertices = np.column_stack([x, y])
    vertices = np.vstack([vertices, vertices[0]])

    codes = np.full(len(vertices), MplPath.LINETO)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY

    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def set_axis_to_patch_extent(ax: plt.Axes, patch: PathPatch) -> None:
    vertices = patch.get_path().vertices
    xmin = float(np.nanmin(vertices[:, 0]))
    xmax = float(np.nanmax(vertices[:, 0]))
    ymin = float(np.nanmin(vertices[:, 1]))
    ymax = float(np.nanmax(vertices[:, 1]))
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - 0.008 * dx, xmax + 0.008 * dx)
    ax.set_ylim(ymin - 0.015 * dy, ymax + 0.015 * dy)


def build_centroid_points(up_geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        up_geo.drop(columns="geometry").copy(),
        geometry=gpd.points_from_xy(up_geo["cen_lon"], up_geo["cen_lat"]),
        crs="EPSG:4326",
    )
    return points.to_crs(PROJ)


def clip_axis_collections(ax: plt.Axes, patch: PathPatch) -> None:
    for collection in ax.collections:
        collection.set_clip_path(patch)


# =============================================================================
# 3. Data preparation
# =============================================================================


def prepare_global_layers():
    ensure_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    up = gpd.read_file(INPUT_GPKG)
    required_fields = {
        "hex_id",
        "cen_lon",
        "cen_lat",
        "audit_group3",
        "is_ag_focal",
        "n_nonag_up_neighbors",
        "local_excess_beta_obs",
        "local_excess_beta_obs_sign",
        "geometry",
    }
    missing_fields = sorted(required_fields - set(up.columns))
    if missing_fields:
        raise ValueError("INPUT_GPKG is missing required fields: " + ", ".join(missing_fields))

    expected_groups = {"Ag_self", "Ag_neighbor_nonAg", "Other_nonAg_UP"}
    actual_groups = set(up["audit_group3"].dropna().unique())
    unexpected = sorted(actual_groups - expected_groups)
    if unexpected:
        raise ValueError("Unexpected audit_group3 values: " + ", ".join(map(str, unexpected)))

    global_bounds = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    up_geo = clip_to_lonlat_extent(up, global_bounds)
    land_geo = clip_to_lonlat_extent(gpd.read_file(LAND_SHP), global_bounds)
    dry_geo = None
    if DRY_SHP is not None:
        dry_geo = clip_to_lonlat_extent(gpd.read_file(DRY_SHP), global_bounds)

    points = build_centroid_points(up_geo)
    up_proj = up_geo.to_crs(PROJ)
    land_proj = land_geo.to_crs(PROJ)
    dry_proj = dry_geo.to_crs(PROJ) if dry_geo is not None else None

    land_proj = simplify_geometry(land_proj, LAND_SIMPLIFY)
    dry_proj = simplify_geometry(dry_proj, DRY_SIMPLIFY)
    up_global = simplify_geometry(up_proj, HEX_SIMPLIFY_GLOBAL)

    return {
        "up_geo": up_geo,
        "up_proj_full": up_proj,
        "up_global": up_global,
        "points": points,
        "land_geo": land_geo,
        "land_proj": land_proj,
        "dry_geo": dry_geo,
        "dry_proj": dry_proj,
    }


# =============================================================================
# 4. Common plotting functions
# =============================================================================


def draw_global_background(ax: plt.Axes, land: gpd.GeoDataFrame, dry: Optional[gpd.GeoDataFrame]) -> PathPatch:
    frame = make_projection_boundary_patch(ax)

    land.plot(ax=ax, color=LAND_COLOR, edgecolor="none", linewidth=0, zorder=1)
    if dry is not None:
        dry.plot(ax=ax, color=DRY_COLOR, edgecolor="none", linewidth=0, alpha=0.86, zorder=2)

    land.boundary.plot(ax=ax, color=COAST_COLOR, linewidth=0.22, alpha=0.72, zorder=10)
    set_axis_to_patch_extent(ax, frame)
    ax.set_axis_off()
    return frame


def plot_group_polygons(ax: plt.Axes, up: gpd.GeoDataFrame, global_scale: bool) -> None:
    other = up.loc[up["audit_group3"] == "Other_nonAg_UP"]
    neighbour = up.loc[up["audit_group3"] == "Ag_neighbor_nonAg"]
    ag = up.loc[up["audit_group3"] == "Ag_self"]

    if global_scale:
        other_alpha, neighbour_alpha, ag_alpha = (
            OTHER_UP_ALPHA_GLOBAL,
            NEIGHBOR_ALPHA_GLOBAL,
            AG_ALPHA_GLOBAL,
        )
    else:
        other_alpha, neighbour_alpha, ag_alpha = (
            OTHER_UP_ALPHA_REGIONAL,
            NEIGHBOR_ALPHA_REGIONAL,
            AG_ALPHA_REGIONAL,
        )

    if not other.empty:
        other.plot(ax=ax, color=OTHER_UP_COLOR, edgecolor="none", linewidth=0, alpha=other_alpha, zorder=3)
    if not neighbour.empty:
        neighbour.plot(ax=ax, color=NEIGHBOR_COLOR, edgecolor="none", linewidth=0, alpha=neighbour_alpha, zorder=4)
    if not ag.empty:
        ag.plot(ax=ax, color=AG_COLOR, edgecolor=FOCAL_EDGE_COLOR, linewidth=FOCAL_EDGE_WIDTH, alpha=ag_alpha, zorder=6)


def add_group_centroid_overlay(ax: plt.Axes, points: gpd.GeoDataFrame) -> None:
    if not USE_GROUP_CENTROID_OVERLAY:
        return
    neighbour = points.loc[points["audit_group3"] == "Ag_neighbor_nonAg"]
    ag = points.loc[points["audit_group3"] == "Ag_self"]

    if not neighbour.empty:
        ax.scatter(
            neighbour.geometry.x,
            neighbour.geometry.y,
            s=NEIGHBOR_GROUP_CENTROID_SIZE,
            color=NEIGHBOR_COLOR,
            alpha=GROUP_CENTROID_ALPHA,
            linewidth=0,
            zorder=7,
        )
    if not ag.empty:
        ax.scatter(
            ag.geometry.x,
            ag.geometry.y,
            s=AG_GROUP_CENTROID_SIZE,
            color=AG_COLOR,
            alpha=min(0.72, GROUP_CENTROID_ALPHA + 0.06),
            linewidth=0,
            zorder=8,
        )


def add_three_group_legend(fig: plt.Figure, rect: Iterable[float]) -> None:
    legend_ax = fig.add_axes(rect)
    legend_ax.set_axis_off()
    handles = [
        Patch(facecolor=AG_COLOR, edgecolor="none", label="Agricultural-expansion focal UP"),
        Patch(facecolor=NEIGHBOR_COLOR, edgecolor="none", label="Non-agricultural UP in agricultural neighbourhoods"),
        Patch(facecolor=OTHER_UP_COLOR, edgecolor="none", label="Other non-agricultural UP"),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        handleheight=0.9,
        columnspacing=2.0,
        borderaxespad=0,
    )


# =============================================================================
# 5. Global three-group map
# =============================================================================


def plot_global_group_map(layers) -> None:
    up = layers["up_global"]
    points = layers["points"]
    land = layers["land_proj"]
    dry = layers["dry_proj"]

    fig = plt.figure(figsize=GLOBAL_FIGSIZE, facecolor="none")
    ax = fig.add_axes(GLOBAL_MAP_AX, facecolor="none")

    frame = draw_global_background(ax, land, dry)
    plot_group_polygons(ax, up, global_scale=True)
    add_group_centroid_overlay(ax, points)

    land.boundary.plot(ax=ax, color=COAST_COLOR, linewidth=0.22, alpha=0.72, zorder=10)
    clip_axis_collections(ax, frame)
    add_three_group_legend(fig, GLOBAL_LEGEND_AX)

    save_figure(fig, "ag_spatial_groups_global_restrained")
    plt.close(fig)


# =============================================================================
# 6. Global local-excess map
# =============================================================================


def calculate_symmetric_local_excess_limit(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("No finite local_excess_beta_obs values are available.")
    limit = float(np.nanpercentile(np.abs(finite.to_numpy()), LOCAL_EXCESS_ABS_PERCENTILE))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(finite.to_numpy())))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Unable to derive a positive local-excess colour limit.")
    return limit


def add_local_excess_point_overlay(
    ax: plt.Axes,
    points: gpd.GeoDataFrame,
    norm: TwoSlopeNorm,
) -> None:
    if not USE_LOCAL_EXCESS_CENTROIDS:
        return

    ag_points = points.loc[points["audit_group3"] == "Ag_self"].copy()
    local_points = ag_points.loc[ag_points["local_excess_beta_obs"].notna()].copy()
    no_neigh_points = ag_points.loc[ag_points["local_excess_beta_obs"].isna()].copy()

    if not no_neigh_points.empty:
        ax.scatter(
            no_neigh_points.geometry.x,
            no_neigh_points.geometry.y,
            s=LOCAL_EXCESS_NO_NEIGHBOR_POINT_SIZE,
            color=NO_NEIGHBOR_COLOR,
            alpha=0.95,
            linewidth=0,
            zorder=7,
        )

    if not local_points.empty:
        vals = local_points["local_excess_beta_obs"].to_numpy()
        ax.scatter(
            local_points.geometry.x,
            local_points.geometry.y,
            c=vals,
            cmap=LOCAL_EXCESS_CMAP,
            norm=norm,
            s=LOCAL_EXCESS_POINT_SIZE,
            alpha=0.95,
            linewidth=LOCAL_EXCESS_POINT_EDGEWIDTH,
            edgecolors=LOCAL_EXCESS_POINT_EDGE_COLOR,
            zorder=8,
        )


def plot_global_local_excess_map(layers) -> None:
    up = layers["up_global"]
    points = layers["points"]
    land = layers["land_proj"]
    dry = layers["dry_proj"]

    ag = up.loc[up["audit_group3"] == "Ag_self"].copy()
    local = ag.loc[ag["local_excess_beta_obs"].notna()].copy()
    no_neighbour = ag.loc[ag["local_excess_beta_obs"].isna()].copy()

    colour_limit = calculate_symmetric_local_excess_limit(local["local_excess_beta_obs"])
    norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0.0, vmax=colour_limit)

    fig = plt.figure(figsize=LOCAL_FIGSIZE, facecolor="none")
    ax = fig.add_axes(LOCAL_MAP_AX, facecolor="none")
    frame = draw_global_background(ax, land, dry)

    # Very light contextual background for all UP hexagons.
    up.plot(ax=ax, color=OTHER_UP_COLOR, edgecolor="none", linewidth=0, alpha=0.08, zorder=3)

    if not no_neighbour.empty:
        no_neighbour.plot(
            ax=ax,
            color=NO_NEIGHBOR_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=NO_NEIGHBOR_ALPHA_GLOBAL,
            zorder=4,
        )

    if not local.empty:
        local.plot(
            ax=ax,
            column="local_excess_beta_obs",
            cmap=LOCAL_EXCESS_CMAP,
            norm=norm,
            edgecolor="none",
            linewidth=0,
            alpha=LOCAL_EXCESS_ALPHA_GLOBAL,
            zorder=5,
        )

    # A restrained point overlay improves global readability without replacing
    # the true polygon layer.
    add_local_excess_point_overlay(ax, points, norm)

    land.boundary.plot(ax=ax, color=COAST_COLOR, linewidth=0.22, alpha=0.72, zorder=10)
    clip_axis_collections(ax, frame)

    # Explanatory note ABOVE the colour bar.
    if SHOW_LOCAL_EXCESS_NOTE:
        note_ax = fig.add_axes(LOCAL_NOTE_AX, facecolor="none")
        note_ax.set_axis_off()
        note_ax.text(
            0.5,
            0.5,
            f"Colours clipped at the {LOCAL_EXCESS_ABS_PERCENTILE:.0f}th percentile of absolute local excess; exact hexagon polygons are retained.",
            ha="center",
            va="center",
            fontsize=7.0,
            color=NOTE_COLOR,
        )

    # Centered horizontal colour bar.
    cax = fig.add_axes(LOCAL_CBAR_AX, facecolor="none")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=LOCAL_EXCESS_CMAP)
    scalar.set_array([])
    colour_bar = fig.colorbar(scalar, cax=cax, orientation="horizontal", extend="both")
    colour_bar.set_label("Local excess in observed expansion trend", fontsize=8.0, labelpad=3)
    colour_bar.ax.tick_params(labelsize=7.2, length=2.3, width=0.55, pad=2)
    colour_bar.outline.set_linewidth(0.55)
    colour_bar.outline.set_edgecolor("#555555")

    # Left-side discrete legend below map.
    legend_ax = fig.add_axes(LOCAL_LEGEND_AX, facecolor="none")
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[Patch(facecolor=NO_NEIGHBOR_COLOR, edgecolor="none", label="No non-agricultural UP neighbours")],
        loc="center",
        frameon=False,
        fontsize=7.8,
        handlelength=1.6,
        borderaxespad=0,
    )

    save_figure(fig, "ag_local_excess_global_restrained")
    plt.close(fig)


# =============================================================================
# 7. Aligned regional polygon maps
# =============================================================================


def projected_bounds_from_lonlat(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    proj: str = PROJ,
    n: int = 250,
) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    lon_bottom = np.linspace(lon_min, lon_max, n)
    lat_bottom = np.full(n, lat_min)
    lat_right = np.linspace(lat_min, lat_max, n)
    lon_right = np.full(n, lon_max)
    lon_top = np.linspace(lon_max, lon_min, n)
    lat_top = np.full(n, lat_max)
    lat_left = np.linspace(lat_max, lat_min, n)
    lon_left = np.full(n, lon_min)

    lon = np.concatenate([lon_bottom, lon_right, lon_top, lon_left])
    lat = np.concatenate([lat_bottom, lat_right, lat_top, lat_left])
    x, y = transformer.transform(lon, lat)
    return (
        float(np.nanmin(x)),
        float(np.nanmin(y)),
        float(np.nanmax(x)),
        float(np.nanmax(y)),
    )


def expand_bounds_to_aspect(
    bounds: Tuple[float, float, float, float],
    target_aspect: float,
    padding_fraction: float = 0.035,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid projected bounds: {bounds}")

    centre_x = 0.5 * (xmin + xmax)
    centre_y = 0.5 * (ymin + ymax)
    current_aspect = width / height
    if current_aspect < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect

    width *= 1.0 + 2.0 * padding_fraction
    height *= 1.0 + 2.0 * padding_fraction
    return (
        centre_x - 0.5 * width,
        centre_y - 0.5 * height,
        centre_x + 0.5 * width,
        centre_y + 0.5 * height,
    )


def clip_projected_to_bounds(
    gdf: Optional[gpd.GeoDataFrame],
    bounds: Tuple[float, float, float, float],
) -> Optional[gpd.GeoDataFrame]:
    if gdf is None:
        return None
    xmin, ymin, xmax, ymax = bounds
    clipped = gpd.clip(gdf, box(xmin, ymin, xmax, ymax))
    return clipped.loc[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()


def style_regional_axis(ax: plt.Axes, bounds: Tuple[float, float, float, float], region_name: str) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0 / REGIONAL_PANEL_ASPECT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)

    ax.text(
        0.5,
        REGION_LABEL_Y,
        region_name,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.4,
        color=TEXT_COLOR,
        clip_on=False,
    )


def plot_regional_maps(layers) -> None:
    up_full = layers["up_proj_full"]
    land_full = layers["land_geo"].to_crs(PROJ)
    dry_full = layers["dry_geo"].to_crs(PROJ) if layers["dry_geo"] is not None else None

    up_full = simplify_geometry(up_full, HEX_SIMPLIFY_REGIONAL)
    land_full = simplify_geometry(land_full, LAND_SIMPLIFY)
    dry_full = simplify_geometry(dry_full, DRY_SIMPLIFY)

    fig, axes = plt.subplots(2, 2, figsize=REGIONAL_FIGSIZE, facecolor="none")
    axes_flat = axes.ravel()
    summary_rows = []

    for ax, (region_name, lonlat_bounds) in zip(axes_flat, REGIONS.items()):
        projected_raw = projected_bounds_from_lonlat(*lonlat_bounds)
        projected_view = expand_bounds_to_aspect(projected_raw, target_aspect=REGIONAL_PANEL_ASPECT, padding_fraction=0.025)

        region_land = clip_projected_to_bounds(land_full, projected_view)
        region_dry = clip_projected_to_bounds(dry_full, projected_view)
        region_up = clip_projected_to_bounds(up_full, projected_view)

        if region_land is not None and not region_land.empty:
            region_land.plot(ax=ax, color=LAND_COLOR, edgecolor="none", linewidth=0, zorder=1)
        if region_dry is not None and not region_dry.empty:
            region_dry.plot(ax=ax, color=DRY_COLOR, edgecolor="none", linewidth=0, alpha=0.84, zorder=2)
        if region_up is not None and not region_up.empty:
            plot_group_polygons(ax, region_up, global_scale=False)
        if region_land is not None and not region_land.empty:
            region_land.boundary.plot(ax=ax, color=COAST_COLOR, linewidth=0.25, alpha=0.72, zorder=10)

        style_regional_axis(ax, projected_view, region_name)

        if region_up is None or region_up.empty:
            n_ag = n_neighbour = n_other = 0
        else:
            n_ag = int((region_up["audit_group3"] == "Ag_self").sum())
            n_neighbour = int((region_up["audit_group3"] == "Ag_neighbor_nonAg").sum())
            n_other = int((region_up["audit_group3"] == "Other_nonAg_UP").sum())

        summary_rows.append(
            {
                "region": region_name,
                "ag_focal_up": n_ag,
                "neighbouring_nonag_up": n_neighbour,
                "other_nonag_up": n_other,
                "input_lon_min": lonlat_bounds[0],
                "input_lon_max": lonlat_bounds[1],
                "input_lat_min": lonlat_bounds[2],
                "input_lat_max": lonlat_bounds[3],
            }
        )

    fig.subplots_adjust(
        left=REGIONAL_LEFT,
        right=REGIONAL_RIGHT,
        bottom=REGIONAL_BOTTOM,
        top=REGIONAL_TOP,
        hspace=REGIONAL_HSPACE,
        wspace=REGIONAL_WSPACE,
    )

    legend_ax = fig.add_axes([0.09, 0.025, 0.82, 0.070], facecolor="none")
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[
            Patch(facecolor=AG_COLOR, edgecolor="none", label="Agricultural-expansion focal UP"),
            Patch(facecolor=NEIGHBOR_COLOR, edgecolor="none", label="Non-agricultural UP in agricultural neighbourhoods"),
            Patch(facecolor=OTHER_UP_COLOR, edgecolor="none", label="Other non-agricultural UP"),
        ],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        columnspacing=2.0,
        borderaxespad=0,
    )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "ag_regional_panel_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_figure(fig, "ag_spatial_groups_regional_aligned")
    plt.close(fig)


# =============================================================================
# 8. Main
# =============================================================================


def main() -> None:
    configure_matplotlib()
    layers = prepare_global_layers()
    plot_global_group_map(layers)
    plot_global_local_excess_map(layers)
    plot_regional_maps(layers)

    print("Finished. Outputs saved in:")
    print(OUTPUT_DIR.resolve())
    print("\nMain map stems:")
    print("  ag_spatial_groups_global_restrained")
    print("  ag_local_excess_global_restrained")
    print("  ag_spatial_groups_regional_aligned")
    print("\nRegional counts:")
    print("  ag_regional_panel_counts.csv")


if __name__ == "__main__":
    main()
