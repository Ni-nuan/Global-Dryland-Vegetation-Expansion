#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agricultural-neighbourhood spatial-audit maps
================================================

This script implements a three-level visualisation strategy:

1. Global native high-DPI map
   - exact 100-km2 UP hexagon polygons
   - no centroid enlargement or other display amplification

2. Global spatially aggregated display map
   - agricultural focal and neighbouring non-agricultural UP hexagons are
     aggregated to a reproducible equal-area display grid
   - this reduces visual overlap while retaining broad global patterns

3. Regional native-detail panels
   - exact 100-km2 polygons are retained for local interpretation

The global land and dryland backgrounds are explicitly loaded from LAND_SHP
and DRY_SHP. All path settings are relative to the directory containing this
script unless an absolute path is supplied.

Required fields in INPUT_GPKG
-----------------------------
hex_id
audit_group3
is_ag_focal
is_nonag_neighbor_of_ag
geometry

Expected values in audit_group3
-------------------------------
Ag_self
Ag_neighbor_nonAg
Other_nonAg_UP
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch
from pyproj import Transformer
from shapely.geometry import box

try:
    # Shapely 2.x
    from shapely import make_valid as _shapely_make_valid
except ImportError:
    try:
        # Some Shapely 1.8 builds
        from shapely.validation import make_valid as _shapely_make_valid
    except ImportError:
        _shapely_make_valid = None


# =============================================================================
# 1. User paths
# =============================================================================

# Paths are resolved relative to the repository root unless absolute.
INPUT_GPKG = Path(
    r"data/processed/agricultural_neighbourhood/up_hex_spatial_audit_integrated.gpkg"
)
LAND_SHP = Path(r"data/external/natural_earth/ne_10m_land.shp")
DRY_SHP: Optional[Path] = Path(
    r"data/external/drylands/drylands_UNCCD_CBD_july2014.shp"
)
OUTPUT_DIR = Path(r"outputs/agricultural_neighbourhood/spatial_audit")


# =============================================================================
# 2. Projection and export settings
# =============================================================================

PROJ = "ESRI:54030"  # Robinson, consistent with the previous map workflow
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -58.0, 85.0

NATIVE_DPI = 900
CLUSTER_DPI = 700
REGIONAL_DPI = 700

EXPORT_PNG = True
EXPORT_PDF = True
EXPORT_SVG = True
EXPORT_TRANSPARENT = True
FONT_FAMILY = "Arial"


# =============================================================================
# 3. Colours and visual hierarchy
# =============================================================================

OCEAN_COLOR = "none"  # transparent background outside land polygons
LAND_COLOR = "#F5F5F5"
DRY_COLOR = "#EEEBDD"
COAST_COLOR = "#686868"
FRAME_COLOR = "#333333"
TEXT_COLOR = "#222222"

OTHER_UP_COLOR = "#D9D9D9"
AG_COLOR = "#D97904"
NEIGHBOR_COLOR = "#9FC7A9"
MIXED_COLOR = "#A99A65"

# Native exact-polygon map
OTHER_UP_ALPHA_NATIVE = 0.26
AG_ALPHA_NATIVE = 0.96
NEIGHBOR_ALPHA_NATIVE = 0.92

# Regional exact-polygon panels
OTHER_UP_ALPHA_REGIONAL = 0.24
AG_ALPHA_REGIONAL = 0.96
NEIGHBOR_ALPHA_REGIONAL = 0.92

# Background simplification in projected metres
LAND_SIMPLIFY = 4500
DRY_SIMPLIFY = 4500
HEX_SIMPLIFY_GLOBAL = 0  # exact hexagons in the native global output
HEX_SIMPLIFY_REGIONAL = 0


# =============================================================================
# 4. Aggregated-display settings
# =============================================================================

# The display grid is not a new analytical unit. It is used only to render the
# broad global pattern without enlarged and overlapping individual symbols.
DISPLAY_GRID_KM = 180
DISPLAY_GRID_M = DISPLAY_GRID_KM * 1000.0

# Squares are inset within each display-grid cell to leave visual separation.
DISPLAY_CELL_FILL_RATIO = 0.72

# Keep all occupied cells by default. Set to 2 to omit isolated single records.
MIN_TOTAL_COUNT_PER_DISPLAY_CELL = 1

# Cells containing both groups are shown as a distinct mixed category.
USE_MIXED_CATEGORY = True

# Alpha varies with the log of the number of source hexagons in each cell.
DISPLAY_ALPHA_MIN = 0.48
DISPLAY_ALPHA_MAX = 0.94

# A faint layer of other non-agricultural UP hexagons provides context.
OTHER_UP_ALPHA_CLUSTERED = 0.12


# =============================================================================
# 5. Figure layouts
# =============================================================================

GLOBAL_FIGSIZE = (11.2, 5.15)
GLOBAL_MAP_AX = [0.020, 0.205, 0.960, 0.765]
GLOBAL_LEGEND_AX = [0.055, 0.075, 0.890, 0.090]

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
# 6. General helpers
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    """Resolve a user path relative to the repository root unless absolute."""
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def validate_inputs() -> Tuple[Path, Path, Optional[Path], Path]:
    input_gpkg = resolve_path(INPUT_GPKG)
    land_shp = resolve_path(LAND_SHP)
    dry_shp = resolve_path(DRY_SHP)
    output_dir = resolve_path(OUTPUT_DIR)

    assert input_gpkg is not None
    assert land_shp is not None
    assert output_dir is not None

    required = [input_gpkg, land_shp]
    if dry_shp is not None:
        required.append(dry_shp)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = (
            "The following required files were not found:\n"
            + "\n".join(missing)
            + "\n\nPlace this script beside the folders referenced in the path "
              "settings, or replace the paths with absolute local paths."
        )
        raise FileNotFoundError(message)

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_gpkg, land_shp, dry_shp, output_dir


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    kwargs = {
        "transparent": EXPORT_TRANSPARENT,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    }
    if EXPORT_PNG:
        fig.savefig(output_dir / f"{stem}.png", dpi=dpi, **kwargs)
    if EXPORT_PDF:
        fig.savefig(output_dir / f"{stem}.pdf", **kwargs)
    if EXPORT_SVG:
        fig.savefig(output_dir / f"{stem}.svg", **kwargs)


def _repair_one_geometry(geometry):
    """Repair one geometry without assuming a specific Shapely version."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        return geometry

    repaired = None
    if _shapely_make_valid is not None:
        try:
            repaired = _shapely_make_valid(geometry)
        except Exception:
            repaired = None

    if repaired is None or repaired.is_empty:
        try:
            repaired = geometry.buffer(0)
        except Exception:
            return None

    # A second buffer(0) is a conservative fallback when make_valid returns an
    # object that remains invalid in an older GEOS build.
    if repaired is not None and not repaired.is_empty and not repaired.is_valid:
        try:
            repaired = repaired.buffer(0)
        except Exception:
            return None

    return repaired if repaired is not None and not repaired.is_empty else None


def repair_geometries(
    gdf: Optional[gpd.GeoDataFrame],
    layer_name: str = "layer",
) -> Optional[gpd.GeoDataFrame]:
    """Repair invalid geometries and remove null/empty records safely."""
    if gdf is None:
        return None

    out = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if out.empty:
        return out

    invalid = ~out.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"Repairing {n_invalid:,} invalid geometries in {layer_name}...")
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].apply(
            _repair_one_geometry
        )
        out = out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()

    return out


def bbox_subset(
    gdf: Optional[gpd.GeoDataFrame],
    bounds: Tuple[float, float, float, float],
) -> Optional[gpd.GeoDataFrame]:
    """Select features whose bounding boxes intersect the requested extent.

    This deliberately avoids geometric intersection. For map panels, Matplotlib
    clips the selected polygons to the axis limits, so an expensive and fragile
    ``gpd.clip`` operation is unnecessary. This also avoids GEOS topology errors
    caused by invalid multipart polygons in some dryland shapefiles.
    """
    if gdf is None:
        return None
    xmin, ymin, xmax, ymax = bounds
    geom_bounds = gdf.geometry.bounds
    mask = (
        (geom_bounds["maxx"] >= xmin)
        & (geom_bounds["minx"] <= xmax)
        & (geom_bounds["maxy"] >= ymin)
        & (geom_bounds["miny"] <= ymax)
    )
    return gdf.loc[mask].copy()


def clip_to_lonlat_extent(
    gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float],
    layer_name: str = "layer",
) -> gpd.GeoDataFrame:
    """Select a longitude/latitude extent without geometric intersection."""
    lon_min, lon_max, lat_min, lat_max = bounds
    if gdf.crs is None:
        raise ValueError("An input layer has no CRS and cannot be projected safely.")

    geo = gdf.to_crs("EPSG:4326")
    geo = repair_geometries(geo, layer_name=layer_name)
    assert geo is not None
    selected = bbox_subset(geo, (lon_min, lat_min, lon_max, lat_max))
    assert selected is not None
    return selected


def simplify_geometry(
    gdf: Optional[gpd.GeoDataFrame],
    tolerance: float,
) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or tolerance <= 0:
        return gdf

    out = gdf.copy()
    valid = out.geometry.notna() & ~out.geometry.is_empty & out.geometry.is_valid
    if valid.any():
        out.loc[valid, "geometry"] = out.loc[valid].geometry.simplify(
            tolerance=tolerance,
            preserve_topology=True,
        )
    return out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()


def make_projection_boundary_patch(
    ax: plt.Axes,
    n: int = 500,
    facecolor: str = OCEAN_COLOR,
    edgecolor: str = FRAME_COLOR,
    linewidth: float = 0.70,
) -> PathPatch:
    """Create the Robinson map frame and use it as a clipping boundary."""
    transformer = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)

    lon_bottom = np.linspace(LON_MIN, LON_MAX, n)
    lat_bottom = np.full(n, LAT_MIN)
    lat_right = np.linspace(LAT_MIN, LAT_MAX, n)
    lon_right = np.full(n, LON_MAX)
    lon_top = np.linspace(LON_MAX, LON_MIN, n)
    lat_top = np.full(n, LAT_MAX)
    lat_left = np.linspace(LAT_MAX, LAT_MIN, n)
    lon_left = np.full(n, LON_MIN)

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
        zorder=0,
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


def clip_axis_collections(ax: plt.Axes, patch: PathPatch) -> None:
    for collection in ax.collections:
        collection.set_clip_path(patch)


# =============================================================================
# 7. Data preparation
# =============================================================================


def prepare_layers(
    input_gpkg: Path,
    land_shp: Path,
    dry_shp: Optional[Path],
) -> Dict[str, Optional[gpd.GeoDataFrame]]:
    global_bounds = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    up = gpd.read_file(input_gpkg)
    required_fields = {
        "hex_id",
        "audit_group3",
        "is_ag_focal",
        "is_nonag_neighbor_of_ag",
        "geometry",
    }
    missing_fields = sorted(required_fields - set(up.columns))
    if missing_fields:
        raise ValueError(
            "INPUT_GPKG is missing required fields: "
            + ", ".join(missing_fields)
        )

    # Preserve the existing group field but recompute a robust plotting class.
    up["is_ag_focal"] = up["is_ag_focal"].fillna(False).astype(bool)
    up["is_nonag_neighbor_of_ag"] = (
        up["is_nonag_neighbor_of_ag"].fillna(False).astype(bool)
    )
    up["plot_group"] = "Other_nonAg_UP"
    up.loc[up["is_nonag_neighbor_of_ag"], "plot_group"] = "Neighbor_nonAg"
    up.loc[up["is_ag_focal"], "plot_group"] = "Ag_focal"

    up_geo = clip_to_lonlat_extent(up, global_bounds, layer_name="UP hexagons")
    land_geo = clip_to_lonlat_extent(
        gpd.read_file(land_shp), global_bounds, layer_name="land background"
    )

    dry_geo: Optional[gpd.GeoDataFrame] = None
    if dry_shp is not None:
        dry_geo = clip_to_lonlat_extent(
            gpd.read_file(dry_shp), global_bounds, layer_name="dryland background"
        )

    up_proj = repair_geometries(up_geo.to_crs(PROJ), layer_name="projected UP hexagons")
    land_proj = repair_geometries(land_geo.to_crs(PROJ), layer_name="projected land background")
    dry_proj = (
        repair_geometries(dry_geo.to_crs(PROJ), layer_name="projected dryland background")
        if dry_geo is not None
        else None
    )
    assert up_proj is not None and land_proj is not None

    land_proj = simplify_geometry(land_proj, LAND_SIMPLIFY)
    dry_proj = simplify_geometry(dry_proj, DRY_SIMPLIFY)
    up_native = simplify_geometry(up_proj, HEX_SIMPLIFY_GLOBAL)

    return {
        "up_geo": up_geo,
        "up_proj_full": up_proj,
        "up_native": up_native,
        "land_geo": land_geo,
        "land_proj": land_proj,
        "dry_geo": dry_geo,
        "dry_proj": dry_proj,
    }


# =============================================================================
# 8. Shared background and legends
# =============================================================================


def draw_global_background(
    ax: plt.Axes,
    land: gpd.GeoDataFrame,
    dry: Optional[gpd.GeoDataFrame],
) -> PathPatch:
    frame = make_projection_boundary_patch(ax)

    land.plot(
        ax=ax,
        color=LAND_COLOR,
        edgecolor="none",
        linewidth=0,
        zorder=1,
    )

    if dry is not None:
        dry.plot(
            ax=ax,
            color=DRY_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=0.88,
            zorder=2,
        )

    land.boundary.plot(
        ax=ax,
        color=COAST_COLOR,
        linewidth=0.22,
        alpha=0.72,
        zorder=20,
    )

    set_axis_to_patch_extent(ax, frame)
    ax.set_axis_off()
    return frame


def add_native_legend(fig: plt.Figure, rect: Iterable[float]) -> None:
    legend_ax = fig.add_axes(rect, facecolor="none")
    legend_ax.set_axis_off()
    handles = [
        Patch(
            facecolor=AG_COLOR,
            edgecolor="none",
            label="Agricultural-expansion focal UP",
        ),
        Patch(
            facecolor=NEIGHBOR_COLOR,
            edgecolor="none",
            label="Non-agricultural UP in agricultural neighbourhoods",
        ),
        Patch(
            facecolor=OTHER_UP_COLOR,
            edgecolor="none",
            label="Other non-agricultural UP",
        ),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        columnspacing=2.0,
        borderaxespad=0,
    )


# =============================================================================
# 9. Global native high-DPI output
# =============================================================================


def plot_global_native(
    layers: Dict[str, Optional[gpd.GeoDataFrame]],
    output_dir: Path,
) -> None:
    up = layers["up_native"]
    land = layers["land_proj"]
    dry = layers["dry_proj"]
    assert up is not None and land is not None

    fig = plt.figure(figsize=GLOBAL_FIGSIZE, facecolor="none")
    ax = fig.add_axes(GLOBAL_MAP_AX, facecolor="none")
    frame = draw_global_background(ax, land, dry)

    other = up.loc[up["plot_group"] == "Other_nonAg_UP"]
    neighbour = up.loc[up["plot_group"] == "Neighbor_nonAg"]
    ag = up.loc[up["plot_group"] == "Ag_focal"]

    if not other.empty:
        other.plot(
            ax=ax,
            color=OTHER_UP_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=OTHER_UP_ALPHA_NATIVE,
            zorder=3,
        )
    if not neighbour.empty:
        neighbour.plot(
            ax=ax,
            color=NEIGHBOR_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=NEIGHBOR_ALPHA_NATIVE,
            zorder=4,
        )
    if not ag.empty:
        ag.plot(
            ax=ax,
            color=AG_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=AG_ALPHA_NATIVE,
            zorder=5,
        )

    land.boundary.plot(
        ax=ax,
        color=COAST_COLOR,
        linewidth=0.22,
        alpha=0.72,
        zorder=20,
    )
    clip_axis_collections(ax, frame)
    add_native_legend(fig, GLOBAL_LEGEND_AX)

    save_figure(
        fig,
        output_dir,
        "ag_spatial_groups_global_native_highdpi",
        NATIVE_DPI,
    )
    plt.close(fig)


# =============================================================================
# 10. Reproducible aggregated global display
# =============================================================================


def build_display_grid(
    up_proj: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    relevant = up_proj.loc[
        up_proj["plot_group"].isin(["Ag_focal", "Neighbor_nonAg"])
    ].copy()
    if relevant.empty:
        raise ValueError("No agricultural focal or neighbouring non-Ag UP hexagons found.")

    centres = relevant.geometry.centroid
    relevant["display_x"] = centres.x
    relevant["display_y"] = centres.y

    # Anchor the display grid to the global Robinson projection rather than to
    # the minimum occupied point, ensuring reproducibility across subsets.
    transformer = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
    anchor_x, anchor_y = transformer.transform(LON_MIN, LAT_MIN)

    relevant["display_ix"] = np.floor(
        (relevant["display_x"] - anchor_x) / DISPLAY_GRID_M
    ).astype(int)
    relevant["display_iy"] = np.floor(
        (relevant["display_y"] - anchor_y) / DISPLAY_GRID_M
    ).astype(int)

    counts = (
        relevant.groupby(["display_ix", "display_iy", "plot_group"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for field in ["Ag_focal", "Neighbor_nonAg"]:
        if field not in counts.columns:
            counts[field] = 0

    counts["total_count"] = counts["Ag_focal"] + counts["Neighbor_nonAg"]
    counts = counts.loc[
        counts["total_count"] >= MIN_TOTAL_COUNT_PER_DISPLAY_CELL
    ].copy()

    def classify_cell(row: pd.Series) -> str:
        n_ag = int(row["Ag_focal"])
        n_neighbour = int(row["Neighbor_nonAg"])
        if n_ag > 0 and n_neighbour == 0:
            return "Ag focal only"
        if n_neighbour > 0 and n_ag == 0:
            return "Neighbour non-Ag only"
        if USE_MIXED_CATEGORY:
            return "Mixed"
        return "Ag focal only" if n_ag >= n_neighbour else "Neighbour non-Ag only"

    counts["display_class"] = counts.apply(classify_cell, axis=1)

    half_size = 0.5 * DISPLAY_GRID_M * DISPLAY_CELL_FILL_RATIO
    geometries = []
    for _, row in counts.iterrows():
        centre_x = anchor_x + (float(row["display_ix"]) + 0.5) * DISPLAY_GRID_M
        centre_y = anchor_y + (float(row["display_iy"]) + 0.5) * DISPLAY_GRID_M
        geometries.append(
            box(
                centre_x - half_size,
                centre_y - half_size,
                centre_x + half_size,
                centre_y + half_size,
            )
        )

    grid = gpd.GeoDataFrame(counts, geometry=geometries, crs=up_proj.crs)

    log_count = np.log1p(grid["total_count"].astype(float))
    if float(log_count.max()) > float(log_count.min()):
        scaled = (log_count - log_count.min()) / (log_count.max() - log_count.min())
    else:
        scaled = pd.Series(np.ones(len(grid)), index=grid.index)
    grid["display_alpha"] = (
        DISPLAY_ALPHA_MIN
        + scaled * (DISPLAY_ALPHA_MAX - DISPLAY_ALPHA_MIN)
    )

    class_colours = {
        "Ag focal only": AG_COLOR,
        "Neighbour non-Ag only": NEIGHBOR_COLOR,
        "Mixed": MIXED_COLOR,
    }
    grid["display_colour"] = [
        to_rgba(class_colours[cls], alpha=float(alpha))
        for cls, alpha in zip(grid["display_class"], grid["display_alpha"])
    ]
    return grid


def plot_global_clustered(
    layers: Dict[str, Optional[gpd.GeoDataFrame]],
    output_dir: Path,
) -> None:
    up = layers["up_proj_full"]
    land = layers["land_proj"]
    dry = layers["dry_proj"]
    assert up is not None and land is not None

    display_grid = build_display_grid(up)

    # Export the exact aggregated-display data for reproducibility.
    display_grid.to_file(
        output_dir / "ag_spatial_groups_clustered_display_data.gpkg",
        driver="GPKG",
    )
    display_grid.drop(columns="geometry").to_csv(
        output_dir / "ag_spatial_groups_clustered_display_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig = plt.figure(figsize=GLOBAL_FIGSIZE, facecolor="none")
    ax = fig.add_axes(GLOBAL_MAP_AX, facecolor="none")
    frame = draw_global_background(ax, land, dry)

    # Other UP hexagons are retained as a subdued exact contextual layer.
    other = up.loc[up["plot_group"] == "Other_nonAg_UP"]
    if not other.empty:
        other.plot(
            ax=ax,
            color=OTHER_UP_COLOR,
            edgecolor="none",
            linewidth=0,
            alpha=OTHER_UP_ALPHA_CLUSTERED,
            zorder=3,
        )

    display_grid.plot(
        ax=ax,
        color=display_grid["display_colour"].tolist(),
        edgecolor="none",
        linewidth=0,
        zorder=5,
    )

    land.boundary.plot(
        ax=ax,
        color=COAST_COLOR,
        linewidth=0.22,
        alpha=0.72,
        zorder=20,
    )
    clip_axis_collections(ax, frame)

    legend_ax = fig.add_axes(GLOBAL_LEGEND_AX, facecolor="none")
    legend_ax.set_axis_off()
    handles = [
        Patch(
            facecolor=AG_COLOR,
            edgecolor="none",
            label=f"Ag-focal display cells ({DISPLAY_GRID_KM} km)",
        ),
        Patch(
            facecolor=NEIGHBOR_COLOR,
            edgecolor="none",
            label="Neighbour non-Ag display cells",
        ),
        Patch(
            facecolor=MIXED_COLOR,
            edgecolor="none",
            label="Mixed display cells",
        ),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        columnspacing=2.0,
        borderaxespad=0,
    )

    save_figure(
        fig,
        output_dir,
        "ag_spatial_groups_global_clustered_display",
        CLUSTER_DPI,
    )
    plt.close(fig)


# =============================================================================
# 11. Regional exact-polygon panels
# =============================================================================


def projected_bounds_from_lonlat(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    n: int = 250,
) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)

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
    padding_fraction: float = 0.025,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    centre_x = 0.5 * (xmin + xmax)
    centre_y = 0.5 * (ymin + ymax)

    if width / height < target_aspect:
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


def clip_projected(
    gdf: Optional[gpd.GeoDataFrame],
    bounds: Tuple[float, float, float, float],
) -> Optional[gpd.GeoDataFrame]:
    """Return a bounding-box subset for regional display.

    The geometries are not intersected with the panel rectangle. Matplotlib
    performs the visual clipping through the axis limits. This is both faster
    and robust to topology defects in source polygons.
    """
    return bbox_subset(gdf, bounds)


def plot_regional_native(
    layers: Dict[str, Optional[gpd.GeoDataFrame]],
    output_dir: Path,
) -> None:
    up = layers["up_proj_full"]
    land_geo = layers["land_geo"]
    dry_geo = layers["dry_geo"]
    assert up is not None and land_geo is not None

    up = simplify_geometry(up, HEX_SIMPLIFY_REGIONAL)
    land = simplify_geometry(land_geo.to_crs(PROJ), LAND_SIMPLIFY)
    dry = (
        simplify_geometry(dry_geo.to_crs(PROJ), DRY_SIMPLIFY)
        if dry_geo is not None
        else None
    )
    assert land is not None

    fig, axes = plt.subplots(
        2,
        2,
        figsize=REGIONAL_FIGSIZE,
        facecolor="none",
    )
    axes_flat = axes.ravel()
    summary_rows = []

    for ax, (region_name, lonlat_bounds) in zip(axes_flat, REGIONS.items()):
        raw_bounds = projected_bounds_from_lonlat(*lonlat_bounds)
        view_bounds = expand_bounds_to_aspect(
            raw_bounds,
            target_aspect=REGIONAL_PANEL_ASPECT,
        )

        region_land = clip_projected(land, view_bounds)
        region_dry = clip_projected(dry, view_bounds)
        region_up = clip_projected(up, view_bounds)

        ax.set_facecolor("none")
        if region_land is not None and not region_land.empty:
            region_land.plot(
                ax=ax,
                color=LAND_COLOR,
                edgecolor="none",
                linewidth=0,
                zorder=1,
            )
        if region_dry is not None and not region_dry.empty:
            region_dry.plot(
                ax=ax,
                color=DRY_COLOR,
                edgecolor="none",
                linewidth=0,
                alpha=0.88,
                zorder=2,
            )

        n_ag = n_neighbour = n_other = 0
        if region_up is not None and not region_up.empty:
            other = region_up.loc[region_up["plot_group"] == "Other_nonAg_UP"]
            neighbour = region_up.loc[region_up["plot_group"] == "Neighbor_nonAg"]
            ag = region_up.loc[region_up["plot_group"] == "Ag_focal"]

            n_other = len(other)
            n_neighbour = len(neighbour)
            n_ag = len(ag)

            if not other.empty:
                other.plot(
                    ax=ax,
                    color=OTHER_UP_COLOR,
                    edgecolor="none",
                    linewidth=0,
                    alpha=OTHER_UP_ALPHA_REGIONAL,
                    zorder=3,
                )
            if not neighbour.empty:
                neighbour.plot(
                    ax=ax,
                    color=NEIGHBOR_COLOR,
                    edgecolor="none",
                    linewidth=0,
                    alpha=NEIGHBOR_ALPHA_REGIONAL,
                    zorder=4,
                )
            if not ag.empty:
                ag.plot(
                    ax=ax,
                    color=AG_COLOR,
                    edgecolor="none",
                    linewidth=0,
                    alpha=AG_ALPHA_REGIONAL,
                    zorder=5,
                )

        if region_land is not None and not region_land.empty:
            region_land.boundary.plot(
                ax=ax,
                color=COAST_COLOR,
                linewidth=0.25,
                alpha=0.72,
                zorder=20,
            )

        xmin, ymin, xmax, ymax = view_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1.0 / REGIONAL_PANEL_ASPECT)
        ax.set_xticks([])
        ax.set_yticks([])
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

        summary_rows.append(
            {
                "region": region_name,
                "ag_focal_up": n_ag,
                "neighbouring_nonag_up": n_neighbour,
                "other_nonag_up": n_other,
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
    handles = [
        Patch(
            facecolor=AG_COLOR,
            edgecolor="none",
            label="Agricultural-expansion focal UP",
        ),
        Patch(
            facecolor=NEIGHBOR_COLOR,
            edgecolor="none",
            label="Non-agricultural UP in agricultural neighbourhoods",
        ),
        Patch(
            facecolor=OTHER_UP_COLOR,
            edgecolor="none",
            label="Other non-agricultural UP",
        ),
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        columnspacing=2.0,
        borderaxespad=0,
    )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "ag_spatial_groups_regional_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_figure(
        fig,
        output_dir,
        "ag_spatial_groups_regional_native_panels",
        REGIONAL_DPI,
    )
    plt.close(fig)


# =============================================================================
# 12. Main
# =============================================================================


def main() -> None:
    configure_matplotlib()
    input_gpkg, land_shp, dry_shp, output_dir = validate_inputs()

    print(f"INPUT_GPKG: {input_gpkg}")
    print(f"LAND_SHP:   {land_shp}")
    print(f"DRY_SHP:    {dry_shp}")
    print(f"OUTPUT_DIR: {output_dir}")

    layers = prepare_layers(input_gpkg, land_shp, dry_shp)

    print("Creating native global high-DPI map...")
    plot_global_native(layers, output_dir)

    print("Creating aggregated global display map...")
    plot_global_clustered(layers, output_dir)

    print("Creating native regional detail panels...")
    plot_regional_native(layers, output_dir)

    print("Finished. Outputs saved in:")
    print(output_dir)
    print("\nMain outputs:")
    print("  ag_spatial_groups_global_native_highdpi")
    print("  ag_spatial_groups_global_clustered_display")
    print("  ag_spatial_groups_regional_native_panels")
    print("  ag_spatial_groups_clustered_display_data.gpkg")
    print("  ag_spatial_groups_clustered_display_data.csv")


if __name__ == "__main__":
    main()
