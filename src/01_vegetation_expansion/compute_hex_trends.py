# -*- coding: utf-8 -*-
"""Reusable axial-hex vegetation-cover trend engine.

The engine is shared by the main NDVI analysis and vegetation threshold/index/
spatial-scale sensitivity runs. Product-specific settings are supplied through
``configs/vegetation/*.yaml``. The locked scientific operations are retained:
annual vegetated fraction from binary masks, tie-corrected Mann-Kendall trend
testing, and Sen's slope using actual year values.
"""

import argparse
import os
import re
import glob
import warnings
from pathlib import Path
import yaml
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import wkb as shapely_wkb

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== Runtime configuration ====================
#
# Product, threshold and spatial-scale differences are supplied through
# configs/vegetation/*.yaml. The statistical and geometric engine below is
# shared across main and sensitivity runs.
#
REPO_ROOT = Path(__file__).resolve().parents[2]

# Populated by apply_config() before the workflow runs.
tif_folder = None
tif_pattern = None
mask_shp = None
raster_crs = None
hex_area_km2 = None
orientation = None
output_prefix = None
n_jobs = None
INDEX_LABEL = "Vegetation index"


def _resolve_repo_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_config(config_path):
    path = Path(config_path).expanduser()
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if "hex_trend" not in config:
        raise KeyError(f"Missing 'hex_trend' section in {path}")
    return config


def apply_config(config):
    global tif_folder, tif_pattern, mask_shp, raster_crs
    global hex_area_km2, orientation, output_prefix, n_jobs, INDEX_LABEL

    cfg = config["hex_trend"]
    required = [
        "tif_folder", "tif_pattern", "mask_vector", "raster_crs",
        "hex_area_km2", "orientation", "output_prefix",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing hex_trend configuration keys: {missing}")

    tif_folder = str(_resolve_repo_path(cfg["tif_folder"]))
    tif_pattern = str(cfg["tif_pattern"])
    mask_shp = str(_resolve_repo_path(cfg["mask_vector"]))
    raster_crs = str(cfg["raster_crs"])
    hex_area_km2 = float(cfg["hex_area_km2"])
    orientation = str(cfg["orientation"])
    output_prefix = str(_resolve_repo_path(cfg["output_prefix"]))

    jobs = cfg.get("n_jobs", "auto")
    if isinstance(jobs, str) and jobs.lower() == "auto":
        n_jobs = max(1, cpu_count() - 1)
    else:
        n_jobs = max(1, int(jobs))

    INDEX_LABEL = str(config.get("index", "Vegetation index"))

    if orientation not in {"flat", "pointy"}:
        raise ValueError("hex_trend.orientation must be 'flat' or 'pointy'")
    if hex_area_km2 <= 0:
        raise ValueError("hex_trend.hex_area_km2 must be positive")

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)


def print_config_summary(config):
    cfg = config["hex_trend"]
    print("=" * 70)
    print(f"Configuration : {config.get('name', '<unnamed>')}")
    print(f"Index         : {INDEX_LABEL}")
    print(f"Binary rasters: {tif_folder}/{tif_pattern}")
    print(f"Dryland mask  : {mask_shp}")
    print(f"Raster CRS    : {raster_crs}")
    print(f"Hex area      : {hex_area_km2:g} km^2")
    print(f"Orientation   : {orientation}")
    print(f"Output prefix : {output_prefix}")
    print(f"Workers       : {n_jobs}")
    print("=" * 70)



def load_world_basemap(target_crs):
    """加载无国界陆地底图（Natural Earth land dissolve）。"""
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except (ImportError, Exception):
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
            world = gpd.read_file(url)

    world_dissolved = world.dissolve()
    return world_dissolved.to_crs(target_crs)


def hex_side_from_area(area_m2: float) -> float:
    """正六边形面积 A = (3*sqrt(3)/2) * s^2 => s = sqrt(2A/(3sqrt3))"""
    return float(np.sqrt((2.0 * area_m2) / (3.0 * np.sqrt(3.0))))


def _hex_polygon(center_x: float, center_y: float, side: float, orientation: str) -> Polygon:
    """
    生成以 (center_x, center_y) 为中心的正六边形 polygon。
    - orientation="flat"  -> 平顶（上边水平）
    - orientation="pointy"-> 尖顶（顶点朝上）
    """
    if orientation == "flat":
        # 平顶：顶点角度 0,60,...（使上边水平）
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300, 0], dtype=np.float64))
    else:
        # 尖顶：整体旋转 30 度
        angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330, 30], dtype=np.float64))

    xs = center_x + side * np.cos(angles)
    ys = center_y + side * np.sin(angles)
    return Polygon(list(zip(xs, ys)))


def _axial_to_xy(q: int, r: int, side: float, orientation: str) -> Tuple[float, float]:
    """
    Axial(q,r) -> (x,y) 标准公式（Red Blob Games）。
    """
    if orientation == "flat":
        x = side * (3.0 / 2.0 * q)
        y = side * (np.sqrt(3.0) * (r + q / 2.0))
    else:
        x = side * (np.sqrt(3.0) * (q + r / 2.0))
        y = side * (3.0 / 2.0 * r)
    return float(x), float(y)


def _compute_q_range(bounds: Tuple[float, float, float, float], side: float, orientation: str, margin: int) -> Tuple[int, int]:
    """
    依据 bbox 粗略计算 q 的范围（含 margin）。
    用步长估计，不需要精确反解（后续会 mask intersects 过滤）。
    """
    minx, miny, maxx, maxy = bounds
    if orientation == "flat":
        dx = 1.5 * side
    else:
        dx = np.sqrt(3.0) * side  # pointy 的 x 主步长

    q_min = int(np.floor((minx / dx))) - margin
    q_max = int(np.ceil((maxx / dx))) + margin
    return q_min, q_max


def _compute_r_range_for_q(bounds: Tuple[float, float, float, float], side: float, orientation: str, q: int, margin: int) -> Tuple[int, int]:
    """
    给定 q，计算能覆盖 bbox 的 r 范围（含 margin）。
    """
    minx, miny, maxx, maxy = bounds

    if orientation == "flat":
        # y = s*sqrt3*(r + q/2) => r = y/(s*sqrt3) - q/2
        denom = side * np.sqrt(3.0)
        r_min = int(np.floor(miny / denom - q / 2.0)) - margin
        r_max = int(np.ceil(maxy / denom - q / 2.0)) + margin
    else:
        # y = s*3/2*r => r = y/(1.5*s)
        denom = 1.5 * side
        r_min = int(np.floor(miny / denom)) - margin
        r_max = int(np.ceil(maxy / denom)) + margin

    return r_min, r_max


def _hex_worker_q_chunk(args):
    """
    Windows-safe worker：生成一个 q 范围内所有 hex，并 intersects mask 过滤。
    args:
      (q_start, q_end, bounds, side, orientation, mask_union_wkb)
    """
    q_start, q_end, bounds, side, orientation, mask_union_wkb = args
    mask_union = shapely_wkb.loads(mask_union_wkb)

    # 先用 bbox 判断减少 intersects 开销
    mask_bounds = mask_union.bounds

    polys: List[Polygon] = []
    margin_r = 0  # r 方向 margin 已在主进程计算时传入

    for q in range(q_start, q_end):
        r_min, r_max = _compute_r_range_for_q(bounds, side, orientation, q, margin=margin_r)
        for r in range(r_min, r_max + 1):
            x, y = _axial_to_xy(q, r, side, orientation)

            # bbox quick reject（中心点离 mask bbox 很远）
            if (x < mask_bounds[0] - 3 * side) or (x > mask_bounds[2] + 3 * side) or (y < mask_bounds[1] - 3 * side) or (y > mask_bounds[3] + 3 * side):
                continue

            h = _hex_polygon(x, y, side, orientation)

            # 再做 intersects
            if h.is_valid and h.intersects(mask_union):
                polys.append(h)

    return polys


def generate_hex_grid_axial(mask_gdf: gpd.GeoDataFrame, side: float, orientation: str = "flat", n_jobs: int = 4, margin: int = 3) -> gpd.GeoDataFrame:
    """
    生成无缝六边形蜂窝格网（Axial 坐标）。
    - 先按 bbox 估算 q/r 范围
    - 并行按 q 分块生成
    - intersects mask_union 过滤到研究区
    """
    mask_gdf = mask_gdf.copy()
    mask_gdf["geometry"] = mask_gdf.geometry.buffer(0)
    mask_gdf = mask_gdf[mask_gdf.is_valid]
    bounds = tuple(mask_gdf.total_bounds.tolist())

    # dissolve union（只做一次）
    mask_union = mask_gdf.geometry.unary_union
    mask_union_wkb = shapely_wkb.dumps(mask_union)

    q_min, q_max = _compute_q_range(bounds, side, orientation, margin=margin)
    q_values = list(range(q_min, q_max + 1))

    # q 分块
    n_jobs = max(1, int(n_jobs))
    chunk = max(1, len(q_values) // (n_jobs * 4))
    q_chunks = [(q_values[i], q_values[min(i + chunk, len(q_values)) - 1] + 1) for i in range(0, len(q_values), chunk)]

    print(f"[HexGrid] orientation={orientation} side={side:.3f}m q_range=[{q_min},{q_max}] chunks={len(q_chunks)} jobs={n_jobs}")

    worker_args = [(qs, qe, bounds, side, orientation, mask_union_wkb) for (qs, qe) in q_chunks]

    if n_jobs == 1:
        parts = [_hex_worker_q_chunk(a) for a in worker_args]
    else:
        with Pool(n_jobs) as pool:
            parts = pool.map(_hex_worker_q_chunk, worker_args)

    all_polys: List[Polygon] = []
    for p in parts:
        all_polys.extend(p)

    print(f"[HexGrid] valid hexes (intersects mask) = {len(all_polys):,}")

    return gpd.GeoDataFrame(geometry=all_polys, crs=mask_gdf.crs)


def load_tif_files(tif_folder: str, tif_pattern: str) -> Tuple[List[str], List[int]]:
    """加载所有年份 TIF，并按年份排序。"""
    search_path = os.path.join(tif_folder, tif_pattern)
    tif_files = glob.glob(search_path)
    if len(tif_files) == 0:
        raise FileNotFoundError(f"未找到符合模式的TIF文件: {search_path}")

    year_file_pairs = []
    for fp in tif_files:
        m = re.search(r"(\d{4})", os.path.basename(fp))
        if m:
            year_file_pairs.append((int(m.group(1)), fp))

    if len(year_file_pairs) == 0:
        raise ValueError("未能从文件名中解析到年份（需要4位数字年份）")

    year_file_pairs.sort(key=lambda x: x[0])
    years = [y for y, _ in year_file_pairs]
    paths = [p for _, p in year_file_pairs]

    print(f"[TIF] found {len(paths)} files, years {min(years)}-{max(years)}")
    return paths, years


def compute_yearly_veg_ratio(args):
    from rasterstats import zonal_stats
    """
    计算单个年份所有六边形的绿化比例：
    - stats=["count","sum"]
    - 二值 NDVI：此处默认 1=无植被，0=有植被 => ratio=(count-sum)/count
      如果你的定义相反，改为 ratio=sum/count
    """
    year, file_path, hex_gdf, chunk_size = args

    n_hexes = len(hex_gdf)
    chunks = [hex_gdf.iloc[i:i + chunk_size] for i in range(0, n_hexes, chunk_size)]

    all_stats = []
    for c in chunks:
        zs = zonal_stats(c, file_path, stats=["count", "sum"], all_touched=False)
        all_stats.extend(zs)

    ratios = []
    for z in all_stats:
        cnt = z.get("count") or 0
        sm = z.get("sum") or 0
        ratio = (cnt - sm) / cnt if cnt > 0 else np.nan
        ratios.append(ratio)

    return year, np.array(ratios, dtype=np.float32)


def parallel_compute_all_years(file_paths: List[str], years: List[int], hex_gdf: gpd.GeoDataFrame, n_jobs: int = 4) -> np.ndarray:
    print(f"\n[Veg] computing yearly vegetation ratio for {len(years)} years ...")
    chunk_size = 100
    args_list = [(y, p, hex_gdf, chunk_size) for y, p in zip(years, file_paths)]

    if n_jobs == 1:
        results = [compute_yearly_veg_ratio(a) for a in args_list]
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(compute_yearly_veg_ratio, args_list)

    results.sort(key=lambda x: x[0])
    n_hexes = len(hex_gdf)
    n_years = len(years)
    mat = np.zeros((n_hexes, n_years), dtype=np.float32)

    for i, (y, arr) in enumerate(results):
        mat[:, i] = arr
        print(f"  done {y} ({i + 1}/{n_years})")

    return mat


def sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return np.nan
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else np.nan


def mann_kendall_test(data: np.ndarray) -> Tuple[float, float]:
    n = len(data)
    if n < 3:
        return np.nan, np.nan

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])

    unique_vals, counts = np.unique(data, return_counts=True)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if len(counts) < n:
        for c in counts:
            if c > 1:
                var_s -= c * (c - 1) * (2 * c + 5) / 18.0

    if var_s == 0:
        return np.nan, np.nan

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return float(tau), float(p_value)


def process_hex_trend_batch(args):
    start_idx, end_idx, veg_matrix, years = args
    n = end_idx - start_idx
    tau_values = np.full(n, np.nan, dtype=np.float32)
    p_values = np.full(n, np.nan, dtype=np.float32)
    slopes = np.full(n, np.nan, dtype=np.float32)

    years_arr = np.array(years, dtype=np.float32)

    for i in range(n):
        idx = start_idx + i
        ts = veg_matrix[idx, :]
        valid = ~np.isnan(ts)
        if np.sum(valid) < 3:
            continue

        vv = ts[valid]
        yy = years_arr[valid]

        tau, p = mann_kendall_test(vv)
        tau_values[i] = tau
        p_values[i] = p
        slopes[i] = sen_slope(yy, vv)

    return start_idx, tau_values, p_values, slopes


def parallel_trend_analysis(veg_matrix: np.ndarray, years: List[int], n_jobs: int = 4):
    n_hexes = veg_matrix.shape[0]
    print(f"\n[Trend] analyzing {n_hexes:,} hexes ...")

    batch_size = max(200, n_hexes // (max(1, n_jobs) * 2))
    batches = [(i, min(i + batch_size, n_hexes)) for i in range(0, n_hexes, batch_size)]
    args_list = [(s, e, veg_matrix, years) for s, e in batches]

    if n_jobs == 1:
        results = [process_hex_trend_batch(a) for a in args_list]
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(process_hex_trend_batch, args_list)

    tau_all = np.full(n_hexes, np.nan, dtype=np.float32)
    p_all = np.full(n_hexes, np.nan, dtype=np.float32)
    slope_all = np.full(n_hexes, np.nan, dtype=np.float32)

    for start, t, p, sl in results:
        tau_all[start:start + len(t)] = t
        p_all[start:start + len(p)] = p
        slope_all[start:start + len(sl)] = sl

    print("[Trend] done.")
    return tau_all, p_all, slope_all


def add_trend_attributes(hex_gdf: gpd.GeoDataFrame, tau, p, slope, veg_matrix, years):
    hex_gdf = hex_gdf.copy()
    hex_gdf["hex_id"] = np.arange(1, len(hex_gdf) + 1, dtype=np.int32)
    hex_gdf["trend_tau"] = tau
    hex_gdf["p_value"] = p
    hex_gdf["sen_slope"] = slope
    hex_gdf["significant"] = (hex_gdf["p_value"] < 0.05).astype(np.int8)

    centroids = hex_gdf.geometry.centroid
    hex_gdf["center_x"] = centroids.x
    hex_gdf["center_y"] = centroids.y

    tmp = hex_gdf.copy()
    tmp = tmp.set_geometry(centroids).to_crs("EPSG:4326")
    hex_gdf["center_lon"] = tmp.geometry.x
    hex_gdf["center_lat"] = tmp.geometry.y

    for i, y in enumerate(years):
        hex_gdf[f"veg_{y}"] = veg_matrix[:, i]

    hex_gdf["veg_mean"] = np.nanmean(veg_matrix, axis=1).astype(np.float32)
    return hex_gdf


def visualize_significant(hex_gdf: gpd.GeoDataFrame, years: List[int], output_png: str):
    world = load_world_basemap(hex_gdf.crs)

    colors = ["#8B0000", "#FF0000", "#FA8072", "#FFFFFF", "#87CEEB", "#4169E1", "#0000FF"]
    cmap = LinearSegmentedColormap.from_list("trend", colors, N=100)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    tau_sig = hex_gdf["trend_tau"].copy()
    tau_sig[hex_gdf["p_value"] >= 0.05] = np.nan

    fig, ax = plt.subplots(figsize=(14, 10))
    world.plot(ax=ax, facecolor="#E8E8E8", edgecolor="#CCCCCC", linewidth=0.3)

    g = hex_gdf.copy()
    g["tau_sig"] = tau_sig
    g.plot(column="tau_sig", cmap=cmap, norm=norm, legend=True,
           edgecolor="none", linewidth=0.0, ax=ax,
           legend_kwds={"label": "Kendall's Tau (显著趋势, p<0.05)", "shrink": 0.8})

    minx, miny, maxx, maxy = hex_gdf.total_bounds
    buf = max(maxx - minx, maxy - miny) * 0.05
    ax.set_xlim(minx - buf, maxx + buf)
    ax.set_ylim(miny - buf, maxy + buf)

    ax.set_title(f"{INDEX_LABEL} Trend Analysis ({years[0]}-{years[-1]})\nMann-Kendall Test (Significant Only, p<0.05)",
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_all(hex_gdf: gpd.GeoDataFrame, years: List[int], output_png_all: str):
    from matplotlib.patches import Patch
    from matplotlib.cm import ScalarMappable

    world = load_world_basemap(hex_gdf.crs)
    colors = ["#8B0000", "#FF0000", "#FA8072", "#FFFFFF", "#87CEEB", "#4169E1", "#0000FF"]
    cmap = LinearSegmentedColormap.from_list("trend", colors, N=100)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    g = hex_gdf.copy()
    g["category"] = 0
    has_data = ~g["trend_tau"].isna()
    g.loc[has_data & (g["p_value"] >= 0.05), "category"] = 1
    g.loc[has_data & (g["p_value"] < 0.05), "category"] = 2

    fig, ax = plt.subplots(figsize=(14, 10))
    world.plot(ax=ax, facecolor="#E8E8E8", edgecolor="#CCCCCC", linewidth=0.3)

    no_data = g[g["category"] == 0]
    if len(no_data) > 0:
        no_data.plot(ax=ax, facecolor="white", edgecolor="none", linewidth=0)

    not_sig = g[g["category"] == 1]
    if len(not_sig) > 0:
        not_sig.plot(ax=ax, facecolor="#FFD700", edgecolor="none", linewidth=0)

    sig = g[g["category"] == 2]
    if len(sig) > 0:
        sig.plot(column="trend_tau", cmap=cmap, norm=norm, ax=ax, edgecolor="none", linewidth=0)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Kendall's Tau (显著趋势)", fontsize=11, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)

    legend_elements = []
    if len(sig) > 0:
        legend_elements.append(Patch(facecolor="#0000FF", edgecolor="black",
                                    label=f"显著上升 (n={len(sig[sig['trend_tau'] > 0])})"))
        legend_elements.append(Patch(facecolor="#8B0000", edgecolor="black",
                                    label=f"显著下降 (n={len(sig[sig['trend_tau'] < 0])})"))
    if len(not_sig) > 0:
        legend_elements.append(Patch(facecolor="#FFD700", edgecolor="black",
                                    label=f"趋势不显著 (n={len(not_sig)})"))
    if len(no_data) > 0:
        legend_elements.append(Patch(facecolor="white", edgecolor="black",
                                    label=f"无数据/no_data (n={len(no_data)})"))
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9, title="图例")

    n_total = len(g)
    n_sig = len(sig)
    n_not = len(not_sig)
    ax.set_title(
        f"{INDEX_LABEL} Trend Analysis - All Hexagons ({years[0]}-{years[-1]})\n"
        f"Total: {n_total:,} | Significant: {n_sig:,} ({(n_sig/n_total*100 if n_total else 0):.1f}%) | "
        f"Not Significant: {n_not:,} ({(n_not/n_total*100 if n_total else 0):.1f}%)",
        fontsize=14, fontweight="bold", pad=20
    )

    minx, miny, maxx, maxy = hex_gdf.total_bounds
    buf = max(maxx - minx, maxy - miny) * 0.05
    ax.set_xlim(minx - buf, maxx + buf)
    ax.set_ylim(miny - buf, maxy + buf)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_png_all, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(hex_gdf: gpd.GeoDataFrame, years: List[int], output_prefix: str):
    output_shp = f"{output_prefix}.shp"
    output_csv = f"{output_prefix}.csv"
    output_png = f"{output_prefix}.png"
    output_png_all = f"{output_prefix}_all.png"

    # Shapefile 字段截断
    rename_dict = {
        "trend_tau": "tau",
        "p_value": "pvalue",
        "sen_slope": "slope",
        "significant": "signif",
        "center_lon": "cen_lon",
        "center_lat": "cen_lat",
        "center_x": "cen_x",
        "center_y": "cen_y",
        "veg_mean": "veg_avg",
    }
    shp_gdf = hex_gdf.rename(columns=rename_dict).copy()

    keep_cols = ["hex_id", "tau", "pvalue", "slope", "signif", "cen_lon", "cen_lat", "veg_avg", "geometry"]
    year_cols = [f"veg_{y}" for y in years]
    keep_cols = keep_cols[:-1] + year_cols[: min(10, len(year_cols))] + ["geometry"]
    shp_gdf = shp_gdf[[c for c in keep_cols if c in shp_gdf.columns]]

    print(f"\n[Save] Shapefile -> {output_shp}")
    shp_gdf.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")

    # CSV 全量字段
    csv_cols = ["hex_id", "trend_tau", "p_value", "sen_slope", "significant", "center_lon", "center_lat", "veg_mean"] + year_cols
    print(f"[Save] CSV -> {output_csv}")
    hex_gdf[csv_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")

    # PNG
    print(f"[Save] PNG (sig) -> {output_png}")
    visualize_significant(hex_gdf, years, output_png)
    print(f"[Save] PNG (all) -> {output_png_all}")
    visualize_all(hex_gdf, years, output_png_all)

    return output_shp, output_csv, output_png, output_png_all


def print_summary(hex_gdf: gpd.GeoDataFrame, years: List[int]):
    print("\n" + "=" * 70)
    print("📊 趋势分析摘要")
    print("=" * 70)
    n_total = len(hex_gdf)
    valid = hex_gdf["trend_tau"].dropna()
    sig = hex_gdf[hex_gdf["p_value"] < 0.05]
    print(f"时间跨度: {years[0]} - {years[-1]} ({len(years)}年)")
    print(f"六边形总数: {n_total:,}")
    print(f"有效分析: {len(valid):,} ({(len(valid)/n_total*100 if n_total else 0):.1f}%)")
    print(f"显著趋势: {len(sig):,} ({(len(sig)/n_total*100 if n_total else 0):.1f}%)")
    if len(sig) > 0:
        inc = sig[sig["trend_tau"] > 0]
        dec = sig[sig["trend_tau"] < 0]
        print("\n显著趋势分布:")
        print(f"  • 上升: {len(inc):,} ({len(inc)/len(sig)*100:.1f}%)")
        print(f"  • 下降: {len(dec):,} ({len(dec)/len(sig)*100:.1f}%)")
        print("\nKendall's Tau (显著):")
        print(f"  • 均值: {sig['trend_tau'].mean():.4f}")
        print(f"  • 中位数: {sig['trend_tau'].median():.4f}")
        print(f"  • 范围: [{sig['trend_tau'].min():.4f}, {sig['trend_tau'].max():.4f}]")
        print("\nSen's slope (显著):")
        print(f"  • 均值: {sig['sen_slope'].mean():.6f}")
        print(f"  • 中位数: {sig['sen_slope'].median():.6f}")
    print("=" * 70)


# ==================== Main program ====================
def run(config, check_only=False):
    import time

    apply_config(config)
    print_config_summary(config)
    if check_only:
        return

    start_time = time.time()
    print("=" * 70)
    print(f"Vegetation trend analysis: {INDEX_LABEL}")
    print("Mann-Kendall + Sen's slope on axial hexagons")
    print("=" * 70)

    print("[1/6] Reading dryland mask ...")
    t1 = time.time()
    mask = gpd.read_file(mask_shp)
    mask = mask.to_crs(raster_crs)
    print(f"mask polygons={len(mask)} ({time.time()-t1:.2f}s)")

    print("\n[2/6] Generating axial hexagon grid ...")
    t2 = time.time()
    side = hex_side_from_area(hex_area_km2 * 1e6)
    print(f"side={side:.3f} m")
    hex_gdf = generate_hex_grid_axial(mask, side, orientation=orientation, n_jobs=n_jobs, margin=3)
    print(f"hexes={len(hex_gdf):,} ({time.time()-t2:.2f}s)")

    print("\n[3/6] Loading annual binary rasters ...")
    t3 = time.time()
    file_paths, years = load_tif_files(tif_folder, tif_pattern)
    print(f"done ({time.time()-t3:.2f}s)")

    print("\n[4/6] Computing annual vegetated fraction ...")
    t4 = time.time()
    veg_matrix = parallel_compute_all_years(file_paths, years, hex_gdf, n_jobs=n_jobs)
    print(f"veg_matrix={veg_matrix.shape[0]:,}x{veg_matrix.shape[1]} ({time.time()-t4:.2f}s)")

    print("\n[5/6] Trend analysis (MK + Sen) ...")
    t5 = time.time()
    tau_values, p_values, sen_slopes = parallel_trend_analysis(veg_matrix, years, n_jobs=n_jobs)
    print(f"done ({time.time()-t5:.2f}s)")

    hex_gdf = add_trend_attributes(hex_gdf, tau_values, p_values, sen_slopes, veg_matrix, years)

    print("\n[6/6] Saving outputs ...")
    t6 = time.time()
    out_shp, out_csv, out_png, out_png_all = save_outputs(hex_gdf, years, output_prefix)
    print(f"done ({time.time()-t6:.2f}s)")

    print_summary(hex_gdf, years)
    print(f"\nTotal elapsed: {time.time()-start_time:.2f}s")
    print("Outputs:")
    print(f"  {out_png}")
    print(f"  {out_png_all}")
    print(f"  {out_shp}")
    print(f"  {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute annual vegetated fraction and MK/Sen trends on axial hexagons."
    )
    parser.add_argument("--config", required=True, help="Path to a vegetation YAML configuration.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate and display the configuration without reading input data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(load_config(args.config), check_only=args.check_config)


if __name__ == "__main__":
    main()
