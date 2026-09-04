# -*- coding: utf-8 -*-
"""
单年份 NDVI 六边形统计 + Shapefile / CSV / PNG 输出（AXIAL 无缝蜂窝格网版）

特点
- 六边形格网：轴坐标(axial)生成（与多年份最终版同一逻辑），严格无缝平铺
- 并行 zonal_stats：count/sum
- veg_ratio 默认算法：二值 NDVI (1=无植被, 0=有植被) => (count - sum) / count
  若你的定义相反（1=有植被），把 VEG_MODE 改为 "one_is_veg"
- 输出：shp/csv/png（全球底图 + 六边形渲染）

依赖
geopandas, rasterio, rasterstats, shapely, numpy, pandas, matplotlib
可选：geodatasets（用于底图；没有也会自动 fallback）
"""

import os
import warnings
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from shapely.prepared import prep as shp_prep
from shapely.wkb import dumps as wkb_dumps, loads as wkb_loads
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

# ===================== USER CONFIG =====================
raster_path   = "NDVI_data_fixed/MODIS_NDVI_2000_measure.tif"
mask_shp      = "Drylands_dataset_fixed/drylands_8857.shp"
raster_crs    = "EPSG:8857"          # NDVI 栅格 CRS（与你的栅格一致）
hex_area_km2  = 3000                 # 六边形面积（km²）
orientation   = "flat"               # "flat"（平顶） or "pointy"（尖顶）
margin_cells  = 2                    # 外扩几圈，避免边缘裁剪漏格

VEG_MODE      = "one_is_noveg"       # "one_is_noveg" 或 "one_is_veg"

output_png    = "veg_2000_hex_axial.png"
output_shp    = "veg_2000_hex_axial.shp"
output_csv    = "veg_2000_hex_axial.csv"

n_jobs = max(1, cpu_count() - 1)
# =======================================================


# -------------------- Basemap --------------------
def load_world_basemap(target_crs):
    """加载无国界世界陆地底图（dissolve 后投影到 target_crs）"""
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
            world = gpd.read_file(url)
    return world.dissolve().to_crs(target_crs)


# -------------------- Hex geometry --------------------
def hex_side_from_area(area_m2: float) -> float:
    """正六边形面积 A = (3*sqrt(3)/2)*s^2 -> s = sqrt(2A/(3sqrt3))"""
    return math.sqrt((2.0 * area_m2) / (3.0 * math.sqrt(3.0)))


def _hex_vertex_offsets(side: float, orientation: str):
    """返回 7 个顶点偏移（含闭合点），与 orientation 严格配套"""
    rot = 0.0 if orientation == "flat" else math.pi / 6.0
    angles = np.linspace(0.0, 2.0 * math.pi, 7) + rot
    return side * np.cos(angles), side * np.sin(angles)


def _grid_dxdy(side: float, orientation: str):
    """理论中心点间距（用于 snap 原点/范围估计）"""
    if orientation == "flat":
        return 1.5 * side, math.sqrt(3.0) * side
    else:
        return math.sqrt(3.0) * side, 1.5 * side


def _axial_to_xy(q: int, r: int, side: float, orientation: str):
    """
    轴坐标(axial) -> 平面坐标
    flat-top:
      x = s * (3/2 * q)
      y = s * (sqrt(3) * (r + q/2))
    pointy-top:
      x = s * (sqrt(3) * (q + r/2))
      y = s * (3/2 * r)
    """
    if orientation == "flat":
        x = side * (1.5 * q)
        y = side * (math.sqrt(3.0) * (r + 0.5 * q))
    else:
        x = side * (math.sqrt(3.0) * (q + 0.5 * r))
        y = side * (1.5 * r)
    return x, y


# -------- Windows-safe worker（顶层函数，可 pickle）--------
def _axial_worker(args):
    """
    args:
      (q_list, r_min, r_max, side, orientation, cos_off, sin_off, x0, y0, mask_wkb)
    """
    q_list, r_min, r_max, side, orientation, cos_off, sin_off, x0, y0, mask_wkb = args
    mask_union = wkb_loads(mask_wkb)
    mask_prep = shp_prep(mask_union)

    out = []
    for q in q_list:
        for r in range(r_min, r_max + 1):
            cx, cy = _axial_to_xy(q, r, side, orientation)
            x = x0 + cx
            y = y0 + cy
            h = Polygon(list(zip(x + cos_off, y + sin_off)))
            if mask_prep.intersects(h):
                out.append(h)
    return out


def generate_hex_grid_axial(mask_gdf: gpd.GeoDataFrame, side: float,
                           orientation: str = "flat", n_jobs: int = 4, margin_cells: int = 2):
    """
    AXIAL 无缝蜂窝格网生成（与多年份最终版同逻辑）
    - 根据 total_bounds 反推 q/r 范围（保守）
    - 并行按 q 分块生成
    - centroid 量化去重（防止边界重复）
    """
    mask_gdf = mask_gdf[mask_gdf.is_valid].copy()
    mask_union = mask_gdf.geometry.unary_union
    minx, miny, maxx, maxy = mask_gdf.total_bounds

    dx, dy = _grid_dxdy(side, orientation)
    cos_off, sin_off = _hex_vertex_offsets(side, orientation)

    # snap 原点到 dx/dy 网格，避免浮点漂移
    x0 = math.floor(minx / dx) * dx
    y0 = math.floor(miny / dy) * dy

    if orientation == "flat":
        q_min = int(math.floor((minx - x0) / (1.5 * side))) - margin_cells - 2
        q_max = int(math.ceil((maxx - x0) / (1.5 * side))) + margin_cells + 2

        def r_from_y(y, q):
            return (y / (math.sqrt(3.0) * side)) - 0.5 * q

        r_candidates = []
        for q in (q_min, q_max):
            r_candidates += [r_from_y((miny - y0), q), r_from_y((maxy - y0), q)]
        r_min = int(math.floor(min(r_candidates))) - margin_cells - 2
        r_max = int(math.ceil(max(r_candidates))) + margin_cells + 2
    else:
        r_min = int(math.floor((miny - y0) / (1.5 * side))) - margin_cells - 2
        r_max = int(math.ceil((maxy - y0) / (1.5 * side))) + margin_cells + 2

        def q_from_x(x, r):
            return (x / (math.sqrt(3.0) * side)) - 0.5 * r

        q_candidates = []
        for r in (r_min, r_max):
            q_candidates += [q_from_x((minx - x0), r), q_from_x((maxx - x0), r)]
        q_min = int(math.floor(min(q_candidates))) - margin_cells - 2
        q_max = int(math.ceil(max(q_candidates))) + margin_cells + 2

    q_all = list(range(q_min, q_max + 1))
    chunk = max(8, len(q_all) // (n_jobs * 4))
    q_chunks = [q_all[i:i + chunk] for i in range(0, len(q_all), chunk)]

    print(f"[格网] orientation={orientation} side={side:.3f} dx={dx:.3f} dy={dy:.3f}")
    print(f"[格网] q=[{q_min},{q_max}] ({len(q_all)}) r=[{r_min},{r_max}] ({r_max - r_min + 1})")
    print(f"[格网] 并行进程: {n_jobs}")

    mask_wkb = wkb_dumps(mask_union)
    args_list = [(qs, r_min, r_max, side, orientation, cos_off, sin_off, x0, y0, mask_wkb)
                 for qs in q_chunks]

    with Pool(processes=n_jobs) as pool:
        parts = pool.map(_axial_worker, args_list)

    hexes = [h for part in parts for h in part]

    # 1mm 量化去重
    seen = set()
    unique = []
    inv = 1e-3
    for h in hexes:
        c = h.centroid
        key = (int(round(c.x / inv)), int(round(c.y / inv)))
        if key not in seen:
            seen.add(key)
            unique.append(h)

    gdf = gpd.GeoDataFrame({"geometry": unique}, crs=mask_gdf.crs)
    print(f"[格网] 有效六边形: {len(gdf):,}（去重前 {len(hexes):,}）")
    return gdf


# -------------------- Zonal stats (parallel) --------------------
def _zonal_worker(args):
    """对一个 hex_batch 做分块 zonal_stats"""
    hex_batch, raster_path, chunk_size = args
    chunks = [hex_batch[i:i + chunk_size] for i in range(0, len(hex_batch), chunk_size)]
    all_stats = []
    for ch in chunks:
        st = zonal_stats(ch, raster_path, stats=["count", "sum"], all_touched=False)
        all_stats.extend(st)
    return all_stats


def parallel_zonal_stats(hex_gdf, raster_path, n_jobs=4):
    total = len(hex_gdf)
    batch_size = max(200, total // (n_jobs * 3))
    chunk_size = 50
    batches = [hex_gdf.iloc[i:i + batch_size] for i in range(0, total, batch_size)]
    print(f"[统计] 六边形={total:,} 批次={len(batches)} 进程={n_jobs}")
    args_list = [(b, raster_path, chunk_size) for b in batches]
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_zonal_worker, args_list)
    out = []
    for r in results:
        out.extend(r)
    return out


def compute_veg_ratio_vectorized(zs, mode="one_is_noveg"):
    """
    mode:
      - "one_is_noveg": 1=无植被,0=有植被 => (count - sum)/count
      - "one_is_veg"  : 1=有植被,0=无植被 => sum/count
    """
    counts = np.array([z.get("count") or 0 for z in zs], dtype=np.float32)
    sums = np.array([z.get("sum") or 0 for z in zs], dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "one_is_veg":
            veg = sums / counts
        else:
            veg = (counts - sums) / counts
        veg[counts == 0] = np.nan
    return veg


# -------------------- Attributes & Outputs --------------------
def add_hex_attributes(hex_gdf):
    """hex_id + 中心点坐标（投影/经纬度）"""
    hex_gdf = hex_gdf.copy()
    hex_gdf["hex_id"] = np.arange(1, len(hex_gdf) + 1)

    cent = hex_gdf.geometry.centroid
    hex_gdf["center_x"] = cent.x
    hex_gdf["center_y"] = cent.y

    g2 = hex_gdf.copy().set_geometry(cent).to_crs("EPSG:4326")
    hex_gdf["center_lon"] = g2.geometry.x
    hex_gdf["center_lat"] = g2.geometry.y
    return hex_gdf


def save_as_shapefile(hex_gdf, output_path):
    out = hex_gdf[["hex_id", "veg_ratio", "center_lon", "center_lat", "center_x", "center_y", "geometry"]].copy()
    out.columns = ["hex_id", "veg_ratio", "cen_lon", "cen_lat", "cen_x", "cen_y", "geometry"]  # shp字段<=10
    out.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[输出] Shapefile: {output_path}")


def save_as_csv(hex_gdf, output_path):
    out = hex_gdf[["hex_id", "veg_ratio", "center_lon", "center_lat"]].copy()
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[输出] CSV: {output_path}  行数={len(out):,}")


def plot_png(hex_gdf, output_png):
    world = load_world_basemap(hex_gdf.crs)
    fig, ax = plt.subplots(figsize=(12, 10))
    world.plot(ax=ax, facecolor="#E8E8E8", edgecolor="#CCCCCC", linewidth=0.3)
    hex_gdf.plot(
        column="veg_ratio",
        cmap="Greens",
        legend=True,
        edgecolor="none",
        linewidth=0.01,
        ax=ax,
        legend_kwds={"label": "Vegetation Fraction", "shrink": 0.8},
    )
    minx, miny, maxx, maxy = hex_gdf.total_bounds
    buf = max(maxx - minx, maxy - miny) * 0.05
    ax.set_xlim(minx - buf, maxx + buf)
    ax.set_ylim(miny - buf, maxy + buf)
    ax.set_axis_off()
    ax.set_title(
        "Vegetation Fraction — Single Year\n(Axial Hex Grid, gap-free)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[输出] PNG: {output_png}")


# ===================== MAIN =====================
if __name__ == "__main__":
    import time

    t0 = time.time()
    print("=" * 70)
    print(f"单年份 NDVI 六边形统计（AXIAL 无缝格网） n_jobs={n_jobs}")
    print("=" * 70)

    # 1) 读取掩膜并投影到栅格 CRS
    mask = gpd.read_file(mask_shp).to_crs(raster_crs)
    print(f"[mask] polygons={len(mask)} crs={mask.crs}")

    # 2) 生成 AXIAL 六边形格网
    side = hex_side_from_area(hex_area_km2 * 1e6)
    print(f"[参数] hex_area={hex_area_km2} km²  side={side:.3f} m  orientation={orientation}")
    hex_gdf = generate_hex_grid_axial(mask, side, orientation=orientation, n_jobs=n_jobs, margin_cells=margin_cells)

    # 3) 并行 zonal stats
    zs = parallel_zonal_stats(hex_gdf, raster_path, n_jobs=n_jobs)
    hex_gdf["veg_ratio"] = compute_veg_ratio_vectorized(zs, mode=VEG_MODE)

    # 4) 添加属性
    hex_gdf = add_hex_attributes(hex_gdf)

    # 5) 输出
    save_as_shapefile(hex_gdf, output_shp)
    save_as_csv(hex_gdf, output_csv)
    plot_png(hex_gdf, output_png)

    print(f"DONE. total={time.time() - t0:.2f}s")
