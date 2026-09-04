# -*- coding: utf-8 -*-
"""
VOD validation for NDVI-derived UP/DOWN hexagons
Accelerated version with multiprocessing zonal extraction.

文件结构要求：
当前代码所在文件夹/
├─ VOD_TREND_fast.py
├─ NDVI_trend_hex_100_up.shp
├─ NDVI_trend_hex_100_down.shp
├─ NDVI_trend_hex_100.csv                  # 可选；默认不使用 background
├─ VODCA_CXKu_tif/
│  ├─ VODCA_CXKu_2000.tif
│  ├─ VODCA_CXKu_2001.tif
│  └─ ...
├─ VOD_annual_max_tif/
│  ├─ VOD_2002.tif
│  ├─ VOD_2003.tif
│  └─ ...
└─ VOD_validation_output/                  # 自动生成

依赖：
pip install geopandas rasterio rasterstats shapely pandas numpy scipy tqdm
"""

import os
import re
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from scipy.stats import norm, mannwhitneyu
from shapely import wkt
from shapely.geometry import shape
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# 0. 工作目录与路径配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

VODCA_FOLDER = "VODCA_CXKu_tif"        # 文件名示例：VODCA_CXKu_2000.tif
VOD_FOLDER = "VOD_annual_max_tif"      # 文件名示例：VOD_2002.tif

UP_SHP = "NDVI_hex_data/NDVI_trend_hex_100_up.shp"
DOWN_SHP = "NDVI_hex_data/NDVI_trend_hex_100_down.shp"
FULL_HEX_CSV = "NDVI_hex_data/NDVI_trend_hex_100.csv"

OUT_DIR = "VOD_validation_output"
os.makedirs(OUT_DIR, exist_ok=True)

PRODUCTS = {
    "VODCA_CXKu": {
        "folder": VODCA_FOLDER,
        "filename_prefix": "VODCA_CXKu",
        "expected_start_year": 2000,
        "expected_end_year": 2021,
        "scale": 1.0,
    },
    "VOD": {
        "folder": VOD_FOLDER,
        "filename_prefix": "VOD",
        "expected_start_year": 2002,
        "expected_end_year": 2022,
        "scale": 1.0,
    },
}


# ============================================================
# 1. 参数设置
# ============================================================

# 对粗分辨率 VOD，median 通常比 mean 更稳
ZONAL_STAT = "median"

# 粗分辨率 raster 与 hex 边界不完全匹配时，True 更稳
ALL_TOUCHED = True

# 每个 hex 至少需要多少比例有效年份才纳入趋势统计
MIN_VALID_FRAC = 0.75

# 是否启用多进程并行提取
USE_MULTIPROCESSING = True

# 并行核心数。建议先用 4；内存充足再改 6 或 8
N_WORKERS = 16

# background/non-UP 对照组最大抽样数量。
# 默认设为 0，即只做 UP/DOWN，速度最快。
# 如果后续想加 non-UP 对照，可改成 60000，并确保 CSV 中有 geometry/WKT/.geo 字段。
MAX_BACKGROUND_HEX = 0

RANDOM_SEED = 42

# 是否保存每个 hex 的逐年 VOD 值
SAVE_ANNUAL_VALUES = True

# VOD 合理值范围，用于剔除 nodata 或异常填充值
VALID_VOD_MIN = -0.5
VALID_VOD_MAX = 10.0


# ============================================================
# 2. 基础函数
# ============================================================

def infer_year_from_name(file_path, prefix):
    """
    从文件名中识别年份。
    支持：
    VODCA_CXKu_2000.tif
    VOD_2002.tif
    """
    filename = os.path.basename(file_path)
    stem = os.path.splitext(filename)[0]

    pattern = rf"^{re.escape(prefix)}[_-]?((?:19|20)\d{{2}})$"
    m = re.search(pattern, stem)

    if m:
        return int(m.group(1))

    m = re.search(r"(19|20)\d{2}", stem)
    if m:
        return int(m.group(0))

    raise ValueError(f"无法从文件名中识别年份：{filename}")


def list_product_tifs(product_name, product_cfg):
    tif_dir = product_cfg["folder"]
    prefix = product_cfg["filename_prefix"]

    if not os.path.exists(tif_dir):
        raise FileNotFoundError(f"{product_name} 文件夹不存在：{tif_dir}")

    files = []
    for filename in os.listdir(tif_dir):
        if filename.lower().endswith((".tif", ".tiff")):
            files.append(os.path.join(tif_dir, filename))

    if len(files) == 0:
        raise FileNotFoundError(f"{product_name} 文件夹中未找到 tif 文件：{tif_dir}")

    year_files = []
    for fp in files:
        try:
            year = infer_year_from_name(fp, prefix)
            year_files.append((year, fp))
        except ValueError:
            print(f"[WARNING] 跳过无法识别年份的文件：{os.path.basename(fp)}")

    if len(year_files) == 0:
        raise ValueError(f"{product_name} 没有任何可识别年份的 tif 文件。")

    year_files = sorted(year_files, key=lambda x: x[0])
    years = [y for y, _ in year_files]

    print(f"\n[INFO] {product_name} 识别到年份：{min(years)}–{max(years)}")
    print(f"[INFO] {product_name} 文件数：{len(year_files)}")

    expected_start = product_cfg.get("expected_start_year")
    expected_end = product_cfg.get("expected_end_year")

    if expected_start is not None and expected_end is not None:
        expected = set(range(expected_start, expected_end + 1))
        actual = set(years)

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        if missing:
            print(f"[WARNING] {product_name} 缺少年份：{missing}")
        if extra:
            print(f"[WARNING] {product_name} 存在预期外年份：{extra}")

    return year_files


def check_product_crs_consistency(product_name, year_files):
    """
    检查同一产品内所有 tif 的 CRS 是否一致。
    若一致，返回该 CRS。
    若不一致，抛出错误，避免错误投影。
    """
    crs_list = []

    for year, tif_path in year_files:
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                raise ValueError(f"{product_name} {year} raster 没有 CRS 信息：{tif_path}")
            crs_list.append(src.crs)

    first_crs = crs_list[0]

    for (year, tif_path), crs in zip(year_files, crs_list):
        if crs != first_crs:
            raise ValueError(
                f"{product_name} 内部 tif CRS 不一致。\n"
                f"第一张 CRS: {first_crs}\n"
                f"异常年份: {year}\n"
                f"异常文件: {tif_path}\n"
                f"异常 CRS: {crs}\n"
                f"请先统一该产品所有 tif 的投影后再运行。"
            )

    return first_crs


def infer_id_column(df):
    candidates = [
        "hex_id", "HEX_ID", "hexid", "HexID",
        "grid_id", "GRID_ID",
        "id", "ID",
        "FID", "fid",
        "OBJECTID", "objectid",
        "system:index"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def ensure_hex_id(gdf, prefix):
    id_col = infer_id_column(gdf)

    if id_col is not None:
        gdf["hex_id_work"] = gdf[id_col].astype(str)
    else:
        gdf["hex_id_work"] = [f"{prefix}_{i}" for i in range(len(gdf))]

    return gdf


def read_sample_shp(path, sample_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件：{path}")

    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"矢量文件为空：{path}")

    if gdf.crs is None:
        raise ValueError(
            f"{path} 没有 CRS 信息。需要先在 GIS 中定义正确投影，不能直接计算。"
        )

    gdf = ensure_hex_id(gdf, sample_name)
    gdf["optical_sample"] = sample_name

    gdf = gdf[["hex_id_work", "optical_sample", "geometry"]].copy()
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()

    return gdf


def try_read_full_csv_as_background(csv_path, up_ids, down_ids):
    """
    可选 background/non-UP 对照组。
    只有当完整 CSV 中有 geometry / WKT / .geo 字段时才会启用。
    默认 MAX_BACKGROUND_HEX = 0，即跳过。
    """
    if MAX_BACKGROUND_HEX == 0:
        print("[INFO] MAX_BACKGROUND_HEX = 0，跳过 background/non-UP。")
        return None

    if not os.path.exists(csv_path):
        print(f"[INFO] 未找到完整 CSV，跳过 background/non-UP：{csv_path}")
        return None

    df = pd.read_csv(csv_path)

    geom_col = None
    for c in ["geometry", "WKT", "wkt", ".geo", "geojson", "GeoJSON"]:
        if c in df.columns:
            geom_col = c
            break

    if geom_col is None:
        print("[INFO] 完整 CSV 未发现 geometry/WKT/.geo 字段，跳过 background/non-UP。")
        return None

    id_col = infer_id_column(df)

    if id_col is not None:
        df["hex_id_work"] = df[id_col].astype(str)
    else:
        df["hex_id_work"] = [f"ALL_{i}" for i in range(len(df))]

    used_ids = up_ids | down_ids
    df = df[~df["hex_id_work"].isin(used_ids)].copy()

    def parse_geom(x):
        if pd.isna(x):
            return None

        s = str(x).strip()

        try:
            if s.startswith("{"):
                return shape(json.loads(s))
            return wkt.loads(s)
        except Exception:
            return None

    df["geometry_parsed"] = df[geom_col].apply(parse_geom)
    df = df[df["geometry_parsed"].notna()].copy()

    if df.empty:
        print("[INFO] 完整 CSV geometry 解析失败，跳过 background/non-UP。")
        return None

    if MAX_BACKGROUND_HEX is not None and len(df) > MAX_BACKGROUND_HEX:
        df = df.sample(n=MAX_BACKGROUND_HEX, random_state=RANDOM_SEED).copy()

    bg = gpd.GeoDataFrame(
        df[["hex_id_work"]].copy(),
        geometry=df["geometry_parsed"],
        crs="EPSG:4326"
    )

    bg["optical_sample"] = "background_nonUP"

    print(f"[INFO] background/non-UP 样本数：{len(bg):,}")

    return bg[["hex_id_work", "optical_sample", "geometry"]].copy()


def build_samples():
    up = read_sample_shp(UP_SHP, "UP")
    down = read_sample_shp(DOWN_SHP, "DOWN")

    if down.crs != up.crs:
        down = down.to_crs(up.crs)

    samples = [up, down]

    bg = try_read_full_csv_as_background(
        FULL_HEX_CSV,
        up_ids=set(up["hex_id_work"]),
        down_ids=set(down["hex_id_work"])
    )

    if bg is not None:
        if bg.crs != up.crs:
            bg = bg.to_crs(up.crs)
        samples.append(bg)

    all_samples = pd.concat(samples, ignore_index=True)
    all_samples = gpd.GeoDataFrame(all_samples, geometry="geometry", crs=up.crs)

    return all_samples


def clean_vod_values(values, scale):
    x = np.asarray(values, dtype="float64") * scale
    x[(x < VALID_VOD_MIN) | (x > VALID_VOD_MAX)] = np.nan
    return x


# ============================================================
# 3. 多进程 VOD 提取函数
# ============================================================

_WORKER_GEOMS = None
_WORKER_ZONAL_STAT = None
_WORKER_ALL_TOUCHED = None
_WORKER_VALID_MIN = None
_WORKER_VALID_MAX = None
_WORKER_SCALE = None


def _init_zonal_worker(geoms, zonal_stat, all_touched, valid_min, valid_max, scale):
    """
    多进程 worker 初始化。
    geoms 已经重投影到当前产品 raster CRS。
    """
    global _WORKER_GEOMS
    global _WORKER_ZONAL_STAT
    global _WORKER_ALL_TOUCHED
    global _WORKER_VALID_MIN
    global _WORKER_VALID_MAX
    global _WORKER_SCALE

    _WORKER_GEOMS = geoms
    _WORKER_ZONAL_STAT = zonal_stat
    _WORKER_ALL_TOUCHED = all_touched
    _WORKER_VALID_MIN = valid_min
    _WORKER_VALID_MAX = valid_max
    _WORKER_SCALE = scale


def _clean_vod_values_worker(values):
    x = np.asarray(values, dtype="float64") * _WORKER_SCALE
    x[(x < _WORKER_VALID_MIN) | (x > _WORKER_VALID_MAX)] = np.nan
    return x


def _extract_one_year_worker(task):
    """
    多进程下提取单个年份。
    注意：该函数必须在顶层，Windows 多进程才能运行。
    """
    year, tif_path = task

    with rasterio.open(tif_path) as src:
        nodata = src.nodata

        stats = zonal_stats(
            vectors=_WORKER_GEOMS,
            raster=tif_path,
            stats=[_WORKER_ZONAL_STAT],
            nodata=nodata,
            all_touched=_WORKER_ALL_TOUCHED,
            geojson_out=False
        )

    vals = [s.get(_WORKER_ZONAL_STAT, np.nan) for s in stats]
    vals = _clean_vod_values_worker(vals)

    return year, vals


def _extract_one_year_serial(year, tif_path, geoms, product_cfg):
    """
    单进程版本，用于调试或内存不足时运行。
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata

        stats = zonal_stats(
            vectors=geoms,
            raster=tif_path,
            stats=[ZONAL_STAT],
            nodata=nodata,
            all_touched=ALL_TOUCHED,
            geojson_out=False
        )

    vals = [s.get(ZONAL_STAT, np.nan) for s in stats]
    vals = clean_vod_values(vals, scale=product_cfg.get("scale", 1.0))

    return year, vals


def extract_annual_vod(product_name, product_cfg, samples_gdf):
    """
    加速版逐年提取 hex-level VOD。

    逻辑：
    1. 检查同一产品所有 tif 的 CRS 是否一致；
    2. 对该产品只重投影一次 hexagon；
    3. 将不同年份 tif 分配给多个进程并行提取；
    4. 输出 hex-year VOD 表。
    """
    year_files = list_product_tifs(product_name, product_cfg)
    years = [y for y, _ in year_files]

    raster_crs = check_product_crs_consistency(product_name, year_files)

    print(f"[INFO] {product_name} raster CRS:")
    print(raster_crs)

    print(f"[INFO] {product_name} 开始一次性重投影 hexagon 到 raster CRS")
    samples_in_raster_crs = samples_gdf.to_crs(raster_crs)

    geoms = list(samples_in_raster_crs.geometry)

    annual_df = samples_gdf[["hex_id_work", "optical_sample"]].copy()

    print(f"[INFO] {product_name} 开始 zonal extraction")
    print(f"[INFO] 样本数：{len(samples_gdf):,}")
    print(f"[INFO] 年份数：{len(years)}")
    print(f"[INFO] USE_MULTIPROCESSING = {USE_MULTIPROCESSING}, N_WORKERS = {N_WORKERS}")

    results = {}

    if USE_MULTIPROCESSING and N_WORKERS > 1:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_zonal_worker,
            initargs=(
                geoms,
                ZONAL_STAT,
                ALL_TOUCHED,
                VALID_VOD_MIN,
                VALID_VOD_MAX,
                product_cfg.get("scale", 1.0)
            )
        ) as executor:

            future_map = {
                executor.submit(_extract_one_year_worker, task): task[0]
                for task in year_files
            }

            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"{product_name} zonal extraction"
            ):
                year, vals = future.result()
                results[year] = vals

    else:
        for year, tif_path in tqdm(year_files, desc=f"{product_name} zonal extraction"):
            year, vals = _extract_one_year_serial(year, tif_path, geoms, product_cfg)
            results[year] = vals

    for year in years:
        annual_df[f"vod_{year}"] = results[year]

    return annual_df, years


# ============================================================
# 4. Mann-Kendall 和 Sen's slope
# ============================================================

def mann_kendall_sen(years, values):
    years = np.asarray(years, dtype="float64")
    values = np.asarray(values, dtype="float64")

    mask = np.isfinite(years) & np.isfinite(values)
    x = years[mask]
    y = values[mask]

    n = len(y)

    if n < 3:
        return np.nan, np.nan, np.nan, n

    s = 0
    slopes = []

    for i in range(n - 1):
        dy = y[(i + 1):] - y[i]
        dx = x[(i + 1):] - x[i]

        s += np.sum(np.sign(dy))

        valid_dx = dx != 0
        if np.any(valid_dx):
            slopes.extend((dy[valid_dx] / dx[valid_dx]).tolist())

    tau = s / (0.5 * n * (n - 1))

    _, counts = np.unique(y, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0

    if var_s <= 0:
        p_value = np.nan
    else:
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_value = 2 * (1 - norm.cdf(abs(z)))

    sen_slope = np.nanmedian(slopes) if len(slopes) > 0 else np.nan

    return tau, p_value, sen_slope, n


def calculate_hex_trends(annual_df, years, product_name):
    vod_cols = [f"vod_{y}" for y in years]
    min_valid_years = max(3, int(np.ceil(len(years) * MIN_VALID_FRAC)))

    values = annual_df[vod_cols].to_numpy(dtype="float64")

    rows = []

    for i in tqdm(range(len(annual_df)), desc=f"{product_name} trend calculation"):
        tau, p_value, slope, n_valid = mann_kendall_sen(years, values[i, :])
        rows.append((tau, p_value, slope, n_valid))

    trend_df = annual_df[["hex_id_work", "optical_sample"]].copy()

    trend_df[["vod_tau", "vod_mk_p", "vod_sen_slope", "n_valid_years"]] = pd.DataFrame(
        rows,
        index=trend_df.index
    )

    trend_df["product"] = product_name
    trend_df["period"] = f"{min(years)}-{max(years)}"
    trend_df["min_required_years"] = min_valid_years
    trend_df["is_valid_trend"] = trend_df["n_valid_years"] >= min_valid_years

    trend_df["positive_vod_slope"] = trend_df["vod_sen_slope"] > 0
    trend_df["negative_vod_slope"] = trend_df["vod_sen_slope"] < 0

    trend_df["significant_positive_vod_trend"] = (
        trend_df["is_valid_trend"]
        & (trend_df["vod_sen_slope"] > 0)
        & (trend_df["vod_mk_p"] < 0.05)
    )

    trend_df["significant_negative_vod_trend"] = (
        trend_df["is_valid_trend"]
        & (trend_df["vod_sen_slope"] < 0)
        & (trend_df["vod_mk_p"] < 0.05)
    )

    def directional_consistency(row):
        if not row["is_valid_trend"]:
            return np.nan

        if row["optical_sample"] == "UP":
            return row["vod_sen_slope"] > 0

        if row["optical_sample"] == "DOWN":
            return row["vod_sen_slope"] < 0

        return np.nan

    trend_df["directionally_consistent_with_optical_class"] = trend_df.apply(
        directional_consistency,
        axis=1
    )

    return trend_df


# ============================================================
# 5. 汇总与检验
# ============================================================

def summarize_product(trend_df):
    rows = []

    for (product, period, sample), sub in trend_df.groupby(["product", "period", "optical_sample"]):
        sub_valid = sub[sub["is_valid_trend"]].copy()
        slopes = sub_valid["vod_sen_slope"].dropna()

        if len(slopes) == 0:
            rows.append({
                "VOD product": product,
                "Period": period,
                "Optical sample": sample,
                "Valid hexagons": 0,
                "Median VOD Sen's slope": np.nan,
                "IQR of VOD Sen's slope": "",
                "Fraction with positive VOD slope (%)": np.nan,
                "Fraction with significant positive VOD trend (%)": np.nan,
                "Fraction with negative VOD slope (%)": np.nan,
                "Fraction with significant negative VOD trend (%)": np.nan,
                "Fraction directionally consistent with optical class (%)": np.nan,
            })
            continue

        q25 = np.nanpercentile(slopes, 25)
        q75 = np.nanpercentile(slopes, 75)

        if sample in ["UP", "DOWN"]:
            direction_frac = (
                np.nanmean(
                    sub_valid["directionally_consistent_with_optical_class"].astype(float)
                ) * 100
            )
        else:
            direction_frac = np.nan

        rows.append({
            "VOD product": product,
            "Period": period,
            "Optical sample": sample,
            "Valid hexagons": int(len(sub_valid)),
            "Median VOD Sen's slope": float(np.nanmedian(slopes)),
            "IQR of VOD Sen's slope": f"{q25:.6g}–{q75:.6g}",
            "Fraction with positive VOD slope (%)": float(np.nanmean(sub_valid["positive_vod_slope"]) * 100),
            "Fraction with significant positive VOD trend (%)": float(np.nanmean(sub_valid["significant_positive_vod_trend"]) * 100),
            "Fraction with negative VOD slope (%)": float(np.nanmean(sub_valid["negative_vod_slope"]) * 100),
            "Fraction with significant negative VOD trend (%)": float(np.nanmean(sub_valid["significant_negative_vod_trend"]) * 100),
            "Fraction directionally consistent with optical class (%)": direction_frac,
        })

    summary = pd.DataFrame(rows)

    sample_order = {
        "UP": 1,
        "DOWN": 2,
        "background_nonUP": 3,
    }

    summary["_sample_order"] = summary["Optical sample"].map(sample_order).fillna(99)
    summary = summary.sort_values(["VOD product", "_sample_order"]).drop(columns="_sample_order")

    return summary


def cliffs_delta(x, y):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    max_n = 20000
    rng = np.random.default_rng(RANDOM_SEED)

    if len(x) > max_n:
        x = rng.choice(x, size=max_n, replace=False)

    if len(y) > max_n:
        y = rng.choice(y, size=max_n, replace=False)

    greater = 0
    less = 0

    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)

    return (greater - less) / (len(x) * len(y))


def pairwise_tests(trend_df):
    rows = []

    sub_all = trend_df[trend_df["is_valid_trend"]].copy()

    comparisons = [
        ("UP", "DOWN"),
        ("UP", "background_nonUP"),
        ("DOWN", "background_nonUP"),
    ]

    if len(sub_all) == 0:
        return pd.DataFrame(rows)

    product = sub_all["product"].iloc[0]

    for group1, group2 in comparisons:
        x1 = sub_all.loc[sub_all["optical_sample"] == group1, "vod_sen_slope"].dropna().values
        x2 = sub_all.loc[sub_all["optical_sample"] == group2, "vod_sen_slope"].dropna().values

        if len(x1) == 0 or len(x2) == 0:
            continue

        try:
            _, p_value = mannwhitneyu(x1, x2, alternative="two-sided")
        except Exception:
            p_value = np.nan

        rows.append({
            "VOD product": product,
            "Comparison": f"{group1} vs {group2}",
            "n_group_1": len(x1),
            "n_group_2": len(x2),
            "Median slope group 1": float(np.nanmedian(x1)),
            "Median slope group 2": float(np.nanmedian(x2)),
            "Median difference group1_minus_group2": float(np.nanmedian(x1) - np.nanmedian(x2)),
            "Mann-Whitney U p": p_value,
            "Cliff's delta": cliffs_delta(x1, x2),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. 单产品运行与分开导出
# ============================================================

def run_one_product(product_name, product_cfg, samples_gdf):
    product_out_dir = os.path.join(OUT_DIR, product_name)
    os.makedirs(product_out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[START] Processing product: {product_name}")
    print("=" * 80)

    annual_df, years = extract_annual_vod(product_name, product_cfg, samples_gdf)

    if SAVE_ANNUAL_VALUES:
        annual_path = os.path.join(product_out_dir, f"{product_name}_annual_values.csv")
        annual_df.to_csv(annual_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {product_name} annual values: {annual_path}")

    trend_df = calculate_hex_trends(annual_df, years, product_name)

    trend_path = os.path.join(product_out_dir, f"{product_name}_hex_trends.csv")
    trend_df.to_csv(trend_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {product_name} hex trends: {trend_path}")

    summary_df = summarize_product(trend_df)

    summary_path = os.path.join(product_out_dir, f"{product_name}_summary_table.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {product_name} summary table: {summary_path}")

    tests_df = pairwise_tests(trend_df)

    tests_path = os.path.join(product_out_dir, f"{product_name}_pairwise_tests.csv")
    tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {product_name} pairwise tests: {tests_path}")

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))

    if len(tests_df) > 0:
        print("\n[PAIRWISE TESTS]")
        print(tests_df.to_string(index=False))
    else:
        print("\n[PAIRWISE TESTS] 未生成。当前默认只做 UP/DOWN；若需要 background，请设置 MAX_BACKGROUND_HEX > 0 且 CSV 需要包含 geometry。")

    print(f"\n[DONE] {product_name}")

    return annual_df, trend_df, summary_df, tests_df


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("[INFO] 当前工作目录：", os.getcwd())

    samples_gdf = build_samples()

    print("\n[INFO] 最终参与提取的样本构成：")
    print(samples_gdf["optical_sample"].value_counts())

    print("\n[INFO] 样本 CRS：")
    print(samples_gdf.crs)

    for product_name, product_cfg in PRODUCTS.items():
        run_one_product(product_name, product_cfg, samples_gdf)

    print("\n" + "=" * 80)
    print("[ALL DONE] 两个 VOD 产品均已分开导出。")
    print("[OUTPUT DIR]", os.path.abspath(OUT_DIR))
    print("=" * 80)


if __name__ == "__main__":
    main()