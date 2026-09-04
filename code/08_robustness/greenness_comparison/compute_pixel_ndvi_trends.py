# -*- coding: utf-8 -*-
"""
逐像素 MODIS NDVI Mann-Kendall 检验与 Sen's slope 估计

输出波段：
Band 1: trend_class
        1  = UP
        0  = 无显著趋势
       -1  = DOWN
       -9999 = NoData、有效年份不足或 MK 方差为 0

Band 2: Sen's slope，单位为 NDVI year-1
Band 3: Mann-Kendall tau
Band 4: Mann-Kendall two-sided p value
Band 5: 有效年份数量

趋势分类标准：
UP:
    tau > 0
    p < 0.05
    Sen's slope > 0

DOWN:
    tau < 0
    p < 0.05
    Sen's slope < 0
"""

import glob
import math
import os
import re
import shutil
from multiprocessing import Pool, cpu_count, freeze_support

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from tqdm import tqdm
import warnings
from scipy.special import erfc

# ==================== 用户配置 ====================

# 年度 NDVI 文件目录
tif_folder = "MODIS_NDVI"

# 文件示例：
# MODIS_NDVI_2000.tif
# MODIS_NDVI_2001.tif
# ...
tif_pattern = "MODIS_NDVI_*.tif"

# 干旱区矢量掩膜
mask_shp = "Drylands_dataset_fixed/drylands_8857.shp"

# 最终输出
output_tif = "MODIS_NDVI_pixel_MK_Sen_8857.tif"

# 临时文件夹
temp_folder = "_temp_MODIS_pixel_MK_Sen"

# 最终等面积投影
target_crs = "EPSG:8857"

# 最终输出分辨率，单位为 m
# 如果希望输出约 500 m 等面积像元，保持 500
target_resolution_m = 500.0

# 每次在一个进程内计算的像元数量
# 数值越大速度通常越快，但内存占用也越高
pixel_batch_size = 20000

# Windows 下建议先限制为 4–6 个进程，避免多个进程同时占用过多内存
n_jobs = min(max(1, cpu_count() - 1), 4)

# MODIS NDVI 缩放系数
#
# None:
#   自动读取栅格 metadata 中的 scale；
#   如果 metadata 未记录 scale，则根据数据范围自动判断。
#
# 0.0001:
#   适用于原始 MODIS int16 NDVI，例如 5000 表示 0.5。
#
# 1.0:
#   适用于已经缩放至 -0.2 至 1.0 的 NDVI。
ndvi_scale_factor = None

ndvi_add_offset = 0.0

# MODIS NDVI 标准有效范围
valid_ndvi_min = -0.2
valid_ndvi_max = 1.0

# 趋势检验参数
alpha = 0.05
min_valid_years = 3

# 分块大小
block_size = 512

# 多进程数量
n_jobs = max(1, cpu_count() - 6)

# 干旱区边界像元判定方式
# False：仅像元中心位于多边形内部时纳入
# True ：与多边形接触的像元均纳入
mask_all_touched = False

# 年度栅格若不完全对齐，自动以第一期栅格为参考，
# 使用最近邻重采样进行读取
alignment_resampling = Resampling.nearest

# 是否保留中间文件
keep_temp_files = False

# 输出 NoData
output_nodata = -9999.0

# ==================================================


# 多进程 worker 中使用的全局变量
_WORKER_SOURCE_DATASETS = []
_WORKER_READERS = []
_WORKER_MASK_DATASET = None
_WORKER_YEARS = None
_WORKER_SCALE_FACTOR = None
_WORKER_ADD_OFFSET = None
_WORKER_VALID_MIN = None
_WORKER_VALID_MAX = None
_WORKER_ALPHA = None
_WORKER_MIN_VALID = None
_WORKER_OUTPUT_NODATA = None


def extract_year(filepath):
    """
    从文件名中提取四位年份。
    """
    filename = os.path.basename(filepath)
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", filename)

    if match is None:
        raise ValueError(f"无法从文件名中识别年份：{filename}")

    return int(match.group(1))


def collect_annual_files():
    """
    收集并按年份排序年度 NDVI 文件。
    """
    search_pattern = os.path.join(tif_folder, tif_pattern)
    filepaths = glob.glob(search_pattern)

    if len(filepaths) == 0:
        raise FileNotFoundError(
            f"未找到年度 NDVI 文件：{search_pattern}"
        )

    records = []

    for filepath in filepaths:
        year = extract_year(filepath)
        records.append((year, os.path.abspath(filepath)))

    records.sort(key=lambda item: item[0])

    years = [item[0] for item in records]
    paths = [item[1] for item in records]

    if len(years) != len(set(years)):
        duplicated = sorted(
            year for year in set(years) if years.count(year) > 1
        )
        raise ValueError(f"检测到重复年份：{duplicated}")

    if len(years) < min_valid_years:
        raise ValueError(
            f"年度栅格数量仅为 {len(years)}，"
            f"少于 min_valid_years={min_valid_years}"
        )

    return years, paths


def detect_scale_factor(first_raster):
    """
    确定 NDVI 缩放系数。

    优先级：
    1. 用户显式配置；
    2. GeoTIFF metadata 中记录的 scale；
    3. 根据样本数据范围自动判断。
    """
    if ndvi_scale_factor is not None:
        return float(ndvi_scale_factor)

    with rasterio.open(first_raster) as src:
        if src.scales and len(src.scales) >= 1:
            metadata_scale = src.scales[0]

            if (
                metadata_scale is not None
                and np.isfinite(metadata_scale)
                and not np.isclose(metadata_scale, 1.0)
            ):
                return float(metadata_scale)

        sample_height = min(src.height, 1024)
        sample_width = min(src.width, 1024)

        sample = src.read(
            1,
            out_shape=(sample_height, sample_width),
            masked=True
        )

        values = sample.compressed().astype(np.float64)

        if values.size == 0:
            raise ValueError(
                "无法从第一期栅格读取有效值，"
                "请显式设置 ndvi_scale_factor。"
            )

        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            raise ValueError(
                "第一期栅格中没有有限值，"
                "请检查数据或显式设置 ndvi_scale_factor。"
            )

        q95 = np.nanpercentile(np.abs(finite_values), 95)

        # 原始 MODIS NDVI 通常位于 -2000 至 10000
        if q95 > 2.0:
            return 0.0001

        return 1.0


def affine_equal(transform_a, transform_b, tolerance=1e-9):
    """
    判断两个仿射变换是否近似一致。
    """
    values_a = np.asarray(tuple(transform_a), dtype=np.float64)
    values_b = np.asarray(tuple(transform_b), dtype=np.float64)

    return np.allclose(
        values_a,
        values_b,
        rtol=0.0,
        atol=tolerance
    )


def raster_is_aligned(dataset, reference):
    """
    判断年度栅格是否与参考栅格完全对齐。
    """
    return (
        dataset.crs == reference["crs"]
        and dataset.width == reference["width"]
        and dataset.height == reference["height"]
        and affine_equal(dataset.transform, reference["transform"])
    )


def generate_windows(width, height, size):
    """
    生成规则分块窗口。
    """
    window_index = 0

    for row_off in range(0, height, size):
        current_height = min(size, height - row_off)

        for col_off in range(0, width, size):
            current_width = min(size, width - col_off)

            yield (
                window_index,
                int(col_off),
                int(row_off),
                int(current_width),
                int(current_height)
            )

            window_index += 1


def create_native_dryland_mask(
    reference_path,
    mask_path,
    mask_output_path
):
    """
    在参考 MODIS 网格上创建干旱区掩膜。

    为避免一次性建立全球尺度数组，按窗口执行矢量栅格化。
    同时返回包含干旱区像元的活动窗口列表。
    """
    print("\n创建参考网格上的干旱区掩膜……")

    with rasterio.open(reference_path) as reference:
        if reference.crs is None:
            raise ValueError(
                f"参考栅格缺少 CRS：{reference_path}"
            )

        drylands = gpd.read_file(mask_path)

        if drylands.crs is None:
            raise ValueError(
                f"干旱区矢量缺少 CRS：{mask_path}"
            )

        drylands = drylands[
            drylands.geometry.notnull()
            & (~drylands.geometry.is_empty)
        ].copy()

        drylands = drylands.to_crs(reference.crs)
        drylands = drylands.reset_index(drop=True)

        if len(drylands) == 0:
            raise ValueError("干旱区矢量中没有有效几何对象。")

        spatial_index = drylands.sindex

        mask_profile = reference.profile.copy()
        mask_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="DEFLATE",
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="YES",
            SPARSE_OK="TRUE"
        )

        all_windows = list(
            generate_windows(
                reference.width,
                reference.height,
                block_size
            )
        )

        active_windows = []

        with rasterio.open(
            mask_output_path,
            "w",
            **mask_profile
        ) as mask_dst:

            iterator = tqdm(
                all_windows,
                desc="Rasterizing dryland mask",
                unit="window"
            )

            for task in iterator:
                (
                    window_index,
                    col_off,
                    row_off,
                    width,
                    height
                ) = task

                window = Window(
                    col_off,
                    row_off,
                    width,
                    height
                )

                current_bounds = window_bounds(
                    window,
                    reference.transform
                )

                candidate_indices = list(
                    spatial_index.intersection(current_bounds)
                )

                if len(candidate_indices) == 0:
                    continue

                window_box = box(*current_bounds)

                shapes = []

                for geometry in drylands.geometry.iloc[candidate_indices]:
                    if geometry.intersects(window_box):
                        shapes.append((geometry, 1))

                if len(shapes) == 0:
                    continue

                mask_array = rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=window_transform(
                        window,
                        reference.transform
                    ),
                    fill=0,
                    default_value=1,
                    all_touched=mask_all_touched,
                    dtype="uint8"
                )

                if np.any(mask_array == 1):
                    mask_dst.write(
                        mask_array,
                        1,
                        window=window
                    )

                    active_windows.append(task)

        print(
            f"总窗口数：{len(all_windows):,}\n"
            f"包含干旱区像元的窗口数：{len(active_windows):,}"
        )

        return active_windows


def mk_sen_block(
    stack,
    years,
    min_valid,
    significance_level
):
    """
    使用 NumPy 分批执行逐像素 Mann-Kendall 检验和 Sen's slope 估计。

    参数
    ----------
    stack : np.ndarray
        形状为 [time, height, width]。
        无效值必须为 np.nan。

    years : np.ndarray
        与时间维度对应的实际年份，例如：
        [2000, 2001, ..., 2022]

    min_valid : int
        最少有效年份数。

    significance_level : float
        显著性水平，例如 0.05。

    返回
    ----------
    trend_class : int8
        1 = UP
        0 = 无显著趋势
        -1 = DOWN
        -128 = 有效年份不足或 MK 方差为 0

    sen_slope : float32
        Sen's slope，单位为 NDVI year-1。

    mk_tau : float32
        Mann-Kendall tau。

    mk_p : float32
        双侧 p 值。

    n_valid_output : int16
        有效年份数。
    """

    n_time, height, width = stack.shape
    n_pixels = height * width

    flat_stack = stack.reshape(n_time, n_pixels).astype(
        np.float32,
        copy=False
    )

    years = np.asarray(years, dtype=np.float64)

    # 所有年份组合 i < j
    pair_i, pair_j = np.triu_indices(n_time, k=1)

    # Sen slope 必须除以实际年份间隔
    year_differences = (
        years[pair_j] - years[pair_i]
    ).astype(np.float32)

    year_differences = year_differences[:, np.newaxis]

    trend_class = np.full(
        n_pixels,
        -128,
        dtype=np.int8
    )

    sen_slope = np.full(
        n_pixels,
        np.nan,
        dtype=np.float32
    )

    mk_tau = np.full(
        n_pixels,
        np.nan,
        dtype=np.float32
    )

    mk_p = np.full(
        n_pixels,
        np.nan,
        dtype=np.float32
    )

    n_valid_output = np.zeros(
        n_pixels,
        dtype=np.int16
    )

    # 分批处理，控制内存占用
    for start in range(0, n_pixels, pixel_batch_size):

        end = min(
            start + pixel_batch_size,
            n_pixels
        )

        current = flat_stack[:, start:end]
        current_pixel_count = end - start

        finite = np.isfinite(current)

        n_valid = finite.sum(
            axis=0
        ).astype(np.int16)

        n_valid_output[start:end] = n_valid

        eligible = n_valid >= min_valid

        if not np.any(eligible):
            continue

        # --------------------------------------------------
        # 1. Mann-Kendall S 统计量
        # --------------------------------------------------

        # 形状：
        # [年份组合数, 当前批次像元数]
        pair_differences = (
            current[pair_j, :]
            - current[pair_i, :]
        ).astype(np.float32, copy=False)

        valid_pairs = np.isfinite(pair_differences)

        positive_count = np.count_nonzero(
            pair_differences > 0,
            axis=0
        )

        negative_count = np.count_nonzero(
            pair_differences < 0,
            axis=0
        )

        s_statistic = (
            positive_count - negative_count
        ).astype(np.float64)

        # --------------------------------------------------
        # 2. Sen's slope
        # --------------------------------------------------

        np.divide(
            pair_differences,
            year_differences,
            out=pair_differences,
            where=valid_pairs
        )

        pair_differences[~valid_pairs] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning
            )

            batch_sen_slope = np.nanmedian(
                pair_differences,
                axis=0
            )

        # 释放不再需要的大数组
        del pair_differences
        del valid_pairs

        # --------------------------------------------------
        # 3. MK ties 校正
        # --------------------------------------------------

        # NaN 会被排在每一列末尾
        sorted_values = np.sort(
            current,
            axis=0
        )

        tie_correction = np.zeros(
            current_pixel_count,
            dtype=np.float64
        )

        run_length = np.ones(
            current_pixel_count,
            dtype=np.int16
        )

        previous_finite = np.isfinite(
            sorted_values[0, :]
        )

        for time_index in range(1, n_time):

            current_finite = np.isfinite(
                sorted_values[time_index, :]
            )

            same_value = (
                current_finite
                & previous_finite
                & (
                    sorted_values[time_index, :]
                    == sorted_values[time_index - 1, :]
                )
            )

            # 当前相同值序列在此处结束
            ended = (
                (~same_value)
                & (run_length > 1)
            )

            if np.any(ended):
                tie_length = run_length[
                    ended
                ].astype(np.float64)

                tie_correction[ended] += (
                    tie_length
                    * (tie_length - 1.0)
                    * (2.0 * tie_length + 5.0)
                )

            run_length[same_value] += 1
            run_length[~same_value] = 1

            previous_finite = current_finite

        # 处理一直延续到最后一个有效位置的 ties
        ended = run_length > 1

        if np.any(ended):
            tie_length = run_length[
                ended
            ].astype(np.float64)

            tie_correction[ended] += (
                tie_length
                * (tie_length - 1.0)
                * (2.0 * tie_length + 5.0)
            )

        del sorted_values

        # --------------------------------------------------
        # 4. MK 方差、Z 值、p 值与 tau
        # --------------------------------------------------

        n_float = n_valid.astype(np.float64)

        variance_s = (
            n_float
            * (n_float - 1.0)
            * (2.0 * n_float + 5.0)
            - tie_correction
        ) / 18.0

        # 按论文规则：
        # 至少 3 个有效年份，且 MK 方差必须大于 0
        statistically_valid = (
            eligible
            & np.isfinite(variance_s)
            & (variance_s > 0.0)
        )

        if not np.any(statistically_valid):
            continue

        z_value = np.zeros(
            current_pixel_count,
            dtype=np.float64
        )

        positive_s = (
            statistically_valid
            & (s_statistic > 0)
        )

        negative_s = (
            statistically_valid
            & (s_statistic < 0)
        )

        zero_s = (
            statistically_valid
            & (s_statistic == 0)
        )

        z_value[positive_s] = (
            s_statistic[positive_s] - 1.0
        ) / np.sqrt(
            variance_s[positive_s]
        )

        z_value[negative_s] = (
            s_statistic[negative_s] + 1.0
        ) / np.sqrt(
            variance_s[negative_s]
        )

        z_value[zero_s] = 0.0

        p_value = np.full(
            current_pixel_count,
            np.nan,
            dtype=np.float64
        )

        p_value[statistically_valid] = erfc(
            np.abs(
                z_value[statistically_valid]
            ) / np.sqrt(2.0)
        )

        denominator = (
            0.5
            * n_float
            * (n_float - 1.0)
        )

        tau_value = np.full(
            current_pixel_count,
            np.nan,
            dtype=np.float64
        )

        tau_value[statistically_valid] = (
            s_statistic[statistically_valid]
            / denominator[statistically_valid]
        )

        # --------------------------------------------------
        # 5. 按论文规则识别 UP 和 DOWN
        # --------------------------------------------------

        batch_trend = np.full(
            current_pixel_count,
            -128,
            dtype=np.int8
        )

        # 有效但未达到显著性标准
        batch_trend[statistically_valid] = 0

        up_mask = (
            statistically_valid
            & (tau_value > 0.0)
            & (p_value < significance_level)
            & (batch_sen_slope > 0.0)
        )

        down_mask = (
            statistically_valid
            & (tau_value < 0.0)
            & (p_value < significance_level)
            & (batch_sen_slope < 0.0)
        )

        batch_trend[up_mask] = 1
        batch_trend[down_mask] = -1

        # --------------------------------------------------
        # 6. 写入当前批次结果
        # --------------------------------------------------

        trend_class[start:end] = batch_trend

        current_slope_output = sen_slope[start:end]
        current_tau_output = mk_tau[start:end]
        current_p_output = mk_p[start:end]

        current_slope_output[statistically_valid] = (
            batch_sen_slope[statistically_valid]
        ).astype(np.float32)

        current_tau_output[statistically_valid] = (
            tau_value[statistically_valid]
        ).astype(np.float32)

        current_p_output[statistically_valid] = (
            p_value[statistically_valid]
        ).astype(np.float32)

    return (
        trend_class.reshape(height, width),
        sen_slope.reshape(height, width),
        mk_tau.reshape(height, width),
        mk_p.reshape(height, width),
        n_valid_output.reshape(height, width)
    )

def initialize_worker(
    annual_paths,
    years,
    reference_definition,
    native_mask_path,
    scale_factor,
    add_offset,
    valid_min,
    valid_max,
    significance_level,
    minimum_valid,
    nodata_value
):
    """
    初始化每个多进程 worker。

    每个 worker 独立打开年度栅格，避免跨进程共享
    GDAL/rasterio 数据集句柄。
    """
    global _WORKER_SOURCE_DATASETS
    global _WORKER_READERS
    global _WORKER_MASK_DATASET
    global _WORKER_YEARS
    global _WORKER_SCALE_FACTOR
    global _WORKER_ADD_OFFSET
    global _WORKER_VALID_MIN
    global _WORKER_VALID_MAX
    global _WORKER_ALPHA
    global _WORKER_MIN_VALID
    global _WORKER_OUTPUT_NODATA

    _WORKER_SOURCE_DATASETS = []
    _WORKER_READERS = []

    for path in annual_paths:
        source = rasterio.open(path)
        _WORKER_SOURCE_DATASETS.append(source)

        if raster_is_aligned(source, reference_definition):
            reader = source
        else:
            reader = WarpedVRT(
                source,
                crs=reference_definition["crs"],
                transform=reference_definition["transform"],
                width=reference_definition["width"],
                height=reference_definition["height"],
                resampling=alignment_resampling
            )

        _WORKER_READERS.append(reader)

    _WORKER_MASK_DATASET = rasterio.open(native_mask_path)
    _WORKER_YEARS = np.asarray(years, dtype=np.int32)
    _WORKER_SCALE_FACTOR = float(scale_factor)
    _WORKER_ADD_OFFSET = float(add_offset)
    _WORKER_VALID_MIN = float(valid_min)
    _WORKER_VALID_MAX = float(valid_max)
    _WORKER_ALPHA = float(significance_level)
    _WORKER_MIN_VALID = int(minimum_valid)
    _WORKER_OUTPUT_NODATA = float(nodata_value)


def process_window(task):
    """
    处理一个活动窗口。
    """
    (
        window_index,
        col_off,
        row_off,
        width,
        height
    ) = task

    window = Window(
        col_off,
        row_off,
        width,
        height
    )

    dryland_mask = _WORKER_MASK_DATASET.read(
        1,
        window=window
    ).astype(bool)

    n_years = len(_WORKER_READERS)

    stack = np.full(
        (n_years, height, width),
        np.nan,
        dtype=np.float32
    )

    for year_index, reader in enumerate(_WORKER_READERS):

        annual_data = reader.read(
            1,
            window=window,
            masked=True
        )

        annual_data = annual_data.filled(
            np.nan
        ).astype(np.float32)

        annual_data = (
            annual_data * _WORKER_SCALE_FACTOR
            + _WORKER_ADD_OFFSET
        )

        invalid = (
            ~np.isfinite(annual_data)
            | (annual_data < _WORKER_VALID_MIN)
            | (annual_data > _WORKER_VALID_MAX)
            | (~dryland_mask)
        )

        annual_data[invalid] = np.nan
        stack[year_index] = annual_data

    (
        trend_class,
        sen_slope,
        mk_tau,
        mk_p,
        n_valid
    ) = mk_sen_block(
        stack,
        _WORKER_YEARS,
        _WORKER_MIN_VALID,
        _WORKER_ALPHA
    )

    output = np.full(
        (5, height, width),
        _WORKER_OUTPUT_NODATA,
        dtype=np.float32
    )

    statistically_valid = (
        dryland_mask
        & (trend_class != -128)
    )

    output[0, statistically_valid] = trend_class[
        statistically_valid
    ].astype(np.float32)

    output[1, statistically_valid] = sen_slope[
        statistically_valid
    ]

    output[2, statistically_valid] = mk_tau[
        statistically_valid
    ]

    output[3, statistically_valid] = mk_p[
        statistically_valid
    ]

    output[4, statistically_valid] = n_valid[
        statistically_valid
    ].astype(np.float32)

    return (
        window_index,
        col_off,
        row_off,
        width,
        height,
        output
    )


def create_native_trend_raster(
    annual_paths,
    years,
    reference_path,
    native_mask_path,
    active_windows,
    native_output_path,
    scale_factor
):
    """
    在参考 MODIS 网格上执行逐像素趋势计算。
    """
    print("\n执行逐像素 Mann-Kendall 和 Sen's slope 计算……")
    print(f"年度数量：{len(years)}")
    print(f"年份范围：{years[0]}–{years[-1]}")
    print(f"NDVI scale factor：{scale_factor}")
    print(f"并行进程数：{n_jobs}")
    print(f"活动窗口数：{len(active_windows):,}")

    with rasterio.open(reference_path) as reference:

        reference_definition = {
            "crs": reference.crs,
            "transform": reference.transform,
            "width": reference.width,
            "height": reference.height
        }

        output_profile = reference.profile.copy()
        output_profile.update(
            driver="GTiff",
            count=5,
            dtype="float32",
            nodata=output_nodata,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="YES",
            SPARSE_OK="TRUE"
        )

        with rasterio.open(
            native_output_path,
            "w",
            **output_profile
        ) as output_dataset:

            output_dataset.set_band_description(
                1,
                "trend_class_-1_DOWN_0_nonsignificant_1_UP"
            )
            output_dataset.set_band_description(
                2,
                "Sen_slope_NDVI_per_year"
            )
            output_dataset.set_band_description(
                3,
                "Mann_Kendall_tau"
            )
            output_dataset.set_band_description(
                4,
                "Mann_Kendall_two_sided_p_value"
            )
            output_dataset.set_band_description(
                5,
                "number_of_valid_years"
            )

            output_dataset.update_tags(
                trend_definition=(
                    "UP: tau>0, p<0.05, Sen slope>0; "
                    "DOWN: tau<0, p<0.05, Sen slope<0"
                ),
                minimum_valid_years=min_valid_years,
                zero_MK_variance="excluded",
                ndvi_scale_factor=scale_factor,
                ndvi_add_offset=ndvi_add_offset,
                first_year=years[0],
                last_year=years[-1],
                analysis_grid="native reference MODIS grid"
            )

            pool = Pool(
                processes=n_jobs,
                initializer=initialize_worker,
                initargs=(
                    annual_paths,
                    years,
                    reference_definition,
                    native_mask_path,
                    scale_factor,
                    ndvi_add_offset,
                    valid_ndvi_min,
                    valid_ndvi_max,
                    alpha,
                    min_valid_years,
                    output_nodata
                )
            )

            try:
                iterator = pool.imap_unordered(
                    process_window,
                    active_windows,
                    chunksize=1
                )

                for result in tqdm(
                    iterator,
                    total=len(active_windows),
                    desc="Pixel MK-Sen",
                    unit="window"
                ):
                    (
                        window_index,
                        col_off,
                        row_off,
                        width,
                        height,
                        output_array
                    ) = result

                    current_window = Window(
                        col_off,
                        row_off,
                        width,
                        height
                    )

                    output_dataset.write(
                        output_array,
                        window=current_window
                    )

            finally:
                pool.close()
                pool.join()


def align_bounds_to_resolution(bounds, resolution):
    """
    将目标范围对齐至固定分辨率。
    """
    left, bottom, right, top = bounds

    aligned_left = math.floor(left / resolution) * resolution
    aligned_bottom = math.floor(bottom / resolution) * resolution
    aligned_right = math.ceil(right / resolution) * resolution
    aligned_top = math.ceil(top / resolution) * resolution

    return (
        aligned_left,
        aligned_bottom,
        aligned_right,
        aligned_top
    )


def reproject_result_to_8857(
    native_result_path,
    mask_path,
    final_output_path
):
    """
    将趋势结果以最近邻方式投影至 EPSG:8857。

    所有波段均使用最近邻重投影，以避免生成不存在的
    趋势类别、p 值、tau 或 Sen slope 混合值。
    """
    print("\n将趋势结果投影至 EPSG:8857……")

    drylands_target = gpd.read_file(mask_path)

    if drylands_target.crs is None:
        raise ValueError("干旱区矢量缺少 CRS。")

    drylands_target = drylands_target[
        drylands_target.geometry.notnull()
        & (~drylands_target.geometry.is_empty)
    ].to_crs(target_crs)

    raw_bounds = tuple(drylands_target.total_bounds)

    (
        left,
        bottom,
        right,
        top
    ) = align_bounds_to_resolution(
        raw_bounds,
        target_resolution_m
    )

    target_width = int(
        math.ceil((right - left) / target_resolution_m)
    )

    target_height = int(
        math.ceil((top - bottom) / target_resolution_m)
    )

    target_transform = from_origin(
        left,
        top,
        target_resolution_m,
        target_resolution_m
    )

    with rasterio.open(native_result_path) as source:

        target_profile = source.profile.copy()
        target_profile.update(
            driver="GTiff",
            crs=target_crs,
            transform=target_transform,
            width=target_width,
            height=target_height,
            count=5,
            dtype="float32",
            nodata=output_nodata,
            compress="DEFLATE",
            predictor=3,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="YES",
            SPARSE_OK="TRUE"
        )

        with rasterio.open(
            final_output_path,
            "w",
            **target_profile
        ) as destination:

            for band_index in range(1, 6):

                reproject(
                    source=rasterio.band(
                        source,
                        band_index
                    ),
                    destination=rasterio.band(
                        destination,
                        band_index
                    ),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=output_nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=output_nodata,
                    resampling=Resampling.nearest,
                    num_threads=n_jobs,
                    init_dest_nodata=True
                )

                description = source.descriptions[
                    band_index - 1
                ]

                if description:
                    destination.set_band_description(
                        band_index,
                        description
                    )

            destination.update_tags(
                **source.tags(),
                output_crs=target_crs,
                output_resolution_m=target_resolution_m,
                reprojection_resampling="nearest"
            )

    print(f"EPSG:8857 输出完成：{final_output_path}")


def print_output_summary(output_path):
    """
    统计最终输出中 UP、DOWN 和非显著像元数量。
    """
    up_count = 0
    down_count = 0
    nonsignificant_count = 0
    valid_count = 0

    with rasterio.open(output_path) as dataset:

        for _, window in dataset.block_windows(1):

            values = dataset.read(
                1,
                window=window
            )

            valid = values != output_nodata

            if not np.any(valid):
                continue

            current = values[valid]

            up_count += int(np.count_nonzero(current == 1))
            down_count += int(np.count_nonzero(current == -1))
            nonsignificant_count += int(
                np.count_nonzero(current == 0)
            )
            valid_count += int(current.size)

    print("\n================ 输出统计 ================")
    print(f"有效趋势像元：{valid_count:,}")
    print(f"UP 像元：{up_count:,}")
    print(f"DOWN 像元：{down_count:,}")
    print(f"非显著像元：{nonsignificant_count:,}")

    if valid_count > 0:
        print(
            f"UP 比例：{up_count / valid_count * 100:.4f}%"
        )
        print(
            f"DOWN 比例：{down_count / valid_count * 100:.4f}%"
        )

    print("==========================================")


def main():

    os.makedirs(temp_folder, exist_ok=True)

    years, annual_paths = collect_annual_files()

    print("================ 输入数据 ================")

    for year, path in zip(years, annual_paths):
        print(f"{year}: {path}")

    reference_path = annual_paths[0]

    scale_factor = detect_scale_factor(reference_path)

    print("\n================ 参数 ================")
    print(f"参考栅格：{reference_path}")
    print(f"缩放系数：{scale_factor}")
    print(f"显著性水平：{alpha}")
    print(f"最少有效年份：{min_valid_years}")
    print(f"目标 CRS：{target_crs}")
    print(f"目标分辨率：{target_resolution_m} m")
    print(f"进程数：{n_jobs}")
    print("======================================")

    native_mask_path = os.path.join(
        temp_folder,
        "dryland_mask_native_grid.tif"
    )

    native_result_path = os.path.join(
        temp_folder,
        "MODIS_NDVI_pixel_MK_Sen_native.tif"
    )

    active_windows = create_native_dryland_mask(
        reference_path=reference_path,
        mask_path=mask_shp,
        mask_output_path=native_mask_path
    )

    if len(active_windows) == 0:
        raise RuntimeError(
            "干旱区掩膜与参考栅格没有重叠像元。"
        )

    create_native_trend_raster(
        annual_paths=annual_paths,
        years=years,
        reference_path=reference_path,
        native_mask_path=native_mask_path,
        active_windows=active_windows,
        native_output_path=native_result_path,
        scale_factor=scale_factor
    )

    reproject_result_to_8857(
        native_result_path=native_result_path,
        mask_path=mask_shp,
        final_output_path=output_tif
    )

    print_output_summary(output_tif)

    if not keep_temp_files:
        shutil.rmtree(
            temp_folder,
            ignore_errors=True
        )
        print("\n临时文件已清理。")
    else:
        print(f"\n临时文件保留于：{temp_folder}")

    print(f"\n最终结果：{output_tif}")


if __name__ == "__main__":
    freeze_support()
    main()