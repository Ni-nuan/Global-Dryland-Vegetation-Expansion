# -*- coding: utf-8 -*-
"""
MODIS逐像素NDVI趋势与100 km²格网植被覆盖趋势的空间对应分析

研究目的
--------
比较两类研究对象：

1. 像素级NDVI趋势
   表征传统逐像素长时序绿度变化：
   - Pixel UP：显著绿度增加
   - Pixel DOWN：显著绿度下降

2. 格网级植被覆盖比例趋势
   表征本文关注的阈值式植被覆盖扩张或收缩：
   - Hex UP：格网内超过NDVI阈值的面积比例显著增加
   - Hex DOWN：格网内超过NDVI阈值的面积比例显著减少

格网分类标准
------------
Hex UP:
    tau > 0
    pvalue < 0.05
    Sen's slope > 0

Hex DOWN:
    tau < 0
    pvalue < 0.05
    Sen's slope < 0

Hex Other:
    具有有效趋势统计结果，但不满足UP或DOWN标准

输入
----
1. MODIS_NDVI_pixel_MK_Sen_8857.tif
   Band 1:
       1 = Pixel UP
       0 = Pixel non-significant
      -1 = Pixel DOWN
   -9999 = NoData

2. Hex_data/NDVI_trend_hex_100.shp
   必须包含：
       tau
       pvalue
       slope

输出
----
1. MODIS_pixel_hex_overlap_8857.tif
   Band 1：像素趋势类别
   Band 2：格网趋势类别
   Band 3：像素—格网组合类别

2. pixel_hex_overlap_matrix.csv
   3×3完整交叉统计矩阵

3. pixel_hex_overlap_metrics.csv
   关键统计指标

4. Hex_data/NDVI_trend_hex_100_with_class.gpkg
   添加UP、DOWN、Other类别后的格网文件
"""

import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm


# ==================== 用户配置 ====================

# 像素级NDVI趋势栅格
pixel_trend_tif = "MODIS_NDVI_pixel_MK_Sen_8857.tif"

# 100 km²格网趋势文件
hex_shp = "Hex_data/NDVI_trend_hex_100.shp"

# 格网趋势统计字段
hex_tau_field = "tau"
hex_pvalue_field = "pvalue"
hex_slope_field = "slope"

# 原始显著性字段，仅用于一致性检查
# 如果不存在，可设置为None
hex_signif_field = "signif"

# 显著性水平
alpha = 0.05

# 像素趋势类别所在波段
pixel_class_band = 1

# 输出文件
output_overlap_tif = "MODIS_pixel_hex_overlap_8857.tif"
output_matrix_csv = "pixel_hex_overlap_matrix.csv"
output_metrics_csv = "pixel_hex_overlap_metrics.csv"

# 保存重新分类后的格网
save_classified_hex = True
output_classified_hex = (
    "Hex_data/NDVI_trend_hex_100_with_class.gpkg"
)

# 栅格化规则
# False：按照像元中心落入哪个格网进行赋值，推荐
# True：只要像元与格网接触就赋值
all_touched = False

# 输出NoData
output_nodata = -9999

# 输出栅格块大小
output_block_size = 512

# ==================================================


# 格网内部临时NoData
HEX_NODATA = -9999

# 趋势类别名称
PIXEL_CLASS_NAMES = {
    -1: "Pixel_DOWN",
    0: "Pixel_non_significant",
    1: "Pixel_UP"
}

HEX_CLASS_NAMES = {
    -1: "Hex_DOWN",
    0: "Hex_other",
    1: "Hex_UP"
}


def ensure_parent_directory(filepath):
    """
    如果输出文件包含文件夹路径，则自动创建父文件夹。
    """
    parent = os.path.dirname(filepath)

    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_ratio(numerator, denominator):
    """
    安全计算比例。
    """
    if denominator == 0:
        return np.nan

    return numerator / denominator


def prevalence_ratio(
    event_inside,
    total_inside,
    event_outside,
    total_outside
):
    """
    计算内部发生率与外部发生率之比。
    """
    inside_rate = safe_ratio(
        event_inside,
        total_inside
    )

    outside_rate = safe_ratio(
        event_outside,
        total_outside
    )

    if (
        not np.isfinite(inside_rate)
        or not np.isfinite(outside_rate)
        or outside_rate == 0
    ):
        return np.nan

    return inside_rate / outside_rate


def odds_ratio(a, b, c, d):
    """
    计算2×2列联表的优势比。

                  Event    Non-event
    Inside          a          b
    Outside         c          d

    如果任意单元格为0，则使用0.5连续性校正。
    """
    values = np.asarray(
        [a, b, c, d],
        dtype=np.float64
    )

    if np.any(values == 0):
        values += 0.5

    a, b, c, d = values

    return (a * d) / (b * c)


def derive_hex_trend_class(
    hexagons,
    tau_field,
    pvalue_field,
    slope_field,
    significance_level
):
    """
    根据论文定义，由tau、pvalue和Sen's slope
    重新构建格网趋势类别。

    返回编码：
         1 = Hex UP
         0 = Hex Other
        -1 = Hex DOWN
     -9999 = 无有效统计结果
    """

    required_fields = [
        tau_field,
        pvalue_field,
        slope_field
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in hexagons.columns
    ]

    if missing_fields:
        raise ValueError(
            f"格网文件缺少必要字段：{missing_fields}\n"
            f"现有字段：{list(hexagons.columns)}"
        )

    tau = pd.to_numeric(
        hexagons[tau_field],
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    pvalue = pd.to_numeric(
        hexagons[pvalue_field],
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    slope = pd.to_numeric(
        hexagons[slope_field],
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid = (
        np.isfinite(tau)
        & np.isfinite(pvalue)
        & np.isfinite(slope)
    )

    trend_code = np.full(
        len(hexagons),
        HEX_NODATA,
        dtype=np.int16
    )

    # 具有有效统计值但不满足UP/DOWN标准
    trend_code[valid] = 0

    up_mask = (
        valid
        & (tau > 0)
        & (pvalue < significance_level)
        & (slope > 0)
    )

    down_mask = (
        valid
        & (tau < 0)
        & (pvalue < significance_level)
        & (slope < 0)
    )

    trend_code[up_mask] = 1
    trend_code[down_mask] = -1

    result = hexagons.copy()

    result["_hex_code"] = trend_code

    result["_hex_class"] = "NoData"
    result.loc[
        result["_hex_code"] == 0,
        "_hex_class"
    ] = "OTHER"

    result.loc[
        result["_hex_code"] == 1,
        "_hex_class"
    ] = "UP"

    result.loc[
        result["_hex_code"] == -1,
        "_hex_class"
    ] = "DOWN"

    return result


def check_signif_consistency(
    hexagons,
    signif_field
):
    """
    检查重新构建的UP/DOWN分类是否与原始signif字段一致。

    signif只能表示显著或不显著，不能区分UP和DOWN。
    """
    audit = {
        "signif_comparable_count": 0,
        "signif_mismatch_count": 0,
        "signif_mismatch_rate": np.nan
    }

    if (
        signif_field is None
        or signif_field not in hexagons.columns
    ):
        print(
            "\n未执行signif一致性检查："
            "字段不存在或已设置为None。"
        )

        return audit

    original_signif = pd.to_numeric(
        hexagons[signif_field],
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    comparable = np.isfinite(original_signif)

    derived_signif = np.isin(
        hexagons["_hex_code"].to_numpy(),
        [-1, 1]
    ).astype(np.int8)

    comparable_count = int(
        np.count_nonzero(comparable)
    )

    if comparable_count == 0:
        return audit

    original_binary = original_signif[
        comparable
    ].astype(np.int8)

    mismatch_count = int(
        np.count_nonzero(
            original_binary
            != derived_signif[comparable]
        )
    )

    mismatch_rate = (
        mismatch_count / comparable_count
    )

    audit = {
        "signif_comparable_count": comparable_count,
        "signif_mismatch_count": mismatch_count,
        "signif_mismatch_rate": mismatch_rate
    }

    print("\n========== signif一致性检查 ==========")
    print(f"可比较格网：{comparable_count:,}")
    print(f"不一致格网：{mismatch_count:,}")
    print(f"不一致比例：{mismatch_rate * 100:.6f}%")
    print("======================================")

    return audit


def repair_geometries(geodataframe):
    """
    修复无效几何并删除空几何。
    """
    result = geodataframe.copy()

    result = result[
        result.geometry.notna()
        & (~result.geometry.is_empty)
    ].copy()

    invalid = ~result.geometry.is_valid

    invalid_count = int(
        np.count_nonzero(invalid)
    )

    if invalid_count > 0:
        print(
            f"\n检测到无效几何：{invalid_count:,}，"
            "正在使用buffer(0)修复……"
        )

        result.loc[
            invalid,
            "geometry"
        ] = result.loc[
            invalid,
            "geometry"
        ].buffer(0)

    result = result[
        result.geometry.notna()
        & (~result.geometry.is_empty)
        & result.geometry.is_valid
    ].copy()

    return result


def calculate_total_blocks(dataset, band_index):
    """
    计算栅格块总数，用于进度条。
    """
    block_height, block_width = (
        dataset.block_shapes[band_index - 1]
    )

    row_blocks = math.ceil(
        dataset.height / block_height
    )

    column_blocks = math.ceil(
        dataset.width / block_width
    )

    return row_blocks * column_blocks


def create_combination_code(
    pixel_array,
    hex_array,
    valid_pair
):
    """
    构建像素趋势—格网趋势组合编码。

    组合编码：
    11 = Pixel UP in Hex UP
    10 = Pixel UP in Hex Other
    12 = Pixel UP in Hex DOWN

    21 = Pixel DOWN in Hex UP
    20 = Pixel DOWN in Hex Other
    22 = Pixel DOWN in Hex DOWN

     1 = Pixel non-significant in Hex UP
     0 = Pixel non-significant in Hex Other
     2 = Pixel non-significant in Hex DOWN
    """

    height, width = pixel_array.shape

    pixel_encoded = np.zeros(
        (height, width),
        dtype=np.int16
    )

    pixel_encoded[pixel_array == 1] = 1
    pixel_encoded[pixel_array == -1] = 2

    hex_encoded = np.zeros(
        (height, width),
        dtype=np.int16
    )

    hex_encoded[hex_array == 1] = 1
    hex_encoded[hex_array == -1] = 2

    combination = (
        pixel_encoded * 10
        + hex_encoded
    ).astype(np.int16)

    combination[~valid_pair] = output_nodata

    return combination


def update_cross_matrix(
    counts,
    pixel_array,
    hex_array,
    valid_pair
):
    """
    更新3×3交叉统计矩阵。

    行顺序：
        Pixel DOWN
        Pixel non-significant
        Pixel UP

    列顺序：
        Hex DOWN
        Hex Other
        Hex UP
    """

    # -1映射为0，0映射为1，1映射为2
    pixel_index = (
        pixel_array[valid_pair] + 1
    ).astype(np.int16)

    hex_index = (
        hex_array[valid_pair] + 1
    ).astype(np.int16)

    encoded = (
        pixel_index * 3
        + hex_index
    )

    block_counts = np.bincount(
        encoded,
        minlength=9
    ).reshape(3, 3)

    counts += block_counts


def build_matrix_dataframe(
    counts,
    pixel_area_km2
):
    """
    将3×3交叉矩阵转换为长表。
    """

    rows = []

    pixel_codes = [-1, 0, 1]
    hex_codes = [-1, 0, 1]

    pixel_totals = counts.sum(axis=1)
    hex_totals = counts.sum(axis=0)
    overall_total = int(counts.sum())

    for row_index, pixel_code in enumerate(
        pixel_codes
    ):
        for column_index, hex_code in enumerate(
            hex_codes
        ):

            count = int(
                counts[row_index, column_index]
            )

            rows.append(
                {
                    "pixel_class":
                        PIXEL_CLASS_NAMES[pixel_code],

                    "hex_class":
                        HEX_CLASS_NAMES[hex_code],

                    "pixel_count":
                        count,

                    "area_km2":
                        count * pixel_area_km2,

                    "share_within_pixel_class":
                        safe_ratio(
                            count,
                            int(pixel_totals[row_index])
                        ),

                    "share_within_hex_class":
                        safe_ratio(
                            count,
                            int(hex_totals[column_index])
                        ),

                    "share_of_all_valid_pairs":
                        safe_ratio(
                            count,
                            overall_total
                        )
                }
            )

    return pd.DataFrame(rows)


def build_metrics_dataframe(
    counts,
    pixel_area_km2,
    unmatched_pixel_count,
    audit
):
    """
    根据交叉矩阵计算关键指标。

    counts行列顺序：
                Hex DOWN  Hex Other  Hex UP
    Pixel DOWN
    Pixel NS
    Pixel UP
    """

    # Pixel DOWN行
    down_in_down = int(counts[0, 0])
    down_in_other = int(counts[0, 1])
    down_in_up = int(counts[0, 2])

    # Pixel non-significant行
    ns_in_down = int(counts[1, 0])
    ns_in_other = int(counts[1, 1])
    ns_in_up = int(counts[1, 2])

    # Pixel UP行
    up_in_down = int(counts[2, 0])
    up_in_other = int(counts[2, 1])
    up_in_up = int(counts[2, 2])

    pixel_up_total = (
        up_in_down
        + up_in_other
        + up_in_up
    )

    pixel_down_total = (
        down_in_down
        + down_in_other
        + down_in_up
    )

    pixel_ns_total = (
        ns_in_down
        + ns_in_other
        + ns_in_up
    )

    valid_pixels_in_hex_up = (
        down_in_up
        + ns_in_up
        + up_in_up
    )

    valid_pixels_in_hex_down = (
        down_in_down
        + ns_in_down
        + up_in_down
    )

    valid_pixels_in_hex_other = (
        down_in_other
        + ns_in_other
        + up_in_other
    )

    valid_pixels_outside_hex_up = (
        valid_pixels_in_hex_down
        + valid_pixels_in_hex_other
    )

    valid_pixels_outside_hex_down = (
        valid_pixels_in_hex_up
        + valid_pixels_in_hex_other
    )

    pixel_up_outside_hex_up = (
        up_in_down
        + up_in_other
    )

    pixel_down_outside_hex_down = (
        down_in_up
        + down_in_other
    )

    non_up_inside_hex_up = (
        down_in_up
        + ns_in_up
    )

    non_up_outside_hex_up = (
        down_in_down
        + down_in_other
        + ns_in_down
        + ns_in_other
    )

    non_down_inside_hex_down = (
        ns_in_down
        + up_in_down
    )

    non_down_outside_hex_down = (
        ns_in_up
        + ns_in_other
        + up_in_up
        + up_in_other
    )

    total_valid_pairs = int(counts.sum())

    metrics = [
        {
            "metric": "all_valid_pixel_hex_pairs",
            "value": total_valid_pairs,
            "definition":
                "具有有效像素趋势和有效格网趋势类别的像元总数"
        },
        {
            "metric": "unmatched_valid_pixel_count",
            "value": unmatched_pixel_count,
            "definition":
                "具有有效像素趋势但未匹配到有效格网的像元数"
        },
        {
            "metric": "pixel_UP_total_count",
            "value": pixel_up_total,
            "definition":
                "全部Pixel UP像元数量"
        },
        {
            "metric": "pixel_DOWN_total_count",
            "value": pixel_down_total,
            "definition":
                "全部Pixel DOWN像元数量"
        },
        {
            "metric": "pixel_non_significant_total_count",
            "value": pixel_ns_total,
            "definition":
                "全部非显著像元数量"
        },
        {
            "metric": "pixel_UP_in_Hex_UP_count",
            "value": up_in_up,
            "definition":
                "位于Hex UP格网内的Pixel UP像元数"
        },
        {
            "metric": "pixel_DOWN_in_Hex_DOWN_count",
            "value": down_in_down,
            "definition":
                "位于Hex DOWN格网内的Pixel DOWN像元数"
        },
        {
            "metric": "pixel_UP_in_Hex_UP_area_km2",
            "value": up_in_up * pixel_area_km2,
            "definition":
                "位于Hex UP格网内的Pixel UP面积"
        },
        {
            "metric": "pixel_DOWN_in_Hex_DOWN_area_km2",
            "value": down_in_down * pixel_area_km2,
            "definition":
                "位于Hex DOWN格网内的Pixel DOWN面积"
        },
        {
            "metric": "UP_capture_rate",
            "value": safe_ratio(
                up_in_up,
                pixel_up_total
            ),
            "definition":
                "全部Pixel UP中位于Hex UP格网内部的比例"
        },
        {
            "metric": "DOWN_capture_rate",
            "value": safe_ratio(
                down_in_down,
                pixel_down_total
            ),
            "definition":
                "全部Pixel DOWN中位于Hex DOWN格网内部的比例"
        },
        {
            "metric": "UP_not_captured_rate",
            "value": safe_ratio(
                up_in_other + up_in_down,
                pixel_up_total
            ),
            "definition":
                "全部Pixel UP中未被Hex UP格网覆盖的比例"
        },
        {
            "metric": "DOWN_not_captured_rate",
            "value": safe_ratio(
                down_in_other + down_in_up,
                pixel_down_total
            ),
            "definition":
                "全部Pixel DOWN中未被Hex DOWN格网覆盖的比例"
        },
        {
            "metric": "Pixel_UP_share_within_Hex_UP",
            "value": safe_ratio(
                up_in_up,
                valid_pixels_in_hex_up
            ),
            "definition":
                "Hex UP内部有效像元中Pixel UP的比例"
        },
        {
            "metric": "Pixel_non_significant_share_within_Hex_UP",
            "value": safe_ratio(
                ns_in_up,
                valid_pixels_in_hex_up
            ),
            "definition":
                "Hex UP内部有效像元中非显著像元的比例"
        },
        {
            "metric": "Pixel_DOWN_share_within_Hex_UP",
            "value": safe_ratio(
                down_in_up,
                valid_pixels_in_hex_up
            ),
            "definition":
                "Hex UP内部有效像元中Pixel DOWN的比例"
        },
        {
            "metric": "Pixel_DOWN_share_within_Hex_DOWN",
            "value": safe_ratio(
                down_in_down,
                valid_pixels_in_hex_down
            ),
            "definition":
                "Hex DOWN内部有效像元中Pixel DOWN的比例"
        },
        {
            "metric": "Pixel_non_significant_share_within_Hex_DOWN",
            "value": safe_ratio(
                ns_in_down,
                valid_pixels_in_hex_down
            ),
            "definition":
                "Hex DOWN内部有效像元中非显著像元的比例"
        },
        {
            "metric": "Pixel_UP_share_within_Hex_DOWN",
            "value": safe_ratio(
                up_in_down,
                valid_pixels_in_hex_down
            ),
            "definition":
                "Hex DOWN内部有效像元中Pixel UP的比例"
        },
        {
            "metric": "Pixel_UP_prevalence_outside_Hex_UP",
            "value": safe_ratio(
                pixel_up_outside_hex_up,
                valid_pixels_outside_hex_up
            ),
            "definition":
                "非Hex UP格网内部有效像元中Pixel UP的比例"
        },
        {
            "metric": "Pixel_DOWN_prevalence_outside_Hex_DOWN",
            "value": safe_ratio(
                pixel_down_outside_hex_down,
                valid_pixels_outside_hex_down
            ),
            "definition":
                "非Hex DOWN格网内部有效像元中Pixel DOWN的比例"
        },
        {
            "metric": "UP_prevalence_ratio",
            "value": prevalence_ratio(
                up_in_up,
                valid_pixels_in_hex_up,
                pixel_up_outside_hex_up,
                valid_pixels_outside_hex_up
            ),
            "definition":
                "Pixel UP在Hex UP内部的发生率除以其在非Hex UP区域的发生率"
        },
        {
            "metric": "DOWN_prevalence_ratio",
            "value": prevalence_ratio(
                down_in_down,
                valid_pixels_in_hex_down,
                pixel_down_outside_hex_down,
                valid_pixels_outside_hex_down
            ),
            "definition":
                "Pixel DOWN在Hex DOWN内部的发生率除以其在非Hex DOWN区域的发生率"
        },
        {
            "metric": "UP_odds_ratio",
            "value": odds_ratio(
                up_in_up,
                non_up_inside_hex_up,
                pixel_up_outside_hex_up,
                non_up_outside_hex_up
            ),
            "definition":
                "Pixel UP出现在Hex UP内部相对于非Hex UP区域的优势比"
        },
        {
            "metric": "DOWN_odds_ratio",
            "value": odds_ratio(
                down_in_down,
                non_down_inside_hex_down,
                pixel_down_outside_hex_down,
                non_down_outside_hex_down
            ),
            "definition":
                "Pixel DOWN出现在Hex DOWN内部相对于非Hex DOWN区域的优势比"
        },
        {
            "metric": "UP_opposite_direction_rate",
            "value": safe_ratio(
                up_in_down,
                pixel_up_total
            ),
            "definition":
                "全部Pixel UP中位于Hex DOWN格网内部的比例"
        },
        {
            "metric": "DOWN_opposite_direction_rate",
            "value": safe_ratio(
                down_in_up,
                pixel_down_total
            ),
            "definition":
                "全部Pixel DOWN中位于Hex UP格网内部的比例"
        },
        {
            "metric": "signif_comparable_count",
            "value": audit[
                "signif_comparable_count"
            ],
            "definition":
                "重新推导分类与原signif字段可进行比较的格网数"
        },
        {
            "metric": "signif_mismatch_count",
            "value": audit[
                "signif_mismatch_count"
            ],
            "definition":
                "重新推导显著性与原signif字段不一致的格网数"
        },
        {
            "metric": "signif_mismatch_rate",
            "value": audit[
                "signif_mismatch_rate"
            ],
            "definition":
                "重新推导显著性与原signif字段不一致的比例"
        }
    ]

    return pd.DataFrame(metrics)


def main():

    ensure_parent_directory(output_overlap_tif)
    ensure_parent_directory(output_matrix_csv)
    ensure_parent_directory(output_metrics_csv)

    if save_classified_hex:
        ensure_parent_directory(
            output_classified_hex
        )

    # ---------------------------------------------------------
    # 1. 读取并分类格网
    # ---------------------------------------------------------

    print("读取格网数据……")

    if not os.path.exists(hex_shp):
        raise FileNotFoundError(
            f"未找到格网文件：{hex_shp}"
        )

    hexagons = gpd.read_file(hex_shp)

    if hexagons.crs is None:
        raise ValueError(
            f"格网文件缺少CRS：{hex_shp}"
        )

    hexagons = repair_geometries(
        hexagons
    )

    print(
        "\n按照tau、pvalue和Sen's slope"
        "重新构建格网UP/DOWN类别……"
    )

    hexagons = derive_hex_trend_class(
        hexagons=hexagons,
        tau_field=hex_tau_field,
        pvalue_field=hex_pvalue_field,
        slope_field=hex_slope_field,
        significance_level=alpha
    )

    print("\n========== 格网分类结果 ==========")
    print(
        hexagons["_hex_class"]
        .value_counts(dropna=False)
    )
    print("==================================")

    up_hex_count = int(
        np.count_nonzero(
            hexagons["_hex_code"].to_numpy() == 1
        )
    )

    down_hex_count = int(
        np.count_nonzero(
            hexagons["_hex_code"].to_numpy() == -1
        )
    )

    other_hex_count = int(
        np.count_nonzero(
            hexagons["_hex_code"].to_numpy() == 0
        )
    )

    nodata_hex_count = int(
        np.count_nonzero(
            hexagons["_hex_code"].to_numpy()
            == HEX_NODATA
        )
    )

    print(f"Hex UP：{up_hex_count:,}")
    print(f"Hex DOWN：{down_hex_count:,}")
    print(f"Hex Other：{other_hex_count:,}")
    print(f"Hex NoData：{nodata_hex_count:,}")

    if up_hex_count == 0:
        raise ValueError(
            "未识别到Hex UP格网，请检查tau、"
            "pvalue和slope字段。"
        )

    if down_hex_count == 0:
        raise ValueError(
            "未识别到Hex DOWN格网，请检查tau、"
            "pvalue和slope字段。"
        )

    audit = check_signif_consistency(
        hexagons,
        hex_signif_field
    )

    if save_classified_hex:
        print(
            f"\n保存重新分类后的格网："
            f"{output_classified_hex}"
        )

        hexagons.to_file(
            output_classified_hex,
            driver="GPKG"
        )

    # ---------------------------------------------------------
    # 2. 打开像素趋势栅格
    # ---------------------------------------------------------

    if not os.path.exists(pixel_trend_tif):
        raise FileNotFoundError(
            f"未找到像素趋势栅格："
            f"{pixel_trend_tif}"
        )

    print("\n打开像素趋势栅格……")

    with rasterio.open(
        pixel_trend_tif
    ) as src:

        if src.crs is None:
            raise ValueError(
                f"像素趋势栅格缺少CRS："
                f"{pixel_trend_tif}"
            )

        if src.count < pixel_class_band:
            raise ValueError(
                f"像素趋势栅格不存在Band "
                f"{pixel_class_band}"
            )

        if hexagons.crs != src.crs:
            print(
                "\n格网CRS与趋势栅格不同，"
                f"正在重投影至：{src.crs}"
            )

            hexagons = hexagons.to_crs(
                src.crs
            )

            hexagons = repair_geometries(
                hexagons
            )

        # 仅保留具有有效格网分类的多边形
        valid_hexagons = hexagons[
            hexagons["_hex_code"]
            != HEX_NODATA
        ].copy()

        if len(valid_hexagons) == 0:
            raise ValueError(
                "没有可用于栅格化的有效格网。"
            )

        spatial_index = (
            valid_hexagons.sindex
        )

        # EPSG:8857为等面积投影
        transform = src.transform

        pixel_area_m2 = abs(
            transform.a * transform.e
            - transform.b * transform.d
        )

        pixel_area_km2 = (
            pixel_area_m2 / 1_000_000.0
        )

        print(
            f"\n栅格CRS：{src.crs}"
        )
        print(
            f"栅格尺寸："
            f"{src.width:,} × {src.height:,}"
        )
        print(
            f"像元面积："
            f"{pixel_area_km2:.8f} km²"
        )

        # -----------------------------------------------------
        # 3. 创建输出栅格
        # -----------------------------------------------------

        output_profile = src.profile.copy()

        output_profile.update(
            driver="GTiff",
            count=3,
            dtype="int16",
            nodata=output_nodata,
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=output_block_size,
            blockysize=output_block_size,
            BIGTIFF="YES"
        )

        # 行列顺序均为：
        # DOWN、Other/NS、UP
        counts = np.zeros(
            (3, 3),
            dtype=np.int64
        )

        unmatched_valid_pixel_count = 0

        total_blocks = calculate_total_blocks(
            src,
            pixel_class_band
        )

        print("\n开始像素—格网空间叠加……")

        with rasterio.open(
            output_overlap_tif,
            "w",
            **output_profile
        ) as dst:

            dst.set_band_description(
                1,
                "pixel_trend_class"
            )

            dst.set_band_description(
                2,
                "hex_trend_class"
            )

            dst.set_band_description(
                3,
                "pixel_hex_combination"
            )

            dst.update_tags(
                pixel_class_definition=(
                    "-1=Pixel_DOWN;"
                    "0=Pixel_non_significant;"
                    "1=Pixel_UP"
                ),
                hex_class_definition=(
                    "-1=Hex_DOWN;"
                    "0=Hex_other;"
                    "1=Hex_UP"
                ),
                combination_definition=(
                    "11=Pixel_UP_in_Hex_UP;"
                    "10=Pixel_UP_in_Hex_other;"
                    "12=Pixel_UP_in_Hex_DOWN;"
                    "21=Pixel_DOWN_in_Hex_UP;"
                    "20=Pixel_DOWN_in_Hex_other;"
                    "22=Pixel_DOWN_in_Hex_DOWN;"
                    "1=Pixel_non_significant_in_Hex_UP;"
                    "0=Pixel_non_significant_in_Hex_other;"
                    "2=Pixel_non_significant_in_Hex_DOWN"
                ),
                hex_UP_definition=(
                    "tau>0, pvalue<0.05, slope>0"
                ),
                hex_DOWN_definition=(
                    "tau<0, pvalue<0.05, slope<0"
                ),
                rasterization_rule=(
                    "pixel_centre"
                    if not all_touched
                    else "all_touched"
                )
            )

            iterator = tqdm(
                src.block_windows(
                    pixel_class_band
                ),
                total=total_blocks,
                desc="Pixel-Hex overlay",
                unit="block"
            )

            for _, window in iterator:

                height = int(window.height)
                width = int(window.width)

                pixel_raw = src.read(
                    pixel_class_band,
                    window=window
                )

                # 仅接受-1、0和1
                valid_pixel = np.isin(
                    pixel_raw,
                    [-1, 0, 1]
                )

                # 初始化当前输出块
                output_block = np.full(
                    (3, height, width),
                    output_nodata,
                    dtype=np.int16
                )

                if not np.any(valid_pixel):
                    dst.write(
                        output_block,
                        window=window
                    )
                    continue

                current_bounds = (
                    src.window_bounds(window)
                )

                candidate_indices = list(
                    spatial_index.intersection(
                        current_bounds
                    )
                )

                if len(candidate_indices) == 0:
                    unmatched_valid_pixel_count += int(
                        np.count_nonzero(
                            valid_pixel
                        )
                    )

                    dst.write(
                        output_block,
                        window=window
                    )
                    continue

                candidates = valid_hexagons.iloc[
                    candidate_indices
                ]

                current_window_box = box(
                    *current_bounds
                )

                candidates = candidates[
                    candidates.geometry.intersects(
                        current_window_box
                    )
                ]

                if len(candidates) == 0:
                    unmatched_valid_pixel_count += int(
                        np.count_nonzero(
                            valid_pixel
                        )
                    )

                    dst.write(
                        output_block,
                        window=window
                    )
                    continue

                shapes = [
                    (
                        geometry,
                        int(code)
                    )
                    for geometry, code
                    in zip(
                        candidates.geometry,
                        candidates["_hex_code"]
                    )
                    if (
                        geometry is not None
                        and not geometry.is_empty
                        and code != HEX_NODATA
                    )
                ]

                if len(shapes) == 0:
                    unmatched_valid_pixel_count += int(
                        np.count_nonzero(
                            valid_pixel
                        )
                    )

                    dst.write(
                        output_block,
                        window=window
                    )
                    continue

                hex_array = rasterize(
                    shapes=shapes,
                    out_shape=(
                        height,
                        width
                    ),
                    transform=(
                        src.window_transform(
                            window
                        )
                    ),
                    fill=HEX_NODATA,
                    all_touched=all_touched,
                    dtype="int16"
                )

                valid_hex = np.isin(
                    hex_array,
                    [-1, 0, 1]
                )

                valid_pair = (
                    valid_pixel
                    & valid_hex
                )

                unmatched = (
                    valid_pixel
                    & (~valid_hex)
                )

                unmatched_valid_pixel_count += int(
                    np.count_nonzero(
                        unmatched
                    )
                )

                if np.any(valid_pair):

                    pixel_class = (
                        pixel_raw.astype(
                            np.int16,
                            copy=False
                        )
                    )

                    update_cross_matrix(
                        counts=counts,
                        pixel_array=pixel_class,
                        hex_array=hex_array,
                        valid_pair=valid_pair
                    )

                    combination = (
                        create_combination_code(
                            pixel_array=pixel_class,
                            hex_array=hex_array,
                            valid_pair=valid_pair
                        )
                    )

                    output_block[
                        0,
                        valid_pair
                    ] = pixel_class[
                        valid_pair
                    ]

                    output_block[
                        1,
                        valid_pair
                    ] = hex_array[
                        valid_pair
                    ]

                    output_block[
                        2,
                        valid_pair
                    ] = combination[
                        valid_pair
                    ]

                # 每个块均写出，避免未写块默认变成0
                dst.write(
                    output_block,
                    window=window
                )

    # ---------------------------------------------------------
    # 4. 输出交叉统计矩阵
    # ---------------------------------------------------------

    matrix_dataframe = (
        build_matrix_dataframe(
            counts=counts,
            pixel_area_km2=pixel_area_km2
        )
    )

    matrix_dataframe.to_csv(
        output_matrix_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # 5. 输出关键统计指标
    # ---------------------------------------------------------

    metrics_dataframe = (
        build_metrics_dataframe(
            counts=counts,
            pixel_area_km2=pixel_area_km2,
            unmatched_pixel_count=(
                unmatched_valid_pixel_count
            ),
            audit=audit
        )
    )

    metrics_dataframe.to_csv(
        output_metrics_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # 6. 控制台输出
    # ---------------------------------------------------------

    display_matrix = pd.DataFrame(
        counts,
        index=[
            "Pixel_DOWN",
            "Pixel_non_significant",
            "Pixel_UP"
        ],
        columns=[
            "Hex_DOWN",
            "Hex_other",
            "Hex_UP"
        ]
    )

    print("\n============================================")
    print("像素趋势—格网趋势交叉矩阵")
    print("============================================")
    print(display_matrix)

    print("\n============================================")
    print("关键指标")
    print("============================================")

    important_metrics = [
        "pixel_UP_total_count",
        "pixel_DOWN_total_count",
        "pixel_UP_in_Hex_UP_count",
        "pixel_DOWN_in_Hex_DOWN_count",
        "pixel_UP_in_Hex_UP_area_km2",
        "pixel_DOWN_in_Hex_DOWN_area_km2",
        "UP_capture_rate",
        "DOWN_capture_rate",
        "UP_not_captured_rate",
        "DOWN_not_captured_rate",
        "Pixel_UP_share_within_Hex_UP",
        "Pixel_DOWN_share_within_Hex_DOWN",
        "UP_prevalence_ratio",
        "DOWN_prevalence_ratio",
        "UP_odds_ratio",
        "DOWN_odds_ratio",
        "UP_opposite_direction_rate",
        "DOWN_opposite_direction_rate"
    ]

    print(
        metrics_dataframe[
            metrics_dataframe[
                "metric"
            ].isin(important_metrics)
        ][
            ["metric", "value"]
        ].to_string(index=False)
    )

    print(
        "\n未匹配到有效格网的像元数："
        f"{unmatched_valid_pixel_count:,}"
    )

    print("\n============================================")
    print("输出文件")
    print("============================================")
    print(
        f"组合分类栅格：{output_overlap_tif}"
    )
    print(
        f"交叉统计矩阵：{output_matrix_csv}"
    )
    print(
        f"关键统计指标：{output_metrics_csv}"
    )

    if save_classified_hex:
        print(
            f"重新分类格网：{output_classified_hex}"
        )

    print("============================================")


if __name__ == "__main__":
    main()