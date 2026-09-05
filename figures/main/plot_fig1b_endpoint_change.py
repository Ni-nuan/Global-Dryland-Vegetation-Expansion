#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fig. 1b
首尾三年平滑端点的植被覆盖比例变化分布

推荐主设置：
    ENDPOINT_METHOD = "median"

定义：
    early_endpoint = median(veg_2000, veg_2001, veg_2002)
    late_endpoint  = median(veg_2020, veg_2021, veg_2022)
    delta_veg_frac = late_endpoint - early_endpoint

可选敏感性设置：
    ENDPOINT_METHOD = "mean"
    ENDPOINT_METHOD = "max"
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 用户配置
# ============================================================

# 建议读取包含 tau、pvalue、slope 和各年度 veg_YYYY 字段的完整CSV
INPUT_CSV = Path(r"hex_data/NDVI_trend_hex_100.csv")

# 输出
OUTPUT_PNG = Path(
    r"Fig1b_delta_UP_3yr_median_violin_box.png"
)
OUTPUT_PDF = Path(
    r"Fig1b_delta_UP_3yr_median_violin_box.pdf"
)
OUTPUT_SVG = Path(
    r"Fig1b_delta_UP_3yr_median_violin_box.svg"
)
OUTPUT_DATA_CSV = Path(
    r"NDVI_trend_hex_100_delta_UP_3yr_median.csv"
)
OUTPUT_SUMMARY_CSV = Path(
    r"NDVI_trend_hex_100_delta_UP_3yr_median_summary.csv"
)

# 是否需要依据 tau、pvalue、slope 从完整数据中重新筛选UP
# 如果输入CSV本身已经只包含UP，可以改为False
FILTER_TO_UP = True

TAU_FIELD = "trend_tau"
PVALUE_FIELD = "p_value"
SLOPE_FIELD = "sen_slope"

ALPHA = 0.05

# 首尾时间窗口
EARLY_YEARS = [2000, 2001, 2002]
LATE_YEARS = [2020, 2021, 2022]

# 推荐使用 median
# 可选值："median"、"mean"、"max"
ENDPOINT_METHOD = "median"

# 每个三年窗口至少需要几个有效年份
# 推荐2：避免单一年度决定端点，同时允许个别年份缺失
MIN_VALID_YEARS_PER_WINDOW = 2

EXPORT_TRANSPARENT = True

# ============================================================


# ============================================================
# 绘图样式
# ============================================================

FIGSIZE = (3.35, 2.25)
DPI = 600
FONT_FAMILY = "Arial"

UP_BLUE = "#2166AC"
UP_LIGHT = "#9ECAE1"
TEXT_COLOR = "#222222"
AXIS_COLOR = "#333333"
ZERO_COLOR = "#8A8A8A"


def ensure_parent(path):
    """创建输出文件父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def aggregate_endpoint(values, method):
    """
    对二维数组按行计算端点统计量。

    values:
        shape = [n_hexagons, n_years]
    """

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning
        )

        if method == "median":
            return np.nanmedian(values, axis=1)

        if method == "mean":
            return np.nanmean(values, axis=1)

        if method == "max":
            return np.nanmax(values, axis=1)

    raise ValueError(
        "ENDPOINT_METHOD必须为："
        "'median'、'mean'或'max'"
    )


def main():

    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 7.5,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "xtick.major.size": 2.8,
        "ytick.major.size": 0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0,
    })

    # --------------------------------------------------------
    # 1. 检查输入
    # --------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_CSV}"
        )

    early_columns = [
        f"veg_{year}"
        for year in EARLY_YEARS
    ]

    late_columns = [
        f"veg_{year}"
        for year in LATE_YEARS
    ]

    required_columns = (
        early_columns
        + late_columns
    )

    if FILTER_TO_UP:
        required_columns += [
            TAU_FIELD,
            PVALUE_FIELD,
            SLOPE_FIELD
        ]

    print(f"读取数据：{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "输入CSV缺少必要字段：\n"
            f"{missing_columns}\n\n"
            f"现有字段包括：{list(df.columns)}"
        )

    print(f"原始记录数：{len(df):,}")

    # --------------------------------------------------------
    # 2. 按论文定义筛选UP
    # --------------------------------------------------------

    if FILTER_TO_UP:

        tau = pd.to_numeric(
            df[TAU_FIELD],
            errors="coerce"
        )

        pvalue = pd.to_numeric(
            df[PVALUE_FIELD],
            errors="coerce"
        )

        slope = pd.to_numeric(
            df[SLOPE_FIELD],
            errors="coerce"
        )

        up_mask = (
            tau.notna()
            & pvalue.notna()
            & slope.notna()
            & (tau > 0)
            & (pvalue < ALPHA)
            & (slope > 0)
        )

        df = df.loc[up_mask].copy()

        print(
            f"按UP标准筛选后的记录数："
            f"{len(df):,}"
        )

    # --------------------------------------------------------
    # 3. 读取首尾三年数据
    # --------------------------------------------------------

    for column in early_columns + late_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    early_values = df[
        early_columns
    ].to_numpy(dtype=np.float64)

    late_values = df[
        late_columns
    ].to_numpy(dtype=np.float64)

    early_valid_count = np.sum(
        np.isfinite(early_values),
        axis=1
    )

    late_valid_count = np.sum(
        np.isfinite(late_values),
        axis=1
    )

    valid_endpoint_mask = (
        (early_valid_count >= MIN_VALID_YEARS_PER_WINDOW)
        & (late_valid_count >= MIN_VALID_YEARS_PER_WINDOW)
    )

    print(
        "满足首尾窗口有效年份要求的记录数："
        f"{int(np.count_nonzero(valid_endpoint_mask)):,}"
    )

    df = df.loc[
        valid_endpoint_mask
    ].copy()

    early_values = early_values[
        valid_endpoint_mask
    ]

    late_values = late_values[
        valid_endpoint_mask
    ]

    early_valid_count = early_valid_count[
        valid_endpoint_mask
    ]

    late_valid_count = late_valid_count[
        valid_endpoint_mask
    ]

    # --------------------------------------------------------
    # 4. 计算首尾复合端点
    # --------------------------------------------------------

    early_endpoint = aggregate_endpoint(
        early_values,
        ENDPOINT_METHOD
    )

    late_endpoint = aggregate_endpoint(
        late_values,
        ENDPOINT_METHOD
    )

    delta = (
        late_endpoint
        - early_endpoint
    )

    finite_delta = np.isfinite(delta)

    df = df.loc[
        finite_delta
    ].copy()

    early_endpoint = early_endpoint[
        finite_delta
    ]

    late_endpoint = late_endpoint[
        finite_delta
    ]

    delta = delta[
        finite_delta
    ]

    early_valid_count = early_valid_count[
        finite_delta
    ]

    late_valid_count = late_valid_count[
        finite_delta
    ]

    n = len(delta)

    if n == 0:
        raise ValueError(
            "没有可用于绘图的有效端点变化记录。"
        )

    # 保存逐格网结果
    df["early_endpoint"] = early_endpoint
    df["late_endpoint"] = late_endpoint
    df["delta_veg_frac"] = delta
    df["early_valid_years"] = early_valid_count
    df["late_valid_years"] = late_valid_count
    df["endpoint_method"] = ENDPOINT_METHOD

    ensure_parent(OUTPUT_DATA_CSV)

    df.to_csv(
        OUTPUT_DATA_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 5. 统计量
    # --------------------------------------------------------

    median = float(
        np.median(delta)
    )

    q1 = float(
        np.percentile(delta, 25)
    )

    q3 = float(
        np.percentile(delta, 75)
    )

    mean = float(
        np.mean(delta)
    )

    positive_count = int(
        np.count_nonzero(delta > 0)
    )

    zero_count = int(
        np.count_nonzero(delta == 0)
    )

    negative_count = int(
        np.count_nonzero(delta < 0)
    )

    summary = pd.DataFrame([
        {
            "endpoint_method": ENDPOINT_METHOD,
            "early_years": "-".join(
                map(str, EARLY_YEARS)
            ),
            "late_years": "-".join(
                map(str, LATE_YEARS)
            ),
            "minimum_valid_years_per_window":
                MIN_VALID_YEARS_PER_WINDOW,
            "N": n,
            "median_delta": median,
            "q1_delta": q1,
            "q3_delta": q3,
            "mean_delta": mean,
            "positive_count": positive_count,
            "positive_fraction":
                positive_count / n,
            "zero_count": zero_count,
            "negative_count": negative_count,
            "negative_fraction":
                negative_count / n
        }
    ])

    ensure_parent(OUTPUT_SUMMARY_CSV)

    summary.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n============== 统计结果 ==============")
    print(f"端点方法：{ENDPOINT_METHOD}")
    print(f"N：{n:,}")
    print(f"Median：{median:.6f}")
    print(f"IQR：{q1:.6f}–{q3:.6f}")
    print(f"Mean：{mean:.6f}")
    print(
        "正变化比例："
        f"{positive_count / n * 100:.3f}%"
    )
    print(
        "负变化比例："
        f"{negative_count / n * 100:.3f}%"
    )
    print("======================================")

    # --------------------------------------------------------
    # 6. 确定显示范围
    # --------------------------------------------------------

    p01 = float(
        np.percentile(delta, 1)
    )

    p99 = float(
        np.percentile(delta, 99)
    )

    x_min = min(
        -0.03,
        p01 - 0.025
    )

    x_max = max(
        1.02,
        p99 + 0.06
    )

    # --------------------------------------------------------
    # 7. 绘图
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        facecolor="white"
    )

    # violin distribution
    parts = ax.violinplot(
        [delta],
        positions=[1],
        vert=False,
        widths=0.66,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor(UP_LIGHT)
        body.set_edgecolor("none")
        body.set_alpha(0.38)

    # boxplot
    ax.boxplot(
        [delta],
        positions=[1],
        vert=False,
        widths=0.22,
        showfliers=False,
        patch_artist=True,
        medianprops={
            "color": UP_BLUE,
            "linewidth": 1.15,
        },
        boxprops={
            "facecolor": "white",
            "edgecolor": UP_BLUE,
            "linewidth": 0.85,
        },
        whiskerprops={
            "color": UP_BLUE,
            "linewidth": 0.85,
        },
        capprops={
            "color": UP_BLUE,
            "linewidth": 0.85,
        },
    )

    # median point
    ax.scatter(
        median,
        1,
        s=14,
        color=UP_BLUE,
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )

    # zero reference
    ax.axvline(
        0,
        color=ZERO_COLOR,
        linewidth=0.65,
        linestyle=(0, (4, 3)),
        zorder=0,
    )

    # statistics annotation
    txt = (
        f"Median = {median:.3f}\n"
        f"IQR = {q1:.3f}–{q3:.3f}\n"
        f"N = {n:,}"
    )

    ax.text(
        0.70,
        0.90,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=TEXT_COLOR,
        linespacing=1.15,
        zorder=10,
    )

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_ylim(
        0.50,
        1.50
    )

    ax.set_yticks([])

    method_label = {
        "median": "median",
        "mean": "mean",
        "max": "maximum"
    }[ENDPOINT_METHOD]

    ax.set_xlabel(
        (
            f"Change in 3-year {method_label} "
            "vegetated fraction\n"
            "(2020–2022 minus 2000–2002)"
        ),
        fontsize=7.8,
        labelpad=4
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(
        AXIS_COLOR
    )
    ax.spines["left"].set_linewidth(
        0.65
    )

    ax.spines["bottom"].set_color(
        AXIS_COLOR
    )
    ax.spines["bottom"].set_linewidth(
        0.65
    )

    ax.tick_params(
        axis="x",
        colors=AXIS_COLOR,
        labelsize=7.2,
        pad=2
    )

    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=0.27,
        top=0.98
    )

    if EXPORT_TRANSPARENT:
        fig.patch.set_alpha(0)

        for current_axis in fig.axes:
            current_axis.set_facecolor(
                "none"
            )

    save_kwargs = {
        "transparent": EXPORT_TRANSPARENT
    }

    ensure_parent(OUTPUT_PNG)
    ensure_parent(OUTPUT_PDF)
    ensure_parent(OUTPUT_SVG)

    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        **save_kwargs
    )

    fig.savefig(
        OUTPUT_PDF,
        **save_kwargs
    )

    fig.savefig(
        OUTPUT_SVG,
        **save_kwargs
    )

    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_DATA_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()