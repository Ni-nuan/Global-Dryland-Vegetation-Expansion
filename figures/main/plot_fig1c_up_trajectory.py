#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1c: temporal trajectory of vegetated fraction across UP hexagons.

- Annual median vegetated fraction is shown as the observed trajectory.
- A fitted linear trend is overlaid as the main visual guide.
- IQR shading shows cross-hexagon spread.
- Clean journal-style formatting for main-text figure assembly.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User-defined paths
# =========================
INPUT_CSV = Path(r"hex_data/NDVI_trend_hex_100_UP_valid_sens_gt0.csv")
OUTPUT_PNG = Path(r"Fig1c_UP_median_trendline_clean.png")
OUTPUT_PDF = Path(r"Fig1c_UP_median_trendline_clean.pdf")
OUTPUT_SVG = Path(r"Fig1c_UP_median_trendline_clean.svg")
OUTPUT_CSV = Path(r"Fig1c_UP_median_trendline_summary.csv")

EXPORT_TRANSPARENT = True

# =========================
# Style
# =========================
FIGSIZE = (3.35, 2.25)
DPI = 600
FONT_FAMILY = "Arial"

LINE_COLOR = "#2166AC"
TREND_COLOR = "#08306B"
BAND_COLOR = "#9ECAE1"
AXIS_COLOR = "#333333"
GRID_COLOR = "#E8E8E8"
TEXT_COLOR = "#222222"

# Add a small right-side x-axis pad so the final-year tick label is not clipped.
X_PAD_LEFT = 0.0
X_PAD_RIGHT = 0.85


def nice_upper_limit(x, step=0.05, min_upper=0.25, max_upper=1.0):
    """Round y-axis upper limit to a clean value."""
    upper = np.nanmax(x)
    upper = upper + 0.03
    upper = np.ceil(upper / step) * step
    return float(np.clip(upper, min_upper, max_upper))


def main():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 7.5,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
    })

    # ---------------------------------------------------------------------
    # Read data
    # ---------------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)

    year_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"veg_\d{4}", c)],
        key=lambda x: int(x.split("_")[1])
    )
    years = np.array([int(c.split("_")[1]) for c in year_cols], dtype=int)

    median = df[year_cols].median(axis=0, skipna=True).to_numpy(dtype=float)
    q25 = df[year_cols].quantile(0.25, axis=0, numeric_only=True).to_numpy(dtype=float)
    q75 = df[year_cols].quantile(0.75, axis=0, numeric_only=True).to_numpy(dtype=float)

    # Linear fit on annual medians.
    # Use relative year for numerical stability and interpretable slope.
    x_rel = years.astype(float) - years.min()
    slope, intercept = np.polyfit(x_rel, median.astype(float), deg=1)
    trend = intercept + slope * x_rel

    out = pd.DataFrame({
        "year": years,
        "veg_frac_median": median,
        "veg_frac_q25": q25,
        "veg_frac_q75": q75,
        "trendline_linear": trend,
        "slope_per_year": np.repeat(slope, len(years)),
        "slope_per_decade": np.repeat(slope * 10, len(years)),
        "intercept_at_start_year": np.repeat(intercept, len(years)),
    })
    out.to_csv(OUTPUT_CSV, index=False)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")

    # IQR band
    ax.fill_between(
        years,
        q25,
        q75,
        color=BAND_COLOR,
        alpha=0.16,
        linewidth=0,
        label="IQR"
    )

    # Annual median
    ax.plot(
        years,
        median,
        color=LINE_COLOR,
        linewidth=1.15,
        alpha=0.80,
        label="Annual median"
    )

    # Small annual markers
    ax.scatter(
        years,
        median,
        s=8,
        color=LINE_COLOR,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.85,
        zorder=4
    )

    # Overall linear trend
    ax.plot(
        years,
        trend,
        color=TREND_COLOR,
        linewidth=1.65,
        linestyle=(0, (4, 2)),
        label="Linear trend",
        zorder=5
    )

    # Minimal annotation
    ax.text(
        0.04,
        0.92,
        f"Slope = {slope * 10:+.3f} per decade",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=TEXT_COLOR
    )

    # Axes
    ax.set_xlabel("Year", fontsize=7.8, labelpad=3)
    ax.set_ylabel("Vegetated fraction", fontsize=7.8, labelpad=3)

    ax.set_xlim(years.min() - X_PAD_LEFT, years.max() + X_PAD_RIGHT)
    ax.set_ylim(0, nice_upper_limit(q75))

    tick_years = np.arange(years.min(), years.max() + 1, 4)
    if years[-1] not in tick_years:
        tick_years = np.append(tick_years, years[-1])
    ax.set_xticks(tick_years)
    ax.set_xticklabels(tick_years.astype(int))

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)

    ax.tick_params(axis="both", colors=AXIS_COLOR, labelsize=7.1, pad=2)

    # Light y-grid only
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # Compact legend
    ax.legend(
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.08),
        handlelength=1.6,
        columnspacing=0.9,
        borderaxespad=0.0
    )

    # Slightly reduce the right edge of the axes to prevent the final-year tick label
    # from being clipped while keeping the panel size close to the original.
    fig.subplots_adjust(left=0.15, right=0.955, bottom=0.19, top=0.88)

    if EXPORT_TRANSPARENT:
        fig.patch.set_alpha(0)
        for a in fig.axes:
            a.set_facecolor("none")

    save_kwargs = dict(transparent=EXPORT_TRANSPARENT)
    fig.savefig(OUTPUT_PNG, dpi=DPI, **save_kwargs)
    fig.savefig(OUTPUT_PDF, **save_kwargs)
    fig.savefig(OUTPUT_SVG, **save_kwargs)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()