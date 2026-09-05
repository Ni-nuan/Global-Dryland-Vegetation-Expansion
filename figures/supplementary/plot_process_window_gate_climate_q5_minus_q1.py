#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-context end-member contrasts along window/gate axes.

Refined export version:
- uses the final method-defined Q5-Q1 end-member contrast;
- replaces Q1-Q5 display text with highest-minus-lowest trend-group language;
- harmonizes typography, colours and export formats with revised Fig. 1-3 style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =============================================================================
# Paths
# =============================================================================
P90_CSV = "outputs/stratification/process_context/process_x_P90_trend_quantile_long.csv"
VPD_CSV = "outputs/stratification/process_context/process_x_VPD_resid_trend_quantile_long.csv"
OUTPUT_CSV = "FIG3_process_delta_window_gate_clim_highest_minus_lowest_data.csv"
OUTPUT_PNG = "FIG3_process_delta_window_gate_clim_highest_minus_lowest.png"
OUTPUT_PDF = "FIG3_process_delta_window_gate_clim_highest_minus_lowest.pdf"
OUTPUT_SVG = "FIG3_process_delta_window_gate_clim_highest_minus_lowest.svg"
EXPORT_TRANSPARENT = True
DPI = 600

# =============================================================================
# Configuration
# =============================================================================
PROCESS_ORDER = [
    "Bare_or_sparse_to_grass_or_forest",
    "Bare_to_Sparse",
    "Ag_expansion",
    "Urban_expansion",
    "No_transition",
    "Other",
]
PROCESS_LABELS = {
    "Bare_or_sparse_to_grass_or_forest": "Bare/sparse to grass/forest",
    "Bare_to_Sparse": "Bare to sparse",
    "Ag_expansion": "Agricultural expansion",
    "Urban_expansion": "Urban expansion",
    "No_transition": "No transition",
    "Other": "Other",
}
PROCESS_COLORS = {
    "Bare_or_sparse_to_grass_or_forest": "#4F9D68",
    "Bare_to_Sparse": "#77B98A",
    "Other": "#B5B5B5",
    "No_transition": "#D9C7A3",
    "Ag_expansion": "#9ECAE1",
    "Urban_expansion": "#8D8D8D",
}
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# =============================================================================
# Style
# =============================================================================
FIGSIZE = (5.35, 2.95)
# Keep the canvas size unchanged, but lift the plotting axes slightly to leave
# one shared x-axis label below both panels. This avoids overlap between two
# repeated long axis labels while preserving panel alignment.
LEFT_AX_POS = [0.36, 0.285, 0.265, 0.60]
RIGHT_AX_POS = [0.68, 0.285, 0.265, 0.60]
BASE_FONT = 6.8
LABEL_FONT = 6.2
TICK_FONT = 6.1
AXIS_LABEL_FONT = 6.8
TITLE_FONT = 7.0
VALUE_FONT = 5.9
GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
ZERO_COLOR = "#555555"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,
        "axes.linewidth": 0.62,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def build_delta(df: pd.DataFrame, qcol: str, driver_name: str, value_col: str) -> pd.DataFrame:
    rows = []
    for g in PROCESS_ORDER:
        sub = df[df["group_type"] == g].copy()
        if sub.empty:
            continue
        sub[qcol] = pd.Categorical(sub[qcol], categories=Q_LABELS, ordered=True)
        sub = sub.sort_values(qcol)
        if (sub[qcol] == "Q1").sum() == 0 or (sub[qcol] == "Q5").sum() == 0:
            continue
        q1 = float(sub.loc[sub[qcol] == "Q1", value_col].iloc[0])
        q5 = float(sub.loc[sub[qcol] == "Q5", value_col].iloc[0])
        rows.append({
            "driver": driver_name,
            "group_type": g,
            "lowest_group_median": q1,
            "highest_group_median": q5,
            "delta_highest_minus_lowest": q5 - q1,
        })
    return pd.DataFrame(rows)


def percent_formatter(x, pos):
    return f"{x:.0f}"


def main() -> None:
    apply_style()
    p90 = pd.read_csv(P90_CSV)
    vpd = pd.read_csv(VPD_CSV)

    plot_df = pd.concat([
        build_delta(p90, "P90_trend_qbin", "Water-supply window", "share_abs_clim_median"),
        build_delta(vpd, "VPD_resid_trend_qbin", "Drying gate", "share_abs_clim_median"),
    ], ignore_index=True)
    plot_df["delta_highest_minus_lowest_pct"] = plot_df["delta_highest_minus_lowest"] * 100.0
    plot_df.to_csv(OUTPUT_CSV, index=False)

    all_vals = plot_df["delta_highest_minus_lowest_pct"].to_numpy(dtype=float)
    max_abs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)), 1.0)
    lim = max_abs * 1.25
    xmin, xmax = -lim, lim
    xticks = np.linspace(np.floor(xmin / 5) * 5, np.ceil(xmax / 5) * 5, 5)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    axes = [fig.add_axes(LEFT_AX_POS), fig.add_axes(RIGHT_AX_POS)]
    panels = [
        ("Water-supply window", "Highest - lowest climate share (percentage points)"),
        ("Drying gate", "Highest - lowest climate share (percentage points)"),
    ]

    for ax, (driver, xlabel) in zip(axes, panels):
        tab = plot_df[plot_df["driver"] == driver].copy()
        tab["group_type"] = pd.Categorical(tab["group_type"], categories=PROCESS_ORDER, ordered=True)
        tab = tab.sort_values("group_type").reset_index(drop=True)
        y = np.arange(len(tab))
        vals = tab["delta_highest_minus_lowest_pct"].to_numpy(dtype=float)
        colors = [PROCESS_COLORS.get(g, "#B5B5B5") for g in tab["group_type"].astype(str)]

        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42, zorder=0)
        ax.axvline(0, color=ZERO_COLOR, lw=0.68, zorder=1)
        ax.barh(y, vals, height=0.50, color=colors, edgecolor="none", zorder=3)

        offset = (xmax - xmin) * 0.018
        for yi, val in zip(y, vals):
            ha = "left" if val >= 0 else "right"
            x_text = val + offset if val >= 0 else val - offset
            ax.text(x_text, yi, f"{val:+.1f}", va="center", ha=ha, fontsize=VALUE_FONT, color=TEXT_COLOR)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(len(tab) - 0.35, -0.65)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        # Use one shared x-axis label for both panels to prevent overlap.
        ax.set_xlabel("")
        ax.set_title(driver, fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
        ax.tick_params(axis="x", labelsize=TICK_FONT, length=2.4, width=0.62, colors=AXIS_COLOR, pad=2)
        ax.tick_params(axis="y", length=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(AXIS_COLOR)
        ax.spines["bottom"].set_linewidth(0.62)

    axes[0].set_yticks(np.arange(len(PROCESS_ORDER)))
    axes[0].set_yticklabels([PROCESS_LABELS[g] for g in PROCESS_ORDER], fontsize=LABEL_FONT, color=TEXT_COLOR)
    axes[0].tick_params(axis="y", pad=5)
    axes[0].set_ylabel("Process context", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=4)
    axes[1].set_yticks(np.arange(len(PROCESS_ORDER)))
    axes[1].set_yticklabels([])

    # Shared x-axis label centred over the two plotting panels.
    shared_x = (LEFT_AX_POS[0] + LEFT_AX_POS[2] / 2 + RIGHT_AX_POS[0] + RIGHT_AX_POS[2] / 2) / 2
    fig.text(
        shared_x,
        0.095,
        "Highest - lowest climate share (percentage points)",
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONT,
        color=TEXT_COLOR,
    )

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
