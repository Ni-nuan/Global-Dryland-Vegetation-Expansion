#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-member contrast in climate share along the SM90 moisture-state gradient.

The final method-defined contrast is Q5-Q1: Q1 is the lowest 20% trend
group and Q5 is the highest 20% trend group. The figure labels use readable
highest-minus-lowest wording.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "FIG5/fig5_UP_hex_plot_ready_core.csv"
OUTPUT_CSV = "FIG3_delta_state_clim_highest_minus_lowest_data.csv"
OUTPUT_PNG = "FIG3_delta_state_clim_highest_minus_lowest.png"
OUTPUT_PDF = "FIG3_delta_state_clim_highest_minus_lowest.pdf"
OUTPUT_SVG = "FIG3_delta_state_clim_highest_minus_lowest.svg"
EXPORT_TRANSPARENT = True
DPI = 600

# =============================================================================
# Configuration
# =============================================================================
ARIDITY_ORDER = ["Hyperarid", "Arid", "Semiarid", "Dry_subhumid"]
ARIDITY_LABELS = {
    "Hyperarid": "Hyperarid",
    "Arid": "Arid",
    "Semiarid": "Semiarid",
    "Dry_subhumid": "Dry subhumid",
}
ARIDITY_COLORS = {
    "Hyperarid": "#9A7A2F",
    "Arid": "#D7A257",
    "Semiarid": "#9ECAE1",
    "Dry_subhumid": "#2166AC",
}
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

FIGSIZE = (3.15, 2.25)
AX_POS = [0.34, 0.25, 0.57, 0.62]

# =============================================================================
# Style
# =============================================================================
BASE_FONT = 6.8
LABEL_FONT = 6.3
TICK_FONT = 6.1
AXIS_LABEL_FONT = 6.8
TITLE_FONT = 6.9
VALUE_FONT = 5.9
GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
ZERO_COLOR = "#555555"


def apply_style() -> None:
    rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,
        "axes.linewidth": 0.62,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def pct_fmt(x, _pos):
    return f"{x * 100:.0f}%"


def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)

    rows = []
    for aridity in ARIDITY_ORDER:
        sub = df[df["aridity_class"] == aridity].copy()
        sub["quantile"] = pd.qcut(sub["SM90_L1_trend"], 5, labels=Q_LABELS, duplicates="drop")
        g = sub.groupby("quantile", observed=False)["share_abs_clim"].median().reindex(Q_LABELS)
        rows.append({
            "driver": "SM90_L1_trend",
            "aridity_class": aridity,
            "lowest_median": float(g.loc["Q1"]),
            "highest_median": float(g.loc["Q5"]),
            "highest_minus_lowest": float(g.loc["Q5"] - g.loc["Q1"]),
        })

    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(OUTPUT_CSV, index=False)

    vals = plot_df["highest_minus_lowest"].to_numpy(dtype=float)
    max_abs = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 0.02)
    lim = max_abs * 1.28

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes(AX_POS)
    ax.set_facecolor("none")

    tab = plot_df.copy()
    tab["label"] = tab["aridity_class"].map(ARIDITY_LABELS)
    y = np.arange(len(tab))
    colors = [ARIDITY_COLORS[a] for a in tab["aridity_class"]]

    ax.barh(y, tab["highest_minus_lowest"], height=0.52, color=colors, edgecolor="none", zorder=3)
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.72, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(tab) - 0.35, -0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(tab["label"], fontsize=LABEL_FONT, color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.set_xlabel("Highest − lowest climate share", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=4)
    ax.set_title("Moisture-state gradient", fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)

    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42, zorder=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(0.62)
    ax.tick_params(axis="x", colors=AXIS_COLOR, labelsize=TICK_FONT, length=2.4, width=0.62, pad=2)
    ax.tick_params(axis="y", length=0, pad=5)

    offset = lim * 0.035
    for yi, val in zip(y, tab["highest_minus_lowest"]):
        ha = "left" if val >= 0 else "right"
        x = val + offset if val >= 0 else val - offset
        ax.text(x, yi, f"{val * 100:+.1f}%", va="center", ha=ha, fontsize=VALUE_FONT, color=TEXT_COLOR)

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
