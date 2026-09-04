#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aridity-class heatmap of median climate share along window-state-gate gradients.

Refined export version preserving the original aggregation logic.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "FIG5/fig5_UP_hex_plot_ready_core.csv"
OUTPUT_CSV = "FIG3_heatmap_homologous_window_state_gate_clim_refined_data.csv"
OUTPUT_PNG = "FIG3_heatmap_homologous_window_state_gate_clim_refined.png"
OUTPUT_PDF = "FIG3_heatmap_homologous_window_state_gate_clim_refined.pdf"
OUTPUT_SVG = "FIG3_heatmap_homologous_window_state_gate_clim_refined.svg"
EXPORT_TRANSPARENT = True
DPI = 600

# =============================================================================
# Configuration
# =============================================================================
ARIDITY_ORDER = ["Hyperarid", "Arid", "Semiarid", "Dry_subhumid"]
ROW_LABELS = ["Hyperarid", "Arid", "Semiarid", "Dry subhumid"]
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TREND_GROUP_LABELS = ["Lowest", "Low", "Middle", "High", "Highest"]
DRIVERS = [
    ("P90_trend", "Window"),
    ("SM90_L1_trend", "State"),
    ("VPD_resid_trend", "Gate"),
]

FIGSIZE = (5.25, 1.95)
GRID_LEFT = 0.13
GRID_RIGHT = 0.90
GRID_BOTTOM = 0.25
GRID_TOP = 0.86

CLIMATE_CMAP = LinearSegmentedColormap.from_list(
    "climate_share_blue",
    ["#F7FBFF", "#D8EAF7", "#9ECAE1", "#6BAED6", "#2166AC"],
    N=256,
)

# =============================================================================
# Style
# =============================================================================
BASE_FONT = 6.8
TICK_FONT = 5.9
AXIS_LABEL_FONT = 6.8
TITLE_FONT = 6.9
CELL_FONT = 5.4
CBAR_FONT = 5.9
GRID_COLOR = "#FFFFFF"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"


def apply_style() -> None:
    rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def build_long(df: pd.DataFrame, driver: str, panel_label: str, value_col: str = "share_abs_clim") -> pd.DataFrame:
    rows = []
    for aridity in ARIDITY_ORDER:
        sub = df[df["aridity_class"] == aridity].copy()
        sub["quantile"] = pd.qcut(sub[driver], 5, labels=Q_LABELS, duplicates="drop")
        g = (
            sub.groupby("quantile", observed=False)[value_col]
            .agg(median="median", n="size")
            .reset_index()
        )
        g["aridity_class"] = aridity
        g["panel"] = panel_label
        g["driver"] = driver
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["quantile"] = pd.Categorical(out["quantile"], categories=Q_LABELS, ordered=True)
    return out[["panel", "driver", "aridity_class", "quantile", "n", "median"]]


def build_matrix(plot_df: pd.DataFrame, driver: str) -> pd.DataFrame:
    return (
        plot_df[plot_df["driver"] == driver]
        .pivot(index="aridity_class", columns="quantile", values="median")
        .reindex(index=ARIDITY_ORDER, columns=Q_LABELS)
    )


def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)
    plot_df = pd.concat([build_long(df, d, p) for d, p in DRIVERS], ignore_index=True)
    plot_df.to_csv(OUTPUT_CSV, index=False)
    mats = {driver: build_matrix(plot_df, driver) for driver, _ in DRIVERS}

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(
        1, 4,
        left=GRID_LEFT, right=GRID_RIGHT, bottom=GRID_BOTTOM, top=GRID_TOP,
        width_ratios=[1, 1, 1, 0.07], wspace=0.22,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    vmin, vmax = 0.0, 0.20
    im = None
    for ax, (driver, panel_label) in zip(axes, DRIVERS):
        mat = mats[driver]
        ax.set_facecolor("none")
        im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=CLIMATE_CMAP)
        ax.set_title(panel_label, fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
        ax.set_xticks(range(len(Q_LABELS)))
        ax.set_xticklabels(TREND_GROUP_LABELS, fontsize=TICK_FONT, color=AXIS_COLOR)
        ax.set_yticks(range(len(ARIDITY_ORDER)))
        if ax is axes[0]:
            ax.set_yticklabels(ROW_LABELS, fontsize=TICK_FONT, color=TEXT_COLOR)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", which="major", length=0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks([x - 0.5 for x in range(1, len(Q_LABELS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(ARIDITY_ORDER))], minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linewidth=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                text_color = "white" if val >= 0.09 else TEXT_COLOR
                ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center", fontsize=CELL_FONT, color=text_color)

    axes[0].set_ylabel("Aridity class", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=4)
    fig.supxlabel("Trend group, from lowest to highest", fontsize=AXIS_LABEL_FONT, y=0.08, color=TEXT_COLOR)

    cax.set_facecolor("none")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Median climate share", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=5)
    cbar.ax.tick_params(labelsize=CBAR_FONT, length=2.2, width=0.55, colors=AXIS_COLOR)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    cbar.outline.set_linewidth(0.55)

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
