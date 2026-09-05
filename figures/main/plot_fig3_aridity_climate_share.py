#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact window-state-gate climate-share curves by aridity class.

Refined export version preserving the original quintile-based aggregation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "FIG5/fig5_UP_hex_plot_ready_core.csv"
OUTPUT_CSV = "FIG3_compact_window_state_gate_clim_refined_data.csv"
OUTPUT_PNG = "FIG3_compact_window_state_gate_clim_refined.png"
OUTPUT_PDF = "FIG3_compact_window_state_gate_clim_refined.pdf"
OUTPUT_SVG = "FIG3_compact_window_state_gate_clim_refined.svg"

# White background gives more stable rendering in Word/PDF viewers
# and avoids dark-mode artefacts from transparent PNGs.
EXPORT_TRANSPARENT = False

# 1200 dpi is preferable for raster export of line-art figures.
# PDF/SVG remain vector outputs.
DPI = 1200

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
TREND_GROUP_LABELS = ["Lowest", "Low", "Middle", "High", "Highest"]

# Variable abbreviations added to x-axis labels to link figure and text directly.
DRIVERS = [
    ("P90_trend", "Water-supply trend group (P90)"),
    ("SM90_L1_trend", "Moisture-state trend group (SM90)"),
    ("VPD_resid_trend", r"Net-drying trend group (VPD$_{\mathrm{resid}}$)"),
]

FIGSIZE = (4.95, 1.90)
AX_Y = 0.23
AX_H = 0.58
AX_W = 0.248
AX_LEFTS = [0.105, 0.395, 0.685]
LEGEND_Y = 0.94

# =============================================================================
# Style
# =============================================================================
# Slightly strengthened relative to the previous version so that the figure
# remains crisp after manuscript scaling.
BASE_FONT = 7.0
TICK_FONT = 6.3
AXIS_LABEL_FONT = 6.6
LEGEND_FONT = 6.0

LINE_WIDTH = 1.25
MARKER_SIZE = 3.2

GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"


def apply_style() -> None:
    rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,

        "axes.linewidth": 0.68,

        # Preserve editable/vector text in exported files
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "axes.unicode_minus": False,

        # Explicit antialiasing for raster rendering
        "text.antialiased": True,
        "lines.antialiased": True,

        # Keep saved background consistent
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def build_summary(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    rows = []

    for aridity in ARIDITY_ORDER:
        sub = df[df["aridity_class"] == aridity].copy()

        sub["quantile"] = pd.qcut(
            sub[driver],
            5,
            labels=Q_LABELS,
            duplicates="drop"
        )

        g = (
            sub.groupby("quantile", observed=False)["share_abs_clim"]
            .agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                n="size",
            )
            .reset_index()
        )

        g["aridity_class"] = aridity
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)

    out["x"] = out["quantile"].map(
        {q: i + 1 for i, q in enumerate(Q_LABELS)}
    )
    out["driver"] = driver

    return out[
        [
            "driver",
            "aridity_class",
            "quantile",
            "x",
            "n",
            "median",
            "q25",
            "q75",
        ]
    ]


def main() -> None:
    apply_style()

    df = pd.read_csv(INPUT_CSV)

    plot_df = pd.concat(
        [build_summary(df, d) for d, _ in DRIVERS],
        ignore_index=True
    )

    plot_df.to_csv(OUTPUT_CSV, index=False)

    ymax = max(
        0.16,
        float(plot_df["q75"].max()) * 1.05
    )

    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI,
        facecolor="white"
    )

    axes = [
        fig.add_axes([left, AX_Y, AX_W, AX_H])
        for left in AX_LEFTS
    ]

    for idx, (ax, (driver, xlabel)) in enumerate(zip(axes, DRIVERS)):

        sub_driver = plot_df[
            plot_df["driver"] == driver
        ]

        for aridity in ARIDITY_ORDER:
            sub = (
                sub_driver[
                    sub_driver["aridity_class"] == aridity
                ]
                .sort_values("x")
            )

            ax.plot(
                sub["x"],
                sub["median"],
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=ARIDITY_LABELS[aridity],
                color=ARIDITY_COLORS[aridity],
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
                zorder=3,
            )

        ax.set_facecolor("white")

        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0, ymax)

        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(
            TREND_GROUP_LABELS,
            fontsize=TICK_FONT,
            color=AXIS_COLOR
        )

        if idx > 0:
            ax.tick_params(labelleft=False)

        ax.set_xlabel(
            xlabel,
            fontsize=AXIS_LABEL_FONT,
            color=TEXT_COLOR,
            labelpad=4
        )

        # Fine adjustment for long panel labels
        xlabel_xpos = [0.50, 0.475, 0.525]
        ax.xaxis.set_label_coords(
            xlabel_xpos[idx],
            -0.175
        )

        ax.grid(
            axis="y",
            color=GRID_COLOR,
            lw=0.45,
            zorder=0
        )
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(AXIS_COLOR)
            ax.spines[spine].set_linewidth(0.68)

        ax.tick_params(
            axis="both",
            which="major",
            length=2.5,
            width=0.68,
            labelsize=TICK_FONT,
            colors=AXIS_COLOR,
            pad=2
        )

    axes[0].set_ylabel(
        "Median climate share",
        fontsize=AXIS_LABEL_FONT,
        color=TEXT_COLOR,
        labelpad=2
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, LEGEND_Y),
        fontsize=LEGEND_FONT,
        handlelength=1.45,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    if EXPORT_TRANSPARENT:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.set_facecolor("none")

        save_kwargs = {
            "transparent": True
        }

    else:
        fig.patch.set_facecolor("white")

        for ax in fig.axes:
            ax.set_facecolor("white")

        save_kwargs = {
            "transparent": False,
            "facecolor": "white"
        }

    # High-resolution raster output
    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        **save_kwargs
    )

    # Vector outputs: preferred for manuscript assembly
    fig.savefig(
        OUTPUT_PDF,
        **save_kwargs
    )

    fig.savefig(
        OUTPUT_SVG,
        **save_kwargs
    )

    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()