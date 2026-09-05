#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aridity-specific climate-share response along a trend-group gradient.

Refined export version with readable trend-group labels and vector outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "FIG5/fig5_UP_hex_plot_ready_core.csv"
OUTPUT_CSV = "FIG3a_vpdresid_trend_share_abs_clim_facets_data_refined.csv"
OUTPUT_PNG = "FIG3a_vpdresid_trend_share_abs_clim_facets_refined.png"
OUTPUT_PDF = "FIG3a_vpdresid_trend_share_abs_clim_facets_refined.pdf"
OUTPUT_SVG = "FIG3a_vpdresid_trend_share_abs_clim_facets_refined.svg"
EXPORT_TRANSPARENT = True
DPI = 600

ORDER = ["Hyperarid", "Arid", "Semiarid", "Dry_subhumid"]
ARIDITY_LABELS = {
    "Hyperarid": "Hyperarid",
    "Arid": "Arid",
    "Semiarid": "Semiarid",
    "Dry_subhumid": "Dry subhumid",
}
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TREND_GROUP_LABELS = ["Lowest", "Low", "Middle", "High", "Highest"]
DRIVER_COL = "VPD_resid_trend"
X_LABEL = "Net-drying trend group"

FIGSIZE = (6.35, 1.95)
AX_LEFTS = [0.085, 0.315, 0.545, 0.775]
AX_Y = 0.24
AX_W = 0.185
AX_H = 0.62
BASE_FONT = 6.8
TICK_FONT = 5.8
TITLE_FONT = 6.7
AXIS_LABEL_FONT = 6.7
LINE_COLOR = "#2166AC"
FILL_COLOR = "#9ECAE1"
GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"


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


def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)
    rows = []
    for aridity in ORDER:
        sub = df[df["aridity_class"] == aridity].copy()
        sub["quantile"] = pd.qcut(sub[DRIVER_COL], 5, labels=Q_LABELS, duplicates="drop")
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
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df["x"] = plot_df["quantile"].map({q: i + 1 for i, q in enumerate(Q_LABELS)})
    plot_df = plot_df[["aridity_class", "quantile", "x", "n", "median", "q25", "q75"]]
    plot_df.to_csv(OUTPUT_CSV, index=False)

    ymax = max(0.16, float(plot_df["q75"].max()) * 1.05)
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    axes = [fig.add_axes([left, AX_Y, AX_W, AX_H]) for left in AX_LEFTS]

    for i, (ax, aridity) in enumerate(zip(axes, ORDER)):
        sub = plot_df[plot_df["aridity_class"] == aridity].sort_values("x")
        ax.fill_between(sub["x"].astype(float), sub["q25"].astype(float), sub["q75"].astype(float), color=FILL_COLOR, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(sub["x"], sub["median"], marker="o", linewidth=1.15, markersize=2.7, color=LINE_COLOR, zorder=3)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0, ymax)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(TREND_GROUP_LABELS, fontsize=TICK_FONT, color=AXIS_COLOR, rotation=0)
        if i == 0:
            ax.set_ylabel("Median climate share", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=3)
        else:
            ax.set_yticklabels([])
        ax.set_title(ARIDITY_LABELS[aridity], fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
        ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=4)
        ax.grid(axis="y", color=GRID_COLOR, lw=0.42, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(AXIS_COLOR)
            ax.spines[spine].set_linewidth(0.62)
        ax.tick_params(axis="both", which="major", length=2.4, width=0.62, labelsize=TICK_FONT, colors=AXIS_COLOR, pad=2)

    if EXPORT_TRANSPARENT:
        fig.patch.set_alpha(0)
        for a in fig.axes:
            a.set_facecolor("none")

    save_kwargs = dict(transparent=EXPORT_TRANSPARENT)
    fig.savefig(OUTPUT_PNG, dpi=DPI, **save_kwargs)
    fig.savefig(OUTPUT_PDF, **save_kwargs)
    fig.savefig(OUTPUT_SVG, **save_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
