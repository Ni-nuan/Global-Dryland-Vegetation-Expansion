#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-context window-gate heat maps: cell counts.

Refined export version with readable trend-group labels, black panel titles and vector outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib import colormaps

INPUT_CSV = "data/processed/process_context/process_context_attribution_hydroclimate_up.csv"
OUTPUT_CSV = "FIG3_support_cellcount_heatmap_masked_data_refined.csv"
OUTPUT_PNG = "FIG3_support_cellcount_heatmap_masked_refined.png"
OUTPUT_PDF = "FIG3_support_cellcount_heatmap_masked_refined.pdf"
OUTPUT_SVG = "FIG3_support_cellcount_heatmap_masked_refined.svg"
EXPORT_TRANSPARENT = True
DPI = 600
MIN_CELL_N = 50

DISPLAY_LABELS = {
    "Other": "Other",
    "No_transition": "No transition",
    "Bare_to_Sparse": "Bare to sparse",
    "Bare_or_sparse_to_grass_or_forest": "Bare/sparse to grass/forest",
    "Ag_expansion": "Agricultural expansion",
    "Urban_expansion": "Urban expansion",
}
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TREND_GROUP_LABELS = ["Lowest", "Low", "Middle", "High", "Highest"]

FIGSIZE = (5.65, 3.65)
BASE_FONT = 6.8
TITLE_FONT = 6.8
TICK_FONT = 5.9
AXIS_LABEL_FONT = 6.8
CBAR_FONT = 6.2
CBAR_LABEL_FONT = 6.8
GRID_COLOR = "#FFFFFF"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
MISSING_CMAP = ListedColormap(["#D9D9D9"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })


def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)
    use = df.dropna(subset=["group_type", "P90_trend", "VPD_resid_trend"]).copy()
    use["P90_q"] = pd.qcut(use["P90_trend"], 5, labels=Q_LABELS)
    use["VPD_q"] = pd.qcut(use["VPD_resid_trend"], 5, labels=Q_LABELS)

    agg = (
        use.groupby(["group_type", "P90_q", "VPD_q"], observed=False)
           .agg(n=("hex_id", "size"))
           .reset_index()
    )
    agg.to_csv(OUTPUT_CSV, index=False)

    group_order = use.groupby("group_type")["hex_id"].size().sort_values(ascending=False).index.tolist()
    heatmaps = {}
    for g in group_order:
        sub = agg[agg["group_type"] == g].copy()
        mat = sub.pivot(index="P90_q", columns="VPD_q", values="n").reindex(index=Q_LABELS, columns=Q_LABELS)
        heatmaps[g] = mat.iloc[::-1]

    all_vals = np.concatenate([m.to_numpy(dtype=float).ravel() for m in heatmaps.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    valid_vals = all_vals[all_vals >= MIN_CELL_N]
    norm = LogNorm(vmin=max(MIN_CELL_N, int(valid_vals.min())), vmax=int(valid_vals.max()))
    cmap = colormaps["YlGnBu"]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(
        2, 4,
        left=0.095, right=0.925, bottom=0.145, top=0.905,
        width_ratios=[1, 1, 1, 0.075],
        wspace=0.22, hspace=0.34,
    )
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    cax = fig.add_subplot(gs[:, 3])

    for ax, g in zip(axes, group_order):
        ax.set_facecolor("none")
        data = heatmaps[g].to_numpy(dtype=float)
        masked = np.ma.masked_where(data < MIN_CELL_N, data)
        ax.imshow(masked, cmap=cmap, norm=norm, interpolation="none", aspect="equal")
        ax.imshow(np.where(data < MIN_CELL_N, 1, np.nan), cmap=MISSING_CMAP, interpolation="none", aspect="equal")

        for i in range(6):
            ax.axhline(i - 0.5, color=GRID_COLOR, lw=0.55, zorder=3)
            ax.axvline(i - 0.5, color=GRID_COLOR, lw=0.55, zorder=3)

        ax.set_xticks(range(5))
        ax.set_xticklabels(TREND_GROUP_LABELS, fontsize=TICK_FONT, color=AXIS_COLOR)
        ax.set_yticks(range(5))
        ax.set_yticklabels(TREND_GROUP_LABELS[::-1], fontsize=TICK_FONT, color=AXIS_COLOR)
        ax.tick_params(length=0, pad=1.5)
        ax.set_title(DISPLAY_LABELS.get(g, g), fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[:3]:
        ax.set_xticklabels([])
    for ax in [axes[1], axes[2], axes[4], axes[5]]:
        ax.set_yticklabels([])

    fig.supxlabel("Net-drying trend group, from lowest to highest", fontsize=AXIS_LABEL_FONT, y=0.055, color=TEXT_COLOR)
    fig.supylabel("Water-supply trend group", fontsize=AXIS_LABEL_FONT, x=0.035, color=TEXT_COLOR)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f"Cell count (masked if <{MIN_CELL_N})", fontsize=CBAR_LABEL_FONT, labelpad=5, color=TEXT_COLOR)
    cb.ax.tick_params(labelsize=CBAR_FONT, length=2.2, width=0.55, colors=AXIS_COLOR)
    cb.outline.set_linewidth(0.55)

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
