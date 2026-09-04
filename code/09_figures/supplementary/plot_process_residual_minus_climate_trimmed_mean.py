#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Six process-context comparison: residual minus climate share.

Refined export version with percentage display, black labels and consistent output formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "data/processed/process_context/process_context_attribution_hydroclimate_up.csv"
OUTPUT_CSV = "FIG3_sixgroups_residual_minus_climate_trimmedmean_data_label_fixed_v2.csv"
OUTPUT_PNG = "FIG3_sixgroups_residual_minus_climate_trimmedmean_label_fixed_v2.png"
OUTPUT_PDF = "FIG3_sixgroups_residual_minus_climate_trimmedmean_label_fixed_v2.pdf"
OUTPUT_SVG = "FIG3_sixgroups_residual_minus_climate_trimmedmean_label_fixed_v2.svg"
EXPORT_TRANSPARENT = True
DPI = 600

PROCESS_COLORS = {
    "Bare_or_sparse_to_grass_or_forest": "#4F9D68",
    "Bare_to_Sparse": "#77B98A",
    "Other": "#B5B5B5",
    "No_transition": "#D9C7A3",
    "Ag_expansion": "#9ECAE1",
    "Urban_expansion": "#8D8D8D",
}
DISPLAY_LABELS = {
    "Other": "Other",
    "No_transition": "No transition",
    "Bare_to_Sparse": "Bare to sparse",
    "Bare_or_sparse_to_grass_or_forest": "Bare/sparse to grass/forest",
    "Ag_expansion": "Agricultural expansion",
    "Urban_expansion": "Urban expansion",
}

FIGSIZE = (5.15, 2.95)
AX_POS = [0.34, 0.23, 0.61, 0.66]
BASE_FONT = 6.8
LABEL_FONT = 6.2
TICK_FONT = 6.1
AXIS_LABEL_FONT = 6.8
VALUE_FONT = 5.8
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


def trimmed_mean(x, proportion=0.10):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    k = int(np.floor(len(x) * proportion))
    if len(x) - 2 * k <= 0:
        return float(np.mean(x))
    return float(np.mean(x[k:len(x) - k]))


def main() -> None:
    apply_style()
    df = pd.read_csv(INPUT_CSV)
    use = df.dropna(subset=["group_type", "share_abs_res", "share_abs_clim"]).copy()
    use["delta_res_minus_clim"] = use["share_abs_res"] - use["share_abs_clim"]

    group_order = use.groupby("group_type")["hex_id"].size().sort_values(ascending=False).index.tolist()
    summary = (
        use.groupby("group_type")
           .agg(
               n=("hex_id", "size"),
               trimmed_mean_delta=("delta_res_minus_clim", lambda x: trimmed_mean(x, 0.10)),
               median_delta=("delta_res_minus_clim", "median"),
           )
           .reset_index()
    )
    summary["group_type"] = pd.Categorical(summary["group_type"], categories=group_order, ordered=True)
    summary = summary.sort_values("group_type").reset_index(drop=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    vals = summary["trimmed_mean_delta"].to_numpy(dtype=float) * 100.0
    med_vals = summary["median_delta"].to_numpy(dtype=float) * 100.0
    labels = [DISPLAY_LABELS.get(g, g) for g in summary["group_type"].astype(str)]
    colors = [PROCESS_COLORS.get(g, "#B5B5B5") for g in summary["group_type"].astype(str)]

    data_min = float(np.nanmin(vals))
    data_max = float(np.nanmax(vals))
    plot_xmin = 0.0 if data_min >= 0 else np.floor((data_min - 2) / 5) * 5
    plot_xmax = max(40.0, np.ceil((data_max + 4) / 5) * 5)
    # Keep the primary metric next to the bar end and reserve the right-side
    # annotation lane only for the secondary distribution/sample-size summary.
    value_pad = max(0.8, plot_xmax * 0.022)
    full_xmax = plot_xmax + 42.0
    detail_x = full_xmax - 1.5

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes(AX_POS)
    y = np.arange(len(summary))

    ax.barh(y, vals, height=0.50, color=colors, edgecolor="none", zorder=3)
    if plot_xmin < 0:
        ax.axvline(0, color="#555555", lw=0.68, zorder=2)

    xticks = np.arange(plot_xmin, plot_xmax + 0.1, 10)
    for x in xticks:
        ax.vlines(x, -0.5, len(summary) - 0.5, colors=GRID_COLOR, linewidth=0.42, zorder=0)
    ax.set_xlim(plot_xmin, full_xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks], fontsize=TICK_FONT, color=AXIS_COLOR)

    for yi, v, med, n in zip(y, vals, med_vals, summary["n"]):
        metric_x = v + value_pad if v >= 0 else v - value_pad
        metric_ha = "left" if v >= 0 else "right"
        ax.text(
            metric_x, yi, f"TMean={v:.1f}%",
            va="center", ha=metric_ha, fontsize=VALUE_FONT, color=TEXT_COLOR,
            clip_on=False,
        )
        ax.text(
            detail_x, yi, f"Median={med:.1f}%; n={n:,}",
            va="center", ha="right", fontsize=VALUE_FONT, color="#555555",
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=LABEL_FONT, color=TEXT_COLOR)
    ax.set_ylim(len(summary) - 0.35, -0.65)
    ax.set_xlabel("Residual - climate share (10% trimmed mean, %)", fontsize=AXIS_LABEL_FONT, color=TEXT_COLOR, labelpad=4)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_linewidth(0.62)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=TICK_FONT, length=2.4, width=0.62, colors=AXIS_COLOR, pad=2)

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
