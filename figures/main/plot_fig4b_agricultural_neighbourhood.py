#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New Figure 4b: agricultural-neighborhood comparison in four aligned panels.

This script restyles the former Fig. 4c as the new Fig. 4b:
1) Observed trend
2) Endpoint change
3) Natural share
4) Natural-dominant rate

Design notes:
- Horizontal lines are lollipop guides from the panel baseline to the group value.
  They are not uncertainty intervals.
- Numeric labels use the same style as the new Fig. 4a: centered above the dot.
- The first panel carries the comparison-group labels.
- Outputs are transparent PNG, PDF and SVG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "outputs/figures/figure4"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = "outputs/agricultural_neighbourhood/ag_three_group_summary.csv"
OUTPUT_PNG = OUTDIR / "figure4b_agricultural_neighbourhood.png"
OUTPUT_PDF = OUTDIR / "figure4b_agricultural_neighbourhood.pdf"
OUTPUT_SVG = OUTDIR / "figure4b_agricultural_neighbourhood.svg"
OUTPUT_CSV = OUTDIR / "figure4b_agricultural_neighbourhood_data.csv"

EXPORT_TRANSPARENT = True
DPI = 600

# =============================================================================
# Groups and colors
# =============================================================================
GROUP_ORDER = ["Ag_self", "Ag_neighbor_nonAg", "Other_nonAg_UP"]
GROUP_LABELS = {
    "Ag_self": "Ag self",
    "Ag_neighbor_nonAg": "Ag-neighbor non-Ag",
    "Other_nonAg_UP": "Other non-Ag UP",
}
GROUP_COLORS = {
    # Same strengthened low-saturation palette as Fig. 4a.
    "Ag_self": "#89C4DD",
    "Ag_neighbor_nonAg": "#6FB785",
    "Other_nonAg_UP": "#AFAFAF",
}

METRICS = [
    "beta_obs_median",
    "endpoint_change_median",
    "share_nat_co2_median",
    "natural_dom_rate",
]
PANEL_TITLES = [
    "Observed trend",
    "Endpoint change",
    "Natural share",
    "Natural-dominant rate",
]
AXIS_LABELS = {
    "beta_obs_median": "Observed trend ($\\beta_{obs}$)",
    "endpoint_change_median": "Endpoint change",
    "share_nat_co2_median": "Natural share (%)",
    "natural_dom_rate": "Natural-dom. rate (%)",
}

# =============================================================================
# Figure style
# =============================================================================
FIGSIZE = (7.65, 2.25)
# Shared with the new Fig. 4a script so the four lower subpanels align
# vertically with the four upper subpanels after manual assembly.
LEFT = 0.165
RIGHT = 0.985
BOTTOM = 0.260
TOP = 0.835
WSPACE = 0.220

BASE_FONT = 6.4
TITLE_FONT = 6.2
LABEL_FONT = 5.8
TICK_FONT = 5.35
VALUE_FONT = 5.30
GRID_COLOR = "#E6E6E6"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
LABEL_HALO = [pe.withStroke(linewidth=1.35, foreground="white")]


def resolve_path(path_like: str) -> str:
    candidates = [
        Path(path_like),
        REPO_ROOT / path_like,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return path_like


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


def style_axes(ax) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.68)
    ax.tick_params(axis="x", labelsize=TICK_FONT, colors=AXIS_COLOR, length=3.2, width=0.62, pad=1.2)
    ax.tick_params(axis="y", length=0, pad=4)


def draw_grid(ax, ticks) -> None:
    for tick in ticks:
        ax.axvline(float(tick), color=GRID_COLOR, lw=0.82, zorder=0)


def draw_lollipop(ax, y, baseline, value, color, label, y_offset=0.12) -> None:
    ax.hlines(y, baseline, value, color=color, lw=2.05, zorder=2)
    ax.scatter(value, y, s=38, color=color, edgecolor="white", linewidth=0.78, zorder=3)
    ax.text(
        value,
        y + y_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=VALUE_FONT,
        color=TEXT_COLOR,
        path_effects=LABEL_HALO,
        zorder=4,
        clip_on=False,
    )


def format_value(metric: str, value: float) -> str:
    if metric in ["share_nat_co2_median", "natural_dom_rate"]:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def clean_decimal(x, pos):
    if abs(x) < 1e-12:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.3f}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    apply_style()
    df = pd.read_csv(resolve_path(INPUT_CSV))
    df = df.set_index("group").loc[GROUP_ORDER].reset_index()
    df.to_csv(OUTPUT_CSV, index=False)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    gs = GridSpec(1, 4, figure=fig, left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    y = np.arange(len(GROUP_ORDER))[::-1]

    fig.text(0.020, 0.520, "Comparison group", rotation=90, va="center", ha="center",
             fontsize=LABEL_FONT, color=TEXT_COLOR)

    for idx, (ax, metric, title) in enumerate(zip(axes, METRICS, PANEL_TITLES)):
        vals = df[metric].astype(float).to_numpy()

        if metric == "natural_dom_rate":
            xmin = max(0.0, float(np.nanmin(vals)) - 0.035)
            xmax = min(1.0, float(np.nanmax(vals)) + 0.045)
            xticks = np.array([0.66, 0.72, 0.78])
            formatter = PercentFormatter(1.0, decimals=0)
        elif metric == "share_nat_co2_median":
            xmin = max(0.0, float(np.nanmin(vals)) - 0.045)
            xmax = min(1.0, float(np.nanmax(vals)) + 0.055)
            xticks = np.array([0.52, 0.56, 0.60])
            formatter = PercentFormatter(1.0, decimals=0)
        elif metric == "endpoint_change_median":
            xmin = 0.0
            xmax = float(np.nanmax(vals)) * 1.16
            xticks = np.array([0.00, 0.08, 0.16])
            formatter = FuncFormatter(clean_decimal)
        else:
            xmin = 0.0
            xmax = float(np.nanmax(vals)) * 1.18
            xticks = np.array([0.000, 0.004, 0.008])
            formatter = FuncFormatter(clean_decimal)

        xticks = xticks[(xticks >= xmin - 1e-12) & (xticks <= xmax + 1e-12)]
        draw_grid(ax, xticks)

        for yi, group in zip(y, GROUP_ORDER):
            row = df.loc[df["group"] == group].iloc[0]
            value = float(row[metric])
            color = GROUP_COLORS[group]
            draw_lollipop(ax, yi, xmin, value, color, format_value(metric, value))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.55, len(GROUP_ORDER) - 0.40)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
        if idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], fontsize=LABEL_FONT, color=TEXT_COLOR)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        ax.set_xlabel(AXIS_LABELS[metric], fontsize=LABEL_FONT, color=TEXT_COLOR, labelpad=3)
        ax.set_title(title, fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
        style_axes(ax)

    if EXPORT_TRANSPARENT:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.set_facecolor("none")

    save_kwargs = dict(dpi=DPI, transparent=EXPORT_TRANSPARENT)
    fig.savefig(OUTPUT_PNG, **save_kwargs)
    fig.savefig(OUTPUT_PDF, transparent=EXPORT_TRANSPARENT)
    fig.savefig(OUTPUT_SVG, transparent=EXPORT_TRANSPARENT)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_SVG}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
