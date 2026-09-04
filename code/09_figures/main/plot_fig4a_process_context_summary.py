#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New Figure 4a: process-context comparison in four aligned panels.

This script merges the former Fig. 4a and Fig. 4b into one compact panel set:
1) Climate share
2) Observed trend
3) Natural share
4) Residual trend

Design notes:
- The first panel carries the process-context labels.
- All numeric labels use the same style: centered above the median point.
- Horizontal bars represent P25-P75 ranges; dots represent P50 / median.
- The residual-trend panel does not draw a zero vertical guide line.
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
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTDIR = REPO_ROOT / "outputs/figures/figure4"
OUTDIR.mkdir(parents=True, exist_ok=True)

INPUT_CLIMATE = "data/processed/process_context/process_context_attribution_hydroclimate_up.csv"
INPUT_ATTR = "data/processed/agricultural_neighbourhood/ag_neighborhood_analysis_table.csv"

OUTPUT_PNG = OUTDIR / "figure4a_process_context_summary.png"
OUTPUT_PDF = OUTDIR / "figure4a_process_context_summary.pdf"
OUTPUT_SVG = OUTDIR / "figure4a_process_context_summary.svg"
OUTPUT_CSV_CLIMATE = OUTDIR / "figure4a_process_context_climate_share_quantiles.csv"
OUTPUT_CSV_ATTR = OUTDIR / "figure4a_process_context_attribution_quantiles.csv"

EXPORT_TRANSPARENT = True
DPI = 600

# =============================================================================
# Process classes and colors
# =============================================================================
PROCESS_ORDER_CANONICAL = [
    "Other",
    "No_transition",
    "Bare_to_Sparse",
    "Bare_or_sparse_to_grass_or_forest",
    "Ag_expansion",
    "Urban_expansion",
]

# Some upstream files used an earlier internal name for the same class.
PROCESS_ALIASES = {
    "Bare_or_sparse_to_grass_or_forest": [
        "Bare_or_sparse_to_grass_or_forest",
        "veg_cover_upgrade",
        "Nat_recovery",
    ],
}

DISPLAY_LABELS = {
    "Other": "Other",
    "No_transition": "No transition",
    "Bare_to_Sparse": "Bare to sparse",
    "Bare_or_sparse_to_grass_or_forest": "Bare/sparse to grass/forest",
    "Ag_expansion": "Agricultural expansion",
    "Urban_expansion": "Urban expansion",
}

PROCESS_COLORS = {
    # Slightly stronger low-saturation palette, harmonized with Fig. 1-3.
    "Other": "#AFAFAF",
    "No_transition": "#D1B982",
    "Bare_to_Sparse": "#6FB785",
    "Bare_or_sparse_to_grass_or_forest": "#43965F",
    "Ag_expansion": "#89C4DD",
    "Urban_expansion": "#838383",
}

PANEL_TITLES = [
    "Climate share",
    "Observed trend",
    "Natural share",
    "Residual trend",
]

# =============================================================================
# Figure style
# =============================================================================
FIGSIZE = (7.65, 2.48)
# These geometry values are shared with the new Fig. 4b script so that
# all eight subpanels align vertically when the two exported files are stacked.
LEFT = 0.165
RIGHT = 0.985
BOTTOM = 0.255
TOP = 0.840
WSPACE = 0.220
WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0]

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
    """Resolve a repository-relative or user-supplied path."""
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


def canonical_values(df: pd.DataFrame, col: str, canonical: str) -> pd.Series:
    """Return values for a canonical process name, accepting known aliases."""
    names = PROCESS_ALIASES.get(canonical, [canonical])
    return df.loc[df[col].isin(names), :]


def build_climate_summary() -> pd.DataFrame:
    df = pd.read_csv(resolve_path(INPUT_CLIMATE))
    df = df.dropna(subset=["group_type", "share_nat_clim"]).copy()

    rows = []
    for process in PROCESS_ORDER_CANONICAL:
        sub_df = canonical_values(df, "group_type", process)
        ser = sub_df["share_nat_clim"].dropna()
        if ser.empty:
            continue
        rows.append({
            "process": process,
            "label": DISPLAY_LABELS[process],
            "q25": ser.quantile(0.25),
            "q50": ser.median(),
            "q75": ser.quantile(0.75),
            "n": int(ser.size),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV_CLIMATE, index=False)
    return out


def build_attr_summary() -> pd.DataFrame:
    df = pd.read_csv(resolve_path(INPUT_ATTR))
    metrics = ["beta_obs", "share_nat_co2", "beta_res_co2"]

    rows = []
    for process in PROCESS_ORDER_CANONICAL:
        sub_df = canonical_values(df, "pathway_type", process)
        if sub_df.empty:
            continue
        rec = {"process": process, "label": DISPLAY_LABELS[process]}
        for metric in metrics:
            ser = sub_df[metric].dropna()
            rec[f"{metric}_q25"] = ser.quantile(0.25)
            rec[f"{metric}_q50"] = ser.median()
            rec[f"{metric}_q75"] = ser.quantile(0.75)
            rec[f"{metric}_n"] = int(ser.size)
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV_ATTR, index=False)
    return out


def style_axes(ax) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.68)
    ax.tick_params(axis="x", labelsize=TICK_FONT, colors=AXIS_COLOR, length=3.2, width=0.62, pad=1.2)
    ax.tick_params(axis="y", length=0, pad=4)


def draw_grid(ax, ticks, skip_zero=False) -> None:
    for tick in ticks:
        if skip_zero and abs(float(tick)) < 1e-12:
            continue
        ax.axvline(float(tick), color=GRID_COLOR, lw=0.82, zorder=0)


def draw_quantile(ax, y, q25, q50, q75, color, label, y_offset=0.12) -> None:
    ax.hlines(y, q25, q75, color=color, lw=2.05, zorder=2)
    cap_h = 0.115
    ax.vlines([q25, q75], y - cap_h, y + cap_h, color=color, lw=1.25, zorder=2)
    ax.scatter(q50, y, s=38, color=color, edgecolor="white", linewidth=0.78, zorder=3)
    ax.text(
        q50,
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


def fmt_attr(metric: str, value: float) -> str:
    if metric == "share_nat_co2":
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def clean_numeric_formatter(x, pos):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.3f}" if abs(x) < 0.01 else f"{x:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    apply_style()
    climate = build_climate_summary()
    attr = build_attr_summary()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    gs = GridSpec(
        1,
        4,
        figure=fig,
        left=LEFT,
        right=RIGHT,
        bottom=BOTTOM,
        top=TOP,
        wspace=WSPACE,
        width_ratios=WIDTH_RATIOS,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    fig.text(0.020, 0.525, "Process context", rotation=90, va="center", ha="center",
             fontsize=LABEL_FONT, color=TEXT_COLOR)

    # -------------------------------------------------------------------------
    # Panel 1: climate-share quantiles
    # -------------------------------------------------------------------------
    ax = axes[0]
    y = np.arange(len(climate))[::-1]
    q75_max = float(climate["q75"].max()) if not climate.empty else 0.30
    x_max = max(0.35, np.ceil((q75_max * 100 + 3) / 5) * 5 / 100.0)
    xticks = np.arange(0, x_max + 1e-9, 0.10)
    draw_grid(ax, xticks)

    for yi, (_, row) in zip(y, climate.iterrows()):
        process = row["process"]
        color = PROCESS_COLORS.get(process, "#B5B5B5")
        draw_quantile(
            ax,
            yi,
            float(row["q25"]),
            float(row["q50"]),
            float(row["q75"]),
            color,
            f"{float(row['q50']) * 100:.1f}%",
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.55, len(climate) - 0.40)
    ax.set_yticks(y)
    ax.set_yticklabels(climate["label"].tolist(), fontsize=LABEL_FONT, color=TEXT_COLOR)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Climate share (%)", fontsize=LABEL_FONT, color=TEXT_COLOR, labelpad=3)
    ax.set_title(PANEL_TITLES[0], fontsize=TITLE_FONT, color=TEXT_COLOR, pad=4)
    style_axes(ax)

    # -------------------------------------------------------------------------
    # Panels 2-4: attribution-position quantiles
    # -------------------------------------------------------------------------
    metric_specs = [
        ("beta_obs", "Observed trend ($\\beta_{obs}$)", PANEL_TITLES[1]),
        ("share_nat_co2", "Natural share (%)", PANEL_TITLES[2]),
        ("beta_res_co2", "Residual trend ($\\beta_{res}$)", PANEL_TITLES[3]),
    ]

    for ax, (metric, xlabel, title) in zip(axes[1:], metric_specs):
        y = np.arange(len(attr))[::-1]
        q25 = attr[f"{metric}_q25"].astype(float).to_numpy()
        q50 = attr[f"{metric}_q50"].astype(float).to_numpy()
        q75 = attr[f"{metric}_q75"].astype(float).to_numpy()

        if metric == "share_nat_co2":
            xmin = max(0.0, float(np.nanmin(q25)) - 0.040)
            xmax = min(1.0, float(np.nanmax(q75)) + 0.055)
            xticks = np.array([0.25, 0.50, 0.75])
            formatter = PercentFormatter(1.0, decimals=0)
            skip_zero_grid = False
        elif metric == "beta_res_co2":
            xmin = float(np.nanmin(q25)) * 1.18
            xmax = max(0.0015, float(np.nanmax(q75)) + 0.0010)
            xticks = np.array([-0.02, -0.01])
            formatter = FuncFormatter(clean_numeric_formatter)
            skip_zero_grid = True
        else:
            xmin = 0.0 if float(np.nanmin(q25)) >= 0 else float(np.nanmin(q25)) * 1.15
            xmax = float(np.nanmax(q75)) * 1.15
            xticks = np.array([0.00, 0.01, 0.02])
            formatter = FuncFormatter(clean_numeric_formatter)
            skip_zero_grid = False

        xticks = xticks[(xticks >= xmin - 1e-12) & (xticks <= xmax + 1e-12)]
        draw_grid(ax, xticks, skip_zero=skip_zero_grid)

        for yi, (_, row) in zip(y, attr.iterrows()):
            process = row["process"]
            color = PROCESS_COLORS.get(process, "#B5B5B5")
            draw_quantile(
                ax,
                yi,
                float(row[f"{metric}_q25"]),
                float(row[f"{metric}_q50"]),
                float(row[f"{metric}_q75"]),
                color,
                fmt_attr(metric, float(row[f"{metric}_q50"])),
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.55, len(attr) - 0.40)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT, color=TEXT_COLOR, labelpad=3)
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
    print(f"Saved: {OUTPUT_CSV_CLIMATE}")
    print(f"Saved: {OUTPUT_CSV_ATTR}")


if __name__ == "__main__":
    main()
