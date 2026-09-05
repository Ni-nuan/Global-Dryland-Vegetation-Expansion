#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact CO2 budget-share responses across window, state and gate gradients.

Refined export version preserving the original summary logic and y-range.
"""

import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "FIG5/fig5_UP_hex_plot_ready_core.csv"
OUTPUT_CSV = "maintext_FIG3b_compact_window_state_gate_co2_formal_zoomed_y040_060.csv"
OUTPUT_PNG = "maintext_FIG3b_compact_window_state_gate_co2_formal_zoomed_y040_060.png"
OUTPUT_PDF = "maintext_FIG3b_compact_window_state_gate_co2_formal_zoomed_y040_060.pdf"
OUTPUT_SVG = "maintext_FIG3b_compact_window_state_gate_co2_formal_zoomed_y040_060.svg"
# White background gives more stable rendering in Word/PDF viewers.
EXPORT_TRANSPARENT = False

# High-resolution raster export; PDF/SVG remain vector.
DPI = 1200

# =============================================================================
# Settings
# =============================================================================
ARIDITY_ORDER = [
    "Hyperarid",
    "Arid",
    "Semiarid",
    "Dry_subhumid",
]

LABELS_MAP = {
    "Hyperarid": "Hyperarid",
    "Arid": "Arid",
    "Semiarid": "Semiarid",
    "Dry_subhumid": "Dry subhumid",
}

# Keep colours identical to the revised climate-share panel.
ARIDITY_COLORS = {
    "Hyperarid": "#9A7A2F",
    "Arid": "#D7A257",
    "Semiarid": "#9ECAE1",
    "Dry_subhumid": "#2166AC",
}

# Variable abbreviations added to link the figure directly to the main text.
DRIVERS = [
    ("P90_trend", "Water-supply trend group (P90)"),
    ("SM90_L1_trend", "Moisture-state trend group (SM90)"),
    ("VPD_resid_trend", r"Net-drying trend group (VPD$_{\mathrm{resid}}$)"),
]

Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

TREND_GROUP_LABELS = [
    "Lowest",
    "Low",
    "Middle",
    "High",
    "Highest",
]

# =============================================================================
# Figure geometry
# =============================================================================
FIGSIZE = (4.95, 1.90)

AX_Y = 0.23
AX_H = 0.58
AX_W = 0.248
AX_LEFTS = [0.105, 0.395, 0.685]

# =============================================================================
# Style
# =============================================================================
# Matched to revised Fig. 3a for visual consistency.
BASE_FONT = 7.0
TICK_FONT = 6.3

# Slightly smaller than general text because x-axis labels are long.
AXIS_LABEL_FONT = 6.6

LEGEND_FONT = 6.0
LINE_WIDTH = 1.25
MARKER_SIZE = 3.2

GRID_COLOR = "#E7E7E7"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": BASE_FONT,

        "axes.linewidth": 0.68,

        # Preserve editable/vector text.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "axes.unicode_minus": False,

        # Improve raster rendering.
        "text.antialiased": True,
        "lines.antialiased": True,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def build_summary(
    df: pd.DataFrame,
    driver: str,
    value_col: str
) -> pd.DataFrame:

    rows = []

    for aridity in ARIDITY_ORDER:

        sub = df[
            df["aridity_class"] == aridity
        ].copy()

        sub["quantile"] = pd.qcut(
            sub[driver],
            5,
            labels=Q_LABELS,
            duplicates="drop"
        )

        g = (
            sub.groupby(
                "quantile",
                observed=False
            )[value_col]
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

    out = pd.concat(
        rows,
        ignore_index=True
    )

    out["x"] = out["quantile"].map(
        {
            q: i + 1
            for i, q in enumerate(Q_LABELS)
        }
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

    # =========================================================================
    # Load and summarise data
    # =========================================================================
    df = pd.read_csv(INPUT_CSV)

    plot_df = pd.concat(
        [
            build_summary(
                df,
                driver,
                "share_abs_co2"
            )
            for driver, _ in DRIVERS
        ],
        ignore_index=True
    )

    plot_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # =========================================================================
    # Create figure
    # =========================================================================
    fig = plt.figure(
        figsize=FIGSIZE,
        dpi=DPI,
        facecolor="white"
    )

    axes = [
        fig.add_axes(
            [left, AX_Y, AX_W, AX_H]
        )
        for left in AX_LEFTS
    ]

    # Slight horizontal adjustment of long x-axis labels.
    # This prevents overlap between "(SM90)" and "Net-drying".
    xlabel_xpos = [
        0.500,
        0.475,
        0.525,
    ]

    # =========================================================================
    # Plot panels
    # =========================================================================
    for i, (ax, (driver, xlabel)) in enumerate(
        zip(axes, DRIVERS)
    ):

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

                label=LABELS_MAP[aridity],
                color=ARIDITY_COLORS[aridity],

                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,

                zorder=3,
            )

        # ---------------------------------------------------------------------
        # Axes
        # ---------------------------------------------------------------------
        ax.set_facecolor("white")

        ax.set_xlim(
            0.8,
            5.2
        )

        ax.set_ylim(
            0.40,
            0.60
        )

        ax.set_xticks(
            range(1, 6)
        )

        ax.set_xticklabels(
            TREND_GROUP_LABELS,
            fontsize=TICK_FONT,
            color=AXIS_COLOR
        )

        ax.set_yticks(
            [
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
            ]
        )

        if i > 0:

            ax.tick_params(
                labelleft=False
            )

        else:

            ax.set_yticklabels(
                [
                    "0.40",
                    "0.45",
                    "0.50",
                    "0.55",
                    "0.60",
                ],
                fontsize=TICK_FONT,
                color=AXIS_COLOR
            )

        # ---------------------------------------------------------------------
        # X-axis labels
        # ---------------------------------------------------------------------
        ax.set_xlabel(
            xlabel,
            fontsize=AXIS_LABEL_FONT,
            color=TEXT_COLOR,
            labelpad=4
        )

        ax.xaxis.set_label_coords(
            xlabel_xpos[i],
            -0.175
        )

        # ---------------------------------------------------------------------
        # Grid
        # ---------------------------------------------------------------------
        ax.grid(
            axis="y",
            color=GRID_COLOR,
            lw=0.45,
            zorder=0
        )

        ax.set_axisbelow(True)

        # ---------------------------------------------------------------------
        # Spines
        # ---------------------------------------------------------------------
        for spine in [
            "top",
            "right"
        ]:
            ax.spines[
                spine
            ].set_visible(False)

        for spine in [
            "left",
            "bottom"
        ]:
            ax.spines[
                spine
            ].set_color(
                AXIS_COLOR
            )

            ax.spines[
                spine
            ].set_linewidth(
                0.68
            )

        # ---------------------------------------------------------------------
        # Ticks
        # ---------------------------------------------------------------------
        ax.tick_params(
            axis="both",
            which="major",
            length=2.5,
            width=0.68,
            labelsize=TICK_FONT,
            colors=AXIS_COLOR,
            pad=2
        )

    # =========================================================================
    # Y-axis label
    # =========================================================================
    axes[0].set_ylabel(
        r"Median XCO$_2$-background budget share",
        fontsize=AXIS_LABEL_FONT,
        color=TEXT_COLOR,
        labelpad=2
    )

    # =========================================================================
    # Legend
    # =========================================================================
    handles, labels = axes[
        0
    ].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,

        ncol=4,

        frameon=False,

        loc="upper center",
        bbox_to_anchor=(
            0.52,
            0.94
        ),

        fontsize=LEGEND_FONT,

        handlelength=1.45,
        columnspacing=0.9,
        handletextpad=0.35,

        borderaxespad=0,
    )

    # =========================================================================
    # Export
    # =========================================================================
    if EXPORT_TRANSPARENT:

        fig.patch.set_alpha(0)

        for ax in fig.axes:
            ax.set_facecolor("none")

        save_kwargs = {
            "transparent": True
        }

    else:

        fig.patch.set_facecolor(
            "white"
        )

        for ax in fig.axes:
            ax.set_facecolor(
                "white"
            )

        save_kwargs = {
            "transparent": False,
            "facecolor": "white",
        }

    # High-resolution raster
    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        **save_kwargs
    )

    # Vector versions
    fig.savefig(
        OUTPUT_PDF,
        **save_kwargs
    )

    fig.savefig(
        OUTPUT_SVG,
        **save_kwargs
    )

    plt.close(fig)

    # =========================================================================
    # Messages
    # =========================================================================
    print(
        f"Saved: {OUTPUT_PNG}"
    )

    print(
        f"Saved: {OUTPUT_PDF}"
    )

    print(
        f"Saved: {OUTPUT_SVG}"
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
