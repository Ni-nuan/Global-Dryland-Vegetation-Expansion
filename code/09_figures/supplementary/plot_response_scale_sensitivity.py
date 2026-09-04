#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined export script for the logit-vs-fraction attribution comparison panel.

Updates relative to the original version:
- style aligned with the revised Fig. 1 / Fig. 2 panels;
- slightly larger, cleaner typography;
- transparent PNG plus vector PDF and SVG outputs;
- editable text in SVG/PDF;
- robust path handling based on script location.
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# User-defined paths
# =============================================================================
# Keep paths relative to the current working directory, consistent with the
# earlier scripts. Run the script from your project folder.
JSON_LOGIT = Path(r'trend_outputs_xco2_vpd_resid/07_trend_attribution_summary_UP.json')
JSON_FRAC = Path(r'trend_outputs_xco2_vpd_resid/07_trend_attribution_summary_frac_UP.json')

OUT_PNG = Path(r'fig2_logit_vs_frac_comparison_vpd_resid.png')
OUT_PDF = Path(r'fig2_logit_vs_frac_comparison_vpd_resid.pdf')
OUT_SVG = Path(r'fig2_logit_vs_frac_comparison_vpd_resid.svg')
OUT_CSV = Path(r'fig2_logit_vs_frac_comparison_vpd_resid.csv')


# =============================================================================
# Styling
# =============================================================================
CLIM_ONLY_COLOR = '#D9D9D9'
CLIM_PLUS_COLOR = '#8DB7D8'
EDGE_COLOR = '#4A4A4A'
TEXT_COLOR = '#222222'

FIGSIZE = (7.8, 3.6)
DPI = 300


# =============================================================================
# Helpers
# =============================================================================
def apply_rcparams() -> None:
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.size': 9.5,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.0,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    })


def load_summary() -> pd.DataFrame:
    with open(JSON_LOGIT, 'r', encoding='utf-8') as f:
        s_logit = json.load(f)
    with open(JSON_FRAC, 'r', encoding='utf-8') as f:
        s_frac = json.load(f)

    df = pd.DataFrame({
        'metric': ['Natural share', 'Natural-dominant fraction'],
        'logit_CLIM_ONLY': [
            float(s_logit['CLIM_ONLY']['share_nat_median']),
            float(s_logit['CLIM_ONLY']['natural_dominant_pct'])
        ],
        'logit_CLIM_PLUS_XCO2': [
            float(s_logit['CLIM_PLUS_XCO2']['share_nat_median']),
            float(s_logit['CLIM_PLUS_XCO2']['natural_dominant_pct'])
        ],
        'frac_CLIM_ONLY': [
            float(s_frac['CLIM_ONLY']['share_nat_median']),
            float(s_frac['CLIM_ONLY']['natural_dominant_pct'])
        ],
        'frac_CLIM_PLUS_XCO2': [
            float(s_frac['CLIM_PLUS_XCO2']['share_nat_median']),
            float(s_frac['CLIM_PLUS_XCO2']['natural_dominant_pct'])
        ],
    })
    return df


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    apply_rcparams()
    plot_df = load_summary()
    plot_df.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)

    group_x = [0, 1]
    group_labels = ['logit', 'frac']
    barw = 0.33
    offsets = [-barw / 2, barw / 2]
    fill_colors = [CLIM_ONLY_COLOR, CLIM_PLUS_COLOR]
    legend_labels = ['CLIM_ONLY', 'CLIM_PLUS_XCO$_2$']

    for ax, title in zip(axes, ['Natural share', 'Natural-dominant fraction']):
        ax.set_facecolor('none')
        row = plot_df.loc[plot_df['metric'] == title].iloc[0]
        vals = [
            [row['logit_CLIM_ONLY'], row['logit_CLIM_PLUS_XCO2']],
            [row['frac_CLIM_ONLY'], row['frac_CLIM_PLUS_XCO2']]
        ]

        for j, x0 in enumerate(group_x):
            for i in range(2):
                y = float(vals[j][i])
                ax.bar(
                    x0 + offsets[i],
                    y,
                    width=barw,
                    color=fill_colors[i],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.7,
                    zorder=3,
                )
                ax.text(
                    x0 + offsets[i],
                    min(y + 0.03, 0.99),
                    f'{y:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=8.3,
                    color=TEXT_COLOR,
                )

        ax.set_title(title, pad=7, color=TEXT_COLOR)
        ax.set_xticks(group_x)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel('Value' if ax is axes[0] else '')
        ax.tick_params(axis='both', colors=TEXT_COLOR, width=0.7, length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(EDGE_COLOR)
        ax.spines['bottom'].set_color(EDGE_COLOR)
        ax.grid(axis='y', color='#E9E9E9', linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=fill_colors[0], edgecolor=EDGE_COLOR, linewidth=0.7, label=legend_labels[0]),
        Patch(facecolor=fill_colors[1], edgecolor=EDGE_COLOR, linewidth=0.7, label=legend_labels[1]),
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.8,
    )

    plt.tight_layout(rect=[0.02, 0.00, 0.98, 0.92])

    fig.savefig(OUT_PNG, dpi=DPI, transparent=True, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PDF, transparent=True, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_SVG, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == '__main__':
    main()
