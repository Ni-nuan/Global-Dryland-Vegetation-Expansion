#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2e: distribution of naturally explained share under two attribution specifications.

Refined main-text style:
- Fixed canvas and axes position shared with Fig. 2d/f.
- Softer boxplot styling consistent with Fig. 1.
- Slightly larger fonts than the original version.
- No bbox_inches='tight', preserving identical panel dimensions.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# User-defined paths
# =========================
INPUT_XLSX = "trend_outputs_xco2_vpd_resid/06_trend_attribution_by_hex_UP.xlsx"
OUTPUT_PNG = "fig2d_natural_share_main_revised_refined.png"
OUTPUT_PDF = "fig2d_natural_share_main_revised_refined.pdf"
OUTPUT_SVG = "fig2d_natural_share_main_revised_refined.svg"
OUTPUT_CSV = "fig2d_natural_share_main_revised_refined.csv"

# If transparent PNGs are displayed on a black viewer background, text can look abnormal.
# Keep True for manuscript assembly on white; set False only for quick visual preview.
EXPORT_TRANSPARENT = True

# =========================
# Shared panel style
# =========================
FIGSIZE = (3.70, 2.65)
DPI = 600
AX_POS = [0.18, 0.24, 0.76, 0.64]

FONT_FAMILY = "Arial"
NEUTRAL = "#D9D9D9"
XCO2_BLUE = "#9ECAE1"
AXIS_COLOR = "#333333"
GRID_COLOR = "#E7E7E7"
TEXT_COLOR = "#222222"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.75,
    "axes.unicode_minus": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": FONT_FAMILY + ":italic",
    "mathtext.bf": FONT_FAMILY + ":bold",
})

# =========================
# Read and prepare data
# =========================
orig = pd.read_excel(INPUT_XLSX)
cols = list(orig.columns)

c1 = "share_nat_clim" if "share_nat_clim" in cols else [
    c for c in cols if "share" in c.lower() and "clim" in c.lower()
][0]
c2 = "share_nat_co2" if "share_nat_co2" in cols else [
    c for c in cols if "share" in c.lower() and "co2" in c.lower()
][0]

df = pd.DataFrame({
    "model": ["CLIM_ONLY"] * orig[c1].notna().sum()
             + ["CLIM_PLUS_XCO2"] * orig[c2].notna().sum(),
    "natural_share": pd.concat([orig[c1].dropna(), orig[c2].dropna()], ignore_index=True),
})

order = ["CLIM_ONLY", "CLIM_PLUS_XCO2"]
data = [df.loc[df["model"] == m, "natural_share"].dropna().to_numpy(dtype=float) for m in order]
medians = [pd.Series(d).median() for d in data]

df.to_csv(OUTPUT_CSV, index=False)

# =========================
# Plot
# =========================
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
ax = fig.add_axes(AX_POS)

bp = ax.boxplot(
    data,
    positions=[0, 1],
    patch_artist=True,
    widths=0.50,
    showfliers=False,
    medianprops=dict(linewidth=1.15, color=AXIS_COLOR),
    whiskerprops=dict(linewidth=0.85, color=AXIS_COLOR),
    capprops=dict(linewidth=0.85, color=AXIS_COLOR),
    boxprops=dict(linewidth=0.85, color=AXIS_COLOR),
)

for patch, fc in zip(bp["boxes"], [NEUTRAL, XCO2_BLUE]):
    patch.set_facecolor(fc)
    patch.set_edgecolor("none")
    patch.set_alpha(0.95)

# Re-draw box outlines lightly after removing patch edges.
for box in bp["boxes"]:
    box.set_edgecolor(AXIS_COLOR)
    box.set_linewidth(0.85)

for xi, med in zip([0, 1], medians):
    ax.text(
        xi,
        med + 0.030,
        f"{med:.4f}",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=TEXT_COLOR,
    )

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, 1.02)
ax.set_xticks([0, 1])
ax.set_xticklabels(["CLIM_ONLY", "CLIM_PLUS_XCO$_2$"], fontsize=8.2, color=TEXT_COLOR)
ax.set_ylabel("Naturally explained share", fontsize=9.2, color=TEXT_COLOR, labelpad=4)

ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=8.2, color=TEXT_COLOR)

ax.text(
    0.02,
    0.98,
    "UP sample, NAT0",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7.8,
    color=TEXT_COLOR,
)

ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=1)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(AXIS_COLOR)
    ax.spines[spine].set_linewidth(0.75)

ax.tick_params(axis="x", length=0, pad=6, colors=TEXT_COLOR)
ax.tick_params(axis="y", length=3.2, width=0.75, colors=TEXT_COLOR, pad=2)

# =========================
# Export
# =========================
# PNG is kept as a transparent-background preview/export.
# PDF and SVG are the main vector outputs for manuscript assembly/editing.
# svg.fonttype='none' keeps text editable in vector editors when the font is available.
if EXPORT_TRANSPARENT:
    fig.patch.set_alpha(0)
    for a in fig.axes:
        a.set_facecolor("none")

save_kwargs = dict(transparent=EXPORT_TRANSPARENT)

fig.savefig(OUTPUT_PNG, dpi=DPI, **save_kwargs)
fig.savefig(OUTPUT_PDF, **save_kwargs)
fig.savefig(OUTPUT_SVG, **save_kwargs)
plt.close(fig)
