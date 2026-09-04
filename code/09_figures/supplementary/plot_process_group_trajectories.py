#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User-defined paths
# =========================
VEG_CSV = Path(r"hex_data/NDVI_trend_hex_100_UP_valid_sens_gt0.csv")
LABEL_CSV = Path(r"outputs/process_context/01_hex_process_labels_UP.csv")
OUTPUT_PNG = Path(r"Fig1_UP_group_mean_trajectories.png")
OUTPUT_CSV = Path(r"Fig1_UP_group_mean_trajectories.csv")

# =========================
# Style
# =========================
FIGSIZE = (5.2, 3.2)
DPI = 600
FONT_FAMILY = "Arial"

GROUP_ORDER = [
    "Bare_to_Sparse",
    "Bare_or_sparse_to_grass_or_forest",
    "No_transition",
    "Ag_expansion",
    "Urban_expansion",
    "Other",
]

def main():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })

    veg = pd.read_csv(VEG_CSV)
    lab = pd.read_csv(LABEL_CSV)

    if "group_type" in lab.columns:
        lab["group_type"] = lab["group_type"].replace({
            "Nat_recovery": "Bare_or_sparse_to_grass_or_forest",
            "Veg_cover_upgrade": "Bare_or_sparse_to_grass_or_forest"
        })

    df = veg.merge(lab[["hex_id", "group_type"]], on="hex_id", how="left")

    year_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"veg_\d{4}", c)],
        key=lambda x: int(x.split("_")[1])
    )
    years = np.array([int(c.split("_")[1]) for c in year_cols], dtype=int)

    existing_groups = [g for g in GROUP_ORDER if g in df["group_type"].dropna().unique()]
    other_groups = [g for g in df["group_type"].dropna().unique() if g not in existing_groups]
    final_groups = existing_groups + sorted(other_groups)

    result_rows = []
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for group in final_groups:
        sub = df[df["group_type"] == group].copy()
        if len(sub) == 0:
            continue

        y = sub[year_cols].mean(axis=0, skipna=True).to_numpy(dtype=float)
        ax.plot(years, y, linewidth=1.8, label=group)

        for yr, val in zip(years, y):
            result_rows.append({
                "group_type": group,
                "year": int(yr),
                "veg_frac_mean": float(val),
                "n_hex": int(len(sub))
            })

    out = pd.DataFrame(result_rows)
    out.to_csv(OUTPUT_CSV, index=False)

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean veg_frac")
    ax.set_ylim(0, 1)
    ax.set_xlim(years.min(), years.max())

    tick_years = years[::2]
    ax.set_xticks(tick_years)
    ax.set_xticklabels(tick_years.astype(int))

    # 关键修改：legend 放图外上方
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.6,
        columnspacing=1.2,
        borderaxespad=0.0
    )

    fig.tight_layout(rect=[0, 0, 1, 0.88], pad=0.6)
    fig.savefig(OUTPUT_PNG, dpi=DPI, transparent=True)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()