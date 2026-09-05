#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cross_phase_table_process_x_driverquant_UP.py

Input:
  02_hex_merged_process_attr_climate_UP.csv
    - must contain group_type
    - attribution columns: share_nat_co2, dominance_co2, share_abs_clim, share_abs_co2, share_abs_res, beta_obs
    - driver trend columns: P90_trend / SM90_L1_trend / VPD_resid_trend (any subset is ok)

Output (OUT_DIR):
  process_x_<DRIVER>_quantile_long.csv     (recommended for plotting)
  process_x_<DRIVER>_quantile_wide.xlsx    (wide pivots for heatmaps + long)
  runinfo_cross_phase.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG (edit paths only)
# =============================================================================
CONFIG = {
    "IN_MERGED_CSV": r"data/processed/process_context/process_context_attribution_hydroclimate_up.csv",
    "OUT_DIR": r"outputs/stratification/process_context",

    # process column
    "PROCESS_COL": "group_type",

    # drivers to stratify by (global quantiles across UP)
    "DRIVERS": [
        "P90_trend",
        "SM90_L1_trend",
        "VPD_resid_trend",
    ],

    # quantile bins
    "N_QUANTILES": 5,

    # minimum samples to keep a cell (else kept but flagged)
    "MIN_CELL_N": 50,

    # metrics to summarize in each cell
    "METRICS": [
        "beta_obs",
        "share_nat_co2",
        "share_abs_clim",
        "share_abs_co2",
        "share_abs_res",
        "offdiag_ratio",              # optional; if missing will be skipped
        "dominant_share_of_offdiag",  # optional; if missing will be skipped
    ],

    # dominance column (binary or bool). If missing, dominance_rate will be NA.
    "DOMINANCE_COL": "dominance_co2",
}

# =============================================================================
# helpers
# =============================================================================
def q25(x): return x.quantile(0.25)
def q75(x): return x.quantile(0.75)

def make_global_qbins(s: pd.Series, nq: int) -> pd.Series:
    """
    Global quantile bins (Q1..Qk) using pd.qcut.
    If duplicates occur (many ties), qcut may drop bins; we label accordingly.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    m = s_num.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    if m.sum() < nq * 10:
        # too few valid observations; still try with fewer bins
        nq_eff = max(2, min(nq, int(max(2, m.sum() // 10))))
    else:
        nq_eff = nq

    try:
        cats = pd.qcut(s_num[m], q=nq_eff, duplicates="drop")
    except ValueError:
        # all values equal or not enough unique values
        return out

    # map to Q1..Qk in sorted order
    cat_codes = cats.cat.codes
    k = int(cat_codes.max() + 1)
    labels = [f"Q{i}" for i in range(1, k + 1)]
    out.loc[m] = pd.Categorical.from_codes(cat_codes, categories=labels, ordered=True)
    return out

def summarize_cells(df: pd.DataFrame, process_col: str, qcol: str, metrics: list[str], dominance_col: str, min_n: int) -> pd.DataFrame:
    use_metrics = [c for c in metrics if c in df.columns]
    has_dom = dominance_col in df.columns

    g = df.groupby([process_col, qcol], dropna=False)

    rows = []
    for (p, q), sub in g:
        n = len(sub)
        row = {
            process_col: p,
            qcol: q,
            "n": n,
            "cell_ok": (n >= min_n)
        }
        if has_dom:
            # dominance can be bool, 0/1, or strings; coerce to numeric
            dom = pd.to_numeric(sub[dominance_col], errors="coerce")
            row["dominance_rate"] = float(dom.mean()) if dom.notna().any() else np.nan
        else:
            row["dominance_rate"] = np.nan

        for c in use_metrics:
            x = pd.to_numeric(sub[c], errors="coerce")
            row[f"{c}_median"] = float(x.median()) if x.notna().any() else np.nan
            row[f"{c}_p25"] = float(q25(x)) if x.notna().any() else np.nan
            row[f"{c}_p75"] = float(q75(x)) if x.notna().any() else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    # sort
    if qcol in out.columns:
        out = out.sort_values([process_col, qcol])
    return out

def write_wide_xlsx(long_df: pd.DataFrame, process_col: str, qcol: str, out_xlsx: Path) -> None:
    """
    Create wide pivots for quick heatmap plotting.
    """
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        long_df.to_excel(w, sheet_name="long", index=False)

        # make pivots for key fields if present
        candidates = [c for c in long_df.columns if c.endswith("_median")] + ["dominance_rate", "n"]
        for c in candidates:
            if c not in long_df.columns:
                continue
            piv = long_df.pivot(index=process_col, columns=qcol, values=c)
            piv.to_excel(w, sheet_name=f"wide_{c[:26]}", index=True)

# =============================================================================
# main
# =============================================================================
def main():
    in_path = Path(CONFIG["IN_MERGED_CSV"])
    out_dir = Path(CONFIG["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    proc = CONFIG["PROCESS_COL"]
    if proc not in df.columns:
        raise KeyError(f"Missing process column '{proc}' in {in_path.name}")

    # Ensure process order is stable (optional): keep as is
    df[proc] = df[proc].astype(str)

    drivers = CONFIG["DRIVERS"]
    nq = int(CONFIG["N_QUANTILES"])
    metrics = CONFIG["METRICS"]
    dom_col = CONFIG["DOMINANCE_COL"]
    min_n = int(CONFIG["MIN_CELL_N"])

    produced = []

    for drv in drivers:
        if drv not in df.columns:
            print(f"[skip] driver not found: {drv}")
            continue

        qcol = f"{drv}_qbin"
        df[qcol] = make_global_qbins(df[drv], nq=nq)

        # drop NA bins (no valid driver trend)
        df_sub = df[df[qcol].notna()].copy()
        if df_sub.empty:
            print(f"[skip] no valid values for driver: {drv}")
            continue

        long = summarize_cells(
            df_sub, process_col=proc, qcol=qcol,
            metrics=metrics, dominance_col=dom_col, min_n=min_n
        )

        # output files
        out_long = out_dir / f"process_x_{drv}_quantile_long.csv"
        out_xlsx = out_dir / f"process_x_{drv}_quantile_wide.xlsx"
        long.to_csv(out_long, index=False)
        write_wide_xlsx(long, process_col=proc, qcol=qcol, out_xlsx=out_xlsx)

        produced.append({"driver": drv, "long_csv": str(out_long), "xlsx": str(out_xlsx)})

    runinfo = {
        "input": str(in_path),
        "process_col": proc,
        "drivers_requested": drivers,
        "n_quantiles_target": nq,
        "dominance_col": dom_col,
        "metrics_requested": metrics,
        "min_cell_n": min_n,
        "outputs": produced,
        "note": "Quantile bins are computed globally across UP for each driver; bins may be fewer than target if ties force qcut to drop duplicates."
    }
    (out_dir / "runinfo_cross_phase.json").write_text(json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done. Outputs:")
    for item in produced:
        print("  -", item["long_csv"])
        print("  -", item["xlsx"])

if __name__ == "__main__":
    main()
