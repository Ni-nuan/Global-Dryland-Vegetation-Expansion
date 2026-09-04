#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_typing_landtransition_v2_up_fiveclass_plus_flag.py

Process typing for UP hexes using 2000->2022 land-transition matrix:
- primary_type: FIVE mutually exclusive pathway families (based on dominant off-diagonal flow)
    1) Ag_expansion      : to cropland(1) and from != 1
    2) Urban_expansion   : to urban(8) and from != 8
    3) Bare_to_Sparse    : 9 -> 6
    4) Bare_or_sparse_to_grass_or_forest: dominant transition from {6,9} to {2,3,4,5,7}
    5) Other             : all remaining dominant transitions (incl. degradation/backslide etc.)
- flag_no_transition: offdiag_total == 0 (no explicit conversion)
- group_type: if flag_no_transition -> "No_transition" else primary_type

Then merge with:
- UP attribution output (logit main): 06_trend_attribution_by_hex_UP.xlsx
- climate trends computed from DIR_UP wide xlsx (P90/VPD90/Tmean/SM90_L1/ET90/SWnet90 + derived VPD_resid trend)

Outputs (OUT_DIR):
  01_hex_process_labels_UP.csv
  02_hex_merged_process_attr_climate_UP.csv
  10_group_summary_by_group_type_UP.xlsx
  11_group_summary_by_group_type_UP.csv
  runinfo.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG (EDIT THESE PATHS)
# =============================================================================
CONFIG = {
    "OUT_DIR": r"outputs/process_context",

    # UP 2000->2022 land transition matrix (.xls)
    "CHANGE_XLS": r"NDVI_trend_hex_100_up_change.xls",

    # UP attribution output (logit main)
    "ATTR_UP_XLSX": r"trend_outputs_xco2_vpd_resid/06_trend_attribution_by_hex_UP.xlsx",

    # Climate wide tables directory (same naming as your pipeline)
    "DIR_UP": r"zonal_tables_parallel_up_V2",
    "FILE_P90": r"W90_Precip.xlsx",
    "FILE_VPD90": r"VPD90_mean.xlsx",
    "FILE_TMEAN": r"T2m90_mean.xlsx",
    "FILE_SM_L1": r"SM_L1_90_mean.xlsx",
    "FILE_ET90": r"ET90_sum.xlsx",            # optional
    "FILE_SWNET90": r"SWnet90_mean.xlsx",     # optional

    "YEAR_MIN": 2000,
    "YEAR_MAX": 2022,

    # VPD_resid settings: pooled OLS in hex-year panel
    "VPD_RESID_INCLUDE_SWNET": True,
    "VPD_RESID_DOUBLE_DEMEAN": True,

    # If your change matrix uses a different id column name, set it here
    # (leave None to auto-detect: HEX_ID / hex_id / first column)
    "HEX_ID_COL": None,
}

# Land-cover class names (1..10)
CLASS_NAMES = {
    1: "cropland",
    2: "forest",
    3: "shrubland",
    4: "grassland",
    5: "lichen_and_moss",
    6: "sparse_vegetation",
    7: "wetland",
    8: "urban",
    9: "bare_areas",
    10: "water_and_ice"
}

# Conservative "natural recovery" target classes (more vegetated than sparse/bare)
VEGETATED_TRANSITION_TO = {2, 3, 4, 5, 7}
BARE_SPARSE_TRANSITION_FROM = {6, 9}

# =============================================================================
# IO helpers
# =============================================================================
def require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

def safe_read_xls(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError as e:
        raise ImportError(
            "Reading .xls requires xlrd>=2.0.1. Install:\n"
            "  pip install xlrd==2.0.1\n"
            f"Original error: {e}"
        )

# =============================================================================
# Transition parsing & typing
# =============================================================================
def parse_code(code_int: int) -> Tuple[int, int]:
    """
    Parse integer code like 102 (1->2) or 1003 (10->3).
    Assumes:
      - 1..9 -> represented in hundreds: fr*100 + to
      - 10 -> represented as 1000 + to
    """
    if code_int >= 1000:
        fr = 10
        to = code_int - 1000
    else:
        fr = code_int // 100
        to = code_int % 100
    return fr, to

def classify_primary(fr: int, to: int) -> str:
    """
    FIVE mutually exclusive classes based on dominant off-diagonal transition.
    """
    if (to == 1) and (fr != 1):
        return "Ag_expansion"
    if (to == 8) and (fr != 8):
        return "Urban_expansion"
    if (fr, to) == (9, 6):
        return "Bare_to_Sparse"
    # bare/sparse-to-grass/forest transition
    if (fr in BARE_SPARSE_TRANSITION_FROM) and (to in VEGETATED_TRANSITION_TO):
        return "Bare_or_sparse_to_grass_or_forest"
    return "Other"

# =============================================================================
# Wide xlsx -> long; trends; VPD_resid
# =============================================================================
def extract_year(col: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", str(col))
    return int(m.group(0)) if m else None

def wide_to_long(path: Path, value_name: str, y0: int, y1: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "hex_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "hex_id"})
    df["hex_id"] = df["hex_id"].astype(str)

    col2year: Dict[str, int] = {}
    for c in df.columns:
        if c == "hex_id":
            continue
        y = extract_year(c)
        if y is not None and y0 <= y <= y1:
            col2year[c] = y
    if not col2year:
        raise ValueError(f"No year columns detected in: {path}")

    keep = ["hex_id"] + list(col2year.keys())
    df = df[keep].copy()
    out = df.melt(id_vars=["hex_id"], var_name="col", value_name=value_name)
    out["year"] = out["col"].map(col2year).astype(int)
    out = out.drop(columns=["col"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out

def trend_per_hex(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = long_df.groupby("hex_id", sort=False)

    def _slope(sub: pd.DataFrame) -> float:
        y = sub[value_col].to_numpy(float)
        x = sub["year"].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 2:
            return np.nan
        xx = x[m]
        yy = y[m]
        vx = np.var(xx, ddof=0)
        if vx == 0:
            return np.nan
        cov = np.mean((xx - xx.mean()) * (yy - yy.mean()))
        return cov / vx

    out = pd.DataFrame({
        "hex_id": g.size().index.astype(str),
        f"{value_col}_trend": g.apply(_slope).values
    })
    return out

def double_demean(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = panel.copy()
    overall = {c: out[c].mean() for c in cols}
    by_hex = out.groupby("hex_id")[cols].transform("mean")
    by_year = out.groupby("year")[cols].transform("mean")
    for c in cols:
        out[c] = out[c] - by_hex[c] - by_year[c] + overall[c]
    return out

def pooled_vpd_resid(panel: pd.DataFrame, include_swnet: bool, do_double_demean: bool) -> pd.Series:
    need = ["VPD90", "Tmean"] + (["SWnet90"] if include_swnet and "SWnet90" in panel.columns else [])
    df = panel[["hex_id", "year"] + need].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.Series(np.nan, index=panel.index, name="VPD_resid")

    if do_double_demean:
        df = double_demean(df, need)

    y = df["VPD90"].to_numpy(float)
    X_cols = ["Tmean"] + (["SWnet90"] if ("SWnet90" in df.columns and include_swnet) else [])
    X = df[X_cols].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), X])

    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (X.T @ y)

    resid = y - (X @ beta)
    key = df["hex_id"].astype(str) + "__" + df["year"].astype(str)
    resid_map = pd.Series(resid, index=key)

    full_key = panel["hex_id"].astype(str) + "__" + panel["year"].astype(str)
    return full_key.map(resid_map).rename("VPD_resid")

# =============================================================================
# Attribution: three components
# =============================================================================
def add_three_components(attr: pd.DataFrame) -> pd.DataFrame:
    out = attr.copy()
    out["beta_clim"] = out["beta_nat_clim"]
    out["beta_co2"] = out["beta_nat_co2"] - out["beta_nat_clim"]
    out["beta_res"] = out["beta_res_co2"]

    denom = (out["beta_clim"].abs() + out["beta_co2"].abs() + out["beta_res"].abs()).replace(0, np.nan)
    out["share_abs_clim"] = out["beta_clim"].abs() / denom
    out["share_abs_co2"] = out["beta_co2"].abs() / denom
    out["share_abs_res"] = out["beta_res"].abs() / denom
    return out

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    out_dir = Path(CONFIG["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Read change matrix ----------
    change_path = Path(CONFIG["CHANGE_XLS"])
    require_exists(change_path)
    df = safe_read_xls(change_path)

    # hex_id standardization
    hex_col = CONFIG.get("HEX_ID_COL")
    if hex_col and hex_col in df.columns:
        df = df.rename(columns={hex_col: "hex_id"})
    else:
        if "HEX_ID" in df.columns:
            df = df.rename(columns={"HEX_ID": "hex_id"})
        elif "hex_id" not in df.columns:
            df = df.rename(columns={df.columns[0]: "hex_id"})
    df["hex_id"] = df["hex_id"].astype(str)

    value_cols = [c for c in df.columns if str(c).startswith("VALUE_")]
    if not value_cols:
        raise ValueError("No VALUE_* columns found in the change matrix.")

    # parse transition codes for each VALUE_ column
    codes = [int(str(c).split("_", 1)[1]) for c in value_cols]
    fr_to = {code: parse_code(code) for code in codes}

    mat = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["total_area"] = mat.sum(axis=1)

    # off-diagonal totals (exclude diagonal 1->1,2->2,...)
    offdiag_cols = []
    diag_cols = []
    for c in value_cols:
        code = int(str(c).split("_", 1)[1])
        fr, to = fr_to[code]
        if fr == to:
            diag_cols.append(c)
        else:
            offdiag_cols.append(c)

    df["diag_total"] = mat[diag_cols].sum(axis=1) if diag_cols else 0.0
    df["offdiag_total"] = mat[offdiag_cols].sum(axis=1) if offdiag_cols else 0.0
    df["offdiag_ratio"] = df["offdiag_total"] / df["total_area"].replace(0, np.nan)

    df["flag_no_transition"] = (df["offdiag_total"] == 0)

    # dominant off-diagonal flow (only meaningful when offdiag_total>0)
    if offdiag_cols:
        offdiag_mat = mat[offdiag_cols].to_numpy()
        dom_idx = offdiag_mat.argmax(axis=1)
        dom_cols = np.array(offdiag_cols)[dom_idx]
        dom_vals = offdiag_mat[np.arange(len(df)), dom_idx]
    else:
        dom_cols = np.array([None] * len(df))
        dom_vals = np.array([0.0] * len(df))

    df["dominant_value_col"] = dom_cols
    df["dominant_area"] = dom_vals
    df["dominant_share_of_offdiag"] = df["dominant_area"] / df["offdiag_total"].replace(0, np.nan)

    # parse dom from/to
    dom_from = []
    dom_to = []
    for col in df["dominant_value_col"].tolist():
        if (col is None) or (pd.isna(col)):
            dom_from.append(np.nan)
            dom_to.append(np.nan)
            continue
        code = int(str(col).replace("VALUE_", ""))
        fr, to = parse_code(code)
        dom_from.append(fr)
        dom_to.append(to)
    df["dom_from"] = dom_from
    df["dom_to"] = dom_to
    df["dom_from_name"] = df["dom_from"].map(CLASS_NAMES)
    df["dom_to_name"] = df["dom_to"].map(CLASS_NAMES)

    # primary type (five classes) for has-transition; keep NA for no-transition
    primary = []
    for nt, fr, to in zip(df["flag_no_transition"].tolist(),
                          df["dom_from"].tolist(),
                          df["dom_to"].tolist()):
        if nt:
            primary.append(pd.NA)
        else:
            primary.append(classify_primary(int(fr), int(to)))
    df["primary_type"] = primary

    # group type: treat No_transition as its own group label for summaries
    df["group_type"] = np.where(df["flag_no_transition"], "No_transition", df["primary_type"].astype(str))

    labels = df[[
        "hex_id",
        "flag_no_transition",
        "primary_type",
        "group_type",
        "total_area", "diag_total", "offdiag_total", "offdiag_ratio",
        "dominant_value_col", "dominant_area", "dominant_share_of_offdiag",
        "dom_from", "dom_to", "dom_from_name", "dom_to_name"
    ]].copy()
    labels.to_csv(out_dir / "01_hex_process_labels_UP.csv", index=False)

    # ---------- 2) Read attribution & add three components ----------
    attr_path = Path(CONFIG["ATTR_UP_XLSX"])
    require_exists(attr_path)
    attr = pd.read_excel(attr_path)
    if "hex_id" not in attr.columns:
        attr = attr.rename(columns={attr.columns[0]: "hex_id"})
    attr["hex_id"] = attr["hex_id"].astype(str)

    need = [
        "beta_obs",
        "beta_nat_clim", "beta_res_clim", "share_nat_clim", "dominance_clim",
        "beta_nat_co2", "beta_res_co2", "share_nat_co2", "dominance_co2",
    ]
    miss = [c for c in need if c not in attr.columns]
    if miss:
        raise KeyError("Attribution table missing required columns:\n" + "\n".join(miss))

    attr = add_three_components(attr)

    # ---------- 3) Climate trends + derived VPD_resid trend ----------
    dir_up = Path(CONFIG["DIR_UP"])
    require_exists(dir_up)
    y0, y1 = int(CONFIG["YEAR_MIN"]), int(CONFIG["YEAR_MAX"])

    def maybe_long(fname: str, vname: str) -> Optional[pd.DataFrame]:
        p = dir_up / fname
        if not p.exists():
            return None
        return wide_to_long(p, vname, y0, y1)

    p90 = maybe_long(CONFIG["FILE_P90"], "P90")
    vpd90 = maybe_long(CONFIG["FILE_VPD90"], "VPD90")
    tmean = maybe_long(CONFIG["FILE_TMEAN"], "Tmean")
    sm = maybe_long(CONFIG["FILE_SM_L1"], "SM90_L1")
    et = maybe_long(CONFIG["FILE_ET90"], "ET90") if CONFIG.get("FILE_ET90") else None
    swnet = maybe_long(CONFIG["FILE_SWNET90"], "SWnet90") if CONFIG.get("FILE_SWNET90") else None

    trend_tables = []
    if p90 is not None:   trend_tables.append(trend_per_hex(p90, "P90"))
    if vpd90 is not None: trend_tables.append(trend_per_hex(vpd90, "VPD90"))
    if tmean is not None: trend_tables.append(trend_per_hex(tmean, "Tmean"))
    if sm is not None:    trend_tables.append(trend_per_hex(sm, "SM90_L1"))
    if et is not None:    trend_tables.append(trend_per_hex(et, "ET90"))
    if swnet is not None: trend_tables.append(trend_per_hex(swnet, "SWnet90"))

    # VPD_resid trend
    vpd_resid_trend = None
    if (vpd90 is not None) and (tmean is not None):
        panel = vpd90.merge(tmean, on=["hex_id", "year"], how="inner")
        if CONFIG["VPD_RESID_INCLUDE_SWNET"] and (swnet is not None):
            panel = panel.merge(swnet, on=["hex_id", "year"], how="left")
        panel["VPD_resid"] = pooled_vpd_resid(
            panel=panel,
            include_swnet=bool(CONFIG["VPD_RESID_INCLUDE_SWNET"]),
            do_double_demean=bool(CONFIG["VPD_RESID_DOUBLE_DEMEAN"])
        )
        vpd_resid_trend = trend_per_hex(panel[["hex_id", "year", "VPD_resid"]], "VPD_resid")

    # merge trend tables
    trends = None
    for t in trend_tables:
        trends = t if trends is None else trends.merge(t, on="hex_id", how="outer")
    if vpd_resid_trend is not None:
        trends = vpd_resid_trend if trends is None else trends.merge(vpd_resid_trend, on="hex_id", how="outer")
    if trends is None:
        trends = pd.DataFrame({"hex_id": attr["hex_id"].unique()})

    # ---------- 4) Merge all ----------
    merged = labels.merge(attr, on="hex_id", how="left").merge(trends, on="hex_id", how="left")
    merged.to_csv(out_dir / "02_hex_merged_process_attr_climate_UP.csv", index=False)

    # ---------- 5) Group summaries (by group_type) ----------
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    summary_vars = [
        # attribution
        "beta_obs",
        "share_nat_clim", "share_nat_co2",
        "share_abs_clim", "share_abs_co2", "share_abs_res",
        "beta_clim", "beta_co2", "beta_res",
        # conversion diagnostics
        "offdiag_total", "offdiag_ratio", "dominant_share_of_offdiag",
        # climate trends
        "P90_trend", "SM90_L1_trend", "VPD90_trend", "VPD_resid_trend", "Tmean_trend", "SWnet90_trend", "ET90_trend"
    ]
    summary_vars = [c for c in summary_vars if c in merged.columns]

    g = merged.groupby("group_type", dropna=False)
    summ = g.agg(
        n_hex=("hex_id", "size"),
        frac_no_transition=("flag_no_transition", "mean"),
        offdiag_total_median=("offdiag_total", "median"),
        offdiag_ratio_median=("offdiag_ratio", "median"),
        dominant_share_offdiag_median=("dominant_share_of_offdiag", "median"),
    ).reset_index()

    for c in summary_vars:
        summ[f"{c}_median"] = g[c].median().values
        summ[f"{c}_p25"] = g[c].apply(q25).values
        summ[f"{c}_p75"] = g[c].apply(q75).values

    summ = summ.sort_values("n_hex", ascending=False)
    summ.to_csv(out_dir / "11_group_summary_by_group_type_UP.csv", index=False)

    # Excel workbook (avoid huge sheets)
    with pd.ExcelWriter(out_dir / "10_group_summary_by_group_type_UP.xlsx", engine="openpyxl") as w:
        labels.to_excel(w, sheet_name="hex_labels", index=False)
        merged.head(20000).to_excel(w, sheet_name="hex_merged_sample", index=False)
        summ.to_excel(w, sheet_name="group_summary", index=False)

    runinfo = {
        "five_primary_types": ["Ag_expansion", "Urban_expansion", "Bare_to_Sparse", "Bare_or_sparse_to_grass_or_forest", "Other"],
        "no_transition_flag": "flag_no_transition (offdiag_total==0)",
        "grouping": "group_type = No_transition if flag else primary_type",
        "bare_sparse_to_grass_forest_definition": {
            "from": sorted(list(BARE_SPARSE_TRANSITION_FROM)),
            "to": sorted(list(VEGETATED_TRANSITION_TO)),
            "note": "semantic rename only; class membership is unchanged from the legacy label"
        },
        "vpd_resid": {
            "include_swnet": bool(CONFIG["VPD_RESID_INCLUDE_SWNET"]),
            "double_demean": bool(CONFIG["VPD_RESID_DOUBLE_DEMEAN"]),
            "model": "pooled OLS: VPD90 ~ Tmean (+SWnet90)"
        }
    }
    (out_dir / "runinfo.json").write_text(json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Outputs -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()
