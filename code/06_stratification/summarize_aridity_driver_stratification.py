#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
summarize_aridity_driver_stratification.py

UP-only. Uses the canonical panel-attribution outputs and the hydroclimate/SEM variable system.

What this script does
---------------------
1) Read UP attribution output (logit main): 06_trend_attribution_by_hex_UP.xlsx
2) Assign aridity class by centroid-in-polygon (UNEP-WCMC 4 classes).
3) Build candidate driver time series from DIR_UP climate xlsx files (wide: hex_id + year columns):
   P90, VPD90, Tmean, SM90 (L1/root), ET90, SWnet90, plus derived VPD_resid.
4) For each driver, compute per-hex metrics:
   bg_mean, bg_p90, trend(OLS slope vs year)
5) 1D stratification: aridity
6) 2D stratification: aridity × driver_metric_bin (quantile bins, default 5; within-aridity)
7) Auto-screening: rank (driver, metric) by gradients / monotonicity / min bin n.

Outputs
-------
OUT_DIR/
  hex_to_aridity_UP.csv
  stratified_UP_aridity_1d_clim.csv
  stratified_UP_aridity_1d_co2.csv
  driver_metrics_UP/<DRIVER>_metrics.csv
  stratified_UP_aridity_x_<DRIVER>__<METRIC>_clim.csv
  stratified_UP_aridity_x_<DRIVER>__<METRIC>_co2.csv
  screening_rank_UP.csv
  aridity_multi_driver_tables_UP.xlsx
  runinfo.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

# -------------------------
# CONFIG (EDIT THESE PATHS)
# -------------------------
CONFIG = {
    # Output directory
    "OUT_DIR": r"outputs/aridity_driver_stratification",

    # UP attribution output (logit main)
    "ATTR_UP_XLSX": r"trend_outputs_xco2_vpd_resid/06_trend_attribution_by_hex_UP.xlsx",

    # UP hex shapefile for aridity assignment (must contain HEX_ID_FIELD)
    "HEX_UP_SHP": r"hex_shp/NDVI_trend_hex_100_up.shp",
    "HEX_ID_FIELD": "hex_id",

    # UNEP-WCMC aridity polygons
    "ARIDITY_HYPERARID": r"Drylands_dataset_fixed/drylands_Hyperarid.shp",
    "ARIDITY_ARID": r"Drylands_dataset_fixed/drylands_arid.shp",
    "ARIDITY_SEMIARID": r"Drylands_dataset_fixed/drylands_Semiarid.shp",
    "ARIDITY_DRYSUBHUMID": r"Drylands_dataset_fixed/drylands_Dry_subhumid.shp",

    # Hydroclimate tables directory + fixed filenames (aligned with the main attribution/SEM workflow)
    "DIR_UP": r"zonal_tables_parallel_up_V2",

    "FILE_P90": r"W90_Precip.xlsx",          # -> P90
    "FILE_VPD90": r"VPD90_mean.xlsx",        # -> VPD90
    "FILE_TMEAN": r"T2m90_mean.xlsx",        # -> Tmean (thermal condition proxy)
    "FILE_SM_L1": r"SM_L1_90_mean.xlsx",     # -> SM90_L1
    "FILE_SM_ROOT": r"SM_root_90_mean.xlsx", # -> SM90_root (if exists)
    "FILE_ET90": r"ET90_sum.xlsx",           # -> ET90 (optional)
    "FILE_SWNET90": r"SWnet90_mean.xlsx",    # -> SWnet90 (optional)

    # Driver set: choose which candidate drivers to stratify
    # Supported keys:
    #   "P90","VPD90","Tmean","SM90_L1","SM90_root","ET90","SWnet90","VPD_resid"
    "DRIVER_KEYS": ["VPD90", "P90", "SM90_L1", "SWnet90", "Tmean", "VPD_resid"],

    # Derived VPD_resid: pooled OLS in the UP hex-year panel space:
    #   VPD90 ~ Tmean (+ SWnet90), residual = VPD_resid
    # Optionally apply double-demean (hex + year) before regression (closer to SEM anomaly space).
    "VPD_RESID_INCLUDE_SWNET": True,
    "VPD_RESID_DOUBLE_DEMEAN": True,

    # Year range (for detecting columns & trend slopes)
    "YEAR_MIN": 2000,
    "YEAR_MAX": 2022,

    # Metrics & binning
    "METRICS": ["bg_mean", "bg_p90", "trend"],
    "BINS": 5,
    "WITHIN_ARIDITY": True,
    "MIN_BINS_REQUIRED": 3,
}

ARIDITY_PRIORITY = ["Hyperarid", "Arid", "Semiarid", "Dry_subhumid"]


# -------------------------
# Helpers: geometry
# -------------------------
def read_layer(path: Path, assumed_epsg: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_epsg, allow_override=True)
    return gdf

def ensure_spatial_index(gdf: gpd.GeoDataFrame, name: str) -> None:
    try:
        _ = gdf.sindex
    except Exception as e:
        raise RuntimeError(
            f"[{name}] spatial index not available. Install rtree (recommended).\n"
            f"Conda: conda install -c conda-forge rtree\n"
            f"Original error: {e}"
        )

def build_aridity(aridity_items: List[Tuple[str, str]]) -> gpd.GeoDataFrame:
    frames = []
    for name, path in aridity_items:
        g = read_layer(Path(path), 4326)[["geometry"]].copy()
        g["aridity_class"] = name
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(out, crs="EPSG:4326")

def assign_aridity_centroid(hex_gdf: gpd.GeoDataFrame,
                            aridity: gpd.GeoDataFrame,
                            hex_epsg_assume: int = 3857) -> pd.DataFrame:
    hx = hex_gdf.copy()
    if hx.crs is None:
        hx = hx.set_crs(hex_epsg_assume, allow_override=True)

    cent = hx.copy()
    cent["geometry"] = cent.geometry.centroid
    cent = cent.to_crs(4326)
    cent["aridity_class"] = pd.NA

    ensure_spatial_index(aridity, "aridity_polygons")

    for cls in ARIDITY_PRIORITY:
        polys = aridity.loc[aridity["aridity_class"] == cls, ["aridity_class", "geometry"]]
        if polys.empty:
            continue
        remaining = cent[cent["aridity_class"].isna()].copy()
        if remaining.empty:
            break
        joined = gpd.sjoin(remaining, polys, predicate="within", how="left")

        right_col = None
        if "aridity_class_right" in joined.columns:
            right_col = "aridity_class_right"
        elif "aridity_class" in joined.columns:
            right_col = "aridity_class"
        if right_col is None:
            raise RuntimeError("Unexpected sjoin output: cannot find aridity class column.")
        cent.loc[joined.index, "aridity_class"] = joined[right_col].values

    cent["aridity_class"] = cent["aridity_class"].fillna("Unclassified")
    return cent.drop(columns=["geometry"])


# -------------------------
# Helpers: attribution
# -------------------------
def normalize_dom(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    return s.replace({
        "natural_dominant": "NATURAL",
        "residual_dominant": "RESIDUAL",
        "Natural": "NATURAL",
        "Residual": "RESIDUAL",
    })

def add_three_component(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["beta_clim"] = out["beta_nat_clim"]
    out["beta_co2"] = out["beta_nat_co2"] - out["beta_nat_clim"]
    out["beta_res"] = out["beta_res_co2"]
    denom = (out["beta_clim"].abs() + out["beta_co2"].abs() + out["beta_res"].abs()).replace(0, np.nan)
    out["share_abs_clim"] = out["beta_clim"].abs() / denom
    out["share_abs_co2"] = out["beta_co2"].abs() / denom
    out["share_abs_res"] = out["beta_res"].abs() / denom
    return out

def summarize(df: pd.DataFrame, group_cols: List[str], share_col: str, dom_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_dom"] = normalize_dom(tmp[dom_col])

    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    g = tmp.groupby(group_cols, dropna=False)
    out = g.agg(
        n_hex=("hex_id", "size"),
        share_median=(share_col, "median"),
        share_p25=(share_col, q25),
        share_p75=(share_col, q75),
        natural_dominant=("_dom", lambda x: (x == "NATURAL").mean()),
        residual_dominant=("_dom", lambda x: (x == "RESIDUAL").mean()),
        beta_obs_median=("beta_obs", "median"),
        beta_nat_clim_median=("beta_nat_clim", "median"),
        beta_nat_co2_median=("beta_nat_co2", "median"),
        beta_res_co2_median=("beta_res_co2", "median"),
        share_abs_clim_median=("share_abs_clim", "median"),
        share_abs_co2_median=("share_abs_co2", "median"),
        share_abs_res_median=("share_abs_res", "median"),
    ).reset_index()
    return out

def delta_share(co2_df: pd.DataFrame, clim_df: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
    m = co2_df.merge(clim_df[on_cols + ["share_median"]], on=on_cols, how="left", suffixes=("_co2", "_clim"))
    out = co2_df.copy()
    out["delta_share_median"] = m["share_median_co2"] - m["share_median_clim"]
    return out


# -------------------------
# Helpers: wide xlsx -> long
# -------------------------
def _extract_year(col: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(col))
    return int(m.group(0)) if m else None

def wide_xlsx_to_long(path: Path, value_name: str, year_min: int, year_max: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 2:
        raise ValueError(f"Wide table seems empty: {path}")
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "hex_id"})
    df["hex_id"] = df["hex_id"].astype(str)

    col2year: Dict[str, int] = {}
    for c in df.columns[1:]:
        y = _extract_year(c)
        if y is not None and (year_min <= y <= year_max):
            col2year[c] = y
    if not col2year:
        raise ValueError(f"No year columns detected in: {path}")

    keep = ["hex_id"] + list(col2year.keys())
    df = df[keep].copy()
    out = df.melt(id_vars=["hex_id"], var_name="col", value_name=value_name)
    out["year"] = out["col"].astype(str).map(col2year).astype(int)
    out = out.drop(columns=["col"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out

def double_demean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    overall = {c: out[c].mean() for c in cols}
    by_hex = out.groupby("hex_id")[cols].transform("mean")
    by_year = out.groupby("year")[cols].transform("mean")
    for c in cols:
        out[c] = out[c] - by_hex[c] - by_year[c] + overall[c]
    return out

def pooled_vpd_resid(panel: pd.DataFrame, include_swnet: bool, do_double_demean: bool) -> pd.Series:
    """
    Construct VPD_resid from pooled OLS:
      VPD90 ~ Tmean (+ SWnet90)
    Optionally apply double-demean (hex+year) before regression to mimic SEM anomaly space.
    """
    need = ["VPD90", "Tmean"]
    if include_swnet and ("SWnet90" in panel.columns):
        need.append("SWnet90")

    df = panel[["hex_id", "year"] + need].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.Series(np.nan, index=panel.index, name="VPD_resid")

    if do_double_demean:
        df = double_demean(df, need)

    # OLS via normal equations (no statsmodels dependency)
    y = df["VPD90"].to_numpy(dtype=float)
    X_cols = ["Tmean"] + (["SWnet90"] if ("SWnet90" in df.columns and include_swnet) else [])
    X = df[X_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), X])  # intercept

    # beta = (X'X)^-1 X'y
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (X.T @ y)

    yhat = X @ beta
    resid = y - yhat

    # map back to full panel index by (hex_id,year)
    key = df["hex_id"].astype(str) + "__" + df["year"].astype(str)
    resid_s = pd.Series(resid, index=key, name="VPD_resid")

    full_key = panel["hex_id"].astype(str) + "__" + panel["year"].astype(str)
    return full_key.map(resid_s).rename("VPD_resid")


# -------------------------
# Metrics per hex from long panel
# -------------------------
def compute_metrics_from_long(long_df: pd.DataFrame, value_col: str, year_min: int, year_max: int) -> pd.DataFrame:
    """
    Input long_df: columns ['hex_id','year', value_col]
    Output per hex: bg_mean, bg_p90, trend(OLS slope vs year)
    """
    df = long_df.copy()
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    g = df.groupby("hex_id", sort=False)

    bg_mean = g[value_col].mean()
    bg_p90 = g[value_col].quantile(0.90)

    # trend slope per hex
    def _slope(sub: pd.DataFrame) -> float:
        y = sub[value_col].to_numpy(dtype=float)
        x = sub["year"].to_numpy(dtype=float)
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

    trend = g.apply(_slope)

    out = pd.DataFrame({
        "hex_id": bg_mean.index.astype(str),
        "bg_mean": bg_mean.values,
        "bg_p90": bg_p90.values,
        "trend": trend.values
    })
    return out


# -------------------------
# Binning & screening
# -------------------------
def quantile_bins(series: pd.Series, q: int) -> pd.Series:
    labels = [f"Q{i+1}" for i in range(q)]
    r = series.astype(float).rank(method="first")
    try:
        return pd.qcut(r, q, labels=labels)
    except Exception:
        uq = int(min(q, max(1, r.nunique())))
        if uq <= 1:
            return pd.Series(["Q1"] * len(series), index=series.index)
        lab2 = [f"Q{i+1}" for i in range(uq)]
        return pd.qcut(r, uq, labels=lab2)

def assign_bins(df: pd.DataFrame, col: str, q: int, within_aridity: bool) -> pd.Series:
    if within_aridity:
        return df.groupby("aridity_class")[col].transform(lambda s: quantile_bins(s, q))
    return quantile_bins(df[col], q)

def spearman_sign(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.corr(y, method="spearman")

def compute_screening_scores(table_co2: pd.DataFrame,
                             aridity_col: str,
                             bin_col: str,
                             min_bins_required: int) -> Dict[str, float]:
    df = table_co2.copy()

    def bin_to_int(b):
        m = re.search(r"Q(\d+)", str(b))
        return int(m.group(1)) if m else np.nan

    df["_bin_i"] = df[bin_col].apply(bin_to_int)
    df = df.dropna(subset=["_bin_i"])
    min_bin_n = float(df["n_hex"].min()) if len(df) else np.nan

    grads_co2, grads_res, mono = [], [], []
    for _, g in df.groupby(aridity_col):
        g = g.sort_values("_bin_i")
        gg = g[["share_abs_co2_median", "share_abs_res_median", "_bin_i"]].dropna()
        if gg["_bin_i"].nunique() < min_bins_required:
            continue
        grads_co2.append(float(gg["share_abs_co2_median"].iloc[-1] - gg["share_abs_co2_median"].iloc[0]))
        grads_res.append(float(gg["share_abs_res_median"].iloc[-1] - gg["share_abs_res_median"].iloc[0]))
        rho = spearman_sign(gg["_bin_i"], gg["share_abs_co2_median"])
        mono.append(1.0 if (np.isfinite(rho) and abs(rho) > 0) else 0.0)

    return {
        "avg_gradient_share_co2": float(np.mean(grads_co2)) if grads_co2 else np.nan,
        "avg_gradient_resid": float(np.mean(grads_res)) if grads_res else np.nan,
        "monotonicity_score": float(np.mean(mono)) if mono else np.nan,
        "min_bin_n": min_bin_n
    }


# -------------------------
# Build drivers from DIR_UP
# -------------------------
def build_driver_long_tables(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Return dict: driver_key -> long table with columns ['hex_id','year', driver_key]
    """
    d = Path(cfg["DIR_UP"])
    y0, y1 = int(cfg["YEAR_MIN"]), int(cfg["YEAR_MAX"])

    def read_long(filename: str, colname: str) -> pd.DataFrame | None:
        p = d / filename
        if not p.exists():
            return None
        return wide_xlsx_to_long(p, colname, y0, y1)

    out: Dict[str, pd.DataFrame] = {}

    # base variables
    p90 = read_long(cfg["FILE_P90"], "P90")
    vpd = read_long(cfg["FILE_VPD90"], "VPD90")
    tmean = read_long(cfg["FILE_TMEAN"], "Tmean")
    sm_l1 = read_long(cfg["FILE_SM_L1"], "SM90_L1")

    # optional
    sm_root = None
    if cfg.get("FILE_SM_ROOT"):
        sm_root = read_long(cfg["FILE_SM_ROOT"], "SM90_root")
    et = None
    if cfg.get("FILE_ET90"):
        et = read_long(cfg["FILE_ET90"], "ET90")
    swnet = None
    if cfg.get("FILE_SWNET90"):
        swnet = read_long(cfg["FILE_SWNET90"], "SWnet90")

    # register if exists
    if p90 is not None: out["P90"] = p90
    if vpd is not None: out["VPD90"] = vpd
    if tmean is not None: out["Tmean"] = tmean
    if sm_l1 is not None: out["SM90_L1"] = sm_l1
    if sm_root is not None: out["SM90_root"] = sm_root
    if et is not None: out["ET90"] = et
    if swnet is not None: out["SWnet90"] = swnet

    # derived VPD_resid requires VPD90 + Tmean (+ SWnet90)
    if ("VPD90" in out) and ("Tmean" in out):
        base = out["VPD90"].merge(out["Tmean"], on=["hex_id", "year"], how="inner")
        if cfg["VPD_RESID_INCLUDE_SWNET"] and ("SWnet90" in out):
            base = base.merge(out["SWnet90"], on=["hex_id", "year"], how="left")

        # compute resid in this merged long-panel
        base["hex_id"] = base["hex_id"].astype(str)
        base["year"] = base["year"].astype(int)

        # create residual series aligned to base rows
        base["VPD_resid"] = pooled_vpd_resid(
            panel=base.rename(columns={"Tmean": "Tmean"}),  # keep names
            include_swnet=bool(cfg["VPD_RESID_INCLUDE_SWNET"]),
            do_double_demean=bool(cfg["VPD_RESID_DOUBLE_DEMEAN"])
        ).to_numpy()

        out["VPD_resid"] = base[["hex_id", "year", "VPD_resid"]].copy()

    return out


# -------------------------
# Main
# -------------------------
def validate_config(cfg: dict) -> None:
    need_paths = [
        "ATTR_UP_XLSX", "HEX_UP_SHP",
        "ARIDITY_HYPERARID", "ARIDITY_ARID", "ARIDITY_SEMIARID", "ARIDITY_DRYSUBHUMID",
        "DIR_UP"
    ]
    for k in need_paths:
        if not cfg.get(k):
            raise ValueError(f"CONFIG missing: {k}")
        if not Path(cfg[k]).exists():
            raise FileNotFoundError(f"Path not found: {cfg[k]}")

def main() -> None:
    validate_config(CONFIG)

    out_dir = Path(CONFIG["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "driver_metrics_UP").mkdir(parents=True, exist_ok=True)

    # aridity polygons
    aridity = build_aridity([
        ("Hyperarid", CONFIG["ARIDITY_HYPERARID"]),
        ("Arid", CONFIG["ARIDITY_ARID"]),
        ("Semiarid", CONFIG["ARIDITY_SEMIARID"]),
        ("Dry_subhumid", CONFIG["ARIDITY_DRYSUBHUMID"]),
    ])

    # hex + aridity assignment
    up_hex = read_layer(Path(CONFIG["HEX_UP_SHP"]), 3857)
    hex_id_field = CONFIG["HEX_ID_FIELD"]
    if hex_id_field not in up_hex.columns:
        raise ValueError(f"Missing field '{hex_id_field}' in {CONFIG['HEX_UP_SHP']}")
    lab = assign_aridity_centroid(up_hex[[hex_id_field, "geometry"]], aridity)
    lab = lab.rename(columns={hex_id_field: "hex_id"})
    lab["hex_id"] = lab["hex_id"].astype(str)
    lab.to_csv(out_dir / "hex_to_aridity_UP.csv", index=False)

    # attribution (UP)
    attr = pd.read_excel(Path(CONFIG["ATTR_UP_XLSX"]))
    if "hex_id" not in attr.columns:
        attr = attr.rename(columns={attr.columns[0]: "hex_id"})
    attr["hex_id"] = attr["hex_id"].astype(str)

    required = [
        "hex_id", "beta_obs",
        "beta_nat_clim", "beta_res_clim", "share_nat_clim", "dominance_clim",
        "beta_nat_co2", "beta_res_co2", "share_nat_co2", "dominance_co2",
    ]
    miss = [c for c in required if c not in attr.columns]
    if miss:
        raise KeyError("Attribution table missing required columns:\n" + "\n".join(miss))

    attr = attr.merge(lab[["hex_id", "aridity_class"]], on="hex_id", how="left")
    attr["aridity_class"] = attr["aridity_class"].fillna("Unclassified")
    attr = add_three_component(attr)

    # 1D
    clim_1d = summarize(attr, ["aridity_class"], "share_nat_clim", "dominance_clim")
    co2_1d = summarize(attr, ["aridity_class"], "share_nat_co2", "dominance_co2")
    co2_1d = delta_share(co2_1d, clim_1d, ["aridity_class"])

    order = ARIDITY_PRIORITY + ["Unclassified"]
    for t in (clim_1d, co2_1d):
        t["aridity_class"] = pd.Categorical(t["aridity_class"], categories=order, ordered=True)
        t.sort_values("aridity_class", inplace=True)
        t["aridity_class"] = t["aridity_class"].astype(str)

    clim_1d.to_csv(out_dir / "stratified_UP_aridity_1d_clim.csv", index=False)
    co2_1d.to_csv(out_dir / "stratified_UP_aridity_1d_co2.csv", index=False)

    # build drivers from DIR_UP (new structure)
    driver_longs = build_driver_long_tables(CONFIG)

    # filter to selected DRIVER_KEYS
    wanted = list(CONFIG["DRIVER_KEYS"])
    available = set(driver_longs.keys())
    use_keys = [k for k in wanted if k in available]
    missing_keys = [k for k in wanted if k not in available]
    if missing_keys:
        print("[WARN] These DRIVER_KEYS are not available (missing input files or deps):", missing_keys)

    metrics = list(CONFIG["METRICS"])
    bins = int(CONFIG["BINS"])
    within_aridity = bool(CONFIG["WITHIN_ARIDITY"])
    min_bins_required = int(CONFIG["MIN_BINS_REQUIRED"])
    y0, y1 = int(CONFIG["YEAR_MIN"]), int(CONFIG["YEAR_MAX"])

    all_tables: Dict[str, pd.DataFrame] = {
        "UP_1D_clim": clim_1d,
        "UP_1D_co2": co2_1d,
    }
    rank_rows = []

    # per driver: compute metrics per hex, merge into attr, then 2D summaries
    for dkey in use_keys:
        long_df = driver_longs[dkey].copy()
        val_col = dkey

        dm = compute_metrics_from_long(long_df, value_col=val_col, year_min=y0, year_max=y1)
        dm.to_csv(out_dir / "driver_metrics_UP" / f"{dkey}_metrics.csv", index=False)

        tmp = attr.merge(dm, on="hex_id", how="left")

        for met in metrics:
            bin_col = f"{dkey}_{met}_bin"
            tmp[bin_col] = assign_bins(tmp, col=met, q=bins, within_aridity=within_aridity)
            group_cols = ["aridity_class", bin_col]

            clim_2d = summarize(tmp, group_cols, "share_nat_clim", "dominance_clim")
            co2_2d = summarize(tmp, group_cols, "share_nat_co2", "dominance_co2")
            co2_2d = delta_share(co2_2d, clim_2d, group_cols)

            clim_2d.to_csv(out_dir / f"stratified_UP_aridity_x_{dkey}__{met}_clim.csv", index=False)
            co2_2d.to_csv(out_dir / f"stratified_UP_aridity_x_{dkey}__{met}_co2.csv", index=False)

            all_tables[f"{dkey}_{met}_clim"] = clim_2d
            all_tables[f"{dkey}_{met}_co2"] = co2_2d

            scores = compute_screening_scores(co2_2d, "aridity_class", bin_col, min_bins_required)
            rank_rows.append({
                "driver": dkey,
                "metric": met,
                "bins": bins,
                "within_aridity": within_aridity,
                **scores
            })

    # ranking
    if rank_rows:
        rank = pd.DataFrame(rank_rows)
        rank["composite_score"] = (
            rank["avg_gradient_share_co2"].abs().fillna(0.0) * 1.0 +
            rank["monotonicity_score"].fillna(0.0) * 0.5
        )
        # penalize small bins (500 is a pragmatic default)
        rank["composite_score"] = rank["composite_score"] * np.minimum(
            1.0, rank["min_bin_n"].fillna(0.0) / 500.0
        )
        rank = rank.sort_values("composite_score", ascending=False)
        rank.to_csv(out_dir / "screening_rank_UP.csv", index=False)
        all_tables["screening_rank_UP"] = rank

    # excel book
    with pd.ExcelWriter(out_dir / "aridity_multi_driver_tables_UP.xlsx", engine="openpyxl") as w:
        for name, df in all_tables.items():
            df.to_excel(w, sheet_name=name[:31], index=False)

    runinfo = {
        "DIR_UP": CONFIG["DIR_UP"],
        "driver_keys_requested": CONFIG["DRIVER_KEYS"],
        "driver_keys_used": use_keys,
        "driver_keys_missing": missing_keys,
        "year_min": y0,
        "year_max": y1,
        "metrics": metrics,
        "bins": bins,
        "within_aridity": within_aridity,
        "VPD_resid": {
            "include_swnet": bool(CONFIG["VPD_RESID_INCLUDE_SWNET"]),
            "double_demean": bool(CONFIG["VPD_RESID_DOUBLE_DEMEAN"]),
            "note": "pooled OLS: VPD90 ~ Tmean (+SWnet90), residual used for stratification"
        },
        "notes": "Drivers aligned with the canonical panel-attribution and SEM variable system."
    }
    (out_dir / "runinfo.json").write_text(json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
