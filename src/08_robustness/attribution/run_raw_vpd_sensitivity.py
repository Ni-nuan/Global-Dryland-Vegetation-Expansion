# -*- coding: utf-8 -*-
"""
run_raw_vpd_sensitivity.py

Raw-VPD sensitivity branch of the panel attribution workflow, retaining
logit-scale main outputs and probability-scale supplementary outputs.

Key points:
- MAIN outputs (logit scale): identical logic to v2a (self-consistent additive decomposition)
- SUPPLEMENT outputs (veg_frac scale): derived from inv_logit(yhat) and veg_frac residuals
- Keeps UP/DOWN only, veg_frac files configured separately, XCO2 panel used (no canonical series)
- Adds engineering runinfo/panel-size JSONs (do NOT affect results)

Outputs per dataset tag (UP/DOWN) into OUTDIR:
  00_runinfo_<TAG>.json
  01_panel_sizes_<TAG>.json
  02_fe_summary_CLIM_ONLY_<TAG>.txt
  02_fe_summary_CLIM_PLUS_CO2_<TAG>.txt
  06_trend_attribution_by_hex_<TAG>.xlsx                (MAIN, logit scale)
  07_trend_attribution_summary_<TAG>.json               (MAIN, logit scale)
  08_year_mean_series_<TAG>.xlsx                        (MAIN, logit scale)
  06_trend_attribution_by_hex_frac_<TAG>.xlsx           (SUPP, veg_frac scale)
  07_trend_attribution_summary_frac_<TAG>.json          (SUPP, veg_frac scale)
  08_year_mean_series_frac_<TAG>.xlsx                   (SUPP, veg_frac scale)
  co2_year_mean_<TAG>.csv
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


# =============================================================================
# USER CONFIG (EDIT THESE PATHS / FILENAMES)
# =============================================================================

# Input directories (ONLY these two)
DIR_UP = r"zonal_tables_parallel_up_V2"
DIR_DOWN = r"zonal_tables_parallel_down_V2"

# Dependent variable file (wide: hex_id + year columns) — set separately.
# You may set a filename (within DIR_UP/DIR_DOWN) OR an absolute path.
FILE_VEGFRAC_UP = r"NDVI_trend_hex_100_up.xlsx"         # e.g., r"veg_frac_UP.xlsx" or r"C:\...\veg_frac_UP.xlsx"
FILE_VEGFRAC_DOWN = r"NDVI_trend_hex_100_down.xlsx"     # e.g., r"veg_frac_DOWN.xlsx" or r"C:\...\veg_frac_DOWN.xlsx"

# Climate filenames (within each dir)
FILE_P90     = r"W90_Precip.xlsx"         # -> P90
FILE_VPD90   = r"VPD90_mean.xlsx"         # -> VPD90
FILE_T90     = r"T2m90_mean.xlsx"         # -> Tmean90
FILE_SM90    = r"SM_L1_90_mean.xlsx"      # -> SM90
FILE_ET90    = r"ET90_sum.xlsx"           # -> ET90 (optional)
FILE_SWNET90 = r"SWnet90_mean.xlsx"       # -> SWnet90 (optional)

# XCO2 filename (within each dir)
FILE_XCO2    = r"CO2_annual_tif.xlsx"     # -> XCO2

# Switches
INCLUDE_ET = True
INCLUDE_SWNET = True
INCLUDE_INTERACTION = True   # P90×VPD90

# Standardization
Z_SCORE_X = True             # z-score covariates on regression sample
CENTER_ONLY = False          # if Z_SCORE_X=False, optionally mean-center

# Years
YEAR_MIN = 2000
YEAR_MAX = 2022

# Logit stability
LOGIT_EPS = 1e-6

# Output directory
OUTDIR = r"trend_outputs_xco2_raw_vpd"


# =============================================================================
# Utilities (robust IO + parsing)
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm_path(base_dir: str, p: str) -> str:
    """If p is absolute, use it; otherwise join with base_dir."""
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


def _extract_year_from_col(col: str) -> int | None:
    """
    Accepts: '2000', 'Y2000', 'year_2000', '2000.0', etc.
    Returns int year if found else None.
    """
    s = str(col)
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def wide_xlsx_to_long(path: str, value_name: str, year_min: int, year_max: int) -> pd.DataFrame:
    """
    Read a wide xlsx where first column is hex_id (or similar), and other columns contain years.
    Output: hex_id (str), year (int), value_name (float).
    """
    df = pd.read_excel(path)
    if df.shape[1] < 2:
        raise ValueError(f"Wide table seems empty: {path}")

    # first column as hex_id
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "hex_id"})
    df["hex_id"] = df["hex_id"].astype(str)

    # map columns -> year
    col2year: Dict[str, int] = {}
    for c in df.columns[1:]:
        y = _extract_year_from_col(c)
        if y is None:
            continue
        if year_min <= y <= year_max:
            col2year[c] = y

    if not col2year:
        raise ValueError(f"No year columns detected in: {path}")

    keep_cols = ["hex_id"] + list(col2year.keys())
    df = df[keep_cols].copy()

    out = df.melt(id_vars=["hex_id"], var_name="col", value_name=value_name)
    out["year"] = out["col"].astype(str).map(col2year).astype(int)
    out = out.drop(columns=["col"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    """Inverse-logit with numerical stability."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def slope_ols(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return np.nan
    xx = x[mask].astype(float)
    yy = y[mask].astype(float)
    vx = np.var(xx, ddof=0)
    if vx == 0:
        return np.nan
    cov = np.mean((xx - xx.mean()) * (yy - yy.mean()))
    return cov / vx


def standardize_fit(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Fit z-score/centering parameters on df[cols] and return transformed df + params."""
    out = df.copy()
    params: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        mu = float(s.mean())
        if Z_SCORE_X:
            sd = float(s.std(ddof=0))
            if (not np.isfinite(sd)) or sd == 0:
                sd = 1.0
            out[c] = (s - mu) / sd
            params[c] = (mu, sd)
        elif CENTER_ONLY:
            out[c] = s - mu
            params[c] = (mu, 1.0)
        else:
            out[c] = s
            params[c] = (0.0, 1.0)
    return out, params


def standardize_apply(df: pd.DataFrame, cols: List[str], params: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu, sd = params[c]
        s = pd.to_numeric(out[c], errors="coerce")
        if Z_SCORE_X or CENTER_ONLY:
            out[c] = (s - mu) / sd
        else:
            out[c] = s
    return out


@dataclass
class DatasetSpec:
    tag: str
    dirpath: str
    veg_file: str


# =============================================================================
# Build panel (UP or DOWN)
# =============================================================================

def build_panel(spec: DatasetSpec) -> pd.DataFrame:
    d = spec.dirpath

    # Dependent (veg_frac)
    veg_path = _norm_path(d, spec.veg_file)
    y = wide_xlsx_to_long(veg_path, "veg_frac", YEAR_MIN, YEAR_MAX)

    # Climate system
    p90   = wide_xlsx_to_long(_norm_path(d, FILE_P90), "P90", YEAR_MIN, YEAR_MAX)
    vpd90 = wide_xlsx_to_long(_norm_path(d, FILE_VPD90), "VPD90", YEAR_MIN, YEAR_MAX)
    tmean = wide_xlsx_to_long(_norm_path(d, FILE_T90), "Tmean90", YEAR_MIN, YEAR_MAX)
    sm90  = wide_xlsx_to_long(_norm_path(d, FILE_SM90), "SM90", YEAR_MIN, YEAR_MAX)

    frames = [y, p90, vpd90, tmean, sm90]

    if INCLUDE_ET:
        frames.append(wide_xlsx_to_long(_norm_path(d, FILE_ET90), "ET90", YEAR_MIN, YEAR_MAX))
    if INCLUDE_SWNET:
        frames.append(wide_xlsx_to_long(_norm_path(d, FILE_SWNET90), "SWnet90", YEAR_MIN, YEAR_MAX))

    # XCO2 panel (no canonical series; each dir uses its own file)
    frames.append(wide_xlsx_to_long(_norm_path(d, FILE_XCO2), "XCO2", YEAR_MIN, YEAR_MAX))

    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on=["hex_id", "year"], how="left")

    panel["y_logit"] = logit(panel["veg_frac"].to_numpy(dtype=float), eps=LOGIT_EPS)

    if INCLUDE_INTERACTION:
        panel["P90xVPD"] = panel["P90"] * panel["VPD90"]

    return panel


# =============================================================================
# Modeling + Scheme A attribution
# =============================================================================

def get_covars(include_xco2: bool) -> List[str]:
    covars = ["P90", "VPD90", "Tmean90", "SM90"]
    if INCLUDE_ET:
        covars.append("ET90")
    if INCLUDE_SWNET:
        covars.append("SWnet90")
    if INCLUDE_INTERACTION:
        covars.append("P90xVPD")
    if include_xco2:
        covars.append("XCO2")
    return covars


def fit_panelols(panel: pd.DataFrame, include_xco2: bool):
    covars = get_covars(include_xco2)
    need_cols = ["hex_id", "year", "y_logit"] + covars

    df = panel[need_cols].copy().replace([np.inf, -np.inf], np.nan)
    df_model = df.dropna(subset=["y_logit"] + covars).copy()

    # Fit standardization params on regression sample
    df_std, params = standardize_fit(df_model, covars)
    df_std = df_std.set_index(["hex_id", "year"]).sort_index()

    y = df_std["y_logit"]
    X = df_std[covars]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    sample_info = {
        "n_obs": int(df_std.shape[0]),
        "n_hex": int(df_std.index.get_level_values(0).nunique()),
        "min_obs_per_hex": int(df_std.groupby(level=0).size().min()),
        "avg_obs_per_hex": float(df_std.groupby(level=0).size().mean()),
        "max_obs_per_hex": int(df_std.groupby(level=0).size().max()),
    }
    return res, covars, params, sample_info


def extract_entity_effects(res) -> Dict[str, float]:
    eff = res.estimated_effects
    if isinstance(eff, pd.DataFrame):
        s = eff.iloc[:, 0] if eff.shape[1] == 1 else eff.mean(axis=1)
    else:
        s = eff

    if isinstance(s.index, pd.MultiIndex):
        ent = s.groupby(level=0).mean()
        ent.index = ent.index.astype(str)
        return ent.to_dict()
    return {}


def predict_schemeA(panel: pd.DataFrame, res, covars: List[str], params: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df = panel[["hex_id", "year", "veg_frac", "y_logit"] + covars].copy().replace([np.inf, -np.inf], np.nan)
    df_std = standardize_apply(df, covars, params)

    beta = res.params.reindex(covars).to_numpy(dtype=float)
    df_std["xbeta"] = df_std[covars].to_numpy(dtype=float) @ beta

    alpha = extract_entity_effects(res)
    df_std["alpha_i"] = df_std["hex_id"].astype(str).map(alpha)
    df_std["yhat"] = df_std["alpha_i"] + df_std["xbeta"]
    df_std["resid"] = df_std["y_logit"] - df_std["yhat"]

    # Supplementary probability-scale (veg_frac) predictions/residuals
    df_std["yhat_frac"] = inv_logit(df_std["yhat"].to_numpy(dtype=float))
    df_std["resid_frac"] = df_std["veg_frac"].to_numpy(dtype=float) - df_std["yhat_frac"].to_numpy(dtype=float)
    return df_std


def trend_by_hex_logit(df_pred: pd.DataFrame) -> pd.DataFrame:
    def _slopes(g: pd.DataFrame) -> pd.Series:
        x = g["year"].to_numpy(dtype=float)
        return pd.Series({
            "beta_obs": slope_ols(g["y_logit"].to_numpy(dtype=float), x),
            "beta_nat": slope_ols(g["yhat"].to_numpy(dtype=float), x),
            "beta_res": slope_ols(g["resid"].to_numpy(dtype=float), x),
        })

    out = df_pred.groupby("hex_id", sort=False).apply(_slopes).reset_index()
    out["share_nat"] = np.abs(out["beta_nat"]) / (np.abs(out["beta_nat"]) + np.abs(out["beta_res"]))
    out["dominance"] = np.where(np.abs(out["beta_nat"]) >= np.abs(out["beta_res"]), "NATURAL", "RESIDUAL")
    return out


def trend_by_hex_frac(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Supplementary trends on veg_frac scale (probability scale)."""
    def _slopes(g: pd.DataFrame) -> pd.Series:
        x = g["year"].to_numpy(dtype=float)
        return pd.Series({
            "beta_obs": slope_ols(g["veg_frac"].to_numpy(dtype=float), x),
            "beta_nat": slope_ols(g["yhat_frac"].to_numpy(dtype=float), x),
            "beta_res": slope_ols(g["resid_frac"].to_numpy(dtype=float), x),
        })

    out = df_pred.groupby("hex_id", sort=False).apply(_slopes).reset_index()
    out["share_nat"] = np.abs(out["beta_nat"]) / (np.abs(out["beta_nat"]) + np.abs(out["beta_res"]))
    out["dominance"] = np.where(np.abs(out["beta_nat"]) >= np.abs(out["beta_res"]), "NATURAL", "RESIDUAL")
    return out


def year_means_logit(df_pred: pd.DataFrame) -> pd.DataFrame:
    return df_pred.groupby("year", as_index=False).agg(
        y_logit_mean=("y_logit", "mean"),
        yhat_mean=("yhat", "mean"),
        resid_mean=("resid", "mean"),
        n=("y_logit", "count")
    )


def year_means_frac(df_pred: pd.DataFrame) -> pd.DataFrame:
    return df_pred.groupby("year", as_index=False).agg(
        veg_frac_mean=("veg_frac", "mean"),
        yhat_frac_mean=("yhat_frac", "mean"),
        resid_frac_mean=("resid_frac", "mean"),
        n=("veg_frac", "count")
    )


# =============================================================================
# Run
# =============================================================================

def run_dataset(spec: DatasetSpec) -> None:
    ensure_dir(OUTDIR)

    print(f"\n=== {spec.tag} ===")
    print(f"[{spec.tag}] dir: {spec.dirpath}")
    print(f"[{spec.tag}] veg file: {spec.veg_file}")

    panel = build_panel(spec)

    # ---- basic runinfo / panel sizes (engineering outputs; do not change results)
    runinfo = {
        "dataset_tag": spec.tag,
        "dir": spec.dirpath,
        "veg_file": spec.veg_file,
        "years": [YEAR_MIN, YEAR_MAX],
        "switches": {
            "include_ET": INCLUDE_ET,
            "include_SWnet": INCLUDE_SWNET,
            "include_interaction_P90xVPD": INCLUDE_INTERACTION,
            "standardize": "zscore(reg_sample)" if Z_SCORE_X else ("center(reg_sample)" if CENTER_ONLY else "none"),
            "logit_eps": LOGIT_EPS,
        },
        "covariates_base": get_covars(include_xco2=False),
        "covariates_with_xco2": get_covars(include_xco2=True),
    }
    safe_json_dump(runinfo, os.path.join(OUTDIR, f"00_runinfo_{spec.tag}.json"))

    panel_sizes = {
        "n_rows_panel": int(panel.shape[0]),
        "n_hex_panel": int(panel["hex_id"].nunique()),
        "n_year_panel": int(panel["year"].nunique()),
        "n_rows_nonmissing_veg_frac": int(panel["veg_frac"].notna().sum()),
        "n_rows_nonmissing_y_logit": int(panel["y_logit"].notna().sum()),
    }
    safe_json_dump(panel_sizes, os.path.join(OUTDIR, f"01_panel_sizes_{spec.tag}.json"))

    # ---- CLIM_ONLY (MAIN: logit-scale)
    res1, cov1, par1, info1 = fit_panelols(panel, include_xco2=False)
    with open(os.path.join(OUTDIR, f"02_fe_summary_CLIM_ONLY_{spec.tag}.txt"), "w", encoding="utf-8") as f:
        f.write(str(res1.summary))

    pred1 = predict_schemeA(panel, res1, cov1, par1)

    attr1 = trend_by_hex_logit(pred1).rename(columns={
        "beta_nat": "beta_nat_clim",
        "beta_res": "beta_res_clim",
        "share_nat": "share_nat_clim",
        "dominance": "dominance_clim",
    })
    yr1 = year_means_logit(pred1).rename(columns={
        "yhat_mean": "yhat_clim_mean",
        "resid_mean": "resid_clim_mean",
    })

    # ---- CLIM_PLUS_XCO2 (MAIN: logit-scale)
    res2, cov2, par2, info2 = fit_panelols(panel, include_xco2=True)
    with open(os.path.join(OUTDIR, f"02_fe_summary_CLIM_PLUS_CO2_{spec.tag}.txt"), "w", encoding="utf-8") as f:
        f.write(str(res2.summary))

    pred2 = predict_schemeA(panel, res2, cov2, par2)

    attr2 = trend_by_hex_logit(pred2).rename(columns={
        "beta_nat": "beta_nat_co2",
        "beta_res": "beta_res_co2",
        "share_nat": "share_nat_co2",
        "dominance": "dominance_co2",
    })
    yr2 = year_means_logit(pred2).rename(columns={
        "yhat_mean": "yhat_co2_mean",
        "resid_mean": "resid_co2_mean",
    })

    # ---- merge MAIN outputs (logit-scale)
    out_hex = attr1.merge(attr2, on=["hex_id", "beta_obs"], how="outer")
    out_year = yr1.merge(yr2, on=["year", "y_logit_mean", "n"], how="outer")

    # ---- supplementary probability-scale attribution (veg_frac)
    attr1f = trend_by_hex_frac(pred1).rename(columns={
        "beta_nat": "beta_nat_clim",
        "beta_res": "beta_res_clim",
        "share_nat": "share_nat_clim",
        "dominance": "dominance_clim",
    })
    yr1f = year_means_frac(pred1).rename(columns={
        "yhat_frac_mean": "yhat_clim_mean",
        "resid_frac_mean": "resid_clim_mean",
    })

    attr2f = trend_by_hex_frac(pred2).rename(columns={
        "beta_nat": "beta_nat_co2",
        "beta_res": "beta_res_co2",
        "share_nat": "share_nat_co2",
        "dominance": "dominance_co2",
    })
    yr2f = year_means_frac(pred2).rename(columns={
        "yhat_frac_mean": "yhat_co2_mean",
        "resid_frac_mean": "resid_co2_mean",
    })

    out_hex_frac = attr1f.merge(attr2f, on=["hex_id", "beta_obs"], how="outer")
    out_year_frac = yr1f.merge(yr2f, on=["year", "veg_frac_mean", "n"], how="outer")

    def _summ(df: pd.DataFrame, share_col: str, dom_col: str) -> Dict[str, float]:
        s = df[share_col].replace([np.inf, -np.inf], np.nan).dropna()
        d = df[dom_col].dropna()
        nat_dom = float((d == "NATURAL").mean()) if len(d) else np.nan
        return {
            "n_hex": int(df["hex_id"].nunique()),
            "share_nat_median": float(s.median()) if len(s) else np.nan,
            "natural_dominant_pct": nat_dom
        }

    summary = {
        "dataset_tag": spec.tag,
        "scale": "logit",
        "years": [YEAR_MIN, YEAR_MAX],
        "include_ET": INCLUDE_ET,
        "include_SWnet": INCLUDE_SWNET,
        "include_interaction_P90xVPD": INCLUDE_INTERACTION,
        "standardize": "zscore(reg_sample)" if Z_SCORE_X else ("center(reg_sample)" if CENTER_ONLY else "none"),
        "CLIM_ONLY": {**_summ(out_hex, "share_nat_clim", "dominance_clim"), "sample_info": info1},
        "CLIM_PLUS_XCO2": {**_summ(out_hex, "share_nat_co2", "dominance_co2"), "sample_info": info2},
    }

    # ---- save (MAIN + SUPP)
    out_hex.to_excel(os.path.join(OUTDIR, f"06_trend_attribution_by_hex_{spec.tag}.xlsx"), index=False)
    safe_json_dump(summary, os.path.join(OUTDIR, f"07_trend_attribution_summary_{spec.tag}.json"))
    out_year.to_excel(os.path.join(OUTDIR, f"08_year_mean_series_{spec.tag}.xlsx"), index=False)

    summary_frac = {
        **summary,
        "scale": "veg_frac",
        "CLIM_ONLY": {**_summ(out_hex_frac, "share_nat_clim", "dominance_clim"), "sample_info": info1},
        "CLIM_PLUS_XCO2": {**_summ(out_hex_frac, "share_nat_co2", "dominance_co2"), "sample_info": info2},
    }
    out_hex_frac.to_excel(os.path.join(OUTDIR, f"06_trend_attribution_by_hex_frac_{spec.tag}.xlsx"), index=False)
    safe_json_dump(summary_frac, os.path.join(OUTDIR, f"07_trend_attribution_summary_frac_{spec.tag}.json"))
    out_year_frac.to_excel(os.path.join(OUTDIR, f"08_year_mean_series_frac_{spec.tag}.xlsx"), index=False)

    # XCO2 annual mean snapshot (per dir)
    panel.groupby("year", as_index=False).agg(
        XCO2_mean=("XCO2", "mean"),
        XCO2_std=("XCO2", "std"),
        n=("XCO2", "count")
    ).to_csv(os.path.join(OUTDIR, f"co2_year_mean_{spec.tag}.csv"), index=False, encoding="utf-8")

    print(f"[{spec.tag}] done. Outputs in: {OUTDIR}")


def main() -> None:
    ensure_dir(OUTDIR)

    specs = [
        DatasetSpec(tag="UP", dirpath=DIR_UP, veg_file=FILE_VEGFRAC_UP),
        DatasetSpec(tag="DOWN", dirpath=DIR_DOWN, veg_file=FILE_VEGFRAC_DOWN),
    ]

    for spec in specs:
        run_dataset(spec)

    print(f"\nAll done. Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
