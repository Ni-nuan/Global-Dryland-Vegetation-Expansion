#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nat0_nat1_nat2_yearfe_diagnostics_v3_vpd_resid_with_time.py

Goal:
  Build NAT0 vs NAT1 vs NAT2 diagnostic chain on the LOGIT scale,
  add a CLIM_PLUS_TIME diagnostic that replaces XCO2 with a centered linear year term,
  and use VPD_resid rather than raw VPD90 in all climate specifications.

Definitions:
  - NAT0: entity (hex) FE, time_effects=False
  - NAT1: entity (hex) FE, time_effects=True
  - NAT2: start from NAT1, but replace its time component with the CO2-explained component:
          gamma_t ≈ E[estimated_effects | year=t]  (proxy of time FE up to a constant)
          gamma_hat_co2(t) = f[CO2_mean(t)] using CO2 or CO2 + CO2^2
          yhat_nat2 = yhat_nat1 - gamma_t + gamma_hat_co2(t)
    (This preserves hex FE and Xβ, only swaps the common time component; slopes are invariant to constants.)
  - CLIM_PLUS_TIME: entity (hex) FE, time_effects=False, local climate predictors plus
          a centered linear year term instead of XCO2. This directly tests whether a
          generic monotonic temporal component produces a similar attribution shift.
  - VPD_resid: residualized atmospheric demand from VPD90 ~ Tmean90 (+ SWnet90),
          optionally after hexagon and year double-demeaning, matching the main panel
          attribution framework.

Outputs (per TAG=UP/DOWN):
  OUTDIR/
    NAT2_timeFE_projection_<TAG>.csv
    NAT_diagnostics_by_hex_<TAG>.xlsx
    NAT_diagnostics_summary_<TAG>.json
"""

from __future__ import annotations

import os, re, json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

# =============================================================================
# USER CONFIG (edit paths)
# =============================================================================
DIR_UP = r"zonal_tables_parallel_up_V2"
DIR_DOWN = r"zonal_tables_parallel_down_V2"

FILE_VEGFRAC_UP = r"NDVI_trend_hex_100_up.xlsx"
FILE_VEGFRAC_DOWN = r"NDVI_trend_hex_100_down.xlsx"

FILE_P90     = r"W90_Precip.xlsx"
FILE_VPD90   = r"VPD90_mean.xlsx"
FILE_TMEAN   = r"T2m90_mean.xlsx"
FILE_SM90    = r"SM_L1_90_mean.xlsx"
FILE_ET90    = r"ET90_sum.xlsx"            # optional
FILE_SWNET90 = r"SWnet90_mean.xlsx"        # optional
FILE_XCO2    = r"CO2_annual_tif.xlsx"

INCLUDE_ET = True
INCLUDE_SWNET = True
INCLUDE_INTERACTION = True   # P90 x VPD_resid

# VPD_resid settings, aligned with the main panel attribution script
VPD_RESID_INCLUDE_SWNET = True
VPD_RESID_DOUBLE_DEMEAN = True

# NAT2 projection settings
NAT2_CO2_QUADRATIC = True   # regress year effects on XCO2 + XCO2^2

YEAR_MIN = 2000
YEAR_MAX = 2022

Z_SCORE_X = True
CENTER_ONLY = False          # only used when Z_SCORE_X=False
LOGIT_EPS = 1e-6

OUTDIR = r"nat_yearfe_contrast_vpdresid_with_time"

# =============================================================================
# Helpers
# =============================================================================
def safe_r2_within(res) -> float:
    """
    Compatible across linearmodels versions.
    Try res.rsquared.within, then res.rsquared (float), then NaN.
    """
    try:
        r2 = res.rsquared
        # Newer versions: res.rsquared is a structure with .within
        if hasattr(r2, "within"):
            return float(r2.within)
        # Some versions: res.rsquared is already a float (often within R2)
        if isinstance(r2, (float, int, np.floating, np.integer)):
            return float(r2)
        # Some versions: dict-like
        if isinstance(r2, dict) and "within" in r2:
            return float(r2["within"])
    except Exception:
        pass
    return float("nan")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _norm_path(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_dir, p)

def _extract_year(col: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(col))
    return int(m.group(0)) if m else None

def wide_xlsx_to_long(path: str, value_name: str, y0: int, y1: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "hex_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "hex_id"})
    df["hex_id"] = df["hex_id"].astype(str)

    col2year = {}
    for c in df.columns:
        if c == "hex_id":
            continue
        y = _extract_year(c)
        if y is not None and y0 <= y <= y1:
            col2year[c] = y
    if not col2year:
        raise ValueError(f"No year columns found in: {path}")

    keep = ["hex_id"] + list(col2year.keys())
    df = df[keep].copy()
    out = df.melt(id_vars=["hex_id"], var_name="col", value_name=value_name)
    out["year"] = out["col"].map(col2year).astype(int)
    out = out.drop(columns=["col"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out

def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def zscore_inplace(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for c in cols:
        x = df[c].astype(float)
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd == 0:
            df[c] = x - mu
            stats[c] = (mu, 0.0)
        else:
            df[c] = (x - mu) / sd
            stats[c] = (mu, sd)
    return stats

def center_inplace(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    stats = {}
    for c in cols:
        x = df[c].astype(float)
        mu = float(np.nanmean(x))
        df[c] = x - mu
        stats[c] = mu
    return stats


def double_demean(panel: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Double-demean columns by hexagon and year, preserving the grand mean."""
    out = panel.copy()
    overall = {c: out[c].mean() for c in cols}
    by_hex = out.groupby("hex_id")[cols].transform("mean")
    by_year = out.groupby("year")[cols].transform("mean")
    for c in cols:
        out[c] = out[c] - by_hex[c] - by_year[c] + overall[c]
    return out


def pooled_vpd_resid(panel: pd.DataFrame,
                     include_swnet: bool = True,
                     do_double_demean: bool = True) -> pd.Series:
    """
    Construct residualized atmospheric demand from pooled OLS:
      VPD90 ~ Tmean90 (+ SWnet90)
    If do_double_demean=True, VPD90, Tmean90 and SWnet90 are first double-demeaned
    by hexagon and year. The returned series is aligned to panel.index.
    """
    need = ["VPD90", "Tmean90"]
    if include_swnet and "SWnet90" in panel.columns:
        need.append("SWnet90")

    df = panel[["hex_id", "year"] + need].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        return pd.Series(np.nan, index=panel.index, name="VPD_resid")

    if do_double_demean:
        df = double_demean(df, need)

    y = df["VPD90"].to_numpy(dtype=float)
    x_cols = ["Tmean90"] + (["SWnet90"] if (include_swnet and "SWnet90" in df.columns) else [])
    X = df[x_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), X])

    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)

    resid = y - X @ beta
    key = df["hex_id"].astype(str) + "__" + df["year"].astype(str)
    resid_map = pd.Series(resid, index=key)
    full_key = panel["hex_id"].astype(str) + "__" + panel["year"].astype(str)
    return full_key.map(resid_map).rename("VPD_resid")

def slope_by_hex(df: pd.DataFrame, ycol: str) -> pd.Series:
    # df: columns [hex_id, year, ycol]
    def _s(sub):
        y = sub[ycol].to_numpy(float)
        x = sub["year"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return np.nan
        xx, yy = x[m], y[m]
        vx = np.var(xx, ddof=0)
        if vx == 0:
            return np.nan
        cov = np.mean((xx - xx.mean()) * (yy - yy.mean()))
        return cov / vx

    return df.groupby("hex_id", sort=False).apply(_s)

def fit_panel(df: pd.DataFrame, y: str, X: List[str], time_effects: bool) -> PanelOLS:
    # linearmodels expects MultiIndex (entity,time)
    d = df.set_index(["hex_id", "year"]).sort_index()
    yv = d[y]
    Xv = d[X]
    mod = PanelOLS(yv, Xv, entity_effects=True, time_effects=time_effects, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res

def get_fitted(res) -> pd.Series:
    # aligned with panel index
    fv = res.fitted_values
    if isinstance(fv, pd.DataFrame):
        fv = fv.iloc[:, 0]
    return fv

def get_effect_proxy_time(res) -> pd.Series:
    """
    Proxy for gamma_t (time FE) from estimated_effects:
      estimated_effects(i,t) = alpha_i + gamma_t (up to normalization)
    Use mean over entities for each time t as proxy (constant shifts don't affect trends).
    """
    eff = res.estimated_effects
    if isinstance(eff, pd.DataFrame):
        eff = eff.iloc[:, 0]
    eff = eff.reset_index()
    # columns likely: ['hex_id','year','estimated_effects']
    valcol = [c for c in eff.columns if c not in ("hex_id", "year")]
    if not valcol:
        raise RuntimeError("Cannot parse estimated_effects output.")
    valcol = valcol[0]
    gamma = eff.groupby("year")[valcol].mean()
    gamma.name = "gamma_proxy"
    return gamma

def ols_co2_projection(y: np.ndarray, x: np.ndarray, include_quadratic: bool = True) -> Tuple[Dict[str, float], float, np.ndarray]:
    """
    Regress y on XCO2 or XCO2 + XCO2^2.
    Returns (coefficients, R2, fitted values aligned to input x). Missing inputs yield NaN fitted values.
    """
    y0 = np.asarray(y, dtype=float)
    x0 = np.asarray(x, dtype=float)
    m = np.isfinite(y0) & np.isfinite(x0)
    fitted = np.full_like(y0, np.nan, dtype=float)
    if m.sum() < 2:
        return {"intercept": np.nan, "xco2": np.nan, "xco2_sq": np.nan}, np.nan, fitted

    yy = y0[m]
    xx = x0[m]
    cols = [np.ones(len(xx)), xx]
    names = ["intercept", "xco2"]
    if include_quadratic:
        cols.append(xx ** 2)
        names.append("xco2_sq")
    X = np.column_stack(cols)

    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    yhat = X @ beta
    ssr = np.sum((yy - yhat) ** 2)
    sst = np.sum((yy - yy.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    fitted[m] = yhat

    coef = {name: float(val) for name, val in zip(names, beta)}
    if "xco2_sq" not in coef:
        coef["xco2_sq"] = np.nan
    return coef, float(r2), fitted

# =============================================================================
# Pipeline per TAG
# =============================================================================
@dataclass
class TagConfig:
    tag: str
    base_dir: str
    vegfile: str

def run_one(cfg: TagConfig) -> None:
    tag = cfg.tag
    base = cfg.base_dir

    # ---- read y and drivers ----
    y_long = wide_xlsx_to_long(_norm_path(base, cfg.vegfile), "veg_frac", YEAR_MIN, YEAR_MAX)

    p90 = wide_xlsx_to_long(_norm_path(base, FILE_P90), "P90", YEAR_MIN, YEAR_MAX)
    vpd = wide_xlsx_to_long(_norm_path(base, FILE_VPD90), "VPD90", YEAR_MIN, YEAR_MAX)
    tmean = wide_xlsx_to_long(_norm_path(base, FILE_TMEAN), "Tmean90", YEAR_MIN, YEAR_MAX)
    sm = wide_xlsx_to_long(_norm_path(base, FILE_SM90), "SM90", YEAR_MIN, YEAR_MAX)
    xco2 = wide_xlsx_to_long(_norm_path(base, FILE_XCO2), "XCO2", YEAR_MIN, YEAR_MAX)

    dfs = [y_long, p90, vpd, tmean, sm, xco2]
    if INCLUDE_ET and os.path.exists(_norm_path(base, FILE_ET90)):
        dfs.append(wide_xlsx_to_long(_norm_path(base, FILE_ET90), "ET90", YEAR_MIN, YEAR_MAX))
    if INCLUDE_SWNET and os.path.exists(_norm_path(base, FILE_SWNET90)):
        dfs.append(wide_xlsx_to_long(_norm_path(base, FILE_SWNET90), "SWnet90", YEAR_MIN, YEAR_MAX))

    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on=["hex_id", "year"], how="inner")

    # ---- construct logit y ----
    df["y_logit"] = logit(df["veg_frac"].to_numpy(float), LOGIT_EPS)

    # ---- residualized atmospheric demand, matching the main attribution model ----
    df["VPD_resid"] = pooled_vpd_resid(
        df,
        include_swnet=bool(VPD_RESID_INCLUDE_SWNET),
        do_double_demean=bool(VPD_RESID_DOUBLE_DEMEAN),
    )

    # ---- design matrix ----
    X_cols = ["P90", "VPD_resid", "Tmean90", "SM90", "XCO2"]
    if INCLUDE_ET and "ET90" in df.columns:
        X_cols.append("ET90")
    if INCLUDE_SWNET and "SWnet90" in df.columns:
        X_cols.append("SWnet90")
    if INCLUDE_INTERACTION:
        df["P90xVPDresid"] = df["P90"] * df["VPD_resid"]
        X_cols.append("P90xVPDresid")

    # Time-replacement diagnostic term. It is not included in X_cols because
    # it replaces XCO2 only in the CLIM_PLUS_TIME diagnostic specification.
    df["year_c"] = df["year"].astype(float) - float(np.nanmean(df["year"].astype(float)))

    # Standardize on regression sample
    zstats = {}
    cstats = {}
    if Z_SCORE_X:
        zstats = zscore_inplace(df, [c for c in X_cols if c != "XCO2"])  # climate
        # XCO2 also z-score (recommended for stability)
        zstats.update(zscore_inplace(df, ["XCO2"]))
        # year_c is z-scored separately for the time-replacement diagnostic
        zstats.update(zscore_inplace(df, ["year_c"]))
    else:
        if CENTER_ONLY:
            cstats = center_inplace(df, X_cols + ["year_c"])

    # Split X sets
    X_clim = [c for c in X_cols if c != "XCO2"]          # climate only
    X_climco2 = X_cols[:]                                # climate + XCO2
    X_climtime = X_clim + ["year_c"]                    # climate + centered linear time

    # ---- Fit models ----
    # NAT0: no year FE
    res_nat0_clim = fit_panel(df, "y_logit", X_clim, time_effects=False)
    res_nat0_co2  = fit_panel(df, "y_logit", X_climco2, time_effects=False)
    res_time      = fit_panel(df, "y_logit", X_climtime, time_effects=False)

    # NAT1: with year FE (diagnostic upper bound)
    res_nat1_clim = fit_panel(df, "y_logit", X_clim, time_effects=True)
    res_nat1_co2  = fit_panel(df, "y_logit", X_climco2, time_effects=True)

    # ---- Fitted series ----
    idx = df.set_index(["hex_id", "year"]).sort_index().index
    y = df.set_index(["hex_id", "year"]).sort_index()["y_logit"]

    yhat_nat0_clim = get_fitted(res_nat0_clim).reindex(idx)
    yhat_nat0_co2  = get_fitted(res_nat0_co2).reindex(idx)
    yhat_time      = get_fitted(res_time).reindex(idx)

    yhat_nat1_clim = get_fitted(res_nat1_clim).reindex(idx)
    yhat_nat1_co2  = get_fitted(res_nat1_co2).reindex(idx)

    # ---- NAT2 (from NAT1_CLIM): project time component onto CO2/background trajectory ----
    gamma_proxy = get_effect_proxy_time(res_nat1_clim)  # index=year
    # CO2_mean per year from the SAME regression sample (hex-year)
    co2_mean = df.groupby("year")["XCO2"].mean()
    coef_proj, r2, gamma_hat_vals = ols_co2_projection(
        gamma_proxy.values,
        co2_mean.reindex(gamma_proxy.index).values,
        include_quadratic=bool(NAT2_CO2_QUADRATIC),
    )
    gamma_hat_co2 = pd.Series(gamma_hat_vals, index=gamma_proxy.index, name="gamma_hat_co2")

    # Build NAT2 prediction by swapping time component in NAT1 fitted
    # yhat_nat2 = yhat_nat1 - gamma_proxy(year) + gamma_hat_co2(year)
    years = idx.get_level_values("year")
    yhat_nat2 = yhat_nat1_clim.copy()
    yhat_nat2 = yhat_nat2 - years.map(gamma_proxy).to_numpy() + years.map(gamma_hat_co2).to_numpy()

    # ---- Trend attribution per hex (on logit scale) ----
    tmp = pd.DataFrame({
        "hex_id": idx.get_level_values(0).astype(str),
        "year": years.astype(int),
        "y_obs": y.values,
        "yhat_nat0_clim": yhat_nat0_clim.values,
        "yhat_nat0_co2": yhat_nat0_co2.values,
        "yhat_time": yhat_time.values,
        "yhat_nat1_clim": yhat_nat1_clim.values,
        "yhat_nat1_co2": yhat_nat1_co2.values,
        "yhat_nat2": yhat_nat2.values,
    })

    # slopes
    beta_obs = slope_by_hex(tmp, "y_obs")

    def slopes(yhat_col: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        beta_nat = slope_by_hex(tmp, yhat_col)
        tmp_res = tmp.copy()
        tmp_res["resid"] = tmp_res["y_obs"] - tmp_res[yhat_col]
        beta_res = slope_by_hex(tmp_res, "resid")
        share_nat = beta_nat.abs() / (beta_nat.abs() + beta_res.abs())
        dominance = (beta_nat.abs() >= beta_res.abs()).astype(int)
        return beta_nat, beta_res, share_nat, dominance, beta_nat + beta_res

    nat0_clim = slopes("yhat_nat0_clim")
    nat0_co2  = slopes("yhat_nat0_co2")
    nat_time  = slopes("yhat_time")
    nat1_clim = slopes("yhat_nat1_clim")
    nat1_co2  = slopes("yhat_nat1_co2")
    nat2      = slopes("yhat_nat2")

    out_hex = pd.DataFrame({
        "hex_id": beta_obs.index.astype(str),
        "beta_obs": beta_obs.values,

        "beta_nat0_clim": nat0_clim[0].values,
        "beta_res0_clim": nat0_clim[1].values,
        "share_nat0_clim": nat0_clim[2].values,
        "dom_nat0_clim": nat0_clim[3].values,

        "beta_nat0_co2": nat0_co2[0].values,
        "beta_res0_co2": nat0_co2[1].values,
        "share_nat0_co2": nat0_co2[2].values,
        "dom_nat0_co2": nat0_co2[3].values,

        "beta_nat_time": nat_time[0].values,
        "beta_res_time": nat_time[1].values,
        "share_nat_time": nat_time[2].values,
        "dom_nat_time": nat_time[3].values,

        "beta_nat1_clim": nat1_clim[0].values,
        "beta_res1_clim": nat1_clim[1].values,
        "share_nat1_clim": nat1_clim[2].values,
        "dom_nat1_clim": nat1_clim[3].values,

        "beta_nat1_co2": nat1_co2[0].values,
        "beta_res1_co2": nat1_co2[1].values,
        "share_nat1_co2": nat1_co2[2].values,
        "dom_nat1_co2": nat1_co2[3].values,

        "beta_nat2": nat2[0].values,
        "beta_res2": nat2[1].values,
        "share_nat2": nat2[2].values,
        "dom_nat2": nat2[3].values,
    })

    # ---- Summaries ----
    def summary_block(prefix: str) -> dict:
        s = out_hex[f"share_{prefix}"]
        d = out_hex[f"dom_{prefix}"]
        return {
            "n_hex": int(out_hex.shape[0]),
            "share_median": float(np.nanmedian(s)),
            "share_p25": float(np.nanpercentile(s, 25)),
            "share_p75": float(np.nanpercentile(s, 75)),
            "natural_dominant_pct": float(np.nanmean(d)),
        }

    summ = {
        "tag": tag,
        "years": [YEAR_MIN, YEAR_MAX],
        "spec": {
            "entity_effects": True,
            "NAT0_time_effects": False,
            "NAT1_time_effects": True,
            "NAT2_time_component": "from NAT1_CLIM time proxy projected onto XCO2_mean(year) + XCO2_mean(year)^2" if NAT2_CO2_QUADRATIC else "from NAT1_CLIM time proxy projected onto XCO2_mean(year)",
            "vpd_used": "VPD_resid",
            "vpd_resid_include_swnet": VPD_RESID_INCLUDE_SWNET,
            "vpd_resid_double_demean": VPD_RESID_DOUBLE_DEMEAN,
            "CLIM_PLUS_TIME": "local climate predictors plus centered linear year term, no XCO2, no unrestricted year FE",
            "interaction": "P90xVPDresid" if INCLUDE_INTERACTION else None,
            "z_score": Z_SCORE_X,
        },
        "NAT0_CLIM": summary_block("nat0_clim"),
        "NAT0_CO2":  summary_block("nat0_co2"),
        "CLIM_PLUS_TIME": summary_block("nat_time"),
        "NAT1_CLIM": summary_block("nat1_clim"),
        "NAT1_CO2":  summary_block("nat1_co2"),
        "NAT2":      summary_block("nat2"),
        "NAT2_timeFE_projection": {
            "coefficients": coef_proj,
            "r2": r2,
            "quadratic": NAT2_CO2_QUADRATIC,
            "note": "gamma_proxy(year) is mean estimated_effects over entities; constant shifts do not affect trend attribution."
        },
        "model_fit": {
            "NAT0_CLIM_r2_within": safe_r2_within(res_nat0_clim),
            "NAT0_CO2_r2_within": safe_r2_within(res_nat0_co2),
            "CLIM_PLUS_TIME_r2_within": safe_r2_within(res_time),
            "NAT1_CLIM_r2_within": safe_r2_within(res_nat1_clim),
            "NAT1_CO2_r2_within": safe_r2_within(res_nat1_co2),
        }
    }

    # ---- Write outputs ----
    ensure_dir(OUTDIR)

    # timeFE projection table (for a simple supplement figure)
    proj = pd.DataFrame({
        "year": gamma_proxy.index.astype(int),
        "gamma_time_proxy": gamma_proxy.values,
        "XCO2_mean_year": co2_mean.reindex(gamma_proxy.index).values,
        "XCO2_mean_year_sq": co2_mean.reindex(gamma_proxy.index).values ** 2,
        "gamma_hat_co2": gamma_hat_co2.reindex(gamma_proxy.index).values,
    })
    proj.to_csv(os.path.join(OUTDIR, f"NAT2_timeFE_projection_{tag}.csv"), index=False)

    # by-hex table
    out_xlsx = os.path.join(OUTDIR, f"NAT_diagnostics_by_hex_{tag}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out_hex.to_excel(w, sheet_name="by_hex", index=False)
        proj.to_excel(w, sheet_name="NAT2_time_projection", index=False)

    summary_rows = []
    for key in ["NAT0_CLIM", "NAT0_CO2", "CLIM_PLUS_TIME", "NAT1_CLIM", "NAT1_CO2", "NAT2"]:
        row = {"specification": key}
        row.update(summ[key])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTDIR, f"NAT_diagnostics_summary_table_{tag}.csv"),
        index=False, encoding="utf-8"
    )

    safe_json(summ, os.path.join(OUTDIR, f"NAT_diagnostics_summary_{tag}.json"))

    # also dump FE summaries (optional quick read)
    with open(os.path.join(OUTDIR, f"FE_summary_NAT0_CLIM_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(str(res_nat0_clim.summary))
    with open(os.path.join(OUTDIR, f"FE_summary_CLIM_PLUS_TIME_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(str(res_time.summary))
    with open(os.path.join(OUTDIR, f"FE_summary_NAT1_CLIM_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(str(res_nat1_clim.summary))

    print(f"[{tag}] done -> {OUTDIR}")

def main():
    ensure_dir(OUTDIR)
    run_one(TagConfig("UP", DIR_UP, FILE_VEGFRAC_UP))
    # DOWN is optional; set as needed
    # run_one(TagConfig("DOWN", DIR_DOWN, FILE_VEGFRAC_DOWN))

if __name__ == "__main__":
    main()
