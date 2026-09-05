#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sem_attribution_with_co2_v1.py

Goal
----
A Python-only "mechanism-first" attribution workflow:
1) Two-way fixed-effect (hex FE + year FE) panel regression -> extract common time component.
2) Explain that common time component with CO2 (and optional quadratic term).
3) SEM on double-demeaned anomalies (within-hex and within-year) to quantify direct/indirect mechanistic pathways.

Dependencies
------------
pip install pandas numpy openpyxl linearmodels statsmodels semopy
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _scalar(x) -> float:
    """Convert semopy stats entries (scalar or 1-element Series/array) to float."""
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        _np = None
        _pd = None
    if x is None:
        return float('nan')
    # pandas Series/DataFrame
    if _pd is not None and isinstance(x, (_pd.Series, _pd.DataFrame)):
        try:
            return float(_np.asarray(x).ravel()[0])
        except Exception:
            try:
                return float(x.iloc[0])
            except Exception:
                return float('nan')
    # array-like
    if _np is not None and isinstance(x, (list, tuple, _np.ndarray)):
        try:
            return float(_np.asarray(x).ravel()[0])
        except Exception:
            return float('nan')
    # scalar
    try:
        return float(x)
    except Exception:
        return float('nan')
try:
    from linearmodels.panel import PanelOLS
except Exception as e:
    raise ImportError(
        "Missing dependency: linearmodels. Install with:\n"
        "  pip install linearmodels\n"
        f"Original error: {e}"
    )

try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError(
        "Missing dependency: statsmodels. Install with:\n"
        "  pip install statsmodels\n"
        f"Original error: {e}"
    )

try:
    from semopy import Model
    from semopy.inspector import inspect
    from semopy.stats import calc_stats
except Exception as e:
    raise ImportError(
        "Missing dependency: semopy. Install with:\n"
        "  pip install semopy\n"
        f"Original error: {e}"
    )


def _extract_time_effects_from_panelols(res, years_hint: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Extract time (year) fixed effects from PanelOLS results robustly.

    Handles a range of linearmodels versions and index types (int, str, Period, datetime, MultiIndex).
    Returns a DataFrame with columns: ['year', 'time_effect'].
    """
    candidates = ["time_effects", "_time_effect", "_time_effects", "time_effect"]
    obj = None
    for name in candidates:
        if hasattr(res, name):
            obj = getattr(res, name)
            if obj is not None:
                break

    if obj is None:
        if hasattr(res, "estimated_effects"):
            obj = getattr(res, "estimated_effects")
        else:
            raise AttributeError(
                "Could not extract time effects from PanelOLS results. "
                "Tried attributes: time_effects, _time_effect, estimated_effects."
            )

    # Normalize to Series with a time-like index
    if isinstance(obj, pd.DataFrame):
        if "time" in obj.columns:
            ser = obj["time"]
        else:
            ser = obj.mean(axis=1)
    elif isinstance(obj, pd.Series):
        ser = obj
    else:
        ser = pd.Series(np.asarray(obj))

    # If MultiIndex with ('entity','time'), take the time level
    if isinstance(ser.index, pd.MultiIndex):
        if "time" in ser.index.names:
            time_idx = ser.index.get_level_values("time")
        else:
            time_idx = ser.index.get_level_values(-1)
        out = pd.DataFrame({"year": time_idx, "time_effect": ser.values})
    else:
        out = pd.DataFrame({"year": ser.index, "time_effect": ser.values})

    # Coerce year
    yr = pd.to_numeric(out["year"], errors="coerce")
    if yr.isna().all():
        yr_dt = pd.to_datetime(out["year"], errors="coerce")
        yr = yr_dt.dt.year

    out["year"] = yr
    out["time_effect"] = pd.to_numeric(out["time_effect"], errors="coerce")

    # Drop bad rows BEFORE int cast (fixes IntCastingNaNError)
    out = out.dropna(subset=["year", "time_effect"]).copy()
    out["year"] = out["year"].astype(int)

    # Mean-center for readability
    out["time_effect"] = out["time_effect"] - out["time_effect"].mean()

    # Deduplicate (if any) by averaging
    out = out.groupby("year", as_index=False)["time_effect"].mean()
    out = out.sort_values("year").reset_index(drop=True)

    if years_hint is not None and len(years_hint) > 0:
        out = out[out["year"].isin(years_hint)].copy().sort_values("year").reset_index(drop=True)

    return out


def _compute_time_effects_from_residuals(fe_df_indexed: pd.DataFrame, x_cols: List[str], beta: pd.Series) -> pd.DataFrame:
    """
    Fallback method to obtain time (year) fixed effects when linearmodels does not expose them.

    Model: y_it = X_it * beta + a_i + g_t + e_it
    Let r_it = y_it - X_it * beta. Then under sum-to-zero constraints:
      g_t = mean_i(r_it) - mean(r_it)
    Works for unbalanced panels using available observations.

    Returns DataFrame: ['year', 'time_effect'] (centered).
    """
    if not isinstance(fe_df_indexed.index, pd.MultiIndex):
        raise ValueError("fe_df_indexed must have MultiIndex (hex_id, year).")
    if "y_logit" not in fe_df_indexed.columns:
        raise ValueError("fe_df_indexed must contain y_logit.")

    y = fe_df_indexed["y_logit"].astype(float)
    X = fe_df_indexed[x_cols].astype(float)
    # align beta
    beta = beta.reindex(x_cols)
    xb = pd.Series(X.values @ beta.values, index=X.index)
    r = y - xb

    # overall mean (weighted by obs count)
    rbar = float(r.mean())
    years = fe_df_indexed.index.get_level_values(-1)
    g = r.groupby(years).mean() - rbar
    out = g.reset_index()
    out.columns = ["year", "time_effect"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["time_effect"] = pd.to_numeric(out["time_effect"], errors="coerce")
    out = out.dropna(subset=["year", "time_effect"]).sort_values("year")
    return out


# =========================
# CONFIG (EDIT THESE)
# =========================
NDVI_XLSX    = r"NDVI_trend_hex_100_up.xlsx"   # or *_down.xlsx
PRECIP_XLSX  = r"zonal_tables_parallel_up_V2\W90_Precip.xlsx"           # P90 supply proxy
VPD_XLSX     = r"zonal_tables_parallel_up_V2\VPD90_mean.xlsx"           # VPD (kPa), computed from T2m & Td2m
TMEAN_XLSX   = r"zonal_tables_parallel_up_V2\T2m90_mean.xlsx"           # Tmean-like thermal condition (°C)
SM_L1_XLSX   = r"zonal_tables_parallel_up_V2\SM_L1_90_mean.xlsx"        # optional (surface layer)
SM_ROOT_XLSX = r"zonal_tables_parallel_up_V2\SM_root_90_mean.xlsx"      # root-zone (layers 1-3 mean)
ET_XLSX      = r"zonal_tables_parallel_up_V2\ET90_sum_new.xlsx"             # optional (mm); total_evaporation_sum (sign fixed)
SWNET_XLSX   = r"zonal_tables_parallel_up_V2\SWnet90_mean.xlsx"         # optional (W/m2); from SSR sum

# Optional: specify sheet names (None => first sheet)
NDVI_SHEET    = None
PRECIP_SHEET  = None
VPD_SHEET     = None
TMEAN_SHEET   = None
SM_L1_SHEET   = None
SM_ROOT_SHEET = None
ET_SHEET      = None
SWNET_SHEET   = None

# Choose which soil moisture proxy to use in FE/SEM
SM_CHOICE = "l1"  # same soil-moisture proxy as main SEM

# Construct VPD_resid for SEM (recommended): VPD_resid = VPD - E[VPD | Tmean (+ SWnet)]
VPD_RESID_INCLUDE_SWNET = True

ONLY_SIGNIFICANT = True
SLOPE_SIGN = "pos"  # "pos" / "neg" / "both"

SLOPE_SIGN = "pos"  # "pos" / "neg" / "both"

YEAR_MIN = 2000
YEAR_MAX = 2022
LOGIT_EPS = 1e-4

FE_X_COLS_CORE = ["P90", "VPD90", "SM90"]  # VPD90 auto-replaced by T90 when USE_T
INCLUDE_ET_IN_FE = True
INCLUDE_SWNET_IN_FE = True

DOUBLE_DEMEAN_FOR_SEM = True
Z_SCORE_FOR_SEM = True

CO2_PATH = r"zonal_tables_parallel_up_V2/CO2_annual_tif.xlsx"  # .csv or .xlsx with year + co2(ppm)
CO2_QUADRATIC = True

OUTPUT_DIR = r"outputs/sem/sensitivity_without_direct_swnet"
WRITE_ANOMALY_SAMPLE = True
ANOMALY_SAMPLE_N = 200000

# =========================
# v10.1a sensitivity toggles
# =========================
# 1) Include thermal condition in SM equation (Tmean -> SM90)
INCLUDE_TMEAN_IN_SM = True

# 2) Include thermal condition in ET equation (Tmean -> ET90)
INCLUDE_TMEAN_IN_ET = True

# 3) Toggle SWnet direct effect on y (SWnet90 -> y_logit).
#    If False, SWnet still influences y indirectly via ET (if ET included) and via covariances.
INCLUDE_SWNET_DIRECT_IN_Y = False

# 4) Toggle P90 direct effect on y (keep True by default).
INCLUDE_P90_DIRECT_IN_Y = True
# =========================
# Helpers
# =========================

def _safe_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_wide_xlsx(path: str, sheet_name=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = pd.read_excel(p, sheet_name=sheet_name)
    if isinstance(obj, dict):
        first_name = list(obj.keys())[0]
        if sheet_name is None:
            print(f"[INFO] {p.name}: multiple sheets; using first: {first_name!r}")
            return obj[first_name]
        if sheet_name in obj:
            return obj[sheet_name]
        print(f"[WARN] {p.name}: sheet {sheet_name!r} not found; using first: {first_name!r}")
        return obj[first_name]
    return obj


def _detect_hex_id_col(df: pd.DataFrame) -> str:
    candidates = ["hex_id", "HEX_ID", "h3_id", "H3_ID", "id", "ID", "hex", "HEX"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find hex id column. Tried: {candidates}")


def _extract_year_from_label(label) -> Optional[int]:
    if isinstance(label, (int, np.integer)):
        y = int(label)
        return y if YEAR_MIN <= y <= YEAR_MAX else None
    if isinstance(label, (float, np.floating)):
        yf = float(label)
        y = int(round(yf))
        if abs(yf - y) < 1e-6 and YEAR_MIN <= y <= YEAR_MAX:
            return y
        return None
    s = str(label).strip()
    if s.isdigit():
        y = int(s)
        return y if YEAR_MIN <= y <= YEAR_MAX else None
    m = re.search(r"(20\d{2})", s)
    if not m:
        return None
    y = int(m.group(1))
    return y if YEAR_MIN <= y <= YEAR_MAX else None


def _year_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if _extract_year_from_label(c) is not None]
    cols.sort(key=lambda c: _extract_year_from_label(c))
    return cols


def _to_long(df: pd.DataFrame, value_name: str, hex_id_col: str) -> pd.DataFrame:
    ycols = _year_cols(df)
    if not ycols:
        raise ValueError(
            f"No year columns detected for '{value_name}'. Expected {YEAR_MIN}-{YEAR_MAX}. "
            f"First 40 columns: {list(df.columns)[:40]}"
        )
    long = df[[hex_id_col] + ycols].melt(id_vars=[hex_id_col], var_name="year", value_name=value_name)
    long["year"] = long["year"].apply(lambda x: _extract_year_from_label(x)).astype(int)
    return long


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _apply_slope_sign_filter(ndvi: pd.DataFrame) -> pd.DataFrame:
    if SLOPE_SIGN not in ("pos", "neg", "both"):
        raise ValueError('SLOPE_SIGN must be one of: "pos", "neg", "both"')
    if SLOPE_SIGN == "both":
        return ndvi
    if "sen_slope" not in ndvi.columns:
        print("[WARN] NDVI has no 'sen_slope'; skipping sign filter.")
        return ndvi
    s = pd.to_numeric(ndvi["sen_slope"], errors="coerce")
    return ndvi[s > 0].copy() if SLOPE_SIGN == "pos" else ndvi[s < 0].copy()


def _zscore(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    out = df.copy()
    stats = {}
    for c in cols:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        stats[c] = (mu, sd)
        if not np.isfinite(sd) or sd == 0:
            out[c] = 0.0
        else:
            out[c] = (out[c] - mu) / sd
    return out, stats


def _double_demean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    overall = {c: out[c].mean() for c in cols}
    by_hex = out.groupby("hex_id")[cols].transform("mean")
    by_year = out.groupby("year")[cols].transform("mean")
    for c in cols:
        out[c] = out[c] - by_hex[c] - by_year[c] + overall[c]
    return out


def _read_co2(path: str) -> pd.DataFrame:
    """
    Read CO2 annual series.

    Supported formats:
    A) Long format: columns include a year column (year/Year/...) and a CO2 column (co2/ppm/...).
    B) Wide format: one row or multiple rows with year-encoded columns, e.g.:
         NOAA_CO2_2000, NOAA_CO2_2001, ... NOAA_CO2_2022
       In this case we melt to long and (if multiple rows) take the mean CO2 across rows per year.

    Output: DataFrame with columns ['year', 'co2_ppm'] for YEAR_MIN..YEAR_MAX.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CO2 file not found: {path}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    elif p.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(p)
    else:
        raise ValueError("CO2_PATH must be .csv or .xlsx")

    # -------- Try LONG format first --------
    # Detect year column
    ycol = None
    for c in ["year", "Year", "YEAR", "yr", "YR"]:
        if c in df.columns:
            ycol = c
            break
    if ycol is None:
        for c in df.columns:
            if re.search(r"year", str(c), flags=re.IGNORECASE):
                ycol = c
                break

    # Detect CO2 column
    ccol = None
    for c in ["co2_ppm", "CO2_ppm", "co2", "CO2", "ppm", "PPM"]:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        for c in df.columns:
            if re.search(r"co2|ppm", str(c), flags=re.IGNORECASE):
                ccol = c
                break

    if ycol is not None and ccol is not None:
        out = df[[ycol, ccol]].rename(columns={ycol: "year", ccol: "co2_ppm"}).copy()
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out = out.dropna(subset=["year"])
        out["year"] = out["year"].astype(int)
        out["co2_ppm"] = pd.to_numeric(out["co2_ppm"], errors="coerce")
        out = out.dropna(subset=["co2_ppm"])
        out = out[(out["year"] >= YEAR_MIN) & (out["year"] <= YEAR_MAX)].sort_values("year")
        if out.empty:
            raise ValueError("CO2 series empty after filtering (long format).")
        return out

    # -------- Fall back to WIDE format --------
    # Detect year-encoded columns
    year_cols = [c for c in df.columns if _extract_year_from_label(c) is not None]
    if not year_cols:
        raise ValueError(
            f"Cannot detect year column (long) nor year-encoded columns (wide) in CO2 file. "
            f"Columns: {list(df.columns)}"
        )
    year_cols.sort(key=lambda c: _extract_year_from_label(c))

    # Melt: id_vars = non-year columns (may include hex_id)
    id_vars = [c for c in df.columns if c not in year_cols]
    long = df[id_vars + year_cols].melt(id_vars=id_vars, var_name="year", value_name="co2_ppm")
    long["year"] = long["year"].apply(lambda x: _extract_year_from_label(x))
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["co2_ppm"] = pd.to_numeric(long["co2_ppm"], errors="coerce")
    long = long.dropna(subset=["year", "co2_ppm"])
    long["year"] = long["year"].astype(int)
    long = long[(long["year"] >= YEAR_MIN) & (long["year"] <= YEAR_MAX)]

    if long.empty:
        raise ValueError("CO2 series empty after melting wide format and filtering.")

    # If multiple rows (e.g., repeated per hex), average to a single global series
    out = long.groupby("year", as_index=False)["co2_ppm"].mean().sort_values("year")
    return out


# =========================
# SEM spec
# =========================
def build_sem_model_text(use_et: bool,
                         use_swnet: bool,
                         include_p90_direct: bool = True,
                         include_t_in_sm: bool = False,
                         include_t_in_et: bool = False,
                         include_swnet_direct_in_y: bool = True) -> str:
    """
    SEM structure (water-condition module + explicit thermal condition):

    Nodes:
      - P90        : supply proxy
      - VPD_resid  : atmospheric dryness net of thermal condition (and optionally SWnet)
      - Tmean      : thermal condition (T2m mean-like)
      - SM90       : storage / soil moisture proxy
      - ET90       : consumption / evapotranspiration (optional)
      - SWnet90    : energy input proxy (optional)
      - y_logit    : logit(veg_frac)

    Core directed paths:
      - SM90  ~ P90 + VPD_resid (+ Tmean optional)
      - ET90  ~ P90 + SM90 + VPD_resid (+ SWnet90 optional) (+ Tmean optional)
      - y_logit ~ [P90 optional] + SM90 + VPD_resid + Tmean + [ET90 optional] + [SWnet90 optional]

    Covariances:
      - P90 ~~ VPD_resid (shared circulation anomalies)
      - P90 ~~ Tmean     (background coupling)
      - If SWnet90 used: P90 ~~ SWnet90, Tmean ~~ SWnet90

    Residual covariance:
      - ET90 ~~ SM90 (recommended when ET is included)

    Note:
      VPD_resid is constructed prior to SEM as the residual from:
        VPD90 ~ Tmean (+ SWnet90),
      applied to the SEM anomaly space (after optional double-demeaning).
    """

    lines: List[str] = []
    lines.append("# Regressions")

    sm_rhs = ["a1*P90", "a2*VPD_resid"]
    if include_t_in_sm:
        sm_rhs.append("a3*Tmean")
    lines.append("SM90 ~ " + " + ".join(sm_rhs))

    if use_et:
        et_rhs = ["b1*P90", "b2*SM90", "b3*VPD_resid"]
        if use_swnet:
            et_rhs.append("b4*SWnet90")
        if include_t_in_et:
            et_rhs.append("b5*Tmean")
        lines.append("ET90 ~ " + " + ".join(et_rhs))

    y_rhs = []
    if include_p90_direct:
        y_rhs.append("c1*P90")
    y_rhs += ["c2*SM90", "c3*VPD_resid", "c4*Tmean"]
    if use_et:
        y_rhs.append("c5*ET90")
    if use_swnet and include_swnet_direct_in_y:
        y_rhs.append("c6*SWnet90")
    lines.append("y_logit ~ " + " + ".join(y_rhs))

    lines.append("")
    lines.append("# Covariances (shared background drivers)")
    lines.append("P90 ~~ VPD_resid")
    lines.append("P90 ~~ Tmean")
    if use_swnet:
        lines.append("P90 ~~ SWnet90")
        lines.append("Tmean ~~ SWnet90")

    lines.append("")
    lines.append("# Residual covariance (shared unobserved controls)")
    if use_et:
        lines.append("ET90 ~~ SM90")

    return "\n".join(lines)


def _fit_vpd_resid(df: pd.DataFrame,
                   vpd_col: str = "VPD90",
                   t_col: str = "Tmean",
                   swnet_col: str = "SWnet90",
                   include_swnet: bool = True) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit VPD90 ~ Tmean (+ SWnet90) and return the residual (VPD_resid) and fitted model.

    The function expects df to already be in the SEM sample space (after optional demeaning),
    and to have no missing values in the requested columns.
    """
    cols = [t_col]
    if include_swnet and (swnet_col in df.columns):
        cols.append(swnet_col)
    X = df[cols].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df[vpd_col].astype(float)
    res = sm.OLS(y, X, missing="drop").fit()
    pred = res.predict(X)
    resid = (y - pred).rename("VPD_resid")
    return resid, res


def compute_effects(par: pd.DataFrame,
                    use_et: bool,
                    use_swnet: bool,
                    include_p90_direct: bool = True,
                    include_t_in_sm: bool = False,
                    include_t_in_et: bool = False,
                    include_swnet_direct_in_y: bool = True) -> pd.DataFrame:
    """
    Compute selected direct/indirect effects from SEM parameters.

    Robust extraction via (lval, op, rval) tuples from semopy inspect() output.
    """

    if par is None or len(par) == 0:
        raise ValueError("SEM parameter table is empty.")

    par2 = par.copy()
    par2.columns = [str(c).strip().lower() for c in par2.columns]

    required = {"lval", "op", "rval"}
    if not required.issubset(set(par2.columns)):
        raise ValueError(f"semopy inspect() output missing required columns {required}. Got: {list(par2.columns)}")

    # estimate column
    if "estimate" in par2.columns:
        est_col = "estimate"
    elif "est" in par2.columns:
        est_col = "est"
    else:
        cand = [c for c in par2.columns if "estim" in c]
        est_col = cand[0] if cand else None
    if est_col is None:
        raise ValueError(f"Cannot find estimate column in semopy inspect() output. Columns: {list(par2.columns)}")

    def coef(lval: str, rval: str) -> float:
        mask = (par2["lval"].astype(str) == str(lval)) & (par2["op"].astype(str) == "~") & (par2["rval"].astype(str) == str(rval))
        if not mask.any():
            return float("nan")
        return float(par2.loc[mask, est_col].iloc[0])

    # SM equation
    a1 = coef("SM90", "P90")
    a2 = coef("SM90", "VPD_resid")
    a3 = coef("SM90", "Tmean") if include_t_in_sm else float("nan")

    # y equation
    c1 = coef("y_logit", "P90") if include_p90_direct else float("nan")
    c2 = coef("y_logit", "SM90")
    c3 = coef("y_logit", "VPD_resid")
    c4 = coef("y_logit", "Tmean")
    c5 = coef("y_logit", "ET90") if use_et else float("nan")
    c6 = coef("y_logit", "SWnet90") if (use_swnet and include_swnet_direct_in_y) else float("nan")

    # ET equation
    if use_et:
        b1 = coef("ET90", "P90")
        b2 = coef("ET90", "SM90")
        b3 = coef("ET90", "VPD_resid")
        b4 = coef("ET90", "SWnet90") if use_swnet else float("nan")
        b5 = coef("ET90", "Tmean") if include_t_in_et else float("nan")
    else:
        b1 = b2 = b3 = b4 = b5 = float("nan")

    rows: List[Tuple[str, str, float]] = []

    # Direct effects
    if include_p90_direct:
        rows.append(("direct", "P90 -> y", c1))
    rows.append(("direct", "SM90 -> y", c2))
    rows.append(("direct", "VPD_resid -> y", c3))
    rows.append(("direct", "Tmean -> y", c4))
    if use_et:
        rows.append(("direct", "ET90 -> y", c5))
    if use_swnet and include_swnet_direct_in_y:
        rows.append(("direct", "SWnet90 -> y", c6))

    # Indirect via SM
    rows.append(("indirect", "P90 -> SM90 -> y", a1 * c2))
    rows.append(("indirect", "VPD_resid -> SM90 -> y", a2 * c2))
    if include_t_in_sm:
        rows.append(("indirect", "Tmean -> SM90 -> y", a3 * c2))

    # Indirect via ET
    if use_et:
        rows.append(("indirect", "P90 -> ET90 -> y", b1 * c5))
        rows.append(("indirect", "SM90 -> ET90 -> y", b2 * c5))
        rows.append(("indirect", "VPD_resid -> ET90 -> y", b3 * c5))
        if use_swnet:
            rows.append(("indirect", "SWnet90 -> ET90 -> y", b4 * c5))
        if include_t_in_et:
            rows.append(("indirect", "Tmean -> ET90 -> y", b5 * c5))

        # Chains that traverse SM then ET
        rows.append(("indirect", "P90 -> SM90 -> ET90 -> y", a1 * b2 * c5))
        rows.append(("indirect", "VPD_resid -> SM90 -> ET90 -> y", a2 * b2 * c5))
        if include_t_in_sm:
            rows.append(("indirect", "Tmean -> SM90 -> ET90 -> y", a3 * b2 * c5))

    eff = pd.DataFrame(rows, columns=["type", "effect", "value"])

    # Totals for key drivers
    def _sum_effects(prefix: str) -> float:
        return float(eff.loc[eff["effect"].str.startswith(prefix), "value"].sum())

    if include_p90_direct:
        p_total = _sum_effects("P90 ->")
        eff = pd.concat([eff, pd.DataFrame([("total", "P90 total -> y", p_total)],
                                           columns=["type", "effect", "value"])], ignore_index=True)

    v_total = _sum_effects("VPD_resid ->")
    eff = pd.concat([eff, pd.DataFrame([("total", "VPD_resid total -> y", v_total)],
                                       columns=["type", "effect", "value"])], ignore_index=True)

    t_total = _sum_effects("Tmean ->")
    eff = pd.concat([eff, pd.DataFrame([("total", "Tmean total -> y", t_total)],
                                       columns=["type", "effect", "value"])], ignore_index=True)

    return eff

# =========================
# MainConvert semopy stats entries (scalar or 1-element Series/array) to float."""
    if x is None:
        return float('nan')
    if isinstance(x, (pd.Series, pd.DataFrame)):
        try:
            return float(np.asarray(x).ravel()[0])
        except Exception:
            return float('nan')
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return float(np.asarray(x).ravel()[0])
        except Exception:
            return float('nan')
    try:
        return float(x)
    except Exception:
        return float('nan')


# =========================
# Main
# =========================

def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    _safe_json({
                "inputs": {
            "NDVI_XLSX": NDVI_XLSX,
            "PRECIP_XLSX": PRECIP_XLSX,
            "VPD_XLSX": VPD_XLSX,
            "TMEAN_XLSX": TMEAN_XLSX,
            "SM_L1_XLSX": SM_L1_XLSX,
            "SM_ROOT_XLSX": SM_ROOT_XLSX,
            "ET_XLSX": ET_XLSX,
            "SWNET_XLSX": SWNET_XLSX,
            "CO2_PATH": CO2_PATH,
        },
        "options": {
            "ONLY_SIGNIFICANT": ONLY_SIGNIFICANT,
            "SLOPE_SIGN": SLOPE_SIGN,
            "SM_CHOICE": SM_CHOICE,
            "VPD_RESID_INCLUDE_SWNET": VPD_RESID_INCLUDE_SWNET,
            "YEAR_RANGE": [YEAR_MIN, YEAR_MAX],
            "LOGIT_EPS": LOGIT_EPS,
            "FE_X_COLS_CORE": FE_X_COLS_CORE,
            "INCLUDE_ET_IN_FE": INCLUDE_ET_IN_FE,
            "INCLUDE_SWNET_IN_FE": INCLUDE_SWNET_IN_FE,
            "DOUBLE_DEMEAN_FOR_SEM": DOUBLE_DEMEAN_FOR_SEM,
            "Z_SCORE_FOR_SEM": Z_SCORE_FOR_SEM,
            "CO2_QUADRATIC": CO2_QUADRATIC,
            "INCLUDE_TMEAN_IN_SM": INCLUDE_TMEAN_IN_SM,
            "INCLUDE_TMEAN_IN_ET": INCLUDE_TMEAN_IN_ET,
            "INCLUDE_SWNET_DIRECT_IN_Y": INCLUDE_SWNET_DIRECT_IN_Y,
            "INCLUDE_P90_DIRECT_IN_Y": INCLUDE_P90_DIRECT_IN_Y,
        }
    }, out_dir / "00_runinfo.json")

    # NDVI
    ndvi = _read_wide_xlsx(NDVI_XLSX, NDVI_SHEET)
    ndvi = ndvi.rename(columns={_detect_hex_id_col(ndvi): "hex_id"})
    if ONLY_SIGNIFICANT and "significant" in ndvi.columns:
        ndvi = ndvi[ndvi["significant"].fillna(0).astype(int) == 1].copy()
    ndvi = _apply_slope_sign_filter(ndvi)
    print(f"[INFO] NDVI rows after filters: {len(ndvi):,}")
    y_panel = _to_long(ndvi, "veg_frac", "hex_id")
    y_panel["y_logit"] = _logit(y_panel["veg_frac"].to_numpy(float))

    # Covariates
    precip = _read_wide_xlsx(PRECIP_XLSX, PRECIP_SHEET)
    precip = precip.rename(columns={_detect_hex_id_col(precip): "hex_id"})
    p_panel = _to_long(precip, "P90", "hex_id")

    
    # VPD (kPa)
    if VPD_XLSX is None:
        raise ValueError("VPD_XLSX is None. Provide VPD90_mean xlsx.")
    vpd = _read_wide_xlsx(VPD_XLSX, VPD_SHEET)
    vpd = vpd.rename(columns={_detect_hex_id_col(vpd): "hex_id"})
    vpd_panel = _to_long(vpd, "VPD90", "hex_id")

    # Thermal condition (Tmean-like, °C)
    if TMEAN_XLSX is None:
        raise ValueError("TMEAN_XLSX is None. Provide T2m90_mean (Tmean-like) xlsx.")
    tmean = _read_wide_xlsx(TMEAN_XLSX, TMEAN_SHEET)
    tmean = tmean.rename(columns={_detect_hex_id_col(tmean): "hex_id"})
    t_panel = _to_long(tmean, "Tmean", "hex_id")

    # Soil moisture proxy (choose root-zone or surface)
    if str(SM_CHOICE).lower() == "root":
        sm_path, sm_sheet = SM_ROOT_XLSX, SM_ROOT_SHEET
    elif str(SM_CHOICE).lower() == "l1":
        sm_path, sm_sheet = SM_L1_XLSX, SM_L1_SHEET
    else:
        raise ValueError(f"Unknown SM_CHOICE={SM_CHOICE}. Use 'root' or 'l1'.")

    if sm_path is None:
        raise ValueError(f"SM_CHOICE='{SM_CHOICE}' but the corresponding XLSX path is None.")

    smtab = _read_wide_xlsx(sm_path, sm_sheet)
    smtab = smtab.rename(columns={_detect_hex_id_col(smtab): "hex_id"})
    sm_panel = _to_long(smtab, "SM90", "hex_id")

    et_panel = None
    if ET_XLSX is not None:
        et = _read_wide_xlsx(ET_XLSX, ET_SHEET)
        et = et.rename(columns={_detect_hex_id_col(et): "hex_id"})
        et_panel = _to_long(et, "ET90", "hex_id")

    sw_panel = None
    if SWNET_XLSX is not None:
        sw = _read_wide_xlsx(SWNET_XLSX, SWNET_SHEET)
        sw = sw.rename(columns={_detect_hex_id_col(sw): "hex_id"})
        sw_panel = _to_long(sw, "SWnet90", "hex_id")

    panel = y_panel.merge(p_panel, on=["hex_id", "year"], how="inner")
    panel = panel.merge(vpd_panel, on=["hex_id", "year"], how="inner")
    panel = panel.merge(t_panel, on=["hex_id", "year"], how="inner")
    panel = panel.merge(sm_panel, on=["hex_id", "year"], how="inner")
    if et_panel is not None:
        panel = panel.merge(et_panel, on=["hex_id", "year"], how="inner")
    if sw_panel is not None:
        panel = panel.merge(sw_panel, on=["hex_id", "year"], how="inner")

    panel = panel[(panel["year"] >= YEAR_MIN) & (panel["year"] <= YEAR_MAX)].copy()

    _safe_json({
        "panel_rows_after_merge": int(len(panel)),
        "panel_hex_after_merge": int(panel["hex_id"].nunique()),
        "has_ET90": bool(et_panel is not None),
        "has_SWnet90": bool(sw_panel is not None),
    }, out_dir / "01_panel_sizes.json")

    # -------------------------
    # FE model -> year effects
    # -------------------------
    fe_x = list(FE_X_COLS_CORE)

    if INCLUDE_ET_IN_FE and ("ET90" in panel.columns):
        fe_x.append("ET90")
    if INCLUDE_SWNET_IN_FE and ("SWnet90" in panel.columns):
        fe_x.append("SWnet90")

    fe_df = panel[["hex_id", "year", "y_logit"] + fe_x].dropna().copy()
    print(f"[INFO] FE usable rows after dropna: {len(fe_df):,}")
    if fe_df.empty:
        raise ValueError("FE dataframe empty after dropna; check missingness & FE_X_COLS_CORE.")

    fe_df = fe_df.set_index(["hex_id", "year"]).sort_index()
    fe_mod = PanelOLS(fe_df["y_logit"], fe_df[fe_x], entity_effects=True, time_effects=True)
    fe_res = fe_mod.fit(cov_type="clustered", cluster_entity=True)

    # ---- Extract year fixed effects ----
    # linearmodels exposes different attributes across versions; if extraction fails,
    # compute time effects from residual decomposition: r_it = y_it - X_it*beta.
    try:
        try:
            years_hint = sorted(set(fe_df.index.get_level_values("year")))
        except Exception:
            years_hint = sorted(set(fe_df.index.get_level_values(-1)))

        te = _extract_time_effects_from_panelols(fe_res, years_hint=years_hint)

        # If the extracted series is suspiciously short, fall back.
        if len(te) < 3:
            raise ValueError(f"Extracted time effects too short (n={len(te)}).")
    except Exception as _e:
        print(f"[WARN] Using residual-based time effects fallback due to: {_e}")
        te = _compute_time_effects_from_residuals(fe_df, x_cols=fe_x, beta=fe_res.params)

    te.to_excel(out_dir / "02_fe_time_effects.xlsx", index=False)

    # -------------------------
    # CO2 explains time effects
    # -------------------------
    co2 = _read_co2(CO2_PATH)
    te2 = te.merge(co2, on="year", how="inner").sort_values("year")
    if te2.empty:
        raise ValueError("time_effect x CO2 merge is empty; check CO2 years.")

    X = te2[["co2_ppm"]].copy()
    if CO2_QUADRATIC:
        X["co2_ppm2"] = X["co2_ppm"] ** 2
    X = sm.add_constant(X)
    y = te2["time_effect"].astype(float)

    ols = sm.OLS(y, X).fit()
    te2["time_effect_hat"] = ols.predict(X)
    te2["resid"] = te2["time_effect"] - te2["time_effect_hat"]
    te2.to_excel(out_dir / "03_co2_time_effect_regression.xlsx", index=False)
    (out_dir / "03_co2_time_effect_regression.txt").write_text(ols.summary().as_text(), encoding="utf-8")

    # -------------------------
    # SEM on anomalies
    # -------------------------

    use_et = ("ET90" in panel.columns) and (ET_XLSX is not None)
    use_swnet = ("SWnet90" in panel.columns) and (SWNET_XLSX is not None)

    # Base variables needed to construct VPD_resid within the SEM sample space
    sem_base_vars = ["y_logit", "P90", "VPD90", "Tmean", "SM90"]
    if use_et:
        sem_base_vars.append("ET90")
    if use_swnet:
        sem_base_vars.append("SWnet90")

    sem_df = panel[["hex_id", "year"] + sem_base_vars].dropna().copy()
    print(f"[INFO] SEM usable rows after dropna (base vars): {len(sem_df):,}")
    if sem_df.empty:
        raise ValueError("SEM dataframe empty after dropna; check missingness for SEM base vars.")

    # Optional: double-demean in the SEM space (recommended for mechanism validation)
    if DOUBLE_DEMEAN_FOR_SEM:
        sem_df = _double_demean(sem_df, sem_base_vars)

    # Construct VPD_resid in the SEM space:
    #   VPD_resid = VPD90 - E[VPD90 | Tmean (+ SWnet90)]
    sem_df["VPD_resid"], vpd_fit = _fit_vpd_resid(
        sem_df,
        vpd_col="VPD90",
        t_col="Tmean",
        swnet_col="SWnet90",
        include_swnet=bool(VPD_RESID_INCLUDE_SWNET),
    )

    # Save VPD_resid regression diagnostics (useful for SI)
    vpd_diag = {
        "formula": "VPD90 ~ Tmean" + (" + SWnet90" if (bool(VPD_RESID_INCLUDE_SWNET) and use_swnet) else ""),
        "n_obs": int(vpd_fit.nobs),
        "r2": float(vpd_fit.rsquared),
        "adj_r2": float(vpd_fit.rsquared_adj),
        "params": {k: float(v) for k, v in vpd_fit.params.items()},
        "pvalues": {k: float(v) for k, v in vpd_fit.pvalues.items()},
    }
    _safe_json(vpd_diag, out_dir / "03_vpd_resid_regression.json")
    try:
        (out_dir / "03_vpd_resid_regression.txt").write_text(vpd_fit.summary().as_text(), encoding="utf-8")
    except Exception:
        pass

    # Final SEM variables
    sem_vars = ["y_logit", "P90", "VPD_resid", "Tmean", "SM90"]
    if use_et:
        sem_vars.append("ET90")
    if use_swnet:
        sem_vars.append("SWnet90")
    z_stats = None
    if Z_SCORE_FOR_SEM:
        sem_df, z_stats = _zscore(sem_df, sem_vars)

    model_text = build_sem_model_text(
        use_et=use_et,
        use_swnet=use_swnet,
        include_p90_direct=bool(INCLUDE_P90_DIRECT_IN_Y),
        include_t_in_sm=bool(INCLUDE_TMEAN_IN_SM),
        include_t_in_et=bool(INCLUDE_TMEAN_IN_ET),
        include_swnet_direct_in_y=bool(INCLUDE_SWNET_DIRECT_IN_Y)
    )
    (out_dir / "04_sem_model.txt").write_text(model_text, encoding="utf-8")

    model = Model(model_text)
    model.fit(sem_df[sem_vars])

    st = calc_stats(model)
    fit = {
        "n_obs": int(len(sem_df)),
        "variables": sem_vars,
        "DOUBLE_DEMEAN_FOR_SEM": bool(DOUBLE_DEMEAN_FOR_SEM),
        "Z_SCORE_FOR_SEM": bool(Z_SCORE_FOR_SEM),
        "CFI": _scalar(st.get("CFI", np.nan)),
        "TLI": _scalar(st.get("TLI", np.nan)),
        "RMSEA": _scalar(st.get("RMSEA", np.nan)),
        "SRMR": _scalar(st.get("SRMR", np.nan)),
        "AIC": _scalar(st.get("AIC", np.nan)),
        "BIC": _scalar(st.get("BIC", np.nan)),
    }
    if z_stats is not None:
        fit["zscore_stats_raw"] = {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in z_stats.items()}
    _safe_json(fit, out_dir / "05_sem_fit_indices.json")

    par = inspect(model)
    par.to_excel(out_dir / "06_sem_parameters.xlsx", index=False)

    eff = compute_effects(
        par,
        use_et=use_et,
        use_swnet=use_swnet,
        include_p90_direct=bool(INCLUDE_P90_DIRECT_IN_Y),
        include_t_in_sm=bool(INCLUDE_TMEAN_IN_SM),
        include_t_in_et=bool(INCLUDE_TMEAN_IN_ET),
        include_swnet_direct_in_y=bool(INCLUDE_SWNET_DIRECT_IN_Y)
    )
    eff.to_excel(out_dir / "07_sem_effects_direct_indirect.xlsx", index=False)

    sem_df[sem_vars].describe().T.to_excel(out_dir / "08_sem_anomaly_summary.xlsx")

    if WRITE_ANOMALY_SAMPLE and ANOMALY_SAMPLE_N and ANOMALY_SAMPLE_N > 0:
        take = min(ANOMALY_SAMPLE_N, len(sem_df))
        sem_df.sample(n=take, random_state=42).to_csv(
            out_dir / "09_sem_anomalies_sample.csv.gz", index=False, compression="gzip"
        )

    print("\n[DONE]")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
