#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_vif_diagnostics.py

Low-cost collinearity diagnostics for the dryland vegetation-cover expansion
panel attribution model.

This version uses an explicit configuration block, following the style of the
main attribution scripts. No command-line arguments are required.

Purpose
-------
Compute variance inflation factors (VIFs) for the same predictor sets used in
run_panel_attribution.py, without changing the attribution model itself.

Diagnostics are computed for:
  - UP and DOWN panels
  - CLIM_ONLY and CLIM_PLUS_XCO2 specifications
  - pooled standardized predictors, matching regression-sample z-scoring
  - entity-demeaned standardized predictors, matching within-hexagon variation
    under hexagon fixed effects

Outputs
-------
  - vif_diagnostics_panel_attribution.csv
  - vif_model_summary.csv
  - vif_correlation_matrices.xlsx
  - vif_diagnostics_panel_attribution.xlsx
  - vif_diagnostics_summary.json

Notes
-----
VIF is a diagnostic of predictor collinearity, not an additional attribution
model. It is intentionally kept separate from the main modelling pipeline to
respect the minimum-experiment principle.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIG: paths and model settings
# =============================================================================

# Original model script to reuse IO and preprocessing functions.
# The canonical model script lives under src/05_attribution in the public tree.
# Edit the path below only if the repository layout changes.
MODEL_SCRIPT = r"../run_panel_attribution.py"

# Input directories (ONLY these two)
DIR_UP = r"zonal_tables_parallel_up_V2"
DIR_DOWN = r"zonal_tables_parallel_down_V2"

# Dependent variable file (wide: hex_id + year columns) -- set separately.
# You may set a filename (within DIR_UP/DIR_DOWN) OR an absolute path.
FILE_VEGFRAC_UP = r"NDVI_trend_hex_100_up.xlsx"
FILE_VEGFRAC_DOWN = r"NDVI_trend_hex_100_down.xlsx"

# Climate filenames (within each dir)
FILE_P90     = r"W90_Precip.xlsx"         # -> P90
FILE_VPD90   = r"VPD90_mean.xlsx"         # -> VPD90, used only to construct VPD_resid
FILE_T90     = r"T2m90_mean.xlsx"         # -> Tmean90
FILE_SM90    = r"SM_L1_90_mean.xlsx"      # -> SM90
FILE_ET90    = r"ET90_sum.xlsx"           # -> ET90, optional
FILE_SWNET90 = r"SWnet90_mean.xlsx"       # -> SWnet90, optional

# XCO2 filename (within each dir)
# If your final vpd-resid script uses NOAA_CO2_EPSG8857.xlsx, change this line accordingly.
FILE_XCO2 = r"CO2_annual_tif.xlsx"        # -> XCO2

# Switches, aligned with the main vpd-resid attribution model
INCLUDE_ET = True
INCLUDE_SWNET = True
INCLUDE_INTERACTION = True   # P90 x VPD_resid

# Standardization, aligned with the main attribution model
Z_SCORE_X = True             # z-score covariates on regression sample
CENTER_ONLY = False          # if Z_SCORE_X=False, optionally mean-center

# Years and logit stability, aligned with the main attribution model
YEAR_MIN = 2000
YEAR_MAX = 2022
LOGIT_EPS = 1e-6

# VPD_resid settings, aligned with the main vpd-resid attribution model
VPD_RESID_INCLUDE_SWNET = True
VPD_RESID_DOUBLE_DEMEAN = True

# Output directory for VIF diagnostics
OUTDIR = r"trend_outputs_co2_vpd_resid_vif"


# =============================================================================
# Model module loading and configuration override
# =============================================================================

def _resolve_relative_path(path_like: str) -> Path:
    """Resolve relative paths first relative to this script, then cwd."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    script_dir = Path(__file__).resolve().parent
    p1 = script_dir / p
    if p1.exists():
        return p1
    return Path.cwd() / p


def load_model_module(model_script: str):
    script_path = _resolve_relative_path(model_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"Cannot find model script: {script_path}\n"
            "Edit MODEL_SCRIPT at the top of this VIF script, or place this file "
            "at the configured canonical repository path."
        )

    # The source model imports linearmodels for PanelOLS. VIF construction only
    # reuses IO and preprocessing functions, so provide a harmless stub if
    # linearmodels is absent in the diagnostic environment. If the user runs the
    # original attribution model, linearmodels is still required there.
    try:
        import linearmodels.panel  # noqa: F401
    except Exception:
        linearmodels_stub = types.ModuleType("linearmodels")
        panel_stub = types.ModuleType("linearmodels.panel")

        class _PanelOLSStub:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "linearmodels is required only for fitting the original PanelOLS "
                    "model, not for VIF diagnostics."
                )

        panel_stub.PanelOLS = _PanelOLSStub
        linearmodels_stub.panel = panel_stub
        sys.modules.setdefault("linearmodels", linearmodels_stub)
        sys.modules.setdefault("linearmodels.panel", panel_stub)

    spec = importlib.util.spec_from_file_location("trend_model_for_vif", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_config_to_model(model) -> None:
    """Force the imported model module to use the explicit config above."""
    model.DIR_UP = DIR_UP
    model.DIR_DOWN = DIR_DOWN
    model.FILE_VEGFRAC_UP = FILE_VEGFRAC_UP
    model.FILE_VEGFRAC_DOWN = FILE_VEGFRAC_DOWN

    model.FILE_P90 = FILE_P90
    model.FILE_VPD90 = FILE_VPD90
    model.FILE_T90 = FILE_T90
    model.FILE_SM90 = FILE_SM90
    model.FILE_ET90 = FILE_ET90
    model.FILE_SWNET90 = FILE_SWNET90
    model.FILE_XCO2 = FILE_XCO2

    model.INCLUDE_ET = bool(INCLUDE_ET)
    model.INCLUDE_SWNET = bool(INCLUDE_SWNET)
    model.INCLUDE_INTERACTION = bool(INCLUDE_INTERACTION)
    model.Z_SCORE_X = bool(Z_SCORE_X)
    model.CENTER_ONLY = bool(CENTER_ONLY)
    model.YEAR_MIN = int(YEAR_MIN)
    model.YEAR_MAX = int(YEAR_MAX)
    model.LOGIT_EPS = float(LOGIT_EPS)
    model.VPD_RESID_INCLUDE_SWNET = bool(VPD_RESID_INCLUDE_SWNET)
    model.VPD_RESID_DOUBLE_DEMEAN = bool(VPD_RESID_DOUBLE_DEMEAN)


# =============================================================================
# VIF utilities
# =============================================================================

def vif_from_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF by regressing each predictor against all other predictors."""
    X = X.copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    rows = []
    n_obs, n_cols = X.shape

    if n_cols == 0 or n_obs == 0:
        return pd.DataFrame(columns=[
            "variable", "vif", "r2_against_others", "n_obs", "n_predictors",
            "rank_others", "condition_number_others"
        ])

    for target in X.columns:
        y = X[target].to_numpy(dtype=float)
        others = [c for c in X.columns if c != target]

        if len(others) == 0:
            vif = 1.0
            r2 = 0.0
            rank = 0
            cond = np.nan
        else:
            Z = X[others].to_numpy(dtype=float)
            # Include an intercept for the auxiliary regression.
            Z = np.column_stack([np.ones(Z.shape[0]), Z])
            rank = int(np.linalg.matrix_rank(Z))
            try:
                cond = float(np.linalg.cond(Z))
            except Exception:
                cond = np.nan

            try:
                coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
                yhat = Z @ coef
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                if not np.isfinite(r2):
                    vif = np.nan
                elif r2 >= 1.0:
                    vif = np.inf
                else:
                    vif = 1.0 / max(1.0 - r2, np.finfo(float).eps)
            except Exception:
                r2 = np.nan
                vif = np.nan

        rows.append({
            "variable": target,
            "vif": float(vif) if np.isfinite(vif) else vif,
            "r2_against_others": float(r2) if np.isfinite(r2) else r2,
            "n_obs": int(n_obs),
            "n_predictors": int(n_cols),
            "rank_others": int(rank),
            "condition_number_others": cond,
        })

    return pd.DataFrame(rows)


def standardize_like_model(model, panel: pd.DataFrame, covars: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    need_cols = ["hex_id", "year", "y_logit"] + covars
    df = panel[need_cols].copy().replace([np.inf, -np.inf], np.nan)
    df_model = df.dropna(subset=["y_logit"] + covars).copy()
    df_std, params = model.standardize_fit(df_model, covars)
    return df_std, params


def entity_demean(df_std: pd.DataFrame, covars: List[str]) -> pd.DataFrame:
    X = df_std[["hex_id"] + covars].copy()
    # Within transformation corresponding to hexagon fixed effects.
    return X[covars] - X.groupby("hex_id")[covars].transform("mean")


def classify_vif(v: float) -> str:
    if not np.isfinite(v):
        return "not finite"
    if v < 5:
        return "low (<5)"
    if v < 10:
        return "moderate (5-10)"
    return "high (>=10)"


def corr_long(X: pd.DataFrame, dataset: str, model_name: str, design: str) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if X.shape[1] == 0 or X.shape[0] == 0:
        return pd.DataFrame()
    corr = X.corr()
    corr.insert(0, "variable", corr.index)
    corr.insert(0, "design", design)
    corr.insert(0, "model", model_name)
    corr.insert(0, "dataset", dataset)
    return corr.reset_index(drop=True)


def run_one(model, dataset: str, panel: pd.DataFrame, include_xco2: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_name = "CLIM_PLUS_XCO2" if include_xco2 else "CLIM_ONLY"
    covars = model.get_covars(include_xco2=include_xco2)
    df_std, _ = standardize_like_model(model, panel, covars)

    outputs = []
    corr_outputs = []

    # 1) Pooled standardized predictors, matching regression-sample scaling.
    X_pooled = df_std[covars].copy()
    vif_pooled = vif_from_matrix(X_pooled)
    vif_pooled.insert(0, "design", "pooled_standardized")
    outputs.append(vif_pooled)
    corr_outputs.append(corr_long(X_pooled, dataset, model_name, "pooled_standardized"))

    # 2) Entity-demeaned standardized predictors, matching within-hexagon FE variation.
    X_within = entity_demean(df_std, covars)
    vif_within = vif_from_matrix(X_within)
    vif_within.insert(0, "design", "entity_demeaned_standardized")
    outputs.append(vif_within)
    corr_outputs.append(corr_long(X_within, dataset, model_name, "entity_demeaned_standardized"))

    vif = pd.concat(outputs, ignore_index=True)
    vif.insert(0, "model", model_name)
    vif.insert(0, "dataset", dataset)
    vif["vif_flag"] = vif["vif"].apply(classify_vif)
    vif["covariates"] = ", ".join(covars)
    vif["n_hex"] = int(df_std["hex_id"].nunique())
    vif["n_year"] = int(df_std["year"].nunique())

    summary = (vif.groupby(["dataset", "model", "design"], as_index=False)
               .agg(n_obs=("n_obs", "max"),
                    n_hex=("n_hex", "max"),
                    n_year=("n_year", "max"),
                    n_predictors=("n_predictors", "max"),
                    max_vif=("vif", "max"),
                    median_vif=("vif", "median"),
                    max_r2_against_others=("r2_against_others", "max")))
    summary["max_vif_flag"] = summary["max_vif"].apply(classify_vif)

    corr_nonempty = [c for c in corr_outputs if c is not None and not c.empty]
    corr_df = pd.concat(corr_nonempty, ignore_index=True) if corr_nonempty else pd.DataFrame()
    return vif, summary, corr_df


# =============================================================================
# Output writing
# =============================================================================

def write_outputs(outdir: Path, vif: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame, model_script_resolved: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    vif_csv = outdir / "vif_diagnostics_panel_attribution.csv"
    summary_csv = outdir / "vif_model_summary.csv"
    corr_xlsx = outdir / "vif_correlation_matrices.xlsx"
    workbook_xlsx = outdir / "vif_diagnostics_panel_attribution.xlsx"
    summary_json = outdir / "vif_diagnostics_summary.json"

    vif.to_csv(vif_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    meta = {
        "model_script": str(model_script_resolved),
        "outdir": str(outdir.resolve()),
        "diagnostic": "variance inflation factor",
        "data_config": {
            "DIR_UP": DIR_UP,
            "DIR_DOWN": DIR_DOWN,
            "FILE_VEGFRAC_UP": FILE_VEGFRAC_UP,
            "FILE_VEGFRAC_DOWN": FILE_VEGFRAC_DOWN,
            "FILE_P90": FILE_P90,
            "FILE_VPD90": FILE_VPD90,
            "FILE_T90": FILE_T90,
            "FILE_SM90": FILE_SM90,
            "FILE_ET90": FILE_ET90,
            "FILE_SWNET90": FILE_SWNET90,
            "FILE_XCO2": FILE_XCO2,
        },
        "model_config": {
            "INCLUDE_ET": INCLUDE_ET,
            "INCLUDE_SWNET": INCLUDE_SWNET,
            "INCLUDE_INTERACTION": INCLUDE_INTERACTION,
            "Z_SCORE_X": Z_SCORE_X,
            "CENTER_ONLY": CENTER_ONLY,
            "YEAR_MIN": YEAR_MIN,
            "YEAR_MAX": YEAR_MAX,
            "VPD_RESID_INCLUDE_SWNET": VPD_RESID_INCLUDE_SWNET,
            "VPD_RESID_DOUBLE_DEMEAN": VPD_RESID_DOUBLE_DEMEAN,
        },
        "designs": ["pooled_standardized", "entity_demeaned_standardized"],
        "interpretation": "VIF is used only as a collinearity diagnostic for the prespecified panel attribution covariates.",
        "threshold_note": "Common descriptive thresholds: VIF < 5 low, 5 <= VIF < 10 moderate, VIF >= 10 high. Thresholds are diagnostic conventions, not hard acceptance rules.",
    }
    summary_json.write_text(
        json.dumps({"metadata": meta, "summary": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Excel outputs are convenience copies. CSV/JSON are the canonical lightweight outputs.
    try:
        with pd.ExcelWriter(workbook_xlsx, engine="openpyxl") as writer:
            readme = pd.DataFrame([
                ["Purpose", "Collinearity diagnostics for the fixed-effects panel attribution model."],
                ["Main diagnostic", "VIF computed on pooled standardized and entity-demeaned standardized predictors."],
                ["Pooled standardized", "Predictors after regression-sample z-scoring, before fixed-effect within transformation."],
                ["Entity-demeaned standardized", "Predictors after regression-sample z-scoring and within-hexagon demeaning; most directly aligned with hexagon fixed effects."],
                ["Interpretation", "VIF < 5 low; 5-10 moderate; >=10 high. Use as diagnostic, not as a separate model."],
                ["Model script", str(model_script_resolved)],
                ["DIR_UP", DIR_UP],
                ["DIR_DOWN", DIR_DOWN],
                ["FILE_XCO2", FILE_XCO2],
            ], columns=["Item", "Description"])
            readme.to_excel(writer, sheet_name="README", index=False)
            summary.to_excel(writer, sheet_name="Model_summary", index=False)
            vif.to_excel(writer, sheet_name="VIF", index=False)
            corr.to_excel(writer, sheet_name="Correlations", index=False)
    except Exception as e:
        print(f"[WARN] Excel workbook export failed: {e}")
        print("[WARN] CSV and JSON outputs were still written.")

    try:
        with pd.ExcelWriter(corr_xlsx, engine="openpyxl") as writer:
            corr.to_excel(writer, sheet_name="Correlations", index=False)
    except Exception as e:
        print(f"[WARN] Correlation workbook export failed: {e}")

    print("\n[DONE] VIF diagnostics written to:")
    print(f"  {vif_csv}")
    print(f"  {summary_csv}")
    print(f"  {workbook_xlsx}")
    print(f"  {summary_json}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    model = load_model_module(MODEL_SCRIPT)
    apply_config_to_model(model)

    model_script_resolved = _resolve_relative_path(MODEL_SCRIPT).resolve()
    outdir = _resolve_relative_path(OUTDIR)

    specs = [
        model.DatasetSpec(tag="UP", dirpath=DIR_UP, veg_file=FILE_VEGFRAC_UP),
        model.DatasetSpec(tag="DOWN", dirpath=DIR_DOWN, veg_file=FILE_VEGFRAC_DOWN),
    ]

    all_vif = []
    all_summary = []
    all_corr = []

    for spec in specs:
        print(f"\n=== Building panel for {spec.tag} ===")
        panel = model.build_panel(spec)
        for include_xco2 in [False, True]:
            vif, summary, corr = run_one(model, spec.tag, panel, include_xco2=include_xco2)
            all_vif.append(vif)
            all_summary.append(summary)
            all_corr.append(corr)

    vif_all = pd.concat(all_vif, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    corr_all = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    write_outputs(outdir, vif_all, summary_all, corr_all, model_script_resolved)


if __name__ == "__main__":
    main()
