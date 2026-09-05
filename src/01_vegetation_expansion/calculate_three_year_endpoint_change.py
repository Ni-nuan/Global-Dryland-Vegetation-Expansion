#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate 3-year-window endpoint change for NDVI-derived UP hexagons.

Definition
----------
Early endpoint = median of veg_2000, veg_2001, veg_2002
Late endpoint  = median of veg_2020, veg_2021, veg_2022

A hexagon is retained for endpoint-change summaries only when BOTH windows
contain at least MIN_VALID_PER_WINDOW non-missing annual observations.

UP definition:
    trend_tau > 0
    p_value < 0.05
    sen_slope > 0

DOWN definition:
    trend_tau < 0
    p_value < 0.05
    sen_slope < 0

The script writes:
1) one summary CSV across all configured input files;
2) one per-hexagon endpoint CSV for each input file.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# User settings
# =============================================================================

# Add/remove files here. Labels can be NDVI thresholds (e.g. "0.20") or
# spatial units (e.g. "75 km2") depending on the table being recalculated.
INPUTS = {
    # "0.20": Path(r"NDVI_trend_hex_100.csv"),
    # "0.22": Path(r"NDVI_22_trend_hex_100.csv"),
    # "0.24": Path(r"NDVI_24_trend_hex_100.csv"),
     "0.16": Path(r"hex_data/NDVI_16_trend_hex_100.csv"),
     "0.18": Path(r"hex_data/NDVI_18_trend_hex_100.csv"),
     "75 km2": Path(r"hex_data/NDVI_trend_hex_75.csv"),
     "125 km2": Path(r"hex_data/NDVI_trend_hex_125.csv"),
}

OUTPUT_SUMMARY = Path("three_year_endpoint_summary.csv")
OUTPUT_DETAIL_DIR = Path("three_year_endpoint_details")

EARLY_YEARS = [2000, 2001, 2002]
LATE_YEARS = [2020, 2021, 2022]
MIN_VALID_PER_WINDOW = 2
P_THRESHOLD = 0.05

# Total global dryland hexagons is useful for threshold-sensitivity tables.
# Set to None if not relevant to the configured inputs.
TOTAL_DRYLAND_HEX = 747_205


# =============================================================================
# Helpers
# =============================================================================

def require_columns(df: pd.DataFrame, cols: list[str], file_path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{file_path} is missing required columns: {missing}"
        )


def row_window_median(
    df: pd.DataFrame,
    cols: list[str],
    min_valid: int,
) -> tuple[pd.Series, pd.Series]:
    """Return row-wise window median and valid-observation count."""
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    n_valid = values.notna().sum(axis=1)
    med = values.median(axis=1, skipna=True)
    med = med.where(n_valid >= min_valid, np.nan)
    return med, n_valid


def summarize_one(label: str, file_path: Path) -> dict:
    df = pd.read_csv(file_path)

    early_cols = [f"veg_{y}" for y in EARLY_YEARS]
    late_cols = [f"veg_{y}" for y in LATE_YEARS]

    required = [
        "hex_id", "trend_tau", "p_value", "sen_slope",
        *early_cols, *late_cols,
    ]
    require_columns(df, required, file_path)

    # Ensure trend fields are numeric.
    for c in ["trend_tau", "p_value", "sen_slope"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid_trend = (
        df["trend_tau"].notna()
        & df["p_value"].notna()
        & df["sen_slope"].notna()
    )

    up = (
        (df["trend_tau"] > 0)
        & (df["p_value"] < P_THRESHOLD)
        & (df["sen_slope"] > 0)
    )

    down = (
        (df["trend_tau"] < 0)
        & (df["p_value"] < P_THRESHOLD)
        & (df["sen_slope"] < 0)
    )

    up_df = df.loc[up].copy()

    early_median, early_n = row_window_median(
        up_df, early_cols, MIN_VALID_PER_WINDOW
    )
    late_median, late_n = row_window_median(
        up_df, late_cols, MIN_VALID_PER_WINDOW
    )

    endpoint_valid = early_median.notna() & late_median.notna()
    delta = late_median - early_median

    valid_delta = delta.loc[endpoint_valid]

    # Save per-hexagon details for auditing/reproducibility.
    detail = pd.DataFrame({
        "hex_id": up_df["hex_id"].values,
        "early_valid_n": early_n.values,
        "late_valid_n": late_n.values,
        "early_window_median": early_median.values,
        "late_window_median": late_median.values,
        "endpoint_change_3yr": delta.values,
        "endpoint_valid": endpoint_valid.values,
    })

    OUTPUT_DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = (
        str(label)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "p")
    )
    detail.to_csv(
        OUTPUT_DETAIL_DIR / f"endpoint_3yr_{safe_label}.csv",
        index=False,
    )

    n_total = len(df)
    n_valid_trend = int(valid_trend.sum())
    n_up = int(up.sum())
    n_down = int(down.sum())
    n_endpoint = int(endpoint_valid.sum())

    result = {
        "label": label,
        "file": str(file_path),
        "total_rows": n_total,
        "valid_trend_hex": n_valid_trend,
        "UP_count": n_up,
        "DOWN_count": n_down,
        "UP_DOWN_ratio": n_up / n_down if n_down else np.nan,
        "endpoint_valid_UP": n_endpoint,
        "endpoint_excluded_UP": n_up - n_endpoint,
        "median_endpoint_change_3yr": valid_delta.median(),
        "q25_endpoint_change_3yr": valid_delta.quantile(0.25),
        "q75_endpoint_change_3yr": valid_delta.quantile(0.75),
    }

    TOTAL_DRYLAND_HEX = {
    "0.16": 747_205,
    "0.18": 747_205,
    "0.20": 747_205,
    "0.22": 747_205,
    "0.24": 747_205,
    "75 km2": 991_851,
    "100 km2": 747_205,
    "125 km2": 599_958,
    }
    
    total_dryland_hex = TOTAL_DRYLAND_HEX.get(label)

    if total_dryland_hex is not None:
        result.update({
            "UP_all_dryland_pct": 100.0 * n_up / total_dryland_hex,
            "DOWN_all_dryland_pct": 100.0 * n_down / total_dryland_hex,
        })

    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    rows = []

    for label, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        rows.append(summarize_one(label, path))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    # Human-readable terminal output.
    display_cols = [
        "label",
        "valid_trend_hex",
        "UP_count",
        "DOWN_count",
        "endpoint_valid_UP",
        "endpoint_excluded_UP",
        "median_endpoint_change_3yr",
        "q25_endpoint_change_3yr",
        "q75_endpoint_change_3yr",
    ]

    print("\nThree-year endpoint summary")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved summary: {OUTPUT_SUMMARY}")
    print(f"Saved per-hexagon details in: {OUTPUT_DETAIL_DIR}/")


if __name__ == "__main__":
    main()
