#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agricultural-neighbourhood analysis for expanding dryland hexagons.

This public reconstruction starts from the frozen processed UP analysis table
and the frozen UP hexagon geometries. It preserves the original comparison,
local-excess, Mann-Whitney U and Cliff's-delta definitions.

Neighbourhood semantics follow the final Supplementary Information wording:
- ring 1: UP hexagons sharing a boundary with the agricultural focal hexagon;
- ring 2: UP hexagons adjacent to any ring-1 hexagon, excluding the focal;
- combined neighbourhood: ring 1 union ring 2;
- agricultural focal hexagons are excluded from the non-agricultural
  background before comparison.

The original script removed ring-1 nodes from its ring-2-only set. That changes
only the standalone ring-2 label, not the combined ring1+ring2 neighbourhood
used by the primary analysis. The reconstruction boundary is documented in docs/preprocessing_reproducibility.md.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_TABLE = REPO_ROOT / "data/processed/agricultural_neighbourhood/ag_neighborhood_analysis_table.csv"
UP_HEXAGONS = REPO_ROOT / "data/processed/vegetation/up_hexagons.gpkg"
OUTPUT_DIR = REPO_ROOT / "outputs/agricultural_neighbourhood"

AG_LABEL = "Ag_expansion"
LEGACY_LABEL_MAP = {
    "veg_cover_upgrade": "Bare_or_sparse_to_grass_or_forest",
    "Nat_recovery": "Bare_or_sparse_to_grass_or_forest",
}


def cliffs_delta(x, y) -> float:
    """Cliff's delta using the same rank/U formulation as the original script."""
    x = np.asarray(pd.Series(x).dropna())
    y = np.asarray(pd.Series(y).dropna())
    if len(x) == 0 or len(y) == 0:
        return np.nan
    pooled = np.concatenate([x, y])
    ranks = pd.Series(pooled).rank(method="average").to_numpy()
    rx = ranks[: len(x)].sum()
    u = rx - len(x) * (len(x) + 1) / 2
    return (2 * u) / (len(x) * len(y)) - 1


def build_adjacency(up_hexagons: gpd.GeoDataFrame) -> dict[int, set[int]]:
    """Build boundary-touching adjacency among the fixed UP sample."""
    gdf = up_hexagons[["hex_id", "geometry"]].copy()
    gdf["hex_id"] = gdf["hex_id"].astype(int)
    gdf["geometry"] = gdf.geometry.buffer(0)
    joined = gpd.sjoin(gdf, gdf, how="inner", predicate="touches")
    pairs = joined[["hex_id_left", "hex_id_right"]].rename(
        columns={"hex_id_left": "src", "hex_id_right": "nbr"}
    )
    pairs = pairs[pairs["src"] != pairs["nbr"]]

    neighbours: dict[int, set[int]] = defaultdict(set)
    for src, nbr in pairs.itertuples(index=False):
        neighbours[int(src)].add(int(nbr))
    return neighbours


def ring_sets(node: int, neighbours: dict[int, set[int]]) -> tuple[set[int], set[int], set[int]]:
    """Return ring1, final-SI ring2, and their union."""
    ring1 = set(neighbours.get(node, set()))
    ring2 = set()
    for nbr in ring1:
        ring2.update(neighbours.get(nbr, set()))
    ring2.discard(node)
    combined = ring1 | ring2
    return ring1, ring2, combined


def summarize(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": group["hex_id"].nunique(),
            "beta_obs_median": group["beta_obs"].median(),
            "endpoint_change_median": group["endpoint_change"].median(),
            "share_nat_co2_median": group["share_nat_co2"].median(),
            "beta_nat_co2_median": group["beta_nat_co2"].median(),
            "beta_res_co2_median": group["beta_res_co2"].median(),
            "natural_dom_rate": (group["dominance_co2"] == "NATURAL").mean(),
            "P90_trend_median": group["slope_P90"].median(),
            "SM90_trend_median": group["slope_SM90"].median(),
            "VPD90_trend_median": group["slope_VPD90"].median(),
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis = pd.read_csv(INPUT_TABLE)
    analysis["hex_id"] = analysis["hex_id"].astype(int)
    analysis["pathway_type"] = analysis["pathway_type"].replace(LEGACY_LABEL_MAP)

    up = gpd.read_file(UP_HEXAGONS, layer="up_hexagons")
    if len(up) != analysis["hex_id"].nunique():
        raise ValueError(
            f"UP geometry/table size mismatch: geometry={len(up):,}, "
            f"table={analysis['hex_id'].nunique():,}"
        )

    neighbours = build_adjacency(up)
    analysis_idx = analysis.set_index("hex_id")
    ag_ids = set(analysis.loc[analysis["pathway_type"] == AG_LABEL, "hex_id"])

    pair_rows = []
    for hid in ag_ids:
        ring1, ring2, combined = ring_sets(hid, neighbours)
        for ring_name, nodes in [
            ("ring1", ring1),
            ("ring2", ring2),
            ("ring1_plus_ring2", combined),
        ]:
            for nbr in nodes:
                if nbr in analysis_idx.index and analysis_idx.at[nbr, "pathway_type"] != AG_LABEL:
                    pair_rows.append((hid, ring_name, nbr))

    neighbour_pairs = pd.DataFrame(
        pair_rows,
        columns=["ag_hex_id", "ring", "neighbor_hex_id"],
    )
    neighbour_pairs.to_csv(OUTPUT_DIR / "ag_nonag_neighbor_pairs.csv", index=False)

    neighbour_ids = set(
        neighbour_pairs.loc[
            neighbour_pairs["ring"] == "ring1_plus_ring2",
            "neighbor_hex_id",
        ]
    )
    ag_self = analysis[analysis["pathway_type"] == AG_LABEL].copy()
    nonag = analysis[analysis["pathway_type"] != AG_LABEL].copy()
    ag_neighbor_nonag = analysis[analysis["hex_id"].isin(neighbour_ids)].copy()
    other_nonag = nonag[~nonag["hex_id"].isin(neighbour_ids)].copy()

    three_group = (
        pd.concat(
            {
                "Ag_self": summarize(ag_self),
                "Ag_neighbor_nonAg": summarize(ag_neighbor_nonag),
                "Other_nonAg_UP": summarize(other_nonag),
            },
            axis=1,
        )
        .T.reset_index()
        .rename(columns={"index": "group"})
    )
    three_group.to_csv(OUTPUT_DIR / "ag_three_group_summary.csv", index=False)

    metrics = [
        "beta_obs",
        "endpoint_change",
        "share_nat_co2",
        "beta_nat_co2",
        "beta_res_co2",
        "slope_P90",
        "slope_SM90",
        "slope_VPD90",
    ]
    contrast_rows = []
    for metric in metrics:
        x = ag_neighbor_nonag[metric].dropna()
        y = other_nonag[metric].dropna()
        _, p_value = mannwhitneyu(x, y, alternative="two-sided")
        contrast_rows.append(
            {
                "metric": metric,
                "ag_neighbor_nonag_n": len(x),
                "other_nonag_n": len(y),
                "ag_neighbor_nonag_median": x.median(),
                "other_nonag_median": y.median(),
                "median_diff": x.median() - y.median(),
                "p_value": p_value,
                "cliffs_delta": cliffs_delta(x, y),
            }
        )
    pd.DataFrame(contrast_rows).to_csv(
        OUTPUT_DIR / "ag_neighbor_vs_other_nonag_summary.csv",
        index=False,
    )

    local_rows = []
    for hid in ag_ids:
        _, _, combined = ring_sets(hid, neighbours)
        ids = [
            nbr
            for nbr in combined
            if nbr in analysis_idx.index and analysis_idx.at[nbr, "pathway_type"] != AG_LABEL
        ]
        if not ids:
            continue
        neigh = analysis.loc[analysis["hex_id"].isin(ids)]
        local_rows.append(
            {
                "hex_id": hid,
                "n_nonAg_neighbors": len(ids),
                "ag_beta_obs": analysis_idx.at[hid, "beta_obs"],
                "neighbor_beta_obs_median": neigh["beta_obs"].median(),
                "local_excess_beta_obs": analysis_idx.at[hid, "beta_obs"]
                - neigh["beta_obs"].median(),
                "ag_share_nat_co2": analysis_idx.at[hid, "share_nat_co2"],
                "neighbor_share_nat_co2_median": neigh["share_nat_co2"].median(),
                "local_excess_share_nat_co2": analysis_idx.at[hid, "share_nat_co2"]
                - neigh["share_nat_co2"].median(),
            }
        )

    local = pd.DataFrame(local_rows)
    local.to_csv(OUTPUT_DIR / "ag_local_excess_table.csv", index=False)

    positive = local.loc[local["local_excess_beta_obs"] > 0, "local_excess_beta_obs"]
    local_summary = pd.DataFrame(
        {
            "n_ag_with_nonAg_neighbors": [len(local)],
            "local_excess_beta_obs_median": [local["local_excess_beta_obs"].median()],
            "local_excess_beta_obs_q25": [local["local_excess_beta_obs"].quantile(0.25)],
            "local_excess_beta_obs_q75": [local["local_excess_beta_obs"].quantile(0.75)],
            "share_positive_local_excess_beta_obs": [
                (local["local_excess_beta_obs"] > 0).mean()
            ],
            "positive_local_excess_beta_obs_median": [positive.median()],
            "local_excess_share_nat_co2_median": [
                local["local_excess_share_nat_co2"].median()
            ],
            "share_positive_local_excess_share_nat_co2": [
                (local["local_excess_share_nat_co2"] > 0).mean()
            ],
        }
    )
    local_summary.to_csv(OUTPUT_DIR / "ag_local_excess_summary.csv", index=False)

    path_comp1 = ag_neighbor_nonag["pathway_type"].value_counts(normalize=True).rename(
        "ag_neighbor_nonag_share"
    )
    path_comp2 = other_nonag["pathway_type"].value_counts(normalize=True).rename(
        "other_nonag_share"
    )
    path_comp = pd.concat([path_comp1, path_comp2], axis=1).fillna(0)
    path_comp["share_diff"] = (
        path_comp["ag_neighbor_nonag_share"] - path_comp["other_nonag_share"]
    )
    path_comp.reset_index().rename(columns={"index": "pathway_type"}).to_csv(
        OUTPUT_DIR / "ag_neighbor_pathway_composition.csv",
        index=False,
    )

    print(f"Done. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
