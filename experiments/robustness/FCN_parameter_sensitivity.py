#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameter-sensitivity analysis for the frozen FCN formal run.

This supplementary script does not retrain any model. It reads the frozen
Validation search outputs produced by the formal main flow and summarizes:

- Stage 1 sensitivity over the frozen Top-K selection table.
- Stage 2 sensitivity over the frozen (lambda_0, lambda_t) grid.

The purpose is to document whether the selected operating point lies in a
stable region rather than at an isolated spike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.common import bundle_paths, load_formal_bundle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_parameter_sensitivity"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_block(title: str, values: Dict[str, object]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for key, value in values.items():
        if isinstance(value, (float, np.floating)):
            if np.isnan(float(value)):
                print(f"{key}: nan")
            else:
                print(f"{key}: {float(value):.6f}")
        else:
            print(f"{key}: {value}")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def add_selected_flags_stage1(df: pd.DataFrame, selected_top_k: int, selected_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["TopK"] = pd.to_numeric(out["TopK"], errors="coerce").astype("Int64")
    out["Threshold"] = pd.to_numeric(out["Threshold"], errors="coerce")
    out["F1"] = pd.to_numeric(out["F1"], errors="coerce")
    out["Precision"] = pd.to_numeric(out["Precision"], errors="coerce")
    out["Recall"] = pd.to_numeric(out["Recall"], errors="coerce")
    out["RetainedEdges"] = pd.to_numeric(out["RetainedEdges"], errors="coerce")
    out["selected_row"] = (
        out["TopK"].astype(int).eq(int(selected_top_k))
        & np.isclose(out["Threshold"].to_numpy(dtype=float), float(selected_threshold), atol=1e-12, rtol=0.0)
    )
    out["delta_f1_from_selected"] = out["F1"] - float(
        out.loc[out["selected_row"], "F1"].iloc[0]
        if bool(out["selected_row"].any())
        else out["F1"].max()
    )
    out["delta_precision_from_selected"] = out["Precision"] - float(
        out.loc[out["selected_row"], "Precision"].iloc[0]
        if bool(out["selected_row"].any())
        else out["Precision"].max()
    )
    out["delta_retained_edges_from_selected"] = out["RetainedEdges"] - float(
        out.loc[out["selected_row"], "RetainedEdges"].iloc[0]
        if bool(out["selected_row"].any())
        else out["RetainedEdges"].min()
    )
    return out


def stage1_topk_summary(df: pd.DataFrame, selected_top_k: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    selected_row = df.loc[df["selected_row"]]
    if selected_row.empty:
        selected_f1 = float(df["F1"].max())
        selected_precision = float(df.loc[df["F1"].eq(df["F1"].max()), "Precision"].max())
        selected_retained = float(df.loc[df["F1"].eq(df["F1"].max()), "RetainedEdges"].min())
    else:
        selected_f1 = float(selected_row["F1"].iloc[0])
        selected_precision = float(selected_row["Precision"].iloc[0])
        selected_retained = float(selected_row["RetainedEdges"].iloc[0])

    for _, row in df.sort_values("TopK", ascending=True, kind="mergesort").iterrows():
        rows.append(
            {
                "TopK": int(row["TopK"]),
                "Threshold": safe_float(row["Threshold"]),
                "Feasible": bool(row["Feasible"]),
                "Recall": safe_float(row["Recall"]),
                "Precision": safe_float(row["Precision"]),
                "F1": safe_float(row["F1"]),
                "RetainedEdges": safe_float(row["RetainedEdges"]),
                "TruePositives": safe_float(row["TruePositives"]),
                "FalsePositives": safe_float(row["FalsePositives"]),
                "FalseNegatives": safe_float(row["FalseNegatives"]),
                "is_selected": bool(row["selected_row"]),
                "delta_f1_from_selected": safe_float(row["F1"]) - selected_f1,
                "delta_precision_from_selected": safe_float(row["Precision"]) - selected_precision,
                "delta_retained_edges_from_selected": safe_float(row["RetainedEdges"]) - selected_retained,
            }
        )
    return pd.DataFrame(rows)


def add_selected_flags_stage2(df: pd.DataFrame, selected_lambda_0: float, selected_lambda_t: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("lambda_0", "lambda_t", "AUC", "AP", "Precision", "Recall", "F1", "PredEdges", "TP", "FP", "FN", "ObjectiveTotal"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["selected_row"] = np.isclose(out["lambda_0"].to_numpy(dtype=float), float(selected_lambda_0), atol=1e-12, rtol=0.0) & np.isclose(
        out["lambda_t"].to_numpy(dtype=float), float(selected_lambda_t), atol=1e-12, rtol=0.0
    )
    selected = out.loc[out["selected_row"]]
    if selected.empty:
        best_idx = out.sort_values(["F1", "Precision", "PredEdges", "lambda_0", "lambda_t"], ascending=[False, False, True, True, True], kind="mergesort").index[0]
        selected = out.loc[[best_idx]]
    selected_f1 = float(selected["F1"].iloc[0])
    selected_precision = float(selected["Precision"].iloc[0])
    selected_pred_edges = float(selected["PredEdges"].iloc[0])
    out["delta_f1_from_selected"] = out["F1"] - selected_f1
    out["delta_precision_from_selected"] = out["Precision"] - selected_precision
    out["delta_pred_edges_from_selected"] = out["PredEdges"] - selected_pred_edges
    out["abs_lambda_0_distance"] = (out["lambda_0"] - float(selected_lambda_0)).abs()
    out["abs_lambda_t_distance"] = (out["lambda_t"] - float(selected_lambda_t)).abs()
    return out


def regular_grid_step(values: pd.Series) -> float:
    unique = sorted(set(float(v) for v in values.dropna().tolist()))
    if len(unique) < 2:
        return float("nan")
    diffs = [b - a for a, b in zip(unique[:-1], unique[1:]) if b > a]
    return float(min(diffs)) if diffs else float("nan")


def stage2_heatmaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot_f1 = df.pivot(index="lambda_0", columns="lambda_t", values="F1").sort_index(axis=0).sort_index(axis=1)
    pivot_precision = df.pivot(index="lambda_0", columns="lambda_t", values="Precision").sort_index(axis=0).sort_index(axis=1)
    pivot_pred_edges = df.pivot(index="lambda_0", columns="lambda_t", values="PredEdges").sort_index(axis=0).sort_index(axis=1)
    return pivot_f1, pivot_precision, pivot_pred_edges


def load_stage1_stage2_tables(bundle: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object]]:
    stage1_search = bundle["stage1"]["validation_retention_search"].copy()
    stage2_search = bundle["stage2"]["validation_search"].copy()
    stage1_params = bundle["stage1"]["selected_params"]
    stage2_params = bundle["stage2"]["selected_params"]
    return stage1_search, stage2_search, stage1_params, stage2_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Root directory containing fcn_outputs_stage1/ stage2/ main_experiments/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for parameter-sensitivity outputs.",
    )
    args = parser.parse_args()

    bundle = load_formal_bundle(args.bundle_root)
    stage1_search, stage2_search, stage1_params, stage2_params = load_stage1_stage2_tables(bundle)
    paths = bundle["paths"]

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else bundle_paths(args.bundle_root).root.parent
        / "experiments"
        / "robustness"
        / "fcn_parameter_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_top_k = int(stage1_params["selected_top_k"])
    selected_threshold = float(stage1_params["selected_threshold"])
    selected_lambda_0 = float(stage2_params["stage2b"]["selected_lambda_0"])
    selected_lambda_t = float(stage2_params["stage2b"]["selected_lambda_t"])

    stage1_enriched = add_selected_flags_stage1(stage1_search, selected_top_k, selected_threshold)
    stage1_summary = stage1_topk_summary(stage1_enriched, selected_top_k)
    stage1_feasible = stage1_enriched.loc[stage1_enriched["Feasible"].astype(bool)].copy()
    stage1_best = stage1_feasible.sort_values(
        ["F1", "Precision", "RetainedEdges", "TopK"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).head(1)

    stage1_selected_row = stage1_enriched.loc[stage1_enriched["selected_row"]]
    stage1_selected_in_feasible = bool(
        not stage1_selected_row.empty and bool(stage1_selected_row.iloc[0]["Feasible"])
    )
    stage1_slice = stage1_enriched.loc[
        stage1_enriched["TopK"].astype(int).between(max(1, selected_top_k - 1), min(5, selected_top_k + 1))
    ].sort_values(["TopK", "F1", "Precision"], ascending=[True, False, False], kind="mergesort")

    save_csv(stage1_enriched, output_dir / "stage1_validation_topk_search_enriched.csv")
    save_csv(stage1_summary, output_dir / "stage1_topk_sensitivity_summary.csv")
    save_csv(stage1_best, output_dir / "stage1_best_validation_row.csv")
    save_csv(stage1_feasible, output_dir / "stage1_feasible_validation_rows.csv")
    save_csv(stage1_selected_row, output_dir / "stage1_selected_row.csv")
    save_csv(stage1_slice, output_dir / "stage1_selected_neighborhood.csv")

    stage2_enriched = add_selected_flags_stage2(stage2_search, selected_lambda_0, selected_lambda_t)
    stage2_heatmap_f1, stage2_heatmap_precision, stage2_heatmap_pred_edges = stage2_heatmaps(stage2_enriched)
    stage2_sorted = stage2_enriched.sort_values(
        ["F1", "Precision", "PredEdges", "lambda_0", "lambda_t"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    stage2_selected_row = stage2_enriched.loc[stage2_enriched["selected_row"]]
    if stage2_selected_row.empty:
        stage2_selected_row = stage2_sorted.head(1)
    stage2_selected_is_best = bool(
        not stage2_selected_row.empty
        and not stage2_sorted.empty
        and np.isclose(
            float(stage2_selected_row["F1"].iloc[0]),
            float(stage2_sorted["F1"].iloc[0]),
            atol=1e-12,
        )
    )

    lambda0_step = regular_grid_step(stage2_enriched["lambda_0"])
    lambdat_step = regular_grid_step(stage2_enriched["lambda_t"])
    if np.isfinite(lambda0_step) and np.isfinite(lambdat_step):
        stage2_neighborhood = stage2_enriched.loc[
            (stage2_enriched["abs_lambda_0_distance"] <= lambda0_step + 1e-12)
            & (stage2_enriched["abs_lambda_t_distance"] <= lambdat_step + 1e-12)
        ].sort_values(
            ["abs_lambda_0_distance", "abs_lambda_t_distance", "F1"],
            ascending=[True, True, False],
            kind="mergesort",
        )
    else:
        stage2_neighborhood = stage2_selected_row.copy()

    stage2_slice_lambda0 = stage2_enriched.loc[
        np.isclose(stage2_enriched["lambda_t"].to_numpy(dtype=float), selected_lambda_t, atol=1e-12, rtol=0.0)
    ].sort_values("lambda_0", ascending=True, kind="mergesort")
    stage2_slice_lambdat = stage2_enriched.loc[
        np.isclose(stage2_enriched["lambda_0"].to_numpy(dtype=float), selected_lambda_0, atol=1e-12, rtol=0.0)
    ].sort_values("lambda_t", ascending=True, kind="mergesort")

    save_csv(stage2_enriched, output_dir / "stage2_validation_grid_enriched.csv")
    save_csv(stage2_sorted, output_dir / "stage2_grid_ranked_by_f1.csv")
    save_csv(stage2_selected_row, output_dir / "stage2_selected_row.csv")
    save_csv(stage2_neighborhood, output_dir / "stage2_selected_neighborhood.csv")
    save_csv(stage2_slice_lambda0, output_dir / "stage2_slice_fixed_lambda_t.csv")
    save_csv(stage2_slice_lambdat, output_dir / "stage2_slice_fixed_lambda_0.csv")
    save_csv(stage2_heatmap_f1.reset_index(), output_dir / "stage2_f1_heatmap.csv")
    save_csv(stage2_heatmap_precision.reset_index(), output_dir / "stage2_precision_heatmap.csv")
    save_csv(stage2_heatmap_pred_edges.reset_index(), output_dir / "stage2_prededges_heatmap.csv")

    best_stage2 = stage2_sorted.head(1)

    run_status = {
        "completed": True,
        "bundle_root": str(paths.root),
        "stage1_selected_top_k": selected_top_k,
        "stage1_selected_threshold": selected_threshold,
        "stage2_selected_lambda_0": selected_lambda_0,
        "stage2_selected_lambda_t": selected_lambda_t,
        "stage1_selected_is_best_feasible": bool(
            stage1_selected_in_feasible
            and not stage1_best.empty
            and np.isclose(
                float(stage1_selected_row["F1"].iloc[0]),
                float(stage1_best["F1"].iloc[0]),
                atol=1e-12,
            )
        ),
        "stage2_selected_is_best": bool(stage2_selected_is_best),
        "stage1_search_rows": int(len(stage1_enriched)),
        "stage2_search_rows": int(len(stage2_enriched)),
        "stage1_best_f1": safe_float(stage1_best["F1"].iloc[0]),
        "stage2_best_f1": safe_float(best_stage2["F1"].iloc[0]),
        "stage1_selected_f1": safe_float(stage1_selected_row["F1"].iloc[0]) if not stage1_selected_row.empty else float("nan"),
        "stage2_selected_f1": safe_float(stage2_selected_row["F1"].iloc[0]) if not stage2_selected_row.empty else float("nan"),
    }
    save_json(run_status, output_dir / "parameter_sensitivity_run_status.json")

    print_block(
        "Parameter Sensitivity Summary",
        {
            "stage1_selected_top_k": selected_top_k,
            "stage1_selected_f1": run_status["stage1_selected_f1"],
            "stage1_best_f1": run_status["stage1_best_f1"],
            "stage2_selected_lambda_0": selected_lambda_0,
            "stage2_selected_lambda_t": selected_lambda_t,
            "stage2_selected_f1": run_status["stage2_selected_f1"],
            "stage2_best_f1": run_status["stage2_best_f1"],
            "stage2_heatmap_shape": f"{stage2_heatmap_f1.shape[0]}x{stage2_heatmap_f1.shape[1]}",
            "output_dir": str(output_dir),
        },
    )

    print("\nPrimary output:")
    print(output_dir / "parameter_sensitivity_run_status.json")


if __name__ == "__main__":
    main()
