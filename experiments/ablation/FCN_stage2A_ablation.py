#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FCN Stage 2A ablation under the frozen formal protocol.

This supplementary script isolates whether the formal one-dimensional Stage 2A
recalibration adds value over using the frozen Stage 1 score directly.

Conditions
----------
raw_stage1_prob:
    Use the frozen Stage 1 probability directly as the local evidence input.

formal_stage2a_lr:
    Fit the formal one-dimensional logistic recalibration on Train only and
    reuse it for Validation and Test.

Protocol
--------
- The frozen Stage 1 split is verified once and never modified.
- Stage 2B penalties are selected on Validation only for each condition.
- Test is evaluated only after the Stage 2A choice and Stage 2B penalties are
  frozen.
- Exact target-wise recovery is reused from the formal Stage 2 module.
- No higher-dimensional local feature set is reintroduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.common import bundle_paths, load_stage1_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STAGE2_MODULE_PATH = PROJECT_ROOT / "stage2" / "FCN_stage2_structured.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_stage2A_ablation"
DEFAULT_LAMBDA0_GRID = tuple(np.round(np.arange(0.00, 0.801, 0.05), 10))
DEFAULT_LAMBDAT_GRID = tuple(np.round(np.arange(0.00, 0.501, 0.05), 10))


def parse_float_grid(value: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val < 0.0:
            raise argparse.ArgumentTypeError("Penalty values must be non-negative.")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    return tuple(sorted(set(vals)))


def grid_to_arg(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):g}" for v in values)


def load_formal_stage2_module():
    spec = importlib.util.spec_from_file_location(
        "formal_fcn_stage2_structured",
        STAGE2_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load formal Stage 2 module: {STAGE2_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def load_stage1_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 output file: {path}")
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def verify_stage1_inputs(
    stage2_mod,
    stage1_dir: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    stage2_mod.verify_stage1_partition(
        train_df,
        "Train",
        stage2_mod.EXPECTED_TRAIN_ROWS,
        stage2_mod.EXPECTED_TRAIN_SESSIONS,
        stage2_mod.EXPECTED_TRAIN_EDGES,
    )
    stage2_mod.verify_stage1_partition(
        validation_df,
        "Validation",
        stage2_mod.EXPECTED_VALIDATION_ROWS,
        stage2_mod.EXPECTED_VALIDATION_SESSIONS,
        stage2_mod.EXPECTED_VALIDATION_EDGES,
    )
    stage2_mod.verify_stage1_partition(
        test_df,
        "Test",
        stage2_mod.EXPECTED_TEST_ROWS,
        stage2_mod.EXPECTED_TEST_SESSIONS,
        stage2_mod.EXPECTED_TEST_EDGES,
    )
    stage2_mod.verify_disjoint_sessions(
        [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]
    )

    stage1_params = stage2_mod.load_and_verify_stage1_params(stage1_dir)
    threshold = float(stage1_params["selected_threshold"])
    top_k = int(stage1_params["selected_top_k"])
    stage2_mod.verify_stage1_retention(train_df, "Train", threshold, top_k)
    stage2_mod.verify_stage1_retention(validation_df, "Validation", threshold, top_k)
    stage2_mod.verify_stage1_retention(test_df, "Test", threshold, top_k)

    print_block(
        "Frozen Stage 1 Inputs",
        {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_sessions": train_df["session_id"].nunique(),
            "validation_sessions": validation_df["session_id"].nunique(),
            "test_sessions": test_df["session_id"].nunique(),
            "stage1_selected_threshold": f"{threshold:.16e}",
            "stage1_selected_top_k": top_k,
            "stage1_retention_orientation": stage1_params["retention_orientation"],
        },
    )
    return stage1_params


def retained_candidate_ranking_metrics(stage2_mod, df: pd.DataFrame) -> Dict[str, float]:
    return stage2_mod.stage2a_candidate_ranking_metrics(df)


def apply_raw_stage2a(stage2_mod, df: pd.DataFrame) -> pd.DataFrame:
    """Use the frozen Stage 1 probability directly, without trainable recalibration."""
    out = df.copy().reset_index(drop=True)
    candidate_mask = stage2_mod.get_numeric(out, "stage1_pred_edge", 0.0).astype(int) == 1
    eps = np.finfo(float).eps
    local_prob = stage2_mod.get_numeric(out, "stage1_prob", 0.0).astype(float)
    local_prob = np.clip(local_prob.to_numpy(dtype=float), eps, 1.0 - eps)
    local_evidence = stage2_mod.logit(local_prob)

    out["stage2_graph_input_pred"] = candidate_mask.astype(int)
    out["stage2a_local_prob"] = local_prob
    out["stage2a_local_evidence"] = local_evidence
    out["stage2_local_evidence"] = out["stage2a_local_evidence"]
    out["stage2_selection_score"] = out["stage2a_local_prob"]
    out["stage2_pred_edge"] = candidate_mask.astype(int)
    out["stage2_score"] = np.where(
        candidate_mask,
        out["stage2a_local_prob"].to_numpy(dtype=float),
        0.0,
    )
    return out


def apply_formal_stage2a(stage2_mod, df: pd.DataFrame, model) -> pd.DataFrame:
    return stage2_mod.run_stage2a(df, model)


def fit_formal_stage2a(stage2_mod, train_df: pd.DataFrame, local_c: float):
    return stage2_mod.fit_stage2a_model(train_df, c_value=local_c)


def final_metrics(stage2_mod, split_name: str, frame: pd.DataFrame) -> Dict[str, object]:
    row = stage2_mod.split_summary(split_name, frame)
    return {
        "split": split_name,
        "Precision": float(row["precision"]),
        "Recall": float(row["recall"]),
        "F1": float(row["f1"]),
        "AUC": float(row.get("auc", float("nan"))),
        "AP": float(row.get("ap", float("nan"))),
        "PredEdges": int(row["pred_edges"]),
        "TP": int(row["tp"]),
        "FP": int(row["fp"]),
        "FN": int(row["fn"]),
    }


def run_condition(
    stage2_mod,
    name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outdir: Path,
    lambda0_grid: Sequence[float],
    lambdat_grid: Sequence[float],
    local_c: float,
) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"Stage 2A ablation: {name}")
    print("=" * 100)

    if name == "raw_stage1_prob":
        model = None
        train_stage2a = apply_raw_stage2a(stage2_mod, train_df)
        validation_stage2a = apply_raw_stage2a(stage2_mod, validation_df)
        test_stage2a = apply_raw_stage2a(stage2_mod, test_df)
        coefficients = pd.DataFrame(
            [
                {
                    "feature": "stage1_prob",
                    "coefficient": np.nan,
                    "abs_coefficient": np.nan,
                    "note": "no_trainable_recalibration",
                }
            ]
        )
    elif name == "formal_stage2a_lr":
        model = fit_formal_stage2a(stage2_mod, train_df, local_c=local_c)
        train_stage2a = apply_formal_stage2a(stage2_mod, train_df, model)
        validation_stage2a = apply_formal_stage2a(stage2_mod, validation_df, model)
        test_stage2a = apply_formal_stage2a(stage2_mod, test_df, model)
        coefficients = stage2_mod.extract_stage2a_coefficients(model)
    else:
        raise ValueError(f"Unknown condition: {name}")

    ranking_rows = []
    for split_name, frame in (
        ("Train", train_stage2a),
        ("Validation", validation_stage2a),
        ("Test", test_stage2a),
    ):
        metrics = retained_candidate_ranking_metrics(stage2_mod, frame)
        ranking_rows.append({"split": split_name, **metrics})
    ranking_df = pd.DataFrame(ranking_rows)

    if name == "raw_stage1_prob":
        selected_lambda_0, selected_lambda_t, validation_search = stage2_mod.validation_grid_search(
            validation_stage2a,
            lambda0_grid=lambda0_grid,
            lambdat_grid=lambdat_grid,
        )
    else:
        selected_lambda_0, selected_lambda_t, validation_search = stage2_mod.validation_grid_search(
            validation_stage2a,
            lambda0_grid=lambda0_grid,
            lambdat_grid=lambdat_grid,
        )

    stage2_mod.reject_upper_grid_boundary("lambda_0", selected_lambda_0, lambda0_grid)
    stage2_mod.reject_upper_grid_boundary("lambda_t", selected_lambda_t, lambdat_grid)

    final_train, train_params = stage2_mod.exact_targetwise_recovery(
        train_stage2a,
        lambda_0=float(selected_lambda_0),
        lambda_t=float(selected_lambda_t),
    )
    final_validation, validation_params = stage2_mod.exact_targetwise_recovery(
        validation_stage2a,
        lambda_0=float(selected_lambda_0),
        lambda_t=float(selected_lambda_t),
    )
    final_test, test_params = stage2_mod.exact_targetwise_recovery(
        test_stage2a,
        lambda_0=float(selected_lambda_0),
        lambda_t=float(selected_lambda_t),
    )

    exactness_rows = []
    for split_name, stage2a_df, final_df in [
        ("Train", train_stage2a, final_train),
        ("Validation", validation_stage2a, final_validation),
        ("Test", test_stage2a, final_test),
    ]:
        exactness_rows.append(
            stage2_mod.exhaustive_targetwise_audit(
                stage2a_df,
                final_df,
                lambda_0=float(selected_lambda_0),
                lambda_t=float(selected_lambda_t),
                split_name=split_name,
            )
        )
    exactness_df = pd.DataFrame(exactness_rows)

    split_rows = []
    for split_name, frame in [
        ("Train", final_train),
        ("Validation", final_validation),
        ("Test", final_test),
    ]:
        split_rows.append(final_metrics(stage2_mod, split_name, frame))
    split_df = pd.DataFrame(split_rows)

    condition_dir = outdir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    save_csv(train_stage2a, condition_dir / "stage2a_train_scored.csv")
    save_csv(validation_stage2a, condition_dir / "stage2a_validation_scored.csv")
    save_csv(test_stage2a, condition_dir / "stage2a_test_scored.csv")
    save_csv(ranking_df, condition_dir / "stage2a_candidate_ranking.csv")
    save_csv(validation_search, condition_dir / "stage2_validation_search.csv")
    save_csv(final_train, condition_dir / "stage2_train_final_graph.csv")
    save_csv(final_validation, condition_dir / "stage2_validation_final_graph.csv")
    save_csv(final_test, condition_dir / "stage2_test_final_graph.csv")
    save_csv(exactness_df, condition_dir / "stage2_exactness_audit.csv")
    save_csv(split_df, condition_dir / "structured_stage2_summary.csv")
    save_csv(coefficients, condition_dir / "stage2a_feature_coefficients.csv")

    save_json(
        {
            "condition": name,
            "selected_lambda_0": float(selected_lambda_0),
            "selected_lambda_t": float(selected_lambda_t),
            "train_params": train_params,
            "validation_params": validation_params,
            "test_params": test_params,
            "stage2a_type": "raw_stage1_prob" if name == "raw_stage1_prob" else "formal_stage2a_lr",
        },
        condition_dir / "stage2_selected_params.json",
    )

    val_row = split_df.loc[split_df["split"] == "Validation"].iloc[0].to_dict()
    test_row = split_df.loc[split_df["split"] == "Test"].iloc[0].to_dict()
    rank_val = ranking_df.loc[ranking_df["split"] == "Validation"].iloc[0].to_dict()
    rank_test = ranking_df.loc[ranking_df["split"] == "Test"].iloc[0].to_dict()

    print_block(
        f"Stage 2A ranking: {name}",
        {
            "Validation_candidate_edges": int(rank_val["candidate_edges"]),
            "Validation_true_edges": int(rank_val["candidate_true_edges"]),
            "Validation_AUC": float(rank_val["auc"]),
            "Validation_AP": float(rank_val["ap"]),
            "Test_candidate_edges": int(rank_test["candidate_edges"]),
            "Test_true_edges": int(rank_test["candidate_true_edges"]),
            "Test_AUC": float(rank_test["auc"]),
            "Test_AP": float(rank_test["ap"]),
        },
    )
    print_block(
        f"Stage 2B result: {name}",
        {
            "selected_lambda_0": float(selected_lambda_0),
            "selected_lambda_t": float(selected_lambda_t),
            "Validation_Precision": float(val_row["Precision"]),
            "Validation_Recall": float(val_row["Recall"]),
            "Validation_F1": float(val_row["F1"]),
            "Validation_PredEdges": int(val_row["PredEdges"]),
            "Test_Precision": float(test_row["Precision"]),
            "Test_Recall": float(test_row["Recall"]),
            "Test_F1": float(test_row["F1"]),
            "Test_AUC": float(test_row["AUC"]),
            "Test_AP": float(test_row["AP"]),
            "Test_PredEdges": int(test_row["PredEdges"]),
        },
    )

    return {
        "condition": name,
        "stage2a_type": "raw_stage1_prob" if name == "raw_stage1_prob" else "formal_stage2a_lr",
        "Validation_Stage2A_AUC": float(rank_val["auc"]),
        "Validation_Stage2A_AP": float(rank_val["ap"]),
        "Test_Stage2A_AUC": float(rank_test["auc"]),
        "Test_Stage2A_AP": float(rank_test["ap"]),
        "selected_lambda_0": float(selected_lambda_0),
        "selected_lambda_t": float(selected_lambda_t),
        "Validation_Precision": float(val_row["Precision"]),
        "Validation_Recall": float(val_row["Recall"]),
        "Validation_F1": float(val_row["F1"]),
        "Validation_AUC": float(val_row["AUC"]),
        "Validation_AP": float(val_row["AP"]),
        "Validation_PredEdges": int(val_row["PredEdges"]),
        "Validation_TP": int(val_row["TP"]),
        "Validation_FP": int(val_row["FP"]),
        "Validation_FN": int(val_row["FN"]),
        "Test_Precision": float(test_row["Precision"]),
        "Test_Recall": float(test_row["Recall"]),
        "Test_F1": float(test_row["F1"]),
        "Test_AUC": float(test_row["AUC"]),
        "Test_AP": float(test_row["AP"]),
        "Test_PredEdges": int(test_row["PredEdges"]),
        "Test_TP": int(test_row["TP"]),
        "Test_FP": int(test_row["FP"]),
        "Test_FN": int(test_row["FN"]),
        "Validation_exactness_passed": bool(
            exactness_df.loc[exactness_df["split"] == "Validation", "passed"].iloc[0]
        ),
        "Test_exactness_passed": bool(
            exactness_df.loc[exactness_df["split"] == "Test", "passed"].iloc[0]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FCN Stage 2A ablation under the frozen formal protocol."
    )
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
        help="Directory for supplementary ablation outputs.",
    )
    parser.add_argument(
        "--local-c",
        type=float,
        default=1.0,
        help="Fixed LogisticRegression C for the formal Stage 2A condition.",
    )
    parser.add_argument(
        "--lambda0-grid",
        type=parse_float_grid,
        default=DEFAULT_LAMBDA0_GRID,
        help="Comma-separated lambda_0 validation grid.",
    )
    parser.add_argument(
        "--lambdat-grid",
        type=parse_float_grid,
        default=DEFAULT_LAMBDAT_GRID,
        help="Comma-separated lambda_t validation grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2_mod = load_formal_stage2_module()

    bundle = load_stage1_outputs(args.bundle_root)
    stage1_params = verify_stage1_inputs(
        stage2_mod,
        bundle_paths(args.bundle_root).stage1_dir,
        bundle["train"],
        bundle["validation"],
        bundle["test"],
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else bundle_paths(args.bundle_root).root.parent
        / "experiments"
        / "ablation"
        / "fcn_stage2A_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record the frozen Stage 1 inputs once for the entire supplementary run.
    save_json(
        {
            "bundle_root": str(bundle_paths(args.bundle_root).root),
            "stage1_selected_threshold": float(stage1_params["selected_threshold"]),
            "stage1_selected_top_k": int(stage1_params["selected_top_k"]),
            "stage1_retention_orientation": stage1_params["retention_orientation"],
            "conditions": ["raw_stage1_prob", "formal_stage2a_lr"],
        },
        output_dir / "stage2A_ablation_run_status.json",
    )

    results = []
    for condition in ("raw_stage1_prob", "formal_stage2a_lr"):
        results.append(
            run_condition(
                stage2_mod=stage2_mod,
                name=condition,
                train_df=bundle["train"],
                validation_df=bundle["validation"],
                test_df=bundle["test"],
                outdir=output_dir,
                lambda0_grid=args.lambda0_grid,
                lambdat_grid=args.lambdat_grid,
                local_c=float(args.local_c),
            )
        )

    summary_df = pd.DataFrame(results)
    save_csv(summary_df, output_dir / "stage2A_ablation_summary.csv")

    print_block("Stage 2A Ablation Summary", summary_df.to_dict(orient="list"))
    print("\nPrimary output:")
    print(output_dir / "stage2A_ablation_summary.csv")


if __name__ == "__main__":
    main()
