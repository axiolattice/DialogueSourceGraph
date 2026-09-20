#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FCN Stage 2 structural ablation: M0 / M1 / M2.

This supplementary script reuses the frozen formal Stage 1 outputs and the
formal Stage 2 exact recovery functions. It does not reimplement Stage 1 or
Stage 2 logic.

Conditions
----------
M0: local evidence only
M1: local evidence + sparse penalty (lambda_0)
M2: local evidence + sparse penalty + target competition (lambda_0, lambda_t)

Protocol
--------
- Stage 2A is fitted once on frozen Train candidates and shared by all
  conditions.
- M0 uses fixed zero penalties.
- M1 selects lambda_0 on Validation only with lambda_t fixed at zero.
- M2 selects (lambda_0, lambda_t) on Validation only.
- Test is evaluated only after the parameters are frozen.
- Exactness is audited by exhaustive subset enumeration for every split.
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_stage2_structure_ablation"
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


def verify_stage1_inputs(stage2_mod, stage1_outputs: Dict[str, object]) -> None:
    stage1_params = stage2_mod.load_and_verify_stage1_params(
        bundle_paths(stage1_outputs["paths"].root).stage1_dir
    )
    stage1_threshold = float(stage1_params["selected_threshold"])
    stage1_top_k = int(stage1_params["selected_top_k"])

    train_df = stage1_outputs["train"]
    validation_df = stage1_outputs["validation"]
    test_df = stage1_outputs["test"]

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
    stage2_mod.verify_stage1_retention(train_df, "Train", stage1_threshold, stage1_top_k)
    stage2_mod.verify_stage1_retention(
        validation_df, "Validation", stage1_threshold, stage1_top_k
    )
    stage2_mod.verify_stage1_retention(test_df, "Test", stage1_threshold, stage1_top_k)

    print_block(
        "Frozen Stage 1 Inputs",
        {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_sessions": train_df["session_id"].nunique(),
            "validation_sessions": validation_df["session_id"].nunique(),
            "test_sessions": test_df["session_id"].nunique(),
            "stage1_selected_threshold": f"{stage1_threshold:.16e}",
            "stage1_selected_top_k": stage1_top_k,
            "stage1_retention_orientation": stage1_params["retention_orientation"],
        },
    )

    return stage1_params


def build_stage2a_frames(stage2_mod, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, local_c: float):
    model = stage2_mod.fit_stage2a_model(train_df, c_value=local_c)
    train_stage2a = stage2_mod.run_stage2a(train_df, model)
    validation_stage2a = stage2_mod.run_stage2a(validation_df, model)
    test_stage2a = stage2_mod.run_stage2a(test_df, model)
    return model, train_stage2a, validation_stage2a, test_stage2a


def candidate_ranking_report(stage2_mod, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name, frame in frames.items():
        metrics = stage2_mod.stage2a_candidate_ranking_metrics(frame)
        rows.append({"split": split_name, **metrics})
    return pd.DataFrame(rows)


def write_stage2b_outputs(
    stage2_mod,
    condition: str,
    lambda0_grid: Sequence[float],
    lambdat_grid: Sequence[float],
    train_stage2a: pd.DataFrame,
    validation_stage2a: pd.DataFrame,
    test_stage2a: pd.DataFrame,
    outdir: Path,
) -> Dict[str, object]:
    condition_dir = outdir / condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    if condition == "M0":
        selected_lambda_0 = 0.0
        selected_lambda_t = 0.0
        validation_search = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "lambda_0": 0.0,
                    "lambda_t": 0.0,
                    "note": "fixed_zero_penalty",
                }
            ]
        )
    elif condition == "M1":
        selected_lambda_0, _, validation_search = stage2_mod.validation_grid_search(
            validation_stage2a,
            lambda0_grid=lambda0_grid,
            lambdat_grid=(0.0,),
        )
        selected_lambda_t = 0.0
        stage2_mod.reject_upper_grid_boundary("lambda_0", selected_lambda_0, lambda0_grid)
    elif condition == "M2":
        selected_lambda_0, selected_lambda_t, validation_search = stage2_mod.validation_grid_search(
            validation_stage2a,
            lambda0_grid=lambda0_grid,
            lambdat_grid=lambdat_grid,
        )
        stage2_mod.reject_upper_grid_boundary("lambda_0", selected_lambda_0, lambda0_grid)
        stage2_mod.reject_upper_grid_boundary("lambda_t", selected_lambda_t, lambdat_grid)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if condition == "M0":
        validation_search = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "lambda_0": selected_lambda_0,
                    "lambda_t": selected_lambda_t,
                    "note": "fixed_zero_penalty",
                }
            ]
        )

    save_csv(validation_search, condition_dir / "stage2_validation_search.csv")

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

    save_csv(final_train, condition_dir / "stage2_train_final_graph.csv")
    save_csv(final_validation, condition_dir / "stage2_validation_final_graph.csv")
    save_csv(final_test, condition_dir / "stage2_test_final_graph.csv")

    audit_rows = []
    for split_name, stage2a_df, final_df in [
        ("Train", train_stage2a, final_train),
        ("Validation", validation_stage2a, final_validation),
        ("Test", test_stage2a, final_test),
    ]:
        audit = stage2_mod.exhaustive_targetwise_audit(
            stage2a_df,
            final_df,
            lambda_0=float(selected_lambda_0),
            lambda_t=float(selected_lambda_t),
            split_name=split_name,
        )
        audit_rows.append(audit)
    audit_df = pd.DataFrame(audit_rows)
    save_csv(audit_df, condition_dir / "stage2_exactness_audit.csv")

    split_rows = []
    for split_name, frame in [
        ("Train", final_train),
        ("Validation", final_validation),
        ("Test", final_test),
    ]:
        split_rows.append(
            {
                "condition": condition,
                **stage2_mod.split_summary(split_name, frame),
            }
        )
    split_df = pd.DataFrame(split_rows)
    save_csv(split_df, condition_dir / "structured_stage2_summary.csv")

    selected_payload = {
        "condition": condition,
        "selected_lambda_0": float(selected_lambda_0),
        "selected_lambda_t": float(selected_lambda_t),
        "train_params": train_params,
        "validation_params": validation_params,
        "test_params": test_params,
    }
    save_json(selected_payload, condition_dir / "stage2_selected_params.json")

    validation_metrics = split_df.loc[split_df["split"] == "Validation"].iloc[0].to_dict()
    test_metrics = split_df.loc[split_df["split"] == "Test"].iloc[0].to_dict()

    return {
        "condition": condition,
        "selected_lambda_0": float(selected_lambda_0),
        "selected_lambda_t": float(selected_lambda_t),
        "Validation_Precision": float(validation_metrics["precision"]),
        "Validation_Recall": float(validation_metrics["recall"]),
        "Validation_F1": float(validation_metrics["f1"]),
        "Validation_AUC": float(validation_metrics.get("auc", float("nan"))),
        "Validation_AP": float(validation_metrics.get("ap", float("nan"))),
        "Validation_PredEdges": int(validation_metrics["pred_edges"]),
        "Validation_TP": int(validation_metrics["tp"]),
        "Validation_FP": int(validation_metrics["fp"]),
        "Validation_FN": int(validation_metrics["fn"]),
        "Test_Precision": float(test_metrics["precision"]),
        "Test_Recall": float(test_metrics["recall"]),
        "Test_F1": float(test_metrics["f1"]),
        "Test_AUC": float(test_metrics.get("auc", float("nan"))),
        "Test_AP": float(test_metrics.get("ap", float("nan"))),
        "Test_PredEdges": int(test_metrics["pred_edges"]),
        "Test_TP": int(test_metrics["tp"]),
        "Test_FP": int(test_metrics["fp"]),
        "Test_FN": int(test_metrics["fn"]),
        "Validation_exactness_passed": bool(audit_df.loc[audit_df["split"] == "Validation", "passed"].iloc[0]),
        "Test_exactness_passed": bool(audit_df.loc[audit_df["split"] == "Test", "passed"].iloc[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FCN Stage 2 structural ablation (M0 / M1 / M2)."
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
        help="Fixed LogisticRegression C for Stage 2A (default: 1.0).",
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
    stage1_params = verify_stage1_inputs(stage2_mod, bundle)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else bundle_paths(args.bundle_root).root.parent
        / "experiments"
        / "ablation"
        / "fcn_stage2_structure_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, train_stage2a, validation_stage2a, test_stage2a = build_stage2a_frames(
        stage2_mod,
        bundle["train"],
        bundle["validation"],
        bundle["test"],
        local_c=float(args.local_c),
    )

    save_csv(train_stage2a, output_dir / "stage2a_train_scored.csv")
    save_csv(validation_stage2a, output_dir / "stage2a_validation_scored.csv")
    save_csv(test_stage2a, output_dir / "stage2a_test_scored.csv")
    save_csv(
        stage2_mod.extract_stage2a_coefficients(model),
        output_dir / "stage2a_feature_coefficients.csv",
    )

    stage2a_ranking = candidate_ranking_report(
        stage2_mod,
        {
            "Train": train_stage2a,
            "Validation": validation_stage2a,
            "Test": test_stage2a,
        },
    )
    save_csv(stage2a_ranking, output_dir / "stage2a_candidate_ranking.csv")

    print_block(
        "Stage 2A Candidate Ranking",
        {
            f"{row['split']}_candidate_edges": int(row["candidate_edges"])
            for _, row in stage2a_ranking.iterrows()
        },
    )

    summary_rows = []
    for condition in ("M0", "M1", "M2"):
        print_block(
            f"Stage 2 Structural Ablation: {condition}",
            {
                "local_c": float(args.local_c),
                "lambda0_grid": grid_to_arg(args.lambda0_grid),
                "lambdat_grid": grid_to_arg(args.lambdat_grid),
            },
        )
        if condition == "M0":
            row = write_stage2b_outputs(
                stage2_mod,
                condition=condition,
                lambda0_grid=(0.0,),
                lambdat_grid=(0.0,),
                train_stage2a=train_stage2a,
                validation_stage2a=validation_stage2a,
                test_stage2a=test_stage2a,
                outdir=output_dir,
            )
        elif condition == "M1":
            row = write_stage2b_outputs(
                stage2_mod,
                condition=condition,
                lambda0_grid=args.lambda0_grid,
                lambdat_grid=(0.0,),
                train_stage2a=train_stage2a,
                validation_stage2a=validation_stage2a,
                test_stage2a=test_stage2a,
                outdir=output_dir,
            )
        else:
            row = write_stage2b_outputs(
                stage2_mod,
                condition=condition,
                lambda0_grid=args.lambda0_grid,
                lambdat_grid=args.lambdat_grid,
                train_stage2a=train_stage2a,
                validation_stage2a=validation_stage2a,
                test_stage2a=test_stage2a,
                outdir=output_dir,
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, output_dir / "stage2_structure_ablation_summary.csv")

    split_rows = []
    for condition in ("M0", "M1", "M2"):
        condition_dir = output_dir / condition
        split_summary = pd.read_csv(
            condition_dir / "structured_stage2_summary.csv",
            low_memory=False,
        )
        split_rows.append(split_summary)
    save_csv(pd.concat(split_rows, ignore_index=True), output_dir / "stage2_structure_ablation_split_summary.csv")

    run_manifest = {
        "completed": True,
        "bundle_root": str(bundle_paths(args.bundle_root).root),
        "output_dir": str(output_dir),
        "stage1_selected_threshold": float(stage1_params["selected_threshold"]),
        "stage1_selected_top_k": int(stage1_params["selected_top_k"]),
        "conditions": ["M0", "M1", "M2"],
    }
    save_json(run_manifest, output_dir / "stage2_structure_ablation_run_status.json")

    print_block("Stage 2 Structural Ablation Summary", summary_df.to_dict(orient="list"))


if __name__ == "__main__":
    main()
