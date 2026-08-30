#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCN Stage 2A Ablation Experiment

Purpose
-------
Compare three Stage 2A local-evidence specifications under exactly the same
frozen Stage 1 inputs and exact Stage 2B structured-recovery protocol:

    1. stage1_only : stage1_prob only
    2. core5       : compact five-feature specification
    3. full23      : legacy full feature specification

Protocol
--------
- Fixed 176 / 44 / 55 Train / Validation / Test split inherited from Stage 1.
- Each Stage 2A model is fitted on Train only.
- Stage 2B (lambda_0, lambda_t) is selected on Validation only for each
  ablation condition.
- Test is evaluated once after selection.
- The exact target-wise Stage 2B solver is shared across all conditions.

Important
---------
This is an ABLATION script, not the formal FCN implementation.
The formal method remains FCN_stage2_structured.py with one-dimensional
stage1_prob recalibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Import the frozen exact solver / metrics / input checks from the formal script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import FCN_stage2_structured as formal


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# stage1_only is the final one-dimensional recalibration specification.
STAGE1_ONLY = ["stage1_prob"]

# Compact 5-feature ablation used in the prior experiment:
# Stage 1 learned score + temporal distance + basic text-length cues.
CORE5 = [
    "stage1_prob",
    "distance_num",
    "q1_len",
    "a1_len",
    "q2_len",
]

# Full-23 is intentionally resolved from the columns present in the Stage 1
# outputs. The exact historical names must be supplied or already present.
#
# To prevent silently inventing/altering the old feature set, this script
# requires --full23-features for the full23 condition unless a sidecar JSON
# named stage2A_full23_features.json exists beside the script.
#
# JSON format:
# {"features": ["stage1_prob", "...", "..."]}


def parse_feature_list(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Feature list cannot be empty.")
    return vals


def resolve_full23_features(args) -> List[str]:
    if args.full23_features:
        features = list(args.full23_features)
    else:
        features = [
            "stage1_prob",
            "stage1_complaint_prob",
            "stage1_challenge_prob",
            "stage1_structured_prob",
            "answer_unsatisfied_score",
            "q2_pressure_score",
            "q2_challenge_score",
            "q2_explain_only_score",
            "evidence_score",
            "structured_bridge_rule_score",
            "structured_answer_state_score",
            "structured_answer_commitment_score",
            "structured_answer_numeric_score",
            "structured_q2_continuation_need_score",
            "structured_followup_action_score",
            "structured_reference_score",
            "structured_shared_object_score",
            "structured_a1_shared_object_score",
            "q1_len",
            "a1_len",
            "q2_len",
            "distance_num",
            "topic_shift_score",
        ]

    if len(features) != 23:
        raise RuntimeError(
            f"full23 must contain exactly 23 feature columns; got {len(features)}."
        )
    if len(set(features)) != 23:
        raise RuntimeError("full23 feature list contains duplicates.")
    return features


def ensure_features(df: pd.DataFrame, features: List[str], condition: str) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{condition}: missing required columns: {missing}. "
            "Do not substitute other columns; use the exact archived feature set."
        )


def load_stage1_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 output file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def retained_candidate_ranking_metrics(df: pd.DataFrame) -> Dict[str, float]:
    retained = formal.get_numeric(df, "stage2_graph_input_pred", 0.0).astype(int).eq(1)
    y = formal.get_numeric(df.loc[retained], "edge_label", 0.0).astype(int).to_numpy()
    s = formal.get_numeric(df.loc[retained], "stage2_local_evidence", 0.0).to_numpy()
    return {
        "candidate_edges": int(retained.sum()),
        "candidate_true_edges": int(y.sum()),
        "auc": formal.safe_auc(y, s),
        "ap": formal.safe_ap(y, s),
    }


def final_split_metrics(split_name: str, df: pd.DataFrame) -> Dict[str, object]:
    y_true = formal.get_numeric(df, "edge_label", 0.0).astype(int).to_numpy()
    y_pred = formal.get_numeric(
        df,
        "final_graph_pred_edge",
        formal.get_numeric(df, "stage2_pred_edge", 0.0),
    ).astype(int).to_numpy()
    y_score = formal.get_numeric(
        df,
        "final_graph_score",
        formal.get_numeric(df, "stage2_score", 0.0),
    ).to_numpy()
    m = formal.binary_metrics(y_true, y_pred, y_score=y_score)
    return {
        "split": split_name,
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "AP": m["ap"],
        "AUC": m["auc"],
        "PredEdges": m["pred_edges"],
        "TP": m["tp"],
        "FP": m["fp"],
        "FN": m["fn"],
    }


def fit_stage2a(
    train_df: pd.DataFrame,
    features: List[str],
    c_value: float,
):
    retained = formal.get_numeric(
        train_df, "stage1_pred_edge", 0.0
    ).astype(int).eq(1)

    fit_df = train_df.loc[retained].copy()
    if fit_df.empty:
        raise RuntimeError("No retained Stage 1 Train candidates.")

    ensure_features(fit_df, features, "Train")

    X = (
        fit_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    y = formal.get_numeric(
        fit_df, "edge_label", 0.0
    ).astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError("Retained Train candidates contain only one class.")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=formal.SEED,
        ),
    )
    model.fit(X, y)
    return model


def apply_stage2a(
    df: pd.DataFrame,
    model,
    features: List[str],
) -> pd.DataFrame:
    ensure_features(df, features, "Apply")

    out = df.copy().reset_index(drop=True)
    retained = formal.get_numeric(
        out, "stage1_pred_edge", 0.0
    ).astype(int).eq(1)

    X = (
        out[features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    out["stage2_graph_input_pred"] = retained.astype(int)
    out["stage2a_local_prob"] = model.predict_proba(X)[:, 1]
    out["stage2_recalibrated_prob"] = model.predict_proba(X)[:, 1]
    out["stage2a_local_evidence"] = model.decision_function(X)
    out["stage2_local_evidence"] = out["stage2a_local_evidence"]
    # Compatibility aliases for downstream exact-solver utilities.
    out["stage2_selection_score"] = out["stage2_recalibrated_prob"]
    out["stage2_pred_edge"] = retained.astype(int)
    out["stage2_score"] = out["stage2_recalibrated_prob"].where(retained, 0.0)
    return out


def run_condition(
    name: str,
    features: List[str],
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    c_value: float,
    lambda0_grid,
    lambdat_grid,
    outdir: Path,
) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"Stage 2A ablation: {name}")
    print("=" * 100)
    print("features:", ", ".join(features))

    model = fit_stage2a(train_raw, features, c_value)

    train = apply_stage2a(train_raw, model, features)
    val = apply_stage2a(val_raw, model, features)
    test = apply_stage2a(test_raw, model, features)

    ranking = {}
    for split_name, frame in (
        ("Train", train), ("Validation", val), ("Test", test)
    ):
        ranking[split_name] = retained_candidate_ranking_metrics(frame)
        r = ranking[split_name]
        print(
            f"{split_name} Stage2A: "
            f"AUC={r['auc']:.6f}, AP={r['ap']:.6f}, "
            f"candidates={r['candidate_edges']}, true={r['candidate_true_edges']}"
        )

    lambda_0, lambda_t, search = formal.validation_grid_search(
        val,
        lambda0_grid=lambda0_grid,
        lambdat_grid=lambdat_grid,
    )

    print(
        f"Validation-selected Stage2B: "
        f"lambda_0={lambda_0:.6f}, lambda_t={lambda_t:.6f}"
    )

    final_train, _ = formal.exact_targetwise_recovery(train, lambda_0, lambda_t)
    final_val, _ = formal.exact_targetwise_recovery(val, lambda_0, lambda_t)
    final_test, _ = formal.exact_targetwise_recovery(test, lambda_0, lambda_t)

    rows = []
    for split_name, frame in (
        ("Train", final_train),
        ("Validation", final_val),
        ("Test", final_test),
    ):
        row = final_split_metrics(split_name, frame)
        rows.append(row)
        print(
            f"{split_name} final: "
            f"P={row['Precision']:.6f}, R={row['Recall']:.6f}, "
            f"F1={row['F1']:.6f}, AP={row['AP']:.6f}, "
            f"PredEdges={row['PredEdges']}"
        )

    condition_dir = outdir / name
    condition_dir.mkdir(parents=True, exist_ok=True)

    search.to_csv(
        condition_dir / "stage2_validation_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(rows).to_csv(
        condition_dir / "structured_stage2_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_test.to_csv(
        condition_dir / "stage2_test_final_graph.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_row = next(r for r in rows if r["split"] == "Test")
    val_row = next(r for r in rows if r["split"] == "Validation")

    return {
        "condition": name,
        "n_features": len(features),
        "features": "|".join(features),
        "Validation_Stage2A_AUC": ranking["Validation"]["auc"],
        "Validation_Stage2A_AP": ranking["Validation"]["ap"],
        "Test_Stage2A_AUC": ranking["Test"]["auc"],
        "Test_Stage2A_AP": ranking["Test"]["ap"],
        "lambda_0": lambda_0,
        "lambda_t": lambda_t,
        "Validation_Precision": val_row["Precision"],
        "Validation_Recall": val_row["Recall"],
        "Validation_F1": val_row["F1"],
        "Test_Precision": test_row["Precision"],
        "Test_Recall": test_row["Recall"],
        "Test_F1": test_row["F1"],
        "Test_AP": test_row["AP"],
        "Test_AUC": test_row["AUC"],
        "Test_PredEdges": test_row["PredEdges"],
        "Test_TP": test_row["TP"],
        "Test_FP": test_row["FP"],
        "Test_FN": test_row["FN"],
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="FCN Stage 2A three-way ablation under the frozen final protocol."
    )
    p.add_argument("--stage1-outdir", required=True)
    p.add_argument(
        "--outdir",
        default=str(HERE / "fcn_outputs_stage2A_ablation"),
    )
    p.add_argument(
        "--recalibration-c",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--lambda0-grid",
        type=formal.parse_float_grid,
        default=formal.DEFAULT_LAMBDA0_GRID,
    )
    p.add_argument(
        "--lambdat-grid",
        type=formal.parse_float_grid,
        default=formal.DEFAULT_LAMBDAT_GRID,
    )
    p.add_argument(
        "--full23-features",
        type=parse_feature_list,
        default=None,
        help="Exact comma-separated archived full-23 feature names.",
    )
    p.add_argument(
        "--skip-full23",
        action="store_true",
        help="Run only stage1_only and core5.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    stage1_dir = Path(args.stage1_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw = load_stage1_split(stage1_dir / "stage1_train_scored.csv")
    val_raw = load_stage1_split(stage1_dir / "stage1_validation_scored.csv")
    test_raw = load_stage1_split(stage1_dir / "stage1_test_scored.csv")

    formal.verify_three_way_inputs(train_raw, val_raw, test_raw)

    conditions = [
        ("stage1_only", STAGE1_ONLY),
        ("core5", CORE5),
    ]
    if not args.skip_full23:
        conditions.append(("full23", resolve_full23_features(args)))

    results = []
    for name, features in conditions:
        results.append(
            run_condition(
                name=name,
                features=features,
                train_raw=train_raw,
                val_raw=val_raw,
                test_raw=test_raw,
                c_value=args.recalibration_c,
                lambda0_grid=args.lambda0_grid,
                lambdat_grid=args.lambdat_grid,
                outdir=outdir,
            )
        )

    summary = pd.DataFrame(results)
    summary.to_csv(
        outdir / "stage2A_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("Stage 2A Ablation Summary")
    print("=" * 100)
    print(
        summary[
            [
                "condition",
                "n_features",
                "Validation_Stage2A_AP",
                "lambda_0",
                "lambda_t",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Test_AP",
                "Test_PredEdges",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print("\nPrimary output:")
    print(outdir / "stage2A_ablation_summary.csv")


if __name__ == "__main__":
    main()
