#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FCN Stage 2: learned local evidence and exact structured graph recovery.

Input
-----
This script consumes the frozen Stage 1 outputs:

    stage1_train_scored.csv
    stage1_validation_scored.csv
    stage1_test_scored.csv

with mutually disjoint 176 / 44 / 55 Train / Validation / Test sessions.

Stage 2A: learned local edge evidence
-------------------------------------
A class-balanced logistic-regression scorer is fitted once on retained Stage 1
Train candidates using a one-dimensional recalibration feature:

    stage1_prob.

For candidate edge (i,j), Stage 2A estimates a local edge score

    p_ij = P(y_ij = 1 | x_ij),

and converts it to additive evidence

    r_ij = log(p_ij / (1 - p_ij)).

Stage 2A performs no graph pruning.

Stage 2B: exact target-wise graph recovery
------------------------------------------
The final graph maximizes

    sum_(i,j) r_ij z_ij
    - lambda_0 sum_(i,j) z_ij
    - lambda_t sum_j C(d_in(j), 2).

Because there is no source-side coupling term, the objective decomposes
exactly by target query. For each target j, candidate edges are sorted by
local evidence and the optimal prefix length k is selected by exhaustive
evaluation of the exact target-wise objective.

Hyperparameter protocol
-----------------------
- Stage 2A is fitted on Train only and is not refitted after validation.
- lambda_0 and lambda_t are selected on Validation only.
- The frozen Stage 2A scorer and Stage 2B parameters are applied once to Test.
- Test labels are never used for fitting or hyperparameter selection.

Legacy SVMP rules, graph caps, budgets, hand-written structural bonuses,
message passing, and greedy graph selection are not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_STAGE1_OUTDIR = PROJECT_DIR / "stage1" / "fcn_outputs_stage1_final"
DEFAULT_STAGE2_OUTDIR = SCRIPT_DIR / "fcn_outputs_stage2_final"

EXPECTED_TRAIN_SESSIONS = 176
EXPECTED_VALIDATION_SESSIONS = 44
EXPECTED_TEST_SESSIONS = 55
SEED = 42

# Formal Stage 2A feature set after ablation: one-dimensional recalibration.
LOCAL_EVIDENCE_FEATURES: Tuple[str, ...] = (
    "stage1_prob",
)

# Broad, regular validation grids centered on the scale used in the earlier
# structured-objective experiments. They can be overridden from the CLI.
DEFAULT_LAMBDA0_GRID = tuple(np.round(np.arange(0.00, 0.801, 0.05), 10))
DEFAULT_LAMBDAT_GRID = tuple(np.round(np.arange(0.00, 0.501, 0.05), 10))


# =============================================================================
# Utilities
# =============================================================================

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


def get_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def sigmoid(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))


def logit(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(
    y_true,
    y_pred,
    y_score=None,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    out: Dict[str, float] = {
        "total": int(len(y_true_arr)),
        "true_edges": int(y_true_arr.sum()),
        "pred_edges": int(y_pred_arr.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
    }

    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=float)
        out["auc"] = safe_auc(y_true_arr, score_arr)
        out["ap"] = safe_ap(y_true_arr, score_arr)

    return out


def ranking_metrics(y_true, y_score) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    return {
        "total": int(len(y)),
        "true_edges": int(y.sum()),
        "auc": safe_auc(y, s),
        "ap": safe_ap(y, s),
    }


def print_block(title: str, values: Dict[str, object]) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    for key, value in values.items():
        if isinstance(value, (float, np.floating)):
            if np.isnan(float(value)):
                print(f"{key}: nan")
            else:
                print(f"{key}: {float(value):.6f}")
        else:
            print(f"{key}: {value}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def verify_three_way_inputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    required = {"session_id", "Q1_row", "Q2_row", "edge_label", "stage1_pred_edge"}
    for name, df in (
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ):
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{name} Stage 1 file is missing columns: {missing}")

    counts = (
        int(train_df["session_id"].nunique()),
        int(validation_df["session_id"].nunique()),
        int(test_df["session_id"].nunique()),
    )
    expected = (
        EXPECTED_TRAIN_SESSIONS,
        EXPECTED_VALIDATION_SESSIONS,
        EXPECTED_TEST_SESSIONS,
    )
    if counts != expected:
        raise RuntimeError(
            f"Stage 1 inputs do not match frozen 176/44/55 split: "
            f"observed={counts}, expected={expected}."
        )

    train_ids = set(train_df["session_id"].astype(str))
    validation_ids = set(validation_df["session_id"].astype(str))
    test_ids = set(test_df["session_id"].astype(str))
    if train_ids & validation_ids:
        raise RuntimeError("Train and Validation sessions overlap.")
    if train_ids & test_ids:
        raise RuntimeError("Train and Test sessions overlap.")
    if validation_ids & test_ids:
        raise RuntimeError("Validation and Test sessions overlap.")


# =============================================================================
# Stage 2A -- learned local edge evidence
# =============================================================================

def stage2a_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct strictly edge-local Stage 2A features."""
    return pd.DataFrame(
        {
            col: get_numeric(df, col, 0.0)
            for col in LOCAL_EVIDENCE_FEATURES
        },
        index=df.index,
    )


def fit_stage2a_model(
    train_df: pd.DataFrame,
    c_value: float = 1.0,
):
    """Fit Stage 2A once on retained Stage 1 Train candidates."""
    candidate_mask = get_numeric(
        train_df,
        "stage1_pred_edge",
        0.0,
    ).astype(int) == 1

    candidate_df = train_df.loc[candidate_mask].copy()
    if candidate_df.empty:
        raise RuntimeError("No retained Stage 1 candidates in Train.")
    if "edge_label" not in candidate_df.columns:
        raise RuntimeError("Train file has no edge_label column.")

    y = get_numeric(candidate_df, "edge_label", 0.0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Retained Train candidates contain only one class.")

    X = stage2a_feature_frame(candidate_df)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ),
    )
    model.fit(X, y)
    model.stage2a_feature_names_ = np.asarray(
        LOCAL_EVIDENCE_FEATURES,
        dtype=object,
    )
    return model


def stage2a_probability(
    df: pd.DataFrame,
    model,
) -> pd.Series:
    X = stage2a_feature_frame(df)
    feature_names = list(
        getattr(model, "stage2a_feature_names_", LOCAL_EVIDENCE_FEATURES)
    )
    prob = model.predict_proba(X[feature_names])[:, 1]
    return pd.Series(
        np.clip(prob, 1e-6, 1.0 - 1e-6),
        index=df.index,
        dtype=float,
    )


def run_stage2a(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Attach Stage 2A evidence; Stage 2A itself performs no pruning."""
    out = df.copy().reset_index(drop=True)

    candidate_mask = (
        get_numeric(out, "stage1_pred_edge", 0.0).astype(int) == 1
    )

    local_prob = stage2a_probability(out, model)
    local_evidence = pd.Series(
        logit(local_prob.to_numpy(dtype=float)),
        index=out.index,
        dtype=float,
    )

    out["stage2_graph_input_pred"] = candidate_mask.astype(int)
    out["stage2a_local_prob"] = local_prob
    out["stage2a_local_evidence"] = local_evidence

    # Compatibility fields for downstream experiment scripts. They contain
    # learned local evidence only; no legacy graph heuristic is reintroduced.
    out["stage2_selection_score"] = local_prob
    out["stage2_pred_edge"] = candidate_mask.astype(int)
    out["stage2_score"] = np.where(
        candidate_mask,
        local_prob.to_numpy(dtype=float),
        0.0,
    )

    return out


def stage2a_candidate_ranking_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mask = get_numeric(df, "stage2_graph_input_pred", 0.0).astype(int) == 1
    if int(mask.sum()) == 0:
        return {
            "candidate_edges": 0,
            "candidate_true_edges": 0,
            "auc": float("nan"),
            "ap": float("nan"),
        }

    y = get_numeric(df.loc[mask], "edge_label", 0.0).astype(int).to_numpy()
    s = get_numeric(
        df.loc[mask],
        "stage2a_local_prob",
        0.0,
    ).to_numpy(dtype=float)

    m = ranking_metrics(y, s)
    return {
        "candidate_edges": int(mask.sum()),
        "candidate_true_edges": int(y.sum()),
        "auc": m["auc"],
        "ap": m["ap"],
    }


# =============================================================================
# Stage 2B -- exact target-wise sparse graph recovery
# =============================================================================

def exact_targetwise_recovery(
    stage2a_df: pd.DataFrame,
    lambda_0: float,
    lambda_t: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Solve the Stage 2B objective exactly by target decomposition."""
    if lambda_0 < 0.0 or lambda_t < 0.0:
        raise ValueError("lambda_0 and lambda_t must be non-negative.")

    out = stage2a_df.copy().reset_index(drop=True)
    evidence = get_numeric(out, "stage2a_local_evidence", 0.0)

    candidate_mask = (
        get_numeric(out, "stage2_graph_input_pred", 0.0).astype(int) == 1
    )

    selected = pd.Series(0, index=out.index, dtype=int)
    target_rank = pd.Series(np.nan, index=out.index, dtype=float)
    incremental_gain = pd.Series(np.nan, index=out.index, dtype=float)
    target_objective = pd.Series(0.0, index=out.index, dtype=float)
    target_selected_degree = pd.Series(0, index=out.index, dtype=int)

    objective_total = 0.0
    target_count = 0

    candidates = out.loc[candidate_mask]

    for _, group in candidates.groupby(
        ["session_id", "Q2_row"],
        sort=False,
        dropna=False,
    ):
        target_count += 1

        # Stable deterministic ordering: descending evidence; original row
        # index breaks exact score ties.
        idxs = group.index.to_numpy(dtype=int)
        order = sorted(
            idxs.tolist(),
            key=lambda idx: (-float(evidence.loc[idx]), int(idx)),
        )

        r = np.asarray(
            [float(evidence.loc[idx]) for idx in order],
            dtype=float,
        )
        m = len(order)

        k_values = np.arange(m + 1, dtype=int)
        prefix = np.concatenate(
            [np.array([0.0]), np.cumsum(r)]
        )
        values = (
            prefix
            - float(lambda_0) * k_values
            - float(lambda_t) * k_values * (k_values - 1) / 2.0
        )

        # np.argmax returns the first maximum, i.e. the smaller k under ties.
        k_star = int(np.argmax(values))
        optimum = float(values[k_star])
        objective_total += optimum

        for rank, idx in enumerate(order, start=1):
            target_rank.loc[idx] = float(rank)
            incremental_gain.loc[idx] = float(
                evidence.loc[idx]
                - float(lambda_0)
                - float(lambda_t) * float(rank - 1)
            )
            target_objective.loc[idx] = optimum
            target_selected_degree.loc[idx] = k_star

        if k_star > 0:
            selected.loc[order[:k_star]] = 1

    final_score = pd.Series(0.0, index=out.index, dtype=float)
    candidate_indices = out.index[candidate_mask]
    if len(candidate_indices):
        final_score.loc[candidate_indices] = sigmoid(
            incremental_gain.loc[candidate_indices]
            .fillna(-30.0)
            .to_numpy(dtype=float)
        )

    out["stage2b_target_rank"] = target_rank
    out["stage2b_incremental_gain"] = incremental_gain
    out["stage2b_target_objective"] = target_objective
    out["stage2_graph_target_degree"] = target_selected_degree

    out["stage2_pruned_by_graph"] = (
        candidate_mask & (selected == 0)
    ).astype(int)

    out["final_graph_pred_edge"] = selected.astype(int)
    out["final_graph_score"] = final_score

    params = {
        "lambda_0": float(lambda_0),
        "lambda_t": float(lambda_t),
        "objective_total": float(objective_total),
        "target_groups": int(target_count),
        "pred_edges": int(selected.sum()),
    }
    return out, params


# =============================================================================
# Validation selection
# =============================================================================

def validation_grid_search(
    validation_stage2a: pd.DataFrame,
    lambda0_grid: Sequence[float],
    lambdat_grid: Sequence[float],
) -> Tuple[float, float, pd.DataFrame]:
    """Select Stage 2B penalties using Validation labels only."""
    rows: List[Dict[str, object]] = []

    y = get_numeric(
        validation_stage2a,
        "edge_label",
        0.0,
    ).astype(int).to_numpy()

    for lambda_0 in lambda0_grid:
        for lambda_t in lambdat_grid:
            recovered, params = exact_targetwise_recovery(
                validation_stage2a,
                lambda_0=float(lambda_0),
                lambda_t=float(lambda_t),
            )

            metrics = binary_metrics(
                y,
                recovered["final_graph_pred_edge"].to_numpy(dtype=int),
                recovered["final_graph_score"].to_numpy(dtype=float),
            )

            rows.append(
                {
                    "lambda_0": float(lambda_0),
                    "lambda_t": float(lambda_t),
                    "AUC": float(metrics.get("auc", float("nan"))),
                    "AP": float(metrics.get("ap", float("nan"))),
                    "Precision": float(metrics["precision"]),
                    "Recall": float(metrics["recall"]),
                    "F1": float(metrics["f1"]),
                    "PredEdges": int(metrics["pred_edges"]),
                    "TP": int(metrics["tp"]),
                    "FP": int(metrics["fp"]),
                    "FN": int(metrics["fn"]),
                    "ObjectiveTotal": float(params["objective_total"]),
                }
            )

    search = pd.DataFrame(rows)

    # Primary model-selection criterion: validation F1.
    # Deterministic ties favor precision, then a smaller graph, then smaller
    # penalties. Test data are not involved.
    ordered = search.sort_values(
        ["F1", "Precision", "PredEdges", "lambda_0", "lambda_t"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    best = ordered.iloc[0]
    return (
        float(best["lambda_0"]),
        float(best["lambda_t"]),
        search,
    )


# =============================================================================
# Output and audit
# =============================================================================

def extract_stage2a_coefficients(model) -> pd.DataFrame:
    """Export standardized Stage 2A logistic coefficients for audit."""
    lr = model.named_steps["logisticregression"]
    names = list(
        getattr(model, "stage2a_feature_names_", LOCAL_EVIDENCE_FEATURES)
    )
    coef = lr.coef_[0]
    return pd.DataFrame(
        {
            "feature": names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values(
        "abs_coefficient",
        ascending=False,
        kind="mergesort",
    )


def split_summary(
    split_name: str,
    final_df: pd.DataFrame,
) -> Dict[str, object]:
    y = get_numeric(final_df, "edge_label", 0.0).astype(int).to_numpy()
    metrics = binary_metrics(
        y,
        final_df["final_graph_pred_edge"].to_numpy(dtype=int),
        final_df["final_graph_score"].to_numpy(dtype=float),
    )

    row: Dict[str, object] = {"split": split_name}
    row.update(metrics)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final FCN Stage 2: three-way input, Train-only Stage 2A, "
            "Validation-selected penalties, and exact target-wise Stage 2B."
        )
    )
    parser.add_argument(
        "--stage1-outdir",
        default=str(DEFAULT_STAGE1_OUTDIR),
    )
    parser.add_argument(
        "--stage2-outdir",
        default=str(DEFAULT_STAGE2_OUTDIR),
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
        help=(
            "Comma-separated lambda_0 validation grid. Default: "
            + grid_to_arg(DEFAULT_LAMBDA0_GRID)
        ),
    )
    parser.add_argument(
        "--lambdat-grid",
        type=parse_float_grid,
        default=DEFAULT_LAMBDAT_GRID,
        help=(
            "Comma-separated lambda_t validation grid. Default: "
            + grid_to_arg(DEFAULT_LAMBDAT_GRID)
        ),
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    stage1_outdir = Path(args.stage1_outdir)
    outdir = Path(args.stage2_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_file = stage1_outdir / "stage1_train_scored.csv"
    validation_file = stage1_outdir / "stage1_validation_scored.csv"
    test_file = stage1_outdir / "stage1_test_scored.csv"

    missing = [
        str(path)
        for path in (train_file, validation_file, test_file)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing final Stage 1 three-way outputs:\n  "
            + "\n  ".join(missing)
        )

    train_df = pd.read_csv(train_file, low_memory=False)
    validation_df = pd.read_csv(validation_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)

    verify_three_way_inputs(
        train_df,
        validation_df,
        test_df,
    )

    print_block(
        "Stage 2 Input Protocol",
        {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_sessions": train_df["session_id"].nunique(),
            "validation_sessions": validation_df["session_id"].nunique(),
            "test_sessions": test_df["session_id"].nunique(),
            "train_stage1_candidates": int(
                get_numeric(train_df, "stage1_pred_edge").sum()
            ),
            "validation_stage1_candidates": int(
                get_numeric(validation_df, "stage1_pred_edge").sum()
            ),
            "test_stage1_candidates": int(
                get_numeric(test_df, "stage1_pred_edge").sum()
            ),
        },
    )

    # -----------------------------------------------------------------
    # Stage 2A: Train once, then freeze.
    # -----------------------------------------------------------------
    local_model = fit_stage2a_model(
        train_df,
        c_value=args.local_c,
    )

    stage2a_train = run_stage2a(train_df, local_model)
    stage2a_validation = run_stage2a(validation_df, local_model)

    # Test scoring is deliberately delayed until after validation selection.
    print_block(
        "Train - Stage 2A Retained-Candidate Ranking",
        stage2a_candidate_ranking_metrics(stage2a_train),
    )
    print_block(
        "Validation - Stage 2A Retained-Candidate Ranking",
        stage2a_candidate_ranking_metrics(stage2a_validation),
    )

    save_csv(
        stage2a_train,
        outdir / "stage2a_train_scored.csv",
    )
    save_csv(
        stage2a_validation,
        outdir / "stage2a_validation_scored.csv",
    )

    coef_df = extract_stage2a_coefficients(local_model)
    save_csv(
        coef_df,
        outdir / "stage2a_feature_coefficients.csv",
    )

    # -----------------------------------------------------------------
    # Stage 2B: select lambda_0 and lambda_t on Validation only.
    # -----------------------------------------------------------------
    selected_lambda0, selected_lambdat, validation_search = (
        validation_grid_search(
            stage2a_validation,
            lambda0_grid=args.lambda0_grid,
            lambdat_grid=args.lambdat_grid,
        )
    )

    save_csv(
        validation_search,
        outdir / "stage2_validation_search.csv",
    )

    selected_search_row = validation_search[
        np.isclose(
            validation_search["lambda_0"].astype(float),
            selected_lambda0,
            rtol=0.0,
            atol=1e-15,
        )
        & np.isclose(
            validation_search["lambda_t"].astype(float),
            selected_lambdat,
            rtol=0.0,
            atol=1e-15,
        )
    ].iloc[0]

    print_block(
        "Validation-selected Stage 2B Parameters",
        {
            "selected_lambda_0": selected_lambda0,
            "selected_lambda_t": selected_lambdat,
            "validation_precision": float(
                selected_search_row["Precision"]
            ),
            "validation_recall": float(
                selected_search_row["Recall"]
            ),
            "validation_f1": float(
                selected_search_row["F1"]
            ),
            "validation_pred_edges": int(
                selected_search_row["PredEdges"]
            ),
            "validation_tp": int(selected_search_row["TP"]),
            "validation_fp": int(selected_search_row["FP"]),
            "validation_fn": int(selected_search_row["FN"]),
        },
    )

    # Freeze parameters. No model refit and no parameter selection occurs
    # after this point.
    final_train, train_params = exact_targetwise_recovery(
        stage2a_train,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
    )
    final_validation, validation_params = exact_targetwise_recovery(
        stage2a_validation,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
    )

    # -----------------------------------------------------------------
    # Formal Test: score and recover exactly once after freezing.
    # -----------------------------------------------------------------
    stage2a_test = run_stage2a(test_df, local_model)
    save_csv(
        stage2a_test,
        outdir / "stage2a_test_scored.csv",
    )

    print_block(
        "Formal Test - Stage 2A Retained-Candidate Ranking",
        stage2a_candidate_ranking_metrics(stage2a_test),
    )

    final_test, test_params = exact_targetwise_recovery(
        stage2a_test,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
    )

    train_metrics = binary_metrics(
        get_numeric(final_train, "edge_label").astype(int),
        final_train["final_graph_pred_edge"],
        final_train["final_graph_score"],
    )
    validation_metrics = binary_metrics(
        get_numeric(final_validation, "edge_label").astype(int),
        final_validation["final_graph_pred_edge"],
        final_validation["final_graph_score"],
    )
    test_metrics = binary_metrics(
        get_numeric(final_test, "edge_label").astype(int),
        final_test["final_graph_pred_edge"],
        final_test["final_graph_score"],
    )

    print_block(
        "Train - Stage 2B Exact Target-wise Recovery",
        train_metrics,
    )
    print_block(
        "Validation - Stage 2B Exact Target-wise Recovery",
        validation_metrics,
    )
    print_block(
        "Formal Test - Stage 2B Exact Target-wise Recovery",
        test_metrics,
    )

    save_csv(
        final_train,
        outdir / "stage2_train_final_graph.csv",
    )
    save_csv(
        final_validation,
        outdir / "stage2_validation_final_graph.csv",
    )
    save_csv(
        final_test,
        outdir / "stage2_test_final_graph.csv",
    )

    summary = pd.DataFrame(
        [
            split_summary("Train", final_train),
            split_summary("Validation", final_validation),
            split_summary("Test", final_test),
        ]
    )
    save_csv(
        summary,
        outdir / "structured_stage2_summary.csv",
    )

    selected_params = {
        "protocol": "train_validation_test_no_refit",
        "stage2a": {
            "model": "StandardScaler + class-balanced LogisticRegression",
            "local_c": float(args.local_c),
            "fit_split": "Train",
            "refit_after_validation": False,
            "feature_count": len(LOCAL_EVIDENCE_FEATURES),
            "features": list(LOCAL_EVIDENCE_FEATURES),
        },
        "stage2b": {
            "solver": "exact_target_wise",
            "objective": (
                "sum(r_ij*z_ij) - lambda_0*sum(z_ij) "
                "- lambda_t*sum_j C(d_in(j),2)"
            ),
            "source_competition_penalty": 0.0,
            "selection_split": "Validation",
            "selection_metric": "F1",
            "selected_lambda_0": float(selected_lambda0),
            "selected_lambda_t": float(selected_lambdat),
            "lambda0_grid": [float(x) for x in args.lambda0_grid],
            "lambdat_grid": [float(x) for x in args.lambdat_grid],
        },
        "test_labels_used_for_fitting_or_selection": False,
        "validation_metrics": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in validation_metrics.items()
        },
        "test_metrics": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in test_metrics.items()
        },
        "objective_totals": {
            "train": float(train_params["objective_total"]),
            "validation": float(validation_params["objective_total"]),
            "test": float(test_params["objective_total"]),
        },
    }

    (outdir / "stage2_selected_params.json").write_text(
        json.dumps(
            selected_params,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 86)
    print("FCN Stage 2 completed successfully")
    print("=" * 86)
    print(f"output_dir: {outdir}")
    print(
        "Stage 2A model fitted once on Train; Stage 2B parameters selected "
        "on Validation; Formal Test evaluated after freezing."
    )
    print(
        "Test graph: stage2_test_final_graph.csv"
    )


if __name__ == "__main__":
    main()
