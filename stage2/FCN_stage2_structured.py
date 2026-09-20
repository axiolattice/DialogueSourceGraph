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
- If a selected penalty lies on the upper edge of its Validation grid, the
  run stops before Test is read so that the grid can be expanded.
- The frozen Stage 2A scorer and Stage 2B parameters are applied once to Test.
- Test labels are never used for fitting or hyperparameter selection.
- The exact solver is independently checked by enumerating every subset of
  every retained target group (at most 2^5 subsets per group).

Legacy SVMP rules, graph caps, budgets, hand-written structural bonuses,
message passing, and greedy graph selection are not used.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_STAGE1_OUTDIR = SCRIPT_DIR / "fcn_outputs_stage1"
DEFAULT_STAGE2_OUTDIR = SCRIPT_DIR / "fcn_outputs_stage2"

EXPECTED_TRAIN_SESSIONS = 176
EXPECTED_VALIDATION_SESSIONS = 44
EXPECTED_TEST_SESSIONS = 55
EXPECTED_TRAIN_ROWS = 39880
EXPECTED_VALIDATION_ROWS = 8379
EXPECTED_TEST_ROWS = 10434
EXPECTED_TRAIN_EDGES = 1744
EXPECTED_VALIDATION_EDGES = 299
EXPECTED_TEST_EDGES = 528
MAX_CANDIDATES_PER_TARGET = 5
SEED = 42

# Formal Stage 2A feature set after ablation: one-dimensional recalibration.
LOCAL_EVIDENCE_FEATURES: Tuple[str, ...] = (
    "stage1_prob",
)
if any("rationale" in feature.lower() for feature in LOCAL_EVIDENCE_FEATURES):
    raise RuntimeError("Formal Stage 2A forbids rationale-derived features.")

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_numeric(df: pd.DataFrame, col: str, split_name: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if bool(invalid.any()):
        raise RuntimeError(
            f"{split_name} column {col} contains {int(invalid.sum())} "
            "missing or non-finite values."
        )
    return values.astype(float)


def verify_stage1_partition(
    df: pd.DataFrame,
    split_name: str,
    expected_rows: int,
    expected_sessions: int,
    expected_true_edges: int,
) -> None:
    required = {
        "edge_id",
        "session_id",
        "Q1_row",
        "Q2_row",
        "edge_label",
        "stage1_prob",
        "stage1_pred_edge",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{split_name} Stage 1 file is missing columns: {missing}")
    if len(df) != expected_rows:
        raise RuntimeError(
            f"{split_name} row count mismatch: observed={len(df)}, "
            f"expected={expected_rows}."
        )
    session_count = int(df["session_id"].astype(str).nunique())
    if session_count != expected_sessions:
        raise RuntimeError(
            f"{split_name} session count mismatch: observed={session_count}, "
            f"expected={expected_sessions}."
        )
    if df["edge_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{split_name} contains duplicated edge_id values.")
    for col in ("session_id", "Q1_row", "Q2_row", "edge_id"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{split_name} contains an empty {col} value.")

    labels = require_finite_numeric(df, "edge_label", split_name)
    predictions = require_finite_numeric(df, "stage1_pred_edge", split_name)
    scores = require_finite_numeric(df, "stage1_prob", split_name)
    if not set(labels.unique()).issubset({0.0, 1.0}):
        raise RuntimeError(f"{split_name} edge_label must contain only 0/1.")
    if not set(predictions.unique()).issubset({0.0, 1.0}):
        raise RuntimeError(f"{split_name} stage1_pred_edge must contain only 0/1.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise RuntimeError(f"{split_name} stage1_prob must lie in [0,1].")
    if int(labels.sum()) != expected_true_edges:
        raise RuntimeError(
            f"{split_name} true-edge count mismatch: observed={int(labels.sum())}, "
            f"expected={expected_true_edges}."
        )

    retained = df.loc[predictions.astype(int).eq(1)]
    if not retained.empty:
        max_target_candidates = int(
            retained.groupby(["session_id", "Q2_row"], dropna=False).size().max()
        )
        if max_target_candidates > MAX_CANDIDATES_PER_TARGET:
            raise RuntimeError(
                f"{split_name} has {max_target_candidates} retained candidates "
                f"for one target; expected at most {MAX_CANDIDATES_PER_TARGET}."
            )


def verify_disjoint_sessions(named_frames: Sequence[Tuple[str, pd.DataFrame]]) -> None:
    session_sets = {
        name: set(frame["session_id"].astype(str))
        for name, frame in named_frames
    }
    names = list(session_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if session_sets[left] & session_sets[right]:
                raise RuntimeError(f"{left} and {right} sessions overlap.")


def load_and_verify_stage1_params(stage1_outdir: Path) -> Dict[str, object]:
    path = stage1_outdir / "stage1_final_selected_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 parameter manifest: {path}")
    params = json.loads(path.read_text(encoding="utf-8"))
    required_values = {
        "retention_orientation": "source_conditioned",
        "retention_group_keys": ["session_id", "Q1_row"],
        "encoder_train_only": True,
        "encoder_training_objective": "positive_pair_mnrl_only",
        "rationale_derived_supervision": False,
        "scorer_refit_after_validation": False,
        "test_labels_used_for_fitting_or_selection": False,
    }
    mismatches = {
        key: {"observed": params.get(key), "required": expected}
        for key, expected in required_values.items()
        if params.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Stage 1 protocol manifest mismatch: {mismatches}")
    expected_session_counts = {
        "train": EXPECTED_TRAIN_SESSIONS,
        "validation": EXPECTED_VALIDATION_SESSIONS,
        "test": EXPECTED_TEST_SESSIONS,
    }
    if params.get("session_counts") != expected_session_counts:
        raise RuntimeError(
            "Stage 1 session-count manifest mismatch: "
            f"observed={params.get('session_counts')}, "
            f"required={expected_session_counts}."
        )
    threshold = float(params.get("selected_threshold", float("nan")))
    top_k = int(params.get("selected_top_k", 0))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("Stage 1 selected_threshold is invalid.")
    if not 1 <= top_k <= MAX_CANDIDATES_PER_TARGET:
        raise RuntimeError("Stage 1 selected_top_k is outside [1,5].")
    return params


def verify_stage1_retention(
    df: pd.DataFrame,
    split_name: str,
    threshold: float,
    top_k: int,
) -> None:
    scores = require_finite_numeric(df, "stage1_prob", split_name)
    ranks = scores.groupby(
        [df["session_id"].astype(str), df["Q1_row"].astype(str)],
        sort=False,
    ).rank(method="first", ascending=False)
    expected = ((scores >= threshold) & (ranks <= top_k)).astype(int)
    observed = require_finite_numeric(
        df, "stage1_pred_edge", split_name
    ).astype(int)
    mismatches = int((expected.to_numpy() != observed.to_numpy()).sum())
    if mismatches:
        raise RuntimeError(
            f"{split_name} Stage 1 retention differs from its frozen "
            f"source-wise policy on {mismatches} rows."
        )


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


def exhaustive_targetwise_audit(
    stage2a_df: pd.DataFrame,
    recovered_df: pd.DataFrame,
    lambda_0: float,
    lambda_t: float,
    split_name: str,
    tolerance: float = 1e-10,
) -> Dict[str, object]:
    """Independently verify the solver against all subsets per target.

    This audit does not use edge labels. With W=5, each target has at most
    five retained candidates, so a complete 2^m enumeration is inexpensive.
    """
    base = stage2a_df.copy().reset_index(drop=True)
    recovered = recovered_df.copy().reset_index(drop=True)
    if len(base) != len(recovered):
        raise RuntimeError(
            f"{split_name} exactness audit received different row counts."
        )

    candidate_mask = (
        get_numeric(base, "stage2_graph_input_pred", 0.0).astype(int) == 1
    )
    solver_selected = get_numeric(
        recovered,
        "final_graph_pred_edge",
        0.0,
    ).astype(int)
    if int(solver_selected.loc[~candidate_mask].sum()) != 0:
        raise RuntimeError(
            f"{split_name} solver selected an edge excluded by Stage 1."
        )

    evidence = require_finite_numeric(
        base,
        "stage2a_local_evidence",
        split_name,
    )
    all_target_groups = 0
    candidate_target_groups = 0
    max_candidates = 0
    enumerated_subsets = 0
    objective_mismatches = 0
    subset_mismatches = 0
    max_objective_gap = 0.0

    for _, group in base.groupby(
        ["session_id", "Q2_row"],
        sort=False,
        dropna=False,
    ):
        all_target_groups += 1
        idxs = [
            int(idx)
            for idx in group.index
            if bool(candidate_mask.loc[idx])
        ]
        if not idxs:
            continue

        candidate_target_groups += 1
        order = sorted(
            idxs,
            key=lambda idx: (-float(evidence.loc[idx]), int(idx)),
        )
        m = len(order)
        max_candidates = max(max_candidates, m)
        if m > MAX_CANDIDATES_PER_TARGET:
            raise RuntimeError(
                f"{split_name} exactness audit found {m} candidates for one "
                f"target; maximum is {MAX_CANDIDATES_PER_TARGET}."
            )

        r = [float(evidence.loc[idx]) for idx in order]
        values: List[float] = []
        for mask in range(1 << m):
            k = int(bin(mask).count("1"))
            selected_evidence = sum(
                r[pos]
                for pos in range(m)
                if mask & (1 << pos)
            )
            values.append(
                float(
                    selected_evidence
                    - float(lambda_0) * k
                    - float(lambda_t) * k * (k - 1) / 2.0
                )
            )

        enumerated_subsets += len(values)
        best_value = max(values)
        solver_mask = 0
        for pos, idx in enumerate(order):
            if int(solver_selected.loc[idx]) == 1:
                solver_mask |= 1 << pos
        solver_value = values[solver_mask]
        objective_gap = max(0.0, float(best_value - solver_value))
        max_objective_gap = max(max_objective_gap, objective_gap)
        if objective_gap > tolerance:
            objective_mismatches += 1

        optimal_masks = {
            mask
            for mask, value in enumerate(values)
            if abs(float(value - best_value)) <= tolerance
        }
        if solver_mask not in optimal_masks:
            subset_mismatches += 1

    audit: Dict[str, object] = {
        "split": split_name,
        "all_target_groups": int(all_target_groups),
        "candidate_target_groups": int(candidate_target_groups),
        "max_candidates_per_target": int(max_candidates),
        "enumerated_subsets": int(enumerated_subsets),
        "objective_mismatches": int(objective_mismatches),
        "subset_mismatches": int(subset_mismatches),
        "max_objective_gap": float(max_objective_gap),
        "tolerance": float(tolerance),
        "passed": bool(
            objective_mismatches == 0 and subset_mismatches == 0
        ),
    }
    if not audit["passed"]:
        raise RuntimeError(f"{split_name} exactness audit failed: {audit}")
    return audit


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


def reject_upper_grid_boundary(
    parameter_name: str,
    selected_value: float,
    grid: Sequence[float],
) -> None:
    """Stop before Test if Validation selects an unexplored upper boundary."""
    values = sorted(set(float(value) for value in grid))
    if len(values) <= 1:
        return
    if np.isclose(
        float(selected_value),
        float(values[-1]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Validation selected {parameter_name}={selected_value:g}, the "
            f"upper boundary of [{values[0]:g}, {values[-1]:g}]. Expand only "
            "this Validation grid and rerun; Formal Test has not been read."
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
    run_status_file = outdir / "stage2_run_status.json"
    run_status = {
        "completed": False,
        "formal_test_loaded": False,
        "message": "Validation-only work in progress; do not report Test results.",
    }
    run_status_file.write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_file = stage1_outdir / "stage1_train_scored.csv"
    validation_file = stage1_outdir / "stage1_validation_scored.csv"
    test_file = stage1_outdir / "stage1_test_scored.csv"
    stage1_params_file = stage1_outdir / "stage1_final_selected_params.json"

    missing = [
        str(path)
        for path in (
            train_file,
            validation_file,
            test_file,
            stage1_params_file,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing final Stage 1 three-way outputs:\n  "
            + "\n  ".join(missing)
        )

    stage1_params = load_and_verify_stage1_params(stage1_outdir)
    stage1_threshold = float(stage1_params["selected_threshold"])
    stage1_top_k = int(stage1_params["selected_top_k"])

    # Physically read only Train and Validation before all model and penalty
    # choices are frozen. Merely checking that the Test path exists above does
    # not load its contents.
    train_df = pd.read_csv(train_file, low_memory=False)
    validation_df = pd.read_csv(validation_file, low_memory=False)
    verify_stage1_partition(
        train_df,
        "Train",
        EXPECTED_TRAIN_ROWS,
        EXPECTED_TRAIN_SESSIONS,
        EXPECTED_TRAIN_EDGES,
    )
    verify_stage1_partition(
        validation_df,
        "Validation",
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_VALIDATION_SESSIONS,
        EXPECTED_VALIDATION_EDGES,
    )
    verify_disjoint_sessions(
        [("Train", train_df), ("Validation", validation_df)]
    )
    verify_stage1_retention(
        train_df,
        "Train",
        stage1_threshold,
        stage1_top_k,
    )
    verify_stage1_retention(
        validation_df,
        "Validation",
        stage1_threshold,
        stage1_top_k,
    )

    print_block(
        "Stage 2 Train / Validation Input Protocol (Test Not Read)",
        {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "train_sessions": train_df["session_id"].nunique(),
            "validation_sessions": validation_df["session_id"].nunique(),
            "train_stage1_candidates": int(
                get_numeric(train_df, "stage1_pred_edge").sum()
            ),
            "validation_stage1_candidates": int(
                get_numeric(validation_df, "stage1_pred_edge").sum()
            ),
            "stage1_retention_orientation": "source_conditioned",
            "stage1_selected_threshold": f"{stage1_threshold:.16e}",
            "stage1_selected_top_k": stage1_top_k,
            "formal_test_status": "not_read",
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

    # A boundary optimum is not accepted as a frozen hyperparameter because
    # it leaves the explored Validation range demonstrably unresolved.
    reject_upper_grid_boundary(
        "lambda_0",
        selected_lambda0,
        args.lambda0_grid,
    )
    reject_upper_grid_boundary(
        "lambda_t",
        selected_lambdat,
        args.lambdat_grid,
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

    train_exactness = exhaustive_targetwise_audit(
        stage2a_train,
        final_train,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
        split_name="Train",
    )
    validation_exactness = exhaustive_targetwise_audit(
        stage2a_validation,
        final_validation,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
        split_name="Validation",
    )

    # -----------------------------------------------------------------
    # Formal Test: score and recover exactly once after freezing.
    # -----------------------------------------------------------------
    test_df = pd.read_csv(test_file, low_memory=False)
    run_status["formal_test_loaded"] = True
    run_status["message"] = (
        "Parameters frozen; Formal Test evaluation in progress."
    )
    run_status_file.write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verify_stage1_partition(
        test_df,
        "Test",
        EXPECTED_TEST_ROWS,
        EXPECTED_TEST_SESSIONS,
        EXPECTED_TEST_EDGES,
    )
    verify_disjoint_sessions(
        [
            ("Train", train_df),
            ("Validation", validation_df),
            ("Test", test_df),
        ]
    )
    verify_stage1_retention(
        test_df,
        "Test",
        stage1_threshold,
        stage1_top_k,
    )
    print_block(
        "Formal Test Input (Loaded After Parameter Freezing)",
        {
            "test_rows": len(test_df),
            "test_sessions": test_df["session_id"].nunique(),
            "test_stage1_candidates": int(
                get_numeric(test_df, "stage1_pred_edge").sum()
            ),
            "formal_test_status": "loaded_after_validation_selection",
        },
    )

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
    test_exactness = exhaustive_targetwise_audit(
        stage2a_test,
        final_test,
        lambda_0=selected_lambda0,
        lambda_t=selected_lambdat,
        split_name="Test",
    )
    exactness_rows = [
        train_exactness,
        validation_exactness,
        test_exactness,
    ]
    save_csv(
        pd.DataFrame(exactness_rows),
        outdir / "stage2_exactness_audit.csv",
    )
    print_block(
        "Stage 2B Exhaustive Exactness Audit",
        {
            "audited_splits": "Train, Validation, Test",
            "candidate_target_groups": int(
                sum(row["candidate_target_groups"] for row in exactness_rows)
            ),
            "enumerated_subsets": int(
                sum(row["enumerated_subsets"] for row in exactness_rows)
            ),
            "max_candidates_per_target": int(
                max(row["max_candidates_per_target"] for row in exactness_rows)
            ),
            "objective_mismatches": int(
                sum(row["objective_mismatches"] for row in exactness_rows)
            ),
            "subset_mismatches": int(
                sum(row["subset_mismatches"] for row in exactness_rows)
            ),
            "passed": all(bool(row["passed"]) for row in exactness_rows),
        },
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
        "protocol": "train_validation_test_no_refit_test_read_after_freeze",
        "rationale_derived_supervision": False,
        "stage1_input": {
            "retention_orientation": stage1_params["retention_orientation"],
            "retention_group_keys": stage1_params["retention_group_keys"],
            "selected_threshold": stage1_threshold,
            "selected_top_k": stage1_top_k,
            "parameter_manifest_sha256": sha256_file(stage1_params_file),
            "train_scored_sha256": sha256_file(train_file),
            "validation_scored_sha256": sha256_file(validation_file),
            "test_scored_sha256": sha256_file(test_file),
        },
        "split_protocol": {
            "train_sessions": EXPECTED_TRAIN_SESSIONS,
            "validation_sessions": EXPECTED_VALIDATION_SESSIONS,
            "test_sessions": EXPECTED_TEST_SESSIONS,
            "sessions_disjoint": True,
            "test_loaded_after_validation_selection": True,
        },
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
            "upper_grid_boundary_rejected": True,
        },
        "exactness_audit": {
            "method": "all_subsets_per_target",
            "max_subsets_per_target": int(2 ** MAX_CANDIDATES_PER_TARGET),
            "splits": {
                row["split"]: row
                for row in exactness_rows
            },
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

    run_status.update(
        {
            "completed": True,
            "formal_test_loaded": True,
            "message": "FCN Stage 2 completed successfully.",
            "selected_lambda_0": float(selected_lambda0),
            "selected_lambda_t": float(selected_lambdat),
            "exactness_audit_passed": True,
        }
    )
    run_status_file.write_text(
        json.dumps(run_status, ensure_ascii=False, indent=2),
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
    print("Exact target-wise solver independently verified by all-subset audit.")
    print(
        "Test graph: stage2_test_final_graph.csv"
    )


if __name__ == "__main__":
    main()
