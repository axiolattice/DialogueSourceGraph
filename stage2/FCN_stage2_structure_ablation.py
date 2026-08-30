#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FCN Stage 2 structural ablation: M0 / M1 / M2.

This script keeps the final formal Stage 2A fixed to the frozen one-dimensional
recalibration learned from Stage 1. It then compares three Stage 2B objectives:

    M0: sum(r_ij z_ij)
    M1: sum(r_ij z_ij) - lambda_0 sum(z_ij)
    M2: sum(r_ij z_ij) - lambda_0 sum(z_ij) - lambda_t sum_j C(d_in(j), 2)

Protocol
--------
- Stage 2A is frozen and identical across all conditions.
- M0 has no tunable parameters.
- M1 selects lambda_0 on Validation only.
- M2 selects (lambda_0, lambda_t) on Validation only.
- Test is reported once after freezing and never used for selection.

The aim is to isolate the incremental value of target competition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_STAGE1_OUTDIR = PROJECT_DIR / "stage1" / "fcn_outputs_stage1_final"
DEFAULT_STAGE2_OUTDIR = SCRIPT_DIR / "fcn_outputs_stage2_structure_ablation"
DEFAULT_LAMBDA0_GRID = tuple(np.round(np.arange(0.00, 0.801, 0.05), 10))
DEFAULT_LAMBDAT_GRID = tuple(np.round(np.arange(0.00, 0.501, 0.05), 10))
EXPECTED_TRAIN_SESSIONS = 176
EXPECTED_VALIDATION_SESSIONS = 44
EXPECTED_TEST_SESSIONS = 55
SEED = 42


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
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

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
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }
    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=float)
        out["auc"] = safe_auc(y_true_arr, score_arr)
        out["ap"] = safe_ap(y_true_arr, score_arr)
    return out


def print_block(title: str, values: Dict[str, object]) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    for key, value in values.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {'nan' if np.isnan(float(value)) else f'{float(value):.6f}'}")
        else:
            print(f"{key}: {value}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_stage1_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 1 output file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def verify_three_way_inputs(train_df, validation_df, test_df) -> None:
    required = {"session_id", "Q1_row", "Q2_row", "edge_label", "stage1_pred_edge"}
    for name, df in (("Train", train_df), ("Validation", validation_df), ("Test", test_df)):
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
        raise RuntimeError(f"Stage 1 inputs do not match frozen 176/44/55 split: observed={counts}, expected={expected}.")
    train_ids = set(train_df["session_id"].astype(str))
    validation_ids = set(validation_df["session_id"].astype(str))
    test_ids = set(test_df["session_id"].astype(str))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Train/Validation/Test session sets overlap.")


def fit_stage2a_model(train_df: pd.DataFrame, c_value: float = 1.0):
    candidate_mask = get_numeric(train_df, "stage1_pred_edge", 0.0).astype(int) == 1
    candidate_df = train_df.loc[candidate_mask].copy()
    if candidate_df.empty:
        raise RuntimeError("No retained Stage 1 candidates in Train.")
    y = get_numeric(candidate_df, "edge_label", 0.0).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError("Retained Train candidates contain only one class.")
    X = pd.DataFrame({"stage1_prob": get_numeric(candidate_df, "stage1_prob", 0.0)}, index=candidate_df.index)
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
    model.stage2a_feature_names_ = np.asarray(["stage1_prob"], dtype=object)
    return model


def run_stage2a(df: pd.DataFrame, model) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    candidate_mask = get_numeric(out, "stage1_pred_edge", 0.0).astype(int) == 1
    X = pd.DataFrame({"stage1_prob": get_numeric(out, "stage1_prob", 0.0)}, index=out.index)
    local_prob = model.predict_proba(X)[:, 1]
    out["stage2_graph_input_pred"] = candidate_mask.astype(int)
    out["stage2a_local_prob"] = np.clip(local_prob, 1e-6, 1.0 - 1e-6)
    out["stage2a_local_evidence"] = logit(out["stage2a_local_prob"].to_numpy(dtype=float))
    out["stage2_local_evidence"] = out["stage2a_local_evidence"]
    out["stage2_selection_score"] = out["stage2a_local_prob"]
    out["stage2_pred_edge"] = candidate_mask.astype(int)
    out["stage2_score"] = np.where(candidate_mask, out["stage2a_local_prob"].to_numpy(dtype=float), 0.0)
    return out


def exact_m0_recovery(stage2a_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = stage2a_df.copy().reset_index(drop=True)
    evidence = get_numeric(out, "stage2a_local_evidence", 0.0).to_numpy(dtype=float)
    candidate_mask = get_numeric(out, "stage2_graph_input_pred", 0.0).astype(int) == 1
    selected = np.zeros(len(out), dtype=int)
    selected[candidate_mask.to_numpy() & (evidence > 0.0)] = 1
    out["final_graph_pred_edge"] = selected
    out["final_graph_score"] = np.where(candidate_mask, sigmoid(evidence), 0.0)
    params = {
        "objective_total": float(np.maximum(evidence[candidate_mask.to_numpy()], 0.0).sum()),
        "target_groups": int(out.loc[candidate_mask, ["session_id", "Q2_row"]].drop_duplicates().shape[0]),
        "pred_edges": int(selected.sum()),
    }
    return out, params


def exact_m1_recovery(stage2a_df: pd.DataFrame, lambda_0: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = stage2a_df.copy().reset_index(drop=True)
    evidence = get_numeric(out, "stage2a_local_evidence", 0.0).to_numpy(dtype=float)
    candidate_mask = get_numeric(out, "stage2_graph_input_pred", 0.0).astype(int) == 1
    threshold = float(lambda_0)
    selected = np.zeros(len(out), dtype=int)
    selected[candidate_mask.to_numpy() & (evidence > threshold)] = 1
    out["final_graph_pred_edge"] = selected
    out["final_graph_score"] = np.where(candidate_mask, sigmoid(evidence - threshold), 0.0)
    params = {
        "lambda_0": float(lambda_0),
        "objective_total": float(np.maximum(evidence[candidate_mask.to_numpy()] - threshold, 0.0).sum()),
        "target_groups": int(out.loc[candidate_mask, ["session_id", "Q2_row"]].drop_duplicates().shape[0]),
        "pred_edges": int(selected.sum()),
    }
    return out, params


def exact_m2_recovery(stage2a_df: pd.DataFrame, lambda_0: float, lambda_t: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = stage2a_df.copy().reset_index(drop=True)
    evidence = get_numeric(out, "stage2a_local_evidence", 0.0)
    candidate_mask = get_numeric(out, "stage2_graph_input_pred", 0.0).astype(int) == 1
    selected = pd.Series(0, index=out.index, dtype=int)
    target_degree = pd.Series(0, index=out.index, dtype=int)
    incremental_gain = pd.Series(np.nan, index=out.index, dtype=float)
    objective_total = 0.0
    target_count = 0
    for _, group in out.loc[candidate_mask].groupby(["session_id", "Q2_row"], sort=False, dropna=False):
        target_count += 1
        idxs = group.index.to_numpy(dtype=int)
        order = sorted(idxs.tolist(), key=lambda idx: (-float(evidence.loc[idx]), int(idx)))
        r = np.asarray([float(evidence.loc[idx]) for idx in order], dtype=float)
        k_values = np.arange(len(order) + 1, dtype=int)
        prefix = np.concatenate([np.array([0.0]), np.cumsum(r)])
        values = prefix - float(lambda_0) * k_values - float(lambda_t) * k_values * (k_values - 1) / 2.0
        k_star = int(np.argmax(values))
        objective_total += float(values[k_star])
        for rank, idx in enumerate(order, start=1):
            target_degree.loc[idx] = k_star
            incremental_gain.loc[idx] = float(evidence.loc[idx] - float(lambda_0) - float(lambda_t) * float(rank - 1))
        if k_star > 0:
            selected.loc[order[:k_star]] = 1
    out["stage2_graph_target_degree"] = target_degree
    out["stage2_pruned_by_graph"] = (candidate_mask & (selected == 0)).astype(int)
    out["final_graph_pred_edge"] = selected.astype(int)
    final_score = pd.Series(0.0, index=out.index, dtype=float)
    if candidate_mask.any():
        final_score.loc[out.index[candidate_mask]] = sigmoid(
            incremental_gain.loc[out.index[candidate_mask]].fillna(-30.0).to_numpy(dtype=float)
        )
    out["final_graph_score"] = final_score
    params = {
        "lambda_0": float(lambda_0),
        "lambda_t": float(lambda_t),
        "objective_total": float(objective_total),
        "target_groups": int(target_count),
        "pred_edges": int(selected.sum()),
    }
    return out, params


def validation_search_m1(validation_stage2a: pd.DataFrame, lambda0_grid: Sequence[float]):
    rows = []
    y = get_numeric(validation_stage2a, "edge_label", 0.0).astype(int).to_numpy()
    for lambda_0 in lambda0_grid:
        recovered, params = exact_m1_recovery(validation_stage2a, lambda_0=float(lambda_0))
        metrics = binary_metrics(y, recovered["final_graph_pred_edge"].to_numpy(dtype=int), recovered["final_graph_score"].to_numpy(dtype=float))
        rows.append(
            {
                "lambda_0": float(lambda_0),
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
    ordered = search.sort_values(["F1", "Precision", "PredEdges", "lambda_0"], ascending=[False, False, True, True], kind="mergesort").reset_index(drop=True)
    best = ordered.iloc[0]
    return float(best["lambda_0"]), search


def validation_search_m2(validation_stage2a: pd.DataFrame, lambda0_grid: Sequence[float], lambdat_grid: Sequence[float]):
    rows = []
    y = get_numeric(validation_stage2a, "edge_label", 0.0).astype(int).to_numpy()
    for lambda_0 in lambda0_grid:
        for lambda_t in lambdat_grid:
            recovered, params = exact_m2_recovery(validation_stage2a, lambda_0=float(lambda_0), lambda_t=float(lambda_t))
            metrics = binary_metrics(y, recovered["final_graph_pred_edge"].to_numpy(dtype=int), recovered["final_graph_score"].to_numpy(dtype=float))
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
    ordered = search.sort_values(["F1", "Precision", "PredEdges", "lambda_0", "lambda_t"], ascending=[False, False, True, True, True], kind="mergesort").reset_index(drop=True)
    best = ordered.iloc[0]
    return float(best["lambda_0"]), float(best["lambda_t"]), search


def parse_args():
    p = argparse.ArgumentParser(description="FCN Stage 2 structural ablation M0/M1/M2.")
    p.add_argument("--stage1-outdir", default=str(DEFAULT_STAGE1_OUTDIR))
    p.add_argument("--outdir", default=str(DEFAULT_STAGE2_OUTDIR))
    p.add_argument("--local-c", type=float, default=1.0)
    p.add_argument("--lambda0-grid", type=parse_float_grid, default=DEFAULT_LAMBDA0_GRID)
    p.add_argument("--lambdat-grid", type=parse_float_grid, default=DEFAULT_LAMBDAT_GRID)
    return p.parse_args()


def stage2a_candidate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mask = get_numeric(df, "stage2_graph_input_pred", 0.0).astype(int) == 1
    if int(mask.sum()) == 0:
        return {
            "candidate_edges": 0,
            "candidate_true_edges": 0,
            "auc": float("nan"),
            "ap": float("nan"),
        }
    y = get_numeric(df.loc[mask], "edge_label", 0.0).astype(int).to_numpy()
    s = get_numeric(df.loc[mask], "stage2a_local_prob", 0.0).to_numpy(dtype=float)
    return {
        "candidate_edges": int(mask.sum()),
        "candidate_true_edges": int(y.sum()),
        "auc": safe_auc(y, s),
        "ap": safe_ap(y, s),
    }


def run_condition(
    name: str,
    train_stage2a: pd.DataFrame,
    val_stage2a: pd.DataFrame,
    test_stage2a: pd.DataFrame,
    outdir: Path,
    lambda0_grid,
    lambdat_grid,
) -> Dict[str, object]:
    print("\n" + "=" * 100)
    print(f"Stage 2 structural ablation: {name}")
    print("=" * 100)
    print_block("Train - Stage 2A Retained-Candidate Ranking", stage2a_candidate_metrics(train_stage2a))
    print_block("Validation - Stage 2A Retained-Candidate Ranking", stage2a_candidate_metrics(val_stage2a))
    condition_dir = outdir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    save_csv(train_stage2a, condition_dir / "stage2a_train_scored.csv")
    save_csv(val_stage2a, condition_dir / "stage2a_validation_scored.csv")
    save_csv(test_stage2a, condition_dir / "stage2a_test_scored.csv")
    if name == "M0":
        final_train, train_params = exact_m0_recovery(train_stage2a)
        final_val, val_params = exact_m0_recovery(val_stage2a)
        final_test, test_params = exact_m0_recovery(test_stage2a)
        search = pd.DataFrame([{"objective": "M0", "note": "no hyperparameters"}])
        selected = {}
    elif name == "M1":
        lambda_0, search = validation_search_m1(val_stage2a, lambda0_grid)
        final_train, train_params = exact_m1_recovery(train_stage2a, lambda_0)
        final_val, val_params = exact_m1_recovery(val_stage2a, lambda_0)
        final_test, test_params = exact_m1_recovery(test_stage2a, lambda_0)
        save_csv(search, condition_dir / "stage2_validation_search.csv")
        selected = {"lambda_0": float(lambda_0)}
    else:
        lambda_0, lambda_t, search = validation_search_m2(val_stage2a, lambda0_grid, lambdat_grid)
        final_train, train_params = exact_m2_recovery(train_stage2a, lambda_0, lambda_t)
        final_val, val_params = exact_m2_recovery(val_stage2a, lambda_0, lambda_t)
        final_test, test_params = exact_m2_recovery(test_stage2a, lambda_0, lambda_t)
        save_csv(search, condition_dir / "stage2_validation_search.csv")
        selected = {"lambda_0": float(lambda_0), "lambda_t": float(lambda_t)}
    save_csv(final_train, condition_dir / "stage2_train_final_graph.csv")
    save_csv(final_val, condition_dir / "stage2_validation_final_graph.csv")
    save_csv(final_test, condition_dir / "stage2_test_final_graph.csv")
    rows = []
    for split_name, frame in (("Train", final_train), ("Validation", final_val), ("Test", final_test)):
        y_true = get_numeric(frame, "edge_label").astype(int).to_numpy()
        rows.append({"split": split_name, **binary_metrics(y_true, frame["final_graph_pred_edge"], frame["final_graph_score"])})
    save_csv(pd.DataFrame(rows), condition_dir / "structured_stage2_summary.csv")
    with open(condition_dir / "stage2_selected_params.json", "w", encoding="utf-8") as f:
        json.dump({"condition": name, **selected, "train_params": train_params, "validation_params": val_params, "test_params": test_params}, f, ensure_ascii=False, indent=2)
    return {
        "condition": name,
        "Validation_Precision": float(rows[1]["precision"]),
        "Validation_Recall": float(rows[1]["recall"]),
        "Validation_F1": float(rows[1]["f1"]),
        "Validation_PredEdges": int(rows[1]["pred_edges"]),
        "Test_Precision": float(rows[2]["precision"]),
        "Test_Recall": float(rows[2]["recall"]),
        "Test_F1": float(rows[2]["f1"]),
        "Test_AP": float(rows[2].get("ap", float("nan"))),
        "Test_AUC": float(rows[2].get("auc", float("nan"))),
        "Test_PredEdges": int(rows[2]["pred_edges"]),
        "Test_TP": int(rows[2]["tp"]),
        "Test_FP": int(rows[2]["fp"]),
        "Test_FN": int(rows[2]["fn"]),
        **selected,
    }


def main() -> None:
    args = parse_args()
    stage1_outdir = Path(args.stage1_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df = load_stage1_split(stage1_outdir / "stage1_train_scored.csv")
    val_df = load_stage1_split(stage1_outdir / "stage1_validation_scored.csv")
    test_df = load_stage1_split(stage1_outdir / "stage1_test_scored.csv")
    verify_three_way_inputs(train_df, val_df, test_df)
    model = fit_stage2a_model(train_df, c_value=args.local_c)
    train_stage2a = run_stage2a(train_df, model)
    val_stage2a = run_stage2a(val_df, model)
    test_stage2a = run_stage2a(test_df, model)
    results = []
    for name in ("M0", "M1", "M2"):
        results.append(run_condition(name, train_stage2a, val_stage2a, test_stage2a, outdir, args.lambda0_grid, args.lambdat_grid))
    summary = pd.DataFrame(results)
    save_csv(summary, outdir / "stage2_structure_ablation_summary.csv")
    print("\n" + "=" * 100)
    print("Stage 2 structural ablation summary")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
