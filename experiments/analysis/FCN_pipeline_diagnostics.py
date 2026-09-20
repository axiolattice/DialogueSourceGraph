"""Summarize the frozen FCN pipeline as a stage-by-stage evidence chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.common import bundle_paths, load_formal_bundle


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_score, dtype=float)))


def safe_ap(y_true: pd.Series, y_score: pd.Series) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(y_score, dtype=float)))


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def summarize_stage1(split_name: str, df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["edge_label"].astype(int)
    y_pred = df["stage1_pred_edge"].astype(int)
    metrics = binary_metrics(y_true, y_pred)
    total = len(df)
    retained = int(y_pred.sum())
    return {
        "split": split_name,
        "stage": "stage1_candidate_retention",
        "total_edges": total,
        "true_edges": int(y_true.sum()),
        "pred_edges": retained,
        "candidate_retention_ratio": float(retained / total) if total else float("nan"),
        "candidate_compression_ratio": float(1.0 - retained / total) if total else float("nan"),
        "stage1_gold_edge_recall": metrics["recall"],
        "stage1_precision": metrics["precision"],
        "stage1_recall": metrics["recall"],
        "stage1_f1": metrics["f1"],
        "stage1_tp": metrics["tp"],
        "stage1_fp": metrics["fp"],
        "stage1_fn": metrics["fn"],
        "stage1_tn": metrics["tn"],
        "stage1_auc": safe_auc(y_true, df["stage1_prob"]),
        "stage1_ap": safe_ap(y_true, df["stage1_prob"]),
        "stage2a_auc": np.nan,
        "stage2a_ap": np.nan,
        "stage2b_input_pred_edges": np.nan,
        "stage2b_final_pred_edges": np.nan,
        "stage2b_deleted_edges": np.nan,
        "stage2b_deleted_tp": np.nan,
        "stage2b_deleted_fp": np.nan,
        "stage2b_deleted_fn": np.nan,
        "stage2b_precision": np.nan,
        "stage2b_recall": np.nan,
        "stage2b_f1": np.nan,
        "stage2b_auc": np.nan,
        "stage2b_ap": np.nan,
        "notes": "Stage 1 source-conditioned candidate retention.",
    }


def summarize_stage2a(split_name: str, df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["edge_label"].astype(int)
    y_score = df["stage2a_local_prob"].astype(float)
    return {
        "split": split_name,
        "stage": "stage2a_local_calibration",
        "total_edges": len(df),
        "true_edges": int(y_true.sum()),
        "pred_edges": np.nan,
        "candidate_retention_ratio": np.nan,
        "candidate_compression_ratio": np.nan,
        "stage1_gold_edge_recall": np.nan,
        "stage1_precision": np.nan,
        "stage1_recall": np.nan,
        "stage1_f1": np.nan,
        "stage1_tp": np.nan,
        "stage1_fp": np.nan,
        "stage1_fn": np.nan,
        "stage1_tn": np.nan,
        "stage1_auc": np.nan,
        "stage1_ap": np.nan,
        "stage2a_auc": safe_auc(y_true, y_score),
        "stage2a_ap": safe_ap(y_true, y_score),
        "stage2b_input_pred_edges": np.nan,
        "stage2b_final_pred_edges": np.nan,
        "stage2b_deleted_edges": np.nan,
        "stage2b_deleted_tp": np.nan,
        "stage2b_deleted_fp": np.nan,
        "stage2b_deleted_fn": np.nan,
        "stage2b_precision": np.nan,
        "stage2b_recall": np.nan,
        "stage2b_f1": np.nan,
        "stage2b_auc": np.nan,
        "stage2b_ap": np.nan,
        "notes": "Stage 2A candidate-set ranking and calibration.",
    }


def summarize_stage2b(split_name: str, df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["edge_label"].astype(int)
    final_pred = df["final_graph_pred_edge"].astype(int)
    input_pred = (
        df["stage2_graph_input_pred"].astype(int)
        if "stage2_graph_input_pred" in df.columns
        else df["stage2_pred_edge"].astype(int)
    )
    metrics = binary_metrics(y_true, final_pred)
    deleted = (input_pred == 1) & (final_pred == 0)
    deleted_edges = int(deleted.sum())
    deleted_tp = int(((y_true == 1) & deleted).sum())
    deleted_fp = int(((y_true == 0) & deleted).sum())
    deleted_fn = int(((y_true == 1) & (input_pred == 0)).sum())
    return {
        "split": split_name,
        "stage": "stage2b_structured_recovery",
        "total_edges": len(df),
        "true_edges": int(y_true.sum()),
        "pred_edges": np.nan,
        "candidate_retention_ratio": np.nan,
        "candidate_compression_ratio": np.nan,
        "stage1_gold_edge_recall": np.nan,
        "stage1_precision": np.nan,
        "stage1_recall": np.nan,
        "stage1_f1": np.nan,
        "stage1_tp": np.nan,
        "stage1_fp": np.nan,
        "stage1_fn": np.nan,
        "stage1_tn": np.nan,
        "stage1_auc": np.nan,
        "stage1_ap": np.nan,
        "stage2a_auc": np.nan,
        "stage2a_ap": np.nan,
        "stage2b_input_pred_edges": int(input_pred.sum()),
        "stage2b_final_pred_edges": int(final_pred.sum()),
        "stage2b_deleted_edges": deleted_edges,
        "stage2b_deleted_tp": deleted_tp,
        "stage2b_deleted_fp": deleted_fp,
        "stage2b_deleted_fn": deleted_fn,
        "stage2b_precision": metrics["precision"],
        "stage2b_recall": metrics["recall"],
        "stage2b_f1": metrics["f1"],
        "stage2b_auc": safe_auc(y_true, df["final_graph_score"]),
        "stage2b_ap": safe_ap(y_true, df["final_graph_score"]),
        "notes": "Stage 2B exact target-wise graph recovery.",
    }


def build_transition_summary(rows: List[Dict[str, float]], stage2_params: Dict[str, object]) -> pd.DataFrame:
    stage2b = stage2_params.get("stage2b", stage2_params)
    selected_lambda_0 = stage2b.get("selected_lambda_0", np.nan)
    selected_lambda_t = stage2b.get("selected_lambda_t", np.nan)
    summary_rows = []
    for split in ["Train", "Validation", "Test"]:
        stage1 = next(r for r in rows if r["split"] == split and r["stage"] == "stage1_candidate_retention")
        stage2a = next(r for r in rows if r["split"] == split and r["stage"] == "stage2a_local_calibration")
        stage2b = next(r for r in rows if r["split"] == split and r["stage"] == "stage2b_structured_recovery")
        summary_rows.append(
            {
                "split": split,
                "raw_candidates": stage1["total_edges"],
                "raw_true_edges": stage1["true_edges"],
                "stage1_retained_edges": stage1["pred_edges"],
                "stage1_gold_edge_recall": stage1["stage1_gold_edge_recall"],
                "stage1_compression_ratio": stage1["candidate_compression_ratio"],
                "stage2a_auc": stage2a["stage2a_auc"],
                "stage2a_ap": stage2a["stage2a_ap"],
                "stage2b_input_pred_edges": stage2b["stage2b_input_pred_edges"],
                "stage2b_final_pred_edges": stage2b["stage2b_final_pred_edges"],
                "stage2b_deleted_edges": stage2b["stage2b_deleted_edges"],
                "stage2b_deleted_tp": stage2b["stage2b_deleted_tp"],
                "stage2b_deleted_fp": stage2b["stage2b_deleted_fp"],
                "stage2b_precision": stage2b["stage2b_precision"],
                "stage2b_recall": stage2b["stage2b_recall"],
                "stage2b_f1": stage2b["stage2b_f1"],
                "stage2b_auc": stage2b["stage2b_auc"],
                "stage2b_ap": stage2b["stage2b_ap"],
                "selected_lambda_0": selected_lambda_0,
                "selected_lambda_t": selected_lambda_t,
            }
        )
    return pd.DataFrame(summary_rows)


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
        help="Directory for diagnostics outputs.",
    )
    args = parser.parse_args()

    bundle = load_formal_bundle(args.bundle_root)
    paths = bundle["paths"]
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else bundle_paths(args.bundle_root).root.parent
        / "experiments"
        / "analysis"
        / "fcn_pipeline_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for split_name, df in [
        ("Train", bundle["stage1"]["train"]),
        ("Validation", bundle["stage1"]["validation"]),
        ("Test", bundle["stage1"]["test"]),
    ]:
        rows.append(summarize_stage1(split_name, df))

    for split_name, df in [
        ("Train", bundle["stage2"]["stage2a_train"]),
        ("Validation", bundle["stage2"]["stage2a_validation"]),
        ("Test", bundle["stage2"]["stage2a_test"]),
    ]:
        rows.append(summarize_stage2a(split_name, df))

    for split_name, df in [
        ("Train", bundle["stage2"]["train_graph"]),
        ("Validation", bundle["stage2"]["validation_graph"]),
        ("Test", bundle["stage2"]["test_graph"]),
    ]:
        rows.append(summarize_stage2b(split_name, df))

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_dir / "pipeline_diagnostics.csv", index=False)

    summary = build_transition_summary(rows, bundle["stage2"]["selected_params"])
    summary.to_csv(output_dir / "stage_transition_summary.csv", index=False)

    print("Formal bundle root:", paths.root)
    print("Diagnostics written to:", output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
