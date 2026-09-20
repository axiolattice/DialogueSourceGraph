#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error analysis for the frozen FCN Test graph.

This script reads the frozen formal Test graph and summarizes false positives
and false negatives in a quantitative, post-hoc way. It does not retrain any
model or alter the formal outputs.

The analysis includes:
- overall Test metrics for the final FCN graph
- consistency check between Stage 2 outputs and main-experiment FCN outputs
- false-positive / false-negative counts by distance, topic-shift bucket,
  and target indegree
- heuristic taxonomy for the main error classes
- top cases for manual inspection
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.common import bundle_paths, load_formal_bundle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_error_analysis"


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


def binary_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: Optional[pd.Series] = None) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_score is not None:
        out["auc"] = safe_auc(y_true, y_score)
        out["ap"] = safe_ap(y_true, y_score)
    return out


def char_jaccard(left: object, right: object) -> float:
    def normalize(text: object) -> set[str]:
        s = "" if pd.isna(text) else str(text).lower()
        return {ch for ch in s if ch.strip()}

    a = normalize(left)
    b = normalize(right)
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def distance_bucket(distance: float) -> str:
    d = int(distance)
    if d <= 1:
        return "d=1"
    if d <= 3:
        return "d=2-3"
    if d <= 5:
        return "d=4-5"
    return "d>=6"


def bucket_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bins = [-np.inf, 0.0, 0.1, 0.25, 0.5, 0.75, np.inf]
    labels = ["<=0", "0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", ">=0.75"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def generic_answer_trigger(text: object) -> bool:
    s = "" if pd.isna(text) else str(text)
    patterns = [
        "谢谢",
        "按照规定",
        "按规定",
        "正在研究",
        "暂无计划",
        "我们会",
        "努力工作",
        "持续",
        "请关注",
        "理性判断",
        "无法作出评价",
        "不便回答",
    ]
    return any(pat in s for pat in patterns)


def question_form_trigger(row: pd.Series) -> bool:
    flags = str(row.get("topic_shift_flags", "") or "")
    bucket = str(row.get("topic_shift_bucket", "") or "")
    q2_len = float(row.get("q2_len", np.nan))
    return (
        "question_form" in flags
        or "short_q2" in flags
        or bucket == "boundary"
        or (np.isfinite(q2_len) and q2_len <= 20)
    )


def jaccard_trigger(row: pd.Series) -> bool:
    q1_q2 = char_jaccard(row.get("Q1"), row.get("Q2"))
    a1_q2 = char_jaccard(row.get("A1"), row.get("Q2"))
    return q1_q2 >= 0.30 or a1_q2 >= 0.30


def classify_fp(row: pd.Series, target_pred_degree: int) -> str:
    distance = float(row.get("distance_num", row.get("distance", np.nan)))
    if np.isfinite(distance) and distance >= 4:
        return "long_distance_carryover"
    if target_pred_degree >= 2:
        return "multiple_source_over_recovery"
    if generic_answer_trigger(row.get("A1")):
        return "vague_answer_false_trigger"
    if question_form_trigger(row):
        return "boundary_question_form"
    if jaccard_trigger(row):
        return "broad_lexical_overlap"
    bucket = str(row.get("topic_shift_bucket", "") or "")
    if bucket not in {"", "keep", "nan"}:
        return "topic_boundary_extension"
    return "other_manual_review"


def classify_fn(row: pd.Series) -> str:
    stage1_pred = int(pd.to_numeric(row.get("stage1_pred_edge", 0), errors="coerce") or 0)
    stage2_pruned = int(pd.to_numeric(row.get("stage2_pruned_by_graph", 0), errors="coerce") or 0)
    distance = float(row.get("distance_num", row.get("distance", np.nan)))
    if stage1_pred == 0:
        return "stage1_recall_miss"
    if stage2_pruned == 1:
        return "stage2_graph_pruning_drop"
    if np.isfinite(distance) and distance >= 4:
        return "long_distance_drop"
    if question_form_trigger(row):
        return "boundary_question_form"
    bucket = str(row.get("topic_shift_bucket", "") or "")
    if bucket not in {"", "keep", "nan"}:
        return "topic_boundary_drop"
    return "other_manual_review"


def taxonomy_table(df: pd.DataFrame, label: str, class_col: str) -> pd.DataFrame:
    counts = (
        df[class_col]
        .fillna("other_manual_review")
        .astype(str)
        .value_counts(dropna=False)
        .reset_index()
    )
    counts.columns = ["category", "count"]
    counts["share"] = counts["count"] / max(len(df), 1)
    counts["label"] = label
    return counts[["label", "category", "count", "share"]]


def secondary_breakdown(
    df: pd.DataFrame,
    primary_col: str,
    secondary_col: str,
    error_label: str,
) -> pd.DataFrame:
    """Summarize observed secondary attributes within each error category."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "primary_category",
                "secondary_attribute",
                "count",
                "share_within_error",
                "share_within_primary",
            ]
        )
    work = df.copy()
    work[primary_col] = work[primary_col].fillna("other_manual_review")
    work[secondary_col] = work[secondary_col].fillna("unrecorded").astype(str)
    counts = (
        work.groupby([primary_col, secondary_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={primary_col: "primary_category", secondary_col: "secondary_attribute"})
    )
    counts.insert(0, "label", error_label)
    total = max(len(work), 1)
    primary_totals = counts.groupby("primary_category")["count"].transform("sum")
    counts["share_within_error"] = counts["count"] / total
    counts["share_within_primary"] = counts["count"] / primary_totals
    return counts.sort_values(
        ["primary_category", "count", "secondary_attribute"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def distribution_by_bucket(df: pd.DataFrame, label: str, col: str, bucket_fn) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame([{"label": label, "bucket": "<missing>", "count": len(df), "share": 1.0}])
    buckets = df[col].apply(lambda x: bucket_fn(x))
    counts = buckets.value_counts(dropna=False).reset_index()
    counts.columns = ["bucket", "count"]
    counts["label"] = label
    counts["share"] = counts["count"] / max(len(df), 1)
    return counts[["label", "bucket", "count", "share"]]


def target_degree_summary(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    if "session_id" not in df.columns or "Q2_row" not in df.columns:
        return pd.DataFrame()
    degree = (
        df.loc[df[pred_col].astype(int).eq(1)]
        .groupby(["session_id", "Q2_row"], dropna=False)
        .size()
        .reset_index(name="pred_indegree")
    )
    merged = df.drop(columns=["pred_indegree"], errors="ignore").merge(
        degree, on=["session_id", "Q2_row"], how="left"
    )
    merged["pred_indegree"] = merged["pred_indegree"].fillna(0).astype(int)
    summary = (
        merged.groupby("pred_indegree", dropna=False)
        .agg(
            rows=("edge_id", "size"),
            positives=("edge_label", "sum"),
            predicted=("final_graph_pred_edge", "sum"),
        )
        .reset_index()
    )
    summary["share"] = summary["rows"] / max(len(merged), 1)
    return summary


def consistency_check(stage2_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    merged = stage2_df[["edge_id", "final_graph_pred_edge", "final_graph_score"]].merge(
        main_df[["edge_id", "final_fcn_pred", "final_fcn_score"]],
        on="edge_id",
        how="inner",
    )
    pred_mismatch = int(
        (merged["final_graph_pred_edge"].astype(int) != merged["final_fcn_pred"].astype(int)).sum()
    )
    score_match = np.isclose(
        merged["final_graph_score"].to_numpy(dtype=float),
        merged["final_fcn_score"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    score_mismatch = int((~score_match).sum())
    return pd.DataFrame(
        [
            {"metric": "matched_rows", "value": int(len(merged))},
            {"metric": "pred_mismatches", "value": pred_mismatch},
            {"metric": "score_mismatches", "value": score_mismatch},
            {"metric": "pred_match_rate", "value": float(1.0 - pred_mismatch / max(len(merged), 1))},
        ]
    )


def parse_args() -> argparse.Namespace:
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
        help="Directory for error-analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_formal_bundle(args.bundle_root)
    paths = bundle["paths"]
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else bundle_paths(args.bundle_root).root.parent
        / "experiments"
        / "analysis"
        / "fcn_error_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage2_test = bundle["stage2"]["test_graph"].copy()
    main_test = bundle["main"]["test_predictions"].copy()

    if "final_graph_pred_edge" not in stage2_test.columns:
        raise RuntimeError("Stage 2 test graph is missing final_graph_pred_edge.")

    y_true = pd.to_numeric(stage2_test["edge_label"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(stage2_test["final_graph_pred_edge"], errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(stage2_test["final_graph_score"], errors="coerce").fillna(0.0)
    metrics = binary_metrics(y_true, y_pred, y_score)

    # Merge the main-test predictions to ensure the frozen package is internally
    # consistent with the main experiment outputs.
    consistency_df = consistency_check(stage2_test, main_test)
    save_csv(consistency_df, output_dir / "error_consistency_check.csv")

    # Error rows for inspection.
    fp = stage2_test.loc[(y_pred == 1) & (y_true == 0)].copy()
    fn = stage2_test.loc[(y_pred == 0) & (y_true == 1)].copy()
    tp = stage2_test.loc[(y_pred == 1) & (y_true == 1)].copy()

    # Target-level predicted indegree for over-recovery detection.
    pred_degree = (
        stage2_test.loc[y_pred.eq(1)]
        .groupby(["session_id", "Q2_row"], dropna=False)
        .size()
        .reset_index(name="pred_indegree")
    )
    stage2_test = stage2_test.merge(pred_degree, on=["session_id", "Q2_row"], how="left")
    stage2_test["pred_indegree"] = stage2_test["pred_indegree"].fillna(0).astype(int)
    stage2_test["distance_bucket"] = stage2_test["distance_num"].apply(distance_bucket)
    stage2_test["q2_bucket"] = stage2_test["q2_len"].apply(lambda x: bucket_series(pd.Series([x])).iloc[0])
    stage2_test["q1_q2_jaccard"] = [
        char_jaccard(q1, q2) for q1, q2 in zip(stage2_test["Q1"], stage2_test["Q2"])
    ]
    stage2_test["a1_q2_jaccard"] = [
        char_jaccard(a1, q2) for a1, q2 in zip(stage2_test["A1"], stage2_test["Q2"])
    ]

    stage2_test["fp_taxonomy"] = "not_fp"
    fp_idx = (y_pred == 1) & (y_true == 0)
    stage2_test.loc[fp_idx, "fp_taxonomy"] = stage2_test.loc[fp_idx].apply(
        lambda row: classify_fp(row, int(row.get("pred_indegree", 0))),
        axis=1,
    )
    stage2_test["fn_taxonomy"] = "not_fn"
    fn_idx = (y_pred == 0) & (y_true == 1)
    stage2_test.loc[fn_idx, "fn_taxonomy"] = stage2_test.loc[fn_idx].apply(
        classify_fn,
        axis=1,
    )

    fp_rows = stage2_test.loc[fp_idx].copy()
    fn_rows = stage2_test.loc[fn_idx].copy()
    tp_rows = stage2_test.loc[(y_pred == 1) & (y_true == 1)].copy()

    fp_rows = fp_rows.sort_values(
        ["final_graph_score", "a1_q2_jaccard", "q1_q2_jaccard"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    fn_rows = fn_rows.sort_values(
        ["final_graph_score", "stage2a_local_prob", "stage1_prob"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    fp_tax = taxonomy_table(fp_rows, "FP", "fp_taxonomy")
    fn_tax = taxonomy_table(fn_rows, "FN", "fn_taxonomy")
    fp_rows["distance_band"] = pd.to_numeric(fp_rows["distance_num"], errors="coerce").apply(
        lambda x: distance_bucket(x) if np.isfinite(x) else "unrecorded"
    )
    fp_secondary = secondary_breakdown(
        fp_rows,
        primary_col="fp_taxonomy",
        secondary_col="distance_band",
        error_label="FP",
    )
    fn_secondary = secondary_breakdown(
        fn_rows,
        primary_col="fn_taxonomy",
        secondary_col="relation_type",
        error_label="FN",
    )

    fp_by_distance = distribution_by_bucket(fp_rows, "FP", "distance_num", distance_bucket)
    fn_by_distance = distribution_by_bucket(fn_rows, "FN", "distance_num", distance_bucket)
    fp_by_q2 = distribution_by_bucket(fp_rows, "FP", "q2_len", lambda x: bucket_series(pd.Series([x])).iloc[0])
    fn_by_q2 = distribution_by_bucket(fn_rows, "FN", "q2_len", lambda x: bucket_series(pd.Series([x])).iloc[0])
    fp_by_pred_degree = target_degree_summary(stage2_test, "final_graph_pred_edge")
    fn_stage1_miss = int(((y_true == 1) & (pd.to_numeric(stage2_test["stage1_pred_edge"], errors="coerce").fillna(0).astype(int) == 0)).sum())
    fn_stage2_drop = int(((y_true == 1) & (pd.to_numeric(stage2_test["stage1_pred_edge"], errors="coerce").fillna(0).astype(int) == 1) & (y_pred == 0)).sum())
    fn_stage_attribution = pd.DataFrame(
        [
            {"metric": "stage1_recall_miss", "count": fn_stage1_miss},
            {"metric": "stage2_graph_pruning_drop", "count": fn_stage2_drop},
            {"metric": "total_fn", "count": int(len(fn_rows))},
        ]
    )

    save_csv(fp_tax, output_dir / "error_fp_taxonomy.csv")
    save_csv(fn_tax, output_dir / "error_fn_taxonomy.csv")
    save_csv(fp_secondary, output_dir / "error_fp_secondary_breakdown.csv")
    save_csv(fn_secondary, output_dir / "error_fn_secondary_breakdown.csv")
    save_csv(fp_by_distance, output_dir / "error_fp_by_distance.csv")
    save_csv(fn_by_distance, output_dir / "error_fn_by_distance.csv")
    save_csv(fp_by_q2, output_dir / "error_fp_by_q2_bucket.csv")
    save_csv(fn_by_q2, output_dir / "error_fn_by_q2_bucket.csv")
    save_csv(fp_by_pred_degree, output_dir / "error_pred_indegree_summary.csv")
    save_csv(fn_stage_attribution, output_dir / "error_fn_stage_attribution.csv")

    review_cols = [
        "edge_id",
        "session_id",
        "Q1_row",
        "Q2_row",
        "distance",
        "distance_num",
        "edge_label",
        "final_graph_pred_edge",
        "final_graph_score",
        "stage1_prob",
        "stage2a_local_prob",
        "stage2b_target_rank",
        "stage2b_incremental_gain",
        "pred_indegree",
        "fp_taxonomy",
        "fn_taxonomy",
        "relation_type",
        "topic_shift_bucket",
        "topic_shift_flags",
        "q1_q2_jaccard",
        "a1_q2_jaccard",
        "Q1",
        "A1",
        "Q2",
        "dep_reason",
    ]
    review_cols = [col for col in review_cols if col in stage2_test.columns]
    save_csv(fp_rows[review_cols], output_dir / "error_fp_cases.csv")
    save_csv(fn_rows[review_cols], output_dir / "error_fn_cases.csv")
    save_csv(tp_rows[review_cols], output_dir / "error_tp_cases.csv")

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": int(len(stage2_test))},
            {"metric": "true_edges", "value": int(y_true.sum())},
            {"metric": "pred_edges", "value": int(y_pred.sum())},
            {"metric": "tp", "value": int(metrics["tp"])},
            {"metric": "fp", "value": int(metrics["fp"])},
            {"metric": "fn", "value": int(metrics["fn"])},
            {"metric": "tn", "value": int(metrics["tn"])},
            {"metric": "precision", "value": float(metrics["precision"])},
            {"metric": "recall", "value": float(metrics["recall"])},
            {"metric": "f1", "value": float(metrics["f1"])},
            {"metric": "auc", "value": float(metrics["auc"])},
            {"metric": "ap", "value": float(metrics["ap"])},
            {"metric": "consistency_pred_mismatches", "value": int(consistency_df.loc[consistency_df["metric"] == "pred_mismatches", "value"].iloc[0])},
            {"metric": "consistency_score_mismatches", "value": int(consistency_df.loc[consistency_df["metric"] == "score_mismatches", "value"].iloc[0])},
        ]
    )
    save_csv(summary, output_dir / "error_summary.csv")

    run_status = {
        "completed": True,
        "bundle_root": str(paths.root),
        "rows": int(len(stage2_test)),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "consistency_pred_mismatches": int(
            consistency_df.loc[consistency_df["metric"] == "pred_mismatches", "value"].iloc[0]
        ),
        "consistency_score_mismatches": int(
            consistency_df.loc[consistency_df["metric"] == "score_mismatches", "value"].iloc[0]
        ),
    }
    save_json(run_status, output_dir / "error_analysis_run_status.json")

    print_block("Error Analysis Summary", run_status)
    print("\nPrimary output:")
    print(output_dir / "error_analysis_run_status.json")


if __name__ == "__main__":
    main()
