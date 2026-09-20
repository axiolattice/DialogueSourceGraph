#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core FCN baselines under the frozen 176/44/55 Train/Validation/Test protocol.

Supervised baselines are fitted on Train only. Thresholds and the Top-K
operating point are selected on Validation only. Test is evaluated only after
all baseline parameters are frozen. Final FCN is loaded from the frozen Stage
2 graph and is not retrained here.

Core comparisons:
BM25; Temporal-distance Prior; General Chinese Sentence Encoder Retrieval;
General Chinese Retrieval Encoder Retrieval; Financial-domain Encoder
Retrieval; General Chinese Pairwise Transformer + Linear Probe; Financial-domain
Pairwise Transformer + Linear Probe; Score Fusion; Pairwise Transformer + top-K
pruning; GraphSAGE; Dir-SAGE; Final FCN.

Standalone comparison mode:
Fine-tuned Financial Pairwise, which updates the financial pair encoder and its
classification head on Train while preserving the established split protocol.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:
    hf_snapshot_download = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_STAGE1_DIR = PROJECT_DIR / "stage1" / "fcn_outputs_stage1"
DEFAULT_STAGE2_DIR = PROJECT_DIR / "stage2" / "fcn_outputs_stage2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_outputs_main_experiments"
DEFAULT_FINETUNED_PAIRWISE_OUTPUT_DIR = SCRIPT_DIR / "fcn_outputs_finetuned_financial_pairwise"
DEFAULT_HF_CACHE_DIR = PROJECT_DIR / "checkpoints" / "hf_cache"

EXPECTED_TRAIN_SESSIONS = 176
EXPECTED_VALIDATION_SESSIONS = 44
EXPECTED_TEST_SESSIONS = 55
SEED = 42

DEFAULT_MODELS = {
    "sentence_encoder": "DMetaSoul/sbert-chinese-general-v2",
    "retrieval_encoder": "moka-ai/m3e-base",
    "financial_encoder": "Langboat/mengzi-bert-base-fin",
    "pairwise_cn": "hfl/chinese-roberta-wwm-ext",
    "pairwise_fin": "yiyanghkust/finbert-tone-chinese",
    "graphsage": "DMetaSoul/sbert-chinese-general-v2",
}

CORE_BASELINE_ORDER = (
    "BM25",
    "Temporal-distance Prior",
    "General Chinese Sentence Encoder Retrieval",
    "General Chinese Retrieval Encoder Retrieval",
    "Financial-domain Encoder Retrieval",
    "General Chinese Pairwise Transformer + Linear Probe",
    "Financial-domain Pairwise Transformer + Linear Probe",
    "Score Fusion",
    "Pairwise Transformer + top-K pruning",
    "GraphSAGE",
    "Dir-SAGE",
    "Final FCN",
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
LOGGER = logging.getLogger("fcn-main-experiments")


# =============================================================================
# Data
# =============================================================================

def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@lru_cache(maxsize=None)
def directory_fingerprint(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing model directory: {path}")
    digest = hashlib.sha256()
    files = sorted(
        [p for p in path.rglob("*") if p.is_file()],
        key=lambda p: str(p.relative_to(path)).replace(os.sep, "/"),
    )
    for file in files:
        rel = str(file.relative_to(path)).replace(os.sep, "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def model_fingerprint(repo_id_or_path: str, cache_dir: str) -> Dict[str, str]:
    resolved = resolve_hf_model(repo_id_or_path, cache_dir)
    return {
        "requested": repo_id_or_path,
        "resolved_path": str(resolved),
    }


def json_sha256(payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256_bytes(blob)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")


def load_session_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"session_id", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {missing}")
    df = df.copy()
    df["session_id"] = df["session_id"].astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    valid = {"train", "validation", "test"}
    bad = sorted(set(df["split"]) - valid)
    if bad:
        raise RuntimeError(f"{path.name} has invalid split labels: {bad}")
    if df["session_id"].duplicated().any():
        dup = df.loc[df["session_id"].duplicated(), "session_id"].head(5).tolist()
        raise RuntimeError(f"{path.name} has duplicate session IDs: {dup}")
    return df.reset_index(drop=True)


def manifest_split_sets(manifest: pd.DataFrame) -> Dict[str, set]:
    grouped = {}
    for split in ("train", "validation", "test"):
        grouped[split] = set(
            manifest.loc[manifest["split"] == split, "session_id"].astype(str)
        )
    return grouped


def verify_split_manifest(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    grouped = manifest_split_sets(manifest)
    observed = {
        "train": set(train_df["session_id"].astype(str)),
        "validation": set(validation_df["session_id"].astype(str)),
        "test": set(test_df["session_id"].astype(str)),
    }
    for split in ("train", "validation", "test"):
        if observed[split] != grouped[split]:
            missing = sorted(grouped[split] - observed[split])
            extra = sorted(observed[split] - grouped[split])
            raise RuntimeError(
                f"{split} session set mismatch. Missing={missing[:5]}, extra={extra[:5]}"
            )


def expected_counts_check(
    name: str,
    df: pd.DataFrame,
    expected_rows: int,
    expected_true_edges: int,
    expected_sessions: int,
) -> None:
    observed = (
        len(df),
        int(df["edge_label"].sum()),
        int(df["session_id"].nunique()),
    )
    expected = (expected_rows, expected_true_edges, expected_sessions)
    if observed != expected:
        raise RuntimeError(
            f"{name} expected rows/edges/sessions {expected}, observed {observed}."
        )


def verify_stage1_artifacts(
    stage1_dir: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    manifest_path: Path,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    params_path = stage1_dir / "stage1_final_selected_params.json"
    params = load_json(params_path)
    manifest = load_session_manifest(manifest_path)

    expected_counts_check("Train", train_df, 39880, 1744, 176)
    expected_counts_check("Validation", validation_df, 8379, 299, 44)
    if test_df is not None:
        expected_counts_check("Test", test_df, 10434, 528, 55)
        verify_split_manifest(train_df, validation_df, test_df, manifest)
    else:
        manifest_sets = manifest_split_sets(manifest)
        if set(train_df["session_id"].astype(str)) != manifest_sets["train"]:
            raise RuntimeError("Train session set mismatch with session manifest.")
        if set(validation_df["session_id"].astype(str)) != manifest_sets["validation"]:
            raise RuntimeError("Validation session set mismatch with session manifest.")
        if train_df["session_id"].astype(str).isin(manifest_sets["validation"]).any():
            raise RuntimeError("Train session overlap with validation manifest.")
        if validation_df["session_id"].astype(str).isin(manifest_sets["train"]).any():
            raise RuntimeError("Validation session overlap with train manifest.")

    for key, expected in {
        "retention_orientation": "source_conditioned",
        "retention_group_keys": ["session_id", "Q1_row"],
    }.items():
        if params.get(key) != expected:
            raise RuntimeError(
                f"Stage 1 parameter manifest mismatch for {key}: {params.get(key)!r}"
            )

    return params


def verify_stage2_artifacts(
    stage2_dir: Path,
    stage1_params: dict,
    validation_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
) -> dict:
    params_path = stage2_dir / "stage2_selected_params.json"
    params = load_json(params_path)
    val_graph = stage2_dir / "stage2_validation_final_graph.csv"
    status = stage2_dir / "stage2_run_status.json"

    for path in (val_graph, status):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
    if test_df is not None:
        test_graph = stage2_dir / "stage2_test_final_graph.csv"
        exactness = stage2_dir / "stage2_exactness_audit.csv"
        for path in (test_graph, exactness):
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

    split_protocol = params.get("split_protocol", {})
    if split_protocol.get("train_sessions") != 176 or split_protocol.get("validation_sessions") != 44:
        raise RuntimeError(f"Stage 2 split protocol mismatch: {split_protocol}")
    if not split_protocol.get("sessions_disjoint", False):
        raise RuntimeError("Stage 2 split protocol does not confirm disjoint sessions.")
    if not split_protocol.get("test_loaded_after_validation_selection", False):
        raise RuntimeError("Stage 2 did not freeze Validation before reading Test.")

    stage1_input = params.get("stage1_input", {})
    for key in ("retention_orientation", "retention_group_keys"):
        if key not in stage1_input:
            raise RuntimeError(f"Stage 2 manifest missing stage1_input.{key}")
    if stage1_input.get("retention_orientation") != stage1_params.get("retention_orientation"):
        raise RuntimeError("Stage 2 stage1_input retention orientation mismatch.")
    if stage1_input.get("retention_group_keys") != stage1_params.get("retention_group_keys"):
        raise RuntimeError("Stage 2 stage1_input retention group keys mismatch.")

    for key, expected in {
        "total": 8379,
        "true_edges": 299,
    }.items():
        if params.get("validation_metrics", {}).get(key) != expected:
            raise RuntimeError(f"Stage 2 validation metric mismatch for {key}.")

    expected_counts_check("Validation", validation_df, 8379, 299, 44)
    if test_df is not None:
        if split_protocol.get("test_sessions") != 55:
            raise RuntimeError(f"Stage 2 test session count mismatch: {split_protocol}")
        for key, expected in {
            "total": 10434,
            "true_edges": 528,
        }.items():
            if params.get("test_metrics", {}).get(key) != expected:
                raise RuntimeError(f"Stage 2 test metric mismatch for {key}.")
        expected_counts_check("Test", test_df, 10434, 528, 55)

    if params.get("rationale_derived_supervision", True) is not False:
        raise RuntimeError("Stage 2 parameter manifest is inconsistent with the no-rationale protocol.")
    if params.get("test_labels_used_for_fitting_or_selection", True) is not False:
        raise RuntimeError("Stage 2 parameter manifest indicates test leakage.")

    exact_df = pd.read_csv(exactness, encoding="utf-8-sig")
    required = {
        "split",
        "objective_mismatches",
        "subset_mismatches",
        "max_objective_gap",
        "passed",
    }
    missing = sorted(required - set(exact_df.columns))
    if missing:
        raise RuntimeError(f"Stage 2 exactness audit missing columns: {missing}")
    bad = exact_df.loc[
        (exact_df["objective_mismatches"].fillna(-1).astype(int) != 0)
        | (exact_df["subset_mismatches"].fillna(-1).astype(int) != 0)
        | (~exact_df["passed"].astype(bool))
    ]
    if not bad.empty:
        raise RuntimeError(
            "Stage 2 exactness audit failed for splits: "
            + ", ".join(bad["split"].astype(str).tolist())
        )
    if (exact_df["max_objective_gap"].fillna(np.inf).astype(float) > 1e-10).any():
        raise RuntimeError("Stage 2 exactness audit max_objective_gap exceeded tolerance.")

    return params


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edge_label" not in out.columns:
        if "dependent" in out.columns:
            out["edge_label"] = (
                out["dependent"].astype(str).str.lower() == "yes"
            ).astype(int)
        elif "label" in out.columns:
            out["edge_label"] = (
                pd.to_numeric(out["label"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            raise ValueError("Cannot derive edge_label.")

    out["edge_label"] = (
        pd.to_numeric(out["edge_label"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "distance_num" in out.columns:
        out["distance_num"] = (
            pd.to_numeric(out["distance_num"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    elif "distance" in out.columns:
        out["distance_num"] = (
            pd.to_numeric(out["distance"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    else:
        out["distance_num"] = (
            pd.to_numeric(out["Q2_row"], errors="coerce")
            - pd.to_numeric(out["Q1_row"], errors="coerce")
        ).fillna(0.0).astype(float)
    return out


def build_pair_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q1 = (
        out["Q1"].map(clean_text)
        if "Q1" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    a1 = (
        out["A1"].map(clean_text)
        if "A1" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    q2 = (
        out["Q2"].map(clean_text)
        if "Q2" in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    out["anchor_text"] = (
        "前序问题：" + q1.fillna("") + " 管理层回答：" + a1.fillna("")
    )
    out["candidate_text"] = "当前追问：" + q2.fillna("")
    return out


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = build_pair_text_columns(normalize_columns(df))
    required = {"session_id", "Q1_row", "Q2_row", "edge_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {missing}")
    return df.reset_index(drop=True)


def verify_three_way_protocol(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    require_fixed_counts: bool = True,
) -> None:
    observed = (
        int(train_df["session_id"].nunique()),
        int(validation_df["session_id"].nunique()),
        int(test_df["session_id"].nunique()),
    )
    expected = (
        EXPECTED_TRAIN_SESSIONS,
        EXPECTED_VALIDATION_SESSIONS,
        EXPECTED_TEST_SESSIONS,
    )
    if require_fixed_counts and observed != expected:
        raise RuntimeError(
            f"Expected 176/44/55 sessions; observed {observed}."
        )

    tr = set(train_df["session_id"].astype(str))
    va = set(validation_df["session_id"].astype(str))
    te = set(test_df["session_id"].astype(str))
    if tr & va or tr & te or va & te:
        raise RuntimeError("Train/Validation/Test session sets overlap.")


def align_graph_to_reference(
    reference_df: pd.DataFrame,
    graph_df: pd.DataFrame,
) -> pd.DataFrame:
    ref = reference_df.reset_index(drop=True)
    graph = graph_df.reset_index(drop=True)

    keys = (
        ["edge_id"]
        if "edge_id" in ref.columns and "edge_id" in graph.columns
        else ["session_id", "Q1_row", "Q2_row"]
    )

    if len(ref) == len(graph):
        same = True
        for col in keys:
            if not ref[col].astype(str).equals(graph[col].astype(str)):
                same = False
                break
        if same:
            return graph

    ref_key = ref[keys].copy()
    graph_work = graph.copy()
    for col in keys:
        ref_key[col] = ref_key[col].astype(str)
        graph_work[col] = graph_work[col].astype(str)

    payload = [c for c in graph_work.columns if c not in keys]
    merged = ref_key.merge(
        graph_work[keys + payload],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if "final_graph_pred_edge" not in merged.columns:
        raise RuntimeError("Final graph alignment failed.")
    if merged["final_graph_pred_edge"].isna().any():
        raise RuntimeError("Unmatched rows during final graph alignment.")
    return merged.reset_index(drop=True)


# =============================================================================
# Metrics and Validation-only selection
# =============================================================================

def safe_auc(y_true, y_score) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(y_score, dtype=float)))


def safe_ap(y_true, y_score) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(y_score, dtype=float)))


def binary_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    score = np.asarray(y_score, dtype=float)
    tp = int(np.logical_and(y == 1, pred == 1).sum())
    fp = int(np.logical_and(y == 0, pred == 1).sum())
    fn = int(np.logical_and(y == 1, pred == 0).sum())
    tn = int(np.logical_and(y == 0, pred == 0).sum())
    return {
        "AUC": safe_auc(y, score),
        "AP": safe_ap(y, score),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "PredEdges": int(pred.sum()),
        "TrueEdges": int(y.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def select_threshold_exact(
    y_true,
    scores,
    eligible_mask=None,
) -> Tuple[float, Dict[str, float]]:
    """Exact threshold maximizing Validation F1.

    Tie order: higher precision, fewer predicted edges, larger threshold.
    When eligible_mask is provided, only eligible rows can be predicted
    positive, but FN is counted over all Validation positives.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.nan_to_num(
        np.asarray(scores, dtype=float),
        nan=0.0,
        posinf=1e30,
        neginf=-1e30,
    )
    eligible = (
        np.ones(len(y), dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool)
    )
    idx = np.where(eligible)[0]
    total_pos = int(y.sum())

    if len(idx) == 0:
        pred = np.zeros(len(y), dtype=int)
        return float("inf"), binary_metrics(y, pred, s)

    order = idx[np.argsort(-s[idx], kind="mergesort")]
    oscore = s[order]
    olabel = y[order]
    ctp = np.cumsum(olabel == 1)
    cfp = np.cumsum(olabel == 0)
    ends = np.r_[
        np.where(oscore[:-1] != oscore[1:])[0],
        len(order) - 1,
    ]

    best_key = None
    best_threshold = None
    for pos in ends:
        tp = int(ctp[pos])
        fp = int(cfp[pos])
        fn = total_pos - tp
        pred_edges = tp + fp
        precision = tp / pred_edges if pred_edges else 0.0
        f1 = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn)
            else 0.0
        )
        threshold = float(oscore[pos])
        key = (f1, precision, -pred_edges, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold

    threshold = float(best_threshold)
    pred = (eligible & (s >= threshold)).astype(int)
    return threshold, binary_metrics(y, pred, s)


def anchor_topk_mask(
    df: pd.DataFrame,
    scores,
    top_k: int,
) -> np.ndarray:
    work = df.reset_index(drop=True)
    s = np.asarray(scores, dtype=float)
    keep = np.zeros(len(work), dtype=bool)
    for _, group in work.groupby(["session_id", "Q1_row"], sort=False):
        idxs = group.index.to_numpy(dtype=int)
        order = sorted(
            idxs.tolist(),
            key=lambda idx: (-float(s[idx]), int(idx)),
        )
        keep[order[: int(top_k)]] = True
    return keep


def select_topk_and_threshold(
    validation_df: pd.DataFrame,
    validation_scores,
    k_values: Sequence[int],
) -> Tuple[int, float, pd.DataFrame]:
    rows = []
    y = validation_df["edge_label"].to_numpy(dtype=int)
    for k in k_values:
        eligible = anchor_topk_mask(validation_df, validation_scores, k)
        theta, metrics = select_threshold_exact(
            y,
            validation_scores,
            eligible_mask=eligible,
        )
        rows.append(
            {
                "TopK": int(k),
                "Threshold": float(theta),
                **metrics,
            }
        )

    search = pd.DataFrame(rows)
    ordered = search.sort_values(
        ["F1", "Precision", "PredEdges", "TopK", "Threshold"],
        ascending=[False, False, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best = ordered.iloc[0]
    return int(best["TopK"]), float(best["Threshold"]), search


def evaluate_threshold_model(
    name: str,
    purpose: str,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_scores,
    test_scores,
):
    theta, _ = select_threshold_exact(
        validation_df["edge_label"],
        val_scores,
    )
    val_pred = (np.asarray(val_scores) >= theta).astype(int)
    test_pred = (np.asarray(test_scores) >= theta).astype(int)

    val_row = {
        "Split": "Validation",
        "Model": name,
        "Purpose": purpose,
        "Threshold": theta,
        "TopK": np.nan,
        **binary_metrics(
            validation_df["edge_label"],
            val_pred,
            val_scores,
        ),
    }
    test_row = {
        "Split": "Test",
        "Model": name,
        "Purpose": purpose,
        "Threshold": theta,
        "TopK": np.nan,
        **binary_metrics(
            test_df["edge_label"],
            test_pred,
            test_scores,
        ),
    }
    return val_row, test_row, np.asarray(test_pred, dtype=int)


def evaluate_topk_model(
    name: str,
    purpose: str,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_scores,
    test_scores,
    k_values: Sequence[int],
):
    k, theta, search = select_topk_and_threshold(
        validation_df,
        val_scores,
        k_values,
    )
    val_eligible = anchor_topk_mask(validation_df, val_scores, k)
    test_eligible = anchor_topk_mask(test_df, test_scores, k)
    val_pred = (
        val_eligible & (np.asarray(val_scores) >= theta)
    ).astype(int)
    test_pred = (
        test_eligible & (np.asarray(test_scores) >= theta)
    ).astype(int)

    val_row = {
        "Split": "Validation",
        "Model": name,
        "Purpose": purpose,
        "Threshold": theta,
        "TopK": k,
        **binary_metrics(
            validation_df["edge_label"],
            val_pred,
            val_scores,
        ),
    }
    test_row = {
        "Split": "Test",
        "Model": name,
        "Purpose": purpose,
        "Threshold": theta,
        "TopK": k,
        **binary_metrics(
            test_df["edge_label"],
            test_pred,
            test_scores,
        ),
    }
    return val_row, test_row, np.asarray(test_pred, dtype=int), search


def evaluate_threshold_fixed(
    name: str,
    purpose: str,
    test_df: pd.DataFrame,
    test_scores,
    threshold: float,
):
    test_pred = (np.asarray(test_scores) >= float(threshold)).astype(int)
    return {
        "Split": "Test",
        "Model": name,
        "Purpose": purpose,
        "Threshold": float(threshold),
        "TopK": np.nan,
        **binary_metrics(
            test_df["edge_label"],
            test_pred,
            test_scores,
        ),
    }, np.asarray(test_pred, dtype=int)


def evaluate_topk_fixed(
    name: str,
    purpose: str,
    test_df: pd.DataFrame,
    test_scores,
    threshold: float,
    top_k: int,
):
    test_eligible = anchor_topk_mask(test_df, test_scores, int(top_k))
    test_pred = (
        test_eligible & (np.asarray(test_scores) >= float(threshold))
    ).astype(int)
    return {
        "Split": "Test",
        "Model": name,
        "Purpose": purpose,
        "Threshold": float(threshold),
        "TopK": int(top_k),
        **binary_metrics(
            test_df["edge_label"],
            test_pred,
            test_scores,
        ),
    }, np.asarray(test_pred, dtype=int)


# =============================================================================
# Baselines retained from the prior FCN_main_experiments.py
# =============================================================================

def simple_tokenize(text: object) -> List[str]:
    value = clean_text(text)
    if not value:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", value)
    return tokens if tokens else list(value)


def bm25_tokenize(text: object) -> List[str]:
    """Character bigram BM25 tokenization with ASCII word preservation."""
    value = clean_text(text)
    if not value:
        return []
    tokens: List[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", value):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend([chunk[i : i + 2] for i in range(len(chunk) - 1)])
        else:
            tokens.append(chunk.lower())
    return tokens if tokens else list(value)


class SimpleBM25:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lens = np.asarray([len(x) for x in corpus], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        df_counts: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            for token in set(doc):
                df_counts[token] += 1
        n_docs = max(len(corpus), 1)
        self.idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df_counts.items()
        }

    def score(self, query_tokens, doc_tokens) -> float:
        tf: Dict[str, int] = defaultdict(int)
        for token in doc_tokens:
            tf[token] += 1
        doc_len = float(len(doc_tokens))
        norm = self.k1 * (
            1.0
            - self.b
            + self.b * (doc_len / self.avgdl if self.avgdl else 0.0)
        )
        score = 0.0
        for token in query_tokens:
            count = tf.get(token, 0)
            if count <= 0:
                continue
            score += (
                self.idf.get(token, 0.0)
                * (count * (self.k1 + 1.0))
                / (count + norm + 1e-9)
            )
        return float(score)


def fit_bm25(train_df: pd.DataFrame) -> SimpleBM25:
    unique_docs = list(dict.fromkeys(train_df["anchor_text"].map(clean_text).tolist()))
    return SimpleBM25([bm25_tokenize(x) for x in unique_docs])


def score_bm25(df: pd.DataFrame, scorer: SimpleBM25) -> np.ndarray:
    values = []
    for row in df.itertuples(index=False):
        values.append(
            scorer.score(
                bm25_tokenize(getattr(row, "candidate_text", "")),
                bm25_tokenize(getattr(row, "anchor_text", "")),
            )
        )
    return np.asarray(values, dtype=float)


def compute_distance_prior_scores(df: pd.DataFrame) -> np.ndarray:
    d = np.clip(
        pd.to_numeric(df["distance_num"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float),
        1.0,
        None,
    )
    return 1.0 / d


def resolve_hf_model(repo_id_or_path: str, cache_dir: str) -> str:
    if os.path.exists(repo_id_or_path):
        return repo_id_or_path
    if hf_snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub unavailable; provide a local model path."
        )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    endpoints = [
        os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        "https://huggingface.co",
    ]
    last_exc = None
    for endpoint in endpoints:
        try:
            return hf_snapshot_download(
                repo_id=repo_id_or_path,
                cache_dir=str(cache_dir),
                local_files_only=False,
                endpoint=endpoint,
            )
        except Exception as exc:
            last_exc = exc
            LOGGER.warning(
                "Failed %s via %s: %s",
                repo_id_or_path,
                endpoint,
                exc,
            )
    raise RuntimeError(f"Cannot load {repo_id_or_path}: {last_exc}")


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(
        min=1e-9
    )


def load_auto_encoder(repo_id: str, cache_dir: str):
    model_path = resolve_hf_model(repo_id, cache_dir)
    return (
        AutoTokenizer.from_pretrained(model_path),
        AutoModel.from_pretrained(model_path),
    )


def encode_auto_texts(
    model,
    tokenizer,
    texts_a,
    texts_b,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(texts_a), int(batch_size)):
            a = list(texts_a[start : start + int(batch_size)])
            b = (
                list(texts_b[start : start + int(batch_size)])
                if texts_b is not None
                else None
            )
            if b is None:
                encoded = tokenizer(
                    a,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    a,
                    b,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model_out = model(**encoded)
            pooled = (
                model_out.pooler_output
                if getattr(model_out, "pooler_output", None) is not None
                else mean_pool(
                    model_out.last_hidden_state,
                    encoded["attention_mask"],
                )
            )
            outputs.append(pooled.detach().cpu().numpy())

    return (
        np.vstack(outputs).astype(np.float32)
        if outputs
        else np.zeros((0, 0), dtype=np.float32)
    )


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.clip(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1),
        1e-12,
        None,
    )
    return np.sum(a * b, axis=1) / denom


def cleanup_model(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_sentence_encoder_scores_3way(
    train_df,
    validation_df,
    test_df,
    repo_id,
    cache_dir,
    batch_size,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    model = SentenceTransformer(model_path)

    def score(df):
        a = np.asarray(
            model.encode(
                df["anchor_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        b = np.asarray(
            model.encode(
                df["candidate_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        return cosine_scores(a, b)

    result = (score(train_df), score(validation_df), score(test_df))
    cleanup_model(model)
    return result


def compute_transformer_retrieval_scores_3way(
    train_df,
    validation_df,
    test_df,
    repo_id,
    cache_dir,
    batch_size,
    max_length,
    use_sentence_transformer: bool = False,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    if use_sentence_transformer:
        model = SentenceTransformer(model_path)

        def score(df):
            a = np.asarray(
                model.encode(
                    df["anchor_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            b = np.asarray(
                model.encode(
                    df["candidate_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            return cosine_scores(a, b)

        result = (score(train_df), score(validation_df), score(test_df))
        cleanup_model(model)
        return result

    tokenizer, model = load_auto_encoder(repo_id, cache_dir)

    def score(df):
        a = encode_auto_texts(
            model,
            tokenizer,
            df["anchor_text"].tolist(),
            None,
            batch_size,
            max_length,
        )
        b = encode_auto_texts(
            model,
            tokenizer,
            df["candidate_text"].tolist(),
            None,
            batch_size,
            max_length,
        )
        return cosine_scores(a, b)

    result = (score(train_df), score(validation_df), score(test_df))
    cleanup_model(model, tokenizer)
    return result


def compute_pairwise_scores_3way(
    train_df,
    validation_df,
    test_df,
    repo_id,
    cache_dir,
    batch_size,
    max_length,
):
    """Same joint-encoding + LR baseline as the prior main script."""
    tokenizer, encoder = load_auto_encoder(repo_id, cache_dir)

    def embed(df):
        return encode_auto_texts(
            encoder,
            tokenizer,
            df["anchor_text"].tolist(),
            df["candidate_text"].tolist(),
            batch_size,
            max_length,
        )

    train_emb = embed(train_df)
    val_emb = embed(validation_df)
    test_emb = embed(test_df)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    model.fit(train_emb, train_df["edge_label"].to_numpy(dtype=int))

    result = (
        model.predict_proba(train_emb)[:, 1].astype(float),
        model.predict_proba(val_emb)[:, 1].astype(float),
        model.predict_proba(test_emb)[:, 1].astype(float),
    )
    cleanup_model(encoder, tokenizer)
    return result


def compute_score_fusion_scores_3way(
    train_df,
    validation_df,
    test_df,
    bm25_scores,
    semantic_scores,
    pairwise_scores,
):
    def inv_distance(df):
        d = np.clip(
            pd.to_numeric(df["distance_num"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float),
            1.0,
            None,
        )
        return 1.0 / d

    train_x = np.column_stack(
        [
            bm25_scores[0],
            semantic_scores[0],
            pairwise_scores[0],
            inv_distance(train_df),
        ]
    )
    val_x = np.column_stack(
        [
            bm25_scores[1],
            semantic_scores[1],
            pairwise_scores[1],
            inv_distance(validation_df),
        ]
    )
    test_x = np.column_stack(
        [
            bm25_scores[2],
            semantic_scores[2],
            pairwise_scores[2],
            inv_distance(test_df),
        ]
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    model.fit(train_x, train_df["edge_label"].to_numpy(dtype=int))
    return (
        model.predict_proba(train_x)[:, 1].astype(float),
        model.predict_proba(val_x)[:, 1].astype(float),
        model.predict_proba(test_x)[:, 1].astype(float),
    )


# =============================================================================
# GraphSAGE baseline
# =============================================================================

class GraphSAGELayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim * 2, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj is row-normalized adjacency with self-loops.
        neigh = adj @ x
        h = torch.cat([x, neigh], dim=-1)
        return torch.nn.functional.gelu(self.norm(self.lin(h)))


class GraphSAGEEdgeModel(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            layers.append(GraphSAGELayer(dims[i], dims[i + 1]))
        self.layers = torch.nn.ModuleList(layers)
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, adj)
        hs = h[src_idx]
        hd = h[dst_idx]
        edge_feat = torch.cat([hs, hd, torch.abs(hs - hd), hs * hd], dim=-1)
        return self.edge_head(edge_feat).squeeze(-1)


class DirSAGELayer(torch.nn.Module):
    """GraphSAGE layer with separate incoming and outgoing aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim * 3, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        in_adj: torch.Tensor,
        out_adj: torch.Tensor,
    ) -> torch.Tensor:
        incoming = in_adj @ x
        outgoing = out_adj @ x
        h = torch.cat([x, incoming, outgoing], dim=-1)
        return torch.nn.functional.gelu(self.norm(self.lin(h)))


class DirSAGEEdgeModel(torch.nn.Module):
    """Directed GraphSAGE edge classifier over the candidate adjacency."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            layers.append(DirSAGELayer(dims[i], dims[i + 1]))
        self.layers = torch.nn.ModuleList(layers)
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        x: torch.Tensor,
        in_adj: torch.Tensor,
        out_adj: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, in_adj, out_adj)
        return h

    def forward(
        self,
        x: torch.Tensor,
        in_adj: torch.Tensor,
        out_adj: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, in_adj, out_adj)
        hs = h[src_idx]
        hd = h[dst_idx]
        edge_feat = torch.cat([hs, hd, torch.abs(hs - hd), hs * hd], dim=-1)
        return self.edge_head(edge_feat).squeeze(-1)


class ScoreTriplet(list):
    """List-like container for cached split scores plus metadata."""
    pass


def _node_text_map(df: pd.DataFrame) -> Dict[Tuple[str, int, str], str]:
    """Map each role-specific turn node to its textual representation."""
    mapping: Dict[Tuple[str, int, str], str] = {}
    for row in df.itertuples(index=False):
        sid = str(getattr(row, "session_id"))
        q1r = int(getattr(row, "Q1_row"))
        q2r = int(getattr(row, "Q2_row"))
        mapping[(sid, q1r, "source")] = (
            clean_text(getattr(row, "Q1", ""))
            + " "
            + clean_text(getattr(row, "A1", ""))
        )
        mapping[(sid, q2r, "target")] = clean_text(getattr(row, "Q2", ""))
    return mapping


def compute_graphsage_scores_3way(
    train_df,
    validation_df,
    test_df,
    repo_id,
    cache_dir,
    batch_size,
    max_length,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    encoder = SentenceTransformer(model_path)

    # Node embeddings are built from turn text using SentenceTransformer mean
    # pooling with embedding normalization.
    all_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    node_map = _node_text_map(all_df)
    node_keys = list(node_map.keys())
    node_texts = [node_map[k] for k in node_keys]
    node_emb = np.asarray(
        encoder.encode(
            node_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    node_embs = {k: node_emb[i] for i, k in enumerate(node_keys)}
    encoder.to("cpu")
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def score_split():
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        def build_split_graphs(split_df: pd.DataFrame):
            graphs = []
            for _, g in split_df.groupby("session_id", sort=False):
                sid = str(g["session_id"].iloc[0])
                node_keys = []
                for row in g.itertuples(index=False):
                    q1r = int(getattr(row, "Q1_row"))
                    q2r = int(getattr(row, "Q2_row"))
                    node_keys.append((sid, q1r, "source"))
                    node_keys.append((sid, q2r, "target"))
                node_keys = sorted(set(node_keys), key=lambda x: (x[1], x[2]))
                idx_map = {k: i for i, k in enumerate(node_keys)}
                x = np.vstack([node_embs[k] for k in node_keys]).astype(np.float32)
                n = len(node_keys)
                adj = np.zeros((n, n), dtype=np.float32)
                np.fill_diagonal(adj, 1.0)
                for row in g.itertuples(index=False):
                    i = idx_map[(sid, int(getattr(row, "Q1_row")), "source")]
                    j = idx_map[(sid, int(getattr(row, "Q2_row")), "target")]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
                deg = adj.sum(axis=1, keepdims=True)
                adj = adj / np.clip(deg, 1.0, None)
                edge_src = np.asarray(
                    [idx_map[(sid, int(x), "source")] for x in g["Q1_row"].tolist()],
                    dtype=np.int64,
                )
                edge_dst = np.asarray(
                    [idx_map[(sid, int(x), "target")] for x in g["Q2_row"].tolist()],
                    dtype=np.int64,
                )
                y = g["edge_label"].to_numpy(dtype=np.float32)
                graphs.append((g.index.to_numpy(dtype=int), x, adj, edge_src, edge_dst, y))
            return graphs

        train_graphs = build_split_graphs(train_df)
        val_graphs = build_split_graphs(validation_df)
        test_graphs = build_split_graphs(test_df)
        in_dim = int(train_graphs[0][1].shape[1]) if train_graphs else int(node_emb.shape[1])
        model = GraphSAGEEdgeModel(in_dim=in_dim, hidden_dim=128, n_layers=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_labels = train_df["edge_label"].to_numpy(dtype=np.float32)
        pos = float(np.clip(train_labels.sum(), 1.0, None))
        neg = float(len(train_labels) - train_labels.sum())
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="sum",
        )

        rng = np.random.default_rng(SEED)
        best_state = None
        best_val_ap = -1.0
        best_epoch = -1
        patience = 5
        bad_epochs = 0
        max_epochs = 50
        for epoch in range(max_epochs):
            model.train()
            order = np.arange(len(train_graphs))
            rng.shuffle(order)
            opt.zero_grad()
            total_loss = 0.0
            for idx in order:
                _, x, adj, src, dst, y = train_graphs[idx]
                x_t = torch.from_numpy(x).to(device)
                adj_t = torch.from_numpy(adj).to(device)
                src_t = torch.from_numpy(src).to(device)
                dst_t = torch.from_numpy(dst).to(device)
                y_t = torch.from_numpy(y).to(device)
                logits = model(x_t, adj_t, src_t, dst_t)
                loss = loss_fn(logits, y_t) / max(len(train_df), 1)
                loss.backward()
                total_loss += float(loss.detach().cpu().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            model.eval()
            val_scores = []
            val_targets = []
            with torch.no_grad():
                for _, x, adj, src, dst, y in val_graphs:
                    x_t = torch.from_numpy(x).to(device)
                    adj_t = torch.from_numpy(adj).to(device)
                    src_t = torch.from_numpy(src).to(device)
                    dst_t = torch.from_numpy(dst).to(device)
                    logits = model(x_t, adj_t, src_t, dst_t)
                    val_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
                    val_targets.append(y)
            val_scores_arr = np.concatenate(val_scores, axis=0).astype(float)
            val_targets_arr = np.concatenate(val_targets, axis=0).astype(int)
            val_ap = safe_ap(val_targets_arr, val_scores_arr)
            if val_ap > best_val_ap + 1e-12:
                best_val_ap = val_ap
                best_epoch = int(epoch + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        def predict(split_df: pd.DataFrame) -> np.ndarray:
            model.eval()
            scores = np.zeros(len(split_df), dtype=float)
            with torch.no_grad():
                for row_idx, x, adj, src, dst, _ in build_split_graphs(split_df):
                    x_t = torch.from_numpy(x).to(device)
                    adj_t = torch.from_numpy(adj).to(device)
                    src_t = torch.from_numpy(src).to(device)
                    dst_t = torch.from_numpy(dst).to(device)
                    logits = model(x_t, adj_t, src_t, dst_t)
                    scores[row_idx] = torch.sigmoid(logits).detach().cpu().numpy()
            return scores.astype(float)

        result = ScoreTriplet([predict(train_df), predict(validation_df), predict(test_df)])
        result.training_metadata = {
            "seed": SEED,
            "best_epoch": int(best_epoch),
            "best_validation_ap": float(best_val_ap),
            "pos_weight": float(pos_weight.detach().cpu().item()),
            "hidden_dim": 128,
            "n_layers": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": patience,
            "max_epochs": max_epochs,
            "early_stopping_metric": "Validation AP",
            "training_split": "Train",
            "message_passing_graph": "W=5 candidate adjacency",
            "gold_edges_used_in_adjacency": False,
        }
        cleanup_model(model)
        return result

    return score_split()


def compute_sentence_encoder_scores_2way(
    train_df,
    validation_df,
    repo_id,
    cache_dir,
    batch_size,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    model = SentenceTransformer(model_path)

    def score(df):
        a = np.asarray(
            model.encode(
                df["anchor_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        b = np.asarray(
            model.encode(
                df["candidate_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        return cosine_scores(a, b)

    result = (score(train_df), score(validation_df))
    frozen = {
        "type": "sentence_transformer_retrieval",
        "model_path": model_path,
        "model_ref": model_path,
        "batch_size": int(batch_size),
    }
    cleanup_model(model)
    return result, frozen


def compute_transformer_retrieval_scores_2way(
    train_df,
    validation_df,
    repo_id,
    cache_dir,
    batch_size,
    max_length,
    use_sentence_transformer: bool = False,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    if use_sentence_transformer:
        model = SentenceTransformer(model_path)

        def score(df):
            a = np.asarray(
                model.encode(
                    df["anchor_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            b = np.asarray(
                model.encode(
                    df["candidate_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            return cosine_scores(a, b)

        result = (score(train_df), score(validation_df))
        frozen = {
            "type": "sentence_transformer_retrieval",
            "model_path": model_path,
            "model_ref": model_path,
            "batch_size": int(batch_size),
        }
        cleanup_model(model)
        return result, frozen

    tokenizer, model = load_auto_encoder(repo_id, cache_dir)

    def score(df):
        a = encode_auto_texts(
            model,
            tokenizer,
            df["anchor_text"].tolist(),
            None,
            batch_size,
            max_length,
        )
        b = encode_auto_texts(
            model,
            tokenizer,
            df["candidate_text"].tolist(),
            None,
            batch_size,
            max_length,
        )
        return cosine_scores(a, b)

    result = (score(train_df), score(validation_df))
    frozen = {
        "type": "autoencoder_retrieval",
        "model_path": model_path,
        "model_ref": model_path,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
    }
    cleanup_model(model, tokenizer)
    return result, frozen


def score_sentence_encoder_single(
    df: pd.DataFrame,
    model_path: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_path)

    def score(frame):
        a = np.asarray(
            model.encode(
                frame["anchor_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        b = np.asarray(
            model.encode(
                frame["candidate_text"].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        return cosine_scores(a, b)

    values = score(df)
    cleanup_model(model)
    return values


def score_transformer_retrieval_single(
    df: pd.DataFrame,
    model_path: str,
    cache_dir: str,
    batch_size: int,
    max_length: int,
    use_sentence_transformer: bool = False,
) -> np.ndarray:
    if use_sentence_transformer:
        model = SentenceTransformer(model_path)

        def score(frame):
            a = np.asarray(
                model.encode(
                    frame["anchor_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            b = np.asarray(
                model.encode(
                    frame["candidate_text"].tolist(),
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            return cosine_scores(a, b)

        values = score(df)
        cleanup_model(model)
        return values

    tokenizer, model = load_auto_encoder(model_path, cache_dir)

    a = encode_auto_texts(
        model,
        tokenizer,
        df["anchor_text"].tolist(),
        None,
        batch_size,
        max_length,
    )
    b = encode_auto_texts(
        model,
        tokenizer,
        df["candidate_text"].tolist(),
        None,
        batch_size,
        max_length,
    )
    values = cosine_scores(a, b)
    cleanup_model(model, tokenizer)
    return values


def fit_pairwise_probe_2way(
    train_df,
    validation_df,
    repo_id,
    cache_dir,
    batch_size,
    max_length,
):
    tokenizer, encoder = load_auto_encoder(repo_id, cache_dir)

    def embed(df):
        return encode_auto_texts(
            encoder,
            tokenizer,
            df["anchor_text"].tolist(),
            df["candidate_text"].tolist(),
            batch_size,
            max_length,
        )

    train_emb = embed(train_df)
    val_emb = embed(validation_df)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    model.fit(train_emb, train_df["edge_label"].to_numpy(dtype=int))

    frozen = {
        "type": "pairwise_probe",
        "model_path": resolve_hf_model(repo_id, cache_dir),
        "model_ref": resolve_hf_model(repo_id, cache_dir),
        "classifier": model,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
    }
    result = (
        model.predict_proba(train_emb)[:, 1].astype(float),
        model.predict_proba(val_emb)[:, 1].astype(float),
    )
    cleanup_model(encoder, tokenizer)
    return result, frozen


def score_pairwise_probe_from_frozen(
    df: pd.DataFrame,
    frozen: dict,
    cache_dir: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer, encoder = load_auto_encoder(frozen["model_path"], cache_dir)
    emb = encode_auto_texts(
        encoder,
        tokenizer,
        df["anchor_text"].tolist(),
        df["candidate_text"].tolist(),
        batch_size,
        max_length,
    )
    scores = frozen["classifier"].predict_proba(emb)[:, 1].astype(float)
    cleanup_model(encoder, tokenizer)
    return scores


def score_finetuned_pairwise_classifier(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Score jointly encoded source--target pairs with an end-to-end classifier."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(df), int(batch_size)):
            batch = df.iloc[start : start + int(batch_size)]
            encoded = tokenizer(
                batch["anchor_text"].tolist(),
                batch["candidate_text"].tolist(),
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            scores.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(scores).astype(float) if scores else np.zeros(0, dtype=float)


def fit_finetuned_financial_pairwise_2way(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    repo_id: str,
    cache_dir: str,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[np.ndarray, dict, List[dict]]:
    """Fine-tune a financial joint pair classifier using Train and Validation only."""
    if epochs < 1:
        raise ValueError("finetuned-pairwise-epochs must be positive.")
    if batch_size < 1:
        raise ValueError("finetuned-pairwise-batch-size must be positive.")

    model_path = resolve_hf_model(repo_id, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model.to(device)

    labels = train_df["edge_label"].to_numpy(dtype=int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("Fine-tuned pairwise training requires both edge classes.")
    class_weights = torch.tensor(
        [1.0, negatives / positives], dtype=torch.float32, device=device
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    checkpoint_path = output_dir / "finetuned_financial_pairwise_best_state.pt"
    history: List[dict] = []
    best = None
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(train_df))
        losses = []
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch = train_df.iloc[idx]
            encoded = tokenizer(
                batch["anchor_text"].tolist(),
                batch["candidate_text"].tolist(),
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            target = torch.as_tensor(
                batch["edge_label"].to_numpy(dtype=np.int64), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded).logits
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        validation_scores = score_finetuned_pairwise_classifier(
            validation_df, model, tokenizer, device, batch_size, max_length
        )
        threshold, validation_metrics = select_threshold_exact(
            validation_df["edge_label"], validation_scores
        )
        row = {
            "Epoch": int(epoch),
            "TrainLoss": float(np.mean(losses)) if losses else float("nan"),
            "Threshold": float(threshold),
            **validation_metrics,
        }
        history.append(row)
        selection_key = (
            float(validation_metrics["F1"]),
            float(validation_metrics["Precision"]),
            -int(validation_metrics["PredEdges"]),
            float(threshold),
        )
        if best is None or selection_key > best["selection_key"]:
            best = {
                "selection_key": selection_key,
                "epoch": int(epoch),
                "threshold": float(threshold),
                "metrics": validation_metrics,
            }
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best is None or best_state is None:
        raise RuntimeError("Fine-tuned pairwise model did not produce a Validation selection.")
    save_torch(checkpoint_path, best_state)
    model.load_state_dict(best_state)
    selected_validation_scores = score_finetuned_pairwise_classifier(
        validation_df, model, tokenizer, device, batch_size, max_length
    )
    frozen = {
        "type": "finetuned_financial_pairwise",
        "model_path": str(model_path),
        "model_ref": repo_id,
        "artifact_path": str(checkpoint_path),
        "selected_epoch": int(best["epoch"]),
        "threshold": float(best["threshold"]),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "class_weight_positive": float(negatives / positives),
    }
    cleanup_model(model, tokenizer)
    return selected_validation_scores, frozen, history


def score_finetuned_pairwise_from_checkpoint(
    df: pd.DataFrame,
    frozen: dict,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(frozen["model_path"])
    model = AutoModelForSequenceClassification.from_pretrained(
        frozen["model_path"], num_labels=2, ignore_mismatched_sizes=True
    )
    model.load_state_dict(load_torch(Path(frozen["artifact_path"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scores = score_finetuned_pairwise_classifier(
        df,
        model,
        tokenizer,
        device,
        int(frozen["batch_size"]),
        int(frozen["max_length"]),
    )
    cleanup_model(model, tokenizer)
    return scores


def fit_score_fusion_2way(
    train_df,
    validation_df,
    bm25_scores,
    semantic_scores,
    pairwise_scores,
):
    def inv_distance(df):
        d = np.clip(
            pd.to_numeric(df["distance_num"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float),
            1.0,
            None,
        )
        return 1.0 / d

    train_x = np.column_stack(
        [
            bm25_scores[0],
            semantic_scores[0],
            pairwise_scores[0],
            inv_distance(train_df),
        ]
    )
    val_x = np.column_stack(
        [
            bm25_scores[1],
            semantic_scores[1],
            pairwise_scores[1],
            inv_distance(validation_df),
        ]
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )
    model.fit(train_x, train_df["edge_label"].to_numpy(dtype=int))
    frozen = {
        "type": "score_fusion",
        "model": model,
        "feature_names": ["BM25", "Semantic", "Pairwise", "InverseDistance"],
    }
    return (
        model.predict_proba(train_x)[:, 1].astype(float),
        model.predict_proba(val_x)[:, 1].astype(float),
    ), frozen


def score_fusion_from_frozen(
    df: pd.DataFrame,
    frozen: dict,
    bm25_scores: np.ndarray,
    semantic_scores: np.ndarray,
    pairwise_scores: np.ndarray,
) -> np.ndarray:
    d = np.clip(
        pd.to_numeric(df["distance_num"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float),
        1.0,
        None,
    )
    x = np.column_stack(
        [
            bm25_scores,
            semantic_scores,
            pairwise_scores,
            1.0 / d,
        ]
    )
    return frozen["model"].predict_proba(x)[:, 1].astype(float)


def _graphsage_node_map(df: pd.DataFrame) -> Dict[Tuple[str, int, str], str]:
    mapping: Dict[Tuple[str, int, str], str] = {}
    for row in df.itertuples(index=False):
        sid = str(getattr(row, "session_id"))
        q1r = int(getattr(row, "Q1_row"))
        q2r = int(getattr(row, "Q2_row"))
        mapping[(sid, q1r, "source")] = (
            clean_text(getattr(row, "Q1", "")) + " " + clean_text(getattr(row, "A1", ""))
        )
        mapping[(sid, q2r, "target")] = clean_text(getattr(row, "Q2", ""))
    return mapping


def _build_dirsage_graphs(
    split_df: pd.DataFrame,
    node_embs: Dict[Tuple[str, int, str], np.ndarray],
    include_labels: bool,
):
    """Build directed candidate graphs without using dependency labels as edges."""
    work_df = split_df.copy()
    work_df["_dirsage_row_pos"] = np.arange(len(work_df), dtype=np.int64)
    graphs = []
    for _, g in work_df.groupby("session_id", sort=False):
        sid = str(g["session_id"].iloc[0])
        node_keys = []
        for row in g.itertuples(index=False):
            q1r = int(getattr(row, "Q1_row"))
            q2r = int(getattr(row, "Q2_row"))
            node_keys.append((sid, q1r, "source"))
            node_keys.append((sid, q2r, "target"))
        node_keys = sorted(set(node_keys), key=lambda x: (x[1], x[2]))
        idx_map = {k: i for i, k in enumerate(node_keys)}
        x = np.vstack([node_embs[k] for k in node_keys]).astype(np.float32)
        n = len(node_keys)
        in_adj = np.zeros((n, n), dtype=np.float32)
        out_adj = np.zeros((n, n), dtype=np.float32)
        for row in g.itertuples(index=False):
            source = idx_map[(sid, int(getattr(row, "Q1_row")), "source")]
            target = idx_map[(sid, int(getattr(row, "Q2_row")), "target")]
            in_adj[target, source] = 1.0
            out_adj[source, target] = 1.0
        in_adj /= np.clip(in_adj.sum(axis=1, keepdims=True), 1.0, None)
        out_adj /= np.clip(out_adj.sum(axis=1, keepdims=True), 1.0, None)
        edge_src = np.asarray(
            [idx_map[(sid, int(x), "source")] for x in g["Q1_row"].tolist()],
            dtype=np.int64,
        )
        edge_dst = np.asarray(
            [idx_map[(sid, int(x), "target")] for x in g["Q2_row"].tolist()],
            dtype=np.int64,
        )
        record = [g["_dirsage_row_pos"].to_numpy(dtype=int), x, in_adj, out_adj, edge_src, edge_dst]
        if include_labels:
            record.append(g["edge_label"].to_numpy(dtype=np.float32))
        graphs.append(tuple(record))
    return graphs


def fit_graphsage_2way(
    train_df,
    validation_df,
    repo_id,
    cache_dir,
    batch_size,
):
    model_path = resolve_hf_model(repo_id, cache_dir)
    encoder = SentenceTransformer(model_path)
    all_df = pd.concat([train_df, validation_df], ignore_index=True)
    node_map = _graphsage_node_map(all_df)
    node_keys = list(node_map.keys())
    node_texts = [node_map[k] for k in node_keys]
    node_emb = np.asarray(
        encoder.encode(
            node_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    node_embs = {k: node_emb[i] for i, k in enumerate(node_keys)}
    cleanup_model(encoder)

    def build_split_graphs(split_df: pd.DataFrame):
        graphs = []
        for _, g in split_df.groupby("session_id", sort=False):
            sid = str(g["session_id"].iloc[0])
            node_keys = []
            for row in g.itertuples(index=False):
                q1r = int(getattr(row, "Q1_row"))
                q2r = int(getattr(row, "Q2_row"))
                node_keys.append((sid, q1r, "source"))
                node_keys.append((sid, q2r, "target"))
            node_keys = sorted(set(node_keys), key=lambda x: (x[1], x[2]))
            idx_map = {k: i for i, k in enumerate(node_keys)}
            x = np.vstack([node_embs[k] for k in node_keys]).astype(np.float32)
            n = len(node_keys)
            adj = np.zeros((n, n), dtype=np.float32)
            np.fill_diagonal(adj, 1.0)
            for row in g.itertuples(index=False):
                i = idx_map[(sid, int(getattr(row, "Q1_row")), "source")]
                j = idx_map[(sid, int(getattr(row, "Q2_row")), "target")]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
            deg = adj.sum(axis=1, keepdims=True)
            adj = adj / np.clip(deg, 1.0, None)
            edge_src = np.asarray(
                [idx_map[(sid, int(x), "source")] for x in g["Q1_row"].tolist()],
                dtype=np.int64,
            )
            edge_dst = np.asarray(
                [idx_map[(sid, int(x), "target")] for x in g["Q2_row"].tolist()],
                dtype=np.int64,
            )
            y = g["edge_label"].to_numpy(dtype=np.float32)
            graphs.append((g.index.to_numpy(dtype=int), x, adj, edge_src, edge_dst, y))
        return graphs

    train_graphs = build_split_graphs(train_df)
    val_graphs = build_split_graphs(validation_df)
    in_dim = int(train_graphs[0][1].shape[1]) if train_graphs else int(node_emb.shape[1])
    model = GraphSAGEEdgeModel(in_dim=in_dim, hidden_dim=128, n_layers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_labels = train_df["edge_label"].to_numpy(dtype=np.float32)
    pos = float(np.clip(train_labels.sum(), 1.0, None))
    neg = float(len(train_labels) - train_labels.sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    rng = np.random.default_rng(SEED)
    best_state = None
    best_val_ap = -1.0
    best_epoch = -1
    patience = 5
    bad_epochs = 0
    max_epochs = 50
    for epoch in range(max_epochs):
        model.train()
        order = np.arange(len(train_graphs))
        rng.shuffle(order)
        opt.zero_grad()
        for idx in order:
            _, x, adj, src, dst, y = train_graphs[idx]
            x_t = torch.from_numpy(x).to(device)
            adj_t = torch.from_numpy(adj).to(device)
            src_t = torch.from_numpy(src).to(device)
            dst_t = torch.from_numpy(dst).to(device)
            y_t = torch.from_numpy(y).to(device)
            logits = model(x_t, adj_t, src_t, dst_t)
            loss = loss_fn(logits, y_t) / max(len(train_df), 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        val_scores = []
        val_targets = []
        with torch.no_grad():
            for _, x, adj, src, dst, y in val_graphs:
                x_t = torch.from_numpy(x).to(device)
                adj_t = torch.from_numpy(adj).to(device)
                src_t = torch.from_numpy(src).to(device)
                dst_t = torch.from_numpy(dst).to(device)
                logits = model(x_t, adj_t, src_t, dst_t)
                val_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
                val_targets.append(y)
        val_scores_arr = np.concatenate(val_scores, axis=0).astype(float)
        val_targets_arr = np.concatenate(val_targets, axis=0).astype(int)
        val_ap = safe_ap(val_targets_arr, val_scores_arr)
        if val_ap > best_val_ap + 1e-12:
            best_val_ap = val_ap
            best_epoch = int(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(split_df: pd.DataFrame) -> np.ndarray:
        model.eval()
        scores = np.zeros(len(split_df), dtype=float)
        with torch.no_grad():
            for row_idx, x, adj, src, dst, _ in build_split_graphs(split_df):
                x_t = torch.from_numpy(x).to(device)
                adj_t = torch.from_numpy(adj).to(device)
                src_t = torch.from_numpy(src).to(device)
                dst_t = torch.from_numpy(dst).to(device)
                logits = model(x_t, adj_t, src_t, dst_t)
                scores[row_idx] = torch.sigmoid(logits).detach().cpu().numpy()
        return scores.astype(float)

    frozen = {
        "type": "graphsage",
        "model_path": model_path,
        "model_ref": model_path,
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_validation_ap": float(best_val_ap),
        "hidden_dim": 128,
        "n_layers": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": patience,
        "max_epochs": max_epochs,
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "message_passing_graph": "W=5 candidate adjacency",
    }
    result = (predict(train_df), predict(validation_df))
    cleanup_model(model)
    return result, frozen


def score_graphsage_from_frozen(
    split_df: pd.DataFrame,
    frozen: dict,
    batch_size: int,
) -> np.ndarray:
    model_path = frozen["model_path"]
    encoder = SentenceTransformer(model_path)
    node_map = _graphsage_node_map(split_df)
    node_keys = list(node_map.keys())
    node_texts = [node_map[k] for k in node_keys]
    node_emb = np.asarray(
        encoder.encode(
            node_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    cleanup_model(encoder)
    node_embs = {k: node_emb[i] for i, k in enumerate(node_keys)}

    graphs = []
    for _, g in split_df.groupby("session_id", sort=False):
        sid = str(g["session_id"].iloc[0])
        node_keys = []
        for row in g.itertuples(index=False):
            q1r = int(getattr(row, "Q1_row"))
            q2r = int(getattr(row, "Q2_row"))
            node_keys.append((sid, q1r, "source"))
            node_keys.append((sid, q2r, "target"))
        node_keys = sorted(set(node_keys), key=lambda x: (x[1], x[2]))
        idx_map = {k: i for i, k in enumerate(node_keys)}
        x = np.vstack([node_embs[k] for k in node_keys]).astype(np.float32)
        n = len(node_keys)
        adj = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(adj, 1.0)
        for row in g.itertuples(index=False):
            i = idx_map[(sid, int(getattr(row, "Q1_row")), "source")]
            j = idx_map[(sid, int(getattr(row, "Q2_row")), "target")]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        deg = adj.sum(axis=1, keepdims=True)
        adj = adj / np.clip(deg, 1.0, None)
        edge_src = np.asarray(
            [idx_map[(sid, int(x), "source")] for x in g["Q1_row"].tolist()],
            dtype=np.int64,
        )
        edge_dst = np.asarray(
            [idx_map[(sid, int(x), "target")] for x in g["Q2_row"].tolist()],
            dtype=np.int64,
        )
        graphs.append((g.index.to_numpy(dtype=int), x, adj, edge_src, edge_dst))

    in_dim = int(graphs[0][1].shape[1]) if graphs else 0
    model = GraphSAGEEdgeModel(in_dim=in_dim, hidden_dim=frozen["hidden_dim"], n_layers=frozen["n_layers"])
    model.load_state_dict(frozen["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores = np.zeros(len(split_df), dtype=float)
    with torch.no_grad():
        for row_idx, x, adj, src, dst in graphs:
            x_t = torch.from_numpy(x).to(device)
            adj_t = torch.from_numpy(adj).to(device)
            src_t = torch.from_numpy(src).to(device)
            dst_t = torch.from_numpy(dst).to(device)
            logits = model(x_t, adj_t, src_t, dst_t)
            scores[row_idx] = torch.sigmoid(logits).detach().cpu().numpy()
    cleanup_model(model)
    return scores.astype(float)


def fit_dirsage_2way(
    train_df,
    validation_df,
    repo_id,
    cache_dir,
    batch_size,
):
    """Fit Dir-SAGE under the same protocol as the GraphSAGE baseline."""
    model_path = resolve_hf_model(repo_id, cache_dir)
    encoder = SentenceTransformer(model_path)
    all_df = pd.concat([train_df, validation_df], ignore_index=True)
    node_map = _graphsage_node_map(all_df)
    node_keys = list(node_map.keys())
    node_texts = [node_map[k] for k in node_keys]
    node_emb = np.asarray(
        encoder.encode(
            node_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    node_embs = {k: node_emb[i] for i, k in enumerate(node_keys)}
    cleanup_model(encoder)

    train_graphs = _build_dirsage_graphs(train_df, node_embs, include_labels=True)
    val_graphs = _build_dirsage_graphs(validation_df, node_embs, include_labels=True)
    in_dim = int(train_graphs[0][1].shape[1]) if train_graphs else int(node_emb.shape[1])
    model = DirSAGEEdgeModel(in_dim=in_dim, hidden_dim=128, n_layers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_labels = train_df["edge_label"].to_numpy(dtype=np.float32)
    pos = float(np.clip(train_labels.sum(), 1.0, None))
    neg = float(len(train_labels) - train_labels.sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="sum")

    rng = np.random.default_rng(SEED)
    best_state = None
    best_val_ap = -1.0
    best_epoch = -1
    patience = 5
    bad_epochs = 0
    max_epochs = 50
    for epoch in range(max_epochs):
        model.train()
        order = np.arange(len(train_graphs))
        rng.shuffle(order)
        opt.zero_grad()
        for idx in order:
            _, x, in_adj, out_adj, src, dst, y = train_graphs[idx]
            logits = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(in_adj).to(device),
                torch.from_numpy(out_adj).to(device),
                torch.from_numpy(src).to(device),
                torch.from_numpy(dst).to(device),
            )
            loss = loss_fn(logits, torch.from_numpy(y).to(device)) / max(len(train_df), 1)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        val_scores = []
        val_targets = []
        with torch.no_grad():
            for _, x, in_adj, out_adj, src, dst, y in val_graphs:
                logits = model(
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(in_adj).to(device),
                    torch.from_numpy(out_adj).to(device),
                    torch.from_numpy(src).to(device),
                    torch.from_numpy(dst).to(device),
                )
                val_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
                val_targets.append(y)
        val_scores_arr = np.concatenate(val_scores, axis=0).astype(float)
        val_targets_arr = np.concatenate(val_targets, axis=0).astype(int)
        val_ap = safe_ap(val_targets_arr, val_scores_arr)
        if val_ap > best_val_ap + 1e-12:
            best_val_ap = val_ap
            best_epoch = int(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(split_df: pd.DataFrame) -> np.ndarray:
        model.eval()
        scores = np.zeros(len(split_df), dtype=float)
        with torch.no_grad():
            for row_idx, x, in_adj, out_adj, src, dst, _ in _build_dirsage_graphs(
                split_df, node_embs, include_labels=True
            ):
                logits = model(
                    torch.from_numpy(x).to(device),
                    torch.from_numpy(in_adj).to(device),
                    torch.from_numpy(out_adj).to(device),
                    torch.from_numpy(src).to(device),
                    torch.from_numpy(dst).to(device),
                )
                scores[row_idx] = torch.sigmoid(logits).detach().cpu().numpy()
        return scores.astype(float)

    frozen = {
        "type": "dirsage",
        "model_path": model_path,
        "model_ref": model_path,
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_validation_ap": float(best_val_ap),
        "hidden_dim": 128,
        "n_layers": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": patience,
        "max_epochs": max_epochs,
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "message_passing_graph": "directed W=5 candidate adjacency",
    }
    result = (predict(train_df), predict(validation_df))
    cleanup_model(model)
    return result, frozen


def score_dirsage_from_frozen(
    split_df: pd.DataFrame,
    frozen: dict,
    batch_size: int,
) -> np.ndarray:
    model_path = frozen["model_path"]
    encoder = SentenceTransformer(model_path)
    node_map = _graphsage_node_map(split_df)
    node_keys = list(node_map.keys())
    node_texts = [node_map[k] for k in node_keys]
    node_emb = np.asarray(
        encoder.encode(
            node_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    cleanup_model(encoder)
    node_embs = {k: node_emb[i] for i, k in enumerate(node_keys)}
    graphs = _build_dirsage_graphs(split_df, node_embs, include_labels=False)

    in_dim = int(graphs[0][1].shape[1]) if graphs else 0
    model = DirSAGEEdgeModel(
        in_dim=in_dim,
        hidden_dim=frozen["hidden_dim"],
        n_layers=frozen["n_layers"],
    )
    model.load_state_dict(frozen["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores = np.zeros(len(split_df), dtype=float)
    with torch.no_grad():
        for row_idx, x, in_adj, out_adj, src, dst in graphs:
            logits = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(in_adj).to(device),
                torch.from_numpy(out_adj).to(device),
                torch.from_numpy(src).to(device),
                torch.from_numpy(dst).to(device),
            )
            scores[row_idx] = torch.sigmoid(logits).detach().cpu().numpy()
    cleanup_model(model)
    return scores.astype(float)


# =============================================================================
# Cache
# =============================================================================

def slugify(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name.lower())
    return re.sub(r"_+", "_", s).strip("_") or "baseline"


def cache_signature(name: str, cache_meta: dict) -> str:
    payload = {"name": name, **cache_meta}
    return json_sha256(payload)[:24]


def cache_file(outdir: Path, name: str, cache_meta: dict) -> Path:
    suffix = "_v4" if name == "GraphSAGE" else ""
    sig = cache_signature(name, cache_meta)
    return outdir / "cache" / f"{slugify(name)}{suffix}_{sig}_scores.npz"


def save_score_cache(path: Path, scores, cache_meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_scores=np.asarray(scores[0], dtype=float),
        validation_scores=np.asarray(scores[1], dtype=float),
        test_scores=np.asarray(scores[2], dtype=float),
        cache_meta_json=np.asarray(json.dumps(cache_meta, ensure_ascii=False, sort_keys=True)),
    )


def load_score_cache(path: Path, lengths, expected_meta: Optional[dict] = None):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    scores = (
        data["train_scores"].astype(float),
        data["validation_scores"].astype(float),
        data["test_scores"].astype(float),
    )
    if tuple(len(x) for x in scores) != tuple(lengths):
        return None
    if expected_meta is not None:
        if "cache_meta_json" not in data.files:
            return None
        stored = json.loads(str(data["cache_meta_json"].item()))
        if stored != expected_meta:
            return None
    return scores


# =============================================================================
# Session bootstrap
# =============================================================================

def session_bootstrap_ci(
    df: pd.DataFrame,
    score,
    pred,
    n_samples: int,
    seed: int,
):
    if n_samples <= 0:
        return []
    work = df[["session_id", "edge_label"]].copy()
    work["_score"] = np.asarray(score, dtype=float)
    work["_pred"] = np.asarray(pred, dtype=int)
    groups = [
        g.reset_index(drop=True)
        for _, g in work.groupby("session_id", sort=False)
    ]
    rng = np.random.default_rng(seed)
    draws = {
        "AUC": [],
        "AP": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
    }
    for _ in range(n_samples):
        boot = pd.concat(
            [
                groups[int(i)]
                for i in rng.integers(0, len(groups), size=len(groups))
            ],
            ignore_index=True,
        )
        m = binary_metrics(
            boot["edge_label"],
            boot["_pred"],
            boot["_score"],
        )
        for key in draws:
            draws[key].append(m[key])

    rows = []
    for metric, values in draws.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        rows.append(
            {
                "Metric": metric,
                "BootstrapMean": float(arr.mean()),
                "CI_Lower": float(np.quantile(arr, 0.025)),
                "CI_Upper": float(np.quantile(arr, 0.975)),
                "BootstrapSamples": n_samples,
            }
        )
    return rows


def paired_f1_bootstrap(
    df,
    fcn_pred,
    baseline_pred,
    n_samples,
    seed,
):
    if n_samples <= 0:
        return {}
    work = df[["session_id", "edge_label"]].copy()
    work["_fcn"] = np.asarray(fcn_pred, dtype=int)
    work["_base"] = np.asarray(baseline_pred, dtype=int)
    groups = [
        g.reset_index(drop=True)
        for _, g in work.groupby("session_id", sort=False)
    ]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_samples):
        boot = pd.concat(
            [
                groups[int(i)]
                for i in rng.integers(0, len(groups), size=len(groups))
            ],
            ignore_index=True,
        )
        y = boot["edge_label"].to_numpy(dtype=int)
        diffs.append(
            f1_score(y, boot["_fcn"], zero_division=0)
            - f1_score(y, boot["_base"], zero_division=0)
        )
    arr = np.asarray(diffs, dtype=float)
    return {
        "F1_Difference_FCN_minus_Baseline_Mean": float(arr.mean()),
        "CI_Lower": float(np.quantile(arr, 0.025)),
        "CI_Upper": float(np.quantile(arr, 0.975)),
        "P_Diff_LE_0": float(np.mean(arr <= 0.0)),
        "BootstrapSamples": n_samples,
    }


# =============================================================================
# CLI and main
# =============================================================================

def parse_topk_values(value: str) -> Tuple[int, ...]:
    vals = sorted(
        {
            int(x.strip())
            for x in str(value).split(",")
            if x.strip()
        }
    )
    if not vals or min(vals) < 1:
        raise ValueError("Top-K values must be positive integers.")
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run core FCN baselines under the final three-way protocol."
    )
    p.add_argument("--stage1-dir", default=str(DEFAULT_STAGE1_DIR))
    p.add_argument("--stage2-dir", default=str(DEFAULT_STAGE2_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--hf-cache-dir", default=str(DEFAULT_HF_CACHE_DIR))

    p.add_argument(
        "--sentence-encoder-model",
        default=DEFAULT_MODELS["sentence_encoder"],
    )
    p.add_argument(
        "--retrieval-encoder-model",
        default=DEFAULT_MODELS["retrieval_encoder"],
    )
    p.add_argument(
        "--financial-encoder-model",
        default=DEFAULT_MODELS["financial_encoder"],
    )
    p.add_argument(
        "--pairwise-cn-model",
        default=DEFAULT_MODELS["pairwise_cn"],
    )
    p.add_argument(
        "--pairwise-fin-model",
        default=DEFAULT_MODELS["pairwise_fin"],
    )
    p.add_argument(
        "--graphsage-model",
        default=DEFAULT_MODELS["graphsage"],
    )

    p.add_argument("--sentence-batch-size", type=int, default=64)
    p.add_argument("--encoder-batch-size", type=int, default=32)
    p.add_argument("--pairwise-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--topk-values", default="1,2,3,4,5")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--dirsage-only",
        action="store_true",
        help="Run only the Dir-SAGE baseline and write standalone result files.",
    )
    p.add_argument(
        "--finetuned-financial-pairwise-only",
        action="store_true",
        help=(
            "Run only the end-to-end Fine-tuned Financial Pairwise baseline "
            "and write standalone result files."
        ),
    )
    p.add_argument("--finetuned-pairwise-epochs", type=int, default=3)
    p.add_argument("--finetuned-pairwise-learning-rate", type=float, default=2e-5)
    p.add_argument("--finetuned-pairwise-weight-decay", type=float, default=0.01)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--skip-failed-baselines", action="store_true")
    p.add_argument(
        "--allow-custom-session-counts",
        action="store_true",
        help="Allow custom session counts for a supplied Stage 1 split.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(stage1_dir / "stage1_train_scored.csv")
    val_df = load_split(stage1_dir / "stage1_validation_scored.csv")
    test_df = load_split(stage1_dir / "stage1_test_scored.csv")
    stage1_manifest_path = stage1_dir / "stage1_split_manifest.csv"
    stage2_manifest_path = stage2_dir / "stage2_selected_params.json"
    stage1_params = verify_stage1_artifacts(
        stage1_dir,
        train_df,
        val_df,
        test_df,
        stage1_manifest_path,
    )
    stage2_params = verify_stage2_artifacts(
        stage2_dir,
        stage1_params,
        val_df,
        test_df,
    )
    verify_three_way_protocol(
        train_df,
        val_df,
        test_df,
        require_fixed_counts=not args.allow_custom_session_counts,
    )
    if not args.allow_custom_session_counts:
        LOGGER.info("Verified fixed 176/44/55 split and matching Stage 1/2 manifests.")

    LOGGER.info(
        "Train %d/%d sessions | Validation %d/%d | Test %d/%d",
        len(train_df),
        train_df["session_id"].nunique(),
        len(val_df),
        val_df["session_id"].nunique(),
        len(test_df),
        test_df["session_id"].nunique(),
    )

    lengths = (len(train_df), len(val_df), len(test_df))
    sentence_model_fingerprint = model_fingerprint(
        args.sentence_encoder_model,
        args.hf_cache_dir,
    )
    retrieval_model_fingerprint = model_fingerprint(
        args.retrieval_encoder_model,
        args.hf_cache_dir,
    )
    financial_model_fingerprint = model_fingerprint(
        args.financial_encoder_model,
        args.hf_cache_dir,
    )
    pairwise_cn_fingerprint = model_fingerprint(
        args.pairwise_cn_model,
        args.hf_cache_dir,
    )
    pairwise_fin_fingerprint = model_fingerprint(
        args.pairwise_fin_model,
        args.hf_cache_dir,
    )
    graphsage_model_fingerprint = model_fingerprint(
        args.graphsage_model,
        args.hf_cache_dir,
    )
    shared_cache_meta = {
        "protocol": "fixed_176_44_55_train_validation_test",
        "text_protocol": "source_conditioned_anchor_candidate_v1",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "validation_sessions": int(val_df["session_id"].nunique()),
        "test_sessions": int(test_df["session_id"].nunique()),
        "train_true_edges": int(train_df["edge_label"].sum()),
        "validation_true_edges": int(val_df["edge_label"].sum()),
        "test_true_edges": int(test_df["edge_label"].sum()),
        "sentence_encoder_model": sentence_model_fingerprint,
        "retrieval_encoder_model": retrieval_model_fingerprint,
        "financial_encoder_model": financial_model_fingerprint,
        "pairwise_cn_model": pairwise_cn_fingerprint,
        "pairwise_fin_model": pairwise_fin_fingerprint,
        "graphsage_model": graphsage_model_fingerprint,
        "sentence_encoder_pooling": "sentence_transformer_normalized",
        "retrieval_encoder_pooling": "sentence_transformer_normalized",
        "financial_encoder_pooling": "auto_encoder_pooler_or_mean",
        "pairwise_text_protocol": "joint_anchor_candidate_encoding",
        "graphsage_graph_protocol": "candidate_adjacency_w5",
        "sentence_batch_size": args.sentence_batch_size,
        "encoder_batch_size": args.encoder_batch_size,
        "pairwise_batch_size": args.pairwise_batch_size,
        "max_length": args.max_length,
        "topk_values": parse_topk_values(args.topk_values),
        "seed": SEED,
    }
    score_bank = {}
    val_rows = []
    test_rows = []
    params = []
    pred_bank = {}
    score_test_bank = {}
    topk_search = []

    def cached(name, fn, cache_meta):
        path = cache_file(outdir, name, cache_meta)
        scores = None if args.no_cache else load_score_cache(
            path,
            lengths,
            expected_meta=cache_meta,
        )
        if scores is not None:
            LOGGER.info("Loaded cache: %s", name)
            return scores
        LOGGER.info("Running: %s", name)
        scores = fn()
        save_score_cache(path, scores, cache_meta)
        if name == "GraphSAGE":
            meta = getattr(scores, "training_metadata", None)
            if meta is not None:
                (outdir / "graphsage_training_metadata.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return scores

    bm25 = lambda: (
        lambda scorer: (
            score_bm25(train_df, scorer),
            score_bm25(val_df, scorer),
            score_bm25(test_df, scorer),
        )
    )(fit_bm25(train_df))

    jobs = [
        (
            "BM25",
            "Surface lexical-overlap baseline.",
            lambda: cached(
                "BM25",
                bm25,
                {
                    **shared_cache_meta,
                    "baseline": "BM25",
                    "tokenizer": "char_bigram_bm25_v1",
                    "dedupe_docs": "unique_anchor_text",
                },
            ),
        ),
        (
            "Temporal-distance Prior",
            "Inverse-distance temporal prior with Validation-selected threshold.",
            lambda: (
                compute_distance_prior_scores(train_df),
                compute_distance_prior_scores(val_df),
                compute_distance_prior_scores(test_df),
            ),
        ),
        (
            "General Chinese Sentence Encoder Retrieval",
            "Off-the-shelf Chinese sentence-encoder cosine retrieval.",
            lambda: cached(
                "General Chinese Sentence Encoder Retrieval",
                lambda: compute_sentence_encoder_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.sentence_encoder_model,
                    args.hf_cache_dir,
                    args.sentence_batch_size,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "General Chinese Sentence Encoder Retrieval",
                    "use_sentence_transformer": True,
                },
            ),
        ),
        (
            "General Chinese Retrieval Encoder Retrieval",
            "General Chinese transformer retrieval by cosine similarity.",
            lambda: cached(
                "General Chinese Retrieval Encoder Retrieval",
                lambda: compute_transformer_retrieval_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.retrieval_encoder_model,
                    args.hf_cache_dir,
                    args.encoder_batch_size,
                    args.max_length,
                    use_sentence_transformer=True,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "General Chinese Retrieval Encoder Retrieval",
                    "use_sentence_transformer": True,
                },
            ),
        ),
        (
            "Financial-domain Encoder Retrieval",
            "Financial-domain transformer retrieval by cosine similarity.",
            lambda: cached(
                "Financial-domain Encoder Retrieval",
                lambda: compute_transformer_retrieval_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.financial_encoder_model,
                    args.hf_cache_dir,
                    args.encoder_batch_size,
                    args.max_length,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "Financial-domain Encoder Retrieval",
                    "use_sentence_transformer": False,
                },
            ),
        ),
        (
            "General Chinese Pairwise Transformer + Linear Probe",
            "Frozen Transformer pair encoding with a linear probe fitted on Train.",
            lambda: cached(
                "General Chinese Pairwise Transformer + Linear Probe",
                lambda: compute_pairwise_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.pairwise_cn_model,
                    args.hf_cache_dir,
                    args.pairwise_batch_size,
                    args.max_length,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "General Chinese Pairwise Transformer + Linear Probe",
                    "fit_split": "Train",
                },
            ),
        ),
        (
            "Financial-domain Pairwise Transformer + Linear Probe",
            "Frozen financial-domain Transformer pair encoding with a linear probe fitted on Train.",
            lambda: cached(
                "Financial-domain Pairwise Transformer + Linear Probe",
                lambda: compute_pairwise_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.pairwise_fin_model,
                    args.hf_cache_dir,
                    args.pairwise_batch_size,
                    args.max_length,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "Financial-domain Pairwise Transformer + Linear Probe",
                    "fit_split": "Train",
                },
            ),
        ),
        (
            "GraphSAGE",
            "Candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
            lambda: cached(
                "GraphSAGE",
                lambda: compute_graphsage_scores_3way(
                    train_df,
                    val_df,
                    test_df,
                    args.graphsage_model,
                    args.hf_cache_dir,
                    args.encoder_batch_size,
                    args.max_length,
                ),
                {
                    **shared_cache_meta,
                    "baseline": "GraphSAGE",
                    "fit_split": "Train",
                    "message_passing_graph": "W=5 candidate adjacency",
                },
            ),
        ),
    ]

    for name, purpose, fn in jobs:
        try:
            scores = fn()
        except Exception:
            if args.skip_failed_baselines:
                LOGGER.exception("Skipping %s", name)
                continue
            raise

        score_bank[name] = scores
        vrow, trow, tpred = evaluate_threshold_model(
            name,
            purpose,
            val_df,
            test_df,
            scores[1],
            scores[2],
        )
        val_rows.append(vrow)
        test_rows.append(trow)
        params.append(
            {
                "Model": name,
                "SelectionSplit": "Validation",
                "SelectionCriterion": "max F1",
                "Threshold": trow["Threshold"],
                "TopK": np.nan,
            }
        )
        pred_bank[name] = tpred
        score_test_bank[name] = scores[2]

    # Score Fusion: same inputs as the prior script.
    deps = (
        "BM25",
        "General Chinese Retrieval Encoder Retrieval",
        "General Chinese Pairwise Transformer + Linear Probe",
    )
    if all(x in score_bank for x in deps):
        scores = compute_score_fusion_scores_3way(
            train_df,
            val_df,
            test_df,
            score_bank["BM25"],
            score_bank["General Chinese Retrieval Encoder Retrieval"],
            score_bank["General Chinese Pairwise Transformer + Linear Probe"],
        )
        score_bank["Score Fusion"] = scores
        vrow, trow, tpred = evaluate_threshold_model(
            "Score Fusion",
            "Train-only fusion of BM25, retrieval, pairwise, and distance.",
            val_df,
            test_df,
            scores[1],
            scores[2],
        )
        val_rows.append(vrow)
        test_rows.append(trow)
        params.append(
            {
                "Model": "Score Fusion",
                "SelectionSplit": "Validation",
                "SelectionCriterion": "max F1",
                "Threshold": trow["Threshold"],
                "TopK": np.nan,
            }
        )
        pred_bank["Score Fusion"] = tpred
        score_test_bank["Score Fusion"] = scores[2]

    # Pairwise Transformer + top-K: jointly select K and theta on Validation.
    if "General Chinese Pairwise Transformer + Linear Probe" in score_bank:
        scores = score_bank["General Chinese Pairwise Transformer + Linear Probe"]
        vrow, trow, tpred, search = evaluate_topk_model(
            "Pairwise Transformer + top-K pruning",
            "Frozen Transformer pair encoding with Validation-selected Top-K pruning.",
            val_df,
            test_df,
            scores[1],
            scores[2],
            parse_topk_values(args.topk_values),
        )
        val_rows.append(vrow)
        test_rows.append(trow)
        params.append(
            {
                "Model": "Pairwise Transformer + top-K pruning",
                "SelectionSplit": "Validation",
                "SelectionCriterion": "max F1",
                "Threshold": trow["Threshold"],
                "TopK": trow["TopK"],
            }
        )
        pred_bank["Pairwise Transformer + top-K pruning"] = tpred
        score_test_bank["Pairwise Transformer + top-K pruning"] = scores[2]
        search.insert(0, "Model", "Pairwise Transformer + top-K pruning")
        topk_search.append(search)

    # Final FCN: frozen Stage-2 output only.
    val_graph = align_graph_to_reference(
        val_df,
        load_split(stage2_dir / "stage2_validation_final_graph.csv"),
    )
    test_graph = align_graph_to_reference(
        test_df,
        load_split(stage2_dir / "stage2_test_final_graph.csv"),
    )
    for frame in (val_graph, test_graph):
        if "final_graph_score" not in frame.columns:
            raise RuntimeError("Missing final_graph_score.")
        if "final_graph_pred_edge" not in frame.columns:
            raise RuntimeError("Missing final_graph_pred_edge.")

    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Final FCN",
            "Purpose": "Frozen revised FCN with exact Stage 2B.",
            "Threshold": np.nan,
            "TopK": np.nan,
            **binary_metrics(
                val_df["edge_label"],
                val_graph["final_graph_pred_edge"],
                val_graph["final_graph_score"],
            ),
        }
    )
    test_rows.append(
        {
            "Split": "Test",
            "Model": "Final FCN",
            "Purpose": "Frozen revised FCN with exact Stage 2B.",
            "Threshold": np.nan,
            "TopK": np.nan,
            **binary_metrics(
                test_df["edge_label"],
                test_graph["final_graph_pred_edge"],
                test_graph["final_graph_score"],
            ),
        }
    )
    params.append(
        {
            "Model": "Final FCN",
            "SelectionSplit": "Validation inside FCN Stage 2",
            "SelectionCriterion": "frozen",
            "Threshold": np.nan,
            "TopK": np.nan,
        }
    )
    pred_bank["Final FCN"] = test_graph["final_graph_pred_edge"].to_numpy(
        dtype=int
    )
    score_test_bank["Final FCN"] = test_graph["final_graph_score"].to_numpy(
        dtype=float
    )

    order = {name: i for i, name in enumerate(CORE_BASELINE_ORDER)}
    val_table = pd.DataFrame(val_rows)
    test_table = pd.DataFrame(test_rows)
    for table in (val_table, test_table):
        table["_order"] = table["Model"].map(order)
        table.sort_values("_order", inplace=True, kind="mergesort")
        table.drop(columns="_order", inplace=True)
        table.reset_index(drop=True, inplace=True)

    val_table.to_csv(
        outdir / "followup_edge_core_baselines_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_table.to_csv(
        outdir / "followup_edge_core_baselines_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(params).to_csv(
        outdir / "followup_edge_core_baselines_selected_params.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if topk_search:
        pd.concat(topk_search, ignore_index=True).to_csv(
            outdir / "pairwise_transformer_topk_validation_search.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Test prediction audit.
    key_cols = [
        c
        for c in ["edge_id", "session_id", "Q1_row", "Q2_row", "edge_label"]
        if c in test_df.columns
    ]
    audit = test_df[key_cols].copy()
    for name in CORE_BASELINE_ORDER:
        if name not in pred_bank:
            continue
        slug = slugify(name)
        audit[f"{slug}_score"] = score_test_bank[name]
        audit[f"{slug}_pred"] = pred_bank[name]
    audit.to_csv(
        outdir / "followup_edge_core_baselines_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Session bootstrap CIs and paired F1 differences versus FCN.
    ci_rows = []
    paired_rows = []
    for name in pred_bank:
        for row in session_bootstrap_ci(
            test_df,
            score_test_bank[name],
            pred_bank[name],
            args.bootstrap_samples,
            args.bootstrap_seed,
        ):
            row["Model"] = name
            ci_rows.append(row)

        if name != "Final FCN" and args.bootstrap_samples > 0:
            paired_rows.append(
                {
                    "Baseline": name,
                    **paired_f1_bootstrap(
                        test_df,
                        pred_bank["Final FCN"],
                        pred_bank[name],
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                    ),
                }
            )

    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(
            outdir / "followup_edge_core_baselines_bootstrap_ci.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            outdir / "followup_edge_core_baselines_paired_f1_vs_fcn.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metadata = {
        "protocol": "fixed_176_44_55_train_validation_test",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sessions": train_df["session_id"].nunique(),
        "validation_sessions": val_df["session_id"].nunique(),
        "test_sessions": test_df["session_id"].nunique(),
        "test_true_edges": int(test_df["edge_label"].sum()),
        "supervised_models_fit_on": "Train only",
        "baseline_model_selection": "Validation only",
        "test_used_for_model_selection": False,
        "models": {
            "sentence_encoder": args.sentence_encoder_model,
            "retrieval_encoder": args.retrieval_encoder_model,
            "financial_encoder": args.financial_encoder_model,
            "pairwise_cn": args.pairwise_cn_model,
            "pairwise_fin": args.pairwise_fin_model,
            "graphsage": args.graphsage_model,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (outdir / "main_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("Core baseline Test results")
    print("=" * 96)
    print(
        test_table[
            [
                "Model",
                "AUC",
                "AP",
                "Precision",
                "Recall",
                "F1",
                "PredEdges",
                "TP",
                "FP",
                "FN",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print(f"\nOutput directory: {outdir}")
    print("Primary table: followup_edge_core_baselines_test.csv")


def run_dirsage_only(args: argparse.Namespace) -> None:
    """Evaluate Dir-SAGE without re-running completed baseline families."""
    stage1_dir = Path(args.stage1_dir)
    outdir = Path(args.output_dir)
    freeze_dir = outdir / "validation_freeze"
    outdir.mkdir(parents=True, exist_ok=True)
    freeze_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(stage1_dir / "stage1_train_scored.csv")
    val_df = load_split(stage1_dir / "stage1_validation_scored.csv")
    stage1_manifest_path = stage1_dir / "stage1_split_manifest.csv"
    verify_stage1_artifacts(stage1_dir, train_df, val_df, stage1_manifest_path)

    dirsage_scores, dirsage_frozen = fit_dirsage_2way(
        train_df,
        val_df,
        args.graphsage_model,
        args.hf_cache_dir,
        args.encoder_batch_size,
    )
    checkpoint_path = freeze_dir / "dirsage_best_state.pt"
    save_torch(checkpoint_path, dirsage_frozen["state_dict"])
    dirsage_theta, _ = select_threshold_exact(val_df["edge_label"], dirsage_scores[1])
    val_pred = (dirsage_scores[1] >= dirsage_theta).astype(int)
    val_row = {
        "Split": "Validation",
        "Model": "Dir-SAGE",
        "Purpose": "Directed candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
        "Threshold": float(dirsage_theta),
        "TopK": np.nan,
        **binary_metrics(val_df["edge_label"], val_pred, dirsage_scores[1]),
    }
    freeze_manifest = {
        "protocol": "pretest_freeze_then_test",
        "baseline": "Dir-SAGE",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "validation_sessions": int(val_df["session_id"].nunique()),
        "selection": {
            "split": "Validation",
            "criterion": "max F1",
            "threshold": float(dirsage_theta),
        },
        "artifact": {
            **{k: v for k, v in dirsage_frozen.items() if k != "state_dict"},
            "artifact_path": str(checkpoint_path),
        },
        "test_loaded": False,
    }
    freeze_manifest_path = outdir / "dirsage_validation_freeze_manifest.json"
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dirsage_frozen["state_dict"] = load_torch(checkpoint_path)
    test_df = load_split(stage1_dir / "stage1_test_scored.csv")
    verify_split_manifest(train_df, val_df, test_df, load_session_manifest(stage1_manifest_path))
    dirsage_test = score_dirsage_from_frozen(
        test_df,
        dirsage_frozen,
        args.encoder_batch_size,
    )
    test_pred = (dirsage_test >= dirsage_theta).astype(int)
    test_row = {
        "Split": "Test",
        "Model": "Dir-SAGE",
        "Purpose": "Directed candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
        "Threshold": float(dirsage_theta),
        "TopK": np.nan,
        **binary_metrics(test_df["edge_label"], test_pred, dirsage_test),
    }

    pd.DataFrame([val_row]).to_csv(
        outdir / "dirsage_validation.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([test_row]).to_csv(
        outdir / "dirsage_test.csv", index=False, encoding="utf-8-sig"
    )
    test_predictions = test_df[
        [c for c in ["edge_id", "session_id", "Q1_row", "Q2_row", "edge_label"] if c in test_df.columns]
    ].copy()
    test_predictions["dirsage_score"] = dirsage_test
    test_predictions["dirsage_pred"] = test_pred
    test_predictions.to_csv(
        outdir / "dirsage_test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    freeze_manifest["test_loaded"] = True
    freeze_manifest["test_rows"] = len(test_df)
    freeze_manifest["test_sessions"] = int(test_df["session_id"].nunique())
    freeze_manifest["test_true_edges"] = int(test_df["edge_label"].sum())
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame([test_row]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nDir-SAGE results: {outdir / 'dirsage_test.csv'}")


def run_finetuned_financial_pairwise_only(args: argparse.Namespace) -> None:
    """Run the task-fine-tuned financial pairwise baseline in isolation."""
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    requested_outdir = Path(args.output_dir)
    if requested_outdir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        outdir = DEFAULT_FINETUNED_PAIRWISE_OUTPUT_DIR
    else:
        outdir = requested_outdir
    if outdir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        raise RuntimeError(
            "Fine-tuned Pairwise outputs must use a separate directory from the completed main experiment."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(stage1_dir / "stage1_train_scored.csv")
    validation_df = load_split(stage1_dir / "stage1_validation_scored.csv")
    stage1_manifest_path = stage1_dir / "stage1_split_manifest.csv"
    stage1_params = verify_stage1_artifacts(
        stage1_dir, train_df, validation_df, stage1_manifest_path
    )

    validation_scores, frozen, history = fit_finetuned_financial_pairwise_2way(
        train_df,
        validation_df,
        args.pairwise_fin_model,
        args.hf_cache_dir,
        outdir,
        args.pairwise_batch_size,
        args.max_length,
        args.finetuned_pairwise_epochs,
        args.finetuned_pairwise_learning_rate,
        args.finetuned_pairwise_weight_decay,
    )
    validation_pred = (validation_scores >= frozen["threshold"]).astype(int)
    validation_row = {
        "Split": "Validation",
        "Model": "Fine-tuned Financial Pairwise",
        "Purpose": "End-to-end financial Transformer pair classifier fitted on Train.",
        "SelectedEpoch": int(frozen["selected_epoch"]),
        "Threshold": float(frozen["threshold"]),
        "TopK": np.nan,
        **binary_metrics(validation_df["edge_label"], validation_pred, validation_scores),
    }
    pd.DataFrame(history).to_csv(
        outdir / "finetuned_financial_pairwise_validation_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([validation_row]).to_csv(
        outdir / "finetuned_financial_pairwise_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    freeze_manifest = {
        "protocol": "pretest_freeze_then_test",
        "baseline": "Fine-tuned Financial Pairwise",
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "validation_sessions": int(validation_df["session_id"].nunique()),
        "train_true_edges": int(train_df["edge_label"].sum()),
        "validation_true_edges": int(validation_df["edge_label"].sum()),
        "model": model_fingerprint(frozen["model_path"], args.hf_cache_dir),
        "selection": {
            "split": "Validation",
            "criterion": "max F1, then precision, fewer predicted edges, and threshold",
            "selected_epoch": int(frozen["selected_epoch"]),
            "threshold": float(frozen["threshold"]),
        },
        "artifact": frozen,
        "test_loaded": False,
    }
    freeze_manifest_path = outdir / "finetuned_financial_pairwise_validation_freeze_manifest.json"
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Test is loaded only after the selected checkpoint and threshold are persisted.
    test_df = load_split(stage1_dir / "stage1_test_scored.csv")
    verify_split_manifest(
        train_df, validation_df, test_df, load_session_manifest(stage1_manifest_path)
    )
    verify_stage1_artifacts(
        stage1_dir, train_df, validation_df, stage1_manifest_path, test_df=test_df
    )
    verify_stage2_artifacts(
        stage2_dir, stage1_params, validation_df, test_df=test_df
    )
    test_scores = score_finetuned_pairwise_from_checkpoint(test_df, frozen)
    test_pred = (test_scores >= frozen["threshold"]).astype(int)
    test_row = {
        "Split": "Test",
        "Model": "Fine-tuned Financial Pairwise",
        "Purpose": "End-to-end financial Transformer pair classifier fitted on Train.",
        "SelectedEpoch": int(frozen["selected_epoch"]),
        "Threshold": float(frozen["threshold"]),
        "TopK": np.nan,
        **binary_metrics(test_df["edge_label"], test_pred, test_scores),
    }
    pd.DataFrame([test_row]).to_csv(
        outdir / "finetuned_financial_pairwise_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = test_df[
        [
            column
            for column in ["edge_id", "session_id", "Q1_row", "Q2_row", "edge_label"]
            if column in test_df.columns
        ]
    ].copy()
    predictions["finetuned_financial_pairwise_score"] = test_scores
    predictions["finetuned_financial_pairwise_pred"] = test_pred
    predictions.to_csv(
        outdir / "finetuned_financial_pairwise_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ci_rows = session_bootstrap_ci(
        test_df,
        test_scores,
        test_pred,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    if ci_rows:
        pd.DataFrame(ci_rows).assign(Model="Fine-tuned Financial Pairwise").to_csv(
            outdir / "finetuned_financial_pairwise_bootstrap_ci.csv",
            index=False,
            encoding="utf-8-sig",
        )

    fcn_graph = align_graph_to_reference(
        test_df, load_split(stage2_dir / "stage2_test_final_graph.csv")
    )
    paired = paired_f1_bootstrap(
        test_df,
        fcn_graph["final_graph_pred_edge"].to_numpy(dtype=int),
        test_pred,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    if paired:
        pd.DataFrame([
            {"Baseline": "Fine-tuned Financial Pairwise", **paired}
        ]).to_csv(
            outdir / "finetuned_financial_pairwise_paired_f1_vs_fcn.csv",
            index=False,
            encoding="utf-8-sig",
        )

    freeze_manifest.update(
        {
            "test_loaded": True,
            "test_rows": len(test_df),
            "test_sessions": int(test_df["session_id"].nunique()),
            "test_true_edges": int(test_df["edge_label"].sum()),
        }
    )
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "protocol": "fixed_176_44_55_train_validation_test",
        "baseline": "Fine-tuned Financial Pairwise",
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "model": model_fingerprint(frozen["model_path"], args.hf_cache_dir),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "validation_freeze_manifest": str(freeze_manifest_path),
    }
    (outdir / "finetuned_financial_pairwise_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame([test_row]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nFine-tuned Financial Pairwise results: {outdir / 'finetuned_financial_pairwise_test.csv'}")


def run_formal_main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)
    outdir = Path(args.output_dir)
    freeze_dir = outdir / "validation_freeze"
    outdir.mkdir(parents=True, exist_ok=True)
    freeze_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(stage1_dir / "stage1_train_scored.csv")
    val_df = load_split(stage1_dir / "stage1_validation_scored.csv")
    stage1_manifest_path = stage1_dir / "stage1_split_manifest.csv"
    stage1_params = verify_stage1_artifacts(
        stage1_dir,
        train_df,
        val_df,
        stage1_manifest_path,
    )

    topk_values = parse_topk_values(args.topk_values)
    freeze_manifest_path = outdir / "validation_freeze_manifest.json"

    model_fingerprints = {
        "sentence_encoder": model_fingerprint(
            args.sentence_encoder_model,
            args.hf_cache_dir,
        ),
        "retrieval_encoder": model_fingerprint(
            args.retrieval_encoder_model,
            args.hf_cache_dir,
        ),
        "financial_encoder": model_fingerprint(
            args.financial_encoder_model,
            args.hf_cache_dir,
        ),
        "pairwise_cn": model_fingerprint(
            args.pairwise_cn_model,
            args.hf_cache_dir,
        ),
        "pairwise_fin": model_fingerprint(
            args.pairwise_fin_model,
            args.hf_cache_dir,
        ),
        "graphsage": model_fingerprint(
            args.graphsage_model,
            args.hf_cache_dir,
        ),
        "dirsage": model_fingerprint(
            args.graphsage_model,
            args.hf_cache_dir,
        ),
    }

    # ------------------------------------------------------------------
    # Pre-Test freezing on Train/Validation only.
    # ------------------------------------------------------------------
    pretest_scores = {}
    frozen_artifacts = {}
    val_rows = []
    selection_rows = []
    test_rows = []
    pred_bank = {}
    score_test_bank = {}
    param_rows = []

    # BM25
    bm25_scorer = fit_bm25(train_df)
    bm25_train = score_bm25(train_df, bm25_scorer)
    bm25_val = score_bm25(val_df, bm25_scorer)
    theta, _ = select_threshold_exact(val_df["edge_label"], bm25_val)
    bm25_val_pred = (bm25_val >= theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "BM25",
            "Purpose": "Surface lexical-overlap baseline.",
            "Threshold": float(theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], bm25_val_pred, bm25_val),
        }
    )
    selection_rows.append(
        {
            "Model": "BM25",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["BM25"] = {
        "scorer": bm25_scorer,
        "threshold": float(theta),
        "artifact_path": str(freeze_dir / "bm25_state.pkl"),
    }
    save_pickle(freeze_dir / "bm25_state.pkl", bm25_scorer)
    pretest_scores["BM25"] = (bm25_train, bm25_val)

    # Temporal distance prior
    dist_train = compute_distance_prior_scores(train_df)
    dist_val = compute_distance_prior_scores(val_df)
    theta, _ = select_threshold_exact(val_df["edge_label"], dist_val)
    dist_val_pred = (dist_val >= theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Temporal-distance Prior",
            "Purpose": "Inverse-distance temporal prior with Validation-selected threshold.",
            "Threshold": float(theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], dist_val_pred, dist_val),
        }
    )
    selection_rows.append(
        {
            "Model": "Temporal-distance Prior",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Temporal-distance Prior"] = {
        "threshold": float(theta),
    }

    # Sentence encoder retrieval
    sent_scores, sent_frozen = compute_sentence_encoder_scores_2way(
        train_df,
        val_df,
        args.sentence_encoder_model,
        args.hf_cache_dir,
        args.sentence_batch_size,
    )
    sent_theta, _ = select_threshold_exact(val_df["edge_label"], sent_scores[1])
    sent_val_pred = (sent_scores[1] >= sent_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "General Chinese Sentence Encoder Retrieval",
            "Purpose": "Off-the-shelf Chinese sentence-encoder cosine retrieval.",
            "Threshold": float(sent_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], sent_val_pred, sent_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "General Chinese Sentence Encoder Retrieval",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(sent_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["General Chinese Sentence Encoder Retrieval"] = {
        **sent_frozen,
        "threshold": float(sent_theta),
    }

    # Retrieval encoder retrieval
    retr_scores, retr_frozen = compute_transformer_retrieval_scores_2way(
        train_df,
        val_df,
        args.retrieval_encoder_model,
        args.hf_cache_dir,
        args.encoder_batch_size,
        args.max_length,
        use_sentence_transformer=True,
    )
    retr_theta, _ = select_threshold_exact(val_df["edge_label"], retr_scores[1])
    retr_val_pred = (retr_scores[1] >= retr_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "General Chinese Retrieval Encoder Retrieval",
            "Purpose": "General Chinese transformer retrieval by cosine similarity.",
            "Threshold": float(retr_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], retr_val_pred, retr_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "General Chinese Retrieval Encoder Retrieval",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(retr_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["General Chinese Retrieval Encoder Retrieval"] = {
        **retr_frozen,
        "threshold": float(retr_theta),
    }
    pretest_scores["General Chinese Retrieval Encoder Retrieval"] = retr_scores

    # Financial-domain encoder retrieval
    fin_scores, fin_frozen = compute_transformer_retrieval_scores_2way(
        train_df,
        val_df,
        args.financial_encoder_model,
        args.hf_cache_dir,
        args.encoder_batch_size,
        args.max_length,
        use_sentence_transformer=False,
    )
    fin_theta, _ = select_threshold_exact(val_df["edge_label"], fin_scores[1])
    fin_val_pred = (fin_scores[1] >= fin_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Financial-domain Encoder Retrieval",
            "Purpose": "Financial-domain transformer retrieval by cosine similarity.",
            "Threshold": float(fin_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], fin_val_pred, fin_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "Financial-domain Encoder Retrieval",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(fin_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Financial-domain Encoder Retrieval"] = {
        **fin_frozen,
        "threshold": float(fin_theta),
    }
    pretest_scores["Financial-domain Encoder Retrieval"] = fin_scores

    # Pairwise probe baselines
    pairwise_cn_scores, pairwise_cn_frozen = fit_pairwise_probe_2way(
        train_df,
        val_df,
        args.pairwise_cn_model,
        args.hf_cache_dir,
        args.pairwise_batch_size,
        args.max_length,
    )
    with open(freeze_dir / "pairwise_cn_state.pkl", "wb") as f:
        pickle.dump(pairwise_cn_frozen["classifier"], f)
    pairwise_cn_frozen["artifact_path"] = str(freeze_dir / "pairwise_cn_state.pkl")
    pairwise_cn_theta, _ = select_threshold_exact(
        val_df["edge_label"], pairwise_cn_scores[1]
    )
    pairwise_cn_val_pred = (pairwise_cn_scores[1] >= pairwise_cn_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "General Chinese Pairwise Transformer + Linear Probe",
            "Purpose": "Frozen Transformer pair encoding with a linear probe fitted on Train.",
            "Threshold": float(pairwise_cn_theta),
            "TopK": np.nan,
            **binary_metrics(
                val_df["edge_label"], pairwise_cn_val_pred, pairwise_cn_scores[1]
            ),
        }
    )
    selection_rows.append(
        {
            "Model": "General Chinese Pairwise Transformer + Linear Probe",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(pairwise_cn_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["General Chinese Pairwise Transformer + Linear Probe"] = {
        **pairwise_cn_frozen,
        "threshold": float(pairwise_cn_theta),
    }
    pretest_scores["General Chinese Pairwise Transformer + Linear Probe"] = pairwise_cn_scores

    pairwise_fin_scores, pairwise_fin_frozen = fit_pairwise_probe_2way(
        train_df,
        val_df,
        args.pairwise_fin_model,
        args.hf_cache_dir,
        args.pairwise_batch_size,
        args.max_length,
    )
    with open(freeze_dir / "pairwise_fin_state.pkl", "wb") as f:
        pickle.dump(pairwise_fin_frozen["classifier"], f)
    pairwise_fin_frozen["artifact_path"] = str(freeze_dir / "pairwise_fin_state.pkl")
    pairwise_fin_theta, _ = select_threshold_exact(
        val_df["edge_label"], pairwise_fin_scores[1]
    )
    pairwise_fin_val_pred = (pairwise_fin_scores[1] >= pairwise_fin_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Financial-domain Pairwise Transformer + Linear Probe",
            "Purpose": "Frozen financial-domain Transformer pair encoding with a linear probe fitted on Train.",
            "Threshold": float(pairwise_fin_theta),
            "TopK": np.nan,
            **binary_metrics(
                val_df["edge_label"], pairwise_fin_val_pred, pairwise_fin_scores[1]
            ),
        }
    )
    selection_rows.append(
        {
            "Model": "Financial-domain Pairwise Transformer + Linear Probe",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(pairwise_fin_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Financial-domain Pairwise Transformer + Linear Probe"] = {
        **pairwise_fin_frozen,
        "threshold": float(pairwise_fin_theta),
    }
    pretest_scores["Financial-domain Pairwise Transformer + Linear Probe"] = pairwise_fin_scores

    # Score fusion
    fusion_scores, fusion_frozen = fit_score_fusion_2way(
        train_df,
        val_df,
        pretest_scores["BM25"],
        pretest_scores["General Chinese Retrieval Encoder Retrieval"],
        pretest_scores["General Chinese Pairwise Transformer + Linear Probe"],
    )
    with open(freeze_dir / "score_fusion_state.pkl", "wb") as f:
        pickle.dump(fusion_frozen["model"], f)
    fusion_frozen["artifact_path"] = str(freeze_dir / "score_fusion_state.pkl")
    fusion_theta, _ = select_threshold_exact(val_df["edge_label"], fusion_scores[1])
    fusion_val_pred = (fusion_scores[1] >= fusion_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Score Fusion",
            "Purpose": "Train-only fusion of BM25, retrieval, pairwise, and distance.",
            "Threshold": float(fusion_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], fusion_val_pred, fusion_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "Score Fusion",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(fusion_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Score Fusion"] = {
        **fusion_frozen,
        "threshold": float(fusion_theta),
    }
    pretest_scores["Score Fusion"] = fusion_scores

    # Pairwise transformer + top-K pruning
    pairwise_topk_k, pairwise_topk_theta, pairwise_topk_search = select_topk_and_threshold(
        val_df,
        pairwise_cn_scores[1],
        topk_values,
    )
    pairwise_topk_val_pred = (
        anchor_topk_mask(val_df, pairwise_cn_scores[1], pairwise_topk_k)
        & (pairwise_cn_scores[1] >= pairwise_topk_theta)
    ).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Pairwise Transformer + top-K pruning",
            "Purpose": "Frozen Transformer pair encoding with Validation-selected Top-K pruning.",
            "Threshold": float(pairwise_topk_theta),
            "TopK": int(pairwise_topk_k),
            **binary_metrics(
                val_df["edge_label"], pairwise_topk_val_pred, pairwise_cn_scores[1]
            ),
        }
    )
    selection_rows.append(
        {
            "Model": "Pairwise Transformer + top-K pruning",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(pairwise_topk_theta),
            "TopK": int(pairwise_topk_k),
        }
    )
    if not pairwise_topk_search.empty:
        pairwise_topk_search.insert(0, "Model", "Pairwise Transformer + top-K pruning")
    frozen_artifacts["Pairwise Transformer + top-K pruning"] = {
        "threshold": float(pairwise_topk_theta),
        "topk": int(pairwise_topk_k),
    }
    pretest_scores["Pairwise Transformer + top-K pruning"] = pairwise_cn_scores

    # GraphSAGE
    graphsage_scores, graphsage_frozen = fit_graphsage_2way(
        train_df,
        val_df,
        args.graphsage_model,
        args.hf_cache_dir,
        args.encoder_batch_size,
    )
    save_torch(freeze_dir / "graphsage_best_state.pt", graphsage_frozen["state_dict"])
    graphsage_frozen["artifact_path"] = str(freeze_dir / "graphsage_best_state.pt")
    graphsage_theta, _ = select_threshold_exact(val_df["edge_label"], graphsage_scores[1])
    graphsage_val_pred = (graphsage_scores[1] >= graphsage_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "GraphSAGE",
            "Purpose": "Candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
            "Threshold": float(graphsage_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], graphsage_val_pred, graphsage_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "GraphSAGE",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(graphsage_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["GraphSAGE"] = {
        **graphsage_frozen,
        "threshold": float(graphsage_theta),
    }
    pretest_scores["GraphSAGE"] = graphsage_scores

    # Dir-SAGE
    dirsage_scores, dirsage_frozen = fit_dirsage_2way(
        train_df,
        val_df,
        args.graphsage_model,
        args.hf_cache_dir,
        args.encoder_batch_size,
    )
    save_torch(freeze_dir / "dirsage_best_state.pt", dirsage_frozen["state_dict"])
    dirsage_frozen["artifact_path"] = str(freeze_dir / "dirsage_best_state.pt")
    dirsage_theta, _ = select_threshold_exact(val_df["edge_label"], dirsage_scores[1])
    dirsage_val_pred = (dirsage_scores[1] >= dirsage_theta).astype(int)
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Dir-SAGE",
            "Purpose": "Directed candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
            "Threshold": float(dirsage_theta),
            "TopK": np.nan,
            **binary_metrics(val_df["edge_label"], dirsage_val_pred, dirsage_scores[1]),
        }
    )
    selection_rows.append(
        {
            "Model": "Dir-SAGE",
            "SelectionSplit": "Validation",
            "SelectionCriterion": "max F1",
            "Threshold": float(dirsage_theta),
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Dir-SAGE"] = {
        **dirsage_frozen,
        "threshold": float(dirsage_theta),
    }
    pretest_scores["Dir-SAGE"] = dirsage_scores

    # Final FCN validation only
    val_graph = align_graph_to_reference(
        val_df,
        load_split(stage2_dir / "stage2_validation_final_graph.csv"),
    )
    val_rows.append(
        {
            "Split": "Validation",
            "Model": "Final FCN",
            "Purpose": "Frozen revised FCN with exact Stage 2B.",
            "Threshold": np.nan,
            "TopK": np.nan,
            **binary_metrics(
                val_df["edge_label"],
                val_graph["final_graph_pred_edge"],
                val_graph["final_graph_score"],
            ),
        }
    )
    selection_rows.append(
        {
            "Model": "Final FCN",
            "SelectionSplit": "Validation inside FCN Stage 2",
            "SelectionCriterion": "frozen",
            "Threshold": np.nan,
            "TopK": np.nan,
        }
    )
    frozen_artifacts["Final FCN"] = {
        "validation_graph_path": str(stage2_dir / "stage2_validation_final_graph.csv"),
    }

    def _artifact_manifest_view(item: dict) -> dict:
        drop_keys = {"scorer", "classifier", "model", "state_dict"}
        return {k: v for k, v in item.items() if k not in drop_keys}

    freeze_manifest = {
        "protocol": "pretest_freeze_then_test",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "validation_sessions": int(val_df["session_id"].nunique()),
        "train_true_edges": int(train_df["edge_label"].sum()),
        "validation_true_edges": int(val_df["edge_label"].sum()),
        "model_fingerprints": model_fingerprints,
        "selection": selection_rows,
        "artifacts": {k: _artifact_manifest_view(v) for k, v in frozen_artifacts.items()},
        "test_loaded": False,
    }
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Reload frozen artifacts from disk before touching Test.
    bm25_scorer = load_pickle(freeze_dir / "bm25_state.pkl")
    pairwise_cn_frozen["classifier"] = load_pickle(freeze_dir / "pairwise_cn_state.pkl")
    pairwise_fin_frozen["classifier"] = load_pickle(freeze_dir / "pairwise_fin_state.pkl")
    fusion_frozen["model"] = load_pickle(freeze_dir / "score_fusion_state.pkl")
    graphsage_frozen["state_dict"] = load_torch(freeze_dir / "graphsage_best_state.pt")
    dirsage_frozen["state_dict"] = load_torch(freeze_dir / "dirsage_best_state.pt")
    gc.collect()

    # ------------------------------------------------------------------
    # Test stage starts only after freezing is written.
    # ------------------------------------------------------------------
    test_df = load_split(stage1_dir / "stage1_test_scored.csv")
    expected_counts_check("Test", test_df, 10434, 528, 55)
    verify_split_manifest(train_df, val_df, test_df, load_session_manifest(stage1_manifest_path))
    verify_stage1_artifacts(
        stage1_dir,
        train_df,
        val_df,
        stage1_manifest_path,
        test_df=test_df,
    )
    verify_stage2_artifacts(
        stage2_dir,
        stage1_params,
        val_df,
        test_df=test_df,
    )

    test_rows = []
    # Score all reusable components on Test only now.
    bm25_test = score_bm25(test_df, bm25_scorer)
    dist_test = compute_distance_prior_scores(test_df)
    sent_test = score_sentence_encoder_single(
        test_df,
        sent_frozen["model_path"],
        args.sentence_batch_size,
    )
    retr_test = score_transformer_retrieval_single(
        test_df,
        retr_frozen["model_path"],
        args.hf_cache_dir,
        args.encoder_batch_size,
        args.max_length,
        use_sentence_transformer=True,
    )
    fin_test = score_transformer_retrieval_single(
        test_df,
        fin_frozen["model_path"],
        args.hf_cache_dir,
        args.encoder_batch_size,
        args.max_length,
        use_sentence_transformer=False,
    )
    pairwise_cn_test = score_pairwise_probe_from_frozen(
        test_df,
        pairwise_cn_frozen,
        args.hf_cache_dir,
        args.pairwise_batch_size,
        args.max_length,
    )
    pairwise_fin_test = score_pairwise_probe_from_frozen(
        test_df,
        pairwise_fin_frozen,
        args.hf_cache_dir,
        args.pairwise_batch_size,
        args.max_length,
    )
    fusion_test = score_fusion_from_frozen(
        test_df,
        fusion_frozen,
        bm25_test,
        retr_test,
        pairwise_cn_test,
    )
    graphsage_test = score_graphsage_from_frozen(
        test_df,
        graphsage_frozen,
        args.encoder_batch_size,
    )
    dirsage_test = score_dirsage_from_frozen(
        test_df,
        dirsage_frozen,
        args.encoder_batch_size,
    )

    # Build test rows and predictions.
    def add_test_threshold_row(name, purpose, scores, theta, k=np.nan):
        pred = (np.asarray(scores) >= float(theta)).astype(int)
        row = {
            "Split": "Test",
            "Model": name,
            "Purpose": purpose,
            "Threshold": float(theta),
            "TopK": k,
            **binary_metrics(test_df["edge_label"], pred, scores),
        }
        test_rows.append(row)
        pred_bank[name] = pred
        score_test_bank[name] = np.asarray(scores, dtype=float)

    add_test_threshold_row(
        "BM25",
        "Surface lexical-overlap baseline.",
        bm25_test,
        frozen_artifacts["BM25"]["threshold"],
    )
    add_test_threshold_row(
        "Temporal-distance Prior",
        "Inverse-distance temporal prior with Validation-selected threshold.",
        dist_test,
        frozen_artifacts["Temporal-distance Prior"]["threshold"],
    )
    add_test_threshold_row(
        "General Chinese Sentence Encoder Retrieval",
        "Off-the-shelf Chinese sentence-encoder cosine retrieval.",
        sent_test,
        frozen_artifacts["General Chinese Sentence Encoder Retrieval"]["threshold"],
    )
    add_test_threshold_row(
        "General Chinese Retrieval Encoder Retrieval",
        "General Chinese transformer retrieval by cosine similarity.",
        retr_test,
        frozen_artifacts["General Chinese Retrieval Encoder Retrieval"]["threshold"],
    )
    add_test_threshold_row(
        "Financial-domain Encoder Retrieval",
        "Financial-domain transformer retrieval by cosine similarity.",
        fin_test,
        frozen_artifacts["Financial-domain Encoder Retrieval"]["threshold"],
    )
    add_test_threshold_row(
        "General Chinese Pairwise Transformer + Linear Probe",
        "Frozen Transformer pair encoding with a linear probe fitted on Train.",
        pairwise_cn_test,
        frozen_artifacts["General Chinese Pairwise Transformer + Linear Probe"]["threshold"],
    )
    add_test_threshold_row(
        "Financial-domain Pairwise Transformer + Linear Probe",
        "Frozen financial-domain Transformer pair encoding with a linear probe fitted on Train.",
        pairwise_fin_test,
        frozen_artifacts["Financial-domain Pairwise Transformer + Linear Probe"]["threshold"],
    )
    add_test_threshold_row(
        "Score Fusion",
        "Train-only fusion of BM25, retrieval, pairwise, and distance.",
        fusion_test,
        frozen_artifacts["Score Fusion"]["threshold"],
    )
    pairwise_topk_pred = (
        anchor_topk_mask(test_df, pairwise_cn_test, frozen_artifacts["Pairwise Transformer + top-K pruning"]["topk"])
        & (pairwise_cn_test >= frozen_artifacts["Pairwise Transformer + top-K pruning"]["threshold"])
    ).astype(int)
    test_rows.append(
        {
            "Split": "Test",
            "Model": "Pairwise Transformer + top-K pruning",
            "Purpose": "Frozen Transformer pair encoding with Validation-selected Top-K pruning.",
            "Threshold": float(frozen_artifacts["Pairwise Transformer + top-K pruning"]["threshold"]),
            "TopK": int(frozen_artifacts["Pairwise Transformer + top-K pruning"]["topk"]),
            **binary_metrics(test_df["edge_label"], pairwise_topk_pred, pairwise_cn_test),
        }
    )
    pred_bank["Pairwise Transformer + top-K pruning"] = pairwise_topk_pred
    score_test_bank["Pairwise Transformer + top-K pruning"] = pairwise_cn_test

    test_rows.append(
        {
            "Split": "Test",
            "Model": "GraphSAGE",
            "Purpose": "Candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
            "Threshold": float(frozen_artifacts["GraphSAGE"]["threshold"]),
            "TopK": np.nan,
            **binary_metrics(test_df["edge_label"], (graphsage_test >= frozen_artifacts["GraphSAGE"]["threshold"]).astype(int), graphsage_test),
        }
    )
    pred_bank["GraphSAGE"] = (graphsage_test >= frozen_artifacts["GraphSAGE"]["threshold"]).astype(int)
    score_test_bank["GraphSAGE"] = graphsage_test

    test_rows.append(
        {
            "Split": "Test",
            "Model": "Dir-SAGE",
            "Purpose": "Directed candidate-adjacency GraphSAGE edge classifier on turn embeddings.",
            "Threshold": float(frozen_artifacts["Dir-SAGE"]["threshold"]),
            "TopK": np.nan,
            **binary_metrics(test_df["edge_label"], (dirsage_test >= frozen_artifacts["Dir-SAGE"]["threshold"]).astype(int), dirsage_test),
        }
    )
    pred_bank["Dir-SAGE"] = (dirsage_test >= frozen_artifacts["Dir-SAGE"]["threshold"]).astype(int)
    score_test_bank["Dir-SAGE"] = dirsage_test

    test_graph = align_graph_to_reference(
        test_df,
        load_split(stage2_dir / "stage2_test_final_graph.csv"),
    )
    test_rows.append(
        {
            "Split": "Test",
            "Model": "Final FCN",
            "Purpose": "Frozen revised FCN with exact Stage 2B.",
            "Threshold": np.nan,
            "TopK": np.nan,
            **binary_metrics(
                test_df["edge_label"],
                test_graph["final_graph_pred_edge"],
                test_graph["final_graph_score"],
            ),
        }
    )
    pred_bank["Final FCN"] = test_graph["final_graph_pred_edge"].to_numpy(dtype=int)
    score_test_bank["Final FCN"] = test_graph["final_graph_score"].to_numpy(dtype=float)

    val_table = pd.DataFrame(val_rows)
    test_table = pd.DataFrame(test_rows)
    order = {name: i for i, name in enumerate(CORE_BASELINE_ORDER)}
    for table in (val_table, test_table):
        table["_order"] = table["Model"].map(order)
        table.sort_values("_order", inplace=True, kind="mergesort")
        table.drop(columns="_order", inplace=True)
        table.reset_index(drop=True, inplace=True)

    val_table.to_csv(
        outdir / "followup_edge_core_baselines_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_table.to_csv(
        outdir / "followup_edge_core_baselines_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selection_rows).to_csv(
        outdir / "followup_edge_core_baselines_selected_params.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not pairwise_topk_search.empty:
        pairwise_topk_search.to_csv(
            outdir / "pairwise_transformer_topk_validation_search.csv",
            index=False,
            encoding="utf-8-sig",
        )

    audit = test_df[[
        c for c in ["edge_id", "session_id", "Q1_row", "Q2_row", "edge_label"] if c in test_df.columns
    ]].copy()
    for name in CORE_BASELINE_ORDER:
        if name not in pred_bank:
            continue
        slug = slugify(name)
        audit[f"{slug}_score"] = score_test_bank[name]
        audit[f"{slug}_pred"] = pred_bank[name]
    audit.to_csv(
        outdir / "followup_edge_core_baselines_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ci_rows = []
    paired_rows = []
    for name in pred_bank:
        for row in session_bootstrap_ci(
            test_df,
            score_test_bank[name],
            pred_bank[name],
            args.bootstrap_samples,
            args.bootstrap_seed,
        ):
            row["Model"] = name
            ci_rows.append(row)
        if name != "Final FCN" and args.bootstrap_samples > 0:
            paired_rows.append(
                {
                    "Baseline": name,
                    **paired_f1_bootstrap(
                        test_df,
                        pred_bank["Final FCN"],
                        pred_bank[name],
                        args.bootstrap_samples,
                        args.bootstrap_seed,
                    ),
                }
            )
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(
            outdir / "followup_edge_core_baselines_bootstrap_ci.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(
            outdir / "followup_edge_core_baselines_paired_f1_vs_fcn.csv",
            index=False,
            encoding="utf-8-sig",
        )

    freeze_manifest["test_loaded"] = True
    freeze_manifest["test_rows"] = len(test_df)
    freeze_manifest["test_sessions"] = int(test_df["session_id"].nunique())
    freeze_manifest["test_true_edges"] = int(test_df["edge_label"].sum())
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "protocol": "pretest_freeze_then_test",
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sessions": int(train_df["session_id"].nunique()),
        "validation_sessions": int(val_df["session_id"].nunique()),
        "test_sessions": int(test_df["session_id"].nunique()),
        "test_true_edges": int(test_df["edge_label"].sum()),
        "supervised_models_fit_on": "Train only",
        "baseline_model_selection": "Validation only",
        "test_used_for_model_selection": False,
        "validation_freeze_manifest": str(freeze_manifest_path),
        "models": model_fingerprints,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (outdir / "main_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("Core baseline Test results")
    print("=" * 96)
    print(
        test_table[
            [
                "Model",
                "AUC",
                "AP",
                "Precision",
                "Recall",
                "F1",
                "PredEdges",
                "TP",
                "FP",
                "FN",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    print(f"\nOutput directory: {outdir}")
    print("Primary table: followup_edge_core_baselines_test.csv")


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.dirsage_only and parsed_args.finetuned_financial_pairwise_only:
        raise ValueError("Choose only one standalone baseline mode.")
    if parsed_args.dirsage_only:
        run_dirsage_only(parsed_args)
    elif parsed_args.finetuned_financial_pairwise_only:
        run_finetuned_financial_pairwise_only(parsed_args)
    else:
        run_formal_main(parsed_args)
