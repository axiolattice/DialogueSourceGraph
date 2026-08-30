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
pruning; Final FCN.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from transformers import AutoModel, AutoTokenizer

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:
    hf_snapshot_download = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_STAGE1_DIR = PROJECT_DIR / "stage1" / "fcn_outputs_stage1_final"
DEFAULT_STAGE2_DIR = PROJECT_DIR / "stage2" / "fcn_outputs_stage2_final_stage1only"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "fcn_outputs_main_experiments"
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
    if observed != expected:
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


# =============================================================================
# Baselines retained from the prior FCN_main_experiments.py
# =============================================================================

def simple_tokenize(text: object) -> List[str]:
    value = clean_text(text)
    if not value:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", value)
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
    return SimpleBM25(
        [simple_tokenize(x) for x in train_df["anchor_text"].tolist()]
    )


def score_bm25(df: pd.DataFrame, scorer: SimpleBM25) -> np.ndarray:
    values = []
    for row in df.itertuples(index=False):
        values.append(
            scorer.score(
                simple_tokenize(getattr(row, "candidate_text", "")),
                simple_tokenize(getattr(row, "anchor_text", "")),
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
):
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


# =============================================================================
# Cache
# =============================================================================

def slugify(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", name.lower())
    return re.sub(r"_+", "_", s).strip("_") or "baseline"


def cache_file(outdir: Path, name: str) -> Path:
    suffix = "_v3" if name == "GraphSAGE" else ""
    return outdir / "cache" / f"{slugify(name)}{suffix}_scores.npz"


def save_score_cache(path: Path, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_scores=np.asarray(scores[0], dtype=float),
        validation_scores=np.asarray(scores[1], dtype=float),
        test_scores=np.asarray(scores[2], dtype=float),
    )


def load_score_cache(path: Path, lengths):
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
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--skip-failed-baselines", action="store_true")
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
    verify_three_way_protocol(train_df, val_df, test_df)

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
    score_bank = {}
    val_rows = []
    test_rows = []
    params = []
    pred_bank = {}
    score_test_bank = {}
    topk_search = []

    def cached(name, fn):
        path = cache_file(outdir, name)
        scores = None if args.no_cache else load_score_cache(path, lengths)
        if scores is not None:
            LOGGER.info("Loaded cache: %s", name)
            return scores
        LOGGER.info("Running: %s", name)
        scores = fn()
        save_score_cache(path, scores)
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
            lambda: cached("BM25", bm25),
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
                ),
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
        "models": DEFAULT_MODELS,
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
    main()
