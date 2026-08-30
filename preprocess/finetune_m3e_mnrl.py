# -*- coding: utf-8 -*-
"""Fine-tune m3e-base for the Stage 1 follow-up retrieval encoder.

This script trains a bi-encoder for the first-stage dependency retrieval task
(`dependent=Yes` vs `dependent=No`) using MNRL plus optional auxiliary signals.
The training recipe can include:
1. positive follow-up pairs from the labeled corpus,
2. repeated hard positives from missed true edges,
3. mined hard negatives from high-score false positives,
4. reason-category prototype alignment derived from LLM explanations / dep_reason,
5. continuation-compatibility ranking on train-only weak labels.

The resulting encoder is the representation layer used by Stage 1 candidate
generation. It is designed for follow-up dependency retrieval rather than plain
semantic similarity.

python3 0_finetune_m3e_mnrl.py \
  --train-file "./train data/fcn_30firms_full_labeled.csv" \
  --base-model "AI-ModelScope/m3e-base" \
  --output-path "./checkpoints/fcn-m3e-base-mnrl" \
  --epochs 3 \
  --batch-size 32 \
  --hard-negative-batch-size 8 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.10 \
  --scale 20.0 \
  --max-seq-length 256 \
  --hard-negative-files "./fcn_outputs_stage1/stage1_train_scored.csv" \
  --hard-negative-min-score 0.90 \
  --hard-negative-weight 0.15 \
  --device auto
"""

import argparse
import math
from dataclasses import dataclass
import gc
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import snapshot_download
from sentence_transformers import InputExample, SentenceTransformer

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from followup_semantics import (
    REASON_CATEGORY_PROMPTS,
    clean_text,
    get_reason_prompt,
    infer_reason_category_from_row,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FILE = "./train data/fcn_30firms_full_labeled.csv"
DEFAULT_BASE_MODEL = "AI-ModelScope/m3e-base"
DEFAULT_OUTPUT_PATH = "./checkpoints/fcn-m3e-base-mnrl"
DEFAULT_MODELSCOPE_CACHE = "./checkpoints"
DEFAULT_HARD_POSITIVE_FILES = ""
DEFAULT_HARD_NEGATIVE_FILES = ""
SEED = 42


@dataclass
class HardNegativeExample:
    anchor: str
    positive: str
    negative: str


@dataclass
class ReasonPrototypeExample:
    anchor: str
    prototype: str
    reason_category: str


@dataclass
class DirectionAuxExample:
    anchor: str
    positive: str
    negative: str
    weight: float


def build_anchor_text(q1: object, a1: object, use_prompts: bool = True, reason_category: str = "") -> str:
    q1_text = clean_text(q1)
    a1_text = clean_text(a1)
    reason_text = ""
    prompt_text = get_reason_prompt(reason_category)
    if prompt_text:
        reason_text = " " + prompt_text
    topic_shift_guard = " 不要把单纯换话题、无关延伸或表面相关内容误判为追问延续。"
    if use_prompts:
        return (
            "[追问关系检索] 上一轮问题："
            + q1_text
            + " 上一轮回答："
            + a1_text
            + reason_text
            + " 判断候选问题是否延续同一主题，或是否属于回答不充分后的继续追问，"
            + "是否是在追问披露细节、股东回报、重组进展、经营规划、业绩治理等业务语义原型。"
            + topic_shift_guard
        )
    return "前序问题：" + q1_text + " 管理层回答：" + a1_text + reason_text + topic_shift_guard


def build_candidate_text(q2: object, use_prompts: bool = True) -> str:
    q2_text = clean_text(q2)
    if use_prompts:
        return "[候选后续问题] " + q2_text
    return "当前追问：" + q2_text


def build_plain_anchor_text(q1: object, a1: object) -> str:
    q1_text = clean_text(q1)
    a1_text = clean_text(a1)
    return "前序问题：" + q1_text + " 管理层回答：" + a1_text


def resolve_model_path(model_id_or_path: str, cache_dir: str = DEFAULT_MODELSCOPE_CACHE) -> str:
    if os.path.exists(model_id_or_path):
        return model_id_or_path
    cache_dir_path = Path(cache_dir)
    cached_repo_path = cache_dir_path / model_id_or_path
    if cached_repo_path.exists():
        logger.info("Resolved ModelScope model %s from local cache path %s", model_id_or_path, cached_repo_path)
        return str(cached_repo_path)

    try:
        model_path = snapshot_download(model_id_or_path, cache_dir=cache_dir, local_files_only=True)
        logger.info("Resolved ModelScope model %s from local cache -> %s", model_id_or_path, model_path)
        return model_path
    except Exception as local_exc:
        logger.warning("Local cache lookup for %s failed: %s", model_id_or_path, local_exc)

    model_path = snapshot_download(model_id_or_path, cache_dir=cache_dir, local_files_only=False)
    logger.info("Downloaded/resolved ModelScope model %s -> %s", model_id_or_path, model_path)
    return model_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_file_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def repeat_count(value: object, multiplier: int = 1) -> int:
    count = pd.to_numeric(value, errors="coerce")
    if pd.isna(count):
        count = 1
    return max(int(count) * max(multiplier, 1), 1)


def dependent_yes_mask(df: pd.DataFrame) -> pd.Series:
    if "dependent" in df.columns:
        return df["dependent"].astype(str).str.lower().isin({"yes", "1", "true"})
    if "edge_label" in df.columns:
        return pd.to_numeric(df["edge_label"], errors="coerce").fillna(0).astype(int) == 1
    raise ValueError("Input table must contain either dependent or edge_label.")


def make_pair_example(row: object, use_prompts: bool) -> InputExample:
    reason_category, _, _ = infer_reason_category_from_row(row)
    anchor = build_anchor_text(
        getattr(row, "Q1", ""),
        getattr(row, "A1", ""),
        use_prompts=use_prompts,
        reason_category=reason_category,
    )
    positive = build_candidate_text(getattr(row, "Q2", ""), use_prompts=use_prompts)
    return InputExample(texts=[anchor, positive])


def load_positive_examples(train_file: str, use_prompts: bool) -> List[InputExample]:
    df = read_table(train_file)
    df = df[dependent_yes_mask(df)].copy()
    for col in ["Q1", "A1", "Q2"]:
        df[col] = df[col].apply(clean_text)

    examples: List[InputExample] = []
    duplicate_anchor_count = 0
    anchor_seen: Dict[str, int] = {}
    for row in df.itertuples(index=False):
        example = make_pair_example(row, use_prompts=use_prompts)
        anchor, positive = example.texts
        if not anchor or not positive:
            continue
        anchor_seen[anchor] = anchor_seen.get(anchor, 0) + 1
        if anchor_seen[anchor] > 1:
            duplicate_anchor_count += 1
        examples.append(example)

    logger.info(
        "Loaded %d positive pairs from %s; duplicate-positive anchors=%d",
        len(examples),
        train_file,
        duplicate_anchor_count,
    )
    return examples


def continuation_negative_rank(row: object) -> Tuple[int, int, int, int]:
    bucket = clean_text(getattr(row, "topic_shift_bucket", "")).lower()
    bucket_rank = {"boundary": 0, "keep": 1, "na": 2, "": 2, "easy": 3}.get(bucket, 2)
    distance = pd.to_numeric(getattr(row, "distance", np.nan), errors="coerce")
    distance_rank = int(distance) if not pd.isna(distance) else 99
    q2_len = len(clean_text(getattr(row, "Q2", "")))
    q1_row = pd.to_numeric(getattr(row, "Q1_row", np.nan), errors="coerce")
    q1_rank = int(q1_row) if not pd.isna(q1_row) else 999999
    return bucket_rank, distance_rank, q2_len, q1_rank


def load_direction_aux_examples(
    train_file: str,
    use_prompts: bool,
    negative_ratio: float,
    seed: int,
) -> List[DirectionAuxExample]:
    df = read_table(train_file)
    required = ["Q1", "A1", "Q2"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Train file missing columns for direction aux task: {missing}")
    if "dependency_type" not in df.columns:
        raise ValueError("Train file missing dependency_type column for direction aux task.")

    df = df.copy()
    for col in required:
        df[col] = df[col].apply(clean_text)

    pos_df = df[df["dependency_type"].astype(str).str.strip().ne("No-Dependency")].copy()
    neg_df = df[df["dependency_type"].astype(str).str.strip().eq("No-Dependency")].copy()
    pos_count = len(pos_df)
    if pos_count == 0:
        raise ValueError("No dependent rows found for direction aux task.")

    if "topic_shift_bucket" in neg_df.columns:
        hard_neg_df = neg_df[
            neg_df["topic_shift_bucket"].astype(str).str.lower().isin({"boundary", "keep"})
        ].copy()
        if not hard_neg_df.empty:
            neg_df = hard_neg_df

    neg_by_session: Dict[str, List[object]] = defaultdict(list)
    neg_by_firm_year: Dict[Tuple[str, str], List[object]] = defaultdict(list)
    neg_global: List[object] = []
    for row in neg_df.itertuples(index=False):
        neg_global.append(row)
        session_id = clean_text(getattr(row, "session_id", ""))
        if session_id:
            neg_by_session[session_id].append(row)
        scode = clean_text(getattr(row, "Scode", ""))
        year = clean_text(getattr(row, "Year", ""))
        if scode and year:
            neg_by_firm_year[(scode, year)].append(row)

    examples: List[DirectionAuxExample] = []
    max_neg_per_pos = int(max(math.ceil(negative_ratio), 1))
    rng = random.Random(seed)

    for row in pos_df.itertuples(index=False):
        anchor = build_plain_anchor_text(getattr(row, "Q1", ""), getattr(row, "A1", ""))
        candidate = build_candidate_text(getattr(row, "Q2", ""), use_prompts=use_prompts)
        if not anchor or not candidate:
            continue

        session_id = clean_text(getattr(row, "session_id", ""))
        scode = clean_text(getattr(row, "Scode", ""))
        year = clean_text(getattr(row, "Year", ""))
        candidate_pools: List[Tuple[str, List[object]]] = []
        if session_id and neg_by_session.get(session_id):
            candidate_pools.append(("session", sorted(neg_by_session[session_id], key=continuation_negative_rank)))
        if scode and year and neg_by_firm_year.get((scode, year)):
            candidate_pools.append(("firm_year", sorted(neg_by_firm_year[(scode, year)], key=continuation_negative_rank)))
        if neg_global:
            shuffled_global = neg_global[:]
            rng.shuffle(shuffled_global)
            candidate_pools.append(("global", sorted(shuffled_global, key=continuation_negative_rank)))

        seen_negatives = set()
        sampled_neg = 0
        for pool_source, pool in candidate_pools:
            for neg_row in pool:
                if sampled_neg >= max_neg_per_pos:
                    break
                neg_key = getattr(neg_row, "edge_id", None) or (
                    clean_text(getattr(neg_row, "session_id", "")),
                    clean_text(getattr(neg_row, "Q1_row", "")),
                    clean_text(getattr(neg_row, "Q2_row", "")),
                )
                if neg_key in seen_negatives:
                    continue
                negative_candidate = build_candidate_text(getattr(neg_row, "Q2", ""), use_prompts=use_prompts)
                if not negative_candidate:
                    continue
                seen_negatives.add(neg_key)
                weight = 1.0
                if pool_source == "session":
                    weight += 0.50
                elif pool_source == "firm_year":
                    weight += 0.30
                bucket_rank, distance_rank, q2_len, _ = continuation_negative_rank(neg_row)
                weight += max(0.0, (3 - bucket_rank)) * 0.10
                if distance_rank <= 3:
                    weight += 0.08
                if q2_len <= 25:
                    weight += 0.08
                if pool_source == "global" and bucket_rank <= 1:
                    weight += 0.05
                examples.append(
                    DirectionAuxExample(
                        anchor=anchor,
                        positive=candidate,
                        negative=negative_candidate,
                        weight=weight,
                    )
                )
                sampled_neg += 1
            if sampled_neg >= max_neg_per_pos:
                break

    logger.info(
        "Loaded %d continuation-ranking triplets from %s; negative_ratio=%.2f",
        len(examples),
        train_file,
        negative_ratio,
    )
    return examples


def load_hard_positive_examples(
    files: Sequence[str],
    use_prompts: bool,
    repeat: int,
    complaint_repeat: int,
    challenge_repeat: int,
) -> List[InputExample]:
    examples: List[InputExample] = []
    seen = set()
    for file_name in files:
        path = Path(file_name)
        if not path.exists():
            logger.warning("Hard-positive file not found, skip: %s", file_name)
            continue
        df = read_table(str(path))
        if "dependent" in df.columns or "edge_label" in df.columns:
            df = df[dependent_yes_mask(df)].copy()
        for col in ["Q1", "A1", "Q2"]:
            if col not in df.columns:
                raise ValueError(f"Hard-positive file {file_name} missing column {col}")
            df[col] = df[col].apply(clean_text)
        file_examples = 0
        for row in df.itertuples(index=False):
            example = make_pair_example(row, use_prompts=use_prompts)
            if not example.texts[0] or not example.texts[1]:
                continue
            key = tuple(example.texts)
            if key in seen:
                continue
            seen.add(key)
            relation_type = str(getattr(row, "relation_type", ""))
            row_repeat = repeat
            if relation_type == "Complaint":
                row_repeat = max(row_repeat, complaint_repeat)
            elif relation_type == "Challenge":
                row_repeat = max(row_repeat, challenge_repeat)
            for _ in range(max(row_repeat, 1)):
                examples.append(example)
            file_examples += 1
        logger.info(
            "Loaded %d unique hard positives from %s repeat=%d complaint_repeat=%d challenge_repeat=%d",
            file_examples,
            file_name,
            repeat,
            complaint_repeat,
            challenge_repeat,
        )
    logger.info("Loaded %d hard-positive training pairs after repeat", len(examples))
    return examples


def infer_score_pred_columns(df: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("topic_shift_score", ""),
        ("final_graph_score", "final_graph_pred_edge"),
        ("stage2_prob", "stage2_pred_edge"),
        ("stage1_prob", "stage1_pred_edge"),
    ]
    for score_col, pred_col in candidates:
        if score_col in df.columns and (not pred_col or pred_col in df.columns):
            return score_col, pred_col
    for score_col in ["final_graph_score", "stage2_prob", "stage1_prob"]:
        if score_col in df.columns:
            return score_col, ""
    raise ValueError("Could not infer score column from hard-negative file.")


def load_hard_negative_examples(
    train_file: str,
    hard_negative_files: Sequence[str],
    use_prompts: bool,
    max_per_anchor: int,
    min_score: float,
) -> List[HardNegativeExample]:
    base = read_table(train_file)
    base = base[dependent_yes_mask(base)].copy()
    for col in ["Q1", "A1", "Q2", "session_id", "Q1_row"]:
        if col not in base.columns:
            raise ValueError(f"Train file missing column {col}")
    positive_by_anchor: Dict[Tuple[str, str], List[str]] = {}
    anchor_text_by_key: Dict[Tuple[str, str], str] = {}
    for row in base.itertuples(index=False):
        key = (str(getattr(row, "session_id")), str(getattr(row, "Q1_row")))
        anchor = build_anchor_text(getattr(row, "Q1", ""), getattr(row, "A1", ""), use_prompts=use_prompts)
        positive = build_candidate_text(getattr(row, "Q2", ""), use_prompts=use_prompts)
        if not anchor or not positive:
            continue
        anchor_text_by_key[key] = anchor
        positive_by_anchor.setdefault(key, []).append(positive)

    triples: List[HardNegativeExample] = []
    seen = set()
    for file_name in hard_negative_files:
        path = Path(file_name)
        if not path.exists():
            logger.warning("Hard-negative file not found, skip: %s", file_name)
            continue
        df = read_table(str(path))
        for col in ["Q1", "A1", "Q2", "session_id", "Q1_row"]:
            if col not in df.columns:
                raise ValueError(f"Hard-negative file {file_name} missing column {col}")
        score_col, pred_col = infer_score_pred_columns(df)
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
        is_negative = ~dependent_yes_mask(df)
        if pred_col:
            pred_positive = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int) == 1
            candidates = df[is_negative & pred_positive & (df[score_col] >= min_score)].copy()
        else:
            candidates = df[is_negative & (df[score_col] >= min_score)].copy()
        candidates = candidates.sort_values(score_col, ascending=False)
        file_triples = 0
        for key, group in candidates.groupby(["session_id", "Q1_row"], sort=False):
            anchor_key = (str(key[0]), str(key[1]))
            positives = positive_by_anchor.get(anchor_key)
            anchor = anchor_text_by_key.get(anchor_key)
            if not positives or not anchor:
                continue
            for _, neg_row in group.head(max_per_anchor).iterrows():
                negative = build_candidate_text(neg_row["Q2"], use_prompts=use_prompts)
                if not negative:
                    continue
                for positive in positives[:2]:
                    triple_key = (anchor, positive, negative)
                    if triple_key in seen:
                        continue
                    seen.add(triple_key)
                    triples.append(HardNegativeExample(anchor=anchor, positive=positive, negative=negative))
                    file_triples += 1
        logger.info("Loaded %d hard-negative triples from %s", file_triples, file_name)
    logger.info("Loaded %d hard-negative triples total", len(triples))
    return triples


def load_targeted_examples(
    targeted_file: str,
    positive_repeat_multiplier: int,
    hard_negative_repeat_multiplier: int,
) -> Tuple[List[InputExample], List[HardNegativeExample], List[ReasonPrototypeExample]]:
    path = Path(targeted_file)
    if not path.exists():
        logger.warning("Targeted fine-tune file not found, skip: %s", targeted_file)
        return [], [], []
    df = read_table(str(path))
    required = {"sample_type", "anchor_text", "positive_text", "repeat_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Targeted file {targeted_file} missing columns: {sorted(missing)}")

    positives: List[InputExample] = []
    hard_negatives: List[HardNegativeExample] = []
    reason_proto_examples: List[ReasonPrototypeExample] = []
    positive_df = df[df["sample_type"].astype(str) == "positive"].copy()
    for row in positive_df.itertuples(index=False):
        anchor = clean_text(getattr(row, "anchor_text", ""))
        positive = clean_text(getattr(row, "positive_text", ""))
        if not anchor or not positive:
            continue
        reason_category = clean_text(getattr(row, "reason_category", ""))
        if not reason_category:
            reason_category, _, _ = infer_reason_category_from_row(row)
        if reason_category in REASON_CATEGORY_PROMPTS:
            prompt_text = REASON_CATEGORY_PROMPTS[reason_category]
            if prompt_text not in anchor:
                anchor = anchor + " " + prompt_text
        for _ in range(repeat_count(getattr(row, "repeat_weight", 1), positive_repeat_multiplier)):
            positives.append(InputExample(texts=[anchor, positive]))

        raw_anchor = build_plain_anchor_text(getattr(row, "Q1", ""), getattr(row, "A1", ""))
        prototype = REASON_CATEGORY_PROMPTS.get(reason_category, "")
        if raw_anchor and prototype:
            reason_proto_examples.append(
                ReasonPrototypeExample(
                    anchor=raw_anchor,
                    prototype=prototype,
                    reason_category=reason_category,
                )
            )

    hard_df = df[df["sample_type"].astype(str) == "hard_negative"].copy()
    if not hard_df.empty and "negative_text" not in hard_df.columns:
        raise ValueError(f"Targeted file {targeted_file} has hard_negative rows but no negative_text column.")
    for row in hard_df.itertuples(index=False):
        anchor = clean_text(getattr(row, "anchor_text", ""))
        positive = clean_text(getattr(row, "positive_text", ""))
        negative = clean_text(getattr(row, "negative_text", ""))
        if not anchor or not positive or not negative:
            continue
        for _ in range(repeat_count(getattr(row, "repeat_weight", 1), hard_negative_repeat_multiplier)):
            hard_negatives.append(HardNegativeExample(anchor=anchor, positive=positive, negative=negative))

    logger.info(
        "Loaded targeted examples from %s: positive_pairs=%d hard_negative_triples=%d reason_proto_pairs=%d",
        targeted_file,
        len(positives),
        len(hard_negatives),
        len(reason_proto_examples),
    )
    if "reason_category" in df.columns:
        logger.info("Targeted reason distribution:\n%s", df["reason_category"].value_counts().to_string())
    return positives, hard_negatives, reason_proto_examples


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def make_unique_anchor_batches(examples: List[InputExample], batch_size: int, seed: int) -> List[List[InputExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    batches: List[List[InputExample]] = []
    remaining = shuffled
    # Multiple positives can share the same anchor. In MNRL, other positives in
    # the same batch are treated as negatives, so each batch must contain unique
    # anchors to avoid false in-batch negatives.
    while remaining:
        current: List[InputExample] = []
        anchors = set()
        next_round: List[InputExample] = []
        for example in remaining:
            anchor = example.texts[0]
            if len(current) < batch_size and anchor not in anchors:
                current.append(example)
                anchors.add(anchor)
            else:
                next_round.append(example)
        if current:
            batches.append(current)
        if len(next_round) == len(remaining):
            # Degenerate guard: should not happen unless batch_size is invalid.
            batches.extend([[example] for example in next_round])
            break
        remaining = next_round
    return batches


def make_hard_negative_batches(
    examples: List[HardNegativeExample],
    batch_size: int,
    seed: int,
) -> List[List[HardNegativeExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def make_reason_proto_batches(
    examples: List[ReasonPrototypeExample],
    batch_size: int,
    seed: int,
) -> List[List[ReasonPrototypeExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    batches: List[List[ReasonPrototypeExample]] = []
    remaining = shuffled
    while remaining:
        current: List[ReasonPrototypeExample] = []
        seen_categories = set()
        next_round: List[ReasonPrototypeExample] = []
        for example in remaining:
            category = example.reason_category
            if len(current) < batch_size and category not in seen_categories:
                current.append(example)
                seen_categories.add(category)
            else:
                next_round.append(example)
        if current:
            batches.append(current)
        if len(next_round) == len(remaining):
            batches.extend([[example] for example in next_round])
            break
        remaining = next_round
    return batches


def make_direction_aux_batches(
    examples: List[DirectionAuxExample],
    batch_size: int,
    seed: int,
) -> List[List[DirectionAuxExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def encode_texts(model: SentenceTransformer, texts: List[str], device: str) -> torch.Tensor:
    features = model.tokenize(texts)
    features = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }
    return model(features)["sentence_embedding"]


class DirectionAuxHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = embedding_dim * 4
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, anchor_emb: torch.Tensor, candidate_emb: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [anchor_emb, candidate_emb, torch.abs(anchor_emb - candidate_emb), anchor_emb * candidate_emb],
            dim=1,
        )
        return self.scorer(features).squeeze(-1)


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
    )


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_one_hard_negative_batch(
    model: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    batch: List[HardNegativeExample],
    hard_negative_weight: float,
    scale: float,
    device: str,
) -> float:
    anchors = [example.anchor for example in batch]
    positives = [example.positive for example in batch]
    negatives = [example.negative for example in batch]
    anchor_emb = F.normalize(encode_texts(model, anchors, device), p=2, dim=1)
    positive_emb = F.normalize(encode_texts(model, positives, device), p=2, dim=1)
    negative_emb = F.normalize(encode_texts(model, negatives, device), p=2, dim=1)
    positive_scores = torch.sum(anchor_emb * positive_emb, dim=1)
    negative_scores = torch.sum(anchor_emb * negative_emb, dim=1)
    logits = torch.stack([positive_scores, negative_scores], dim=1) * scale
    labels = torch.zeros(len(batch), dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels) * hard_negative_weight
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_value = float(loss.item())
    del anchor_emb, positive_emb, negative_emb, positive_scores, negative_scores, logits, labels, loss
    return loss_value


def train_one_reason_proto_batch(
    model: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    batch: List[ReasonPrototypeExample],
    proto_weight: float,
    scale: float,
    device: str,
) -> float:
    anchors = [example.anchor for example in batch]
    prototypes = [example.prototype for example in batch]
    anchor_emb = F.normalize(encode_texts(model, anchors, device), p=2, dim=1)
    proto_emb = F.normalize(encode_texts(model, prototypes, device), p=2, dim=1)
    scores = torch.matmul(anchor_emb, proto_emb.t()) * scale
    labels = torch.arange(len(batch), dtype=torch.long, device=device)
    loss = F.cross_entropy(scores, labels) * proto_weight
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    loss_value = float(loss.item())
    del anchor_emb, proto_emb, scores, labels, loss
    return loss_value


def train_one_direction_aux_batch(
    model: SentenceTransformer,
    aux_head: DirectionAuxHead,
    optimizer: torch.optim.Optimizer,
    batch: List[DirectionAuxExample],
    aux_weight: float,
    margin: float,
    device: str,
) -> float:
    anchors = [example.anchor for example in batch]
    positives = [example.positive for example in batch]
    negatives = [example.negative for example in batch]
    example_weights = torch.tensor([example.weight for example in batch], dtype=torch.float32, device=device)
    anchor_emb = F.normalize(encode_texts(model, anchors, device), p=2, dim=1)
    positive_emb = F.normalize(encode_texts(model, positives, device), p=2, dim=1)
    negative_emb = F.normalize(encode_texts(model, negatives, device), p=2, dim=1)
    positive_score = aux_head(anchor_emb, positive_emb)
    negative_score = aux_head(anchor_emb, negative_emb)
    losses = F.relu(margin - positive_score + negative_score)
    loss = (losses * example_weights).sum() / example_weights.sum().clamp_min(1e-6)
    loss = loss * aux_weight
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_head.parameters()), 1.0)
    optimizer.step()
    loss_value = float(loss.item())
    del anchor_emb, positive_emb, negative_emb, positive_score, negative_score, losses, example_weights, loss
    return loss_value


def train_direction_aux_batch_safely(
    model: SentenceTransformer,
    aux_head: DirectionAuxHead,
    optimizer: torch.optim.Optimizer,
    batch: List[DirectionAuxExample],
    aux_weight: float,
    margin: float,
    device: str,
) -> Tuple[float, int]:
    try:
        return train_one_direction_aux_batch(
            model=model,
            aux_head=aux_head,
            optimizer=optimizer,
            batch=batch,
            aux_weight=aux_weight,
            margin=margin,
            device=device,
        ), 1
    except BaseException as exc:
        if not is_cuda_oom(exc) or len(batch) <= 1:
            raise
        clear_cuda_cache()
        mid = len(batch) // 2
        logger.warning("CUDA OOM on direction-aux batch size=%d; retry with %d + %d", len(batch), mid, len(batch) - mid)
        left_loss, left_steps = train_direction_aux_batch_safely(
            model, aux_head, optimizer, batch[:mid], aux_weight, margin, device
        )
        right_loss, right_steps = train_direction_aux_batch_safely(
            model, aux_head, optimizer, batch[mid:], aux_weight, margin, device
        )
        return left_loss + right_loss, left_steps + right_steps


def train_reason_proto_batch_safely(
    model: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    batch: List[ReasonPrototypeExample],
    proto_weight: float,
    scale: float,
    device: str,
) -> Tuple[float, int]:
    try:
        return train_one_reason_proto_batch(
            model=model,
            optimizer=optimizer,
            batch=batch,
            proto_weight=proto_weight,
            scale=scale,
            device=device,
        ), 1
    except BaseException as exc:
        if not is_cuda_oom(exc) or len(batch) <= 1:
            raise
        clear_cuda_cache()
        mid = len(batch) // 2
        logger.warning("CUDA OOM on reason-proto batch size=%d; retry with %d + %d", len(batch), mid, len(batch) - mid)
        left_loss, left_steps = train_reason_proto_batch_safely(
            model, optimizer, batch[:mid], proto_weight, scale, device
        )
        right_loss, right_steps = train_reason_proto_batch_safely(
            model, optimizer, batch[mid:], proto_weight, scale, device
        )
        return left_loss + right_loss, left_steps + right_steps


def train_hard_negative_batch_safely(
    model: SentenceTransformer,
    optimizer: torch.optim.Optimizer,
    batch: List[HardNegativeExample],
    hard_negative_weight: float,
    scale: float,
    device: str,
) -> Tuple[float, int]:
    try:
        return train_one_hard_negative_batch(
            model=model,
            optimizer=optimizer,
            batch=batch,
            hard_negative_weight=hard_negative_weight,
            scale=scale,
            device=device,
        ), 1
    except BaseException as exc:
        if not is_cuda_oom(exc) or len(batch) <= 1:
            raise
        clear_cuda_cache()
        mid = len(batch) // 2
        logger.warning("CUDA OOM on hard-negative batch size=%d; retry with %d + %d", len(batch), mid, len(batch) - mid)
        left_loss, left_steps = train_hard_negative_batch_safely(
            model, optimizer, batch[:mid], hard_negative_weight, scale, device
        )
        right_loss, right_steps = train_hard_negative_batch_safely(
            model, optimizer, batch[mid:], hard_negative_weight, scale, device
        )
        return left_loss + right_loss, left_steps + right_steps


def train_mnrl_manual(
    model: SentenceTransformer,
    examples: List[InputExample],
    hard_negative_examples: List[HardNegativeExample],
    reason_proto_examples: List[ReasonPrototypeExample],
    direction_aux_examples: List[DirectionAuxExample],
    direction_aux_head: DirectionAuxHead | None,
    epochs: int,
    batch_size: int,
    hard_negative_batch_size: int,
    hard_negative_weight: float,
    reason_proto_batch_size: int,
    reason_proto_weight: float,
    direction_aux_batch_size: int,
    direction_aux_weight: float,
    direction_aux_margin: float,
    learning_rate: float,
    warmup_ratio: float,
    scale: float,
    device: str,
) -> None:
    trainable_params = list(model.parameters())
    if direction_aux_head is not None:
        trainable_params += list(direction_aux_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    mnrl_steps = int(np.ceil(len(examples) / max(batch_size, 1)))
    hard_negative_steps = int(np.ceil(len(hard_negative_examples) / max(hard_negative_batch_size, 1)))
    reason_proto_steps = int(np.ceil(len(reason_proto_examples) / max(reason_proto_batch_size, 1)))
    direction_aux_steps = int(np.ceil(len(direction_aux_examples) / max(direction_aux_batch_size, 1)))
    total_steps = max((mnrl_steps + hard_negative_steps + reason_proto_steps + direction_aux_steps) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        return 1.0

    model.train()
    global_step = 0
    for epoch in range(epochs):
        batches = make_unique_anchor_batches(examples, batch_size, seed=SEED + epoch)
        epoch_loss = 0.0
        epoch_hard_negative_loss = 0.0
        epoch_reason_proto_loss = 0.0
        epoch_direction_aux_loss = 0.0
        epoch_steps = 0
        epoch_hard_negative_steps = 0
        epoch_reason_proto_steps = 0
        epoch_direction_aux_steps = 0
        for batch in batches:
            anchors = [example.texts[0] for example in batch]
            positives = [example.texts[1] for example in batch]
            if len(batch) <= 1:
                continue
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_factor(global_step)
            anchor_emb = encode_texts(model, anchors, device)
            positive_emb = encode_texts(model, positives, device)
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            positive_emb = F.normalize(positive_emb, p=2, dim=1)
            scores = torch.matmul(anchor_emb, positive_emb.t()) * scale
            labels = torch.arange(len(batch), dtype=torch.long, device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_steps += 1
            global_step += 1

        if hard_negative_weight > 0 and hard_negative_examples:
            hard_negative_batches = make_hard_negative_batches(
                hard_negative_examples,
                hard_negative_batch_size,
                seed=SEED * 13 + epoch,
            )
            for batch in hard_negative_batches:
                if not batch:
                    continue
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * lr_factor(global_step)
                loss_sum, step_count = train_hard_negative_batch_safely(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    hard_negative_weight=hard_negative_weight,
                    scale=scale,
                    device=device,
                )
                epoch_hard_negative_loss += loss_sum
                epoch_hard_negative_steps += step_count
                global_step += step_count
        if reason_proto_weight > 0 and reason_proto_examples:
            reason_proto_batches = make_reason_proto_batches(
                reason_proto_examples,
                reason_proto_batch_size,
                seed=SEED * 29 + epoch,
            )
            for batch in reason_proto_batches:
                if not batch:
                    continue
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * lr_factor(global_step)
                loss_sum, step_count = train_reason_proto_batch_safely(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    proto_weight=reason_proto_weight,
                    scale=scale,
                    device=device,
                )
                epoch_reason_proto_loss += loss_sum
                epoch_reason_proto_steps += step_count
                global_step += step_count
        if direction_aux_weight > 0 and direction_aux_examples and direction_aux_head is not None:
            direction_aux_batches = make_direction_aux_batches(
                direction_aux_examples,
                direction_aux_batch_size,
                seed=SEED * 41 + epoch,
            )
            for batch in direction_aux_batches:
                if not batch:
                    continue
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * lr_factor(global_step)
                loss_sum, step_count = train_direction_aux_batch_safely(
                    model=model,
                    aux_head=direction_aux_head,
                    optimizer=optimizer,
                    batch=batch,
                    aux_weight=direction_aux_weight,
                    margin=direction_aux_margin,
                    device=device,
                )
                epoch_direction_aux_loss += loss_sum
                epoch_direction_aux_steps += step_count
                global_step += step_count
        logger.info(
            (
                "Epoch %d/%d | mnrl_batches=%d | mnrl_loss=%.4f | "
                "hardneg_batches=%d | hardneg_loss=%.4f | "
                "reason_proto_batches=%d | reason_proto_loss=%.4f | "
                "continuation_aux_batches=%d | continuation_aux_loss=%.4f | lr=%.2e"
            ),
            epoch + 1,
            epochs,
            epoch_steps,
            epoch_loss / max(epoch_steps, 1),
            epoch_hard_negative_steps,
            epoch_hard_negative_loss / max(epoch_hard_negative_steps, 1),
            epoch_reason_proto_steps,
            epoch_reason_proto_loss / max(epoch_reason_proto_steps, 1),
            epoch_direction_aux_steps,
            epoch_direction_aux_loss / max(epoch_direction_aux_steps, 1),
            optimizer.param_groups[0]["lr"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune m3e-base with MultipleNegativesRankingLoss for FCN Stage 1.")
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--modelscope-cache-dir", default=DEFAULT_MODELSCOPE_CACHE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negative-batch-size", type=int, default=0, help="0 means use --batch-size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=20.0, help="MNRL similarity scale.")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum token length for anchor/candidate encoding.")
    parser.add_argument(
        "--hard-positive-files",
        default=DEFAULT_HARD_POSITIVE_FILES,
        help="Comma-separated CSV files containing missed true edges to repeat as hard positives.",
    )
    parser.add_argument(
        "--hard-positive-repeat",
        type=int,
        default=3,
        help="Repeat each unique hard positive this many times.",
    )
    parser.add_argument(
        "--hard-positive-complaint-repeat",
        type=int,
        default=5,
        help="Repeat missed Complaint edges at least this many times.",
    )
    parser.add_argument(
        "--hard-positive-challenge-repeat",
        type=int,
        default=5,
        help="Repeat missed Challenge edges at least this many times.",
    )
    parser.add_argument(
        "--hard-negative-files",
        default=DEFAULT_HARD_NEGATIVE_FILES,
        help="Comma-separated scored stage output files used to mine high-score false positives.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.0,
        help="Weight for hard-negative contrastive loss. Set 0 to disable.",
    )
    parser.add_argument(
        "--hard-negative-max-per-anchor",
        type=int,
        default=3,
        help="Maximum mined false-positive negatives per anchor from each scored file.",
    )
    parser.add_argument(
        "--hard-negative-min-score",
        type=float,
        default=0.90,
        help="Minimum stage score for mined hard negatives. Use 0.90 for high-score false positives.",
    )
    parser.add_argument(
        "--targeted-file",
        default="",
        help=(
            "Optional targeted fine-tuning CSV. "
            "If provided, positive rows are added to MNRL and hard_negative rows are added to "
            "the optional hard-negative loss."
        ),
    )
    parser.add_argument(
        "--targeted-only",
        action="store_true",
        help="Train only on --targeted-file instead of also loading all positives from --train-file.",
    )
    parser.add_argument("--targeted-positive-repeat-multiplier", type=int, default=1)
    parser.add_argument("--targeted-hard-negative-repeat-multiplier", type=int, default=1)
    parser.add_argument("--reason-proto-weight", type=float, default=0.0, help="Weight for reason-category prototype alignment loss.")
    parser.add_argument("--reason-proto-batch-size", type=int, default=6, help="Batch size for reason-category prototype alignment.")
    parser.add_argument("--direction-aux-weight", type=float, default=0.15, help="Weight for continuation compatibility ranking loss.")
    parser.add_argument("--direction-aux-batch-size", type=int, default=32, help="Batch size for continuation compatibility ranking.")
    parser.add_argument("--direction-aux-negative-ratio", type=float, default=6.0, help="Maximum hard negatives per dependent example for the continuation auxiliary task.")
    parser.add_argument("--direction-aux-margin", type=float, default=0.25, help="Ranking margin for continuation compatibility.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-followup-prompts", action="store_true", help="Disable asymmetric follow-up retrieval prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    use_prompts = not args.no_followup_prompts

    examples: List[InputExample] = []
    hard_positive_examples: List[InputExample] = []
    targeted_positive_examples: List[InputExample] = []
    targeted_hard_negative_examples: List[HardNegativeExample] = []
    targeted_reason_proto_examples: List[ReasonPrototypeExample] = []
    direction_aux_examples: List[DirectionAuxExample] = []
    if not args.targeted_only:
        examples = load_positive_examples(args.train_file, use_prompts=use_prompts)
        hard_positive_examples = load_hard_positive_examples(
            parse_file_list(args.hard_positive_files),
            use_prompts=use_prompts,
            repeat=args.hard_positive_repeat,
            complaint_repeat=args.hard_positive_complaint_repeat,
            challenge_repeat=args.hard_positive_challenge_repeat,
        )
        examples.extend(hard_positive_examples)
    if args.targeted_file:
        (
            targeted_positive_examples,
            targeted_hard_negative_examples,
            targeted_reason_proto_examples,
        ) = load_targeted_examples(
            targeted_file=args.targeted_file,
            positive_repeat_multiplier=args.targeted_positive_repeat_multiplier,
            hard_negative_repeat_multiplier=args.targeted_hard_negative_repeat_multiplier,
        )
        examples.extend(targeted_positive_examples)
    if not examples:
        raise RuntimeError("No positive examples found for MNRL fine-tuning.")

    if args.direction_aux_weight > 0:
        direction_aux_examples = load_direction_aux_examples(
            train_file=args.train_file,
            use_prompts=use_prompts,
            negative_ratio=args.direction_aux_negative_ratio,
            seed=SEED,
        )

    hard_negative_batch_size = args.hard_negative_batch_size or args.batch_size
    hard_negative_examples: List[HardNegativeExample] = []
    if args.hard_negative_weight > 0:
        if not args.targeted_only:
            hard_negative_examples = load_hard_negative_examples(
                train_file=args.train_file,
                hard_negative_files=parse_file_list(args.hard_negative_files),
                use_prompts=use_prompts,
                max_per_anchor=args.hard_negative_max_per_anchor,
                min_score=args.hard_negative_min_score,
            )
        hard_negative_examples.extend(targeted_hard_negative_examples)

    model_path = resolve_model_path(args.base_model, cache_dir=args.modelscope_cache_dir)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logger.info("Loading base encoder from %s on %s", model_path, device)
    model = SentenceTransformer(model_path, device=device)
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
        logger.info("Set SentenceTransformer max_seq_length=%d", model.max_seq_length)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "Start manual MNRL fine-tuning: pairs=%d hard_positive_pairs=%d "
            "hard_negative_triples=%d batch_size=%d hardneg_batch_size=%d "
            "direction_aux_examples=%d direction_aux_weight=%.2f direction_aux_batch_size=%d "
            "epochs=%d lr=%.2e warmup_ratio=%.2f prompts=%s"
        ),
        len(examples),
        len(hard_positive_examples),
        len(hard_negative_examples),
        args.batch_size,
        hard_negative_batch_size,
        len(direction_aux_examples),
        args.direction_aux_weight,
        args.direction_aux_batch_size,
        args.epochs,
        args.learning_rate,
        args.warmup_ratio,
        use_prompts,
    )
    train_mnrl_manual(
        model=model,
        examples=examples,
        hard_negative_examples=hard_negative_examples,
        reason_proto_examples=targeted_reason_proto_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hard_negative_batch_size=hard_negative_batch_size,
        hard_negative_weight=args.hard_negative_weight,
        reason_proto_batch_size=args.reason_proto_batch_size,
        reason_proto_weight=args.reason_proto_weight,
        direction_aux_batch_size=args.direction_aux_batch_size,
        direction_aux_weight=args.direction_aux_weight,
        direction_aux_examples=direction_aux_examples,
        direction_aux_head=DirectionAuxHead(
            embedding_dim=model.get_sentence_embedding_dimension(),
        ).to(device) if args.direction_aux_weight > 0 else None,
        direction_aux_margin=args.direction_aux_margin,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        scale=args.scale,
        device=device,
    )
    model.save(str(output_path))

    with open(output_path / "fcn_finetune_readme.txt", "w", encoding="utf-8") as file:
        file.write(
            "Fine-tuned with MultipleNegativesRankingLoss for follow-up retrieval.\n"
            f"base_model={args.base_model}\n"
            f"train_file={args.train_file}\n"
            f"epochs={args.epochs}\n"
            f"batch_size={args.batch_size}\n"
            f"hard_negative_batch_size={hard_negative_batch_size}\n"
            f"learning_rate={args.learning_rate}\n"
            f"max_seq_length={args.max_seq_length}\n"
            f"use_followup_prompts={use_prompts}\n"
            f"hard_positive_files={args.hard_positive_files}\n"
            f"hard_positive_repeat={args.hard_positive_repeat}\n"
            f"hard_positive_complaint_repeat={args.hard_positive_complaint_repeat}\n"
            f"hard_positive_challenge_repeat={args.hard_positive_challenge_repeat}\n"
            f"hard_positive_pairs={len(hard_positive_examples)}\n"
            f"targeted_file={args.targeted_file}\n"
            f"targeted_only={args.targeted_only}\n"
            f"targeted_positive_pairs={len(targeted_positive_examples)}\n"
            f"targeted_hard_negative_triples={len(targeted_hard_negative_examples)}\n"
            f"targeted_reason_proto_pairs={len(targeted_reason_proto_examples)}\n"
            f"targeted_positive_repeat_multiplier={args.targeted_positive_repeat_multiplier}\n"
            f"targeted_hard_negative_repeat_multiplier={args.targeted_hard_negative_repeat_multiplier}\n"
            f"reason_proto_weight={args.reason_proto_weight}\n"
            f"reason_proto_batch_size={args.reason_proto_batch_size}\n"
            f"direction_aux_weight={args.direction_aux_weight}\n"
            f"direction_aux_batch_size={args.direction_aux_batch_size}\n"
            f"direction_aux_negative_ratio={args.direction_aux_negative_ratio}\n"
            f"direction_aux_margin={args.direction_aux_margin}\n"
            f"direction_aux_examples={len(direction_aux_examples)}\n"
            f"hard_negative_files={args.hard_negative_files}\n"
            f"hard_negative_weight={args.hard_negative_weight}\n"
            f"hard_negative_max_per_anchor={args.hard_negative_max_per_anchor}\n"
            f"hard_negative_min_score={args.hard_negative_min_score}\n"
            f"hard_negative_triples={len(hard_negative_examples)}\n"
        )
    logger.info("Fine-tuned model saved to: %s", output_path)


if __name__ == "__main__":
    main()
