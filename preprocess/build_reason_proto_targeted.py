# -*- coding: utf-8 -*-
"""Build a targeted dataset for semantic prototype distillation.

This script is a preprocessing step for Stage 1 representation learning. It
reads the FCN labeled follow-up dataset, extracts a compact reason category
from the explanation text (preferring `thought_process` when available and
falling back to `dep_reason`), and writes a targeted file for
`finetune_m3e_mnrl.py`.

By default the output keeps positive rows only. Optional flags can add
same-category hard negatives and seed-FP pattern-matched hard negatives, so
the encoder learns boundary cases instead of memorizing a prompt template. The
generated file stores raw text fields by default, leaving prompt scaffolding to
the training script.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
import random
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from followup_semantics import (
    REASON_CATEGORY_PATTERNS,
    REASON_CATEGORY_PROMPTS,
    clean_text,
    infer_reason_category,
    infer_reason_category_from_row,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
LOGGER = logging.getLogger("reason-proto-targeted")

DEFAULT_INPUT_FILE = str(PROJECT_DIR / "train data" / "fcn_30firms_full_labeled.csv")
DEFAULT_OUTPUT_FILE = str(PROJECT_DIR / "train data" / "targeted_reason_proto_augmented.csv")
SEED = 42

def read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def dependent_yes_mask(df: pd.DataFrame) -> pd.Series:
    if "dependent" in df.columns:
        return df["dependent"].astype(str).str.lower().isin({"yes", "1", "true"})
    if "edge_label" in df.columns:
        return pd.to_numeric(df["edge_label"], errors="coerce").fillna(0).astype(int) == 1
    raise ValueError("Input table must contain either dependent or edge_label.")


def build_anchor_text(q1: object, a1: object, reason_category: str) -> str:
    q1_text = clean_text(q1)
    a1_text = clean_text(a1)
    reason_text = REASON_CATEGORY_PROMPTS.get(reason_category, "")
    return q1_text + " " + a1_text + (" " + reason_text if reason_text else "")


def build_positive_text(q2: object) -> str:
    return clean_text(q2)


def build_hard_negative_text(q2: object) -> str:
    return clean_text(q2)


def build_pair_text(row: object) -> str:
    q1_text = clean_text(getattr(row, "Q1", ""))
    a1_text = clean_text(getattr(row, "A1", ""))
    q2_text = clean_text(getattr(row, "Q2", ""))
    return " ".join(part for part in [q1_text, a1_text, q2_text] if part)


def infer_fp_pattern_category_from_row(row: object) -> str:
    q2_text = clean_text(getattr(row, "Q2", ""))
    category = infer_reason_category(q2_text)
    if category:
        return category
    qa_text = clean_text(getattr(row, "Q1", "")) + " " + clean_text(getattr(row, "A1", ""))
    category = infer_reason_category(qa_text)
    return category or "UNMAPPED"


def negative_priority_key(row: object, reason_category: str) -> Tuple[int, int, int]:
    """Rank hard negatives by locality and textual challenge strength."""
    session_id = clean_text(getattr(row, "session_id", ""))
    q1_row = clean_text(getattr(row, "Q1_row", ""))
    local_hit = 1 if session_id else 0
    anchor_hit = 1 if q1_row else 0
    q2_text = clean_text(getattr(row, "Q2", ""))
    strength = 0
    if reason_category == "answer_pressure":
        strength += 2
    if any(token in q2_text for token in ("为什么", "为何", "请回答", "正面回答", "明确回答")):
        strength += 1
    if any(token in q2_text for token in ("是否", "能否", "如何", "具体", "详细")):
        strength += 1
    return (local_hit, anchor_hit, strength)


def default_repeat_weight(reason_category: str) -> int:
    # Keep repetition conservative; the user asked not to introduce lots of
    # negatives, so we keep positives balanced and only slightly upweight the
    # rarest category.
    if reason_category == "shareholder_return":
        return 2
    if reason_category in {"restructuring", "strategy_planning", "performance_governance"}:
        return 2
    if reason_category == "answer_pressure":
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a targeted semantic-prototype dataset from LLM explanations. Positive rows are kept by default."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--add-same-category-hard-negatives",
        action="store_true",
        help="Add same-category hard negatives by pairing each positive anchor with a negative Q2 from the same reason category.",
    )
    parser.add_argument(
        "--same-category-hard-negative-k",
        type=int,
        default=1,
        help="Maximum same-category hard negatives to sample per positive row.",
    )
    parser.add_argument(
        "--add-seed-fp-pattern-hard-negatives",
        action="store_true",
        help="Mine additional hard negatives using high-score false-positive seed patterns from a scored Stage 1 file.",
    )
    parser.add_argument(
        "--seed-fp-file",
        default="",
        help="Scored Stage 1 CSV used as the seed pattern source for additional hard negatives.",
    )
    parser.add_argument(
        "--seed-fp-min-score",
        type=float,
        default=0.90,
        help="Minimum stage score for seed false-positive rows when mining pattern-based hard negatives.",
    )
    parser.add_argument(
        "--seed-fp-hard-negative-k",
        type=int,
        default=1,
        help="Maximum pattern-matched hard negatives to sample per positive row.",
    )
    return parser.parse_args()


def load_seed_fp_rows(seed_fp_file: str, min_score: float) -> pd.DataFrame:
    path = Path(seed_fp_file)
    if not path.exists():
        raise FileNotFoundError(f"Seed FP file not found: {seed_fp_file}")
    df = read_table(str(path))
    required = {"Q1", "A1", "Q2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Seed FP file missing columns: {sorted(missing)}")
    if "edge_label" in df.columns:
        df = df[pd.to_numeric(df["edge_label"], errors="coerce").fillna(0).astype(int) == 0].copy()
    if "dependent" in df.columns:
        df = df[df["dependent"].astype(str).str.lower().isin({"no", "0", "false"})].copy()
    if "stage1_pred_edge" in df.columns:
        df = df[pd.to_numeric(df["stage1_pred_edge"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "stage1_prob" in df.columns:
        df["stage1_prob"] = pd.to_numeric(df["stage1_prob"], errors="coerce").fillna(0.0)
        df = df[df["stage1_prob"] >= min_score].copy()
    for col in ["Q1", "A1", "Q2"]:
        df[col] = df[col].apply(clean_text)
    df = df.reset_index(drop=True)
    return df


def mine_pattern_matched_hard_negatives(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    seed_fp_df: pd.DataFrame,
    k_per_positive: int,
) -> List[Dict[str, object]]:
    if seed_fp_df.empty or neg_df.empty or pos_df.empty or k_per_positive <= 0:
        return []

    # Build seed pattern centroids by coarse reason category inferred from the
    # high-score false positives' own question text.
    seed_df = seed_fp_df.copy()
    seed_df["_seed_category"] = seed_df.apply(infer_fp_pattern_category_from_row, axis=1)
    seed_df["_seed_pair_text"] = seed_df.apply(build_pair_text, axis=1)
    seed_categories = [cat for cat in seed_df["_seed_category"].dropna().astype(str).unique().tolist() if cat]
    if not seed_categories:
        return []

    seed_centroids: Dict[str, str] = {}
    for category in seed_categories:
        texts = seed_df.loc[seed_df["_seed_category"] == category, "_seed_pair_text"].tolist()
        texts = [text for text in texts if clean_text(text)]
        if texts:
            seed_centroids[category] = " ".join(texts)
    if not seed_centroids:
        return []

    neg_work = neg_df.copy().reset_index(drop=True)
    neg_work["_neg_category"] = neg_work.apply(
        lambda row: infer_reason_category(build_hard_negative_text(getattr(row, "Q2", "")))
        or infer_reason_category(build_pair_text(row))
        or "UNMAPPED",
        axis=1,
    )
    neg_work["_neg_pair_text"] = neg_work.apply(build_pair_text, axis=1)

    # Use character n-grams so the matcher can react to company names, issue
    # phrases, and recurring complaint / governance templates without requiring
    # a heavy embedding model.
    corpus = neg_work["_neg_pair_text"].tolist() + list(seed_centroids.values())
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=50000)
    tfidf = vectorizer.fit_transform(corpus)
    neg_matrix = tfidf[: len(neg_work)]
    centroid_matrix = tfidf[len(neg_work) :]

    centroid_index = {category: idx for idx, category in enumerate(seed_centroids.keys())}
    centroid_scores: Dict[str, np.ndarray] = {}
    for category, idx in centroid_index.items():
        centroid_vec = centroid_matrix[idx].toarray().ravel()
        norm = np.linalg.norm(centroid_vec)
        if norm > 0:
            centroid_vec = centroid_vec / norm
        scores = np.asarray(neg_matrix @ centroid_vec).ravel()
        centroid_scores[category] = scores

    # Fallback centroid for categories without their own seed patterns.
    fallback_centroid_vec = centroid_matrix.mean(axis=0).A1 if centroid_matrix.shape[0] > 0 else np.array([])
    fallback_norm = np.linalg.norm(fallback_centroid_vec)
    if fallback_norm > 0:
        fallback_centroid_vec = fallback_centroid_vec / fallback_norm
    fallback_scores = np.asarray(neg_matrix @ fallback_centroid_vec).ravel() if fallback_centroid_vec.size else np.zeros(len(neg_work))

    records: List[Dict[str, object]] = []
    seen_triples = set()
    pos_rows = pos_df.copy().reset_index(drop=True)
    pos_rows["_reason_category"] = pos_rows.apply(lambda row: infer_reason_category_from_row(row)[0], axis=1)
    pos_rows["_anchor_text"] = pos_rows.apply(
        lambda row: build_anchor_text(
            getattr(row, "Q1", ""),
            getattr(row, "A1", ""),
            reason_category=infer_reason_category_from_row(row)[0],
        ),
        axis=1,
    )
    pos_rows["_positive_text"] = pos_rows.apply(lambda row: build_positive_text(getattr(row, "Q2", "")), axis=1)

    for _, pos_row in pos_rows.iterrows():
        category = clean_text(pos_row["_reason_category"]) or "UNMAPPED"
        anchor = clean_text(pos_row["_anchor_text"])
        positive = clean_text(pos_row["_positive_text"])
        if not anchor or not positive:
            continue
        if category in centroid_scores:
            scores = centroid_scores[category].copy()
        else:
            scores = fallback_scores.copy()

        # Prefer negatives from the same coarse category as the seed pattern,
        # but keep a fallback path if the category is sparse.
        if category != "UNMAPPED":
            cat_mask = neg_work["_neg_category"].astype(str).eq(category)
            if cat_mask.any():
                boosted_scores = scores.copy()
                boosted_scores[cat_mask.to_numpy()] += 0.10
                scores = boosted_scores

        same_scode = neg_work["Scode"].astype(str).eq(str(pos_row.get("Scode", ""))).to_numpy() if "Scode" in neg_work.columns else None
        same_year = neg_work["Year"].astype(str).eq(str(pos_row.get("Year", ""))).to_numpy() if "Year" in neg_work.columns else None
        if same_scode is not None:
            scores = scores + (0.08 * same_scode.astype(float))
        if same_year is not None:
            scores = scores + (0.03 * same_year.astype(float))

        order = np.argsort(scores)[::-1]
        sampled = 0
        local_seen = set()
        for idx in order:
            if sampled >= max(k_per_positive, 0):
                break
            negative_text = clean_text(neg_work.iloc[idx]["Q2"])
            if not negative_text or negative_text in local_seen:
                continue
            local_seen.add(negative_text)
            triple = (anchor, positive, negative_text)
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
            seed_cat = clean_text(neg_work.iloc[idx]["_neg_category"]) or "UNMAPPED"
            records.append(
                {
                    "sample_type": "hard_negative",
                    "anchor_text": anchor,
                    "positive_text": positive,
                    "negative_text": negative_text,
                    "repeat_weight": 1,
                    "reason_category": category,
                    "positive_reason_category": category,
                    "seed_fp_match_category": seed_cat,
                    "seed_fp_match_score": float(scores[idx]),
                    "Q1": clean_text(pos_row.get("Q1", "")),
                    "A1": clean_text(pos_row.get("A1", "")),
                    "Q2": clean_text(pos_row.get("Q2", "")),
                    "session_id": pos_row.get("session_id", ""),
                    "Q1_row": pos_row.get("Q1_row", ""),
                    "Q2_row": pos_row.get("Q2_row", ""),
                    "distance": pos_row.get("distance", ""),
                    "relation_type": pos_row.get("relation_type", ""),
                    "dep_reason": pos_row.get("dep_reason", ""),
                    "seed_fp_source_count": int((seed_df["_seed_category"].astype(str) == category).sum()),
                }
            )
            sampled += 1

    LOGGER.info(
        "Mined %d pattern-matched hard negatives from %d seed false-positive rows across %d categories",
        len(records),
        len(seed_df),
        len(seed_centroids),
    )
    if records:
        out_df = pd.DataFrame(records)
        LOGGER.info("Pattern-matched hard negative distribution:\n%s", out_df["positive_reason_category"].value_counts().to_string())
    return records


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    raw_df = read_table(str(input_path))
    required = {"Q1", "A1", "Q2"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {sorted(missing)}")
    pos_df = raw_df[dependent_yes_mask(raw_df)].copy()

    records: List[Dict[str, object]] = []
    stats: Dict[str, int] = {k: 0 for k in REASON_CATEGORY_PROMPTS}
    unmapped = 0
    negatives_by_category: DefaultDict[str, List[object]] = defaultdict(list)
    negatives_by_category_session: DefaultDict[Tuple[str, str], List[object]] = defaultdict(list)
    neg_pool = raw_df[~dependent_yes_mask(raw_df)].copy()

    if args.add_same_category_hard_negatives:
        for row in neg_pool.itertuples(index=False):
            reason_category, _, _ = infer_reason_category_from_row(row)
            if reason_category:
                negatives_by_category[reason_category].append(row)
                negatives_by_category_session[(reason_category, clean_text(getattr(row, "session_id", "")))].append(row)
        rng = random.Random(SEED)

    seed_fp_df = pd.DataFrame()
    if args.add_seed_fp_pattern_hard_negatives:
        if not args.seed_fp_file:
            raise ValueError("--add-seed-fp-pattern-hard-negatives requires --seed-fp-file")
        seed_fp_df = load_seed_fp_rows(args.seed_fp_file, args.seed_fp_min_score)
    for row in pos_df.itertuples(index=False):
        reason_category, source_field, source_text = infer_reason_category_from_row(row)
        if not reason_category:
            continue
        anchor_text = build_anchor_text(getattr(row, "Q1", ""), getattr(row, "A1", ""), reason_category)
        positive_text = build_positive_text(getattr(row, "Q2", ""))
        if not clean_text(anchor_text) or not clean_text(positive_text):
            continue
        repeat_weight = default_repeat_weight(reason_category)
        if reason_category in stats:
            stats[reason_category] += 1
        else:
            unmapped += 1
        records.append(
            {
                "sample_type": "positive",
                "anchor_text": anchor_text,
                "positive_text": positive_text,
                "repeat_weight": repeat_weight,
                "positive_reason_category": reason_category,
                "reason_source_field": source_field,
                "reason_source_text": source_text,
                "Q1": clean_text(getattr(row, "Q1", "")),
                "A1": clean_text(getattr(row, "A1", "")),
                "Q2": clean_text(getattr(row, "Q2", "")),
                "session_id": getattr(row, "session_id", ""),
                "Q1_row": getattr(row, "Q1_row", ""),
                "Q2_row": getattr(row, "Q2_row", ""),
                "distance": getattr(row, "distance", ""),
                "relation_type": getattr(row, "relation_type", ""),
                "dep_reason": getattr(row, "dep_reason", ""),
                "thought_process": getattr(row, "thought_process", ""),
                "label": getattr(row, "label", ""),
                "done": getattr(row, "done", ""),
            }
        )
        if args.add_same_category_hard_negatives and reason_category:
            session_id = clean_text(getattr(row, "session_id", ""))
            candidate_pools = []
            if session_id:
                candidate_pools.append(negatives_by_category_session.get((reason_category, session_id), []))
            candidate_pools.append(negatives_by_category.get(reason_category, []))
            seen_negative_texts = set()
            sampled_count = 0
            ordered_candidates = []
            for pool in candidate_pools:
                if not pool:
                    continue
                shuffled_pool = pool[:]
                rng.shuffle(shuffled_pool)
                ordered_candidates.extend(sorted(shuffled_pool, key=lambda r: negative_priority_key(r, reason_category), reverse=True))
            for neg_row in ordered_candidates:
                if sampled_count >= max(args.same_category_hard_negative_k, 0):
                    break
                negative_text = build_hard_negative_text(getattr(neg_row, "Q2", ""))
                negative_text = clean_text(negative_text)
                if not negative_text or negative_text in seen_negative_texts:
                    continue
                seen_negative_texts.add(negative_text)
                neg_reason_category, neg_source_field, neg_source_text = infer_reason_category_from_row(neg_row)
                records.append(
                    {
                        "sample_type": "hard_negative",
                        "anchor_text": anchor_text,
                        "positive_text": positive_text,
                        "negative_text": negative_text,
                        "repeat_weight": 1,
                        "positive_reason_category": reason_category,
                        "reason_source_field": source_field,
                        "reason_source_text": source_text,
                        "negative_reason_category": neg_reason_category,
                        "negative_source_field": neg_source_field,
                        "negative_source_text": neg_source_text,
                        "negative_session_id": getattr(neg_row, "session_id", ""),
                        "negative_Q1_row": getattr(neg_row, "Q1_row", ""),
                        "negative_distance": getattr(neg_row, "distance", ""),
                        "negative_relation_type": getattr(neg_row, "relation_type", ""),
                        "Q1": clean_text(getattr(row, "Q1", "")),
                        "A1": clean_text(getattr(row, "A1", "")),
                        "Q2": clean_text(getattr(row, "Q2", "")),
                        "session_id": getattr(row, "session_id", ""),
                        "Q1_row": getattr(row, "Q1_row", ""),
                        "Q2_row": getattr(row, "Q2_row", ""),
                        "distance": getattr(row, "distance", ""),
                        "relation_type": getattr(row, "relation_type", ""),
                        "dep_reason": getattr(row, "dep_reason", ""),
                        "thought_process": getattr(row, "thought_process", ""),
                        "label": getattr(row, "label", ""),
                        "done": getattr(row, "done", ""),
                    }
                )
                sampled_count += 1

    if args.add_seed_fp_pattern_hard_negatives and not seed_fp_df.empty:
        pattern_records = mine_pattern_matched_hard_negatives(
            pos_df=pos_df,
            neg_df=neg_pool,
            seed_fp_df=seed_fp_df,
            k_per_positive=args.seed_fp_hard_negative_k,
        )
        records.extend(pattern_records)

    out_df = pd.DataFrame(records)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Wrote targeted positive file to %s", out_path)
    LOGGER.info("Rows: %d", len(out_df))
    if not out_df.empty:
        LOGGER.info("Reason distribution:\n%s", out_df["reason_category"].value_counts(dropna=False).to_string())
        LOGGER.info("Repeat weight distribution:\n%s", out_df["repeat_weight"].value_counts(dropna=False).to_string())
    LOGGER.info("Unmapped positives kept=%d", unmapped)
    LOGGER.info("Category counts (before repeat): %s", stats)


if __name__ == "__main__":
    main()
