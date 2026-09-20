# -*- coding: utf-8 -*-
"""Train the formal FCN retrieval encoder with Train-only positive-pair MNRL.

The script deliberately exposes no rationale, targeted, hard-negative, or
auxiliary-loss path.  A fixed session manifest is mandatory; only sessions
marked ``train`` can contribute labeled pairs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import logging
import os
import random
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from sentence_transformers import InputExample, SentenceTransformer

LOGGER = logging.getLogger("fcn_encoder_training")
EXPECTED_SESSION_COUNTS = {"train": 176, "validation": 44, "test": 55}
INPUT_FORMAT = "raw_q1_a1__q2_v1"


def clean_text(value: object) -> str:
    """Convert a value to normalized, single-space text."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return " ".join(str(value).split())


def build_anchor_text(q1: object, a1: object) -> str:
    """Build the encoder input for a candidate source turn."""
    return f"前序问题：{clean_text(q1)} 管理层回答：{clean_text(a1)}"


def build_candidate_text(q2: object) -> str:
    """Build the encoder input for a later query."""
    return f"当前追问：{clean_text(q2)}"


def text_protocol_fingerprint() -> str:
    """Hash the executable text-normalization/template contract."""
    normalized_functions = []
    for function in (clean_text, build_anchor_text, build_candidate_text):
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        # Documentation changes do not alter the executable protocol.
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        normalized_functions.append(ast.dump(node, include_attributes=False))
    payload = "\n".join(normalized_functions).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


TEXT_PROTOCOL_SHA256 = text_protocol_fingerprint()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the formal FCN encoder using Train-only MNRL."
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--session-manifest", required=True)
    parser.add_argument("--base-model", default="AI-ModelScope/m3e-base")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--modelscope-cache-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision when available (default: enabled).",
    )
    return parser.parse_args()


def require_file(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_split(value: object) -> str:
    split = clean_text(value).lower()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    return aliases.get(split, split)


def load_session_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, encoding="utf-8-sig", dtype={"session_id": str})
    required = {"session_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Session manifest missing columns: {sorted(missing)}")

    manifest = manifest.loc[:, ["session_id", "split"]].copy()
    manifest["session_id"] = manifest["session_id"].map(clean_text)
    manifest["split"] = manifest["split"].map(normalize_split)
    if (manifest["session_id"] == "").any():
        raise ValueError("Session manifest contains an empty session_id.")
    duplicates = manifest.loc[manifest["session_id"].duplicated(False), "session_id"].unique()
    if len(duplicates):
        raise ValueError(f"Session manifest contains duplicate session_id values: {duplicates[:10].tolist()}")

    allowed = set(EXPECTED_SESSION_COUNTS)
    observed_splits = set(manifest["split"])
    if observed_splits != allowed:
        raise ValueError(
            f"Manifest splits must be exactly {sorted(allowed)}; observed {sorted(observed_splits)}."
        )
    counts = manifest.groupby("split")["session_id"].nunique().to_dict()
    for split, expected in EXPECTED_SESSION_COUNTS.items():
        observed = int(counts.get(split, 0))
        if observed != expected:
            raise ValueError(
                f"Manifest {split} session count is {observed}; expected {expected}."
            )
    return manifest


def positive_mask(frame: pd.DataFrame) -> pd.Series:
    if "dependent" in frame.columns:
        return frame["dependent"].astype(str).str.strip().str.lower().isin(
            {"yes", "1", "true"}
        )
    if "edge_label" in frame.columns:
        labels = pd.to_numeric(frame["edge_label"], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("edge_label must contain only 0/1 values.")
        return labels.astype(int).eq(1)
    raise ValueError("Training data must contain dependent or edge_label.")


def load_train_examples(
    train_file: Path, manifest: pd.DataFrame
) -> tuple[list[InputExample], dict[str, int]]:
    frame = pd.read_csv(train_file, encoding="utf-8-sig", dtype={"session_id": str})
    required = {"session_id", "Q1", "A1", "Q2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["session_id"] = frame["session_id"].map(clean_text)
    if (frame["session_id"] == "").any():
        raise ValueError("Training data contains an empty session_id.")

    manifest_ids = set(manifest["session_id"])
    data_ids = set(frame["session_id"])
    unknown = sorted(data_ids - manifest_ids)
    absent = sorted(manifest_ids - data_ids)
    if unknown:
        raise ValueError(f"Training data contains sessions absent from manifest: {unknown[:10]}")
    if absent:
        raise ValueError(f"Manifest contains sessions absent from training data: {absent[:10]}")

    split_by_session = manifest.set_index("session_id")["split"]
    frame["_split"] = frame["session_id"].map(split_by_session)
    train = frame.loc[frame["_split"].eq("train")].copy()
    if set(train["session_id"]) & set(manifest.loc[manifest["split"] != "train", "session_id"]):
        raise AssertionError("Validation/Test session reached the encoder training subset.")

    positives = train.loc[positive_mask(train)].copy()
    examples: list[InputExample] = []
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    for row in positives.itertuples(index=False):
        anchor = build_anchor_text(getattr(row, "Q1"), getattr(row, "A1"))
        candidate = build_candidate_text(getattr(row, "Q2"))
        if not clean_text(getattr(row, "Q1")) or not clean_text(getattr(row, "A1")) or not clean_text(getattr(row, "Q2")):
            continue
        key = (anchor, candidate)
        if key in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(key)
        examples.append(InputExample(texts=[anchor, candidate]))

    if len(examples) < 2:
        raise RuntimeError("Fewer than two unique Train positive pairs were found.")
    stats = {
        "all_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "train_positive_rows": int(len(positives)),
        "unique_train_positive_pairs": int(len(examples)),
        "duplicate_positive_pairs_removed": int(duplicate_pairs),
    }
    return examples, stats


def make_unique_anchor_batches(
    examples: list[InputExample], batch_size: int, seed: int
) -> list[list[InputExample]]:
    if batch_size < 2:
        raise ValueError("batch-size must be at least 2 for MNRL.")
    remaining = examples[:]
    random.Random(seed).shuffle(remaining)
    batches: list[list[InputExample]] = []
    while remaining:
        batch: list[InputExample] = []
        anchors: set[str] = set()
        deferred: list[InputExample] = []
        for example in remaining:
            anchor = example.texts[0]
            if len(batch) < batch_size and anchor not in anchors:
                batch.append(example)
                anchors.add(anchor)
            else:
                deferred.append(example)
        if len(batch) >= 2:
            batches.append(batch)
        elif batch:
            LOGGER.warning("Dropping final singleton MNRL batch.")
        if len(deferred) == len(remaining):
            raise RuntimeError("Unable to construct unique-anchor MNRL batches.")
        remaining = deferred
    return batches


def encode(model: SentenceTransformer, texts: Iterable[str], device: str) -> torch.Tensor:
    features = model.tokenize(list(texts))
    features = {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}
    return model(features)["sentence_embedding"]


def train_mnrl(
    model: SentenceTransformer,
    examples: list[InputExample],
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batches_per_epoch = make_unique_anchor_batches(examples, args.batch_size, args.seed)
    total_steps = max(len(batches_per_epoch) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(warmup_steps, 1), 1.0) if warmup_steps else 1.0,
    )
    amp_enabled = bool(args.amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history: list[dict[str, float | int]] = []
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        batches = make_unique_anchor_batches(examples, args.batch_size, args.seed + epoch)
        loss_sum = 0.0
        for batch in batches:
            anchors = [example.texts[0] for example in batch]
            positives = [example.texts[1] for example in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                anchor_embeddings = F.normalize(encode(model, anchors, device), p=2, dim=1)
                positive_embeddings = F.normalize(encode(model, positives, device), p=2, dim=1)
                logits = torch.matmul(anchor_embeddings, positive_embeddings.T) * args.scale
                labels = torch.arange(len(batch), device=device)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            loss_sum += float(loss.detach().cpu())
        epoch_record = {
            "epoch": epoch + 1,
            "batches": len(batches),
            "mean_mnrl_loss": loss_sum / max(len(batches), 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        LOGGER.info("Epoch %d/%d: batches=%d mean_loss=%.6f", epoch + 1, args.epochs, len(batches), epoch_record["mean_mnrl_loss"])
    return {"global_steps": global_step, "amp_enabled": amp_enabled, "history": history}


def resolve_model_path(model_id_or_path: str, cache_dir: str) -> str:
    local = Path(model_id_or_path).expanduser()
    if local.exists():
        return str(local.resolve())
    return snapshot_download(model_id_or_path, cache_dir=cache_dir)


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return requested


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1).")
    set_seed(args.seed)

    train_file = require_file(args.train_file, "Training file")
    session_manifest_file = require_file(args.session_manifest, "Session manifest")
    manifest = load_session_manifest(session_manifest_file)
    examples, data_stats = load_train_examples(train_file, manifest)
    device = select_device(args.device)
    resolved_base_model = resolve_model_path(args.base_model, args.modelscope_cache_dir)
    LOGGER.info("Loading %s on %s", resolved_base_model, device)
    model = SentenceTransformer(resolved_base_model, device=device)
    model.max_seq_length = args.max_seq_length
    training_stats = train_mnrl(model, examples, args, device)

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    split_counts = manifest.groupby("split")["session_id"].nunique().astype(int).to_dict()
    provenance = {
        "formal_fcn_encoder": True,
        "training_objective": "positive_pair_mnrl_only",
        "input_format": INPUT_FORMAT,
        "text_protocol_sha256": TEXT_PROTOCOL_SHA256,
        "rationale_derived_supervision": False,
        "targeted_supervision": False,
        "hard_negative_supervision": False,
        "direction_auxiliary_supervision": False,
        "train_only": True,
        "seed": args.seed,
        "train_file": str(train_file),
        "train_file_sha256": sha256_file(train_file),
        "session_manifest": str(session_manifest_file),
        "session_manifest_sha256": sha256_file(session_manifest_file),
        "session_counts": split_counts,
        "base_model": args.base_model,
        "resolved_base_model": resolved_base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "scale": args.scale,
        "max_seq_length": args.max_seq_length,
        "device": device,
        "data": data_stats,
        "training": training_stats,
    }
    provenance_path = output_path / "encoder_provenance_manifest.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved encoder to %s", output_path)
    LOGGER.info("Saved provenance manifest to %s", provenance_path)


if __name__ == "__main__":
    main()
