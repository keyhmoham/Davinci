"""Reproducible synthetic long-context retrieval evaluation utilities.

These utilities measure tokenized prompt length with the supplied tokenizer.
Synthetic retrieval is a plumbing check, not evidence of real-world 500K context.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def build_needle_example(
    example_id: int,
    *,
    documents: int,
    position_fraction: float,
    seed: int,
    language: str = "en",
) -> dict[str, Any]:
    if not isinstance(example_id, int) or isinstance(example_id, bool) or example_id < 0:
        raise ValueError("example_id must be a non-negative integer")
    if not isinstance(documents, int) or isinstance(documents, bool) or documents < 1:
        raise ValueError("documents must be a positive integer")
    if not 0.0 <= position_fraction <= 1.0:
        raise ValueError("position_fraction must be in [0, 1]")
    if language not in {"en", "fa"}:
        raise ValueError("language must be 'en' or 'fa'")
    rng = random.Random(seed)
    key = f"dv_fact_{example_id:08d}"
    answer = f"dv_value_{rng.randrange(10**12):012d}"
    if language == "fa":
        distractors = [
            "این سند درباره روندهای منطقه‌ای و گزارش بازار است.",
            "خلاصه پروژه جزئیات پیاده‌سازی و اقدامات بعدی را شرح می‌دهد.",
            "یادداشت جلسه شامل تصمیم‌ها و وظایف گروه است.",
        ]
        needle = f"شناسه {key} به مقدار {answer} مربوط است."
        question = f"مقدار مربوط به شناسه {key} چیست؟"
    else:
        distractors = [
            "The report describes regional trends and market conditions.",
            "The project summary records implementation details and next steps.",
            "The meeting note lists decisions and team action items.",
        ]
        needle = f"Record: {key} is associated with {answer}."
        question = f"What value is associated with {key}?"
    chunks = [f"Document {i}: {rng.choice(distractors)}" for i in range(documents)]
    insertion = min(documents, max(0, round(position_fraction * documents)))
    chunks.insert(insertion, needle)
    prompt = "\n\n".join(chunks) + "\n\n" + question
    return {
        "id": example_id,
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "synthetic": True,
            "seed": seed,
            "language": language,
            "position_fraction": position_fraction,
            "needle_document_index": insertion,
            "distractor_documents": documents,
        },
    }


def measure_token_lengths(
    examples: Iterable[dict[str, Any]],
    encode: Callable[[str], Sequence[int]],
) -> list[dict[str, Any]]:
    """Attach exact tokenizer lengths; encoder must return token IDs."""
    measured = []
    for example in examples:
        if not isinstance(example, dict) or not isinstance(example.get("prompt"), str):
            raise ValueError("each example must be a mapping with a prompt string")
        ids = encode(example["prompt"])
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ValueError("tokenizer must return a non-empty sequence of token IDs")
        row = dict(example)
        row["metadata"] = dict(example.get("metadata", {}))
        row["metadata"]["token_count"] = len(ids)
        measured.append(row)
    return measured


def score_retrieval(predictions: Sequence[str], examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(predictions) != len(examples) or not examples:
        raise ValueError("predictions and examples must have equal non-zero lengths")
    groups: dict[str, list[bool]] = {}
    for prediction, example in zip(predictions, examples):
        if not isinstance(prediction, str):
            raise TypeError("predictions must be strings")
        metadata = example.get("metadata", {})
        group = f"{metadata.get('language', 'unknown')}@{metadata.get('position_fraction', 'unknown')}"
        groups.setdefault(group, []).append(prediction.strip() == example["answer"].strip())
    by_group = {
        key: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
        for key, values in sorted(groups.items())
    }
    correct = sum(item["correct"] for item in by_group.values())
    return {"correct": correct, "total": len(examples), "accuracy": correct / len(examples), "groups": by_group}


def write_jsonl_and_manifest(
    examples: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    tokenizer_id: str,
    seed: int,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot write an empty evaluation set")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise ValueError("tokenizer_id must be explicit")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in examples)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "task": "synthetic_needle_retrieval",
        "synthetic_only": True,
        "tokenizer_id": tokenizer_id,
        "seed": seed,
        "examples": len(examples),
        "jsonl_sha256": digest,
        "warning": "Synthetic sanity check only; not a real-world long-context capability benchmark.",
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
