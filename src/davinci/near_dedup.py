"""Approximate SimHash/LSH near-duplicate candidate discovery.

This module reports candidate pairs for human review. It does not delete records
or make a semantic-equivalence decision.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .data_audit import normalize_persian_text

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MASK64 = (1 << 64) - 1


def _shingles(text: str, ngram_size: int) -> set[str]:
    tokens = _TOKEN_RE.findall(text.casefold())
    if len(tokens) < ngram_size:
        return set(tokens)
    return {" ".join(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)}


def _simhash(shingles: set[str]) -> int:
    if not shingles:
        return 0
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8,
                                               person=b"Davinci1").digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def find_near_duplicate_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    text_field: str = "text",
    id_field: str = "source_row",
    jaccard_threshold: float = 0.85,
    max_hamming_distance: int = 6,
    ngram_size: int = 5,
    max_bucket_size: int = 500,
) -> dict[str, Any]:
    """Return approximate near-duplicate candidate pairs for manual review.

    Uses 64-bit SimHash with eight 8-bit LSH bands to shortlist candidates,
    then verifies each candidate using exact Jaccard similarity over word
    n-gram shingles. Large LSH buckets are skipped to bound worst-case work.
    """
    if not isinstance(jaccard_threshold, (int, float)) or isinstance(jaccard_threshold, bool) or not 0 < jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    if not isinstance(max_hamming_distance, int) or isinstance(max_hamming_distance, bool) or not 0 <= max_hamming_distance <= 64:
        raise ValueError("max_hamming_distance must be an integer in [0, 64]")
    for name, value in (("ngram_size", ngram_size), ("max_bucket_size", max_bucket_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    signatures: list[int] = []
    shingle_sets: list[set[str]] = []
    identifiers: list[Any] = []
    pairs: list[dict[str, Any]] = []
    candidate_comparisons = 0
    skipped_overflow_buckets = 0
    too_short = 0

    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {row_index} must be a mapping")
        text = record.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"record {row_index} must contain string field {text_field!r}")
        normalized = normalize_persian_text(text)
        shingles = _shingles(normalized, ngram_size)
        if not shingles:
            too_short += 1
            continue
        signature = _simhash(shingles)
        candidates: set[int] = set()
        for band in range(8):
            value = (signature >> (band * 8)) & 0xFF
            bucket = buckets[(band, value)]
            if len(bucket) > max_bucket_size:
                skipped_overflow_buckets += 1
                continue
            candidates.update(bucket)
        for candidate in candidates:
            candidate_comparisons += 1
            distance = (signature ^ signatures[candidate]).bit_count()
            if distance > max_hamming_distance:
                continue
            intersection = len(shingles & shingle_sets[candidate])
            union = len(shingles | shingle_sets[candidate])
            similarity = intersection / union if union else 1.0
            if similarity >= jaccard_threshold:
                pairs.append({
                    "source_row": record.get(id_field, row_index),
                    "duplicate_candidate_of": identifiers[candidate],
                    "jaccard_similarity": similarity,
                    "simhash_hamming_distance": distance,
                })
        current_index = len(signatures)
        signatures.append(signature)
        shingle_sets.append(shingles)
        identifiers.append(record.get(id_field, row_index))
        for band in range(8):
            value = (signature >> (band * 8)) & 0xFF
            buckets[(band, value)].append(current_index)

    return {
        "pairs": pairs,
        "records_indexed": len(signatures),
        "candidate_comparisons": candidate_comparisons,
        "skipped_overflow_buckets": skipped_overflow_buckets,
        "too_short": too_short,
        "method": "64-bit SimHash, 8x8-bit LSH bands, exact Jaccard over word n-grams",
        "warning": "Approximate candidate discovery only; manually review false positives before filtering.",
    }