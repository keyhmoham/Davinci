"""Reusable, conservative data-audit primitives for Persian/bilingual corpora.

These utilities do not identify language, establish dataset licenses, or decide
whether content is safe to train on. They expose deterministic measurements and
explicit rejection reasons so those decisions remain auditable.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

_ARABIC_TO_PERSIAN = str.maketrans({"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9"})
_DIRECTIONAL_CONTROLS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_SCRIPT = re.compile(r"[A-Za-z]")


def normalize_persian_text(text: str) -> str:
    """Apply conservative Unicode normalization; preserve ZWNJ and punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFC", text).translate(_ARABIC_TO_PERSIAN)
    text = text.replace("\u0640", "")
    text = _DIRECTIONAL_CONTROLS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _EXCESS_NEWLINES.sub("\n\n", text).strip()


def content_sha256(text: str) -> str:
    """Hash UTF-8 text after the caller has chosen its normalization policy."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_statistics(text: str) -> dict[str, int | float]:
    """Return transparent length/script statistics; script ratios are not language IDs."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    alpha_count = sum(char.isalpha() for char in text)
    arabic_count = sum(bool(_ARABIC_SCRIPT.fullmatch(char)) for char in text if char.isalpha())
    latin_count = sum(bool(_LATIN_SCRIPT.fullmatch(char)) for char in text if char.isalpha())
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "alpha_char_count": alpha_count,
        "arabic_script_alpha_count": arabic_count,
        "latin_alpha_count": latin_count,
        "arabic_script_alpha_ratio": arabic_count / alpha_count if alpha_count else 0.0,
        "latin_alpha_ratio": latin_count / alpha_count if alpha_count else 0.0,
        "replacement_char_count": text.count("\ufffd"),
        "line_count": len(text.splitlines()) if text else 0,
    }


def quality_reasons(
    text: str,
    *,
    min_chars: int = 40,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.5,
    max_replacement_chars: int = 0,
) -> list[str]:
    """Return deterministic heuristic rejection reasons; no language is required."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    for name, value in (("min_chars", min_chars), ("max_chars", max_chars),
                        ("max_replacement_chars", max_replacement_chars)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars")
    if (not isinstance(min_alpha_ratio, (int, float)) or isinstance(min_alpha_ratio, bool)
            or not 0.0 <= min_alpha_ratio <= 1.0):
        raise ValueError("min_alpha_ratio must be in [0, 1]")
    stats = text_statistics(text)
    reasons = []
    if not text.strip():
        reasons.append("empty")
    if stats["char_count"] < min_chars:
        reasons.append("too_short")
    if stats["char_count"] > max_chars:
        reasons.append("too_long")
    if stats["alpha_char_count"] == 0 and stats["char_count"] > 0:
        reasons.append("low_alphabetic_ratio")
    elif stats["alpha_char_count"] and (
        stats["alpha_char_count"] / max(stats["char_count"], 1) < min_alpha_ratio
    ):
        reasons.append("low_alphabetic_ratio")
    if stats["replacement_char_count"] > max_replacement_chars:
        reasons.append("replacement_characters")
    return reasons


def deduplicate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    text_field: str = "text",
    source_row_field: str = "source_row",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first normalized-text occurrence and report later duplicate lineage."""
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    first_by_hash: dict[str, Any] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        if text_field not in record:
            raise ValueError(f"record {index} is missing {text_field!r}")
        normalized = normalize_persian_text(record[text_field])
        digest = content_sha256(normalized)
        source_row = record.get(source_row_field, index)
        if digest in first_by_hash:
            duplicates.append({
                "source_row": source_row,
                "duplicate_of": first_by_hash[digest],
                "sha256": digest,
            })
            continue
        first_by_hash[digest] = source_row
        item = dict(record)
        item[text_field] = normalized
        item["sha256"] = digest
        kept.append(item)
    return kept, duplicates


def assign_split(
    digest: str,
    *,
    validation_fraction: float = 0.01,
    seed: int = 42,
) -> str:
    """Assign a stable train/validation split from a content digest and seed."""
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("digest must be a lowercase SHA-256 hex string")
    if (not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool)
            or not 0.0 <= validation_fraction <= 1.0):
        raise ValueError("validation_fraction must be in [0, 1]")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    bucket = int(hashlib.sha256(f"{seed}:{digest}".encode("ascii")).hexdigest()[:16], 16)
    threshold = int(validation_fraction * (1 << 64))
    return "validation" if bucket < threshold else "train"
