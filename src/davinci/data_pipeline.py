"""Strict JSONL and provenance-manifest helpers for training data.

The helpers verify recorded provenance and file integrity. They do not decide
whether a dataset license is legally sufficient; approval remains an explicit
human review decision recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


class DatasetManifestError(ValueError):
    """Raised when a dataset manifest is incomplete or inconsistent."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON object manifest and reject malformed or non-object JSON."""
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(f"cannot read dataset manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetManifestError("dataset manifest root must be a JSON object")
    return value


def validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    require_license_approval: bool = True,
    intended_purpose: str | None = None,
) -> None:
    """Validate provenance and explicit approval fields before model training."""
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    required = ("dataset_id", "dataset_revision", "purpose", "license_reviewed_by_user",
                "license_approved_for_use")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise DatasetManifestError(f"manifest is missing required fields: {', '.join(missing)}")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"].strip():
        raise DatasetManifestError("dataset_id must be a non-empty string")
    if not isinstance(manifest["dataset_revision"], str) or not manifest["dataset_revision"].strip():
        raise DatasetManifestError("dataset_revision must be an immutable revision string")
    if not isinstance(manifest["purpose"], str) or not manifest["purpose"].strip():
        raise DatasetManifestError("purpose must be a non-empty string")
    if intended_purpose is not None and manifest["purpose"] != intended_purpose:
        raise DatasetManifestError(
            f"dataset purpose {manifest['purpose']!r} does not match {intended_purpose!r}"
        )
    if not isinstance(manifest["license_reviewed_by_user"], bool):
        raise DatasetManifestError("license_reviewed_by_user must be an explicit boolean")
    if not isinstance(manifest["license_approved_for_use"], bool):
        raise DatasetManifestError("license_approved_for_use must be an explicit boolean")
    if require_license_approval:
        if manifest["license_reviewed_by_user"] is not True:
            raise PermissionError("dataset license has not been explicitly reviewed")
        if manifest["license_approved_for_use"] is not True:
            raise PermissionError("dataset has not been approved for the intended use")
    notes = manifest.get("license_review_notes")
    if require_license_approval and (not isinstance(notes, str) or not notes.strip()):
        raise DatasetManifestError("approved manifest must include license_review_notes")


def verify_dataset_artifact(
    data_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    hash_field: str = "combined_jsonl_sha256",
) -> str:
    """Verify an artifact against its manifest's recorded SHA-256 field."""
    if hash_field not in manifest:
        raise DatasetManifestError(f"manifest is missing artifact hash field {hash_field!r}")
    expected = manifest[hash_field]
    if not isinstance(expected, str) or len(expected) != 64:
        raise DatasetManifestError(f"{hash_field} must be a 64-character SHA-256 hex digest")
    actual = sha256_file(data_path)
    if actual.lower() != expected.lower():
        raise DatasetManifestError(
            f"artifact SHA-256 mismatch for {data_path}: expected {expected}, got {actual}"
        )
    return actual


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object records, failing with a precise line number on bad input."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            yield value


def iter_text_records(
    path: str | Path,
    *,
    text_field: str = "text",
    split: str | None = None,
    split_field: str = "split",
) -> Iterator[dict[str, Any]]:
    """Yield non-empty text records, optionally selecting one explicit split."""
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string")
    if split is not None and (not isinstance(split, str) or not split):
        raise ValueError("split must be None or a non-empty string")
    for line_number, record in enumerate(iter_jsonl(path), 1):
        if split is not None and record.get(split_field) != split:
            continue
        text = record.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {line_number} has no non-empty string in {text_field!r}")
        yield record
