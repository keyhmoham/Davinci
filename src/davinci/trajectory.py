"""Validation, deduplication, and leakage-aware splitting for agent trajectories."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def validate_trajectory(item: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(item, Mapping):
        return ["trajectory must be an object"]
    if not isinstance(item.get("trajectory_id"), str) or not item["trajectory_id"].strip():
        issues.append("trajectory_id must be a non-empty string")
    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return issues + ["messages must be a list with at least two entries"]
    metadata = item.get("metadata")
    if not isinstance(metadata, Mapping):
        issues.append("metadata must be an object")

    pending_tool: str | None = None
    user_seen = False
    assistant_seen = False
    for index, message in enumerate(messages):
        prefix = f"message {index}: "
        if not isinstance(message, Mapping):
            issues.append(prefix + "must be an object")
            continue
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            issues.append(prefix + "invalid role")
            continue
        if role == "user":
            user_seen = True
        if role == "assistant":
            assistant_seen = True

        if message.get("type") == "tool_call":
            name = message.get("name")
            arguments = message.get("arguments")
            if role != "assistant":
                issues.append(prefix + "tool_call must be assistant-authored")
            if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
                issues.append(prefix + "tool name is missing or invalid")
            if not isinstance(arguments, Mapping):
                issues.append(prefix + "tool arguments must be an object")
            else:
                try:
                    json.dumps(dict(arguments), ensure_ascii=False, allow_nan=False)
                except (TypeError, ValueError):
                    issues.append(prefix + "tool arguments must be strict JSON-serializable")
            if pending_tool is not None:
                issues.append(prefix + f"previous tool call {pending_tool!r} has no result")
            pending_tool = name if isinstance(name, str) and _TOOL_NAME.fullmatch(name) else None
            continue

        if role == "tool":
            name = message.get("name")
            if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
                issues.append(prefix + "tool result name is missing or invalid")
            if not isinstance(message.get("content"), str):
                issues.append(prefix + "tool result content must be a string")
            if not isinstance(message.get("trusted"), bool):
                issues.append(prefix + "tool result must declare trusted as a boolean")
            if pending_tool is None:
                issues.append(prefix + "tool result has no preceding assistant tool call")
            elif name != pending_tool:
                issues.append(prefix + f"tool result name {name!r} does not match pending call {pending_tool!r}")
            pending_tool = None
            continue

        if pending_tool is not None:
            issues.append(prefix + f"expected tool result for {pending_tool!r} before another message")
            pending_tool = None
        if not isinstance(message.get("content"), str):
            issues.append(prefix + "content must be a string")

    if pending_tool is not None:
        issues.append(f"trajectory ends without a result for tool {pending_tool!r}")
    if not user_seen:
        issues.append("trajectory must contain a user message")
    if not assistant_seen:
        issues.append("trajectory must contain an assistant message")
    last = messages[-1]
    if not isinstance(last, Mapping) or last.get("role") != "assistant" or last.get("type") == "tool_call":
        issues.append("trajectory must end with a final assistant message")
    return issues


def trajectory_hash(item: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be strict JSON-serializable") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_trajectories(items: Sequence[Any]) -> dict[str, Any]:
    seen: dict[str, int] = {}
    invalid = 0
    duplicates = []
    issues_report = []
    valid = 0
    for index, item in enumerate(items):
        issues = validate_trajectory(item)
        if issues:
            invalid += 1
            issues_report.append({"index": index, "issues": issues})
        else:
            valid += 1
        try:
            digest = trajectory_hash(item)
        except (TypeError, ValueError):
            continue
        if digest in seen:
            duplicates.append({"index": index, "duplicate_of": seen[digest], "sha256": digest})
        else:
            seen[digest] = index
    return {
        "total": len(items),
        "valid": valid,
        "invalid": invalid,
        "duplicates": len(duplicates),
        "duplicate_rows": duplicates,
        "issues": issues_report,
    }


def split_trajectories(
    items: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Keep exact duplicate trajectories and shared source-document groups in one split."""
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise ValueError("validation_fraction must be numeric")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    records = list(items)
    if not records:
        raise ValueError("items must not be empty")
    for item in records:
        issues = validate_trajectory(item)
        if issues:
            raise ValueError("cannot split invalid trajectory: " + "; ".join(issues))

    parent = list(range(len(records)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_for_key: dict[str, int] = {}
    keys_by_item = []
    for index, item in enumerate(records):
        keys = ["content:" + trajectory_hash(item)]
        source_id = item.get("source_document_id")
        if source_id is not None and str(source_id):
            keys.append("source:" + str(source_id))
        keys_by_item.append(keys)
        for key in keys:
            if key in first_for_key:
                union(index, first_for_key[key])
            else:
                first_for_key[key] = index

    component_keys: dict[int, list[str]] = {}
    for index, keys in enumerate(keys_by_item):
        component_keys.setdefault(find(index), []).extend(keys)

    assignments: dict[int, str] = {}
    train, validation = [], []
    for index, item in enumerate(records):
        root = find(index)
        if root not in assignments:
            stable_key = min(component_keys[root])
            digest = hashlib.sha256(f"{seed}:{stable_key}".encode("utf-8")).hexdigest()
            bucket = int(digest[:16], 16) / float(16**16)
            assignments[root] = "validation" if bucket < validation_fraction else "train"
        (validation if assignments[root] == "validation" else train).append(item)
    if not train or not validation:
        raise ValueError("split produced an empty partition; add trajectories or adjust validation_fraction/seed")
    return train, validation


def export_jsonl(items: Sequence[Mapping[str, Any]], path: str | Path) -> int:
    """Write valid, exact-deduplicated trajectories as one JSON object per line."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            if validate_trajectory(item):
                continue
            digest = trajectory_hash(item)
            if digest in seen:
                continue
            seen.add(digest)
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count
