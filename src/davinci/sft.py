"""Canonical chat serialization and assistant-only SFT utilities for Davinci."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .data_audit import normalize_persian_text


_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "tool_call": "<|tool_call|>",
    "tool_result": "<|tool_result|>",
    "end": "<|endofturn|>",
}


def _strict_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tool payload must be strict JSON-serializable") from exc


def tool_call_content(name: str, arguments: Mapping[str, Any]) -> str:
    if not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError("tool name must match [A-Za-z][A-Za-z0-9_.-]{0,63}")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    return SPECIAL_TOKENS["tool_call"] + _strict_json({"name": name, "arguments": dict(arguments)})


def tool_result_content(name: str, result: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool name must be a non-empty string")
    return SPECIAL_TOKENS["tool_result"] + _strict_json({"name": name, "result": result})


def _message_content(message: Mapping[str, Any], role: str) -> str:
    """Normalize plain chat and recorded trajectory tool events to canonical payloads."""
    if role == "assistant" and (message.get("type") == "tool_call" or "tool_call" in message):
        call = message.get("tool_call", message)
        if not isinstance(call, Mapping):
            raise TypeError("assistant tool_call must be a mapping")
        return tool_call_content(call.get("name"), call.get("arguments"))
    content = message.get("content", "")
    if not isinstance(content, str):
        raise TypeError("message content must be a string")
    if role == "tool" and message.get("name") and not content.startswith(SPECIAL_TOKENS["tool_result"]):
        try:
            result = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ))
        except (json.JSONDecodeError, ValueError):
            result = content
        return tool_result_content(message["name"], result)
    return content

def serialize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Serialize each message once; the final assistant end marker ends the turn."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    output = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unsupported role at message {index}: {role!r}")
        try:
            content = _message_content(message, role)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"message {index}: {exc}") from exc
        output.append(SPECIAL_TOKENS[role] + content + SPECIAL_TOKENS["end"])
    return "".join(output)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Support SentencePieceProcessor and callable Hugging Face tokenizer interfaces."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
    elif hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, out_type=int)
        except TypeError:
            ids = tokenizer.encode(text)
    else:
        raise TypeError("tokenizer must be callable or expose encode()")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        raise ValueError("tokenizer returned batched IDs; expected one unbatched sequence")
    return [int(token_id) for token_id in ids]


def build_masked_example(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, list[int]]:
    """Mask system/user/tool observations; supervise assistant text and tool-call JSON."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence")
    input_ids: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"unsupported role at message {index}: {role!r}")
        try:
            content = _message_content(message, role)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"message {index}: {exc}") from exc
        chunk = encode_text(tokenizer, SPECIAL_TOKENS[role] + content + SPECIAL_TOKENS["end"])
        input_ids.extend(chunk)
        labels.extend(chunk if role == "assistant" else [-100] * len(chunk))
    if not input_ids or not any(token != -100 for token in labels):
        raise ValueError("example must contain at least one assistant target token")
    return {"input_ids": input_ids, "labels": labels}


def _group_keys(example: Mapping[str, Any]) -> list[str]:
    keys = []
    for field in ("source_document_id", "document_id"):
        value = example.get(field)
        if value is not None and str(value):
            keys.append(f"{field}:{value}")
    payload = example.get("messages")
    if payload is None:
        payload = example
    keys.append("content:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        user_parts = [
            normalize_persian_text(message["content"])
            for message in payload
            if isinstance(message, Mapping) and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        prompt = "\n".join(part for part in user_parts if part).strip()
        if prompt:
            keys.append("prompt:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    return keys


def split_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split connected document/content groups so neither source groups nor exact duplicates leak."""
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise ValueError("validation_fraction must be numeric")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    items = list(examples)
    if not items:
        raise ValueError("examples must not be empty")

    parent = list(range(len(items)))

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
    keys_by_item: list[list[str]] = []
    for index, example in enumerate(items):
        if not isinstance(example, Mapping):
            raise TypeError("every example must be a mapping")
        keys = _group_keys(example)
        keys_by_item.append(keys)
        for key in keys:
            if key in first_for_key:
                union(index, first_for_key[key])
            else:
                first_for_key[key] = index

    component_keys: dict[int, list[str]] = {}
    for index, keys in enumerate(keys_by_item):
        root = find(index)
        component_keys.setdefault(root, []).extend(keys)

    assignments = {}
    train, validation = [], []
    for index, example in enumerate(items):
        root = find(index)
        if root not in assignments:
            stable_key = min(component_keys[root])
            digest = hashlib.sha256(f"{seed}:{stable_key}".encode("utf-8")).hexdigest()
            bucket = int(digest[:16], 16) / float(16**16)
            assignments[root] = "validation" if bucket < validation_fraction else "train"
        (validation if assignments[root] == "validation" else train).append(example)
    if not train or not validation:
        raise ValueError("split produced an empty partition; add examples or adjust validation_fraction/seed")
    return train, validation

def collate_masked(batch: Sequence[Mapping[str, Sequence[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")
    if not isinstance(pad_id, int) or isinstance(pad_id, bool) or pad_id < 0:
        raise ValueError("pad_id must be a non-negative integer")
    max_length = max(len(item["input_ids"]) for item in batch)
    if max_length == 0:
        raise ValueError("examples must contain at least one token")
    inputs, labels = [], []
    for item in batch:
        ids, targets = list(item["input_ids"]), list(item["labels"])
        if len(ids) != len(targets):
            raise ValueError("input_ids and labels lengths must match")
        padding = max_length - len(ids)
        inputs.append(ids + [pad_id] * padding)
        labels.append(targets + [-100] * padding)
    input_tensor = torch.tensor(inputs, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    if not torch.any(label_tensor[:, 1:] != -100):
        raise ValueError("batch contains no next-token assistant targets")
    return input_tensor, label_tensor


def causal_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and matching labels [B,T]")
    if logits.shape[1] < 2 or not torch.any(labels[:, 1:] != -100):
        raise ValueError("at least one next-token assistant target is required")
    return F.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
