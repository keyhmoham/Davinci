"""Canonical chat serialization and assistant-only SFT utilities for Davinci."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


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
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool arguments must be a mapping")
    return SPECIAL_TOKENS["tool_call"] + _strict_json({"name": name, "arguments": dict(arguments)})


def tool_result_content(name: str, result: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool name must be a non-empty string")
    return SPECIAL_TOKENS["tool_result"] + _strict_json({"name": name, "result": result})


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
        content = message.get("content", "")
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string")
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
        content = message.get("content", "")
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string")
        chunk = encode_text(tokenizer, SPECIAL_TOKENS[role] + content + SPECIAL_TOKENS["end"])
        input_ids.extend(chunk)
        labels.extend(chunk if role == "assistant" else [-100] * len(chunk))
    if not input_ids or not any(token != -100 for token in labels):
        raise ValueError("example must contain at least one assistant target token")
    return {"input_ids": input_ids, "labels": labels}


def _group_key(example: Mapping[str, Any]) -> str:
    for field in ("source_document_id", "document_id", "example_id", "id"):
        value = example.get(field)
        if value is not None and str(value):
            return f"{field}:{value}"
    payload = example.get("messages", example)
    return "content:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def split_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Deterministically split by document/example identity, not shuffled row index."""
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise ValueError("validation_fraction must be numeric")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    train, validation = [], []
    assignments: dict[str, str] = {}
    for example in examples:
        if not isinstance(example, Mapping):
            raise TypeError("every example must be a mapping")
        key = _group_key(example)
        split = assignments.get(key)
        if split is None:
            digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
            bucket = int(digest[:16], 16) / float(16**16)
            split = "validation" if bucket < validation_fraction else "train"
            assignments[key] = split
        (validation if split == "validation" else train).append(example)
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
