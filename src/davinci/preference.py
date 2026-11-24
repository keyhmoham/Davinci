"""Validated preference data and Direct Preference Optimization primitives."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .data_audit import normalize_persian_text, content_sha256, assign_split


class PreferenceDataError(ValueError):
    """Raised for malformed preference comparisons."""


def validate_preference_pair(record: Mapping[str, Any]) -> dict[str, str]:
    """Validate one prompt/chosen/rejected comparison without guessing fields."""
    if not isinstance(record, Mapping):
        raise TypeError("preference record must be a mapping")
    values: dict[str, str] = {}
    for field in ("prompt", "chosen", "rejected"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PreferenceDataError(f"{field} must be a non-empty string")
        values[field] = value.strip()
    if values["chosen"] == values["rejected"]:
        raise PreferenceDataError("chosen and rejected responses must differ")
    pair_id = record.get("pair_id")
    if pair_id is not None and (not isinstance(pair_id, str) or not pair_id.strip()):
        raise PreferenceDataError("pair_id must be a non-empty string when provided")
    return {
        "pair_id": pair_id.strip() if isinstance(pair_id, str) else "",
        "prompt": values["prompt"],
        "chosen": values["chosen"],
        "rejected": values["rejected"],
    }


def preference_split(prompt: str, *, validation_fraction: float = 0.02, seed: int = 42) -> str:
    """Assign all comparisons for a normalized prompt to the same split."""
    normalized = normalize_persian_text(prompt)
    return assign_split(content_sha256(normalized), validation_fraction=validation_fraction, seed=seed)


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed next-token log-probabilities and target-token counts per row.

    Labels use -100 for ignored positions. Causal shifting is applied internally:
    logits at t score labels at t+1. Padding and ignored targets are never scored.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch tensors")
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence dimensions must match")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2 for causal scoring")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must be a tensor with the same shape as labels")
        if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("attention_mask must have an integer or boolean dtype")
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    if attention_mask is not None:
        valid = valid & attention_mask[:, 1:].bool()
    vocab_size = logits.shape[-1]
    invalid_ids = valid & ((shifted_labels < 0) | (shifted_labels >= vocab_size))
    if bool(invalid_ids.any()):
        raise ValueError("a non-masked label is outside the logits vocabulary")
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = F.log_softmax(logits[:, :-1].float(), dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    counts = valid.sum(dim=-1)
    return token_logps.sum(dim=-1), counts


def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor, *, beta: float = 0.1, reduction: str = "mean") -> dict[str, torch.Tensor]:
    """Compute the standard DPO objective from summed sequence log-probabilities."""
    tensors = (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("all log-probability inputs must be torch tensors")
    shape = policy_chosen_logps.shape
    if len(shape) != 1 or any(value.shape != shape for value in tensors[1:]):
        raise ValueError("all log-probability inputs must be one-dimensional with equal shape")
    if not all(torch.is_floating_point(value) for value in tensors):
        raise TypeError("log-probability tensors must have floating-point dtype")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("log-probability tensors must contain only finite values")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    per_pair_loss = -F.logsigmoid(logits)
    if reduction == "mean":
        losses = per_pair_loss.mean()
    elif reduction == "sum":
        losses = per_pair_loss.sum()
    else:
        losses = per_pair_loss
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return {
        "loss": losses,
        "per_pair_loss": per_pair_loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_margin": chosen_rewards - rejected_rewards,
        "preference_accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        "logits": logits,
    }