"""Token-weighted language-model and exact-match evaluation helpers."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .data_audit import normalize_persian_text


def evaluate_language_model(
    model: torch.nn.Module,
    batches: Iterable[Any],
    *,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate token-weighted causal loss and perplexity, restoring model mode.

    Each batch may be input_ids, (input_ids, labels), or a mapping with
    input_ids and optional labels. Labels use -100 for ignored positions.
    """
    if max_batches is not None and (not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches <= 0):
        raise ValueError("max_batches must be a positive integer or None")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0
    try:
        with torch.no_grad():
            for batch in batches:
                if max_batches is not None and batches_seen >= max_batches:
                    break
                if isinstance(batch, Mapping):
                    input_ids = batch.get("input_ids")
                    labels = batch.get("labels", input_ids)
                elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                    input_ids, labels = batch
                else:
                    input_ids, labels = batch, batch
                if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
                    raise TypeError("each batch must provide input_ids and labels tensors")
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                if labels.shape != input_ids.shape:
                    raise ValueError("labels must have the same shape as input_ids")
                if labels.ndim != 2 or labels.shape[1] < 2:
                    raise ValueError("evaluation batches must have [batch, sequence>=2] labels")
                target_count = int(labels[:, 1:].ne(-100).sum().item())
                if target_count == 0:
                    continue
                output = model(input_ids, labels=labels)
                loss = output.get("loss") if isinstance(output, Mapping) else None
                if loss is None or not bool(torch.isfinite(loss)):
                    raise ValueError("model returned a missing or non-finite evaluation loss")
                total_loss += float(loss.detach().float().cpu()) * target_count
                total_tokens += target_count
                batches_seen += 1
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation produced no supervised next-token targets")
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else float("inf")
    return {"loss": mean_loss, "perplexity": perplexity, "tokens": total_tokens, "batches": batches_seen}


def exact_match_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float | int]:
    """Compute exact match after the repository Persian text normalization."""
    pred_list = list(predictions)
    ref_list = list(references)
    if len(pred_list) != len(ref_list):
        raise ValueError("predictions and references must have equal lengths")
    if not pred_list:
        raise ValueError("at least one prediction/reference pair is required")
    if any(not isinstance(value, str) for value in pred_list + ref_list):
        raise TypeError("predictions and references must contain strings")
    correct = sum(normalize_persian_text(p).strip() == normalize_persian_text(r).strip()
                  for p, r in zip(pred_list, ref_list))
    return {"correct": correct, "total": len(pred_list), "accuracy": correct / len(pred_list)}


def grouped_exact_match_metrics(
    predictions: Iterable[str],
    references: Iterable[str],
    groups: Iterable[str],
) -> dict[str, dict[str, float | int]]:
    """Report normalized exact match overall by explicit language/task group.

    Group labels are supplied by the benchmark manifest; this function never
    infers a language from text. That avoids silently misclassifying code-mixed
    Persian/English examples.
    """
    pred_list = list(predictions)
    ref_list = list(references)
    group_list = list(groups)
    if not pred_list or len(pred_list) != len(ref_list) or len(pred_list) != len(group_list):
        raise ValueError("predictions, references, and groups must have equal non-zero lengths")
    if any(not isinstance(value, str) for value in pred_list + ref_list):
        raise TypeError("predictions and references must contain strings")
    if any(not isinstance(group, str) or not group.strip() for group in group_list):
        raise ValueError("group labels must be non-empty strings")
    correct_by_group: dict[str, int] = {}
    total_by_group: dict[str, int] = {}
    for prediction, reference, group in zip(pred_list, ref_list, group_list):
        key = group.strip()
        total_by_group[key] = total_by_group.get(key, 0) + 1
        correct_by_group[key] = correct_by_group.get(key, 0) + int(
            normalize_persian_text(prediction).strip()
            == normalize_persian_text(reference).strip()
        )
    result: dict[str, dict[str, float | int]] = {
        group: {
            "correct": correct_by_group[group],
            "total": total_by_group[group],
            "accuracy": correct_by_group[group] / total_by_group[group],
        }
        for group in sorted(total_by_group)
    }
    total = len(pred_list)
    correct = sum(correct_by_group.values())
    result["overall"] = {"correct": correct, "total": total, "accuracy": correct / total}
    return result
