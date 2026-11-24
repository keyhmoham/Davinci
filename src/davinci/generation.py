"""Cached autoregressive generation for the canonical Davinci decoder."""
from __future__ import annotations

from typing import Optional

import torch


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits.float(), dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_original = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        logits = logits.masked_fill(remove_original, float("-inf"))
    return logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one token from [B,V] logits; temperature <= 0 selects greedy."""
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError("logits must have non-empty [batch, vocabulary] shape")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be a finite number")
    if not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError("temperature must be a finite number")
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = _filter_logits(logits.float() / float(temperature), top_k=top_k, top_p=top_p)
    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all() or torch.any(probabilities.sum(dim=-1) <= 0):
        raise ValueError("sampling filters removed all valid tokens")
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate with one prefill and cached single-token decode steps.

    Returns the prompt concatenated with generated tokens. For a batch, decoding
    stops when every row has emitted EOS; already-finished rows are padded with
    EOS while other rows continue.
    """
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a [batch, sequence] tensor")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids batch and sequence dimensions must be non-empty")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if eos_token_id is not None and (
        not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool)
        or eos_token_id < 0 or eos_token_id >= model.config.vocab_size
    ):
        raise ValueError("eos_token_id must be a valid vocabulary index")
    if max_new_tokens == 0:
        return input_ids

    was_training = model.training
    model.eval()
    try:
        logits, caches = model.prefill(input_ids)
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        prompt_length = input_ids.shape[1]
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished = finished | (next_token[:, 0] == eos_token_id)
            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 == max_new_tokens or (eos_token_id is not None and bool(finished.all())):
                break
            logits = model.decode(next_token, prompt_length + step, caches)
        return generated
    finally:
        model.train(was_training)
