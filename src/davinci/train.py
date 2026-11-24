"""Small, explicit training utilities for the canonical Davinci model."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Iterable, Optional
import torch
import torch.nn.functional as F

def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                   ignore_index: int = -100) -> torch.Tensor:
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [B,T,V] and input_ids [B,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("Logit batch/sequence dimensions must match input_ids")
    if logits.shape[1] < 2:
        raise ValueError("At least two tokens are required for next-token loss")
    return F.cross_entropy(
        logits[:, :-1, :].float().contiguous().view(-1, logits.shape[-1]),
        input_ids[:, 1:].contiguous().view(-1),
        ignore_index=ignore_index,
    )

def train_microbatch(model, batch, optimizer, *, scaler=None, amp_dtype=None,
                     max_grad_norm: Optional[float] = 1.0,
                     gradient_accumulation_steps: int = 1,
                     microbatch_index: int = 0):
    """Backprop one microbatch; caller steps optimizer after the accumulation window."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    device = next(model.parameters()).device
    batch = batch.to(device, non_blocking=True)
    use_amp = amp_dtype is not None and device.type == "cuda"
    amp_context = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else nullcontext())
    with amp_context:
        output = model(batch, labels=batch)
        if isinstance(output, dict):
            loss = output.get("loss")
            if loss is None:
                loss = causal_lm_loss(output["logits"], batch)
        else:
            loss = causal_lm_loss(output, batch)
        scaled_loss = loss / gradient_accumulation_steps
    if not torch.isfinite(scaled_loss):
        raise FloatingPointError("Non-finite loss; optimizer step aborted")
    if scaler is not None and scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()
    should_step = (microbatch_index + 1) % gradient_accumulation_steps == 0
    grad_norm = None
    if should_step:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": loss.detach(), "optimizer_step": should_step,
            "grad_norm": grad_norm}

@torch.no_grad()
def evaluate_loss(model, batches: Iterable, max_batches: int = 20,
                  amp_dtype=None) -> Optional[float]:
    if max_batches < 1:
        raise ValueError("max_batches must be >= 1")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    values = []
    try:
        for i, batch in enumerate(batches):
            if i >= max_batches:
                break
            batch = batch.to(device, non_blocking=True)
            use_amp = amp_dtype is not None and device.type == "cuda"
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                output = model(batch, labels=batch)
                loss = output["loss"] if isinstance(output, dict) else causal_lm_loss(output, batch)
            if loss is not None and torch.isfinite(loss):
                values.append(float(loss.float().cpu()))
    finally:
        model.train(was_training)
    return sum(values) / len(values) if values else None
