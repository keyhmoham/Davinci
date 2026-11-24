"""Synthetic benchmark for the experimental compressed KV memory.

Run from the repository root:
    PYTHONPATH=. python experiments/benchmark_compressed_memory.py

This is a microbenchmark, not a model-quality or 500K-context evaluation.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from src.davinci.long_context import CompressedKVMemory, CompressedMemoryConfig


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def retrieval_recall(
    key: torch.Tensor,
    query: torch.Tensor,
    group_size: int,
    selected_count: int,
    historical_group_count: int,
) -> float:
    """Recall of pooled-key top-k groups against groups ranked by best token score."""
    # Inputs: key [1, H, T, D], query [1, H, Q, D].
    _, heads, token_count, dim = key.shape
    groups = min(token_count // group_size, historical_group_count)
    usable = groups * group_size
    if groups == 0:
        return float("nan")
    grouped = key[:, :, :usable].reshape(1, heads, groups, group_size, dim)
    pooled = grouped.mean(dim=3)
    q = query.float()
    pooled_scores = torch.einsum("bhqd,bhgd->bhqg", q, pooled.float()) / (dim ** 0.5)
    # Use exact max-over-token dot product as a retrieval oracle.
    token_scores = torch.stack([
        torch.einsum("bhqd,bhgd->bhqg", q, grouped[:, :, :, offset, :].float()) / (dim ** 0.5)
        for offset in range(group_size)
    ], dim=0).amax(dim=0)
    k = min(selected_count, groups)
    predicted = pooled_scores.topk(k, dim=-1).indices
    oracle = token_scores.topk(k, dim=-1).indices
    hits = (predicted.unsqueeze(-1) == oracle.unsqueeze(-2)).any(dim=-1).float()
    return float(hits.mean().item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--recent-window", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for name in ("tokens", "recent_window", "group_size", "top_k", "heads", "head_dim", "queries"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace(chr(95), chr(45))} must be positive")
    if args.tokens <= args.recent_window:
        parser.error("--tokens must exceed --recent-window")
    if args.tokens % args.group_size:
        parser.error("--tokens must be divisible by --group-size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    torch.manual_seed(7)
    key = torch.randn(1, args.heads, args.tokens, args.head_dim, device=device)
    value = torch.randn_like(key)
    query = torch.randn(1, args.heads, args.queries, args.head_dim, device=device)
    query_positions = torch.full((args.queries,), args.tokens, dtype=torch.long, device=device)

    cache = CompressedKVMemory(CompressedMemoryConfig(
        recent_window=args.recent_window,
        compression_group_size=args.group_size,
        retrieval_top_k=args.top_k,
    ))
    synchronize(device)
    start = time.perf_counter()
    cache.append(key, value, torch.arange(args.tokens, device=device))
    synchronize(device)
    append_ms = (time.perf_counter() - start) * 1000

    for _ in range(3):
        cache.attend(query, query_positions)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(10):
        cache.attend(query, query_positions)
    synchronize(device)
    attend_ms = (time.perf_counter() - start) * 100

    recall = retrieval_recall(key, query, args.group_size, args.top_k, cache.memory_key.shape[2])
    full_kv_bytes = (key.numel() + value.numel()) * key.element_size()
    stored_kv_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (cache.recent_key, cache.recent_value, cache.memory_key, cache.memory_value)
    )
    report = {
        "warning": "Synthetic microbenchmark only; not a 500K-context or task-quality result.",
        "device": str(device),
        "tokens": args.tokens,
        "recent_window": args.recent_window,
        "group_size": args.group_size,
        "retrieval_top_k": args.top_k,
        "compressed_groups": int(cache.memory_key.shape[2]),
        "pooled_group_retrieval_recall_at_k": recall,
        "append_ms": append_ms,
        "mean_attend_ms_over_10_calls": attend_ms,
        "full_kv_mib": full_kv_bytes / (1024 ** 2),
        "stored_kv_mib": stored_kv_bytes / (1024 ** 2),
        "kv_storage_ratio": stored_kv_bytes / full_kv_bytes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
