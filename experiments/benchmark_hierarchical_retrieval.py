"""Synthetic comparison of flat and hierarchical retrieval.

Run from the repository root:
    PYTHONPATH=. python experiments/benchmark_hierarchical_retrieval.py
This measures retrieval diagnostics only, not end-to-end model quality.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from src.davinci.hierarchical_retrieval import (
    HierarchicalRetrievalConfig,
    hierarchical_retrieve,
)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--coarse-top-k", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for name in ("entries", "block_size", "coarse_top_k", "top_k",
                 "heads", "head_dim", "queries"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    device = torch.device(args.device)
    torch.manual_seed(41)
    memory = torch.randn(1, args.heads, args.entries, args.head_dim, device=device)
    query = torch.randn(1, args.heads, args.queries, args.head_dim, device=device)
    cfg = HierarchicalRetrievalConfig(args.block_size, args.coarse_top_k, args.top_k)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Flat exact reference.
    sync()
    start = time.perf_counter()
    flat = torch.einsum("bhqd,bhmd->bhqm", query, memory) / (args.head_dim ** 0.5)
    flat_scores, flat_indices = flat.topk(min(args.top_k, args.entries), dim=-1)
    sync()
    flat_ms = (time.perf_counter() - start) * 1000

    for _ in range(2):
        hierarchical_retrieve(memory, query, cfg)
    sync()
    start = time.perf_counter()
    for _ in range(5):
        selected_indices, selected_scores = hierarchical_retrieve(memory, query, cfg)
    sync()
    hierarchical_ms = (time.perf_counter() - start) * 200

    hits = (selected_indices.unsqueeze(-1) == flat_indices.unsqueeze(-2)).any(dim=-1)
    recall = float(hits.float().mean().item())
    blocks = (args.entries + args.block_size - 1) // args.block_size
    selected_blocks = min(args.coarse_top_k, blocks)
    candidate_count = min(args.entries, selected_blocks * args.block_size)
    report = {
        "warning": "Synthetic retrieval microbenchmark; not a model-quality or 500K-context result.",
        "device": str(device),
        "entries": args.entries,
        "queries": args.queries,
        "block_size": args.block_size,
        "coarse_top_k": args.coarse_top_k,
        "final_top_k": min(args.top_k, args.entries),
        "candidate_entries_per_query": candidate_count,
        "candidate_fraction": candidate_count / args.entries,
        "flat_exact_ms": flat_ms,
        "hierarchical_mean_ms_over_5_calls": hierarchical_ms,
        "flat_topk_recall": recall,
        "selected_score_max_abs_error_for_hits": (
            float((selected_scores - flat_scores).abs().max().item()) if recall == 1.0 else None
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
