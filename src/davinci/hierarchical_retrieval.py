"""Experimental two-stage retrieval over compressed KV memory.

The index first ranks contiguous blocks by mean-key similarity, then reranks
individual entries inside the selected blocks. It is approximate whenever
coarse_top_k excludes blocks. It does not modify the canonical decoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HierarchicalRetrievalConfig:
    block_size: int = 64
    coarse_top_k: int = 8
    top_k: int = 32

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@torch.no_grad()
def hierarchical_retrieve(
    memory_key: torch.Tensor,
    query: torch.Tensor,
    config: HierarchicalRetrievalConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (indices, scores), each [B,H,Q,K], for selected memory entries.

    Inputs are memory_key [B,H,M,D] and query [B,H,Q,D]. The returned K is
    min(top_k, M). Candidate discovery costs O(M/block_size) coarse scores
    plus exact scoring of at most coarse_top_k*block_size entries per query.
    """
    cfg = config or HierarchicalRetrievalConfig()
    cfg.validate()
    if memory_key.ndim != 4 or query.ndim != 4:
        raise ValueError("memory_key and query must have [B,H,T,D] shapes")
    if memory_key.shape[:2] != query.shape[:2] or memory_key.shape[-1] != query.shape[-1]:
        raise ValueError("query and memory batch/head/head_dim dimensions must match")
    if memory_key.shape[2] == 0 or query.shape[2] == 0:
        raise ValueError("memory and query sequences must be non-empty")
    if not memory_key.is_floating_point() or not query.is_floating_point():
        raise ValueError("query and memory keys must be floating-point tensors")
    if memory_key.device != query.device or memory_key.dtype != query.dtype:
        raise ValueError("query and memory keys must share device and dtype")

    batch, heads, memory_len, dim = memory_key.shape
    blocks = (memory_len + cfg.block_size - 1) // cfg.block_size
    padded_len = blocks * cfg.block_size
    pad = padded_len - memory_len
    # Zero-pad keys, but exclude padded positions from both ranking stages.
    padded = torch.nn.functional.pad(memory_key.float(), (0, 0, 0, pad))
    block_keys = padded.reshape(batch, heads, blocks, cfg.block_size, dim)
    counts = torch.full((blocks,), cfg.block_size, device=memory_key.device, dtype=torch.float32)
    if pad:
        counts[-1] = cfg.block_size - pad
    centroids = block_keys.sum(dim=3) / counts[None, None, :, None]

    q = query.float()
    coarse_scores = torch.einsum("bhqd,bhgd->bhqg", q, centroids) / (dim ** 0.5)
    coarse_count = min(cfg.coarse_top_k, blocks)
    selected_blocks = coarse_scores.topk(coarse_count, dim=-1).indices

    offsets = torch.arange(cfg.block_size, device=memory_key.device)
    candidate_indices = (selected_blocks.unsqueeze(-1) * cfg.block_size + offsets).reshape(
        batch, heads, query.shape[2], -1
    )
    valid_candidates = candidate_indices < memory_len
    safe_indices = candidate_indices.clamp(max=memory_len - 1)
    gather_index = safe_indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
    expanded_memory = memory_key.unsqueeze(2).expand(-1, -1, query.shape[2], -1, -1)
    candidates = torch.gather(expanded_memory, 3, gather_index).float()
    fine_scores = (candidates * q.unsqueeze(3)).sum(dim=-1) / (dim ** 0.5)
    fine_scores = fine_scores.masked_fill(~valid_candidates, float("-inf"))
    result_count = min(cfg.top_k, memory_len)
    scores, positions = fine_scores.topk(result_count, dim=-1)
    indices = torch.gather(safe_indices, 3, positions)
    return indices, scores
