"""Experimental compressed historical KV memory (approximate; not wired into the canonical decoder)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

@dataclass
class CompressedMemoryConfig:
    recent_window: int = 2048
    compression_group_size: int = 16
    retrieval_top_k: int = 32
    def validate(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.compression_group_size > self.recent_window:
            raise ValueError("compression_group_size must not exceed recent_window")

class CompressedKVMemory:
    """Exact recent KV plus mean-pooled historical groups; approximate attention."""
    def __init__(self, config: Optional[CompressedMemoryConfig] = None):
        self.config = config or CompressedMemoryConfig()
        self.config.validate()
        self.recent_key = self.recent_value = self.recent_positions = None
        self.memory_key = self.memory_value = self.end_positions = None

    def append(self, key: torch.Tensor, value: torch.Tensor, positions: torch.Tensor) -> None:
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("key/value must have matching [B,H,T,D] shapes")
        if not key.is_floating_point() or not value.is_floating_point():
            raise ValueError("key/value tensors must use floating-point dtypes")
        if key.shape[2] == 0:
            raise ValueError("append requires at least one KV token")
        if positions.ndim != 1 or positions.numel() != key.shape[2]:
            raise ValueError("positions must have shape [T]")
        if positions.device != key.device:
            raise ValueError("positions and KV tensors must be on the same device")
        if positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("positions must use an integer dtype")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError("key and value must have the same device and dtype")
        if positions.numel() > 1 and torch.any(positions[1:] <= positions[:-1]):
            raise ValueError("positions must be strictly increasing within each append")
        if self.recent_key is not None:
            if key.shape[:2] != self.recent_key.shape[:2] or key.shape[-1] != self.recent_key.shape[-1]:
                raise ValueError("KV batch/head/head_dim shape changed")
            if key.device != self.recent_key.device or key.dtype != self.recent_key.dtype:
                raise ValueError("appended KV tensors must match cached device and dtype")
            if positions.dtype != self.recent_positions.dtype:
                raise ValueError("appended positions must match cached position dtype")
            if positions.numel() and self.recent_positions.numel() and positions[0] <= self.recent_positions[-1]:
                raise ValueError("positions must be strictly increasing")
            key = torch.cat((self.recent_key, key), dim=2)
            value = torch.cat((self.recent_value, value), dim=2)
            positions = torch.cat((self.recent_positions, positions), dim=0)
        self.recent_key, self.recent_value, self.recent_positions = key, value, positions
        overflow = key.shape[2] - self.config.recent_window
        group = self.config.compression_group_size
        count = min(key.shape[2], ((overflow + group - 1) // group) * group)
        count = (count // group) * group
        if count:
            old_k, old_v = key[:, :, :count], value[:, :, :count]
            groups = count // group
            pooled_k = old_k.reshape(*old_k.shape[:2], groups, group, old_k.shape[-1]).mean(dim=3)
            pooled_v = old_v.reshape(*old_v.shape[:2], groups, group, old_v.shape[-1]).mean(dim=3)
            ends = positions[:count].reshape(groups, group)[:, -1]
            if self.memory_key is None:
                self.memory_key, self.memory_value, self.end_positions = pooled_k, pooled_v, ends
            else:
                self.memory_key = torch.cat((self.memory_key, pooled_k), dim=2)
                self.memory_value = torch.cat((self.memory_value, pooled_v), dim=2)
                self.end_positions = torch.cat((self.end_positions, ends), dim=0)
            self.recent_key, self.recent_value = key[:, :, count:], value[:, :, count:]
            self.recent_positions = positions[count:]

    def attend(self, query: torch.Tensor, query_positions: torch.Tensor,
               scale: Optional[float] = None) -> torch.Tensor:
        if query.ndim != 4 or self.recent_key is None:
            raise ValueError("append KV entries before calling attend")
        if query_positions.ndim != 1 or query_positions.numel() != query.shape[2]:
            raise ValueError("query_positions must have shape [Q]")
        if query.device != self.recent_key.device or query_positions.device != query.device:
            raise ValueError("query, query_positions, and cached KV tensors must share a device")
        if not query.is_floating_point():
            raise ValueError("query must use a floating-point dtype")
        if query_positions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("query_positions must use an integer dtype")
        if query.dtype != self.recent_key.dtype:
            raise ValueError("query and cached KV tensors must share a dtype")
        b, h, q_len, dim = query.shape
        if query.shape[:2] != self.recent_key.shape[:2] or dim != self.recent_key.shape[-1]:
            raise ValueError("query/cache batch, head, and head_dim dimensions must match")

        keys, values, masks = [], [], []
        if self.memory_key is not None and self.memory_key.shape[2]:
            scores = torch.einsum("bhqd,bhmd->bhqm", query.float(), self.memory_key.float())
            scores = scores / (dim ** 0.5)
            valid = self.end_positions[None, :] < query_positions[:, None]
            scores = scores.masked_fill(~valid[None, None], float("-inf"))
            k = min(self.config.retrieval_top_k, self.memory_key.shape[2])
            indices = scores.topk(k, dim=-1).indices
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, -1, dim)
            mk = self.memory_key.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
            mv = self.memory_value.unsqueeze(2).expand_as(mk)
            keys.append(torch.gather(mk, 3, gather_idx))
            values.append(torch.gather(mv, 3, gather_idx))
            masks.append(torch.gather(valid[None, None].expand(b,h,-1,-1), 3, indices))

        recent_valid = self.recent_positions[None, :] <= query_positions[:, None]
        rk = self.recent_key.unsqueeze(2).expand(-1,-1,q_len,-1,-1)
        rv = self.recent_value.unsqueeze(2).expand_as(rk)
        keys.append(rk)
        values.append(rv)
        masks.append(recent_valid[None,None].expand(b,h,-1,-1))
        all_k = torch.cat(keys, dim=3).reshape(b,h,q_len,-1,dim)
        all_v = torch.cat(values, dim=3).reshape(b,h,q_len,-1,dim)
        allowed = torch.cat(masks, dim=3)
        # SDPA expects [B,H,Q,K]; fold query-specific K/V into batch-query rows.
        q = query.reshape(b*h*q_len,1,1,dim)
        k_tensor = all_k.reshape(b*h*q_len,1,-1,dim)
        v_tensor = all_v.reshape(b*h*q_len,1,-1,dim)
        mask = allowed.reshape(b*h*q_len,1,1,-1)
        out = F.scaled_dot_product_attention(q,k_tensor,v_tensor,attn_mask=mask,
                                             dropout_p=0.0,is_causal=False,scale=scale)
        out = out.reshape(b,h,q_len,dim)
        # Define queries with no visible cache entries as zero outputs.
        has_visible_key = allowed.any(dim=-1)
        return out.masked_fill(~has_visible_key[:, :, :, None], 0)
