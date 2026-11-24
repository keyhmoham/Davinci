"""Canonical Davinci decoder reference; chunking does not make global attention 500K-scalable."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DavinciConfig:
    vocab_size: int = 32000
    hidden_size: int = 1280
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 3328
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    local_window: int = 4096
    global_every_n_layers: int = 4
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    attention_chunk_size: int = 128
    global_attention_chunk_size: int = 64

    def validate(self):
        integer_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "local_window": self.local_window,
            "global_every_n_layers": self.global_every_n_layers,
            "attention_chunk_size": self.attention_chunk_size,
            "global_attention_chunk_size": self.global_attention_chunk_size,
        }
        invalid = [
            name for name, value in integer_fields.items()
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ]
        if invalid:
            raise ValueError(
                f"Configuration values must be positive integers: {', '.join(invalid)}"
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for split-half RoPE")
        for name, value in (("norm_eps", self.norm_eps), ("rope_theta", self.rope_theta),
                            ("rope_scaling_factor", self.rope_scaling_factor)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        if (not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool)
                or not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be a finite number in [0, 1)")
        for name, value in (
            ("tie_word_embeddings", self.tie_word_embeddings),
            ("gradient_checkpointing", self.gradient_checkpointing),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

    @classmethod
    def from_dict(cls, values):
        obj = cls(**values)
        obj.validate()
        return obj


def count_parameters(config: DavinciConfig) -> int:
    """Return the exact parameter count implied by this bias-free decoder config."""
    if isinstance(config, dict):
        config = DavinciConfig.from_dict(config)
    if not isinstance(config, DavinciConfig):
        raise TypeError("config must be a DavinciConfig or config dictionary")
    config.validate()
    d, f = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * config.head_dim
    embedding = config.vocab_size * d
    attention = d * d + 2 * d * kv + d * d
    mlp = 3 * d * f
    layer_norms = 2 * d
    output_head = 0 if config.tie_word_embeddings else config.vocab_size * d
    return embedding + config.num_hidden_layers * (attention + mlp + layer_norms) + d + output_head


def apply_rope(x, positions, theta, scaling_factor=1.0):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    if (not isinstance(scaling_factor, (int, float)) or isinstance(scaling_factor, bool)
            or not math.isfinite(scaling_factor) or scaling_factor <= 0):
        raise ValueError("scaling_factor must be a finite positive number")
    angles = torch.outer(positions.float() / float(scaling_factor), inv)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(x.dtype)[None,None], angles.sin().to(x.dtype)[None,None]
    half = d // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)

class KVCache:
    """One layer's KV cache; local caches are bounded during incremental decode."""
    def __init__(self, window: Optional[int]):
        self.window, self.key, self.value, self.positions = window, None, None, None
    def append(self, key, value, positions, trim=True):
        if self.key is None:
            self.key, self.value, self.positions = key, value, positions
        else:
            self.key = torch.cat((self.key, key), dim=2)
            self.value = torch.cat((self.value, value), dim=2)
            self.positions = torch.cat((self.positions, positions), dim=0)
        # During multi-token prefill, retain all keys until attention is computed:
        # deeper-layer hidden states for early prompt tokens are still required.
        if trim:
            self.trim()
        return self.key, self.value, self.positions

    def trim(self):
        if self.window is not None and self.key is not None and self.key.shape[2] > self.window:
            self.key, self.value = self.key[:,:,-self.window:,:], self.value[:,:,-self.window:,:]
            self.positions = self.positions[-self.window:]

class GQAAttention(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.c = c
        self.heads, self.kv_heads = c.num_attention_heads, c.num_key_value_heads
        self.groups, self.head_dim = self.heads // self.kv_heads, c.head_dim
        self.is_global = (layer_idx + 1) % c.global_every_n_layers == 0
        self.window = c.local_window
        d = c.hidden_size
        self.q_proj = nn.Linear(d, self.heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads*self.head_dim, d, bias=False)
    def forward(self, x, positions, cache=None):
        b,t,_ = x.shape
        q = self.q_proj(x).view(b,t,self.heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(b,t,self.kv_heads,self.head_dim).transpose(1,2)
        q, k = apply_rope(q, positions, self.c.rope_theta, self.c.rope_scaling_factor), apply_rope(k, positions, self.c.rope_theta, self.c.rope_scaling_factor)
        if cache is not None:
            expected_window = None if self.is_global else self.window
            if cache.window != expected_window:
                raise ValueError("Cache type does not match attention layer")
            k, v, key_positions = cache.append(k, v, positions, trim=(t == 1))
        else:
            key_positions = positions
        if self.groups != 1:
            k, v = k.repeat_interleave(self.groups,1), v.repeat_interleave(self.groups,1)
        dropout_p = self.c.dropout if self.training else 0.0
        if self.is_global:
            if cache is not None:
                # Cached decode has fewer queries than keys; build only this small query mask.
                delta = positions[:,None] - key_positions[None,:]
                allowed = delta >= 0
                y = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,
                    dropout_p=dropout_p,is_causal=False)
            else:
                # Exact global attention in query chunks. This bounds the explicit
                # causal mask to [global_chunk_size, sequence], while retaining
                # every key/value and exactly the same attention semantics.
                pieces = []
                chunk = self.c.global_attention_chunk_size
                for start in range(0,t,chunk):
                    end = min(start+chunk,t)
                    qpos = positions[start:end]
                    last = end  # positions are contiguous arange(T) in full-sequence forward
                    kp = key_positions[:last]
                    allowed = qpos[:,None] >= kp[None,:]
                    pieces.append(F.scaled_dot_product_attention(q[:,:,start:end,:],
                        k[:,:,:last,:],v[:,:,:last,:],attn_mask=allowed,
                        dropout_p=dropout_p,is_causal=False))
                y = torch.cat(pieces,dim=2)
        else:
            # Exact causal sliding-window attention. Chunking bounds the temporary
            # attention mask to [chunk_size, local_window + chunk_size], not [T,T].
            pieces = []
            chunk = self.c.attention_chunk_size
            for start in range(0,t,chunk):
                end = min(start+chunk,t)
                qpos = positions[start:end]
                if cache is None:
                    first, last = max(0,start-self.window+1), end
                else:
                    lower = qpos[0] - self.window + 1
                    first = int(torch.searchsorted(key_positions,lower,right=False).item())
                    last = int(torch.searchsorted(key_positions,qpos[-1],right=True).item())
                kc,vc = k[:,:,first:last,:],v[:,:,first:last,:]
                kp = key_positions[first:last]
                delta = qpos[:,None] - kp[None,:]
                allowed = (delta >= 0) & (delta < self.window)
                pieces.append(F.scaled_dot_product_attention(q[:,:,start:end,:],kc,vc,
                    attn_mask=allowed,dropout_p=dropout_p,is_causal=False))
            y = torch.cat(pieces,dim=2)
        # Prefill needs the full prompt for every query, then can evict old local KV.
        if cache is not None and t > 1:
            cache.trim()
        return self.o_proj(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d,f,bias=False), nn.Linear(d,f,bias=False), nn.Linear(f,d,bias=False)
    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))

class DecoderBlock(nn.Module):
    def __init__(self,c,i):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(c.hidden_size,c.norm_eps), GQAAttention(c,i)
        self.mlp_norm, self.mlp = RMSNorm(c.hidden_size,c.norm_eps), SwiGLU(c.hidden_size,c.intermediate_size)
    def forward(self,x,positions,cache=None):
        x = x + self.attn(self.attn_norm(x),positions,cache)
        return x + self.mlp(self.mlp_norm(x))

class DavinciForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = DavinciConfig.from_dict(config) if isinstance(config,dict) else config
        self.config.validate()
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(c,i) for i in range(c.num_hidden_layers)])
        self.final_norm = RMSNorm(c.hidden_size,c.norm_eps)
        self.lm_head = nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
    def make_caches(self):
        return [KVCache(None if b.attn.is_global else self.config.local_window) for b in self.layers]
    def forward(self,input_ids,labels=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        pos = torch.arange(input_ids.shape[1],device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for block in self.layers:
            if self.training and self.config.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x,pos)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            if input_ids.shape[1] < 2:
                raise ValueError("At least two tokens are required for next-token loss")
            targets = labels[:,1:].contiguous().view(-1)
            if not torch.any(targets != -100):
                raise ValueError("labels contain no next-token targets (all are ignore_index)")
            loss = F.cross_entropy(logits[:,:-1,:].contiguous().view(-1,logits.shape[-1]),
                                   targets,ignore_index=-100)
        return {"logits":logits,"loss":loss}
    @torch.no_grad()
    def prefill(self,input_ids):
        caches, pos = self.make_caches(), torch.arange(input_ids.shape[1],device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for block,cache in zip(self.layers,caches):
            x = block(x,pos,cache)
        return self.lm_head(self.final_norm(x[:,-1:,:])),caches
    @torch.no_grad()
    def decode(self,input_ids,position,caches):
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode accepts one token per call")
        pos = torch.tensor([position],device=input_ids.device)
        x = self.embed_tokens(input_ids)
        for block,cache in zip(self.layers,caches):
            x = block(x,pos,cache)
        return self.lm_head(self.final_norm(x))
    def save_pretrained(self,directory):
        import json
        from pathlib import Path
        p=Path(directory); p.mkdir(parents=True,exist_ok=True)
        (p/"config.json").write_text(json.dumps(asdict(self.config),indent=2),encoding="utf-8")
        torch.save(self.state_dict(),p/"model.pt")
    @classmethod
    def from_pretrained(cls,directory,map_location="cpu"):
        import json
        from pathlib import Path
        p=Path(directory)
        c=DavinciConfig.from_dict(json.loads((p/"config.json").read_text(encoding="utf-8")))
        model=cls(c)
        model.load_state_dict(torch.load(p/"model.pt",map_location=map_location,weights_only=True))
        return model
