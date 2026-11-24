"""Davinci model and training package."""
from .model import DavinciConfig, DavinciForCausalLM, KVCache
from .train import causal_lm_loss, evaluate_loss, train_microbatch
from .generation import generate, sample_next_token
from .data_audit import (
    assign_split, content_sha256, deduplicate_records, normalize_persian_text,
    quality_reasons, text_statistics,
)
from .tool_call import (
    ToolCall, ToolCallError, ToolDefinition, ToolExecutionError,
    ToolRegistry, parse_tool_call,
)

__all__ = [
    "DavinciConfig", "DavinciForCausalLM", "KVCache",
    "causal_lm_loss", "evaluate_loss", "train_microbatch",
    "generate", "sample_next_token",
    "assign_split", "content_sha256", "deduplicate_records", "normalize_persian_text",
    "quality_reasons", "text_statistics",
    "ToolCall", "ToolCallError", "ToolDefinition", "ToolExecutionError",
    "ToolRegistry", "parse_tool_call",
]
