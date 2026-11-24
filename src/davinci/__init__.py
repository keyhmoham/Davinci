"""Davinci model and training package."""
from .model import DavinciConfig, DavinciForCausalLM, KVCache, count_parameters
from .train import causal_lm_loss, evaluate_loss, train_microbatch
from .generation import generate, sample_next_token
from .data_audit import (
    assign_split, content_sha256, deduplicate_records, normalize_persian_text,
    quality_reasons, text_statistics,
)
from .sft import (
    SPECIAL_TOKENS, build_masked_example, causal_sft_loss, collate_masked,
    serialize_messages, split_examples, tool_call_content, tool_result_content,
)
from .tool_call import (
    ToolCall, ToolCallError, ToolDefinition, ToolExecutionError,
    ToolRegistry, parse_tool_call,
)

__all__ = [
    "DavinciConfig", "DavinciForCausalLM", "KVCache", "count_parameters",
    "causal_lm_loss", "evaluate_loss", "train_microbatch",
    "generate", "sample_next_token",
    "SPECIAL_TOKENS", "build_masked_example", "causal_sft_loss", "collate_masked",
    "serialize_messages", "split_examples", "tool_call_content", "tool_result_content",
    "assign_split", "content_sha256", "deduplicate_records", "normalize_persian_text",
    "quality_reasons", "text_statistics",
    "ToolCall", "ToolCallError", "ToolDefinition", "ToolExecutionError",
    "ToolRegistry", "parse_tool_call",
]
