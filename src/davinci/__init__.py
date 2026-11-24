"""Davinci model and training package."""
from .model import DavinciConfig, DavinciForCausalLM, KVCache, count_parameters
from .train import causal_lm_loss, evaluate_loss, train_microbatch
from .generation import generate, sample_next_token
from .data_pipeline import (
    DatasetManifestError, iter_jsonl, iter_text_records, load_dataset_manifest,
    sha256_file, validate_dataset_manifest, verify_dataset_artifact,
)
from .data_audit import (
    assign_split, content_sha256, deduplicate_records, normalize_persian_text,
    quality_reasons, text_statistics,
)
from .sft import (
    SPECIAL_TOKENS, build_masked_example, causal_sft_loss, collate_masked,
    serialize_messages, split_examples, tool_call_content, tool_result_content,
)
from .preference import (
    PreferenceDataError, dpo_loss, preference_split, sequence_log_probs,
    validate_preference_pair,
)
from .trajectory import audit_trajectories, export_jsonl, split_trajectories, trajectory_hash, validate_trajectory
from .tool_call import (
    ToolCall, ToolCallError, ToolDefinition, ToolExecutionError,
    ToolRegistry, parse_tool_call,
)

__all__ = [
    "DavinciConfig", "DavinciForCausalLM", "KVCache", "count_parameters",
    "causal_lm_loss", "evaluate_loss", "train_microbatch",
    "generate", "sample_next_token",
    "DatasetManifestError", "iter_jsonl", "iter_text_records", "load_dataset_manifest",
    "sha256_file", "validate_dataset_manifest", "verify_dataset_artifact",
    "assign_split", "content_sha256", "deduplicate_records", "normalize_persian_text",
    "quality_reasons", "text_statistics",
    "SPECIAL_TOKENS", "build_masked_example", "causal_sft_loss", "collate_masked",
    "serialize_messages", "split_examples", "tool_call_content", "tool_result_content",
    "PreferenceDataError", "dpo_loss", "preference_split", "sequence_log_probs",
    "validate_preference_pair",
    "audit_trajectories", "export_jsonl", "split_trajectories", "trajectory_hash", "validate_trajectory",
    "ToolCall", "ToolCallError", "ToolDefinition", "ToolExecutionError",
    "ToolRegistry", "parse_tool_call",
]