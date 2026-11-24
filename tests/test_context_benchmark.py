import torch

from experiments.benchmark_context import benchmark_length
from src.davinci.model import DavinciConfig, DavinciForCausalLM


def test_context_benchmark_reports_prefill_decode_and_cache_metrics():
    model = DavinciForCausalLM(DavinciConfig(
        vocab_size=31,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        local_window=8,
        global_every_n_layers=2,
    )).eval()
    result = benchmark_length(model, 12, device=torch.device("cpu"), decode_steps=2)
    assert result["sequence_tokens"] == 12
    assert result["decode_steps"] == 2
    assert result["prefill_seconds"] >= 0
    assert result["decode_seconds_total"] >= 0
    assert result["prefill_tokens_per_second"] > 0
    assert result["decode_tokens_per_second"] > 0
    assert result["kv_cache_bytes"] > 0
    assert result["peak_device_memory_bytes"] is None
