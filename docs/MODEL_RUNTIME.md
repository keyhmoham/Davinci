# Davinci model reference package

The canonical short-sequence decoder lives in `src/davinci/model.py`. Notebooks should import this module rather than maintaining independent copies of the model.

## Components

- RMSNorm, RoPE, grouped-query attention (GQA), SwiGLU, tied embeddings
- Hybrid schedule: every `global_every_n_layers` layer is global; other layers use a causal local window
- Full-sequence forward and one-token incremental decoding with per-layer KV caches
- JSON config + PyTorch state-dict checkpoint save/load

## Quick start

```python
import torch
from src.davinci import DavinciConfig, DavinciForCausalLM

config = DavinciConfig(
    vocab_size=32000,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=352,
    local_window=128,
    global_every_n_layers=2,
)
model = DavinciForCausalLM(config)
tokens = torch.randint(0, config.vocab_size, (2, 64))
result = model(tokens, labels=tokens)
print(result["loss"])
```

## Tests

From the repository root, with PyTorch and pytest installed:

```bash
PYTHONPATH=. python -m pytest -q tests/test_model.py
```

The tests cover finite forward/loss/gradients, full-forward versus prefill/decode parity, tied embeddings, and checkpoint round-trip. The tests have been added to the repository; they must still be run in the target Colab/runtime before this implementation is considered environment-verified.

## Important limits

This is a correctness-oriented reference, not a production training or serving kernel. Its attention mask is dense and has quadratic sequence-length memory/work characteristics. It is **not** suitable for 500K-token prefill/training. The current RoPE is standard fixed-frequency RoPE; no validated 500K positional-extension method is included. There is no packed-sequence training, distributed training, optimizer/scheduler, gradient checkpointing integration, paged KV cache, or optimized block-sparse attention here.

The default architecture configuration is a project hypothesis. Freeze the tokenizer vocabulary size and positional strategy before training; re-run parameter-count and memory/throughput measurements after any config change.
