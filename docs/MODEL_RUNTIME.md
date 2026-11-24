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
from src.davinci import DavinciConfig, DavinciForCausalLM, train_microbatch

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
PYTHONPATH=. python -m pytest -q tests/test_model.py tests/test_train.py
```

The tests cover finite forward/loss/gradients, full-forward versus prefill/decode parity, tied embeddings, checkpoint round-trip, causal loss, real-model optimizer updates with gradient accumulation, and evaluation-mode restoration. The tests have been added to the repository; they must still be run in the target Colab/runtime before this implementation is considered environment-verified.

## Important limits

This is a correctness-oriented reference, not a production training or serving kernel. Its attention mask is dense and has quadratic sequence-length memory/work characteristics. It is **not** suitable for 500K-token prefill/training. The current RoPE is standard fixed-frequency RoPE; no validated 500K positional-extension method is included. The training helpers provide a minimal single-process optimizer-step path, but there is no distributed training, optimizer-state resume orchestration, architecture-integrated gradient checkpointing, paged KV cache, or optimized block-sparse attention here.

The default architecture configuration is a project hypothesis. Freeze the tokenizer vocabulary size and positional strategy before training; re-run parameter-count and memory/throughput measurements after any config change.


## Training helper API

`src/davinci/train.py` provides:

- `causal_lm_loss(logits, input_ids)`: next-token cross-entropy with an optional ignored-label index.
- `train_microbatch(...)`: one backward pass, optional CUDA autocast/scaler, gradient accumulation, clipping, and optimizer stepping at the end of an accumulation window.
- `evaluate_loss(...)`: bounded evaluation that restores the model's previous train/eval mode.

Example:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
for microbatch_index, token_batch in enumerate(batches):
    result = train_microbatch(
        model, token_batch, optimizer,
        gradient_accumulation_steps=4,
        microbatch_index=microbatch_index,
        amp_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    if result["optimizer_step"]:
        print("loss", float(result["loss"]))
```

The helper expects the caller to zero gradients before starting an accumulation window and to ensure the final window is complete. It does not implement a full resumable training controller.
