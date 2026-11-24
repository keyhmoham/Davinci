# Reproducible training checkpoints

The canonical helper is `src.davinci.checkpoint`.

```python
from src.davinci.checkpoint import save_training_checkpoint, load_training_checkpoint

save_training_checkpoint(
    "run/checkpoint.pt", model, optimizer,
    scheduler=scheduler, scaler=scaler,
    step=optimizer_step, micro_step=micro_step,
    metadata={
        "corpus_sha256": corpus_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "training_config_sha256": training_config_sha256,
    },
    data_state={"epoch": epoch, "record_offset": record_offset},
)
state = load_training_checkpoint(
    "run/checkpoint.pt", model, optimizer,
    scheduler=scheduler, scaler=scaler,
    expected_metadata={"corpus_sha256": corpus_sha256},
)
optimizer_step = state["step"]
micro_step = state["micro_step"]
```

The file is written to a temporary sibling and atomically replaced. It stores model,
optimizer, optional scheduler/scaler, Python/NumPy/Torch CPU and CUDA RNG states,
optional parameter gradients (needed to resume inside gradient accumulation), and
caller-supplied data cursor state. The caller remains responsible for making its
data iterator deterministic and restoring the cursor/shuffle buffer exactly.
For the first Colab training runs, keep `NUM_WORKERS=0`.

**Security:** checkpoints contain Python pickle-backed optimizer and RNG objects.
Loading uses `weights_only=False`; only load files produced by a trusted run.
Do not download and load arbitrary checkpoint files.

Metadata is caller-defined. At minimum, bind a checkpoint to corpus, tokenizer,
and training-configuration SHA-256 values. This helper checks only the keys passed
through `expected_metadata`; it does not compute those hashes or prove dataset
license approval.

This feature does not make training distributed or guarantee bitwise determinism
across different PyTorch/CUDA versions, hardware, or nondeterministic kernels.
