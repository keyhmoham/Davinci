# Cached generation

The canonical cached generation helper is `src/davinci/generation.py`. It uses
`model.prefill()` once and `model.decode()` for subsequent tokens. The generic
full-prefix generator in Notebook 14 remains a slow reference baseline.

## Behavior

- Greedy decoding when `temperature <= 0`; otherwise temperature sampling.
- Optional top-k and nucleus (top-p) filtering.
- Optional EOS handling. In batched generation, finished rows receive EOS while
  unfinished rows continue; generation stops when every row is finished.
- The model's original train/eval mode is restored after generation.
- Output includes the prompt and generated continuation.

This is a simple single-request reference helper, not a serving engine. It does
not implement streaming, continuous batching, beam search, speculative decoding,
repetition penalties, stop-string matching, structured output constraints, or
tool authorization. Use the separate strict [tool-call protocol](TOOL_CALL_PROTOCOL.md)
for host-controlled tool execution.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_generation.py
```
