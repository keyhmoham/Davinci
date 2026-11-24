# Evaluation protocol

`src/davinci/evaluation.py` provides token-weighted causal language-model loss/perplexity and normalized exact-match metrics. It is intentionally small; benchmark-specific prompt construction and scoring remain in the evaluation notebooks.

## Language-model metrics

`evaluate_language_model(model, batches)` accepts batches as input-ID tensors, `(input_ids, labels)` pairs, or mappings with `input_ids` and optional `labels`. Labels use `-100` for ignored positions. It weights each batch loss by the number of supervised next-token targets rather than averaging batch means equally, restores the model's previous train/eval mode, and reports token count and perplexity.

```python
from src.davinci.evaluation import evaluate_language_model

metrics = evaluate_language_model(model, validation_loader, max_batches=100)
print(metrics)  # loss, perplexity, tokens, batches
```

`exact_match_metrics` applies the repository's Persian text normalization before exact string comparison. This is suitable for tightly specified extraction/synthetic retrieval tasks, not open-ended answer quality.

## Benchmark reporting requirements

Record:

- exact model, tokenizer, and code revisions/hashes;
- dataset ID, immutable revision, license review, and split manifest;
- language and task category, prompt template, and evaluator version;
- decoding parameters, max input/output lengths, and random seed;
- hardware, PyTorch/CUDA versions, peak memory, prefill latency, decode latency, and tokens/sec;
- aggregate metrics and disaggregated Persian, English, and code-mixed results;
- contamination checks, uncertainty, and representative failure cases.

Do not treat synthetic needle retrieval or training-set loss as a comprehensive benchmark. Long-context claims require natural long-document data, position-distributed evidence, multi-hop tasks, and repeated evaluation at each claimed context length.