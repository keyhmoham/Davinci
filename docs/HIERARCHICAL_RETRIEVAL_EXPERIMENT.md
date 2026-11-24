# Hierarchical retrieval experiment

This is a standalone, approximate two-stage candidate index for the compressed
historical KV entries described in [COMPRESSED_MEMORY_EXPERIMENT.md](COMPRESSED_MEMORY_EXPERIMENT.md).
It is not wired into `Davinci`'s canonical attention path.

## Algorithm

1. Partition the compressed memory keys into contiguous blocks.
2. Compute a mean key for each block and score block means against each query.
3. Select at most `coarse_top_k` blocks.
4. Score individual keys only inside those selected blocks and return the best
   `top_k` entries.

For M memory entries, block size B, Q queries, and C selected blocks, this implementation first rebuilds block centroids in O(MD) work per call, scores approximately M/B centroids per query, and scores at most C*B individual entries per query. It therefore reduces fine-stage query-key comparisons when C*B << M, but centroid construction itself still scans all M keys. A production implementation would need incrementally maintained or cached centroids to avoid rebuilding them for each query batch. This two-stage approximation does not guarantee recall. If all blocks are selected, the result is exact relative to flat dot-product top-k.

## Important limitations

- Mean-key block ranking can exclude a block containing an individually
  relevant key; recall must be measured on representative workloads.
- The implementation computes block means at query time. A production index
  should build/update centroids incrementally and define invalidation semantics.
- This index ranks keys only; it does not perform causal masking or KV attention.
  The caller must apply absolute-position visibility masks before using results.
- No model-quality, throughput, or 500K-context claim is made.
- It is not integrated into the decoder or the existing compressed-memory class.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_hierarchical_retrieval.py
```


## Synthetic benchmark

Run:

```bash
PYTHONPATH=. python experiments/benchmark_hierarchical_retrieval.py
PYTHONPATH=. python experiments/benchmark_hierarchical_retrieval.py --device cpu --entries 4096 --block-size 32 --coarse-top-k 4
```

The report includes the candidate fraction, latency against flat exact top-k,
and recall of the flat exact top-k set. Results depend strongly on query/key
distribution and hardware. The benchmark is synthetic and does not justify
using this index in the model without representative retrieval and downstream
task evaluations.
