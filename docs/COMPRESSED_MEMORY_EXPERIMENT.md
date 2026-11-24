# Experimental compressed historical KV memory

This experiment is separate from the canonical decoder. It is not a drop-in replacement for exact global attention and is not enabled by default.

## Method
- Keep the newest recent window of keys and values exactly.
- Mean-pool complete older contiguous groups.
- Store each group's final absolute token position.
- Retrieve up to top-k historical groups per query/head by scaled query-key similarity.
- Apply causal masking to retrieved groups and exact recent tokens before scaled dot-product attention.

The method is approximate: pooled K/V entries are not equivalent to attending to every original token. Retrieval may miss relevant information. There is no learned compressor, learned retriever, memory-position RoPE correction, or canonical decoder integration.

## Complexity and limitations
Let T be total history, W the recent window, G the compression group size, and K the retrieval count. Stored historical entries are approximately (T-W)/G. Retrieval currently scores against every compressed entry, so retrieval cost remains linear in the number of memory groups. The attention stage uses at most W+K entries per query.

This is a short-sequence research prototype, not evidence of 500K-context capability. Benchmark memory, latency, retrieval recall, and downstream task accuracy before integration.

## Tests
Run: PYTHONPATH=. python -m pytest -q tests/test_long_context.py
