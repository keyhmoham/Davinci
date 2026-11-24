# Supervised fine-tuning data contract

`src/davinci/sft.py` is the canonical chat serializer and assistant-only masking implementation.

## Canonical message format

- `system`, `user`, and `tool` messages are context only and receive `-100` labels.
- Assistant text and assistant tool-call payloads are supervised, including the assistant structural markers and end-of-turn marker.
- Recorded agent trajectories may represent a tool call structurally (`type: tool_call`, `name`, `arguments`) rather than as a preformatted `content` string. The SFT helper converts these to the canonical `<|tool_call|>` plus strict JSON payload.
- Recorded tool observations (`role: tool`, `name`, `content`) are normalized to `<|tool_result|>` plus a JSON payload when the content is valid JSON; otherwise the observation is represented as a JSON string. Tool observations remain masked from the loss.
- Already-canonical tool-call/result content is not wrapped a second time.

## Leakage-aware splits

`split_examples` groups records by source/document identifiers, exact serialized content, and normalized user prompts. Different assistant answers to the same user prompt therefore remain in one partition. This is a conservative grouping rule; it does not detect paraphrased prompts or semantic leakage. Use immutable source IDs where available and run separate contamination checks.

## Validation

```bash
PYTHONPATH=. python -m pytest -q tests/test_sft.py
```

These tests validate serialization, target masking, structured trajectory conversion, and split grouping. They do not establish that a response is factually correct, that tool observations are trustworthy, or that the source data is licensed for training. Human review and dataset-manifest approval remain mandatory.