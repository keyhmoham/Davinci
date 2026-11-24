# Davinci model card template

> **Status:** research implementation and training workflow. This repository state does not establish that a production checkpoint has been trained or evaluated. Complete every bracketed field from real run artifacts before publishing model weights or making capability claims.

## Model details

- **Name:** Davinci
- **Version / checkpoint ID:** [fill from immutable release]
- **Architecture:** decoder-only causal language model; RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU, tied embeddings, hybrid local/global attention.
- **Target scale:** approximately 500M parameters for the documented 32K-vocabulary configuration; publish the exact measured parameter count for the released tokenizer/config.
- **Languages:** Persian-first with English and code-mixed support as an objective, not a verified capability until measured.
- **Tokenizer:** [SentencePiece algorithm, vocabulary size, model SHA-256, special-token IDs]
- **Context length:** [maximum context length actually trained and independently evaluated]

## Intended use

[Describe the validated research and application use cases.] Do not claim reliable Persian reasoning, tool use, long-context reasoning, or general-purpose performance without task-specific held-out evaluation.

## Out-of-scope / prohibited claims

- No 500K context claim based only on the current hybrid attention implementation. Global attention layers still have quadratic sequence-length compute.
- No capability claim based on synthetic data, tiny-model smoke tests, training loss alone, or repository CI.
- No claim that schema-valid tool calls are safe or correct without host authorization and execution safeguards.

## Training data and provenance

Complete this section from the exact training manifests:

| Stage | Dataset IDs and immutable revisions | License / privacy review | Artifact hashes |
|---|---|---|---|
| Pretraining | [fill] | [fill] | [fill] |
| SFT | [fill] | [fill] | [fill] |
| Preference optimization | [fill or not used] | [fill] | [fill] |

Record filtering, normalization, exact/near deduplication, language/script composition, split method, contamination checks, and known source limitations. Public dataset availability is not proof of permission for model training or redistribution.

## Training procedure

- Repository commit: [fill]
- Model config: [attach exact JSON]
- Tokenizer hash: [fill]
- Optimizer, schedule, precision, batch/sequence length, gradient accumulation: [fill]
- Training tokens and optimizer steps: [fill]
- Hardware and software versions: [fill]
- Checkpoint/resume validation: [fill]
- Known training interruptions or instability: [fill]

## Evaluation

Publish exact dataset revisions, split hashes, prompt templates, evaluator versions, decoding settings, hardware, and confidence intervals.

| Evaluation | Persian | English | Code-mixed | Notes |
|---|---:|---:|---:|---|
| Held-out perplexity | [fill] | [fill] | [fill] | [fill] |
| Instruction following | [fill] | [fill] | [fill] | [fill] |
| Reasoning | [fill] | [fill] | [fill] | [fill] |
| Tool-call validity | [fill] | [fill] | [fill] | [fill] |
| Long-context retrieval by length | [fill] | [fill] | [fill] | [fill] |
| Safety / refusal / privacy | [fill] | [fill] | [fill] | [fill] |

## Limitations and risks

[Document known weaknesses, hallucination behavior, safety and privacy risks, training-data gaps, language/domain coverage, long-context limitations, and inference constraints.]

## Deployment

Use the documented serving API only behind authentication, TLS, network controls, request limits, and monitoring. The reference server does not execute tool calls and does not provide continuous batching, paged attention, or distributed serving.

## Change history

[Record changes between released checkpoints, including tokenizer, architecture, data, training, and evaluation changes.]
