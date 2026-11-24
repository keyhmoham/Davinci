# Davinci release checklist

Do not publish model weights until the relevant checks are complete and reproducible.

## Data and rights

- [ ] Every training dataset has a completed dataset card and immutable revision.
- [ ] Every applicable upstream license and intended use has been reviewed.
- [ ] Privacy, consent, retention, and personal-data checks are documented.
- [ ] Exact and near-duplicate checks are recorded.
- [ ] Train/validation/test leakage checks are recorded.
- [ ] Dataset and split artifacts have SHA-256 hashes.

## Tokenizer and model

- [ ] Tokenizer selection is based on Persian, English, mixed-language, and code evaluation.
- [ ] Tokenizer vocabulary and special-token IDs are frozen and recorded.
- [ ] Model configuration and exact parameter count are recorded.
- [ ] Training checkpoints can be resumed and verified.
- [ ] Training run manifest records code revision, data hashes, tokenizer hash, hardware, and software versions.

## Evaluation

- [ ] Held-out Persian and English language-model evaluations are complete.
- [ ] Instruction-following and safety evaluations are complete.
- [ ] Tool-call validity and host-boundary tests are complete.
- [ ] Long-context performance is tested at every claimed context length.
- [ ] Comparisons against baselines are reproducible and not based only on synthetic data.
- [ ] Results include limitations, uncertainty, and failure cases.

## Serving and security

- [ ] Model and tokenizer lineage is checked at load time.
- [ ] Authentication, TLS, rate limits, and request limits are configured.
- [ ] Tool execution is isolated from model inference and host-controlled.
- [ ] Logs redact secrets and sensitive user content.
- [ ] Concurrency, memory limits, shutdown, and failure behavior are tested.

## Publication

- [ ] Model card is complete and accurately describes training/evaluation status.
- [ ] Dataset cards and required attributions are published as permitted.
- [ ] Model weights, tokenizer, config, and checksums are versioned.
- [ ] License and usage terms are explicit.
- [ ] A reproducible inference example and release notes are included.
- [ ] CI is green on the exact release commit.