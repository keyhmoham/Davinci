# Davinci release-readiness checklist

This checklist distinguishes repository engineering from an actually trained/released model. A green CPU CI run does not complete any GPU, dataset, or quality gate.

## Repository engineering

- [x] Canonical model implementation and parameter-count tests.
- [x] Training, generation, evaluation, and strict tool-call primitives.
- [x] Dataset provenance manifests, explicit license gates, SHA-256 verification, deterministic splits, Persian normalization, exact deduplication, and near-duplicate candidate reporting.
- [x] SentencePiece tokenizer training/evaluation notebook with train-split-only input policy and special-token checks.
- [x] Atomic trusted training-state checkpoints, including optimizer/scheduler/scaler, RNG, optional gradients, metadata fingerprints, and caller data cursors.
- [x] Resumable pretraining, SFT, and DPO Colab workflows.
- [x] Prompt/source-grouped SFT splits with exported split hashes; structured tool traces preserve trust labels in context and remain masked from assistant loss.
- [x] DPO notebook preserves prompt context during response truncation and rejects empty response-target masks.
- [x] OpenAI-style single-process HTTP serving reference with streaming, bearer authentication, request limits, and strict request schemas.
- [x] CPU regression suite and notebook syntax/integrity checks.

## Required before any real training run

- [ ] Run each source dataset's card/license/privacy/consent review. Do not set manifest approval fields until that review is complete.
- [ ] Produce immutable raw/processed data manifests; inspect source composition, script ratios, quality rejection counts, duplicates, and near-duplicate candidates.
- [ ] Run tokenizer training in Colab; inspect Persian, English, code-mixed token fertility and special-token IDs; freeze the selected tokenizer.
- [ ] Verify corpus/tokenizer hashes in the pretraining notebook and execute a tiny overfit test.
- [ ] Execute a short GPU pilot; record peak VRAM/RAM, tokens/sec, loss curves, skipped AMP steps, and a kill/restart recovery test.
- [ ] Establish a measured compute budget and a feasible target configuration before allocating the 500M model.

## Required before claiming a trained Davinci model

- [ ] Complete pretraining on the approved corpus and publish the exact checkpoint, code revision, dataset/tokenizer hashes, hardware, and training manifest.
- [ ] Evaluate held-out perplexity and task suites separately for Persian, English, code-mixed text, and relevant domains.
- [ ] Complete SFT only with approved, audited instruction data; verify tool-call serialization and assistant-only loss masking.
- [ ] Complete preference optimization only with genuine human-reviewed comparisons; report held-out preference and capability regressions.
- [ ] Run independent contamination, safety, refusal, privacy, and tool-call validation.
- [ ] Publish model weights, tokenizer, model card, license/provenance notices, evaluation results, and reproducible inference instructions.

## Required before claiming 500K context

- [ ] Replace or augment quadratic global attention with a validated long-context architecture; current global layers remain quadratic in sequence length.
- [ ] Integrate any compressed/hierarchical memory experiment into the decoder only after causal-position, retrieval, and numerical correctness tests.
- [ ] Train/continue-train at increasing lengths and report natural-document retrieval and multi-hop task performance at every claimed length.
- [ ] Measure prefill/decode latency, KV/memory footprint, retrieval recall, and OOM boundaries on the actual target hardware.
- [ ] Do not treat synthetic needle tests or a successful memory microbenchmark as proof of 500K reasoning.

## Current evidence boundary

The repository's automated CI currently validates Python CPU behavior and notebook structure. The Colab notebooks have not been executed end-to-end in this workflow; no trained weights, dataset approvals, GPU throughput, or model-quality results are established here. Those items require the user's reviewed datasets and an available accelerator/runtime.
