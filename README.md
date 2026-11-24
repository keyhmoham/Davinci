# Davinci

See [MODEL_CARD.md](MODEL_CARD.md) for the current research status, [DATASET_CARD_TEMPLATE.md](DATASET_CARD_TEMPLATE.md) for dataset provenance, and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) before publishing weights.

Davinci is an experimental Persian-first bilingual (Persian/English) language-model research project. The repository contains a proposed ~500M-parameter decoder, data/tokenizer/training notebooks, evaluation utilities, and explicitly experimental long-context retrieval prototypes.

> **Status:** research implementation and training pipeline scaffolding. This repository does not contain a pretrained, instruction-tuned, or validated 500K-context model. Notebook cells and cloud training runs must be executed in the target environment before their outputs can be treated as experimental results.

## Start here

1. Read the architecture and compute assumptions: [500M architecture planning](notebooks/Davinci_500M_Architecture_Compute_Planning.ipynb).
2. Prepare and audit text data: [Persian dataset pipeline](notebooks/Davinci_Persian_Dataset_Pipeline.ipynb), then [merge audited pretraining sources](notebooks/Davinci_Corpus_Merge_Builder.ipynb).
3. Train and evaluate the tokenizer: [Tokenizer training and evaluation](notebooks/Davinci_Tokenizer_Training_Evaluation.ipynb).
4. Build SFT/reasoning data: [SFT and agent data](notebooks/Davinci_SFT_Reasoning_Agent_Data.ipynb) and [agent trajectory builder](notebooks/Davinci_Agent_Trajectory_Dataset_Builder.ipynb).
5. Inspect the canonical decoder: [`src/davinci/model.py`](src/davinci/model.py) and [runtime documentation](docs/MODEL_RUNTIME.md).
6. Run the small-scale training engine: [pretraining data loader and training engine](notebooks/Davinci_Pretraining_DataLoader_Training_Engine.ipynb). Begin with the notebook's `tiny` preset; do not select the target preset until memory and throughput are measured on the actual accelerator.
7. Evaluate and export: [evaluation harness](notebooks/Davinci_Evaluation_Harness_Baselines.ipynb), [inference/generation/export](notebooks/Davinci_Inference_Generation_Export.ipynb), and [end-to-end tiny integration test](notebooks/Davinci_End_to_End_Tiny_Integration_Test.ipynb).

## Repository map

- `src/davinci/model.py`: canonical PyTorch decoder reference (RMSNorm, RoPE, GQA, SwiGLU, hybrid local/global causal attention, KV cache), with an exact parameter-count utility.
- `configs/davinci_500m.json`: frozen baseline architecture configuration; `count_parameters` computes 508,960,000 parameters for this bias-free tied-embedding design.
- `src/davinci/train.py`: minimal single-process training/evaluation helpers, accumulation, clipping, and optional CUDA AMP/GradScaler.
- `src/davinci/generation.py`: cached autoregressive generation using canonical prefill/decode, with greedy, top-k, top-p, and EOS handling.
- `src/davinci/tool_call.py`: strict JSON tool-call parsing and host-controlled allowlisted dispatch; generated code is never evaluated.
- `src/davinci/data_audit.py`: deterministic Persian/bilingual normalization, quality diagnostics, exact deduplication, and hash-based dataset splitting.
- `src/davinci/near_dedup.py`: approximate SimHash/LSH candidate discovery with Jaccard verification; emits review candidates rather than deleting data automatically.
- `src/davinci/data_pipeline.py`: strict dataset-manifest validation, explicit license-review gates, artifact SHA-256 verification, and streaming JSONL readers.
- `src/davinci/sft.py`: canonical chat serialization, tool-trace payload formatting, assistant-only target masking, deterministic grouped splits, and SFT loss helpers.
- `src/davinci/preference.py`: preference-pair validation, prompt-grouped deterministic splits, response-only sequence log-probabilities, and the DPO objective.
- `src/davinci/server.py`: optional single-process OpenAI-compatible HTTP/SSE inference API with bearer authentication and request limits.
- `src/davinci/evaluation.py`: token-weighted perplexity, normalized exact match, and explicit language/task-grouped evaluation metrics.
- `src/davinci/trajectory.py`: strict agent-trajectory validation, tool-call/result pairing, exact-content deduplication, leakage-aware splitting, and JSONL export.
- `src/davinci/long_context.py`: experimental compressed historical KV memory; approximate and not integrated into the canonical decoder.
- `src/davinci/hierarchical_retrieval.py`: experimental two-stage candidate retrieval over compressed keys; approximate and not integrated into the canonical decoder.
- `notebooks/`: data, tokenizer, architecture, training, evaluation, and integration workflows.
- `tests/`: short-sequence correctness and regression tests.
- `experiments/`: synthetic microbenchmarks; these are not model-quality benchmarks.
- `docs/`: runtime and experiment limitations.

## Local setup

Python 3.11+ is recommended. Install the editable package and test extras:

```bash
python -m pip install -e ".[test]"
```

Install optional inference-server dependencies when needed:

```bash
python -m pip install -e ".[server]"
```

Run the complete CPU regression suite:

```bash
python -m pytest -q tests
```

Run the experimental microbenchmarks separately:

```bash
PYTHONPATH=. python experiments/benchmark_compressed_memory.py --device cpu
PYTHONPATH=. python experiments/benchmark_hierarchical_retrieval.py --device cpu
```

The benchmark outputs are synthetic diagnostics. Record exact hardware, software versions, command-line arguments, and JSON output when comparing changes.

## Architecture snapshot

The proposed target configuration is 28 decoder layers, hidden size 1,280, 20 query heads, 4 key/value heads, head dimension 64, SwiGLU intermediate size 3,328, and a 32K vocabulary. The architecture notebook calculates a parameter estimate; it is not a guarantee of training feasibility on a particular GPU.

The reference decoder uses exact causal sliding-window attention in non-global layers and exact full-history causal attention in designated global layers. Query chunking reduces temporary attention-mask memory, but global attention still has quadratic compute. The current implementation is not a 500K-token context solution. A validated positional-extension method, scalable global retrieval/memory, long-context training, and task-level evaluation are still required.

## Experimental memory prototypes

The compressed-memory and hierarchical-retrieval modules are intentionally separate from the canonical decoder:

- [Compressed memory design and limitations](docs/COMPRESSED_MEMORY_EXPERIMENT.md)
- [Hierarchical retrieval design and limitations](docs/HIERARCHICAL_RETRIEVAL_EXPERIMENT.md)
- [Tool-call protocol and host-side security boundary](docs/TOOL_CALL_PROTOCOL.md)
- [Dataset audit contracts and limitations](docs/DATA_AUDIT.md)
- [Dataset provenance, manifest, and split contracts](docs/DATA_PIPELINE.md)
- [Cached generation API and limitations](docs/GENERATION.md)
- [DPO preference data and optimization](docs/PREFERENCE_OPTIMIZATION.md)
- [OpenAI-compatible serving](docs/SERVING.md)
- [Evaluation protocol and metrics](docs/EVALUATION.md)

Both prototypes need representative retrieval evaluation and downstream task-quality tests before integration. Their existence does not establish 500K-context capability.

## Training and data safety

- Audit source dataset licenses, provenance, duplication, language balance, personal data, and synthetic-data contamination before training.
- The repository currently has no `LICENSE` file. Public visibility is not a license grant; choose and add an explicit repository license before inviting reuse.
- Keep raw data, private records, and credentials out of Git. The repository is public, so the Colab notebooks clone it without a GitHub token. Use credentials only for separate private storage or writeback operations, and never embed them in notebook source.
- Start with the tiny model preset and a small verified dataset. Measure peak allocated/reserved GPU memory, tokens/sec, loss curves, and validation behavior before scaling.
- Only load trusted training checkpoints. The resumable notebook checkpoint uses pickle-backed state.
- Do not publish benchmark numbers unless the run, hardware, software, dataset manifest, and exact code revision are recorded.

## Current verification boundary

GitHub Actions runs the CPU/Python 3.11 regression suite, including decoder, training, generation, tool-call, data-audit, data-pipeline, trajectory, and notebook-integrity tests. The suite has passed in CI; this verifies repository tests only, not GPU training, Colab execution, dataset quality, or model capability. The committed notebooks have not been executed end-to-end here and contain no verified training outputs. Treat notebook stages as reproducible workflows that still require execution in the target environment, not completed training results.


## Complete notebook index

All notebooks are stored in `notebooks/` and can be opened in GitHub or Colab. The repository is public, so read-only cloning does not require a GitHub token.

| Stage | Notebook |
|---|---|
| Architecture and sizing | [500M Architecture and Compute Planning](notebooks/Davinci_500M_Architecture_Compute_Planning.ipynb) |
| Persian corpus preparation | [Persian Dataset Pipeline](notebooks/Davinci_Persian_Dataset_Pipeline.ipynb) |
| Multi-source corpus merge | [Corpus Merge Builder](notebooks/Davinci_Corpus_Merge_Builder.ipynb) |
| Instruction and reasoning data | [SFT Reasoning and Agent Data](notebooks/Davinci_SFT_Reasoning_Agent_Data.ipynb) |
| Tokenizer | [Tokenizer Training and Evaluation](notebooks/Davinci_Tokenizer_Training_Evaluation.ipynb) |
| Decoder implementation | [PyTorch Model Implementation](notebooks/Davinci_PyTorch_Model_Implementation.ipynb) |
| Pretraining | [Pretraining DataLoader and Training Engine](notebooks/Davinci_Pretraining_DataLoader_Training_Engine.ipynb) |
| Training pilot | [Pretraining Optimization MiniRun](notebooks/Davinci_Pretraining_Optimization_MiniRun.ipynb) |
| Long context | [Long Context Curriculum](notebooks/Davinci_Long_Context_Curriculum_Evaluation.ipynb) |
| Attention and cache | [Efficient Attention and KV Cache](notebooks/Davinci_Efficient_Attention_KV_Cache.ipynb) |
| Runtime | [Runtime Integration Benchmark](notebooks/Davinci_Runtime_Integration_Benchmark.ipynb) |
| Evaluation | [Evaluation Harness and Baselines](notebooks/Davinci_Evaluation_Harness_Baselines.ipynb) |
| Supervised fine-tuning | [SFT Chat Fine-tuning](notebooks/Davinci_SFT_Training_Chat_Finetuning.ipynb) |
| Agent trajectories | [Agent Trajectory Dataset Builder](notebooks/Davinci_Agent_Trajectory_Dataset_Builder.ipynb) |
| Preference data | [Human Preference Dataset Builder](notebooks/Davinci_Preference_Dataset_Builder.ipynb) |
| Preference optimization | [DPO Preference Optimization](notebooks/Davinci_DPO_Preference_Optimization.ipynb) |
| Inference | [Inference, Generation, and Export](notebooks/Davinci_Inference_Generation_Export.ipynb) |
| HTTP serving | [Serving documentation](docs/SERVING.md) |
| Repository checks | [Repository Audit Checks](notebooks/Davinci_Repository_Audit_Checks.ipynb) |
| End-to-end smoke test | [End-to-End Tiny Integration Test](notebooks/Davinci_End_to_End_Tiny_Integration_Test.ipynb) |
| Long-context synthetic eval | [Long Context Curriculum Synthetic Evaluation](notebooks/Davinci_Long_Context_Curriculum_Synthetic_Evaluation.ipynb) |


## Reproducible training checkpoints

The canonical checkpoint API is in [`src/davinci/checkpoint.py`](src/davinci/checkpoint.py), with the trusted-checkpoint contract and resume example in [`docs/TRAINING_CHECKPOINTS.md`](docs/TRAINING_CHECKPOINTS.md). It atomically saves model/optimizer/scheduler/scaler state, Python/NumPy/PyTorch RNG states, optional in-flight gradients, and caller-supplied data-cursor state. The pretraining Colab notebook uses this helper and verifies a run fingerprint before resume. Exact data replay still depends on the caller's deterministic loader and cursor; the current notebook keeps `NUM_WORKERS=0`.

Only load checkpoints produced by a trusted run: optimizer and RNG state require pickle-backed PyTorch serialization. CPU CI tests the serialization/recovery contract; no Colab or GPU training run has been performed here.

## Long-context evaluation and profiling

The experimental synthetic retrieval helpers live in `src/davinci/long_context_eval.py`. They generate deterministic Persian/English needle tasks, measure lengths with the selected tokenizer, and write a JSONL plus SHA-256 manifest. The Colab notebook `notebooks/Davinci_Long_Context_Curriculum_Synthetic_Evaluation.ipynb` uses the trained SentencePiece model at the documented Drive path; it stops if that artifact is missing rather than estimating token counts from characters.

For a checkpoint-specific prefill/decode and KV-cache microbenchmark:

```bash
PYTHONPATH=. python experiments/benchmark_context.py \
  --checkpoint /path/to/davinci-checkpoint \
  --lengths 1024,2048,4096,8192 \
  --decode-steps 16
```

The benchmark uses random token IDs and reports hardware-specific timing and memory only. It does not score language quality, and a successful run is not evidence that 500K context is usable. Increase lengths gradually on the target hardware and record OOMs as results, not as successful measurements.
