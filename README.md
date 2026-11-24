# Davinci

Davinci is an experimental Persian-first bilingual (Persian/English) language-model research project. The repository contains a proposed ~500M-parameter decoder, data/tokenizer/training notebooks, evaluation utilities, and explicitly experimental long-context retrieval prototypes.

> **Status:** research implementation and training pipeline scaffolding. This repository does not contain a pretrained, instruction-tuned, or validated 500K-context model. Notebook cells and cloud training runs must be executed in the target environment before their outputs can be treated as experimental results.

## Start here

1. Read the architecture and compute assumptions: [500M architecture planning](notebooks/Davinci_500M_Architecture_Compute_Planning.ipynb).
2. Prepare and audit text data: [Persian dataset pipeline](notebooks/Davinci_Persian_Dataset_Pipeline.ipynb).
3. Train and evaluate the tokenizer: [Tokenizer training and evaluation](notebooks/Davinci_Tokenizer_Training_Evaluation.ipynb).
4. Build SFT/reasoning data: [SFT and agent data](notebooks/Davinci_SFT_Reasoning_Agent_Data.ipynb) and [agent trajectory builder](notebooks/Davinci_Agent_Trajectory_Dataset_Builder.ipynb).
5. Inspect the canonical decoder: [`src/davinci/model.py`](src/davinci/model.py) and [runtime documentation](docs/MODEL_RUNTIME.md).
6. Run the small-scale training engine: [pretraining data loader and training engine](notebooks/Davinci_Pretraining_DataLoader_Training_Engine.ipynb). Begin with the notebook's `tiny` preset; do not select the target preset until memory and throughput are measured on the actual accelerator.
7. Evaluate and export: [evaluation harness](notebooks/Davinci_Evaluation_Harness_Baselines.ipynb), [inference/generation/export](notebooks/Davinci_Inference_Generation_Export.ipynb), and [end-to-end tiny integration test](notebooks/Davinci_End_to_End_Tiny_Integration_Test.ipynb).

## Repository map

- `src/davinci/model.py`: canonical PyTorch decoder reference (RMSNorm, RoPE, GQA, SwiGLU, hybrid local/global causal attention, KV cache).
- `src/davinci/train.py`: minimal single-process training/evaluation helpers, accumulation, clipping, and optional CUDA AMP/GradScaler.
- `src/davinci/tool_call.py`: strict JSON tool-call parsing and host-controlled allowlisted dispatch; generated code is never evaluated.
- `src/davinci/long_context.py`: experimental compressed historical KV memory; approximate and not integrated into the canonical decoder.
- `src/davinci/hierarchical_retrieval.py`: experimental two-stage candidate retrieval over compressed keys; approximate and not integrated into the canonical decoder.
- `notebooks/`: data, tokenizer, architecture, training, evaluation, and integration workflows.
- `tests/`: short-sequence correctness and regression tests.
- `experiments/`: synthetic microbenchmarks; these are not model-quality benchmarks.
- `docs/`: runtime and experiment limitations.

## Local setup

Python 3.11+ is recommended. Install PyTorch for your platform and pytest:

```bash
python -m pip install torch pytest
```

Run the model, training, tool-call, notebook, and memory regression tests from the repository root:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_model.py \
  tests/test_train.py \
  tests/test_long_context.py \
  tests/test_hierarchical_retrieval.py \
  tests/test_tool_call.py \
  tests/test_notebook_integrity.py
```

Run the experimental microbenchmarks separately:

```bash
PYTHONPATH=. python experiments/benchmark_compressed_memory.py --device cpu
PYTHONPATH=. python experiments/benchmark_hierarchical_retrieval.py --device cpu
```

The benchmark outputs are synthetic diagnostics. Record the exact hardware, software versions, command-line arguments, and JSON output when comparing changes.

## Architecture snapshot

The proposed target configuration is 28 decoder layers, hidden size 1,280, 20 query heads, 4 key/value heads, head dimension 64, SwiGLU intermediate size 3,328, and a 32K vocabulary. The architecture notebook calculates a parameter estimate; it is not a guarantee of training feasibility on a particular GPU.

The reference decoder uses exact causal sliding-window attention in non-global layers and exact full-history causal attention in designated global layers. Query chunking reduces temporary attention-mask memory, but global attention still has quadratic compute. The current implementation is not a 500K-token context solution. A validated positional-extension method, scalable global retrieval/memory, long-context training, and task-level evaluation are still required.

## Experimental memory prototypes

The compressed-memory and hierarchical-retrieval modules are intentionally separate from the canonical decoder:

- [Compressed memory design and limitations](docs/COMPRESSED_MEMORY_EXPERIMENT.md)
- [Hierarchical retrieval design and limitations](docs/HIERARCHICAL_RETRIEVAL_EXPERIMENT.md)
- [Tool-call protocol and host-side security boundary](docs/TOOL_CALL_PROTOCOL.md)

Both prototypes need representative retrieval evaluation and downstream task-quality tests before integration. Their existence does not establish 500K-context capability.

## Training and data safety

- Audit source dataset licenses, provenance, duplication, language balance, personal data, and synthetic-data contamination before training.
- Keep raw data and credentials out of Git. The Colab training notebook reads a GitHub token from a Colab Secret named `GITHUB_TOKEN`; never paste the token into a notebook cell or commit it.
- Start with the tiny model preset and a small verified dataset. Measure peak allocated/reserved GPU memory, tokens/sec, loss curves, and validation behavior before scaling.
- Only load trusted training checkpoints. The resumable notebook checkpoint uses pickle-backed state.
- Do not publish benchmark numbers unless the run, hardware, software, dataset manifest, and exact code revision are recorded.

## Current verification boundary

Tests are maintained in the repository. Isolated local CPU checks passed for the hierarchical retrieval primitive (7 focused checks) and the strict tool-call protocol (16 focused checks); neither was a full checkout test run. All 17 committed notebooks were fetched and their JSON/nbformat structure was checked; they contain no saved execution outputs. A CI syntax/integrity test now checks notebook JSON and pure-Python code cells, but GitHub Actions currently fails before a runner is assigned (`runner_id=0`, no steps), so the repository test suite has **not** been verified by CI. The Colab notebooks have not been executed end-to-end here. Treat unexecuted notebook stages as instructions/code to run and validate, not completed training results.


## Complete notebook index

All notebooks are stored in `notebooks/` and can be opened in GitHub or Colab (the repository is private, so Colab must have access to your GitHub account).

| Stage | Notebook |
|---|---|
| Architecture and sizing | [500M Architecture and Compute Planning](notebooks/Davinci_500M_Architecture_Compute_Planning.ipynb) |
| Persian corpus preparation | [Persian Dataset Pipeline](notebooks/Davinci_Persian_Dataset_Pipeline.ipynb) |
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
| Inference | [Inference, Generation, and Export](notebooks/Davinci_Inference_Generation_Export.ipynb) |
| Repository checks | [Repository Audit Checks](notebooks/Davinci_Repository_Audit_Checks.ipynb) |
| End-to-end smoke test | [End-to-End Tiny Integration Test](notebooks/Davinci_End_to_End_Tiny_Integration_Test.ipynb) |
| Long-context synthetic eval | [Long Context Curriculum Synthetic Evaluation](notebooks/Davinci_Long_Context_Curriculum_Synthetic_Evaluation.ipynb) |
