# Davinci model card (research status)

## Model status

**Status:** research implementation; no pretrained or instruction-tuned model weights are released by this repository at this time.

Davinci is an experimental Persian-first Persian/English language-model project. The repository contains a proposed ~500M-parameter decoder implementation, data preparation and training workflows, preference optimization primitives, evaluation notebooks, and an optional reference inference server.

## Intended use

- Research and engineering experiments on Persian, English, and code-mixed language modeling.
- Reproducible evaluation of decoder, tokenizer, training, and inference components.

Do not treat this repository as a validated production model or as a model with demonstrated safety, reliability, or 500K-token context capability.

## Architecture

The proposed baseline uses 28 decoder layers, hidden size 1,280, 20 query heads, 4 key/value heads, head dimension 64, SwiGLU intermediate size 3,328, and a 32K vocabulary. The parameter-count utility reports 508,960,000 parameters for the bias-free tied-embedding configuration.

The reference decoder combines exact sliding-window attention with periodic exact global-attention layers. Global layers remain quadratic in sequence length. The experimental compressed-memory and hierarchical-retrieval modules are not integrated into the canonical decoder and are approximate.

## Training status

- Pretraining: workflow implemented; no full 500M training run is claimed.
- Supervised fine-tuning: workflow implemented; no completed SFT checkpoint is claimed.
- Agent/tool training: trajectory validation and data preparation implemented; task correctness requires separate verification.
- Preference optimization: DPO objective and a Colab pilot workflow implemented; no DPO-trained checkpoint is claimed.
- Long context: research prototypes and synthetic evaluations exist; 128K/256K/500K capability is not established.

## Data

No training dataset is bundled. Dataset source, immutable revision, license review, privacy review, preprocessing, deduplication, and split manifests must be recorded for each training run. Missing license metadata is not permission. The project does not automatically approve dataset use.

## Evaluation

Repository CI validates code and unit tests on CPU. It does not validate model quality, GPU performance, Colab notebook execution, data rights, or long-context capability. Publish model-quality results only with exact model/checkpoint hashes, dataset manifests, code revision, hardware, software versions, and reproducible evaluation commands.

## Limitations and risks

- No released weights or independently validated benchmark scores.
- No established 500K context capability.
- No production-grade distributed training or continuous-batching inference server.
- Tool execution must remain host-controlled; the model server does not execute generated tool calls.
- Outputs may be incorrect, biased, unsafe, or fabricated; human review is required for consequential uses.
- Dataset licensing, privacy, and contamination are responsibilities of the dataset operator and model trainer.

## Release artifacts required

Before publishing a checkpoint, attach a completed model card with training configuration, tokenizer identity, exact dataset manifest hashes, compute details, evaluation results, known limitations, and a reproducible inference example.