# Preference data and DPO

`src/davinci/preference.py` provides strict preference-pair validation, prompt-grouped deterministic splitting, causal sequence log-probability scoring, and the standard Direct Preference Optimization (DPO) loss.

## Data schema

One JSON object per line:

```json
{
  "pair_id": "optional-stable-id",
  "prompt": "user request",
  "chosen": "preferred response",
  "rejected": "less-preferred response"
}
```

Chosen and rejected responses must be non-empty and different. Preference labels must come from an explicit human or documented evaluation process; the repository does not generate or certify them.

## Objective

For policy and reference sequence log-probabilities, the implementation computes:

\[
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta\left(
\log\frac{\pi_\theta(y_w\mid x)}{\pi_\theta(y_l\mid x)} -
\log\frac{\pi_{ref}(y_w\mid x)}{\pi_{ref}(y_l\mid x)}
\right)\right)\right].
\]

`sequence_log_probs` applies causal shifting and ignores prompt/padding positions marked with `-100` or a false attention mask. It returns summed response log-probabilities and scored-token counts. The DPO objective uses summed sequence log-probabilities; response-length normalization is not silently applied.

## Reproducible split and provenance

Comparisons sharing the same normalized prompt must remain in one split. `preference_split` assigns the normalized prompt hash to a deterministic train/validation partition. The preference builder emits a manifest containing the source identity, immutable revision, explicit license review/approval booleans, notes, and output SHA-256.

Before training, validate the manifest for `purpose="preference"` and verify `preference_jsonl_sha256`. Approval is false by default and must be explicitly reviewed.

## Training notebook

- [Human Preference Dataset Builder](../notebooks/Davinci_Preference_Dataset_Builder.ipynb)
- [DPO Preference Optimization](../notebooks/Davinci_DPO_Preference_Optimization.ipynb)

The notebook uses a frozen reference model and a full-model policy update. It is a correctness-oriented baseline, not an optimized large-model recipe: LoRA/QLoRA, distributed training, reference-model offloading, and automatic preference verification are not implemented. A full 500M run may exceed common free GPU memory.

## Evaluation requirements

Evaluate against held-out prompts and compare with the SFT checkpoint. Report win rate and confidence intervals, Persian and English separately, response quality, refusal behavior, tool-call validity, and regressions on general language tasks. Never treat training preference accuracy as a general model-quality benchmark.