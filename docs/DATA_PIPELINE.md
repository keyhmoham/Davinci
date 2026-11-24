# Dataset pipeline contracts

The repository separates deterministic text preparation from legal and policy decisions. The code can validate that a review was recorded; it cannot determine whether a dataset license actually permits a particular use.

## Manifest requirements

Training manifests must record:

- `dataset_id` and an immutable `dataset_revision`;
- the intended `purpose` (for example, `pretraining` or `instruction`);
- `license_reviewed_by_user` and `license_approved_for_use` as explicit booleans;
- a non-empty `license_review_notes` entry describing the human review;
- artifact SHA-256 values for every exported data file.

Approval must remain false until a person has reviewed the dataset card, upstream components, commercial-use terms, redistribution restrictions, privacy implications, and the exact intended use. Missing license metadata means unknown, not permission.

## Reproducible corpus split

The Persian dataset notebook normalizes text, performs exact normalized-text deduplication, then assigns each retained document to a split using:

```python
bucket = int(SHA256(f"{seed}:{content_sha256}")[:16], 16)
```

The document is validation data when the bucket is below the configured fraction of the 64-bit bucket range; otherwise it is training data. The combined JSONL and the separate train/validation JSONL files are fingerprinted in the manifest. The tokenizer notebook reads only records marked `split=train` from this corpus. The pretraining notebook independently derives the same split from normalized content so that held-out validation documents are not used to train the tokenizer or model.

## Python API

```python
from src.davinci.data_pipeline import (
    load_dataset_manifest, validate_dataset_manifest,
    verify_dataset_artifact, iter_jsonl, iter_text_records,
)

manifest = load_dataset_manifest("dataset_manifest.json")
validate_dataset_manifest(manifest, intended_purpose="pretraining")
verify_dataset_artifact("corpus.jsonl", manifest)
train_records = iter_text_records("corpus.jsonl", split="train")
```

The JSONL readers are streaming and fail closed on malformed JSON, non-object records, or missing/empty text. Artifact verification streams the file while computing SHA-256 and does not load the corpus into memory.

## Limits

- Exact deduplication does not detect paraphrases or near-duplicates.
- Script ratios are not language identification.
- A manifest does not prove the upstream data is accurate, lawful, private, or free of contamination.
- A hash proves byte identity relative to a manifest, not the quality of the data.
- The Colab notebooks still require execution against the selected dataset and tokenizer in the target runtime. CI validates Python tests and notebook syntax, not external Hub downloads or GPU training.