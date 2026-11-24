# Dataset audit primitives

`src/davinci/data_audit.py` provides reusable, dependency-light functions for a
repeatable Persian/bilingual corpus preparation pipeline.

## Contract

- Normalize Unicode to NFC; map Arabic Yeh/Kaf variants to Persian forms; remove
  tatweel and directional controls; preserve ZWNJ, punctuation, and digits.
- Report character, word, alphabetic, Arabic-script, Latin-script, and replacement
  character statistics. Script ratios are diagnostics, **not language identification**.
- Return explicit heuristic quality-rejection reasons. No Persian-only threshold is
  applied, so useful English examples in a bilingual corpus are not discarded.
- Deduplicate on normalized UTF-8 SHA-256 and preserve the first row's provenance.
- Assign train/validation membership deterministically from content hash and seed,
  preventing identical content from leaking across splits when the same digest is used.

## Usage

```python
from src.davinci.data_audit import (
    normalize_persian_text, quality_reasons, deduplicate_records, assign_split,
)

text = normalize_persian_text(raw_text)
reasons = quality_reasons(text, min_chars=40)
if not reasons:
    # Keep source ID, source revision, source row, license review state, and hash.
    ...
```

## Deliberate limits

These helpers do not fetch datasets, inspect licenses, identify language, detect
near-duplicates, remove personal information, or certify data for training. Dataset
revision/config/split and text-column selection must be explicit. A human must review
the upstream dataset card and every applicable license. Keep raw source records
separate from normalized derivatives and retain a provenance manifest.
