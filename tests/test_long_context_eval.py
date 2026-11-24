import json

import pytest

from src.davinci.long_context_eval import (
    build_needle_example,
    measure_token_lengths,
    score_retrieval,
    write_jsonl_and_manifest,
)


def test_needle_examples_are_reproducible_and_stratified():
    a = build_needle_example(3, documents=10, position_fraction=0.5, seed=9)
    b = build_needle_example(3, documents=10, position_fraction=0.5, seed=9)
    assert a == b
    assert a["metadata"]["needle_document_index"] == 5
    fa = build_needle_example(4, documents=10, position_fraction=1.0, seed=9, language="fa")
    assert fa["metadata"]["needle_document_index"] == 10
    assert "شناسه" in fa["prompt"]


def test_needle_example_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        build_needle_example(-1, documents=2, position_fraction=0.5, seed=1)
    with pytest.raises(ValueError):
        build_needle_example(1, documents=2, position_fraction=1.1, seed=1)
    with pytest.raises(ValueError):
        build_needle_example(1, documents=2, position_fraction=0.5, seed=1, language="de")


def test_measure_lengths_uses_actual_tokenizer_output():
    examples = [build_needle_example(1, documents=3, position_fraction=0.0, seed=2)]
    measured = measure_token_lengths(examples, lambda text: text.split())
    assert measured[0]["metadata"]["token_count"] == len(examples[0]["prompt"].split())
    assert "token_count" not in examples[0]["metadata"]


def test_measure_lengths_rejects_empty_tokenizer_output():
    examples = [build_needle_example(1, documents=3, position_fraction=0.0, seed=2)]
    with pytest.raises(ValueError, match="non-empty"):
        measure_token_lengths(examples, lambda text: [])


def test_retrieval_metrics_report_language_and_position_groups():
    examples = [
        build_needle_example(1, documents=3, position_fraction=0.0, seed=1),
        build_needle_example(2, documents=3, position_fraction=1.0, seed=2, language="fa"),
    ]
    result = score_retrieval([examples[0]["answer"], "wrong"], examples)
    assert result["correct"] == 1
    assert result["groups"]["en@0.0"]["accuracy"] == 1.0
    assert result["groups"]["fa@1.0"]["accuracy"] == 0.0


def test_manifest_hash_matches_written_jsonl(tmp_path):
    examples = [build_needle_example(1, documents=3, position_fraction=0.5, seed=3)]
    manifest = write_jsonl_and_manifest(
        examples, tmp_path / "eval.jsonl", tokenizer_id="test-tokenizer@abc", seed=3
    )
    import hashlib

    assert manifest["jsonl_sha256"] == hashlib.sha256((tmp_path / "eval.jsonl").read_bytes()).hexdigest()
    saved = json.loads((tmp_path / "eval.jsonl.manifest.json").read_text())
    assert saved == manifest
    assert saved["synthetic_only"] is True
