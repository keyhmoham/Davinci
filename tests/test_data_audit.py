import pytest

from src.davinci.data_audit import (
    assign_split,
    content_sha256,
    deduplicate_records,
    normalize_persian_text,
    quality_reasons,
    text_statistics,
)


def test_persian_normalization_preserves_zwnj_and_normalizes_variants():
    value = "ميكـروم\u200f  خوب\r\n\r\n\r\n"
    assert normalize_persian_text(value) == "میکروم خوب"


def test_normalization_is_idempotent():
    value = "كلمه\t\tبعد\n\n\nسطر"
    once = normalize_persian_text(value)
    assert normalize_persian_text(once) == once


def test_statistics_report_scripts_without_claiming_language():
    stats = text_statistics("سلام hello")
    assert stats["arabic_script_alpha_count"] == 4
    assert stats["latin_alpha_count"] == 5
    assert stats["arabic_script_alpha_ratio"] == pytest.approx(4 / 9)


def test_quality_reasons_are_explicit_and_bilingual_safe():
    text = "سلام hello " * 8
    assert quality_reasons(text, min_chars=1, min_alpha_ratio=0.1) == []
    assert "too_short" in quality_reasons("short", min_chars=20)
    assert "replacement_characters" in quality_reasons("abc\ufffd", min_chars=0)


def test_deduplication_uses_normalized_content_and_keeps_first_provenance():
    rows = [
        {"source_row": 10, "text": "ميك"},
        {"source_row": 11, "text": "میک"},
        {"source_row": 12, "text": "متن دیگر"},
    ]
    kept, duplicates = deduplicate_records(rows)
    assert [row["source_row"] for row in kept] == [10, 12]
    assert duplicates[0]["source_row"] == 11
    assert duplicates[0]["duplicate_of"] == 10
    assert len(kept[0]["sha256"]) == 64


def test_split_is_deterministic_and_seeded():
    digest = content_sha256("stable example")
    assert assign_split(digest, seed=7) == assign_split(digest, seed=7)
    assert assign_split(digest, validation_fraction=0.0) == "train"
    assert assign_split(digest, validation_fraction=1.0) == "validation"


@pytest.mark.parametrize("kwargs", [
    {"min_chars": -1},
    {"max_chars": 0, "min_chars": 1},
    {"min_alpha_ratio": 1.1},
])
def test_quality_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        quality_reasons("text", **kwargs)


def test_missing_text_field_and_non_string_text_fail_closed():
    with pytest.raises(ValueError, match="missing"):
        deduplicate_records([{"id": 1}])
    with pytest.raises(TypeError):
        normalize_persian_text(None)
