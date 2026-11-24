import pytest

from src.davinci.near_dedup import find_near_duplicate_pairs


def test_near_duplicate_audit_finds_high_overlap_candidate():
    base = " ".join(f"token{i}" for i in range(80))
    variant = base.replace("token79", "replacement79")
    result = find_near_duplicate_pairs([
        {"id": "a", "text": base}, {"id": "b", "text": variant}
    ], id_field="id", jaccard_threshold=0.85)
    assert result["records_indexed"] == 2
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["source_row"] == "b"
    assert result["pairs"][0]["duplicate_candidate_of"] == "a"
    assert result["pairs"][0]["jaccard_similarity"] >= 0.85


def test_near_duplicate_audit_uses_persian_normalization():
    result = find_near_duplicate_pairs([
        {"source_row": 1, "text": " ".join(["مي‌روم"] * 8)},
        {"source_row": 2, "text": " ".join(["می‌روم"] * 8)},
    ], jaccard_threshold=0.99)
    assert result["pairs"]


def test_near_duplicate_audit_reports_short_records_and_bucket_overflow():
    rows = [{"source_row": i, "text": "same " * 10} for i in range(4)]
    result = find_near_duplicate_pairs(rows, max_bucket_size=1)
    assert result["too_short"] == 0
    assert result["skipped_overflow_buckets"] > 0


def test_near_duplicate_audit_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="jaccard_threshold"):
        find_near_duplicate_pairs([], jaccard_threshold=0)
    with pytest.raises(ValueError, match="max_hamming_distance"):
        find_near_duplicate_pairs([], max_hamming_distance=65)
    with pytest.raises(ValueError, match="string field"):
        find_near_duplicate_pairs([{"text": None}])