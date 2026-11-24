import json
import pytest

from src.davinci.trajectory import (
    audit_trajectories, export_jsonl, split_trajectories, trajectory_hash,
    validate_trajectory,
)


def valid_trace(identifier="t1", source=None):
    item = {
        "trajectory_id": identifier,
        "language": "fa",
        "messages": [
            {"role": "user", "content": "محاسبه کن"},
            {"role": "assistant", "type": "tool_call", "name": "add", "arguments": {"a": 2, "b": 3}},
            {"role": "tool", "name": "add", "content": "{\"sum\":5}", "trusted": False},
            {"role": "assistant", "content": "پاسخ: ۵"},
        ],
        "metadata": {"source": "verified", "verified": True},
    }
    if source is not None:
        item["source_document_id"] = source
    return item


def test_valid_tool_trajectory_and_pairing():
    assert validate_trajectory(valid_trace()) == []
    broken = valid_trace()
    broken["messages"][2]["name"] = "other"
    assert any("does not match" in issue for issue in validate_trajectory(broken))


def test_rejects_missing_result_and_untrusted_flag():
    broken = valid_trace()
    broken["messages"].pop(2)
    assert any("without a result" in issue for issue in validate_trajectory(broken))
    broken = valid_trace()
    del broken["messages"][2]["trusted"]
    assert any("trusted" in issue for issue in validate_trajectory(broken))


def test_tool_arguments_must_be_strict_json():
    broken = valid_trace()
    broken["messages"][1]["arguments"] = {"bad": float("nan")}
    assert any("strict JSON" in issue for issue in validate_trajectory(broken))


def test_audit_reports_invalid_and_duplicate_rows():
    first = valid_trace("one")
    second = valid_trace("one")
    broken = valid_trace("bad")
    broken["messages"][-1]["role"] = "tool"
    report = audit_trajectories([first, second, broken])
    assert report["valid"] == 2
    assert report["invalid"] == 1
    assert report["duplicates"] == 1
    assert report["duplicate_rows"][0]["duplicate_of"] == 0


def test_split_keeps_exact_duplicates_and_source_document_groups_together():
    a = valid_trace("a", source="doc-a")
    b = valid_trace("b", source="doc-a")
    duplicate = valid_trace("duplicate", source="doc-b")
    other = valid_trace("other")
    other["messages"][0]["content"] = "different task"
    other["messages"][-1]["content"] = "different answer"
    train, validation = split_trajectories([a, b, duplicate, other], validation_fraction=0.5, seed=42)
    train_ids = {x["trajectory_id"] for x in train}
    val_ids = {x["trajectory_id"] for x in validation}
    assert ({"a", "b", "duplicate"} <= train_ids) or ({"a", "b", "duplicate"} <= val_ids)
    assert train_ids.isdisjoint(val_ids)


def test_export_writes_newline_delimited_valid_unique_records(tmp_path):
    first = valid_trace("a")
    duplicate = valid_trace("a")
    invalid = valid_trace("bad")
    invalid["messages"].pop(2)
    path = tmp_path / "nested" / "traces.jsonl"
    assert export_jsonl([first, duplicate, invalid], path) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trajectory_id"] == "a"
