import json

import pytest

from src.davinci.data_pipeline import (
    DatasetManifestError,
    iter_jsonl,
    iter_text_records,
    load_dataset_manifest,
    sha256_file,
    validate_dataset_manifest,
    verify_dataset_artifact,
)


def approved_manifest():
    return {
        "dataset_id": "example/corpus",
        "dataset_revision": "0123456789abcdef",
        "purpose": "pretraining",
        "license_reviewed_by_user": True,
        "license_approved_for_use": True,
        "license_review_notes": "Reviewed the dataset card and upstream terms for this use.",
    }


def test_manifest_requires_explicit_review_and_approval():
    manifest = approved_manifest()
    validate_dataset_manifest(manifest, intended_purpose="pretraining")
    pending = dict(manifest, license_reviewed_by_user=False)
    with pytest.raises(PermissionError, match="not been explicitly reviewed"):
        validate_dataset_manifest(pending)
    unapproved = dict(manifest, license_approved_for_use=False)
    with pytest.raises(PermissionError, match="not been approved"):
        validate_dataset_manifest(unapproved)


def test_manifest_validation_rejects_missing_provenance_and_wrong_purpose():
    with pytest.raises(DatasetManifestError, match="missing required fields"):
        validate_dataset_manifest({"dataset_id": "x"})
    with pytest.raises(DatasetManifestError, match="does not match"):
        validate_dataset_manifest(approved_manifest(), intended_purpose="instruction")


def test_manifest_loader_requires_json_object(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(approved_manifest()), encoding="utf-8")
    assert load_dataset_manifest(path)["dataset_id"] == "example/corpus"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="JSON object"):
        load_dataset_manifest(path)


def test_artifact_hash_is_verified_and_tampering_fails(tmp_path):
    artifact = tmp_path / "data.jsonl"
    artifact.write_text('{"text":"سلام"}\n', encoding="utf-8")
    digest = sha256_file(artifact)
    assert verify_dataset_artifact(artifact, {"combined_jsonl_sha256": digest}) == digest
    artifact.write_text('{"text":"changed"}\n', encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="mismatch"):
        verify_dataset_artifact(artifact, {"combined_jsonl_sha256": digest})


def test_jsonl_iterator_reports_line_and_requires_objects(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"text":"one"}\n\nnot-json\n', encoding="utf-8")
    iterator = iter_jsonl(path)
    assert next(iterator) == {"text": "one"}
    with pytest.raises(ValueError, match="line 3"):
        next(iterator)
    path.write_text('[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        list(iter_jsonl(path))


def test_text_iterator_filters_requested_split(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"text":"train text","split":"train"}\n'
        '{"text":"validation text","split":"validation"}\n',
        encoding="utf-8",
    )
    rows = list(iter_text_records(path, split="train"))
    assert rows == [{"text": "train text", "split": "train"}]
    assert len(list(iter_text_records(path))) == 2


def test_text_iterator_rejects_empty_text(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"text":"  "}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty string"):
        list(iter_text_records(path))
