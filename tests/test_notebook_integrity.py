"""Lightweight structural/syntax audit for committed Jupyter notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _pure_python_source(source: str) -> str | None:
    lines = source.splitlines()
    if any(line.lstrip().startswith(("%%", "%", "!")) for line in lines):
        return None  # IPython/shell syntax is runtime-specific; validate notebook JSON instead.
    return source


def test_all_notebooks_are_valid_nbformat_json():
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert paths, "No notebooks found"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("nbformat") == 4, f"{path.name}: expected nbformat 4"
        assert isinstance(data.get("cells"), list), f"{path.name}: cells must be a list"
        for index, cell in enumerate(data["cells"]):
            assert cell.get("cell_type") in {"code", "markdown", "raw"}, (
                f"{path.name}: cell {index} has invalid cell_type"
            )
            assert isinstance(cell.get("source"), (str, list)), (
                f"{path.name}: cell {index} source must be text or text lines"
            )


def test_pure_python_notebook_cells_compile():
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(data["cells"]):
            if cell.get("cell_type") != "code":
                continue
            raw = cell.get("source", "")
            source = "".join(raw) if isinstance(raw, list) else raw
            source = _pure_python_source(source)
            if source is None or not source.strip():
                continue
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                raise AssertionError(
                    f"{path.name}, code cell {index}: {exc}"
                ) from exc


def _notebook_source(name: str) -> str:
    path = NOTEBOOKS / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    )


def test_pretraining_notebook_defines_and_checks_corpus_manifest():
    source = _notebook_source("Davinci_Pretraining_DataLoader_Training_Engine.ipynb")
    assert "CORPUS_MANIFEST_PATH =" in source
    assert "CORPUS_MANIFEST_PATH.is_file()" in source
    assert 'get("license_approved_for_use") is not True' in source
    assert '"corpus_manifest_sha256":file_sha256(CORPUS_MANIFEST_PATH)' in source


def test_public_colab_notebooks_do_not_require_github_secrets():
    names = (
        "Davinci_Pretraining_DataLoader_Training_Engine.ipynb",
        "Davinci_SFT_Training_Chat_Finetuning.ipynb",
        "Davinci_Agent_Trajectory_Dataset_Builder.ipynb",
    )
    for name in names:
        source = _notebook_source(name)
        assert "GITHUB_TOKEN" not in source, f"{name} still requires a private-repository token"
        assert "https://github.com/keyhmoham/Davinci.git" in source


def test_sft_fingerprint_and_run_manifest_record_dataset_hash():
    source = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    assert source.count('"dataset_manifest_sha256"') == 2  # run fingerprint and exported manifest
    assert "sft_manifest_hash=file_sha256(SFT_MANIFEST_PATH)" in source


def test_corpus_pipeline_exports_fingerprinted_deterministic_splits():
    source = _notebook_source("Davinci_Persian_Dataset_Pipeline.ipynb")
    assert "def stable_split(" in source
    assert 'row["split"] = stable_split(row["sha256"])' in source
    assert "_train.jsonl" in source and "_validation.jsonl" in source
    assert '"train_jsonl_sha256"' in source
    assert '"validation_jsonl_sha256"' in source
    assert '"license_approved_for_use": LICENSE_APPROVED_FOR_USE' in source


def test_tokenizer_uses_only_training_partition_of_split_corpus():
    source = _notebook_source("Davinci_Tokenizer_Training_Evaluation.ipynb")
    assert '"allowed_splits":["train"]' in source
    assert 'obj.get(split_field) not in allowed_splits' in source


def test_preference_builder_keeps_license_approval_opt_in():
    source = _notebook_source("Davinci_Preference_Dataset_Builder.ipynb")
    assert "LICENSE_REVIEWED=False" in source
    assert "LICENSE_APPROVED_FOR_USE=False" in source
    assert "'purpose':'preference'" in source
    assert "'preference_jsonl_sha256':sha256_file(OUTPUT_JSONL)" in source


def test_dpo_notebook_checks_manifest_and_uses_frozen_reference():
    source = _notebook_source("Davinci_DPO_Preference_Optimization.ipynb")
    assert "validate_dataset_manifest(manifest,intended_purpose='preference')" in source
    assert "verify_dataset_artifact(PREFERENCE_JSONL,manifest" in source
    assert "reference.eval()" in source
    assert "parameter.requires_grad_(False)" in source
    assert "sequence_log_probs" in source


def test_sft_split_is_shared_by_tokenizer_and_finetuning():
    builder = _notebook_source("Davinci_SFT_Reasoning_Agent_Data.ipynb")
    tokenizer = _notebook_source("Davinci_Tokenizer_Training_Evaluation.ipynb")
    trainer = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    assert 'record["split"]=assign_chat_split(record)' in builder
    assert '"split_field":"split","allowed_splits":["train"]' in tokenizer
    assert 'item["split"] not in {"train","validation"}' in trainer
    assert 'item for item in examples if item["split"]=="validation"' in trainer


def test_pretraining_uses_audited_manifest_split_labels():
    corpus = _notebook_source("Davinci_Persian_Dataset_Pipeline.ipynb")
    pretraining = _notebook_source("Davinci_Pretraining_DataLoader_Training_Engine.ipynb")
    assert "\\u2066-\\u2069" in corpus
    assert 'VALIDATION_FRACTION=float(manifest_validation_fraction)' in pretraining
    assert 'obj.get("split")==self.split' in pretraining


def test_sft_and_dpo_runs_verify_upstream_tokenizer_lineage():
    sft = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    dpo = _notebook_source("Davinci_DPO_Preference_Optimization.ipynb")
    assert "PRETRAINING_RUN_MANIFEST=BASE_MODEL_DIR.parent" in sft
    assert 'PRETRAINING_MANIFEST.get("tokenizer_sha256")!=sha256_file(TOKENIZER_MODEL)' in sft
    assert "SFT_RUN_MANIFEST=BASE_SFT_DIR.parent" in dpo
    assert "sft_run_manifest.get('tokenizer_sha256')!=sha256_file(TOKENIZER_MODEL)" in dpo
    assert "SFT_RUN_MANIFEST_SHA256=sha256_file(SFT_RUN_MANIFEST)" in dpo


def test_long_context_notebook_uses_exported_selected_tokenizer():
    source = _notebook_source("Davinci_Long_Context_Curriculum_Evaluation.ipynb")
    assert "davinci_sentencepiece.model" in source
    assert "davinci_unigram_32000.model" not in source


def test_multi_source_corpus_merge_is_provenance_gated_and_disk_deduplicated():
    source = _notebook_source("Davinci_Corpus_Merge_Builder.ipynb")
    assert "validate_dataset_manifest(manifest,intended_purpose='pretraining')" in source
    assert "verify_dataset_artifact(data_path,manifest,hash_field='combined_jsonl_sha256')" in source
    assert "sqlite3.connect(SEEN_DB)" in source
    assert "CREATE TABLE seen (digest TEXT PRIMARY KEY)" in source
    assert "license_approved_for_use':all(" in source
    assert "combined_jsonl_sha256':sha256_file(OUTPUT)" in source


def test_corpus_merge_exports_near_duplicate_review_report_without_auto_filtering():
    source = _notebook_source("Davinci_Corpus_Merge_Builder.ipynb")
    assert "find_near_duplicate_pairs" in source
    assert "near_duplicate_candidates.json" in source
    assert "near_duplicate_audit" in source



def test_sft_and_dpo_notebooks_use_canonical_resumable_checkpoints():
    sft = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    dpo = _notebook_source("Davinci_DPO_Preference_Optimization.ipynb")
    for source in (sft, dpo):
        assert "save_training_checkpoint" in source
        assert "load_training_checkpoint" in source
        assert "expected_metadata=" in source
        assert "latest_" in source
        assert "torch.Generator().manual_seed(SEED+epoch)" in source
    assert "SFT_CHECKPOINT_METADATA" in sft
    assert "DPO_CHECKPOINT_METADATA" in dpo
    assert "epoch_loss_sum" in sft
    assert "'batch_index':batch_index" in dpo
    assert "os.replace(latest_tmp,OUTPUT_DIR/'latest_dpo_checkpoint.pt')" in dpo



def test_single_source_default_matches_multi_source_merge_inputs():
    source = _notebook_source("Davinci_Persian_Dataset_Pipeline.ipynb")
    merger = _notebook_source("Davinci_Corpus_Merge_Builder.ipynb")
    assert 'selected_key = "naab"' in source
    assert "'key':'naab'" in merger
    assert "'key':'persian_blog'" in merger
    assert 'jsonl_path = PROC / f"{selected_key}.jsonl"' in source



def test_merged_corpus_manifest_reports_script_profile_without_language_claims():
    source = _notebook_source("Davinci_Corpus_Merge_Builder.ipynb")
    assert "text_statistics(normalized)" in source
    assert "'script_profile_not_language_id'" in source
    assert "'arabic_script_alpha_ratio'" in source
    assert "not necessarily Persian" in source



def test_pretraining_run_manifest_records_throughput_and_peak_memory():
    source = _notebook_source("Davinci_Pretraining_DataLoader_Training_Engine.ipynb")
    assert "torch.cuda.reset_peak_memory_stats()" in source
    assert "tokens_per_second=seen_tokens/elapsed_seconds" in source
    assert '"peak_cuda_allocated_bytes":peak_cuda_allocated_bytes' in source
    assert '"peak_cuda_reserved_bytes":peak_cuda_reserved_bytes' in source
    assert '"session_optimizer_steps":global_step-session_start_step' in source
    assert "progress.close()\nelapsed_seconds=max(time.time()-start,1e-9)" in source
    assert '"total_tokens_seen":total_tokens_seen+seen_tokens' in source
    assert 'int(state["data_state"].get("total_tokens_seen",0))' in source


def test_sft_notebook_preserves_structured_tool_events_until_canonical_encoder():
    source = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    assert '"type":"tool_call"' in source
    assert '"arguments":call.get("arguments")' in source
    assert '"name":item.get("name")' in source
    assert '"trusted":item["trusted"]' in source
    assert 'tool_result_content(item["name"],item.get("content"))' not in source


def test_sft_notebook_applies_prompt_grouped_split_before_training():
    source = _notebook_source("Davinci_SFT_Training_Chat_Finetuning.ipynb")
    assert "train_examples,val_examples=split_examples(examples,validation_fraction=0.05,seed=SEED)" in source
    assert 'item["split"]=="train"' not in source
    assert 'item["split"]=="validation"' not in source
