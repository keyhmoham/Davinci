import pytest
import torch

from src.davinci.sft import (
    SPECIAL_TOKENS, build_masked_example, causal_sft_loss, collate_masked,
    serialize_messages, split_examples, tool_call_content, tool_result_content,
)


class CharacterTokenizer:
    def __init__(self):
        self.table = {}
    def encode(self, text, out_type=int):
        ids = []
        for char in text:
            if char not in self.table:
                self.table[char] = len(self.table) + 1
            ids.append(self.table[char])
        return ids


def test_serialization_uses_only_registered_token_spellings():
    messages = [
        {"role": "user", "content": "سلام"},
        {"role": "assistant", "content": "درود"},
    ]
    serialized = serialize_messages(messages)
    assert serialized.endswith(SPECIAL_TOKENS["end"])
    assert "<|end|>" not in serialized
    assert "<|turn_end|>" not in serialized


def test_tool_call_and_result_payloads_are_strict_json():
    call = tool_call_content("add", {"left": 2, "right": 3})
    result = tool_result_content("add", {"sum": 5})
    assert call.startswith(SPECIAL_TOKENS["tool_call"])
    assert result.startswith(SPECIAL_TOKENS["tool_result"])
    with pytest.raises(ValueError):
        tool_call_content("add", {"value": float("nan")})


def test_assistant_only_masking_supervises_tool_calls_not_observations():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "user", "content": "compute"},
        {"role": "assistant", "content": tool_call_content("add", {"left": 1, "right": 2})},
        {"role": "tool", "content": tool_result_content("add", {"sum": 3})},
        {"role": "assistant", "content": "3"},
    ]
    example = build_masked_example(messages, tokenizer)
    assert any(value != -100 for value in example["labels"])
    assert example["labels"] != example["input_ids"]
    assert len(example["input_ids"]) == len(example["labels"])


def test_recorded_trajectory_tool_events_are_canonicalized_and_masked_correctly():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "user", "content": "compute"},
        {"role": "assistant", "type": "tool_call", "name": "add", "arguments": {"left": 1, "right": 2}},
        {"role": "tool", "name": "add", "content": '{"sum":3}', "trusted": False},
        {"role": "assistant", "content": "3"},
    ]
    serialized = serialize_messages(messages)
    assert tool_call_content("add", {"left": 1, "right": 2}) in serialized
    assert tool_result_content("add", {"sum": 3}) in serialized
    example = build_masked_example(messages, tokenizer)
    call_payload = tool_call_content("add", {"left": 1, "right": 2})
    result_payload = tool_result_content("add", {"sum": 3})
    call_start = serialized.index(call_payload)
    result_start = serialized.index(result_payload)
    assert all(value != -100 for value in example["labels"][call_start:call_start + len(call_payload)])
    assert all(value == -100 for value in example["labels"][result_start:result_start + len(result_payload)])

def test_group_split_keeps_same_document_and_duplicate_content_together():
    examples = [
        {"id": "a", "messages": [{"role": "user", "content": "same"}]},
        {"id": "b", "messages": [{"role": "user", "content": "same"}]},
        {"id": "e", "source_document_id": "doc-2", "messages": [{"role": "user", "content": "same"}]},
        {"id": "f", "messages": [{"role": "user", "content": "other"}]},
        {"id": "c", "source_document_id": "doc-1", "messages": [{"role": "user", "content": "x"}]},
        {"id": "d", "source_document_id": "doc-1", "messages": [{"role": "user", "content": "y"}]},
    ]
    train, validation = split_examples(examples, validation_fraction=0.5, seed=42)
    train_ids = {item["id"] for item in train}
    val_ids = {item["id"] for item in validation}
    assert ({"c", "d"} <= train_ids) or ({"c", "d"} <= val_ids)
    assert ({"a", "b", "e"} <= train_ids) or ({"a", "b", "e"} <= val_ids)
    assert "f" in val_ids
    assert not (train_ids & val_ids)


def test_split_keeps_identical_user_prompts_together_despite_different_answers():
    examples = [
        {"id": "a", "messages": [{"role": "user", "content": "سلام؟"}, {"role": "assistant", "content": "پاسخ اول"}]},
        {"id": "b", "messages": [{"role": "user", "content": "سلام؟"}, {"role": "assistant", "content": "پاسخ دوم"}]},
        {"id": "c", "messages": [{"role": "user", "content": "یک سؤال دیگر"}, {"role": "assistant", "content": "پاسخ سوم"}]},
        {"id": "d", "messages": [{"role": "user", "content": "سؤال چهارم"}, {"role": "assistant", "content": "پاسخ چهارم"}]},
    ]
    train, validation = split_examples(examples, validation_fraction=0.5, seed=42)
    train_ids = {item["id"] for item in train}
    validation_ids = {item["id"] for item in validation}
    assert ({"a", "b"} <= train_ids) or ({"a", "b"} <= validation_ids)
    assert train_ids.isdisjoint(validation_ids)

def test_collator_masks_padding_and_sft_loss_is_finite():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    ids, labels = collate_masked(batch, pad_id=0)
    assert ids.shape == labels.shape == (2, 3)
    assert labels[1, 2].item() == -100
    logits = torch.randn(2, 3, 8, requires_grad=True)
    loss = causal_sft_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()


def test_empty_or_invalid_sft_targets_are_rejected():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="assistant target"):
        build_masked_example([{"role": "user", "content": "hi"}], tokenizer)
    with pytest.raises(ValueError, match="empty"):
        split_examples([], validation_fraction=0.1)
