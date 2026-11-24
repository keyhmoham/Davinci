import math

import pytest
import torch

from src.davinci.preference import (
    PreferenceDataError, dpo_loss, preference_split, sequence_log_probs,
    validate_preference_pair,
)


def test_preference_pair_validation_and_normalization():
    row = validate_preference_pair({"prompt": "  سؤال  ", "chosen": " جواب خوب ", "rejected": "جواب بد"})
    assert row == {"pair_id": "", "prompt": "سؤال", "chosen": "جواب خوب", "rejected": "جواب بد"}
    with pytest.raises(PreferenceDataError, match="must differ"):
        validate_preference_pair({"prompt": "x", "chosen": "same", "rejected": "same"})
    with pytest.raises(PreferenceDataError, match="non-empty"):
        validate_preference_pair({"prompt": "", "chosen": "a", "rejected": "b"})


def test_preference_split_is_stable_for_normalized_prompt():
    assert preference_split("مي‌روم", seed=9) == preference_split("می‌روم", seed=9)
    assert preference_split("x", validation_fraction=0.0) == "train"
    assert preference_split("x", validation_fraction=1.0) == "validation"


def test_sequence_log_probs_applies_causal_shift_and_masking():
    logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    labels = torch.tensor([[0, 0, -100]])
    logps, counts = sequence_log_probs(logits, labels)
    expected = torch.log_softmax(logits[:, :-1].float(), -1)[0, 0, 0]
    assert counts.tolist() == [1]
    assert logps.item() == pytest.approx(expected.item())
    logps.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_sequence_log_probs_respects_attention_mask_and_rejects_bad_labels():
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    _, counts = sequence_log_probs(logits, labels, attention_mask=mask)
    assert counts.tolist() == [1, 3]
    bad = labels.clone(); bad[0, 1] = 99
    with pytest.raises(ValueError, match="outside"):
        sequence_log_probs(logits, bad)


def test_dpo_loss_prefers_policy_with_chosen_response_and_has_gradients():
    chosen = torch.tensor([-1.0, -0.5], requires_grad=True)
    rejected = torch.tensor([-2.0, -1.5], requires_grad=True)
    ref_chosen = torch.tensor([-1.0, -1.0])
    ref_rejected = torch.tensor([-1.5, -1.2])
    result = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    assert result["loss"].ndim == 0
    assert 0.0 <= result["preference_accuracy"].item() <= 1.0
    result["loss"].backward()
    assert chosen.grad is not None and torch.isfinite(chosen.grad).all()


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_dpo_rejects_invalid_beta(beta):
    x = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(x, x, x, x, beta=beta)


def test_dpo_rejects_shape_empty_and_nonfinite_inputs():
    x = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        dpo_loss(x, x.repeat(2), x, x)
    empty = torch.empty(0)
    with pytest.raises(ValueError, match="one-dimensional"):
        dpo_loss(empty, empty, empty, empty)
    bad = torch.tensor([math.inf])
    with pytest.raises(ValueError, match="finite"):
        dpo_loss(bad, x, x, x)