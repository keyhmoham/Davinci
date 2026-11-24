import random

import numpy as np
import pytest
import torch

from src.davinci.checkpoint import load_training_checkpoint, save_training_checkpoint
from src.davinci.model import DavinciConfig, DavinciForCausalLM


def make_model():
    return DavinciForCausalLM(DavinciConfig(
        vocab_size=29, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, local_window=8, global_every_n_layers=2,
    ))


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_and_progress(tmp_path):
    torch.manual_seed(7)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    ids = torch.randint(0, 29, (2, 8))
    model(ids, labels=ids)["loss"].backward()
    optimizer.step()
    scheduler.step()
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "nested" / "state.pt"
    save_training_checkpoint(
        path, model, optimizer, scheduler=scheduler, step=4, micro_step=2,
        metadata={"corpus_sha256": "a" * 64, "tokenizer_sha256": "b" * 64},
        data_state={"epoch": 1, "record_offset": 123, "shuffle_buffer": [4, 8]},
    )
    restored = make_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-4)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    info = load_training_checkpoint(
        path, restored, restored_optimizer, scheduler=restored_scheduler,
        expected_metadata={"corpus_sha256": "a" * 64},
    )
    for key, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert info["step"] == 4 and info["micro_step"] == 2
    assert info["data_state"]["record_offset"] == 123
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_checkpoint_restores_rng_states(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "rng.pt"
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    save_training_checkpoint(path, model, optimizer, step=0)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_training_checkpoint(path, model, optimizer)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_checkpoint_preserves_partial_accumulation_gradients(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    ids = torch.randint(0, 29, (2, 8))
    model(ids, labels=ids)["loss"].backward()
    expected = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    path = tmp_path / "grad.pt"
    save_training_checkpoint(path, model, optimizer, step=3, micro_step=1)
    restored = make_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    load_training_checkpoint(path, restored, restored_optimizer)
    for name, parameter in restored.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])
        else:
            assert parameter.grad is None


def test_checkpoint_rejects_dataset_mismatch_before_mutating_model(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "mismatch.pt"
    save_training_checkpoint(path, model, optimizer, step=1, metadata={"dataset": "old"})
    target = make_model()
    before = {k: v.detach().clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_training_checkpoint(
            path, target, torch.optim.AdamW(target.parameters()),
            expected_metadata={"dataset": "new"},
        )
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_checkpoint_requires_matching_scheduler_and_scaler(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "plain.pt"
    save_training_checkpoint(path, model, optimizer, step=0)
    with pytest.raises(ValueError, match="Scheduler presence"):
        load_training_checkpoint(
            path, make_model(), torch.optim.AdamW(make_model().parameters()),
            scheduler=torch.optim.lr_scheduler.StepLR(
                torch.optim.AdamW(make_model().parameters()), step_size=1
            ),
        )
