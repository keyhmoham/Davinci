import pytest
import torch

from src.davinci import DavinciConfig, DavinciForCausalLM
from src.davinci.generation import generate, sample_next_token


def tiny_model():
    return DavinciForCausalLM(DavinciConfig(
        vocab_size=43, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, local_window=6, global_every_n_layers=2,
        attention_chunk_size=3, global_attention_chunk_size=2,
    ))


def slow_greedy(model, input_ids, max_new_tokens):
    generated = input_ids.clone()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(generated)["logits"][:, -1, :]
                token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, token), dim=1)
        return generated
    finally:
        model.train(was_training)


def test_cached_greedy_generation_matches_full_prefix_reference():
    torch.manual_seed(53)
    model = tiny_model().eval()
    prompt = torch.randint(0, model.config.vocab_size, (2, 9))
    expected = slow_greedy(model, prompt, 5)
    actual = generate(model, prompt, max_new_tokens=5, temperature=0)
    assert torch.equal(actual, expected)


def test_generation_respects_eos_and_restores_training_mode():
    model = tiny_model().train()
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))
    # Greedy output token is used as EOS, so generation stops after one token.
    with torch.no_grad():
        eos = int(model(prompt)["logits"][:, -1, :].argmax(dim=-1)[0])
    output = generate(model, prompt, max_new_tokens=10, eos_token_id=eos, temperature=0)
    assert output.shape[1] >= prompt.shape[1] + 1
    assert model.training


def test_generation_zero_new_tokens_returns_prompt():
    model = tiny_model().eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    assert torch.equal(generate(model, prompt, max_new_tokens=0), prompt)


def test_sampling_top_k_and_top_p_produce_valid_token_ids():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    token = sample_next_token(logits, temperature=0.7, top_k=2, top_p=0.9)
    assert token.shape == (1, 1)
    assert int(token.item()) in {2, 3}


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1},
    {"max_new_tokens": 1.5},
    {"top_k": 0},
    {"top_p": 0.0},
    {"eos_token_id": 999},
])
def test_generation_rejects_invalid_arguments(kwargs):
    model = tiny_model().eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        generate(model, prompt, **kwargs)
