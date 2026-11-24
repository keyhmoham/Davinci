import tempfile
import torch
from src.davinci.model import DavinciConfig, DavinciForCausalLM

def tiny():
    return DavinciConfig(vocab_size=97, hidden_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, local_window=5, global_every_n_layers=2)

def test_forward_loss_and_gradients():
    c=tiny(); m=DavinciForCausalLM(c)
    ids=torch.randint(0,c.vocab_size,(2,12))
    out=m(ids,labels=ids)
    assert out["logits"].shape==(2,12,c.vocab_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

def test_prefill_decode_parity_with_local_and_global_layers():
    torch.manual_seed(7); m=DavinciForCausalLM(tiny()).eval()
    ids=torch.randint(0,m.config.vocab_size,(2,17))
    with torch.no_grad():
        expected=m(ids)["logits"][:,-1:,:]
        _,cache=m.prefill(ids[:,:-1])
        actual=m.decode(ids[:,-1:],16,cache)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)

def test_tied_embeddings_and_checkpoint_roundtrip():
    m=DavinciForCausalLM(tiny()).eval()
    assert m.lm_head.weight.data_ptr()==m.embed_tokens.weight.data_ptr()
    ids=torch.randint(0,m.config.vocab_size,(1,8))
    with torch.no_grad(): expected=m(ids)["logits"]
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d); restored=DavinciForCausalLM.from_pretrained(d).eval()
        with torch.no_grad(): actual=restored(ids)["logits"]
    torch.testing.assert_close(actual,expected)


def test_gradient_checkpointing_backward():
    c=tiny(); c.gradient_checkpointing=True
    m=DavinciForCausalLM(c).train()
    ids=torch.randint(0,c.vocab_size,(2,9))
    loss=m(ids,labels=ids)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_invalid_loss_inputs_raise_clear_errors():
    m=DavinciForCausalLM(tiny())
    one=torch.randint(0,m.config.vocab_size,(1,1))
    try:
        m(one,labels=one)
        assert False, "expected short-sequence error"
    except ValueError as exc:
        assert "two tokens" in str(exc)
    ids=torch.randint(0,m.config.vocab_size,(1,4))
    labels=torch.full_like(ids,-100)
    try:
        m(ids,labels=labels)
        assert False, "expected empty-target error"
    except ValueError as exc:
        assert "no next-token targets" in str(exc)


def test_chunked_local_attention_matches_single_chunk():
    torch.manual_seed(11)
    c1=tiny(); c1.attention_chunk_size=3
    c2=tiny(); c2.attention_chunk_size=64
    m1=DavinciForCausalLM(c1).eval()
    m2=DavinciForCausalLM(c2).eval()
    m2.load_state_dict(m1.state_dict())
    ids=torch.randint(0,c1.vocab_size,(2,23))
    with torch.no_grad():
        a=m1(ids)["logits"]
        b=m2(ids)["logits"]
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)


def test_chunked_local_attention_cache_parity():
    torch.manual_seed(13)
    c=tiny(); c.attention_chunk_size=4
    m=DavinciForCausalLM(c).eval()
    ids=torch.randint(0,c.vocab_size,(1,21))
    with torch.no_grad():
        expected=m(ids)["logits"][:,-1:,:]
        _,cache=m.prefill(ids[:,:-1])
        actual=m.decode(ids[:,-1:],20,cache)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)


def test_chunked_global_attention_matches_single_chunk():
    torch.manual_seed(17)
    c1=tiny(); c1.global_attention_chunk_size=3
    c2=tiny(); c2.global_attention_chunk_size=64
    m1=DavinciForCausalLM(c1).eval()
    m2=DavinciForCausalLM(c2).eval()
    m2.load_state_dict(m1.state_dict())
    ids=torch.randint(0,c1.vocab_size,(2,19))
    with torch.no_grad():
        a=m1(ids)["logits"]
        b=m2(ids)["logits"]
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)


def _dense_attention_reference(attn, x, positions):
    """Independent short-sequence reference for exact causal attention."""
    import torch.nn.functional as F
    b,t,_=x.shape
    q=attn.q_proj(x).view(b,t,attn.heads,attn.head_dim).transpose(1,2)
    k=attn.k_proj(x).view(b,t,attn.kv_heads,attn.head_dim).transpose(1,2)
    v=attn.v_proj(x).view(b,t,attn.kv_heads,attn.head_dim).transpose(1,2)
    from src.davinci.model import apply_rope
    q=apply_rope(q,positions,attn.c.rope_theta,attn.c.rope_scaling_factor)
    k=apply_rope(k,positions,attn.c.rope_theta,attn.c.rope_scaling_factor)
    if attn.groups != 1:
        k=k.repeat_interleave(attn.groups,1)
        v=v.repeat_interleave(attn.groups,1)
    delta=positions[:,None]-positions[None,:]
    allowed=delta>=0
    if not attn.is_global:
        allowed=allowed & (delta<attn.window)
    y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,
        dropout_p=0.0,is_causal=False)
    return attn.o_proj(y.transpose(1,2).contiguous().view(b,t,-1))


def test_chunked_attention_matches_independent_dense_reference():
    from src.davinci.model import GQAAttention
    torch.manual_seed(23)
    x=torch.randn(2,17,64)
    positions=torch.arange(x.shape[1])
    for layer_idx in (0,1):  # local layer and global layer
        c=tiny()
        c.attention_chunk_size=3
        c.global_attention_chunk_size=4
        attn=GQAAttention(c,layer_idx).eval()
        with torch.no_grad():
            actual=attn(x,positions)
            expected=_dense_attention_reference(attn,x,positions)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)


def test_config_rejects_non_integer_and_non_finite_values():
    import pytest
    with pytest.raises(ValueError, match="positive integers"):
        DavinciConfig(num_hidden_layers=2.5).validate()
    with pytest.raises(ValueError, match="positive integers"):
        DavinciConfig(attention_chunk_size=True).validate()
    with pytest.raises(ValueError, match="norm_eps"):
        DavinciConfig(norm_eps=0.0).validate()
    with pytest.raises(ValueError, match="rope_theta"):
        DavinciConfig(rope_theta=float("nan")).validate()
    with pytest.raises(ValueError, match="dropout"):
        DavinciConfig(dropout=float("inf")).validate()

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        DavinciConfig(gradient_checkpointing=1).validate()
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        DavinciConfig(tie_word_embeddings="yes").validate()


def test_parameter_count_matches_instantiated_tiny_model():
    from src.davinci.model import count_parameters
    model = DavinciForCausalLM(tiny())
    assert count_parameters(model.config) == sum(p.numel() for p in model.parameters())


def test_canonical_500m_config_has_expected_parameter_count():
    import json
    from pathlib import Path
    from src.davinci.model import count_parameters
    path = Path(__file__).resolve().parents[1] / "configs" / "davinci_500m.json"
    config = DavinciConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert count_parameters(config) == 508_960_000



def test_linear_rope_interpolation_matches_scaled_positions():
    from src.davinci.model import apply_rope
    torch.manual_seed(29)
    x = torch.randn(2, 4, 9, 16)
    positions = torch.arange(9)
    interpolated = apply_rope(x, positions, 10000.0, scaling_factor=4.0)
    reference = apply_rope(x, positions.float() / 4.0, 10000.0)
    torch.testing.assert_close(interpolated, reference, rtol=1e-6, atol=1e-6)


def test_rope_scaling_factor_is_validated_and_used_by_attention():
    import pytest
    with pytest.raises(ValueError, match="rope_scaling_factor"):
        DavinciConfig(rope_scaling_factor=0.0).validate()
    c = tiny()
    c.rope_scaling_factor = 2.0
    model = DavinciForCausalLM(c).eval()
    ids = torch.randint(0, c.vocab_size, (1, 11))
    with torch.no_grad():
        expected = model(ids)["logits"][:, -1:, :]
        _, caches = model.prefill(ids[:, :-1])
        actual = model.decode(ids[:, -1:], 10, caches)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
