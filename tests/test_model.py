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
