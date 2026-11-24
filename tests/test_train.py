import torch
from src.davinci import DavinciConfig, DavinciForCausalLM
from src.davinci.train import causal_lm_loss, train_microbatch, evaluate_loss

def tiny_model():
    return DavinciForCausalLM(DavinciConfig(
        vocab_size=41, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, local_window=8, global_every_n_layers=2))

def test_causal_loss_matches_model_loss():
    model=tiny_model()
    ids=torch.randint(0,41,(2,9))
    logits=model(ids)["logits"]
    torch.testing.assert_close(causal_lm_loss(logits,ids),model(ids,labels=ids)["loss"])

def test_accumulated_real_model_training_updates_weights():
    torch.manual_seed(13)
    model=tiny_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    ids=torch.randint(0,41,(2,9))
    before=model.embed_tokens.weight.detach().clone()
    first=train_microbatch(model,ids,optimizer,gradient_accumulation_steps=2,microbatch_index=0)
    assert not first["optimizer_step"]
    assert torch.equal(before,model.embed_tokens.weight)
    second=train_microbatch(model,ids,optimizer,gradient_accumulation_steps=2,microbatch_index=1)
    assert second["optimizer_step"]
    assert not torch.equal(before,model.embed_tokens.weight)

def test_evaluation_restores_training_mode():
    model=tiny_model().train()
    ids=torch.randint(0,41,(2,9))
    value=evaluate_loss(model,[ids,ids],max_batches=2)
    assert value is not None and value>0
    assert model.training
