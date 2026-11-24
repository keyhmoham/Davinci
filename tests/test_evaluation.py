import pytest
import torch

from src.davinci.evaluation import evaluate_language_model, exact_match_metrics
from src.davinci.model import DavinciConfig, DavinciForCausalLM


def tiny_model():
    config = DavinciConfig(vocab_size=31, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, local_window=4, global_every_n_layers=2)
    return DavinciForCausalLM(config)


def test_language_model_evaluation_is_token_weighted_and_restores_mode():
    torch.manual_seed(7)
    model = tiny_model().train()
    first = torch.randint(0, 31, (1, 4))
    second = torch.randint(0, 31, (2, 5))
    labels_first = first.clone()
    labels_second = second.clone()
    labels_second[0, 3:] = -100
    batches = [(first, labels_first), (second, labels_second)]
    with torch.no_grad():
        losses=[]; tokens=0
        for ids, labels in batches:
            count=int(labels[:,1:].ne(-100).sum())
            losses.append(float(model(ids,labels=labels)["loss"])*count)
            tokens+=count
    expected=sum(losses)/tokens
    metrics=evaluate_language_model(model,batches)
    assert metrics["loss"] == pytest.approx(expected,rel=1e-6)
    assert metrics["tokens"] == tokens
    assert metrics["batches"] == 2
    assert metrics["perplexity"] == pytest.approx(torch.exp(torch.tensor(expected)).item(),rel=1e-6)
    assert model.training is True


def test_language_model_evaluation_restores_eval_mode_and_skips_empty_target_batch():
    model=tiny_model().eval()
    ids=torch.randint(0,31,(1,4))
    labels=torch.full_like(ids,-100)
    valid=ids.clone()
    metrics=evaluate_language_model(model,[(ids,labels),(ids,valid)])
    assert metrics["batches"] == 1
    assert model.training is False


def test_language_model_evaluation_rejects_empty_or_invalid_batches():
    model=tiny_model()
    with pytest.raises(ValueError,match="no supervised"):
        evaluate_language_model(model,[])
    with pytest.raises(ValueError,match="same shape"):
        evaluate_language_model(model,[(torch.ones(1,3,dtype=torch.long),torch.ones(1,2,dtype=torch.long))])
    with pytest.raises(ValueError,match="max_batches"):
        evaluate_language_model(model,[],max_batches=0)


def test_exact_match_metrics_normalizes_persian_and_checks_lengths():
    result=exact_match_metrics(["مي‌روم","سلام"],["می‌روم","خوب"] )
    assert result == {"correct":1,"total":2,"accuracy":0.5}
    with pytest.raises(ValueError,match="equal lengths"):
        exact_match_metrics(["a"],[])
    with pytest.raises(ValueError,match="at least one"):
        exact_match_metrics([],[])


def test_grouped_exact_match_reports_explicit_language_and_task_breakdowns():
    from src.davinci.evaluation import grouped_exact_match_metrics

    result = grouped_exact_match_metrics(
        ["مي‌روم", "hello", "2"],
        ["می‌روم", "bye", "2"],
        ["fa", "en", "math"],
    )
    assert result["fa"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert result["en"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert result["overall"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}


def test_grouped_exact_match_rejects_missing_or_invalid_group_labels():
    from src.davinci.evaluation import grouped_exact_match_metrics

    with pytest.raises(ValueError, match="equal non-zero"):
        grouped_exact_match_metrics(["a"], ["a"], [])
    with pytest.raises(ValueError, match="non-empty"):
        grouped_exact_match_metrics(["a"], ["a"], [" "])
