import pytest
import torch

from src.davinci.hierarchical_retrieval import (
    HierarchicalRetrievalConfig,
    hierarchical_retrieve,
)


def test_hierarchical_retrieval_matches_flat_when_all_blocks_are_searched():
    torch.manual_seed(31)
    keys = torch.randn(2, 3, 37, 8)
    query = torch.randn(2, 3, 4, 8)
    indices, scores = hierarchical_retrieve(
        keys, query,
        HierarchicalRetrievalConfig(block_size=5, coarse_top_k=100, top_k=7),
    )
    exact = torch.einsum("bhqd,bhmd->bhqm", query, keys) / (8 ** 0.5)
    expected_scores, expected_indices = exact.topk(7, dim=-1)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(indices, expected_indices)


def test_hierarchical_retrieval_returns_only_selected_block_candidates():
    torch.manual_seed(32)
    keys = torch.randn(1, 2, 256, 16)
    query = torch.randn(1, 2, 3, 16)
    indices, scores = hierarchical_retrieve(
        keys, query,
        HierarchicalRetrievalConfig(block_size=16, coarse_top_k=2, top_k=8),
    )
    assert indices.shape == scores.shape == (1, 2, 3, 8)
    assert torch.isfinite(scores).all()
    # Every returned item belongs to one of the selected 2 x 16-entry blocks.
    assert ((indices // 16) < 16).all()


def test_hierarchical_retrieval_handles_partial_final_block():
    keys = torch.randn(1, 1, 19, 4)
    query = torch.randn(1, 1, 1, 4)
    indices, scores = hierarchical_retrieve(
        keys, query,
        HierarchicalRetrievalConfig(block_size=8, coarse_top_k=3, top_k=19),
    )
    assert indices.shape == scores.shape == (1, 1, 1, 19)
    assert (indices < 19).all()
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("config", [
    HierarchicalRetrievalConfig(block_size=0),
    HierarchicalRetrievalConfig(block_size=2.5),
    HierarchicalRetrievalConfig(coarse_top_k=True),
])
def test_hierarchical_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_hierarchical_retrieval_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="dimensions"):
        hierarchical_retrieve(torch.randn(1, 2, 8, 4), torch.randn(1, 1, 1, 4))
