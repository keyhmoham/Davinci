import torch
import pytest
from src.davinci.long_context import CompressedKVMemory, CompressedMemoryConfig

def test_compressed_memory_keeps_recent_and_compresses_old_groups():
    cache=CompressedKVMemory(CompressedMemoryConfig(recent_window=4,compression_group_size=2,retrieval_top_k=2))
    key=torch.arange(12,dtype=torch.float32).view(1,1,6,2)
    value=key+100
    cache.append(key,value,torch.arange(6))
    assert cache.recent_key.shape[2]==4
    assert cache.memory_key.shape[2]==1
    torch.testing.assert_close(cache.memory_key[0,0,0],key[0,0,:2].mean(0))
    torch.testing.assert_close(cache.memory_value[0,0,0],value[0,0,:2].mean(0))
    assert cache.end_positions.tolist()==[1]

def test_compressed_memory_attention_is_finite_and_causal():
    torch.manual_seed(5)
    cache=CompressedKVMemory(CompressedMemoryConfig(recent_window=4,compression_group_size=2,retrieval_top_k=3))
    key=torch.randn(2,2,10,8); value=torch.randn_like(key)
    cache.append(key,value,torch.arange(10))
    query=torch.randn(2,2,2,8)
    out=cache.attend(query,torch.tensor([9,10]))
    assert out.shape==query.shape
    assert torch.isfinite(out).all()

def test_compressed_memory_rejects_invalid_position_order():
    cache=CompressedKVMemory(CompressedMemoryConfig(recent_window=4,compression_group_size=2))
    x=torch.randn(1,1,2,4)
    cache.append(x,x,torch.tensor([4,5]))
    with pytest.raises(ValueError):
        cache.append(x,x,torch.tensor([5,6]))

def test_compressed_memory_config_validation():
    with pytest.raises(ValueError):
        CompressedKVMemory(CompressedMemoryConfig(recent_window=0))
