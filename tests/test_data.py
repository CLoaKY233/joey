import numpy as np
from joey.data import pack_tokens, PackedShardDataset


def test_pack_exact_blocks():
    tokens = list(range(10, 10 + 25))  # 25 tokens
    blocks = pack_tokens(tokens, block_len=10)
    assert blocks.shape == (2, 10)      # 25 -> 2 full blocks, remainder dropped
    assert blocks.dtype == np.int32


def test_pack_ids_preserved():
    tokens = list(range(100, 120))
    blocks = pack_tokens(tokens, block_len=5)
    assert blocks.flatten().tolist() == list(range(100, 120))


def test_shard_dataset_roundtrip(tmp_path):
    blocks = np.arange(60, dtype=np.int32).reshape(6, 10)
    shard = tmp_path / "shard0.npy"
    np.save(shard, blocks)
    ds = PackedShardDataset([str(shard)], block_len=10)
    assert len(ds) == 6
    item = ds[3]
    assert item.shape == (10,)
    assert item.tolist() == list(range(30, 40))
