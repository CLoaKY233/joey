import numpy as np
import torch
from torch.utils.data import Dataset


def pack_tokens(tokens, block_len: int) -> np.ndarray:
    """Drop the remainder, reshape into [n_blocks, block_len] int32."""
    n = (len(tokens) // block_len) * block_len
    arr = np.asarray(tokens[:n], dtype=np.int32)
    return arr.reshape(-1, block_len)


class PackedShardDataset(Dataset):
    def __init__(self, shard_paths, block_len: int):
        self.block_len = block_len
        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths]
        self.lengths = [s.shape[0] for s in self.shards]
        self.cum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        shard_i = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = idx - self.cum[shard_i]
        row = np.asarray(self.shards[shard_i][local], dtype=np.int64)
        return torch.from_numpy(row)


def build_shards_from_fineweb(tokenizer, out_dir, block_len, target_tokens,
                              tokens_per_shard=50_000_000):
    """Stream FineWeb-Edu, tokenize, write packed .npy shards. Used in real runs."""
    import os
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    buf, written, shard_i = [], 0, 0
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for ex in ds:
        buf.extend(tokenizer.encode(ex["text"]))
        buf.append(tokenizer.eos_id)
        if len(buf) >= tokens_per_shard:
            blocks = pack_tokens(buf, block_len)
            np.save(os.path.join(out_dir, f"shard{shard_i}.npy"), blocks)
            written += blocks.size
            shard_i += 1
            buf = []
            if written >= target_tokens:
                break
    if buf:
        np.save(os.path.join(out_dir, f"shard{shard_i}.npy"),
                pack_tokens(buf, block_len))
