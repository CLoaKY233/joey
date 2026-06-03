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
                              tokens_per_shard=50_000_000, batch_docs=1000,
                              on_commit=None):
    """Stream FineWeb-Edu, tokenize (batched), write packed .npy shards.
    on_commit(shard_path) is called after each shard so callers can persist a
    Modal volume mid-build."""
    import os
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    buf, written, shard_i = [], 0, 0
    texts = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    def flush_texts():
        nonlocal buf
        for ids in tokenizer.encode_batch(texts):
            buf.extend(ids)
            buf.append(tokenizer.eos_id)
        texts.clear()

    for ex in ds:
        texts.append(ex["text"])
        if len(texts) >= batch_docs:
            flush_texts()
        if len(buf) >= tokens_per_shard:
            path = os.path.join(out_dir, f"shard{shard_i}.npy")
            blocks = pack_tokens(buf, block_len)
            np.save(path, blocks)
            written += blocks.size
            shard_i += 1
            buf = []
            print(f"shard {shard_i} written, ~{written/1e6:.0f}M tokens", flush=True)
            if on_commit:
                on_commit(path)
            if written >= target_tokens:
                return
    flush_texts()
    if buf:
        path = os.path.join(out_dir, f"shard{shard_i}.npy")
        np.save(path, pack_tokens(buf, block_len))
        if on_commit:
            on_commit(path)
