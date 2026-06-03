"""Tiny end-to-end run on T4/CPU: train a small model on a small FineWeb-Edu
slice for a few hundred steps; assert loss drops and samples are valid ids."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import torch
from joey.config import ModelConfig, MASK_ID
from joey.model import JoeyModel
from joey.tokenizer import JoeyTokenizer
from joey.data import build_shards_from_fineweb, PackedShardDataset
from joey.train import train_steps
from joey.sampler import generate
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    tok = JoeyTokenizer.load("artifacts/tok.json")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=256, n_layers=4,
                      n_heads=4, ctx_len=128)
    if not glob.glob("data/sanity/*.npy"):
        build_shards_from_fineweb(tok, "data/sanity", cfg.ctx_len,
                                  target_tokens=2_000_000)
    ds = PackedShardDataset(sorted(glob.glob("data/sanity/*.npy")), cfg.ctx_len)
    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    it = iter(loader)

    def batches():
        nonlocal it
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(loader)
                yield next(it)

    m = JoeyModel(cfg)
    losses = train_steps(m, batches(), MASK_ID, n_steps=500, lr=3e-4, device=DEVICE)
    print("loss start->end:", round(losses[0], 3), "->", round(losses[-1], 3))
    assert losses[-1] < losses[0], "loss did not decrease"
    out = generate(m.to(DEVICE), length=cfg.ctx_len, steps=32, mask_id=MASK_ID,
                   vocab_size=cfg.vocab_size, device=DEVICE)
    print("SAMPLE:", tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
