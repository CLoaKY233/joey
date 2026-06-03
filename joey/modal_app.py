"""Modal harness for Joey.

Two stages, so we never pay A100 rates for CPU work:
  1. build_data  (CPU) — trains the tokenizer and builds packed shards into a
     persistent volume. Cheap.
  2. run_training (A100-40GB) — trains the 170M diffusion model from the volume,
     bf16 + EMA, with an hours kill-switch.

Run the whole pipeline:   modal run joey/modal_app.py --max-hours 9
"""
import modal

app = modal.App("joey-diffusion")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "tokenizers", "datasets", "numpy")
    .add_local_python_source("joey")
)
vol = modal.Volume.from_name("joey-data", create_if_missing=True)

TOK_PATH = "/vol/tok.json"
SHARD_DIR = "/vol/shards"
CKPT_PATH = "/vol/joey_base.pt"


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=32768,
              timeout=60 * 60 * 6)
def build_data(tokenizer_docs: int = 400_000, target_tokens: int = 2_000_000_000,
               vocab_size: int = 16384, ctx_len: int = 256):
    import os
    import glob
    from datasets import load_dataset
    from joey.tokenizer import JoeyTokenizer
    from joey.data import build_shards_from_fineweb

    # 1. tokenizer (trained in-cloud on a FineWeb-Edu slice)
    if not os.path.exists(TOK_PATH):
        os.makedirs("/vol", exist_ok=True)
        corpus = "/vol/corpus.txt"
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        with open(corpus, "w") as f:
            for i, ex in zip(range(tokenizer_docs), ds):
                f.write(ex["text"] + "\n")
        print(f"wrote {tokenizer_docs} docs; training tokenizer...", flush=True)
        JoeyTokenizer.train([corpus], vocab_size=vocab_size).save(TOK_PATH)
        vol.commit()
        print("tokenizer saved", flush=True)

    # 2. packed shards
    existing = sorted(glob.glob(f"{SHARD_DIR}/*.npy"))
    have = sum(__import__("numpy").load(p, mmap_mode="r").size for p in existing)
    if have < target_tokens:
        tok = JoeyTokenizer.load(TOK_PATH)
        build_shards_from_fineweb(tok, SHARD_DIR, ctx_len, target_tokens,
                                  on_commit=lambda _p: vol.commit())
        vol.commit()
    print("data ready", flush=True)


@app.function(image=image, gpu="A100", volumes={"/vol": vol},
              timeout=60 * 60 * 11)
def run_training(max_hours: float = 9.0, max_steps: int = 200_000,
                 batch_size: int = 96, grad_accum: int = 2):
    import glob
    import torch
    from joey.config import ModelConfig, TrainConfig, MASK_ID
    from joey.model import JoeyModel
    from joey.tokenizer import JoeyTokenizer
    from joey.data import PackedShardDataset
    from joey.train import train
    from joey.sampler import generate

    tok = JoeyTokenizer.load(TOK_PATH)
    cfg = ModelConfig(vocab_size=tok.vocab_size)        # full 170M config
    tcfg = TrainConfig(batch_size=batch_size, max_hours=max_hours,
                       max_steps=max_steps, warmup_steps=2000,
                       ckpt_every=2000, sample_every=500)
    ds = PackedShardDataset(sorted(glob.glob(f"{SHARD_DIR}/*.npy")), cfg.ctx_len)
    print(f"dataset blocks: {len(ds)} | params: "
          f"{sum(p.numel() for p in JoeyModel(cfg).parameters())/1e6:.0f}M", flush=True)

    def sampler_cb(model, step):
        out = generate(model, length=cfg.ctx_len, steps=64, mask_id=MASK_ID,
                       vocab_size=cfg.vocab_size, device="cuda")
        print(f"[{step}] {tok.decode(out[0].tolist())[:240]}", flush=True)
        vol.commit()

    train(JoeyModel(cfg), ds, cfg, tcfg, MASK_ID, "cuda", CKPT_PATH,
          sampler_cb=sampler_cb, grad_accum=grad_accum)
    vol.commit()
    print("training complete", flush=True)


@app.local_entrypoint()
def main(max_hours: float = 9.0, target_tokens: int = 2_000_000_000):
    build_data.remote(target_tokens=target_tokens)
    run_training.remote(max_hours=max_hours)
