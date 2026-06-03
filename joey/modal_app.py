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
               vocab_size: int = 16384, ctx_len: int = 256,
               then_train: bool = True, max_hours: float = 9.0):
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

    # Chain training server-side so it survives the local caller disconnecting.
    if then_train:
        print("spawning run_training...", flush=True)
        run_training.spawn(max_hours=max_hours)


@app.function(image=image, gpu="A100", volumes={"/vol": vol},
              timeout=60 * 60 * 11)
def run_training(max_hours: float = 9.0, max_steps: int = 200_000,
                 batch_size: int = 128, grad_accum: int = 4, lr: float = 4e-4):
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
                       max_steps=max_steps, warmup_steps=2000, lr=lr,
                       ckpt_every=2000, sample_every=500)
    ds = PackedShardDataset(sorted(glob.glob(f"{SHARD_DIR}/*.npy")), cfg.ctx_len)
    print(f"dataset blocks: {len(ds)} | params: "
          f"{sum(p.numel() for p in JoeyModel(cfg).parameters())/1e6:.0f}M", flush=True)

    def sampler_cb(model, step):
        out = generate(model, length=cfg.ctx_len, steps=64, mask_id=MASK_ID,
                       vocab_size=cfg.vocab_size, device="cuda",
                       rep_penalty=1.3, top_p=0.9)
        print(f"[{step}] {tok.decode(out[0].tolist())[:240]}", flush=True)
        vol.commit()

    train(JoeyModel(cfg), ds, cfg, tcfg, MASK_ID, "cuda", CKPT_PATH,
          sampler_cb=sampler_cb, grad_accum=grad_accum)
    vol.commit()
    print("training complete", flush=True)


@app.function(image=image, gpu="A100", volumes={"/vol": vol},
              timeout=60 * 60 * 2)
def sft_and_eval(sft_steps: int = 3000):
    """Finetune the current base checkpoint on DailyDialog and print sample
    conversations — a preview of conversational behavior + loop-breaking."""
    import torch
    from joey.config import MASK_ID, BOS_ID, EOS_ID
    from joey.tokenizer import JoeyTokenizer
    from joey.train import load_ckpt, save_ckpt
    from joey.sft import build_dailydialog, run_sft
    from joey.sampler import generate

    tok = JoeyTokenizer.load(TOK_PATH)
    model, cfg, base_step = load_ckpt(CKPT_PATH, use_ema=True)
    print(f"base checkpoint step {base_step}", flush=True)
    data = build_dailydialog(tok, cfg.ctx_len)
    print(f"dailydialog pairs: {len(data)}", flush=True)
    run_sft(model, data, "cuda", steps=sft_steps, lr=1e-4, batch_size=32)
    save_ckpt(model, cfg, "/vol/joey_chat.pt", sft_steps)
    vol.commit()

    model.eval()
    prompts = ["Hi!", "How are you?", "What is your name?",
               "Do you like music?", "What did you do today?", "Goodbye!"]
    print("\n===== SAMPLE CONVERSATIONS =====", flush=True)
    for msg in prompts:
        ids = torch.tensor([[BOS_ID] + tok.encode(msg) + [EOS_ID]])
        total = min(cfg.ctx_len, ids.shape[1] + 40)
        out = generate(model, length=total, steps=96, mask_id=MASK_ID,
                       vocab_size=cfg.vocab_size, prompt_ids=ids, device="cuda",
                       rep_penalty=1.3, top_p=0.9)
        resp = out[0, ids.shape[1]:].tolist()
        if EOS_ID in resp:
            resp = resp[:resp.index(EOS_ID)]
        reply = tok.decode(resp)
        print(f"you> {msg}\njoey> {reply}\n", flush=True)


@app.local_entrypoint()
def main(max_hours: float = 9.0, target_tokens: int = 2_000_000_000):
    # spawn() runs server-side and returns immediately, so the whole pipeline
    # (build_data -> run_training) survives the local process disconnecting.
    call = build_data.spawn(target_tokens=target_tokens, then_train=True,
                            max_hours=max_hours)
    print(f"spawned build_data: {call.object_id}")
    print("pipeline running server-side; safe to disconnect. Watch with:")
    print("  modal app logs joey-diffusion")
