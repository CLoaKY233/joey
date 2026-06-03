"""Instruction finetune the base checkpoint with response-only masking
(LLaDA-style SFT). Each example is [BOS] prompt [EOS] response, padded to
ctx_len; only response tokens are ever masked / scored."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from joey.config import MASK_ID, BOS_ID, EOS_ID, PAD_ID
from joey.tokenizer import JoeyTokenizer
from joey.train import load_ckpt, save_ckpt, cosine_lr
from joey.diffusion import sft_diffusion_loss


def _add_pair(X, R, tok, ctx, prompt_text, resp_text):
    prompt = tok.encode(prompt_text.strip())
    resp = tok.encode(" " + resp_text.strip())
    ids = [BOS_ID] + prompt + [EOS_ID] + resp + [EOS_ID]
    rmask = [False] * (len(prompt) + 2) + [True] * (len(resp) + 1)
    ids = ids[:ctx] + [PAD_ID] * max(0, ctx - len(ids))
    rmask = rmask[:ctx] + [False] * max(0, ctx - len(rmask))
    X.append(ids)
    R.append(rmask)


def build(tok, ctx):
    """Everyday-conversation SFT from DailyDialog: each consecutive utterance
    pair (u_i -> u_{i+1}) becomes a prompt->response example."""
    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    X, R = [], []
    for ex in ds:
        turns = [t for t in ex["dialog"] if t.strip()]
        for i in range(len(turns) - 1):
            _add_pair(X, R, tok, ctx, turns[i], turns[i + 1])
    return TensorDataset(torch.tensor(X), torch.tensor(R, dtype=torch.bool))


def main():
    tok = JoeyTokenizer.load("artifacts/tok.json")
    model, cfg, _ = load_ckpt("artifacts/joey_base.pt", use_ema=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()
    data = build(tok, cfg.ctx_len)
    loader = DataLoader(data, batch_size=16, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    step, total = 0, 3000
    while step < total:
        for x, r in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, 1e-4, 100, total)
            opt.zero_grad()
            loss = sft_diffusion_loss(model, x.to(dev), r.to(dev), MASK_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"sft step {step} loss {loss.item():.3f}", flush=True)
            if step >= total:
                break
    save_ckpt(model, cfg, "artifacts/joey_chat.pt", step)
    print("saved artifacts/joey_chat.pt")


if __name__ == "__main__":
    main()
