"""Reusable instruction/conversation SFT pieces (response-only masking),
shared by scripts/sft.py and the Modal preview function."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from joey.config import MASK_ID, BOS_ID, EOS_ID, PAD_ID
from joey.diffusion import sft_diffusion_loss
from joey.train import cosine_lr


def _add_pair(X, R, tok, ctx, prompt_text, resp_text):
    prompt = tok.encode(prompt_text.strip())
    resp = tok.encode(" " + resp_text.strip())
    ids = [BOS_ID] + prompt + [EOS_ID] + resp + [EOS_ID]
    rmask = [False] * (len(prompt) + 2) + [True] * (len(resp) + 1)
    ids = ids[:ctx] + [PAD_ID] * max(0, ctx - len(ids))
    rmask = rmask[:ctx] + [False] * max(0, ctx - len(rmask))
    X.append(ids)
    R.append(rmask)


def build_dailydialog(tok, ctx, max_pairs=None):
    """Each consecutive utterance pair (u_i -> u_{i+1}) is a prompt->response."""
    from datasets import load_dataset
    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
    X, R = [], []
    for ex in ds:
        turns = [t for t in ex["dialog"] if t.strip()]
        for i in range(len(turns) - 1):
            _add_pair(X, R, tok, ctx, turns[i], turns[i + 1])
            if max_pairs and len(X) >= max_pairs:
                return TensorDataset(torch.tensor(X), torch.tensor(R, dtype=torch.bool))
    return TensorDataset(torch.tensor(X), torch.tensor(R, dtype=torch.bool))


def run_sft(model, dataset, device, steps=3000, lr=1e-4, batch_size=32,
            warmup=100, log_every=200):
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    step = 0
    while step < steps:
        for x, r in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, lr, warmup, steps)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = sft_diffusion_loss(model, x.to(device), r.to(device), MASK_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"sft step {step}/{steps} loss {loss.item():.3f}", flush=True)
            if step >= steps:
                break
    return model
