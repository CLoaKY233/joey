import math
import time
import torch
from joey.model import JoeyModel
from joey.diffusion import diffusion_loss
from joey.config import ModelConfig


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def save_ckpt(model, cfg, path, step):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, path)


def load_ckpt(path, map_location="cpu"):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**blob["cfg"])
    model = JoeyModel(cfg)
    model.load_state_dict(blob["model"])
    return model, cfg, blob["step"]


def train_steps(model, batch_iter, mask_id, n_steps, lr=3e-4, warmup=10,
                grad_clip=1.0, device="cpu"):
    """Run n_steps; return list of losses."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    losses = []
    for step in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, lr, warmup, n_steps)
        x = next(batch_iter).to(device)
        opt.zero_grad()
        loss = diffusion_loss(model, x, mask_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(loss.item())
    return losses


def train(model, dataset, cfg, train_cfg, mask_id, device, ckpt_path,
          sampler_cb=None):
    """Full run with checkpointing + hours kill-switch. Used on T4/A100."""
    from torch.utils.data import DataLoader
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=2)
    start, step = time.time(), 0
    while step < train_cfg.max_steps:
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, train_cfg.lr, train_cfg.warmup_steps,
                                    train_cfg.max_steps)
            opt.zero_grad()
            loss = diffusion_loss(model, batch.to(device), mask_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            step += 1
            if step % train_cfg.sample_every == 0:
                print(f"step {step} loss {loss.item():.3f}")
                if sampler_cb:
                    sampler_cb(model, step)
            if step % train_cfg.ckpt_every == 0:
                save_ckpt(model, cfg, ckpt_path, step)
            if (time.time() - start) / 3600 > train_cfg.max_hours:
                print("HOURS KILL-SWITCH hit; checkpointing and stopping.")
                save_ckpt(model, cfg, ckpt_path, step)
                return
            if step >= train_cfg.max_steps:
                break
    save_ckpt(model, cfg, ckpt_path, step)
