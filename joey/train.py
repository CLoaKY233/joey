import copy
import math
import os
import time
import torch
from joey.model import JoeyModel
from joey.diffusion import diffusion_loss
from joey.config import ModelConfig


class EMA:
    """Exponential moving average of model weights — diffusion samples are
    noticeably cleaner when drawn from EMA weights rather than the raw model."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def param_groups(model, weight_decay):
    """Decoupled weight decay: decay matmul weights, not biases/norms/embeddings."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in n or "ln" in n or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def save_ckpt(model, cfg, path, step):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, path)


def load_ckpt(path, map_location="cpu", use_ema=True):
    """Load a checkpoint. Prefers EMA weights for inference when present."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(**blob["cfg"])
    model = JoeyModel(cfg)
    weights = blob["ema"] if (use_ema and "ema" in blob) else blob["model"]
    model.load_state_dict(weights)
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


def save_ckpt_ema(model, ema, cfg, path, step):
    """Checkpoint that stores both raw and EMA weights."""
    torch.save({"model": model.state_dict(), "ema": ema.shadow,
                "cfg": cfg.__dict__, "step": step}, path)


def train(model, dataset, cfg, train_cfg, mask_id, device, ckpt_path,
          sampler_cb=None, ema_decay=0.999, grad_accum=1):
    """Full run: bf16 autocast, EMA weights, decoupled weight decay, cosine LR,
    checkpoint+resume, and an hours kill-switch. Used on the A100."""
    from torch.utils.data import DataLoader
    model.to(device).train()
    opt = torch.optim.AdamW(param_groups(model, train_cfg.weight_decay),
                            lr=train_cfg.lr, betas=(0.9, 0.95))
    ema = EMA(model, ema_decay)
    step = 0
    # Resume if a checkpoint already exists on the volume.
    if os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if "ema" in blob:
            ema.shadow = {k: v.to(device) for k, v in blob["ema"].items()}
        step = blob.get("step", 0)
        print(f"resumed from {ckpt_path} at step {step}")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True,
                        persistent_workers=True)
    start = time.time()
    opt.zero_grad()
    while step < train_cfg.max_steps:
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, train_cfg.lr, train_cfg.warmup_steps,
                                    train_cfg.max_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = diffusion_loss(model, batch.to(device), mask_id) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_cfg.grad_clip)
                opt.step()
                opt.zero_grad()
                ema.update(model)
            step += 1
            if step % train_cfg.sample_every == 0:
                print(f"step {step} loss {loss.item() * grad_accum:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"elapsed {(time.time()-start)/3600:.2f}h", flush=True)
                if sampler_cb:
                    em = JoeyModel(cfg).to(device)
                    ema.copy_to(em)
                    em.eval()
                    sampler_cb(em, step)
                    model.train()
            if step % train_cfg.ckpt_every == 0:
                save_ckpt_ema(model, ema, cfg, ckpt_path, step)
            if (time.time() - start) / 3600 > train_cfg.max_hours:
                print("HOURS KILL-SWITCH hit; checkpointing and stopping.")
                save_ckpt_ema(model, ema, cfg, ckpt_path, step)
                return
            if step >= train_cfg.max_steps:
                break
    save_ckpt_ema(model, ema, cfg, ckpt_path, step)
