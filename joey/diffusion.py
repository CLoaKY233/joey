import torch
import torch.nn.functional as F

EPS = 1e-4


def mask_tokens(x, t, mask_id):
    """Mask each token independently with probability t[b]. Returns (x_t, mask)."""
    probs = t[:, None].expand_as(x)
    mask = torch.rand_like(x, dtype=torch.float) < probs
    x_t = torch.where(mask, torch.full_like(x, mask_id), x)
    return x_t, mask


def diffusion_loss(model, x, mask_id, force_t=None):
    """1/t-weighted cross-entropy on masked positions only (MDLM weighting)."""
    B = x.shape[0]
    t = force_t if force_t is not None else torch.rand(B, device=x.device)
    t = t.clamp(min=EPS, max=1.0)
    x_t, mask = mask_tokens(x, t, mask_id)
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    logits = model(x_t, t)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction="none"
    ).view(B, -1)
    weight = (1.0 / t)[:, None].expand_as(ce)
    masked_loss = (ce * mask.float() * weight).sum() / mask.float().sum()
    return masked_loss
