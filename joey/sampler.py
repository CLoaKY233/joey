import math
import torch


def _apply_rep_penalty(logits, present_ids, penalty):
    """Divide logits of already-present tokens (CTRL-style) to discourage loops."""
    if penalty == 1.0 or present_ids.numel() == 0:
        return logits
    sel = logits[..., present_ids]
    logits[..., present_ids] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return logits


def _top_p_filter(logits, top_p):
    """Keep the smallest set of tokens whose cumulative prob >= top_p; mask the rest."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sp = sorted_logits.softmax(-1)
    remove = sp.cumsum(-1) - sp > top_p
    remove_scattered = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_scattered, float("-inf"))


@torch.no_grad()
def generate(model, length, steps, mask_id, vocab_size,
             prompt_ids=None, greedy=False, device="cpu",
             rep_penalty=1.0, top_p=1.0):
    """Remasking (MaskGIT/LLaDA-style) iterative decoding.

    Each step: predict EVERY non-fixed position, keep only the highest-confidence
    ones (a cosine schedule grows the kept set 0 -> all over `steps`), and re-mask
    the rest so they are re-decided next step with more context. Re-deciding is
    what breaks the repetition loops that permanent-commit sampling falls into.
    prompt_ids (if given) is a clean prefix that stays fixed."""
    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(1, length, dtype=torch.bool, device=device)
    if prompt_ids is not None:
        p = prompt_ids.to(device)
        n = min(p.shape[1], length)
        x[0, :n] = p[0, :n]
        fixed[0, :n] = True

    nonfix = (~fixed)[0]
    n_unfixed = int(nonfix.sum())
    if n_unfixed == 0:
        return x

    for step in range(steps):
        t_val = max(1.0 - step / steps, 1e-4)
        t = torch.full((1,), t_val, device=device)
        logits = model(x, t)
        logits[..., mask_id] = float("-inf")              # never emit [MASK]
        present = x[(x != mask_id)].unique()
        logits = _apply_rep_penalty(logits, present, rep_penalty)
        logits = _top_p_filter(logits, top_p)
        probs = logits.softmax(-1)                        # [1, T, V]
        if greedy:
            conf, pred = probs.max(-1)                    # [1, T]
        else:
            pred = torch.multinomial(probs[0], 1).view(1, -1)
            conf = probs[0].gather(-1, pred[0, :, None]).squeeze(-1)[None]

        # How many non-fixed positions to KEEP this step (cosine schedule 0 -> all).
        keep = int(round(n_unfixed * (1.0 - math.cos(math.pi / 2 * (step + 1) / steps))))
        keep = max(1, min(keep, n_unfixed))
        # Rebuild x: fixed prefix stays; everything else masked except the top-`keep`
        # most-confident predictions.
        new_x = x.clone()
        new_x[0, nonfix] = mask_id
        scores = conf[0].masked_fill(fixed[0], -1.0)
        top = scores.argsort(descending=True)[:keep]
        new_x[0, top] = pred[0, top]
        x = new_x

    # Safety: fill any leftover masks greedily at the lowest noise level.
    still = (x == mask_id)
    if still.any():
        logits = model(x, torch.full((1,), 1e-4, device=device))
        logits[..., mask_id] = float("-inf")
        x[still] = logits.argmax(-1)[still]
    return x
