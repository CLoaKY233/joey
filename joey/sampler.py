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
    cum = sorted_logits.softmax(-1).cumsum(-1)
    remove = cum - sorted_logits.softmax(-1) > top_p   # keep tokens up to the crossing
    remove_scattered = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_scattered, float("-inf"))


@torch.no_grad()
def generate(model, length, steps, mask_id, vocab_size,
             prompt_ids=None, greedy=False, device="cpu",
             rep_penalty=1.0, top_p=1.0):
    """Iterative unmasking. Start all-[MASK]; each step predict every masked
    position, then commit the most-confident ones so the number still masked
    decreases to zero by the final step. prompt_ids (if given) is a clean prefix
    that stays fixed. rep_penalty/top_p curb the repetition loops common to
    diffusion sampling (defaults are no-ops)."""
    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(1, length, dtype=torch.bool, device=device)
    if prompt_ids is not None:
        p = prompt_ids.to(device)
        n = min(p.shape[1], length)
        x[0, :n] = p[0, :n]
        fixed[0, :n] = True

    n_unfixed = int((~fixed).sum())
    for step in range(steps):
        masked = (x == mask_id) & ~fixed
        if masked.sum() == 0:
            break
        t_val = 1.0 - step / steps
        t = torch.full((1,), max(t_val, 1e-4), device=device)
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

        # How many should remain masked after this step (linear schedule to 0).
        keep_masked = int(round(n_unfixed * (1.0 - (step + 1) / steps)))
        n_commit = max(1, int(masked.sum()) - keep_masked)
        scores = conf.masked_fill(~masked, -1.0)
        order = scores[0].argsort(descending=True)[:n_commit]
        x[0, order] = pred[0, order]

    # Fill any stragglers greedily at the lowest noise level.
    still = (x == mask_id)
    if still.any():
        logits = model(x, torch.full((1,), 1e-4, device=device))
        logits[..., mask_id] = float("-inf")
        x[still] = logits.argmax(-1)[still]
    return x
