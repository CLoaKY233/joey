import torch
from joey.config import ModelConfig, MASK_ID
from joey.model import JoeyModel
from joey.diffusion import mask_tokens, diffusion_loss


def test_t_zero_no_mask():
    x = torch.randint(4, 64, (8, 16))
    xt, m = mask_tokens(x, torch.zeros(8), MASK_ID)
    assert m.sum() == 0
    assert torch.equal(xt, x)


def test_t_one_all_mask():
    x = torch.randint(4, 64, (8, 16))
    xt, m = mask_tokens(x, torch.ones(8), MASK_ID)
    assert m.all()
    assert (xt == MASK_ID).all()


def test_masked_positions_become_mask_id():
    x = torch.randint(4, 64, (8, 16))
    xt, m = mask_tokens(x, torch.full((8,), 0.5), MASK_ID)
    assert (xt[m] == MASK_ID).all()
    assert torch.equal(xt[~m], x[~m])


def test_loss_ignores_unmasked():
    # With zero masked tokens, loss must be 0 (nothing to predict).
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    x = torch.randint(4, 64, (4, 16))
    loss = diffusion_loss(m, x, MASK_ID, force_t=torch.zeros(4))
    assert loss.item() == 0.0


def test_overfit_one_batch():
    # THE correctness gate: model must drive loss near zero on a single batch.
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # One sequence tiled across the batch: at high t the input is all-[MASK],
    # so distinct targets would be unlearnable. Memorizing a single sequence is
    # the correct overfit gate for a diffusion model.
    x = torch.randint(4, 64, (1, 16)).expand(8, 16).contiguous()
    last = None
    for _ in range(300):
        opt.zero_grad()
        loss = diffusion_loss(m, x, MASK_ID)
        loss.backward()
        opt.step()
        last = loss.item()
    assert last < 0.5
