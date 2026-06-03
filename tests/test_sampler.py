import torch
from joey.config import ModelConfig, MASK_ID
from joey.model import JoeyModel
from joey.sampler import generate
from joey.diffusion import diffusion_loss


def _tiny():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)


def test_output_shape_and_no_masks():
    m = JoeyModel(_tiny()).eval()
    out = generate(m, length=16, steps=8, mask_id=MASK_ID, vocab_size=64)
    assert out.shape == (1, 16)
    assert (out != MASK_ID).all()


def test_ids_in_range():
    m = JoeyModel(_tiny()).eval()
    out = generate(m, length=16, steps=4, mask_id=MASK_ID, vocab_size=64)
    assert (out >= 0).all() and (out < 64).all()


def test_varying_steps_no_crash():
    m = JoeyModel(_tiny()).eval()
    for s in (1, 2, 8, 32):
        out = generate(m, length=16, steps=s, mask_id=MASK_ID, vocab_size=64)
        assert out.shape == (1, 16)


def test_prompt_prefix_preserved():
    m = JoeyModel(_tiny()).eval()
    prompt = torch.tensor([[5, 6, 7]])
    out = generate(m, length=16, steps=8, mask_id=MASK_ID, vocab_size=64,
                   prompt_ids=prompt)
    assert out[0, :3].tolist() == [5, 6, 7]


def test_recovers_overfit_sequence():
    # Train to memorize one sequence, then sampling should reproduce it.
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    target = torch.randint(4, 64, (1, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(400):
        opt.zero_grad()
        diffusion_loss(m, target.expand(8, 16), MASK_ID).backward()
        opt.step()
    m.eval()
    out = generate(m, length=16, steps=16, mask_id=MASK_ID, vocab_size=64,
                   greedy=True)
    # Most positions should match the memorized sequence.
    assert (out == target).float().mean() > 0.7
