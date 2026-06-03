import torch
from joey.config import ModelConfig
from joey.model import JoeyModel


def _tiny_cfg():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)


def test_forward_shape():
    cfg = _tiny_cfg()
    m = JoeyModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    t = torch.rand(2)
    out = m(x, t)
    assert out.shape == (2, cfg.ctx_len, cfg.vocab_size)


def test_attention_is_bidirectional():
    # Changing the LAST token must change the prediction at the FIRST position.
    cfg = _tiny_cfg()
    m = JoeyModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.ctx_len))
    t = torch.full((1,), 0.5)
    with torch.no_grad():
        a = m(x, t)[0, 0].clone()
        x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        b = m(x2, t)[0, 0]
    assert not torch.allclose(a, b)


def test_gradients_flow():
    cfg = _tiny_cfg()
    m = JoeyModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    t = torch.rand(2)
    m(x, t).sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_param_count_reasonable():
    # The real 150M config should land near 150M (130M-190M).
    cfg = ModelConfig()
    n = sum(p.numel() for p in JoeyModel(cfg).parameters())
    assert 130_000_000 < n < 190_000_000
