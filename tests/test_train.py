import torch
from joey.config import ModelConfig, TrainConfig, MASK_ID
from joey.model import JoeyModel
from joey.train import (cosine_lr, train_steps, save_ckpt, load_ckpt,
                        EMA, param_groups, save_ckpt_ema)


def _tiny():
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)


def test_cosine_lr_warmup_then_decay():
    base = 3e-4
    assert cosine_lr(0, base, warmup=10, total=100) < base       # warming up
    assert abs(cosine_lr(10, base, warmup=10, total=100) - base) < 1e-9  # peak
    assert cosine_lr(100, base, warmup=10, total=100) < base * 0.1       # decayed


def test_train_reduces_loss():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    data = torch.randint(4, 64, (16, 16))
    def batches():
        while True:
            yield data[torch.randint(0, 16, (8,))]
    losses = train_steps(m, batches(), MASK_ID, n_steps=100, lr=1e-3)
    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    path = str(tmp_path / "ck.pt")
    save_ckpt(m, cfg, path, step=42)
    m2, cfg2, step = load_ckpt(path)
    assert step == 42 and cfg2.d_model == 32
    x = torch.randint(4, 64, (1, 16)); t = torch.rand(1)
    assert torch.allclose(m(x, t), m2(x, t))


def test_param_groups_split():
    m = JoeyModel(_tiny())
    groups = param_groups(m, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    # every trainable param lands in exactly one group
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    assert n == n_grouped


def test_ema_tracks_then_diverges():
    m = JoeyModel(_tiny())
    ema = EMA(m, decay=0.9)
    before = ema.shadow["ln_f.weight"].clone()
    with torch.no_grad():
        m.ln_f.weight.add_(1.0)   # shift weights
    ema.update(m)
    after = ema.shadow["ln_f.weight"]
    # EMA moved toward the new value but not all the way (decay < 1)
    assert not torch.allclose(after, before)
    assert not torch.allclose(after, m.ln_f.weight)


def test_load_ckpt_prefers_ema(tmp_path):
    cfg = _tiny()
    m = JoeyModel(cfg)
    ema = EMA(m)
    # make EMA weights distinct from raw
    for v in ema.shadow.values():
        if v.dtype.is_floating_point:
            v.add_(0.5)
    path = str(tmp_path / "e.pt")
    save_ckpt_ema(m, ema, cfg, path, step=7)
    m_ema, _, step = load_ckpt(path, use_ema=True)
    m_raw, _, _ = load_ckpt(path, use_ema=False)
    assert step == 7
    sd_ema, sd_raw = m_ema.state_dict(), m_raw.state_dict()
    assert not torch.allclose(sd_ema["ln_f.weight"], sd_raw["ln_f.weight"])
