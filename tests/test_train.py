import torch
from joey.config import ModelConfig, TrainConfig, MASK_ID
from joey.model import JoeyModel
from joey.train import cosine_lr, train_steps, save_ckpt, load_ckpt


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
