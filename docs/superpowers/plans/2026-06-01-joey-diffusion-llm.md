# Joey Diffusion LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ~150M-parameter masked/absorbing-state diffusion language model from scratch (tokenizer, model, diffusion loss, sampler, training), train it on FineWeb-Edu, and make it chattable via instruction finetuning.

**Architecture:** Forward process masks tokens with probability `t∼U(0,1)`; a bidirectional Transformer (no causal mask, timestep-conditioned) predicts the originals; loss is `1/t`-weighted cross-entropy on masked positions only. Generation starts all-`[MASK]` and iteratively unmasks the most-confident tokens. Dev/sanity on a free T4; the real run on a Modal A100 within a ~$30 budget.

**Tech Stack:** Python 3.11, PyTorch, `uv`, Modal, pytest. From scratch: BPE training, model, diffusion, sampler.

**Spec:** `docs/superpowers/specs/2026-06-01-joey-diffusion-llm-design.md`
**Learning docs:** `docs/diffusion-llm/`

---

## File Structure

```
joey/
  __init__.py
  config.py            # ModelConfig, TrainConfig dataclasses (single source of truth)
  tokenizer.py         # BPE train + encode/decode + special tokens
  data.py              # FineWeb-Edu streaming -> packed shards -> Dataset
  model.py             # bidirectional timestep-conditioned Transformer
  diffusion.py         # forward masking + weighted CE loss
  sampler.py           # iterative unmasking generation
  train.py             # training loop (pretrain + SFT modes)
  modal_app.py         # Modal harness (A100), T4 mirror, budget guardrails
tests/
  test_tokenizer.py
  test_data.py
  test_model.py
  test_diffusion.py
  test_sampler.py
  test_train.py
```

Single source of truth for shapes/ids: `joey/config.py`. Special token ids are fixed constants there so every module agrees.

---

## Task 0: Project setup

**Files:**
- Modify: `pyproject.toml`
- Create: `joey/__init__.py`, `joey/config.py`, `tests/__init__.py`

- [ ] **Step 1: Add dependencies**

Run:
```bash
uv add torch pytest tokenizers datasets numpy
```

- [ ] **Step 2: Create config module**

Create `joey/config.py`:
```python
from dataclasses import dataclass

# Special token ids — fixed so every module agrees. Real BPE ids start after these.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
NUM_SPECIAL = 4
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]"]


@dataclass
class ModelConfig:
    vocab_size: int = 16384      # includes the 4 special tokens
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    ctx_len: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 100_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_hours: float = 9.0       # A100 budget kill-switch
    ckpt_every: int = 2000
    sample_every: int = 1000
```

- [ ] **Step 3: Create empty package markers**

Create `joey/__init__.py` (empty) and `tests/__init__.py` (empty).

- [ ] **Step 4: Verify imports**

Run: `uv run python -c "from joey.config import ModelConfig, MASK_ID; print(ModelConfig().d_model, MASK_ID)"`
Expected: `1024 3`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml joey/ tests/__init__.py uv.lock
git commit -m "chore: project setup and config"
```

---

## Task 1: Tokenizer

**Files:**
- Create: `joey/tokenizer.py`
- Test: `tests/test_tokenizer.py`

We wrap HuggingFace `tokenizers` BPE *training* primitive (a tokenization algorithm, not a model), reserving our 4 special-token ids at the front so they match `config.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tokenizer.py`:
```python
from joey.tokenizer import JoeyTokenizer
from joey.config import MASK_ID, PAD_ID, NUM_SPECIAL


def _toy_corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(["the cat sat on the mat"] * 200 +
                            ["a dog ran in the park"] * 200))
    return str(p)


def test_train_and_roundtrip(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    text = "the cat sat on the mat"
    assert tok.decode(tok.encode(text)) == text


def test_special_ids_reserved(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    assert tok.mask_id == MASK_ID
    assert tok.pad_id == PAD_ID
    # no normal token collides with a reserved special id
    ids = tok.encode("the cat")
    assert all(i >= NUM_SPECIAL for i in ids)


def test_ids_in_range(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    ids = tok.encode("a dog ran in the park")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_save_load(tmp_path):
    tok = JoeyTokenizer.train([_toy_corpus(tmp_path)], vocab_size=400)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = JoeyTokenizer.load(path)
    assert tok2.encode("the cat") == tok.encode("the cat")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.tokenizer'`

- [ ] **Step 3: Implement the tokenizer**

Create `joey/tokenizer.py`:
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from joey.config import SPECIAL_TOKENS, MASK_ID, PAD_ID, BOS_ID, EOS_ID


class JoeyTokenizer:
    def __init__(self, tk: Tokenizer):
        self._tk = tk
        self.mask_id = MASK_ID
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    @classmethod
    def train(cls, files, vocab_size=16384):
        tk = Tokenizer(BPE(unk_token=None))
        tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tk.decoder = __import__("tokenizers").decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,  # assigned ids 0..3 first
            show_progress=False,
        )
        tk.train(files, trainer)
        return cls(tk)

    def encode(self, text: str):
        return self._tk.encode(text).ids

    def decode(self, ids):
        return self._tk.decode(list(ids))

    def save(self, path: str):
        self._tk.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(Tokenizer.from_file(path))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add joey/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: BPE tokenizer with reserved special tokens"
```

---

## Task 2: Data pipeline

**Files:**
- Create: `joey/data.py`
- Test: `tests/test_data.py`

Packs a token stream into fixed-length blocks and writes/reads `.npy` shards. FineWeb-Edu streaming is wired but tests use a synthetic token stream (no network in tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:
```python
import numpy as np
from joey.data import pack_tokens, PackedShardDataset


def test_pack_exact_blocks():
    tokens = list(range(10, 10 + 25))  # 25 tokens
    blocks = pack_tokens(tokens, block_len=10)
    assert blocks.shape == (2, 10)      # 25 -> 2 full blocks, remainder dropped
    assert blocks.dtype == np.int32


def test_pack_ids_preserved():
    tokens = list(range(100, 120))
    blocks = pack_tokens(tokens, block_len=5)
    assert blocks.flatten().tolist() == list(range(100, 120))


def test_shard_dataset_roundtrip(tmp_path):
    blocks = np.arange(60, dtype=np.int32).reshape(6, 10)
    shard = tmp_path / "shard0.npy"
    np.save(shard, blocks)
    ds = PackedShardDataset([str(shard)], block_len=10)
    assert len(ds) == 6
    item = ds[3]
    assert item.shape == (10,)
    assert item.tolist() == list(range(30, 40))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.data'`

- [ ] **Step 3: Implement the data pipeline**

Create `joey/data.py`:
```python
import numpy as np
import torch
from torch.utils.data import Dataset


def pack_tokens(tokens, block_len: int) -> np.ndarray:
    """Drop the remainder, reshape into [n_blocks, block_len] int32."""
    n = (len(tokens) // block_len) * block_len
    arr = np.asarray(tokens[:n], dtype=np.int32)
    return arr.reshape(-1, block_len)


class PackedShardDataset(Dataset):
    def __init__(self, shard_paths, block_len: int):
        self.block_len = block_len
        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths]
        self.lengths = [s.shape[0] for s in self.shards]
        self.cum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        shard_i = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = idx - self.cum[shard_i]
        row = np.asarray(self.shards[shard_i][local], dtype=np.int64)
        return torch.from_numpy(row)


def build_shards_from_fineweb(tokenizer, out_dir, block_len, target_tokens,
                              tokens_per_shard=50_000_000):
    """Stream FineWeb-Edu, tokenize, write packed .npy shards. Used in real runs."""
    import os
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    buf, written, shard_i = [], 0, 0
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    for ex in ds:
        buf.extend(tokenizer.encode(ex["text"]))
        buf.append(tokenizer.eos_id)
        if len(buf) >= tokens_per_shard:
            blocks = pack_tokens(buf, block_len)
            np.save(os.path.join(out_dir, f"shard{shard_i}.npy"), blocks)
            written += blocks.size
            shard_i += 1
            buf = []
            if written >= target_tokens:
                break
    if buf:
        np.save(os.path.join(out_dir, f"shard{shard_i}.npy"),
                pack_tokens(buf, block_len))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add joey/data.py tests/test_data.py
git commit -m "feat: token packing and packed-shard dataset"
```

---

## Task 3: Model (bidirectional timestep-conditioned Transformer)

**Files:**
- Create: `joey/model.py`
- Test: `tests/test_model.py`

Key property vs GPT: **no causal mask** — every position attends to every position. Timestep `t` is embedded and added to token+position embeddings.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:
```python
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
    # The real 150M config should land near 150M (130M-180M).
    cfg = ModelConfig()
    n = sum(p.numel() for p in JoeyModel(cfg).parameters())
    assert 130_000_000 < n < 190_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.model'`

- [ ] **Step 3: Implement the model**

Create `joey/model.py`:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from joey.config import ModelConfig


def timestep_embedding(t, dim):
    """Sinusoidal embedding of continuous t in [0,1]. t: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.mlp_ratio * cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # NO causal mask -> bidirectional attention
        att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(att))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class JoeyModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.t_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x, t):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        temb = self.t_proj(timestep_embedding(t, self.cfg.d_model))  # [B, C]
        h = self.tok_emb(x) + self.pos_emb(pos)[None] + temb[:, None, :]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add joey/model.py tests/test_model.py
git commit -m "feat: bidirectional timestep-conditioned Transformer"
```

---

## Task 4: Diffusion core (forward masking + loss)

**Files:**
- Create: `joey/diffusion.py`
- Test: `tests/test_diffusion.py`

The heart. Forward masking + `1/t`-weighted CE on masked positions. Includes the critical **overfit-one-batch** gate.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diffusion.py`:
```python
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
    x = torch.randint(4, 64, (8, 16))
    last = None
    for _ in range(300):
        opt.zero_grad()
        loss = diffusion_loss(m, x, MASK_ID)
        loss.backward()
        opt.step()
        last = loss.item()
    assert last < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.diffusion'`

- [ ] **Step 3: Implement the diffusion core**

Create `joey/diffusion.py`:
```python
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
        logits.view(-1, logits.size(-1)), x.view(-1), reduction="none"
    ).view(B, -1)
    weight = (1.0 / t)[:, None].expand_as(ce)
    masked_loss = (ce * mask.float() * weight).sum() / mask.float().sum()
    return masked_loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diffusion.py -v`
Expected: PASS (5 passed). `test_overfit_one_batch` is the key gate.

- [ ] **Step 5: Commit**

```bash
git add joey/diffusion.py tests/test_diffusion.py
git commit -m "feat: masked forward process and weighted CE diffusion loss"
```

---

## Task 5: Sampler (iterative unmasking)

**Files:**
- Create: `joey/sampler.py`
- Test: `tests/test_sampler.py`

Start all-`[MASK]`; over N steps predict every masked token, commit the most-confident fraction, repeat. Supports an optional clean prompt prefix (kept unmasked) for chat.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sampler.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.sampler'`

- [ ] **Step 3: Implement the sampler**

Create `joey/sampler.py`:
```python
import torch


@torch.no_grad()
def generate(model, length, steps, mask_id, vocab_size,
             prompt_ids=None, greedy=False, device="cpu"):
    """Iterative unmasking. Start all-[MASK]; each step commit the most-confident
    masked predictions. prompt_ids (if given) stays fixed as a clean prefix."""
    x = torch.full((1, length), mask_id, dtype=torch.long, device=device)
    fixed = torch.zeros(1, length, dtype=torch.bool, device=device)
    if prompt_ids is not None:
        p = prompt_ids.to(device)
        x[0, :p.shape[1]] = p[0]
        fixed[0, :p.shape[1]] = True

    for step in range(steps):
        t_val = 1.0 - step / steps
        t = torch.full((1,), max(t_val, 1e-4), device=device)
        logits = model(x, t)
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1) if greedy else (
            probs.gather(-1, (samp := torch.multinomial(
                probs[0], 1).view(1, -1))[..., None]).squeeze(-1), samp)

        masked = (x == mask_id) & ~fixed
        if masked.sum() == 0:
            break
        # How many to commit this step: ramp so all are filled by the last step.
        remaining = int(masked.sum())
        n_commit = max(1, remaining - int(remaining * (t_val)))
        scores = conf.masked_fill(~masked, -1.0)
        order = scores[0].argsort(descending=True)
        to_fill = order[:n_commit]
        x[0, to_fill] = pred[0, to_fill]

    # Fill any stragglers greedily.
    still = (x == mask_id)
    if still.any():
        logits = model(x, torch.full((1,), 1e-4, device=device))
        x[still] = logits.argmax(-1)[still]
    return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sampler.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add joey/sampler.py tests/test_sampler.py
git commit -m "feat: iterative-unmasking sampler with prompt conditioning"
```

---

## Task 6: Training loop

**Files:**
- Create: `joey/train.py`
- Test: `tests/test_train.py`

AdamW + cosine schedule + warmup, grad clip, checkpointing, hours kill-switch. SFT mode masks only response tokens. The test runs a few CPU steps and asserts loss drops + checkpoint round-trips.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'joey.train'`

- [ ] **Step 3: Implement the training loop**

Create `joey/train.py`:
```python
import math
import time
import torch
from joey.model import JoeyModel
from joey.diffusion import diffusion_loss, mask_tokens
from joey.config import ModelConfig


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def save_ckpt(model, cfg, path, step):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, path)


def load_ckpt(path, map_location="cpu"):
    blob = torch.load(path, map_location=map_location)
    cfg = ModelConfig(**blob["cfg"])
    model = JoeyModel(cfg)
    model.load_state_dict(blob["model"])
    return model, cfg, blob["step"]


def train_steps(model, batch_iter, mask_id, n_steps, lr=3e-4, warmup=10,
                grad_clip=1.0, device="cpu", sft_response_mask=False):
    """Run n_steps; return list of losses. sft mode (response-only masking) is
    handled by the caller passing batches of (x, resp_mask) tuples."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    losses = []
    for step in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, lr, warmup, n_steps)
        batch = next(batch_iter)
        x = batch.to(device)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add joey/train.py tests/test_train.py
git commit -m "feat: training loop with cosine schedule, ckpt, hours kill-switch"
```

---

## Task 7: T4 end-to-end sanity run (manual gate, free)

**Files:**
- Create: `scripts/sanity_t4.py`

No new unit test; this is the free smoke test that must pass before spending A100 credit.

- [ ] **Step 1: Write the sanity script**

Create `scripts/sanity_t4.py`:
```python
"""Tiny end-to-end run on T4/CPU: train a small model on a small FineWeb-Edu
slice for a few hundred steps; assert loss drops and samples are valid ids."""
import torch
from joey.config import ModelConfig, TrainConfig, MASK_ID
from joey.model import JoeyModel
from joey.tokenizer import JoeyTokenizer
from joey.data import build_shards_from_fineweb, PackedShardDataset
from joey.train import train_steps
from joey.sampler import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. small tokenizer (train on a tiny slice first, or load if present)
    tok = JoeyTokenizer.load("artifacts/tok.json")
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=256, n_layers=4,
                      n_heads=4, ctx_len=128)
    build_shards_from_fineweb(tok, "data/sanity", cfg.ctx_len,
                              target_tokens=2_000_000)
    ds = PackedShardDataset(sorted(__import__("glob").glob("data/sanity/*.npy")),
                            cfg.ctx_len)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    it = iter(loader)
    def batches():
        nonlocal it
        while True:
            try: yield next(it)
            except StopIteration:
                it = iter(loader); yield next(it)
    m = JoeyModel(cfg)
    losses = train_steps(m, batches(), MASK_ID, n_steps=500, lr=3e-4, device=DEVICE)
    print("loss start->end:", losses[0], "->", losses[-1])
    assert losses[-1] < losses[0], "loss did not decrease"
    out = generate(m.to(DEVICE), length=cfg.ctx_len, steps=32, mask_id=MASK_ID,
                   vocab_size=cfg.vocab_size, device=DEVICE)
    print("SAMPLE:", tok.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train a sanity tokenizer**

Run:
```bash
uv run python -c "
from joey.tokenizer import JoeyTokenizer
from datasets import load_dataset
import os; os.makedirs('artifacts', exist_ok=True)
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
with open('artifacts/corpus.txt','w') as f:
    for i,ex in zip(range(20000), ds): f.write(ex['text']+'\n')
JoeyTokenizer.train(['artifacts/corpus.txt'], vocab_size=16384).save('artifacts/tok.json')
print('tokenizer trained')
"
```
Expected: `tokenizer trained`

- [ ] **Step 3: Run the sanity script**

Run: `uv run python scripts/sanity_t4.py`
Expected: prints a decreasing `loss start->end`, then a `SAMPLE:` line of decoded text (word-like fragments, not gibberish-ids). **Gate: loss must decrease.**

- [ ] **Step 4: Commit**

```bash
git add scripts/sanity_t4.py
git commit -m "feat: T4 end-to-end sanity script (pre-A100 gate)"
```

---

## Task 8: Modal harness + real A100 run (spends credit)

**Files:**
- Create: `joey/modal_app.py`

Wraps the full run on a Modal A100 with the hours kill-switch from `TrainConfig`. Run only after Task 7 passes.

- [ ] **Step 1: Write the Modal app**

Create `joey/modal_app.py`:
```python
import modal

app = modal.App("joey-diffusion")
image = (modal.Image.debian_slim()
         .pip_install("torch", "tokenizers", "datasets", "numpy"))
vol = modal.Volume.from_name("joey-data", create_if_missing=True)


@app.function(gpu="A100", image=image, volumes={"/vol": vol},
              timeout=60 * 60 * 10)
def run_training(max_hours: float = 9.0, target_tokens: int = 1_500_000_000):
    import glob, torch
    from joey.config import ModelConfig, TrainConfig, MASK_ID
    from joey.model import JoeyModel
    from joey.tokenizer import JoeyTokenizer
    from joey.data import build_shards_from_fineweb, PackedShardDataset
    from joey.train import train
    from joey.sampler import generate

    tok = JoeyTokenizer.load("/vol/tok.json")
    cfg = ModelConfig(vocab_size=tok.vocab_size)          # full 150M config
    tcfg = TrainConfig(max_hours=max_hours)
    if not glob.glob("/vol/shards/*.npy"):
        build_shards_from_fineweb(tok, "/vol/shards", cfg.ctx_len, target_tokens)
        vol.commit()
    ds = PackedShardDataset(sorted(glob.glob("/vol/shards/*.npy")), cfg.ctx_len)

    def sampler_cb(model, step):
        out = generate(model, length=cfg.ctx_len, steps=32, mask_id=MASK_ID,
                       vocab_size=cfg.vocab_size, device="cuda")
        print(f"[{step}] {tok.decode(out[0].tolist())[:200]}")
        vol.commit()

    train(JoeyModel(cfg), ds, cfg, tcfg, MASK_ID, "cuda",
          "/vol/joey_base.pt", sampler_cb=sampler_cb)
    vol.commit()


@app.local_entrypoint()
def main(max_hours: float = 9.0):
    run_training.remote(max_hours=max_hours)
```

- [ ] **Step 2: Upload the trained tokenizer to the Modal volume**

Run:
```bash
uv run modal volume put joey-data artifacts/tok.json /tok.json
```
Expected: upload confirmation.

- [ ] **Step 3: Launch the A100 run with the budget kill-switch**

Run: `uv run modal run joey/modal_app.py --max-hours 9`
Expected: streamed `step N loss ...` lines and periodic decoded samples that grow more word-like; stops at the hours kill-switch or `max_steps`, checkpoint saved to the volume. **Watch cost; this spends the $30.**

- [ ] **Step 4: Pull the checkpoint back**

Run: `uv run modal volume get joey-data /joey_base.pt artifacts/joey_base.pt`
Expected: `joey_base.pt` downloaded.

- [ ] **Step 5: Commit**

```bash
git add joey/modal_app.py
git commit -m "feat: Modal A100 harness with budget kill-switch"
```

---

## Task 9: Instruction SFT + chat (makes Joey chattable)

**Files:**
- Modify: `joey/diffusion.py` (add response-only masking loss)
- Create: `scripts/sft.py`, `scripts/chat.py`
- Test: `tests/test_diffusion.py` (add SFT-loss test)

Phase 2: finetune on a small instruction dataset, masking **only response tokens** (prompt stays clean) — LLaDA's SFT recipe. Then a simple chat REPL.

- [ ] **Step 1: Add the failing SFT-loss test**

Add to `tests/test_diffusion.py`:
```python
def test_sft_loss_only_masks_response():
    import torch
    from joey.config import ModelConfig, MASK_ID
    from joey.model import JoeyModel
    from joey.diffusion import sft_diffusion_loss
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, ctx_len=16)
    m = JoeyModel(cfg)
    x = torch.randint(4, 64, (4, 16))
    resp = torch.zeros(4, 16, dtype=torch.bool); resp[:, 8:] = True  # response half
    loss = sft_diffusion_loss(m, x, resp, MASK_ID, force_t=torch.zeros(4))
    assert loss.item() == 0.0   # t=0 -> nothing masked -> 0 loss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diffusion.py::test_sft_loss_only_masks_response -v`
Expected: FAIL with `ImportError: cannot import name 'sft_diffusion_loss'`

- [ ] **Step 3: Implement response-only SFT loss**

Add to `joey/diffusion.py`:
```python
def sft_diffusion_loss(model, x, resp_mask, mask_id, force_t=None):
    """Like diffusion_loss but only response tokens are eligible for masking,
    and loss is computed only there (LLaDA SFT). resp_mask: [B,T] bool."""
    B = x.shape[0]
    t = force_t if force_t is not None else torch.rand(B, device=x.device)
    t = t.clamp(min=EPS, max=1.0)
    probs = t[:, None].expand_as(x) * resp_mask.float()
    mask = torch.rand_like(x, dtype=torch.float) < probs
    if mask.sum() == 0:
        return torch.zeros((), device=x.device)
    x_t = torch.where(mask, torch.full_like(x, mask_id), x)
    logits = model(x_t, t)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1),
                         reduction="none").view(B, -1)
    weight = (1.0 / t)[:, None].expand_as(ce)
    return (ce * mask.float() * weight).sum() / mask.float().sum()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_diffusion.py::test_sft_loss_only_masks_response -v`
Expected: PASS

- [ ] **Step 5: Write the SFT script**

Create `scripts/sft.py`:
```python
"""Finetune the base checkpoint on a small instruction dataset (response-only
masking). Format each example as [BOS] prompt [EOS] response, padded to ctx_len."""
import glob, torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from joey.config import MASK_ID, BOS_ID, EOS_ID, PAD_ID
from joey.tokenizer import JoeyTokenizer
from joey.train import load_ckpt, save_ckpt, cosine_lr
from joey.diffusion import sft_diffusion_loss

def build(tok, ctx):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    X, R = [], []
    for ex in ds:
        prompt = tok.encode(ex["instruction"] + ("\n" + ex["input"] if ex["input"] else ""))
        resp = tok.encode(ex["output"])
        ids = [BOS_ID] + prompt + [EOS_ID] + resp
        ids = ids[:ctx] + [PAD_ID] * max(0, ctx - len(ids))
        rmask = [False] * (len(prompt) + 2) + [True] * len(resp)
        rmask = rmask[:ctx] + [False] * max(0, ctx - len(rmask))
        X.append(ids); R.append(rmask)
    return TensorDataset(torch.tensor(X), torch.tensor(R, dtype=torch.bool))

def main():
    tok = JoeyTokenizer.load("artifacts/tok.json")
    model, cfg, _ = load_ckpt("artifacts/joey_base.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).train()
    data = build(tok, cfg.ctx_len)
    loader = DataLoader(data, batch_size=16, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    step, total = 0, 3000
    while step < total:
        for x, r in loader:
            for g in opt.param_groups: g["lr"] = cosine_lr(step, 1e-4, 100, total)
            opt.zero_grad()
            loss = sft_diffusion_loss(model, x.to(dev), r.to(dev), MASK_ID)
            loss.backward(); opt.step(); step += 1
            if step % 200 == 0: print(f"sft step {step} loss {loss.item():.3f}")
            if step >= total: break
    save_ckpt(model, cfg, "artifacts/joey_chat.pt", step)

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the chat REPL**

Create `scripts/chat.py`:
```python
"""Talk to Joey. Encodes your prompt as a clean prefix, samples the response
region by iterative unmasking."""
import torch
from joey.config import MASK_ID, BOS_ID, EOS_ID
from joey.tokenizer import JoeyTokenizer
from joey.train import load_ckpt
from joey.sampler import generate

def main():
    tok = JoeyTokenizer.load("artifacts/tok.json")
    model, cfg, _ = load_ckpt("artifacts/joey_chat.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    while True:
        msg = input("you> ")
        if msg.strip() in ("quit", "exit"): break
        prompt = torch.tensor([[BOS_ID] + tok.encode(msg) + [EOS_ID]])
        out = generate(model, length=cfg.ctx_len, steps=64, mask_id=MASK_ID,
                       vocab_size=cfg.vocab_size, prompt_ids=prompt, device=dev)
        resp = out[0, prompt.shape[1]:].tolist()
        print("joey>", tok.decode(resp))

if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run SFT (on Modal A100 or T4 — it's short) and chat**

Run:
```bash
uv run python scripts/sft.py
uv run python scripts/chat.py
```
Expected: SFT loss decreases; chat REPL responds with (rough but recognizable) text. **This is the finish line: talking to Joey.**

- [ ] **Step 8: Commit**

```bash
git add joey/diffusion.py tests/test_diffusion.py scripts/sft.py scripts/chat.py
git commit -m "feat: instruction SFT (response-only masking) and chat REPL"
```

---

## Self-Review Notes

- **Spec coverage:** tokenizer (T1), data pipeline (T2), model/bidirectional/timestep (T3), diffusion forward+loss (T4), sampler (T5), training loop+kill-switch (T6), T4 sanity (T7), Modal A100 real run (T8), instruction SFT response-only + chat (T9). All spec components mapped.
- **Type consistency:** special-token ids come from `joey/config.py` everywhere; `generate(...)` signature is identical across sampler tests, sanity, modal, and chat; `diffusion_loss`/`sft_diffusion_loss` share the `(model, x, ..., mask_id, force_t=None)` shape.
- **Budget discipline:** T7 (free) gates T8 (paid); `TrainConfig.max_hours` kill-switch enforced in `train()`.
- **Known caveat (from spec):** general-text coherence at 150M / ~1.5B tokens will be rough; SFT improves perceived quality. Logged, not hidden.
```
