# Joey — A Diffusion LLM From Scratch (Design Spec)

**Date:** 2026-06-01
**Status:** Approved design → ready for implementation plan

## Goal

Build a masked/absorbing-state diffusion language model (MDLM / LLaDA-style) **entirely from
scratch**, trained on general web text, small enough to train within ~$30 of Modal A100 time, that
we can chat with — and, above all, **fully understand**. The artifact is a working diffusion LM;
the real prize is deep understanding of every part (architecture, attention, the noising/denoising
loop, sampling).

Learning context and method comparison live in `docs/diffusion-llm/` (overview + masked/D3PM/SEDD).

## Non-Goals

- Not ChatGPT-quality open-domain chat — physically impossible at this scale/budget.
- No use of HuggingFace Trainer, `diffusers`, or pretrained weights. We write the model, diffusion,
  sampler, and tokenizer training ourselves.

## Decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Diffusion method | **Masked / absorbing-state** (MDLM/LLaDA) | Simplest loss (weighted CE), forgiving to debug, transfers to LLaDA |
| Dataset | **FineWeb-Edu slice**, ~1–2B tokens | General-domain text; we are compute-bound, not data-bound |
| Tokenizer | **Own BPE, ~16k vocab** + special tokens | Right-sized for a small model; learn tokenization from scratch |
| Model size | **~150M params** (`d_model 1024, 12 layers, 16 heads, ctx 256`) | Best quality within budget |
| Attention | **Bidirectional, no causal mask** | Diffusion refines the whole sequence in parallel |
| Infra | **T4 (local, free) for dev + sanity; A100 (Modal) for real run** | Spend $30 credit only on a proven pipeline |
| Stack | Python 3.11, PyTorch, `uv`, Modal | — |

## Budget Discipline

- ~$30 ≈ ~8–12 A100-hours total. A100-40GB ≈ $2.5/h, 80GB ≈ $3.7/h.
- All development and a full end-to-end **sanity run happen free on the T4**.
- A100 credit is spent **only once the full loop is proven on T4**.
- Hard **kill-switch on wall-clock hours** + checkpoint/resume so a crash never wastes credit.
- Realistic token budget for the real run: ~1–2B tokens (≈ 20 tokens/param for 150M is ~3B; we cap
  to fit credit and log the shortfall honestly).

## Architecture

```
FineWeb-Edu  ->  BPE tokenizer  ->  packed shards (len 256)
                                          |
                                   mask each token w.p. t   (forward process, fixed)
                                          |
                                bidirectional Transformer (+ timestep t)
                                          |
                              1/t-weighted cross-entropy on MASKED positions only
                                          |  (after training)
                          all-[MASK]  ->  iterative unmask (N steps)  ->  text
```

### Components (each a focused, independently testable module)

**1. Tokenizer** — own BPE, ~16k vocab, trained on a FineWeb-Edu slice. Special tokens:
`[MASK]` (absorbing state), `[PAD]`, `[BOS]`, `[EOS]`.
- *Interface:* `encode(str) -> ids`, `decode(ids) -> str`, reserved special-token ids.
- *Tests:* encode→decode round-trip; `[MASK]`/`[PAD]`/`[BOS]`/`[EOS]` reserved and stable; all ids
  within `[0, vocab)`.

**2. Data pipeline** — stream FineWeb-Edu, tokenize, pack into fixed-length blocks (256 tokens),
shard to disk; target ~1–2B tokens. Streaming so we never hold the corpus in memory.
- *Interface:* iterable/Dataset of `LongTensor[batch, 256]`.
- *Tests:* every block is exactly 256 long; all ids in range; shards reload deterministically.

**3. Model** — decoder-style Transformer with **bidirectional full self-attention (no causal
mask)** and timestep conditioning. Config: `d_model 1024, n_layers 12, n_heads 16, ctx 256,
vocab ~16k` → ~150M params. Hand-written: token + positional embeddings, sinusoidal/learned
timestep embedding injected into blocks, multi-head full self-attention, MLP blocks (GELU),
pre-norm (LayerNorm/RMSNorm), final vocab head (optionally weight-tied).
- *Interface:* `model(x_ids[B,256], t[B]) -> logits[B,256,vocab]`.
- *Tests:* output shape; attention is full (position 0 influenced by position 255); gradients flow
  to all params; param count ≈ 150M.

**4. Diffusion core** — the heart, kept tiny and heavily tested.
- Forward: sample `t ~ U(0,1)`; mask each token independently w.p. `t` → `[MASK]`.
- Loss: cross-entropy on **masked positions only**, weighted by `1/t` (MDLM weighting).
- *Tests:* `t→0` ⇒ ~no masks; `t→1` ⇒ ~all masks; loss ignores unmasked positions; model can
  **overfit a single batch** to near-zero loss (the key correctness gate).

**5. Sampler** — start from all-`[MASK]`; over `N` steps predict all masked tokens, **commit the
most-confident** fraction, re-mask the rest, repeat until none masked. Exposes the
steps↔quality knob.
- *Interface:* `generate(prompt_ids?, steps, length) -> ids`.
- *Tests:* outputs valid ids; varying `steps` never crashes; with an overfit model, reproduces the
  memorized sequence.

**6. Training loop** — AdamW, cosine LR schedule with warmup, mixed precision (bf16/fp16),
gradient clipping, periodic checkpointing, loss + sample-generation logging.
- *Phase 1 — base pretrain:* mask anywhere across packed blocks.
- *Phase 2 — instruction SFT:* small chat/instruction dataset; **mask only the response tokens**
  (LLaDA-style), prompt stays clean. This is what makes it chattable.

**7. Modal harness** — wrap training for A100; mirror path for local T4 dev. Budget guardrails:
max-hours kill-switch, checkpoint resume, cost logging.

## Build & Verification Order

Each step is verified before the next begins:

1. Tokenizer (train + round-trip tests)
2. Data pipeline (packed shards, range tests)
3. Model forward (shape/attention/grad tests)
4. Diffusion core → **overfit one batch to ~0 loss** (primary correctness gate)
5. Sampler (recovers overfit sequence)
6. Small **T4 end-to-end run** (loss decreases, samples become word-like) — free
7. **A100 real run** on Modal (base pretrain, ~1–2B tokens) — spends credit
8. Instruction SFT → chat
9. Talk to Joey

## Testing Strategy

- TDD per module: write the interface test first, then implement (see component tests above).
- The two gates that catch most bugs: **encode→decode round-trip** and **overfit-one-batch**.
- End-to-end smoke test on T4 before any paid compute.

## Risks & Mitigations

- *Burning credit on a bug* → full T4 sanity run + hours kill-switch + resume.
- *Rough coherence on general text at 150M* → expected; accept and log honestly; instruction SFT
  improves perceived quality. TinyStories remains a fallback to demonstrate the loop works.
- *Sampler quality* → tune steps + confidence schedule; this is the main quality knob.
- *Tokenizer/vocab bugs poisoning everything* → strong round-trip tests up front.

## Stack

Python 3.11 · PyTorch · `uv` (deps) · Modal (cloud). From scratch: model, diffusion, sampler, BPE
training. Allowed libs: PyTorch, a BPE trainer primitive only if we choose (else hand-rolled),
datasets streaming for FineWeb-Edu ingestion.
