# Joey 🐤 — a diffusion language model, from scratch

[![Weights on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/cloaky/joey)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A small **masked-diffusion LLM** built from the ground up in PyTorch — custom tokenizer,
> bidirectional Transformer, the diffusion noising/denoising objective, and an iterative-unmasking
> sampler. No HuggingFace `Trainer`, no pretrained weights — every component is hand-written.

**Status: 🚧 work in progress.** A ~170M base model is trained and chats at a basic level; the
current focus is scaling the model up for genuine coherence (see [Roadmap](#roadmap)).

Unlike GPT-style models that write left-to-right one token at a time, Joey generates text by
**iterative denoising**: it starts from a fully `[MASK]`ed sequence and progressively fills it in,
re-deciding low-confidence tokens along the way. This is the MDLM / LLaDA family of discrete
diffusion language models.

---

## Why I built this

To actually *understand* how diffusion LLMs work by implementing one end to end — the architecture,
attention, the forward/reverse diffusion processes, and sampling — and to train something I can talk
to on a tiny budget. The conceptual write-ups I made along the way live in
[`docs/diffusion-llm/`](docs/diffusion-llm/) (overview + masked diffusion + D3PM + SEDD, with
mermaid diagrams).

## How it works (in one diagram)

```
FineWeb-Edu ── BPE ──▶ packed token blocks
                            │
                   mask each token w.p. t        (forward process — fixed)
                            │
              bidirectional Transformer(+ t)     (reverse process — learned)
                            │
        1/t-weighted cross-entropy on masked positions
                            │  (after training)
   all-[MASK] ──▶ predict · keep confident · re-mask rest ──▶ text   (sampling)
```

- **Forward process:** corrupt text by replacing tokens with `[MASK]` at a random rate `t`.
- **Reverse process:** a bidirectional, timestep-conditioned Transformer predicts the originals.
- **Loss:** cross-entropy on the masked positions only, `1/t`-weighted (the MDLM objective).
- **Sampling:** MaskGIT/LLaDA-style **remasking** — predict everything, keep the most confident
  tokens, re-mask the rest, repeat. This is what breaks the repetition loops naive samplers fall into.

## Architecture

| Property | Value |
|---|---|
| Parameters | ~170M |
| Backbone | Bidirectional Transformer (no causal mask), timestep-conditioned |
| `d_model` / layers / heads | 1024 / 12 / 16 |
| Context length | 256 tokens |
| Vocabulary | 16,384 (custom ByteLevel BPE + `[PAD] [BOS] [EOS] [MASK]`) |
| MLP | 4× GELU, pre-norm, weight-tied head |
| Diffusion | Masked / absorbing-state (MDLM / LLaDA family) |

## Training

| Stage | Details |
|---|---|
| Data | FineWeb-Edu (general web text), ~2B tokens, own 16K BPE tokenizer |
| Base | A100-40GB, bf16 + EMA, cosine LR + warmup, 174K steps (~6h), gradient-accum, hours kill-switch |
| Fine-tune | DailyDialog, response-only masking (LLaDA-style SFT) |
| Sampler | Remasking (MaskGIT/LLaDA) + repetition penalty + top-p |

## Results (honest)

After base training + conversation fine-tuning, Joey greets correctly, forms grammatical sentences,
and stays in a conversational register:

```
you> Hi!
joey> Oh, I am right! It's my favorite, we have always been there for a long time...

you> Do you like music?
joey> I don't know that much. But I think there is no one...
```

It is **fluent but not yet truly coherent** — correct local grammar without sustained global meaning.
That's the signature of a *capacity* ceiling: 170M parameters is small, and the model had essentially
converged for its size. Getting genuinely meaningful output is primarily a question of **scale**
(more parameters + tokens), which is the next milestone. More samples:
[`docs/joey-first-conversations.md`](docs/joey-first-conversations.md).

## Repo layout

```
joey/
  config.py       # model / training config + special-token ids
  tokenizer.py    # ByteLevel BPE (train + encode/decode)
  data.py         # streaming tokenization → packed shards → dataset
  model.py        # bidirectional, timestep-conditioned Transformer
  diffusion.py    # forward masking + 1/t-weighted CE loss (+ SFT variant)
  sampler.py      # remasking iterative-unmasking generation
  train.py        # training loop: bf16, EMA, cosine LR, checkpoint/resume
  sft.py          # conversational fine-tuning (response-only masking)
  modal_app.py    # cloud training/eval orchestration (Modal)
scripts/          # sanity run, SFT, interactive chat
tests/            # unit tests for every module
docs/             # diffusion-LLM explainers + sample conversations
```

## Setup & usage

Requires [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
uv run pytest          # run the test suite

# get the trained weights from the Hugging Face Hub
uv run hf download cloaky/joey joey_chat.pt tok.json --local-dir artifacts

# chat with the model
uv run python scripts/chat.py
```

Pretrained weights (`joey_chat.pt`, `joey_base.pt`, `tok.json`) live on the
[Hugging Face Hub](https://huggingface.co/cloaky/joey).

Training is orchestrated on [Modal](https://modal.com) (`joey/modal_app.py`); a local sanity run is
in `scripts/sanity_t4.py`.

## Roadmap

- [x] From-scratch tokenizer, model, diffusion loss, sampler, training loop
- [x] Base pretraining on ~2B tokens + conversational SFT
- [x] Remasking sampler to eliminate repetition loops
- [ ] **Scale up the model (~400M → 1B)** for real coherence — *in progress*
- [ ] Larger / cleaner instruction-tuning data
- [ ] Classifier-free guidance for conditional sampling
- [ ] Longer context

## References

- Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (MDLM), NeurIPS 2024
- Nie et al., *Large Language Diffusion Models* (LLaDA), 2025
- Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM), NeurIPS 2021
- Lou, Meng, Ermon, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD), ICML 2024
- Chang et al., *MaskGIT: Masked Generative Image Transformers*, CVPR 2022

## License

MIT — see [LICENSE](LICENSE).
