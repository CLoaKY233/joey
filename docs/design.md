# Design notes

Design decisions and trade-offs behind Joey. For the conceptual background on diffusion language
models, see [`diffusion-llm/`](diffusion-llm/).

## Goal

Build a masked-diffusion LLM from scratch — small enough to train on a tight budget, but complete
enough to understand every moving part and actually chat with. Learning is the point, so nothing is
delegated to high-level training frameworks.

## Why masked / absorbing-state diffusion

Discrete text diffusion has a few formulations (masked/MDLM, D3PM, SEDD). Masked diffusion was chosen
because:

- The loss collapses to **weighted cross-entropy on masked positions** — simple to implement and
  forgiving to debug at small scale.
- It is the formulation behind **LLaDA**, so the learning transfers to a real, scaled-up model.
- It maps cleanly onto a familiar BERT-style masked-prediction objective, run at a *random* mask rate.

D3PM (transition-matrix diffusion) and SEDD (score-entropy) are more general/powerful but carry
heavier math and more ways to get subtly wrong — documented in `diffusion-llm/` as future reading.

## Key components and boundaries

Each module has a single responsibility and is unit-tested in isolation:

- **Tokenizer** — own ByteLevel BPE with reserved special-token ids (`[PAD] [BOS] [EOS] [MASK]`).
- **Data** — stream a web corpus, tokenize in batches, pack into fixed-length blocks, shard to disk.
- **Model** — a Transformer with **full (bidirectional) attention** and timestep conditioning.
  Bidirectionality matters: diffusion refines the whole sequence at once, so there is no causal mask.
- **Diffusion** — the forward masking process and the `1/t`-weighted cross-entropy loss; plus a
  response-only variant for instruction tuning.
- **Sampler** — remasking iterative decoding (predict all, keep confident, re-mask the rest).
- **Train / SFT** — the optimization loops, checkpointing, and fine-tuning.

## Notable engineering decisions

- **Bidirectional attention, timestep-conditioned.** A sinusoidal timestep embedding is added to the
  token/position embeddings so the model knows the current noise level.
- **`1/t` loss weighting.** Turns the masked cross-entropy into a proper diffusion likelihood bound
  (the MDLM simplification).
- **Remasking sampler.** Permanent-commit sampling causes repetition collapse, because masked
  positions are predicted independently and the model stamps the same high-probability token into
  many slots. Re-masking low-confidence tokens each step lets the model correct itself — the standard
  cure (MaskGIT / LLaDA).
- **Memory.** Training memory is dominated by the output logits tensor (`batch × seq × vocab`), not
  the parameters — so effective batch size is reached via gradient accumulation rather than a large
  microbatch.
- **EMA weights** for cleaner samples; **bf16** autocast; **cosine LR with warmup**;
  **checkpoint/resume** and an hours-based cost kill-switch for cloud runs.

## Known limitations

At ~170M parameters the model is fluent but capacity-limited — it learns grammar and conversational
register but not sustained meaning. Improving this is primarily a matter of scale (parameters +
tokens), which is the project's current direction.
