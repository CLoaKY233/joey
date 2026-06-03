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
