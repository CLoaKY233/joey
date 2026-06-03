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
