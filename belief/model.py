"""PyTorch model B: bidirectional transformer denoiser for masked discrete diffusion.

Inputs (int64): cur/last/age/xt [B,64], inv [B,7], ply [B], to_move [B].
Output: logits [B,64,7] over {empty, enemy P N B R Q K} for every square
(only unseen squares are meaningful). No time embedding: MDLM's SUBS
parameterisation is time-independent.

xt vocabulary (9): 0-6 revealed value, 7 MASK, 8 VIS (square is visible).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CUR, N_LAST, N_AGE, N_XT, N_OUT = 14, 8, 6, 9, 7
MASK, VIS = 7, 8
N_INV, N_COUNT, N_PLY = 7, 10, 32


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    # ablation switches: a disabled input is replaced by a constant index,
    # so the ONNX signature never changes
    use_inv: bool = True
    use_last: bool = True
    use_age: bool = True


class Block(nn.Module):
    def __init__(self, d: int, h: int, ff: int, p: float):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class BeliefDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.cur = nn.Embedding(N_CUR, d)
        self.last = nn.Embedding(N_LAST, d)
        self.age = nn.Embedding(N_AGE, d)
        self.xt = nn.Embedding(N_XT, d)
        self.pos = nn.Embedding(64, d)
        self.inv_type = nn.Embedding(N_INV, d)
        self.inv_count = nn.Embedding(N_COUNT, d)
        self.ply = nn.Embedding(N_PLY, d)
        self.to_move = nn.Embedding(2, d)
        self.glob = nn.Embedding(2, d)          # marks the ply / to_move tokens
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_OUT)
        self.register_buffer('sq_idx', torch.arange(64), persistent=False)
        self.register_buffer('inv_idx', torch.arange(N_INV), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, cur, last, age, xt, inv, ply, to_move):
        cfg = self.cfg
        if not cfg.use_last:
            last = torch.zeros_like(last)
        if not cfg.use_age:
            age = torch.zeros_like(age)
        if not cfg.use_inv:
            inv = torch.zeros_like(inv)
        sq = self.cur(cur) + self.last(last) + self.age(age) + self.xt(xt) + self.pos(self.sq_idx)
        inv_tok = self.inv_type(self.inv_idx) + self.inv_count(inv.clamp(0, N_COUNT - 1))
        ply_tok = (self.ply(ply) + self.glob.weight[0]).unsqueeze(1)
        tm_tok = (self.to_move(to_move) + self.glob.weight[1]).unsqueeze(1)
        x = self.drop(torch.cat([sq, inv_tok, ply_tok, tm_tok], dim=1))     # [B, 73, d]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, :64]))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def config_for_params(target: float) -> ModelConfig:
    """Rough presets for the size sweep (0.1M / 0.5M / 0.8M / 2M)."""
    presets = {0.1: (64, 2, 2, 256), 0.5: (112, 3, 4, 448), 0.8: (128, 4, 4, 512), 2.0: (192, 5, 6, 768)}
    key = min(presets, key=lambda k: abs(math.log(k / target)))
    d, l, h, ff = presets[key]
    return ModelConfig(d_model=d, n_layers=l, n_heads=h, d_ff=ff)


def build(cfg: ModelConfig | dict) -> BeliefDenoiser:
    if isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)
    return BeliefDenoiser(cfg)


__all__ = ['ModelConfig', 'BeliefDenoiser', 'build', 'n_params', 'config_for_params', 'asdict', 'MASK', 'VIS']
