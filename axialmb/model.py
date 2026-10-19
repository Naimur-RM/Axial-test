"""Axial set-prediction network over a whole data matrix.

Input is ``N`` rows by ``p`` variables. Each block attends across variables
inside a row, then across rows inside a variable through induced-set attention,
so memory is linear in ``N``. There is no positional encoding: the network is
invariant to row order and equivariant to column order.

After the blocks, rows are averaged into one token per variable. A few column
layers mix those tokens, with attention biased by pairwise statistics.

Three heads:

- pair: one membership logit per pair, symmetric by construction
- role: none, adjacent (parent or child), or spouse
- set size: a distribution over ``|MB(i)|`` in ``0..MAX_MB``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import MAX_MB, MAX_STATES
from .stats import N_COL, N_PAIR


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d={d} must be divisible by heads={heads}")
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, query, context, mask=None):
        b, lq, d = query.shape
        lk = context.shape[1]
        q = self.q(query).view(b, lq, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.view(b, lk, self.h, d // self.h).transpose(1, 2)
        v = v.view(b, lk, self.h, d // self.h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(b, lq, d))


class InducedSetAttention(nn.Module):
    """``m`` learned points read the rows, then the rows read the points."""

    def __init__(self, d, heads, m):
        super().__init__()
        self.points = nn.Parameter(torch.randn(m, d) * 0.02)
        self.read = Attention(d, heads)
        self.write = Attention(d, heads)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        pts = self.points.unsqueeze(0).expand(x.shape[0], -1, -1)
        summary = pts + self.read(pts, x)
        return self.write(x, self.ln(summary))


class AxialBlock(nn.Module):
    def __init__(self, d, heads, inducing, ff_mult=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.row_attn = Attention(d, heads)
        self.col_attn = InducedSetAttention(d, heads, inducing)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, row_mask):
        b, n, p, d = h.shape
        x = self.ln1(h).reshape(b * n, p, d)
        mask = row_mask.unsqueeze(1).expand(b, n, p).reshape(b * n, 1, 1, p)
        h = h + self.drop(self.row_attn(x, x, mask).reshape(b, n, p, d))
        x = self.ln2(h).permute(0, 2, 1, 3).reshape(b * p, n, d)
        col = self.col_attn(x).reshape(b, p, n, d).permute(0, 2, 1, 3)
        h = h + self.drop(col)
        return h + self.drop(self.ff(self.ln3(h)))


class ColumnLayer(nn.Module):
    """Self-attention among variable tokens, biased by pairwise statistics."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.bias = nn.Sequential(nn.Linear(N_PAIR, d // 2), nn.GELU(), nn.Linear(d // 2, heads))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, pair_feat, col_mask):
        bias = self.bias(pair_feat).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~col_mask[:, None, None, :], float("-inf"))
        x = self.ln1(h)
        h = h + self.attn(x, x, bias.to(x.dtype))
        return h + self.ff(self.ln2(h))


class AxialMB(nn.Module):
    def __init__(self, d=64, heads=4, blocks=4, inducing=32, col_layers=2,
                 dropout=0.0, grad_checkpoint=False):
        super().__init__()
        self.d = d
        self.grad_checkpoint = grad_checkpoint
        self.val_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.cat_emb = nn.Embedding(MAX_STATES, d)
        self.freq_emb = nn.Linear(1, d)
        self.type_emb = nn.Embedding(2, d)
        self.col_proj = nn.Linear(N_COL, d)
        self.blocks = nn.ModuleList(
            AxialBlock(d, heads, inducing, dropout=dropout) for _ in range(blocks)
        )
        self.ln_pool = nn.LayerNorm(d)
        self.columns = nn.ModuleList(ColumnLayer(d, heads) for _ in range(col_layers))
        self.pair_proj = nn.Linear(N_PAIR, d)
        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(),
        )
        self.member_out = nn.Linear(d, 1)
        self.role_out = nn.Linear(d, 3)
        self.count_mlp = nn.Sequential(
            nn.Linear(3 * d + 1, 2 * d), nn.GELU(), nn.Linear(2 * d, MAX_MB + 1),
        )

    def forward(self, x, codes, freq, is_cat, col_feat, pair_feat, col_mask):
        """
        x, freq: (B, N, p) float. codes: (B, N, p) long. is_cat, col_mask: (B, p) bool.
        col_feat: (B, p, N_COL). pair_feat: (B, p, p, N_PAIR).
        """
        b, n, p = x.shape
        cont = self.val_emb(x.unsqueeze(-1))
        cat = self.cat_emb(codes) + self.freq_emb(freq.unsqueeze(-1))
        h = torch.where(is_cat[:, None, :, None], cat, cont)
        col = self.type_emb(is_cat.long()) + self.col_proj(col_feat)
        h = h + col.unsqueeze(1)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                h = checkpoint(blk, h, col_mask, use_reentrant=False)
            else:
                h = blk(h, col_mask)
        tok = self.ln_pool(h.mean(dim=1))
        for layer in self.columns:
            tok = layer(tok, pair_feat, col_mask)

        hi = tok.unsqueeze(2).expand(b, p, p, self.d)
        hj = tok.unsqueeze(1).expand(b, p, p, self.d)
        pair_in = torch.cat([hi * hj, (hi - hj).abs(), hi + hj, self.pair_proj(pair_feat)], dim=-1)
        g = self.pair_mlp(pair_in)
        member = self.member_out(g).squeeze(-1)
        role = self.role_out(g)

        valid = col_mask[:, :, None] & col_mask[:, None, :]
        valid = valid & ~torch.eye(p, dtype=torch.bool, device=x.device)[None]
        vf = valid.unsqueeze(-1).to(g.dtype)
        n_valid = vf.sum(2).clamp(min=1.0)
        g_mean = (g * vf).sum(2) / n_valid
        g_max = g.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(2)
        g_max = torch.where(torch.isfinite(g_max), g_max, torch.zeros_like(g_max))
        soft = (torch.sigmoid(member.float()) * valid.float()).sum(2, keepdim=True) / MAX_MB
        count = self.count_mlp(torch.cat([tok, g_mean, g_max, soft.to(tok.dtype)], dim=-1))
        limit = valid.sum(2, keepdim=True)
        ks = torch.arange(MAX_MB + 1, device=x.device).view(1, 1, -1)
        count = count.float().masked_fill(ks > limit, float("-inf"))
        return dict(member=member.float(), role=role.float(), count=count, valid=valid)


def build_model(cfg) -> AxialMB:
    return AxialMB(
        d=cfg["d"],
        heads=cfg["heads"],
        blocks=cfg["blocks"],
        inducing=cfg["inducing"],
        col_layers=cfg["col_layers"],
        dropout=cfg.get("dropout", 0.0),
        grad_checkpoint=cfg.get("grad_checkpoint", True),
    )


def invariance_check(model, batch, atol=1e-4):
    """Row shuffles leave every output unchanged. Column shuffles permute it."""
    model.eval()
    with torch.no_grad():
        base = model(**batch)["member"]
        n = batch["x"].shape[1]
        perm = torch.randperm(n)
        rows = {k: (v[:, perm] if k in ("x", "codes", "freq") else v) for k, v in batch.items()}
        d_rows = (model(**rows)["member"] - base).abs().max().item()
        p = batch["x"].shape[2]
        cp = torch.randperm(p)
        cols = {}
        for k, v in batch.items():
            if k in ("x", "codes", "freq"):
                cols[k] = v[:, :, cp]
            elif k in ("is_cat", "col_mask", "col_feat"):
                cols[k] = v[:, cp]
            elif k == "pair_feat":
                cols[k] = v[:, cp][:, :, cp]
        out = model(**cols)["member"]
        d_cols = (out - base[:, cp][:, :, cp]).abs().max().item()
    return dict(row_invariance=d_rows, column_equivariance=d_cols,
                pass_rows=d_rows < atol, pass_cols=d_cols < atol)
