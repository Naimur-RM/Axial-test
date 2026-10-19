"""Turn worlds (or a real table) into padded tensors for the network."""

from __future__ import annotations

import numpy as np
import torch

from .config import MAX_STATES
from .stats import N_COL, N_PAIR, category_frequency, pair_and_column_features


def relabel_codes(codes, is_cat, n_states, rng):
    """Random permutation of category labels per discrete column."""
    out = codes.copy()
    for j in np.flatnonzero(is_cat):
        k = int(max(n_states[j], codes[:, j].max() + 1))
        k = min(k, MAX_STATES)
        perm = rng.permutation(k)
        out[:, j] = perm[np.clip(codes[:, j], 0, k - 1)]
    return out


def table_inputs(x, codes, is_cat, n_states):
    """Network inputs for one matrix, before padding.

    Discrete columns are re-standardized from ``codes`` so that the correlation
    features follow the same (possibly relabeled) category codes the network sees.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    is_cat = np.asarray(is_cat, bool)
    for j in np.flatnonzero(is_cat):
        c = codes[:, j].astype(np.float64)
        x[:, j] = (c - c.mean()) / (c.std() + 1e-8)
    pair, col = pair_and_column_features(x, codes, is_cat, n_states)
    freq = category_frequency(codes, is_cat)
    return dict(x=np.asarray(x, np.float32), codes=codes, freq=freq,
                is_cat=np.asarray(is_cat, bool), col_feat=col, pair_feat=pair)


def collate(items, device, labels=None):
    """Pad to the largest ``p`` in the batch. All items share ``N``."""
    b = len(items)
    n = items[0]["x"].shape[0]
    p = max(it["x"].shape[1] for it in items)
    x = np.zeros((b, n, p), np.float32)
    codes = np.zeros((b, n, p), np.int64)
    freq = np.zeros((b, n, p), np.float32)
    is_cat = np.zeros((b, p), bool)
    col_mask = np.zeros((b, p), bool)
    col_feat = np.zeros((b, p, N_COL), np.float32)
    pair_feat = np.zeros((b, p, p, N_PAIR), np.float32)
    for i, it in enumerate(items):
        q = it["x"].shape[1]
        x[i, :, :q] = it["x"]
        codes[i, :, :q] = it["codes"]
        freq[i, :, :q] = it["freq"]
        is_cat[i, :q] = it["is_cat"]
        col_mask[i, :q] = True
        col_feat[i, :q] = it["col_feat"]
        pair_feat[i, :q, :q] = it["pair_feat"]
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    batch = dict(
        x=t(x, torch.float32), codes=t(codes, torch.long), freq=t(freq, torch.float32),
        is_cat=t(is_cat, torch.bool), col_feat=t(col_feat, torch.float32),
        pair_feat=t(pair_feat, torch.float32), col_mask=t(col_mask, torch.bool),
    )
    if labels is None:
        return batch
    member = np.zeros((b, p, p), np.float32)
    role = np.zeros((b, p, p), np.int64)
    pair_mask = np.zeros((b, p, p), bool)
    count = np.zeros((b, p), np.int64)
    count_mask = np.zeros((b, p), bool)
    for i, lab in enumerate(labels):
        q = lab["member"].shape[0]
        member[i, :q, :q] = lab["member"]
        role[i, :q, :q] = lab["role"]
        pair_mask[i, :q, :q] = lab["pair_mask"]
        count[i, :q] = lab["count"]
        count_mask[i, :q] = lab["count_mask"]
    target = dict(
        member=t(member, torch.float32), role=t(role, torch.long),
        pair_mask=t(pair_mask, torch.bool), count=t(count, torch.long),
        count_mask=t(count_mask, torch.bool),
    )
    return batch, target


def world_item(world, rng=None):
    codes = world["codes"]
    if rng is not None:
        codes = relabel_codes(codes, world["is_cat"], world["n_states"], rng)
    return table_inputs(world["X"], codes, world["is_cat"], world["n_states"])
