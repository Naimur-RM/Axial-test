"""Markov boundaries from one forward pass per view.

The network gives, for each variable ``i``, a membership probability for every
partner ``j`` and a distribution over the boundary size ``|MB(i)|``. The
returned boundary is the ``k_i`` most probable partners, with ``k_i`` the most
probable size. Nothing is tuned after training.

Test-time views average out sampling noise and category coding: view 0 is the
full table, the others use a random 80% of rows and a random relabeling of every
discrete column. The network is exactly equivariant to column order, so column
shuffles would add nothing and are not used.
"""

from __future__ import annotations

import numpy as np
import torch

from .batch import collate, relabel_codes, table_inputs
from .config import MAX_STATES


def infer_types(matrix, max_states=MAX_STATES):
    """A column is discrete when it holds at most ``max_states`` integer values."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n, p = matrix.shape
    is_cat = np.zeros(p, bool)
    codes = np.zeros((n, p), np.int64)
    n_states = np.zeros(p, np.int64)
    for j in range(p):
        col = matrix[:, j]
        levels = np.unique(col)
        if len(levels) <= max_states and np.allclose(col, np.round(col)):
            is_cat[j] = True
            codes[:, j] = np.searchsorted(levels, col)
            n_states[j] = len(levels)
    return codes, is_cat, n_states


def _views(x, codes, is_cat, n_states, n_views, frac, rng, n_cap):
    n = x.shape[0]
    for v in range(n_views):
        if v == 0 and n <= n_cap:
            rows = np.arange(n)
            c = codes
        else:
            size = min(n_cap, max(32, int(round(frac * n))))
            rows = np.sort(rng.choice(n, size=min(size, n), replace=False))
            c = relabel_codes(codes, is_cat, n_states, rng) if v > 0 else codes
        yield table_inputs(x[rows], c[rows], is_cat, n_states)


@torch.no_grad()
def predict_scores(model, x, codes=None, is_cat=None, n_states=None, device="cpu",
                   n_views=8, frac=0.8, seed=0, n_cap=2000):
    """Averaged membership probabilities (p, p) and size distributions (p, K+1)."""
    x = np.asarray(x, dtype=np.float64)
    if codes is None:
        codes, is_cat, n_states = infer_types(x)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    rng = np.random.default_rng(seed)
    model.eval()
    prob_sum, count_sum, role_sum = None, None, None
    for item in _views(x, codes, is_cat, n_states, n_views, frac, rng, n_cap):
        batch = collate([item], device)
        out = model(**batch)
        prob = torch.sigmoid(out["member"][0]).cpu().numpy()
        count = torch.softmax(out["count"][0], -1).cpu().numpy()
        role = torch.softmax(out["role"][0], -1).cpu().numpy()
        prob_sum = prob if prob_sum is None else prob_sum + prob
        count_sum = count if count_sum is None else count_sum + count
        role_sum = role if role_sum is None else role_sum + role
    prob = prob_sum / n_views
    np.fill_diagonal(prob, 0.0)
    return prob, count_sum / n_views, role_sum / n_views


def decode(prob, count_prob, rows=None, candidates=None):
    """``MB(i)`` = the ``argmax`` size of the most probable partners."""
    p = prob.shape[0]
    rows = range(p) if rows is None else rows
    out = {}
    for i in rows:
        cand = [j for j in (range(p) if candidates is None else candidates) if j != i]
        k = int(np.argmax(count_prob[i]))
        k = min(k, len(cand))
        order = sorted(cand, key=lambda j: -prob[i, j])
        out[int(i)] = sorted(int(j) for j in order[:k])
    return out


def predict_boundaries(model, x, codes=None, is_cat=None, n_states=None, device="cpu",
                       n_views=8, frac=0.8, seed=0, rows=None, candidates=None):
    prob, count_prob, role_prob = predict_scores(
        model, x, codes, is_cat, n_states, device=device, n_views=n_views, frac=frac, seed=seed,
    )
    return dict(
        boundaries=decode(prob, count_prob, rows=rows, candidates=candidates),
        prob=prob,
        count_prob=count_prob,
        role_prob=role_prob,
        size=count_prob.argmax(1),
    )
