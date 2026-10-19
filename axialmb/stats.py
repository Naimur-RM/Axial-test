"""Pairwise and per-column statistics fed to the network as inputs.

Pair features, all symmetric:

0. absolute Pearson correlation
1. absolute partial correlation from a Ledoit-Wolf precision matrix. For
   Gaussian data this is nonzero exactly on the Markov boundary.
2. normalized mutual information on binned values (categories for discrete
   columns, quantile bins for continuous ones)
3. absolute partial correlation of Gaussian-copula scores, which keeps
   monotone nonlinear dependence

Column features: discrete flag, number of states over ``MAX_STATES``, and
entropy of the binned column.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf

from .config import MAX_STATES

N_PAIR = 4
N_COL = 3
_BINS = MAX_STATES


def _partial_corr(z):
    try:
        prec = LedoitWolf(assume_centered=True).fit(z).precision_
    except Exception:
        return np.zeros((z.shape[1], z.shape[1]))
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return np.abs(pc)


def _binned(x, codes, is_cat):
    n, p = x.shape
    out = np.zeros((n, p), dtype=np.int64)
    for j in range(p):
        if is_cat[j]:
            out[:, j] = np.clip(codes[:, j], 0, _BINS - 1)
        else:
            edges = np.quantile(x[:, j], np.linspace(0, 1, 6)[1:-1])
            out[:, j] = np.searchsorted(edges, x[:, j], side="right")
    return out


def _mutual_information(binned):
    n, p = binned.shape
    onehot = np.zeros((n, p * _BINS), dtype=np.float32)
    rows = np.arange(n)
    for j in range(p):
        onehot[rows, j * _BINS + binned[:, j]] = 1.0
    joint = (onehot.T @ onehot).reshape(p, _BINS, p, _BINS).transpose(0, 2, 1, 3) / n
    marg = onehot.sum(0).reshape(p, _BINS) / n
    outer = marg[:, None, :, None] * marg[None, :, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log(joint / np.maximum(outer, 1e-12)), 0.0)
    mi = terms.sum(axis=(2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(marg > 0, marg * np.log(marg), 0.0).sum(1)
    denom = np.maximum(np.minimum(ent[:, None], ent[None, :]), 1e-6)
    nmi = np.clip(mi / denom, 0.0, 1.0)
    np.fill_diagonal(nmi, 0.0)
    return nmi, ent


def pair_and_column_features(x, codes, is_cat, n_states):
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape
    corr = np.abs((x.T @ x) / n)
    np.fill_diagonal(corr, 0.0)
    pcorr = _partial_corr(x)
    binned = _binned(x, codes, is_cat)
    nmi, ent = _mutual_information(binned)
    scores = ndtri((rankdata(x, axis=0) - 0.5) / n)
    scores = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    copula = _partial_corr(scores)
    pair = np.stack([corr, pcorr, nmi, copula], axis=-1)
    col = np.stack([
        np.asarray(is_cat, dtype=np.float64),
        np.asarray(n_states, dtype=np.float64) / MAX_STATES,
        ent / np.log(_BINS),
    ], axis=-1)
    pair = np.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    col = np.nan_to_num(col, nan=0.0).astype(np.float32)
    return pair, col


def category_frequency(codes, is_cat):
    """Frequency of each cell's category in its column. It does not depend on the labels."""
    n, p = codes.shape
    freq = np.zeros((n, p), dtype=np.float32)
    for j in range(p):
        if is_cat[j]:
            counts = np.bincount(codes[:, j], minlength=MAX_STATES).astype(np.float32) / n
            freq[:, j] = counts[codes[:, j]]
    return freq
