"""IAMB, the classical grow-shrink Markov boundary algorithm.

Continuous columns use Fisher-z partial correlation. Discrete columns use a
conditional G-test. Both return the boundary set. Neither returns an orientation.
"""

from __future__ import annotations

import math

import numpy as np


def _is_discrete(x, y, max_card=20) -> bool:
    cols = [y] + [x[:, j] for j in range(x.shape[1])]
    return all(len(np.unique(col)) <= max_card for col in cols)


def _fisher_z(r, n, k):
    r = float(np.clip(r, -0.999999, 0.999999))
    dof = n - k - 3
    if dof <= 0:
        return 1.0
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _partial_corr(x, y, j, cond):
    xj = x[:, j]
    if cond:
        a = np.column_stack([np.ones(len(y)), x[:, cond]])
        xj = xj - a @ np.linalg.lstsq(a, xj, rcond=None)[0]
        yr = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    else:
        xj, yr = xj - xj.mean(), y - y.mean()
    return float(xj @ yr) / (np.linalg.norm(xj) * np.linalg.norm(yr) + 1e-12)


def _g_stat(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a_levels, a_inv = np.unique(a, return_inverse=True)
    b_levels, b_inv = np.unique(b, return_inverse=True)
    table = np.zeros((len(a_levels), len(b_levels)))
    np.add.at(table, (a_inv, b_inv), 1)
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    mask = table > 0
    g = float(2 * np.sum(table[mask] * np.log(table[mask] / np.maximum(expected[mask], 1e-12))))
    df = (len(a_levels) - 1) * (len(b_levels) - 1)
    return g, df


def _g_pvalue(x, y, j, cond):
    from scipy.stats import chi2

    xj = x[:, j]
    if not cond:
        g, df = _g_stat(xj, y)
    else:
        keys = np.ascontiguousarray(x[:, cond])
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        g, df = 0.0, 0
        for stratum in np.unique(inverse):
            mask = inverse == stratum
            if int(mask.sum()) < 8:
                continue
            gs, dfs = _g_stat(xj[mask], y[mask])
            g += gs
            df += dfs
    if df <= 0:
        return 1.0
    return float(chi2.sf(g, df))


def iamb(x, y, alpha=0.05, max_cond=8):
    """Incremental association Markov boundary of ``y`` inside ``x``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = x.shape
    discrete = _is_discrete(x, y)
    mb: list[int] = []
    changed = True
    while changed:
        changed = False
        best_j, best_p = None, alpha
        for j in range(p):
            if j in mb:
                continue
            cond = mb[:max_cond]
            if discrete:
                pval = _g_pvalue(x, y, j, cond)
            else:
                pval = _fisher_z(_partial_corr(x, y, j, cond), n, len(cond))
            if pval < best_p:
                best_p, best_j = pval, j
        if best_j is not None:
            mb.append(best_j)
            changed = True
    for j in list(mb):
        cond = [z for z in mb if z != j][:max_cond]
        if discrete:
            pval = _g_pvalue(x, y, j, cond)
        else:
            pval = _fisher_z(_partial_corr(x, y, j, cond), n, len(cond))
        if pval >= alpha:
            mb.remove(j)
    return sorted(mb)
