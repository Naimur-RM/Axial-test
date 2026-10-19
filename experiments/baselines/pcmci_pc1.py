"""PC1, the parent-selection stage of PCMCI.

PC1 removes lagged links by partial correlation, conditioning on the strongest
remaining parents and growing the conditioning set. Contemporaneous links are
not in this stage; PCMCI tests those later with MCI. The comparison here is
against PC1 only, on the lagged parent set.

If tigramite is installed, its ``run_pc_stable`` is used. Otherwise this file
runs the same removal rule with a Fisher-z partial correlation.
"""

from __future__ import annotations

import math

import numpy as np


def _align(series, index, lag, start):
    # value of variable `index` at time t - lag, for t = start .. T-1
    t = series.shape[0]
    return series[start - lag: t - lag, index]


def _partial_corr_pvalue(target, candidate, conds):
    if conds:
        a = np.column_stack([np.ones(len(target))] + conds)
        target = target - a @ np.linalg.lstsq(a, target, rcond=None)[0]
        candidate = candidate - a @ np.linalg.lstsq(a, candidate, rcond=None)[0]
    else:
        target = target - target.mean()
        candidate = candidate - candidate.mean()
    nt, nc = np.linalg.norm(target), np.linalg.norm(candidate)
    if nt < 1e-12 or nc < 1e-12:
        return 0.0, 1.0
    r = float(np.clip(target @ candidate / (nt * nc), -0.999999, 0.999999))
    dof = len(target) - len(conds) - 3
    if dof <= 0:
        return r, 1.0
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return r, p


def pc1_parents(series, tau_max, alpha=0.05, max_cond_dim=5):
    """Return ``parents[j] = {(i, lag)}`` for lag in 1..tau_max, including own lags."""
    series = np.asarray(series, dtype=np.float64)
    t, d = series.shape
    start = tau_max
    if t - start < 20:
        raise ValueError(f"series length {t} is too short for tau_max={tau_max}")
    parents = {
        j: [(i, lag) for lag in range(1, tau_max + 1) for i in range(d)]
        for j in range(d)
    }
    strength = {j: {link: 0.0 for link in parents[j]} for j in range(d)}
    max_dim = min(max_cond_dim, d * tau_max - 1)
    for cond_dim in range(0, max_dim + 1):
        removals = {j: [] for j in range(d)}
        for j in range(d):
            ordered = sorted(parents[j], key=lambda link: -abs(strength[j].get(link, 0.0)))
            y = _align(series, j, 0, start)
            for link in ordered:
                others = [other for other in ordered if other != link]
                if len(others) < cond_dim:
                    continue
                cond_links = others[:cond_dim]
                conds = [_align(series, i, lag, start) for i, lag in cond_links]
                i, lag = link
                val, pval = _partial_corr_pvalue(y, _align(series, i, lag, start), conds)
                strength[j][link] = val
                if pval > alpha:
                    removals[j].append(link)
        for j in range(d):
            for link in removals[j]:
                if link in parents[j]:
                    parents[j].remove(link)
                    strength[j].pop(link, None)
    return {j: set(links) for j, links in parents.items()}


def tigramite_pc1(series, tau_max, alpha=0.05):
    """Published PC1, when tigramite is installed."""
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    frame = pp.DataFrame(np.asarray(series, dtype=np.float64))
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    parents = pcmci.run_pc_stable(tau_max=tau_max, pc_alpha=alpha)
    # tigramite stores parents[j] as a list of (i, -lag)
    out = {}
    for j, links in parents.items():
        out[j] = {(int(i), int(-lag)) for i, lag in links if lag < 0}
    return out


def pc1(series, tau_max, alpha=0.05, max_cond_dim=5):
    try:
        import tigramite  # noqa: F401
        found = tigramite_pc1(series, tau_max=tau_max, alpha=alpha)
        return found, {"implementation": "tigramite.run_pc_stable", "alpha": alpha, "tau_max": tau_max}
    except Exception:
        found = pc1_parents(series, tau_max=tau_max, alpha=alpha, max_cond_dim=max_cond_dim)
        return found, {"implementation": "fisherz_pc1", "alpha": alpha, "tau_max": tau_max}
