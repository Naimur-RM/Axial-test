"""Synthetic prior over whole graphs.

A world is an observational sample from a DAG over ``p`` observed variables.
The label is the moral graph: ``member[i, j] = 1`` when ``j`` is in the Markov
boundary of ``i`` (parent, child, or co-parent of a child). Hidden confounders
are eliminated from the moral graph by connecting their neighbors, which is the
boundary of the observed marginal.

Four kinds of world:

- ``continuous``: the mechanisms in ``MECHANISMS`` plus random-feature GPs.
- ``discrete``: Dirichlet conditional tables with 2 to 5 states, and
  noisy or near-deterministic logic nodes (OR, AND, XOR, noisy-OR).
- ``mixed``: each node is discrete or continuous at random.
- ``lagged``: a vector autoregression laid out as ``[current, lags]``. Only
  pairs that involve a current value are labeled, and a current value's
  boundary is its set of lagged parents.
"""

from __future__ import annotations

import numpy as np

from .config import MAX_FEATURES, MAX_MB, MAX_STATES

FAMILIES = [
    "erdos_renyi", "scale_free", "small_world", "modular", "layered",
    "tree", "lattice", "collider_hub", "latent_confounder",
]
MECHANISMS = [
    "linear_gaussian", "linear_nongauss", "mlp", "poly", "sigmoid",
    "threshold", "multiplicative", "heteroscedastic", "heavy_tail", "gp",
]
KINDS = ["continuous", "discrete", "mixed", "lagged"]
LOGIC = ["or", "and", "xor", "noisy_or"]


def _parents_by_family(p, family, rng):
    """Parent lists over a topological order 0..p-1. Edges always go i -> j with i < j."""
    pa = [[] for _ in range(p)]
    if family == "erdos_renyi":
        rho = rng.uniform(1.0, 3.0) / max(p - 1, 1)
        for j in range(1, p):
            cand = np.arange(j)
            sel = cand[rng.random(j) < rho]
            pa[j] = list(sel[-6:])
    elif family == "scale_free":
        m = rng.integers(1, 4)
        outdeg = np.ones(p)
        for j in range(1, p):
            k = min(int(m), j)
            w = outdeg[:j] / outdeg[:j].sum()
            pa[j] = list(rng.choice(j, size=k, replace=False, p=w))
            outdeg[pa[j]] += 1
    elif family == "small_world":
        k = int(rng.integers(1, 4))
        for j in range(1, p):
            base = [i for i in range(max(0, j - k), j)]
            if rng.random() < 0.15 and j > 4:
                base = base[:-1] + [int(rng.integers(0, j))]
            pa[j] = list(dict.fromkeys(base))
    elif family == "modular":
        nb = int(rng.integers(2, 6))
        blk = rng.integers(0, nb, p)
        for j in range(1, p):
            for i in range(j):
                pr = 0.25 if blk[i] == blk[j] else 0.01
                if rng.random() < pr:
                    pa[j].append(i)
            pa[j] = pa[j][-6:]
    elif family == "layered":
        nl = int(rng.integers(2, 6))
        lay = np.sort(rng.integers(0, nl, p))
        for j in range(p):
            prev = np.where(lay == lay[j] - 1)[0]
            if len(prev):
                k = min(int(rng.integers(1, 4)), len(prev))
                pa[j] = list(rng.choice(prev, size=k, replace=False))
    elif family == "tree":
        for j in range(1, p):
            pa[j] = [int(rng.integers(0, j))]
    elif family == "lattice":
        side = int(np.ceil(np.sqrt(p)))
        for j in range(p):
            r, c = divmod(j, side)
            for i in ([j - 1] if c > 0 else []) + ([j - side] if r > 0 else []):
                if 0 <= i < j:
                    pa[j].append(i)
    elif family == "collider_hub":
        lo = max(1, p // 3)
        hubs = set(rng.choice(np.arange(lo, p), size=min(p - lo, max(2, p // 8)), replace=False).tolist())
        for j in range(1, p):
            if j in hubs:
                k = min(int(rng.integers(2, 6)), j)
                pa[j] = list(rng.choice(j, size=k, replace=False))
            elif rng.random() < 0.3:
                pa[j] = [int(rng.integers(0, j))]
    elif family == "latent_confounder":
        rho = rng.uniform(1.0, 2.5) / max(p - 1, 1)
        for j in range(1, p):
            cand = np.arange(j)
            sel = cand[rng.random(j) < rho]
            pa[j] = list(sel[-5:])
    else:
        raise ValueError(family)
    return [[int(i) for i in parents] for parents in pa]


def _noise(mech, rng, n, scale):
    if mech == "linear_nongauss":
        if rng.random() < 0.5:
            return rng.laplace(0, scale, n)
        return rng.uniform(-1.7 * scale, 1.7 * scale, n)
    if mech == "heavy_tail":
        return rng.standard_t(3, n) * scale * 0.6
    return rng.normal(0, scale, n)


def _apply_mech(vals, mech, rng, noise_s):
    n, k = vals.shape
    if k == 0:
        return _noise(mech, rng, n, 1.0)
    w = rng.uniform(0.6, 1.5, k) * rng.choice([-1.0, 1.0], k)
    lin = vals @ w
    if mech in ("linear_gaussian", "linear_nongauss", "heavy_tail"):
        out = lin
    elif mech == "mlp":
        h = int(rng.integers(3, 7))
        w1 = rng.normal(0, 1.2, (k, h))
        w2 = rng.normal(0, 1.2, h)
        out = np.tanh(vals @ w1) @ w2
    elif mech == "poly":
        w2 = rng.uniform(0.2, 0.7, k) * rng.choice([-1.0, 1.0], k)
        out = lin + (vals ** 2) @ w2
    elif mech == "sigmoid":
        out = 2.0 * np.tanh(lin)
    elif mech == "threshold":
        out = lin + 1.5 * np.sign(lin) * (np.abs(lin) > 0.8)
    elif mech == "multiplicative":
        extra = vals[:, 0] * vals[:, -1] if k > 1 else vals[:, 0] ** 2
        out = lin + extra * rng.uniform(0.4, 0.9)
    elif mech == "heteroscedastic":
        s = 0.4 + 0.8 * np.abs(np.tanh(lin))
        return lin + rng.normal(0, 1, n) * s * noise_s
    elif mech == "gp":
        # Additive random-feature approximation of a GP with an RBF kernel.
        out = np.zeros(n)
        for c in range(k):
            length = rng.uniform(0.5, 2.0)
            omega = rng.normal(0, 1.0 / length, 40)
            phase = rng.uniform(0, 2 * np.pi, 40)
            amp = rng.normal(0, 1.0, 40) * np.sqrt(2.0 / 40)
            out += np.cos(np.outer(vals[:, c], omega) + phase) @ amp
    else:
        out = lin
    sd = out.std()
    if sd > 1e-8:
        out = out / sd
    return out + _noise(mech, rng, n, noise_s)


def _quantile_bins(v, k=3):
    edges = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, v, side="right")


def _discrete_inputs(data, parents, is_cat):
    cols = []
    for i in parents:
        cols.append(data[:, i].astype(np.int64) if is_cat[i] else _quantile_bins(data[:, i]))
    return np.column_stack(cols) if cols else np.zeros((data.shape[0], 0), dtype=np.int64)


def _sample_cpt(configs, card, rng, alpha):
    n = configs.shape[0]
    if configs.shape[1] == 0:
        prob = rng.dirichlet(np.full(card, alpha))
        return rng.choice(card, size=n, p=prob)
    _, inverse = np.unique(configs, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    table = rng.dirichlet(np.full(card, alpha), size=int(inverse.max()) + 1)
    cdf = np.cumsum(table[inverse], axis=1)
    draw = rng.random(n)
    return np.minimum((draw[:, None] > cdf).sum(axis=1), card - 1)


def _sample_logic(configs, rng, kind, flip):
    n = configs.shape[0]
    active = (configs > 0).astype(np.int64)
    if kind == "or":
        out = active.max(axis=1)
    elif kind == "and":
        out = active.min(axis=1)
    elif kind == "xor":
        out = active.sum(axis=1) % 2
    else:
        inhibit = rng.uniform(0.05, 0.4, active.shape[1])
        p_off = np.prod(np.where(active == 1, inhibit, 1.0), axis=1)
        leak = rng.uniform(0.0, 0.05)
        out = (rng.random(n) > p_off * (1 - leak)).astype(np.int64)
    flips = rng.random(n) < flip
    return np.where(flips, 1 - out, out)


def _continuous_inputs(data, parents, is_cat, rng):
    """Numeric parent inputs. A discrete parent enters through random per-state effects."""
    cols = []
    for i in parents:
        if is_cat[i]:
            levels = int(data[:, i].max()) + 1
            effect = rng.normal(0, 1.0, levels)
            cols.append(effect[data[:, i].astype(np.int64)])
        else:
            cols.append(data[:, i])
    return np.column_stack(cols) if cols else np.zeros((data.shape[0], 0))


def _moral_graph(pa, p):
    member = np.zeros((p, p), dtype=bool)
    adjacent = np.zeros((p, p), dtype=bool)
    for j in range(p):
        for i in pa[j]:
            member[i, j] = member[j, i] = True
            adjacent[i, j] = adjacent[j, i] = True
        for a in pa[j]:
            for b in pa[j]:
                if a != b:
                    member[a, b] = True
    np.fill_diagonal(member, False)
    return member, adjacent


def _eliminate(member, hidden):
    """Marginalize hidden nodes from the moral graph: their neighbors become a clique."""
    member = member.copy()
    for h in hidden:
        nbrs = np.flatnonzero(member[h])
        for a in nbrs:
            member[a, nbrs] = True
        member[h, :] = False
        member[:, h] = False
    np.fill_diagonal(member, False)
    return member


def _roles(member_obs, pa, keep):
    pos = {g: i for i, g in enumerate(keep)}
    out = []
    for g in keep:
        parents = {pos[i] for i in pa[g] if i in pos}
        children = {pos[j] for j in keep if g in pa[j]}
        boundary = set(np.flatnonzero(member_obs[pos[g]]).tolist())
        out.append(dict(
            parents=parents,
            children=children,
            spouses=boundary - parents - children,
            boundary=boundary,
        ))
    return out


def _labels(member_obs, adjacent_obs, mask=None, count_mask=None):
    p = member_obs.shape[0]
    role = np.where(member_obs, np.where(adjacent_obs, 1, 2), 0).astype(np.int64)
    pair_mask = np.ones((p, p), dtype=bool) if mask is None else mask
    np.fill_diagonal(pair_mask, False)
    count = np.minimum(member_obs.sum(1), MAX_MB).astype(np.int64)
    cmask = np.ones(p, dtype=bool) if count_mask is None else count_mask
    return dict(
        member=member_obs.astype(np.int64),
        role=role,
        pair_mask=pair_mask,
        count=count,
        count_mask=cmask,
    )


def _finish(values, is_cat, n_states):
    values = np.asarray(values, dtype=np.float64)
    x = (values - values.mean(0)) / (values.std(0) + 1e-8)
    codes = np.where(is_cat[None, :], np.clip(values, 0, MAX_STATES - 1), 0).astype(np.int64)
    return x.astype(np.float32), codes, is_cat.astype(bool), n_states.astype(np.int64)


def graph_world(rng, p, n, kind, family=None, mech=None, discrete_style="any"):
    """One DAG world of ``p`` observed variables and ``n`` rows."""
    if p > MAX_FEATURES:
        raise ValueError(f"p={p} exceeds MAX_FEATURES={MAX_FEATURES}")
    family = family or FAMILIES[int(rng.integers(len(FAMILIES)))]
    n_hidden = int(rng.integers(2, 7)) if family == "latent_confounder" else 0
    p_gen = p + n_hidden
    pa = _parents_by_family(p_gen, family, rng)

    if kind == "continuous":
        cat = np.zeros(p_gen, dtype=bool)
    elif kind == "discrete":
        cat = np.ones(p_gen, dtype=bool)
    else:
        cat = rng.random(p_gen) < 0.5
    card = np.where(cat, rng.choice([2, 2, 3, 3, 4, 5], size=p_gen), 0)
    use_logic = np.zeros(p_gen, dtype=bool)
    if discrete_style == "any":
        use_logic = cat & (np.array([len(x) for x in pa]) >= 2) & (rng.random(p_gen) < 0.2)
    card[use_logic] = 2
    mech = mech or MECHANISMS[int(rng.integers(len(MECHANISMS)))]
    noise_s = float(rng.uniform(0.3, 1.5))
    alpha = float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))

    data = np.zeros((n, p_gen))
    for j in range(p_gen):
        if cat[j]:
            configs = _discrete_inputs(data, pa[j], cat)
            if use_logic[j]:
                logic = LOGIC[int(rng.integers(len(LOGIC)))]
                flip = float(rng.choice([0.0, 0.01, 0.03, 0.05]))
                data[:, j] = _sample_logic(configs, rng, logic, flip)
            else:
                data[:, j] = _sample_cpt(configs, int(card[j]), rng, alpha)
        else:
            vals = _continuous_inputs(data, pa[j], cat, rng)
            out = _apply_mech(vals, mech, rng, noise_s)
            data[:, j] = out / (out.std() + 1e-8)

    member, adjacent = _moral_graph(pa, p_gen)
    hidden = []
    if n_hidden:
        degree = member.sum(1)
        eligible = np.flatnonzero((degree >= 2) & (degree <= 4))
        if len(eligible) < n_hidden:
            eligible = np.flatnonzero(degree >= 1)
        take = min(n_hidden, len(eligible))
        hidden = [int(h) for h in rng.choice(eligible, size=take, replace=False)] if take else []
        if len(hidden) < n_hidden:
            spare = [j for j in range(p_gen) if j not in set(hidden)]
            hidden += [int(h) for h in rng.choice(spare, size=n_hidden - len(hidden), replace=False)]
        member = _eliminate(member, hidden)
    keep = [j for j in range(p_gen) if j not in set(hidden)]
    member_obs = member[np.ix_(keep, keep)]
    adjacent_obs = adjacent[np.ix_(keep, keep)] & member_obs

    x, codes, is_cat, n_states = _finish(data[:, keep], cat[keep], card[keep])
    out = dict(
        X=x, codes=codes, is_cat=is_cat, n_states=n_states,
        kind=kind, family=family, mech=mech if kind != "discrete" else "dirichlet_cpt",
        roles=_roles(member_obs, pa, keep),
    )
    out.update(_labels(member_obs, adjacent_obs))
    return out


def lagged_world(rng, n, d=None, lags=None, p_max=MAX_FEATURES, nonlinear=None):
    """A VAR laid out as ``[current_0..current_{d-1}, lag columns]``.

    Lag column ``d + j * L + (l - 1)`` holds series ``j`` at lag ``l``.
    """
    while True:
        d = d or int(rng.integers(2, 9))
        lags = lags or int(rng.integers(1, 6))
        if d * (lags + 1) <= p_max:
            break
        d, lags = None, None
    nonlinear = bool(rng.random() < 0.5) if nonlinear is None else nonlinear
    density = rng.uniform(0.1, 0.4)
    coef = np.zeros((lags, d, d))
    parents = {i: set() for i in range(d)}
    for i in range(d):
        own = int(rng.integers(1, lags + 1))
        coef[own - 1, i, i] = rng.uniform(0.2, 0.6) * rng.choice([-1.0, 1.0])
        parents[i].add((i, own))
        for ell in range(lags):
            for j in range(d):
                if j != i and rng.random() < density / lags:
                    coef[ell, i, j] = rng.uniform(0.15, 0.5) * rng.choice([-1.0, 1.0])
                    parents[i].add((j, ell + 1))
    companion = np.zeros((d * lags, d * lags))
    companion[:d] = np.concatenate(list(coef), axis=1)
    if lags > 1:
        companion[d:, :-d] = np.eye(d * (lags - 1))
    radius = np.abs(np.linalg.eigvals(companion)).max()
    if radius > 0.9:
        coef *= 0.9 / radius
    burn = 200
    length = n + lags + burn
    series = np.zeros((length, d))
    laplace = rng.random() < 0.3
    for t in range(lags, length):
        acc = np.zeros(d)
        for ell in range(lags):
            z = series[t - ell - 1]
            acc += coef[ell] @ (np.tanh(z) if nonlinear else z)
        eps = rng.laplace(0, 0.7, d) if laplace else rng.normal(0, 1.0, d)
        series[t] = acc + eps
    series = series[burn:]
    design = lagged_matrix(series, lags)
    p = design.shape[1]
    member = np.zeros((p, p), dtype=bool)
    for i in range(d):
        for j, ell in parents[i]:
            c = d + j * lags + (ell - 1)
            member[i, c] = member[c, i] = True
    mask = np.zeros((p, p), dtype=bool)
    mask[:d, :] = True
    mask[:, :d] = True
    cmask = np.zeros(p, dtype=bool)
    cmask[:d] = True
    is_cat = np.zeros(p, dtype=bool)
    x, codes, is_cat, n_states = _finish(design, is_cat, np.zeros(p))
    roles = []
    for c in range(p):
        if c < d:
            pset = {d + j * lags + (ell - 1) for j, ell in parents[c]}
            roles.append(dict(parents=pset, children=set(), spouses=set(), boundary=set(pset)))
        else:
            roles.append(None)
    out = dict(
        X=x, codes=codes, is_cat=is_cat, n_states=n_states,
        kind="lagged", family="var", mech="tanh" if nonlinear else "linear",
        roles=roles, d=d, lags=lags, true_parents=parents,
        series=series[lags:].astype(np.float32),
    )
    out.update(_labels(member, member, mask=mask, count_mask=cmask))
    return out


def lagged_matrix(series, lags):
    """``[current, lag 1..L of each series]`` with the column order of ``lagged_world``."""
    series = np.asarray(series, dtype=np.float64)
    t, d = series.shape
    rows = t - lags
    out = np.zeros((rows, d * (lags + 1)))
    out[:, :d] = series[lags:]
    for j in range(d):
        for ell in range(1, lags + 1):
            out[:, d + j * lags + (ell - 1)] = series[lags - ell: t - ell, j]
    return out


def sample_world(rng, cfg, kind=None, families=None, p=None, n=None):
    kind = kind or rng.choice(KINDS, p=cfg["kind_probs"])
    n = n or int(rng.integers(cfg["N_min"], cfg["N_max"] + 1))
    if kind == "lagged":
        return lagged_world(rng, n, p_max=min(cfg["p_max"], MAX_FEATURES))
    p = p or int(rng.integers(cfg["p_min"], cfg["p_max"] + 1))
    family = None
    if families:
        family = families[int(rng.integers(len(families)))]
    return graph_world(rng, p, n, kind, family=family)
