"""
Pilot for mechanism-shift localization.

World: random DAG over p variables, two environments (E=0/1). A sparse set T
of nodes has its mechanism P(X_j | PA_j) changed in E=1 (mean, noise-scale, or
function shift). Descendants of T change marginally but NOT in mechanism.

Test for node j given a conditioning set S:
    H0: E _||_ X_j | X_S
Environment-side CRT: resample E* ~ Bernoulli(e_hat(X_S)), cross-fitted.
Statistic: max |z| over residual features of X_j (r, r^2, r*X_s).

Decision rules compared:
    marginal      S = {}                (what drift monitors do)
    full          S = all other vars    (domain classifier style)
    oracle_pa     S = true parents      (identifiable target)
    max3          max p over {marginal, full, oracle_pa}
    greedy        max p over sets visited by a greedy separating-set search
                  (classical, no learned proposer)
    proposer_*    max p over {marginal, noisy-parent proposals}
                  (stand-in for an amortized proposer with controlled errors)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression, Ridge

G = {
    "lin": lambda x: x,
    "tanh": lambda x: 1.3 * np.tanh(1.5 * x),
    "sin": lambda x: 1.2 * np.sin(1.5 * x),
    "sq": lambda x: (x ** 2 - 1) / np.sqrt(2),
}


# ------------------------------------------------------------------ world
def make_world(p, rng, deg=2.0, n_shift=2):
    W = np.zeros((p, p))
    F = {}
    for a in range(p):
        for b in range(a + 1, p):
            if rng.random() < deg / (p - 1):
                W[a, b] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
                F[(a, b)] = str(rng.choice(list(G)))
    has_child = [j for j in range(p) if np.any(W[j] != 0)]
    pool = has_child if len(has_child) >= n_shift else list(range(p))
    T = sorted(int(t) for t in rng.choice(pool, n_shift, replace=False))
    kind, W1 = {}, W.copy()
    delta, gamma = {}, {}
    for t in T:
        k = str(rng.choice(["mean", "scale", "func"]))
        pa = np.flatnonzero(W[:, t])
        if k == "func" and len(pa) == 0:
            k = "mean"
        kind[t] = k
        delta[t] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
        gamma[t] = rng.uniform(1.8, 2.5)
        if k == "func":
            for a in pa:
                W1[a, t] = rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
                if np.sign(W1[a, t]) == np.sign(W[a, t]):
                    W1[a, t] = -W1[a, t]
    world = dict(p=p, W=W, W1=W1, F=F, T=T, kind=kind, delta=delta, gamma=gamma,
                 sdn=rng.uniform(0.5, 1.0, p), mu0=np.zeros(p), s0=np.ones(p),
                 raw_sd0=np.ones(p))
    sample(world, 20000, 0, rng, fit_stats=True)
    return world


def sample(world, n, env, rng, fit_stats=False):
    p, W, W1, F = world["p"], world["W"], world["W1"], world["F"]
    X = np.zeros((n, p))
    for j in range(p):
        shifted = env == 1 and j in world["kind"]
        k = world["kind"].get(j)
        Wj = W1[:, j] if (shifted and k == "func") else W[:, j]
        pa = np.flatnonzero(W[:, j])
        u = np.zeros(n)
        for a in pa:
            u += Wj[a] * G[F[(a, j)]](X[:, a])
        sd = world["sdn"][j] * (world["gamma"][j] if shifted and k == "scale" else 1.0)
        v = u + sd * rng.standard_normal(n)
        if fit_stats:
            world["mu0"][j], world["s0"][j] = v.mean(), v.std()
        if shifted and k == "mean":
            v = v + world["delta"][j] * world["s0"][j]
        X[:, j] = (v - world["mu0"][j]) / world["s0"][j]
    return X


def descendants(W, T):
    p = W.shape[0]
    reach = set()
    stack = list(T)
    while stack:
        a = stack.pop()
        for b in np.flatnonzero(W[a] != 0):
            if b not in reach:
                reach.add(int(b))
                stack.append(int(b))
    return reach


# ------------------------------------------------------------------ test
def _basis(XS, cubic):
    n, s = XS.shape
    cols = [XS, XS ** 2]
    if cubic:
        cols.append(XS ** 3)
    if s <= 4:
        for a in range(s):
            for b in range(a + 1, s):
                cols.append((XS[:, a] * XS[:, b])[:, None])
    return np.column_stack(cols)


def env_crt(E, xj, XS, B, rng, folds=2):
    """p-value for H0: E _||_ X_j | X_S via environment-side CRT."""
    n = len(E)
    if XS.shape[1] == 0:
        e_hat = np.full(n, E.mean())
        m_hat = np.full(n, xj.mean())
    else:
        e_hat, m_hat = np.zeros(n), np.zeros(n)
        idx = rng.permutation(n)
        parts = np.array_split(idx, folds)
        Be, Bm = _basis(XS, False), _basis(XS, True)
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(Be[tr], E[tr])
            e_hat[te] = clf.predict_proba(Be[te])[:, 1]
            # m_hat uses X only (never E), so it does not affect validity
            m_hat[te] = Ridge(alpha=1.0).fit(Bm[tr], xj[tr]).predict(Bm[te])
    e_hat = np.clip(e_hat, 0.02, 0.98)
    r = xj - m_hat
    feats = [r, r ** 2 - np.mean(r ** 2)]
    for s in range(XS.shape[1]):
        feats.append(r * XS[:, s])
    Fm = np.column_stack(feats)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-12)
    v = e_hat * (1 - e_hat)
    den = np.sqrt((v[:, None] * Fm ** 2).sum(0)) + 1e-12
    t_obs = np.max(np.abs((E - e_hat) @ Fm) / den)
    Es = (rng.random((B, n)) < e_hat[None, :]).astype(float)
    t_null = np.max(np.abs((Es - e_hat[None, :]) @ Fm) / den[None, :], axis=1)
    return (1 + np.sum(t_null >= t_obs)) / (B + 1)


# ------------------------------------------------------------------ rules
def greedy_search(E, X, j, B, rng, alpha, depth=3, pool_m=6):
    """Classical: look for a separating set. Declared iff every visited set rejects."""
    p = X.shape[1]
    others = [k for k in range(p) if k != j]
    c = np.abs([np.corrcoef(X[:, k], X[:, j])[0, 1] for k in others])
    pool = [others[i] for i in np.argsort(-c)[:pool_m]]
    S, p_max, tests = [], env_crt(E, X[:, j], X[:, []], B, rng), 1
    while p_max <= alpha and len(S) < depth:
        best, best_p = None, -1
        for k in pool:
            if k in S:
                continue
            pv = env_crt(E, X[:, j], X[:, S + [k]], B, rng)
            tests += 1
            if pv > best_p:
                best, best_p = k, pv
        if best is None:
            break
        S.append(best)
        p_max = max(p_max, best_p)
    return p_max, tests


def noisy_parents(W, j, rng, miss, n_false, p):
    pa = list(np.flatnonzero(W[:, j]))
    keep = [a for a in pa if rng.random() > miss]
    non = [k for k in range(p) if k != j and k not in pa]
    fp = list(rng.choice(non, min(n_false, len(non)), replace=False)) if n_false else []
    return sorted(int(a) for a in keep + fp)


def run_one(args):
    rep, p, n, B, alpha = args
    rng = np.random.default_rng(1000 + rep * 31 + n)
    world = make_world(p, rng)
    X = np.vstack([sample(world, n, 0, rng), sample(world, n, 1, rng)])
    E = np.r_[np.zeros(n), np.ones(n)]
    T = set(world["T"])
    D = descendants(world["W"], T) - T
    rows = []
    for j in range(p):
        cat = "target" if j in T else ("descendant" if j in D else "other")
        pa = list(np.flatnonzero(world["W"][:, j]))
        rest = [k for k in range(p) if k != j]
        pm = env_crt(E, X[:, j], X[:, []], B, rng)
        pf = env_crt(E, X[:, j], X[:, rest], B, rng)
        po = env_crt(E, X[:, j], X[:, pa], B, rng) if pa else pm
        res = {"marginal": pm, "full": pf, "oracle_pa": po,
               "max3": max(pm, pf, po)}
        if pm <= alpha:  # otherwise every max-rule is already non-significant
            res["greedy"], ntest = greedy_search(E, X, j, B, rng, alpha)
            for tag, miss, nf in (("proposer_good", 0.1, 1), ("proposer_bad", 0.4, 2)):
                cands = [noisy_parents(world["W"], j, rng, miss, nf, p) for _ in range(3)]
                ps = [pm] + [env_crt(E, X[:, j], X[:, c], B, rng) for c in cands if c]
                res[tag] = max(ps)
        else:
            res["greedy"] = res["proposer_good"] = res["proposer_bad"] = pm
            ntest = 1
        rows.append({"rep": rep, "n": n, "j": j, "cat": cat,
                     "kind": world["kind"].get(j, ""), "n_pa": len(pa),
                     "greedy_tests": ntest, **res})
    return rows


def summarize(rows, alpha):
    import collections
    rules = ["marginal", "full", "oracle_pa", "max3", "greedy", "proposer_good", "proposer_bad"]
    out = {}
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["n"], r["cat"])].append(r)
    for (n, cat), rs in sorted(g.items()):
        out[f"n={n} {cat} (count={len(rs)})"] = {
            k: round(float(np.mean([x[k] <= alpha for x in rs])), 3) for k in rules}
    g2 = collections.defaultdict(list)
    for r in rows:
        if r["cat"] == "target":
            g2[(r["n"], r["kind"])].append(r)
    for (n, kind), rs in sorted(g2.items()):
        out[f"n={n} target/{kind} (count={len(rs)})"] = {
            k: round(float(np.mean([x[k] <= alpha for x in rs])), 3) for k in rules}
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    p, B, alpha = 12, 500, 0.05
    jobs = [(r, p, n, B, alpha) for n in (300, 1000) for r in range(reps)]
    t = time.time()
    with Pool(10) as pool:
        rows = sum(pool.map(run_one, jobs), [])
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "rows.json"), "w"))
    summ = summarize(rows, alpha)
    json.dump(summ, open(os.path.join(os.path.dirname(__file__), "summary.json"), "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s")
    for k, v in summ.items():
        print(k, v)
