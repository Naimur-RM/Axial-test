"""
Pilot round 3: the two remaining issues from round 2.

1. Residual Type-I inflation on descendants (shifted parents -> poor overlap
   between environments). Compare clipping e_hat at [0.02, 0.98] (rounds 1-2)
   with overlap trimming: test only rows with e_hat in [0.05, 0.95].
   Trimming is valid because the row selection depends on X_S only.
2. Bad proposer misses a shifted parent. Instead of max with a full greedy
   search (too conservative), "extend": for each proposed set c also test
   c + {k} for the m variables most correlated with X_j, and take the max.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time, collections
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import Ridge
from pilot import make_world, sample, descendants, noisy_parents, _basis
from pilot2 import propensity, _gbm

LO, HI = 0.05, 0.95


def crt_given(E, xj, XS, e_hat, B, rng, mode, folds=2):
    n = len(E)
    if XS.shape[1] == 0:
        m_hat = np.full(n, xj.mean())
    else:
        m_hat = np.zeros(n)
        parts = np.array_split(rng.permutation(n), folds)
        Bm = _basis(XS, True)
        for f in range(folds):
            te = parts[f]
            tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            m_hat[te] = Ridge(alpha=1.0).fit(Bm[tr], xj[tr]).predict(Bm[te])
    if mode == "clip":
        keep = np.ones(n, bool)
        e_hat = np.clip(e_hat, 0.02, 0.98)
    else:
        keep = (e_hat > LO) & (e_hat < HI)
        if keep.sum() < 60:
            return 1.0, 0.0
    E, e_hat, r, XS = E[keep], e_hat[keep], (xj - m_hat)[keep], XS[keep]
    feats = [r, r ** 2 - np.mean(r ** 2)] + [r * XS[:, s] for s in range(XS.shape[1])]
    Fm = np.column_stack(feats)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-12)
    v = e_hat * (1 - e_hat)
    den = np.sqrt((v[:, None] * Fm ** 2).sum(0)) + 1e-12
    t_obs = np.max(np.abs((E - e_hat) @ Fm) / den)
    Es = (rng.random((B, len(E))) < e_hat[None, :]).astype(float)
    t_null = np.max(np.abs((Es - e_hat[None, :]) @ Fm) / den[None, :], axis=1)
    return (1 + np.sum(t_null >= t_obs)) / (B + 1), keep.mean()


def crt(E, X, j, S, B, rng, mode="trim"):
    XS = X[:, list(S)]
    return crt_given(E, X[:, j], XS, propensity(E, XS, rng, "gbm"), B, rng, mode)[0]


def corr_pool(X, j, m):
    others = [k for k in range(X.shape[1]) if k != j]
    c = np.abs([np.corrcoef(X[:, k], X[:, j])[0, 1] for k in others])
    return [others[i] for i in np.argsort(-c)[:m]]


def proposer_rule(E, X, j, cands, B, rng, extend_m=0):
    sets = {tuple(sorted(c)) for c in cands}
    if extend_m:
        pool = corr_pool(X, j, extend_m)
        for c in list(sets):
            for k in pool:
                if k not in c:
                    sets.add(tuple(sorted(c + (k,))))
    ps = [crt(E, X, j, list(s), B, rng) for s in sets]
    return max(ps), len(ps)


def greedy(E, X, j, B, rng, alpha, depth=3, pool_m=6):
    pool = corr_pool(X, j, pool_m)
    S, n_t = [], 1
    p_max = crt(E, X, j, [], B, rng)
    while p_max <= alpha and len(S) < depth:
        best, best_p = None, -1
        for k in pool:
            if k in S:
                continue
            pv = crt(E, X, j, S + [k], B, rng)
            n_t += 1
            if pv > best_p:
                best, best_p = k, pv
        if best is None:
            break
        S.append(best)
        p_max = max(p_max, best_p)
    return p_max, n_t


def run_one(args):
    rep, p, n, B, alpha = args
    rng = np.random.default_rng(9000 + rep * 31 + n)
    world = make_world(p, rng)
    X = np.vstack([sample(world, n, 0, rng), sample(world, n, 1, rng)])
    E = np.r_[np.zeros(n), np.ones(n)]
    Nb = 20000
    Xb = np.vstack([sample(world, Nb, 0, rng), sample(world, Nb, 1, rng)])
    Eb = np.r_[np.zeros(Nb), np.ones(Nb)]
    T = set(world["T"])
    D = descendants(world["W"], T) - T
    rows = []
    for j in range(p):
        cat = "target" if j in T else ("descendant" if j in D else "other")
        pa = [int(a) for a in np.flatnonzero(world["W"][:, j])]
        res = {}
        XS = X[:, pa]
        e_gbm = propensity(E, XS, rng, "gbm")
        e_big = _gbm().fit(Xb[:, pa], Eb).predict_proba(XS)[:, 1] if pa else np.full(2 * n, 0.5)
        res["oracle_pa/gbm/clip"], _ = crt_given(E, X[:, j], XS, e_gbm, B, rng, "clip")
        res["oracle_pa/gbm/trim"], kept = crt_given(E, X[:, j], XS, e_gbm, B, rng, "trim")
        res["oracle_pa/big/clip"], _ = crt_given(E, X[:, j], XS, e_big, B, rng, "clip")
        res["oracle_pa/big/trim"], _ = crt_given(E, X[:, j], XS, e_big, B, rng, "trim")
        res["kept_frac"] = kept
        for tag, miss, nf in (("good", 0.1, 1), ("bad", 0.4, 2)):
            cands = [tuple(noisy_parents(world["W"], j, rng, miss, nf, p)) for _ in range(3)]
            res[f"proposer_{tag}"], res[f"proposer_{tag}/n_tests"] = proposer_rule(E, X, j, cands, B, rng)
            res[f"proposer_{tag}+extend"], res[f"proposer_{tag}+extend/n_tests"] = \
                proposer_rule(E, X, j, cands, B, rng, extend_m=3)
        res["greedy"], res["greedy/n_tests"] = greedy(E, X, j, B, rng, alpha)
        rows.append({"rep": rep, "n": n, "j": j, "cat": cat,
                     "kind": world["kind"].get(j, ""), **res})
    return rows


RULES = ["oracle_pa/gbm/clip", "oracle_pa/gbm/trim", "oracle_pa/big/clip", "oracle_pa/big/trim",
         "proposer_good", "proposer_good+extend", "proposer_bad", "proposer_bad+extend", "greedy"]


def summarize(rows, alpha):
    out = {}
    for key in ("cat", "kind"):
        g = collections.defaultdict(list)
        for r in rows:
            if key == "kind" and r["cat"] != "target":
                continue
            g[(r["n"], r[key])].append(r)
        for (n, c), rs in sorted(g.items()):
            name = f"n={n} {'target/' if key == 'kind' else ''}{c} (count={len(rs)})"
            out[name] = {k: round(float(np.mean([x[k] <= alpha for x in rs])), 3) for k in RULES}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        out[f"n={n} mean tests per variable"] = {
            k: round(float(np.mean([r[k + "/n_tests"] for r in rs])), 2)
            for k in ("proposer_good", "proposer_good+extend", "proposer_bad",
                      "proposer_bad+extend", "greedy")}
        out[f"n={n} mean kept fraction after trimming"] = round(float(np.mean([r["kept_frac"] for r in rs])), 3)
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    p, B, alpha = 12, 500, 0.05
    jobs = [(r, p, n, B, alpha) for n in (300, 1000) for r in range(reps)]
    t = time.time()
    with Pool(11) as pool:
        rows = sum(pool.map(run_one, jobs), [])
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "rows3.json"), "w"))
    summ = summarize(rows, alpha)
    json.dump(summ, open(os.path.join(here, "summary3.json"), "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s")
    for k, v in summ.items():
        print(k, v)
