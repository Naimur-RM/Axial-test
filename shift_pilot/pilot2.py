"""
Pilot round 2: fix the two failure modes found in round 1.

1. Validity: does the oracle-parent test inflate because the propensity model
   P(E | X_S) is misspecified (logistic-quadratic) or because of estimation
   noise? Compare: logit2 (round 1), gbm (cross-fitted boosting), and big
   (boosting fit on 20k fresh samples per environment, ~ the true propensity).
2. Power: drop the empty set from the candidate family unless the proposed
   parent set is empty, and do not gate on the marginal test.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time, collections
import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from pilot import make_world, sample, descendants, noisy_parents, _basis


def _gbm():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                          max_leaf_nodes=8, min_samples_leaf=20,
                                          early_stopping=False, random_state=0)


def propensity(E, XS, rng, model, folds=2):
    n = len(E)
    if XS.shape[1] == 0:
        return np.full(n, E.mean())
    e_hat = np.zeros(n)
    parts = np.array_split(rng.permutation(n), folds)
    Be = _basis(XS, False) if model == "logit2" else XS
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        clf = LogisticRegression(C=1.0, max_iter=1000) if model == "logit2" else _gbm()
        clf.fit(Be[tr], E[tr])
        e_hat[te] = clf.predict_proba(Be[te])[:, 1]
    return e_hat


def crt_given(E, xj, XS, e_hat, B, rng, folds=2):
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
    e_hat = np.clip(e_hat, 0.02, 0.98)
    r = xj - m_hat
    feats = [r, r ** 2 - np.mean(r ** 2)] + [r * XS[:, s] for s in range(XS.shape[1])]
    Fm = np.column_stack(feats)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-12)
    v = e_hat * (1 - e_hat)
    den = np.sqrt((v[:, None] * Fm ** 2).sum(0)) + 1e-12
    t_obs = np.max(np.abs((E - e_hat) @ Fm) / den)
    Es = (rng.random((B, n)) < e_hat[None, :]).astype(float)
    t_null = np.max(np.abs((Es - e_hat[None, :]) @ Fm) / den[None, :], axis=1)
    return (1 + np.sum(t_null >= t_obs)) / (B + 1)


def crt(E, X, j, S, B, rng, model):
    XS = X[:, list(S)]
    return crt_given(E, X[:, j], XS, propensity(E, XS, rng, model), B, rng)


def greedy(E, X, j, B, rng, alpha, model, depth=3, pool_m=6):
    p = X.shape[1]
    others = [k for k in range(p) if k != j]
    c = np.abs([np.corrcoef(X[:, k], X[:, j])[0, 1] for k in others])
    pool = [others[i] for i in np.argsort(-c)[:pool_m]]
    S, visited = [], [()]
    p_max = crt(E, X, j, [], B, rng, model)
    while p_max <= alpha and len(S) < depth:
        best, best_p = None, -1
        for k in pool:
            if k in S:
                continue
            pv = crt(E, X, j, S + [k], B, rng, model)
            if pv > best_p:
                best, best_p = k, pv
        if best is None:
            break
        S.append(best)
        visited.append(tuple(S))
        p_max = max(p_max, best_p)
    return p_max, len(visited)


def run_one(args):
    rep, p, n, B, alpha = args
    rng = np.random.default_rng(5000 + rep * 31 + n)
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
        for model in ("logit2", "gbm"):
            res[f"oracle_pa/{model}"] = crt(E, X, j, pa, B, rng, model)
        if pa:
            big = _gbm().fit(Xb[:, pa], Eb)
            res["oracle_pa/big"] = crt_given(E, X[:, j], X[:, pa],
                                             big.predict_proba(X[:, pa])[:, 1], B, rng)
        else:
            res["oracle_pa/big"] = res["oracle_pa/gbm"]
        model = "gbm"
        for tag, miss, nf in (("proposer_good", 0.1, 1), ("proposer_bad", 0.4, 2)):
            cands = {tuple(noisy_parents(world["W"], j, rng, miss, nf, p)) for _ in range(3)}
            ps = [crt(E, X, j, list(c), B, rng, model) for c in cands]
            res[tag] = max(ps)
            res[tag + "/n_tests"] = len(ps)
        res["greedy"], res["greedy/n_tests"] = greedy(E, X, j, B, rng, alpha, model)
        res["proposer_bad+greedy"] = max(res["proposer_bad"], res["greedy"])
        rows.append({"rep": rep, "n": n, "j": j, "cat": cat,
                     "kind": world["kind"].get(j, ""), **res})
    return rows


RULES = ["oracle_pa/logit2", "oracle_pa/gbm", "oracle_pa/big",
         "proposer_good", "proposer_bad", "greedy", "proposer_bad+greedy"]


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
            for k in ("proposer_good", "proposer_bad", "greedy")}
    return out


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    p, B, alpha = 12, 500, 0.05
    jobs = [(r, p, n, B, alpha) for n in (300, 1000) for r in range(reps)]
    t = time.time()
    with Pool(11) as pool:
        rows = sum(pool.map(run_one, jobs), [])
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(rows, open(os.path.join(here, "rows2.json"), "w"))
    summ = summarize(rows, alpha)
    json.dump(summ, open(os.path.join(here, "summary2.json"), "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s")
    for k, v in summ.items():
        print(k, v)
