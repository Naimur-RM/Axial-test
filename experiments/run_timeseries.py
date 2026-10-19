"""Lagged parent recovery against PC1, the first stage of PCMCI.

Synthetic linear and nonlinear VARs have a known parent set. The real series is
the Box-Jenkins gas furnace. Two claims are scored there, because the book fixes
them and does not fix a unique autoregressive order:

- CO2 should depend on the gas rate at lags 3, 4 and 5.
- Past CO2 should not be selected as a parent of the gas input.

AxialMB v2 reads the window ``[current values, lags 1..L of every series]`` in
one pass. The current values have no children inside the window, so the
boundary of a current value is its lagged parent set. Only lag columns are
candidates.
"""

from __future__ import annotations

import numpy as np

from axialmb.metrics import set_scores
from axialmb.prior import lagged_matrix

from .baselines.pcmci_pc1 import pc1
from .common import (ALPHAS, OURS, Scoreboard, add_best_alpha, average_precision, axial_predict,
                     load_axial, print_table, write_json)
from .datasets import (GAS_FURNACE_FORBIDDEN, GAS_FURNACE_INPUT_LAGS, lagged_columns,
                       load_gas_furnace, var_process)

SETTINGS = [
    ("linear_sparse", dict(d=5, lag=2, n=2000, nonlinear=False, density=0.15)),
    ("nonlinear_sparse", dict(d=5, lag=2, n=2000, nonlinear=True, density=0.15)),
    ("linear_denser", dict(d=6, lag=2, n=2000, nonlinear=False, density=0.3)),
]


def axial_links(model, series, lag, device, seed=0):
    """``({series i: {(j, lag)}}, {series i: {(j, lag): prob}})``."""
    d = series.shape[1]
    window = lagged_matrix(series, lag)
    colmap = lagged_columns(d, lag)
    sets, prob = axial_predict(model, window, device, rows=list(range(d)),
                               candidates=sorted(colmap), seed=seed)
    found = {i: {colmap[c] for c in sets[i]} for i in range(d)}
    scores = {i: {colmap[c]: float(prob[i, c]) for c in colmap} for i in range(d)}
    return found, scores


def _gas_claims(found):
    co2, gas = set(found.get(1, ())), set(found.get(0, ()))
    delay = GAS_FURNACE_INPUT_LAGS
    return {
        "input_delay_recall": len(co2 & delay) / len(delay),
        "input_delay_hits": sorted(co2 & delay),
        "exogeneity_violations": sorted(gas & GAS_FURNACE_FORBIDDEN),
        "n_exogeneity_violations": len(gas & GAS_FURNACE_FORBIDDEN),
        "parents_of_gas": sorted(gas),
        "parents_of_co2": sorted(co2),
    }


def run(args, run_dir):
    device = args.device
    model, _cfg = load_axial(args.checkpoint, device)
    settings = SETTINGS[:2] if args.quick else SETTINGS
    seeds = list(range(1 if args.quick else args.seeds))
    synthetic = {}
    for name, kw in settings:
        board = Scoreboard()
        for seed in seeds:
            rng = np.random.default_rng([920_000, args.seed, seed])
            series, truth = var_process(rng=rng, **kw)
            d, lag = kw["d"], kw["lag"]
            cand = list(lagged_columns(d, lag).values())
            for a in ALPHAS:
                found, _info = pc1(series, tau_max=lag, alpha=a)
                for i in range(d):
                    if truth[i]:
                        board.add(f"pc1@{a}", (name, seed, i), set_scores(found.get(i, ()), truth[i]))
            if model is not None:
                found, scores = axial_links(model, series, lag, device, seed)
                for i in range(d):
                    if truth[i]:
                        board.add(OURS, (name, seed, i), set_scores(found[i], truth[i]),
                                  ap=average_precision(scores[i], truth[i], cand))
            print(f"  {name} seed {seed}")
        best = add_best_alpha(board, "pc1")
        synthetic[name] = {"setting": kw, "seeds": seeds, "pc1_best_alpha": best, "summary": board.summary()}
        print_table(f"VAR {name}", synthetic[name]["summary"])

    gas = load_gas_furnace()
    gas_out = {"note": gas["note"], "tau_max": gas["tau_max"], "pc1": {}}
    for a in ALPHAS:
        found, info = pc1(gas["series"], tau_max=gas["tau_max"], alpha=a)
        gas_out["pc1"][str(a)] = _gas_claims(found) | {"implementation": info.get("implementation")}
    if model is not None:
        runs = []
        for seed in seeds:
            found, scores = axial_links(model, gas["series"], gas["tau_max"], device, seed)
            runs.append(_gas_claims(found) | {"seed": seed})
            if seed == seeds[0]:
                gas_out[f"{OURS}_co2_scores"] = {f"{j},{ell}": s for (j, ell), s in scores[1].items()}
        gas_out[OURS] = runs
    else:
        gas_out[OURS] = {"status": "skipped", "reason": f"no checkpoint at {args.checkpoint}"}

    payload = {
        "task": "timeseries_vs_pcmci_pc1",
        "what_is_measured": (
            "Lagged parent set. PC1 is the parent-screening stage of PCMCI, "
            "before the MCI tests. No orientation is scored."
        ),
        "alphas": list(ALPHAS),
        "synthetic": synthetic,
        "gas_furnace": gas_out,
    }
    print("\nGas furnace: delay 3/4/5 recall and CO2 -> gas violations")
    for a, c in gas_out["pc1"].items():
        print(f"  pc1@{a}: recall {c['input_delay_recall']:.2f}, violations {c['n_exogeneity_violations']}")
    if isinstance(gas_out[OURS], list):
        for c in gas_out[OURS]:
            print(f"  {OURS} seed {c['seed']}: recall {c['input_delay_recall']:.2f}, "
                  f"violations {c['n_exogeneity_violations']}, CO2 parents {c['parents_of_co2']}")
    write_json(run_dir / "timeseries.json", payload)
    return payload
