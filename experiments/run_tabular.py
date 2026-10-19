"""Tabular benchmarks and the real Sachs observational table.

Benchmarks are forward samples from bnlearn networks whose Markov boundaries
are known exactly: survey, asia, sachs, child, insurance, alarm. Each seed is a
fresh sample of ``n`` rows. None of these networks is used in training or model
selection. The real baseline is the Sachs CD3/CD28 log-expression table, scored
against the consensus boundary.

Classical tests see the integer categories. AxialMB v2 sees the categories with
their type, and the StablePFN discovery head and AVICI see the z-scored matrix.
"""

from __future__ import annotations

import numpy as np

from .baselines.classical_mb import iamb
from .baselines.causal import moral_score
from .baselines.stablepfn_mb import load_stablepfn
from .common import (ALPHAS, OURS, Scoreboard, add_best_alpha, average_precision, axial_predict,
                     load_axial, print_table, score_target, stable_predict, write_json)
from .datasets import load_sachs_real, sample_bnlearn
from .run_boundary import graph_methods

BENCHMARKS = ["survey", "asia", "sachs", "child", "insurance", "alarm"]


def _evaluate(board, table, seed, model, stable, device, discrete, stable_rows, errors):
    names = table["names"]
    p = len(names)
    neural = table["standard"]
    if discrete:
        classical = table["codes"]
        codes = table["codes"].astype(np.int64)
        is_cat = np.ones(p, bool)
        n_states = codes.max(0) + 1
    else:
        classical = np.asarray(table["standard"], dtype=np.float64)
        codes, is_cat, n_states = None, None, None
    graphs = graph_methods(classical, neural, names)
    for m, (_b, info, _e) in graphs.items():
        if info.get("status") != "ok":
            errors.append({"seed": seed, "method": m, **info})
    targets = [t for t, nm in enumerate(names) if table["roles"][nm]["boundary"]]
    ours, prob = None, None
    if model is not None:
        ours, prob = axial_predict(model, classical if discrete else neural, device, codes, is_cat,
                                   n_states, rows=targets, seed=seed)
    for t in targets:
        name = names[t]
        roles = table["roles"][name]
        idx_truth = {names.index(v) for v in roles["boundary"]}
        key = (table["name"], seed, name)
        keep = [j for j in range(p) if j != t]
        for m, (b, _info, edge) in graphs.items():
            if b is None or name not in b:
                continue
            ap = None
            if edge is not None:
                ap = average_precision(moral_score(edge)[t], idx_truth, keep)
            board.add(m, key, score_target(b[name], roles), ap=ap)
        for a in ALPHAS:
            sel = iamb(classical[:, keep], classical[:, t], alpha=a)
            board.add(f"iamb@{a}", key, score_target({names[keep[j]] for j in sel}, roles))
        if ours is not None:
            board.add(OURS, key, score_target({names[j] for j in ours[t]}, roles),
                      ap=average_precision(prob[t], idx_truth, keep))
        if stable is not None:
            try:
                sel, scores, used = stable_predict(stable, neural[:, keep], neural[:, t], device, seed)
                full = np.zeros(p)
                full[keep] = scores
                board.add("stablepfn_discovery", key, score_target({names[keep[j]] for j in sel}, roles),
                          ap=average_precision(full, idx_truth, keep))
                stable_rows.setdefault(table["name"], set()).add(used)
            except Exception as exc:
                errors.append({"seed": seed, "method": "stablepfn_discovery", "target": name,
                               "reason": f"{type(exc).__name__}: {exc}"})


def _finish(board):
    a_iamb = add_best_alpha(board, "iamb")
    a_pc = add_best_alpha(board, "pc")
    return {"iamb_best_alpha": a_iamb, "pc_best_alpha": a_pc, "summary": board.summary()}


def run(args, run_dir):
    device = args.device
    model, _cfg = load_axial(args.checkpoint, device)
    stable = None
    if args.stablepfn_checkpoint:
        try:
            stable = load_stablepfn(args.stablepfn_checkpoint, device=device)
        except Exception as exc:
            print(f"StablePFN checkpoint failed: {exc}")
    names = ["survey", "asia"] if args.quick else BENCHMARKS
    seeds = list(range(1 if args.quick else args.seeds))
    n_samples = 2000
    datasets, stable_rows = {}, {}
    for name in names:
        board, errors = Scoreboard(), []
        for seed in seeds:
            print(f"  benchmark {name} n={n_samples} seed {seed}")
            table = sample_bnlearn(name, n_samples, seed=args.seed * 1000 + seed)
            _evaluate(board, table, seed, model, stable, device, True, stable_rows, errors)
        datasets[name] = _finish(board) | {"errors": errors, "seeds": seeds, "n": n_samples}
        print_table(f"bnlearn {name}", datasets[name]["summary"])

    print("  real baseline: Sachs CD3/CD28 observational")
    real = load_sachs_real()
    board, errors = Scoreboard(), []
    _evaluate(board, real, 0, model, stable, device, False, stable_rows, errors)
    datasets[real["name"]] = _finish(board) | {"errors": errors, "note": real["note"],
                                               "n": int(real["standard"].shape[0])}
    print_table("Sachs observational (consensus boundary)", datasets[real["name"]]["summary"])

    payload = {
        "task": "tabular_benchmark_and_real",
        "what_is_measured": "Boundary overlap on bnlearn samples and on the real Sachs table.",
        "alphas": list(ALPHAS),
        "stablepfn_rows_used": {k: sorted(v) for k, v in stable_rows.items()},
        "datasets": datasets,
    }
    write_json(run_dir / "tabular.json", payload)
    return payload
