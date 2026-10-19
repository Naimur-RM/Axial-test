"""AVICI and classical causal discovery, scored on the Markov boundary.

PC and GES estimate a CPDAG. AVICI gives edge probabilities, thresholded at its
default 0.5. Each graph is moralized, and the reported set for a target is its
neighborhood: parents, children, and spouses. No orientation metric is
computed. AUPRC uses the AVICI moral score ``max(P_ij, P_ji, max_k P_ik P_jk)``.
"""

from __future__ import annotations

import numpy as np

from axialmb.prior import FAMILIES, graph_world

from .baselines.causal import avici_boundaries, ges_boundaries, moral_score, pc_boundaries, try_method
from .common import (ALPHAS, OURS, Scoreboard, add_best_alpha, average_precision, axial_predict,
                     load_axial, print_table, score_target, write_json)
from .run_mb import WORLD_SPECS, world_matrices

EVAL_SEED_OFFSET = 910_000


def graph_methods(classical, neural, names, alphas=ALPHAS):
    """``{method: (boundaries by name or None, info, edge_prob or None)}``."""
    out = {}
    for a in alphas:
        b, info = try_method(lambda a=a: pc_boundaries(classical, names, alpha=a, max_cond=3))
        out[f"pc@{a}"] = (b, info, None)
    b, info = try_method(lambda: ges_boundaries(classical, names))
    out["ges"] = (b, info, None)
    b, info = try_method(lambda: avici_boundaries(neural, names))
    out["avici"] = (b, info, info.pop("_edge_prob", None))
    return out


def run(args, run_dir):
    device = args.device
    model, _cfg = load_axial(args.checkpoint, device)
    seeds = list(range(args.seeds))
    p, n = (8, 400) if args.quick else (14, 1000)
    train_families = [f for f in FAMILIES if f != "lattice"]
    board = Scoreboard()
    notes, details = {}, []
    for seed in seeds:
        for w_idx, (label, spec) in enumerate(WORLD_SPECS):
            rng = np.random.default_rng([EVAL_SEED_OFFSET, args.seed, seed, w_idx])
            spec = dict(spec)
            spec.setdefault("family", train_families[int(rng.integers(len(train_families)))])
            world = graph_world(rng, p, n, **spec)
            classical, neural = world_matrices(world)
            names = [f"v{j}" for j in range(p)]
            targets = [i for i, r in enumerate(world["roles"]) if r["boundary"]]
            methods = graph_methods(classical, neural, names)
            ours, prob = (None, None)
            if model is not None:
                ours, prob = axial_predict(model, world["X"], device, world["codes"], world["is_cat"],
                                           world["n_states"], rows=targets, seed=seed)
            for m, (_b, info, _e) in methods.items():
                if info.get("status") != "ok":
                    notes.setdefault(m, []).append(info)
            for t in targets:
                roles = world["roles"][t]
                key = (label, seed, t)
                others = [j for j in range(p) if j != t]
                for m, (b, _info, edge) in methods.items():
                    if b is None:
                        continue
                    pred = {int(v[1:]) for v in b.get(names[t], ())}
                    ap = None
                    if edge is not None:
                        ap = average_precision(moral_score(edge)[t], roles["boundary"], others)
                    board.add(m, key, score_target(pred, roles), ap=ap)
                if ours is not None:
                    board.add(OURS, key, score_target(ours[t], roles),
                              ap=average_precision(prob[t], roles["boundary"], others))
            details.append({"seed": seed, "world": label, "family": world["family"], "mech": world["mech"]})
            print(f"  seed {seed} {label:<16} {world['family']:<16} {world['mech']}")

    best = add_best_alpha(board, "pc")
    payload = {
        "task": "causal_discovery_boundary",
        "what_is_measured": (
            "Moral-graph neighborhood of each target (parents, children, spouses). "
            "Orientation accuracy is not computed."
        ),
        "seeds": seeds, "p": p, "n": n,
        "pc_alphas": list(ALPHAS), "pc_best_alpha": best,
        "summary": board.summary(),
        "failures": notes,
        "worlds": details,
    }
    print_table("Synthetic boundary recovery: PC, GES, AVICI vs AxialMB v2", payload["summary"])
    write_json(run_dir / "boundary.json", payload)
    return payload
