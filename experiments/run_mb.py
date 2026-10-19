"""Classical IAMB against the StablePFN discovery head and AxialMB v2.

Synthetic worlds from the v2 generator, drawn from a seed space that training
never uses. Each seed gives four worlds: continuous, discrete with Dirichlet
tables, mixed, and the held-out lattice family. Every method returns a
boundary set, scored against parents ∪ children ∪ spouses. Orientation is not
scored.

IAMB runs at every alpha in ``ALPHAS``. ``iamb_best_alpha`` is the alpha that
did best on this very data, an oracle best case that favors the baseline.
"""

from __future__ import annotations

import numpy as np

from axialmb.prior import FAMILIES, graph_world

from .baselines.classical_mb import iamb
from .baselines.stablepfn_mb import load_stablepfn
from .common import (ALPHAS, OURS, Scoreboard, add_best_alpha, average_precision, axial_predict,
                     load_axial, print_table, score_target, stable_predict, write_json)

EVAL_SEED_OFFSET = 900_000
WORLD_SPECS = [
    ("continuous", dict(kind="continuous")),
    ("discrete", dict(kind="discrete", discrete_style="cpt")),
    ("mixed", dict(kind="mixed")),
    ("lattice_holdout", dict(kind="continuous", family="lattice")),
]


def world_matrices(world):
    """Classical tests see category codes; neural methods see the z-scored table."""
    classical = np.where(world["is_cat"][None, :], world["codes"], world["X"]).astype(np.float64)
    neural = (classical - classical.mean(0)) / (classical.std(0) + 1e-8)
    return classical, neural.astype(np.float32)


def run(args, run_dir):
    device = args.device
    model, _cfg = load_axial(args.checkpoint, device)
    stable, stable_note = None, "no StablePFN checkpoint"
    if args.stablepfn_checkpoint:
        try:
            stable = load_stablepfn(args.stablepfn_checkpoint, device=device)
            stable_note = args.stablepfn_checkpoint
        except Exception as exc:
            stable_note = f"failed to load: {type(exc).__name__}: {exc}"

    seeds = list(range(args.seeds))
    p, n, per_world = (10, 300, 3) if args.quick else (18, 800, 6)
    train_families = [f for f in FAMILIES if f != "lattice"]
    board = Scoreboard()
    details, stable_rows = [], []
    for seed in seeds:
        for w_idx, (label, spec) in enumerate(WORLD_SPECS):
            rng = np.random.default_rng([EVAL_SEED_OFFSET, args.seed, seed, w_idx])
            spec = dict(spec)
            spec.setdefault("family", train_families[int(rng.integers(len(train_families)))])
            world = graph_world(rng, p, n, **spec)
            classical, neural = world_matrices(world)
            targets = [i for i, r in enumerate(world["roles"]) if r["boundary"]]
            targets = sorted(rng.choice(targets, min(per_world, len(targets)), replace=False).tolist())
            ours, prob = (None, None)
            if model is not None:
                ours, prob = axial_predict(model, world["X"], device, world["codes"], world["is_cat"],
                                           world["n_states"], rows=targets, seed=seed)
            rec = {"seed": seed, "world": label, "family": world["family"], "mech": world["mech"],
                   "targets": targets}
            for t in targets:
                roles = world["roles"][t]
                key = (label, seed, t)
                keep = [j for j in range(p) if j != t]
                for a in ALPHAS:
                    sel = iamb(classical[:, keep], classical[:, t], alpha=a)
                    board.add(f"iamb@{a}", key, score_target({keep[j] for j in sel}, roles))
                if ours is not None:
                    board.add(OURS, key, score_target(ours[t], roles),
                              ap=average_precision(prob[t], roles["boundary"], keep))
                if stable is not None:
                    try:
                        sel, scores, used = stable_predict(stable, neural[:, keep], neural[:, t], device, seed)
                        full = np.zeros(p)
                        full[keep] = scores
                        board.add("stablepfn_discovery", key, score_target({keep[j] for j in sel}, roles),
                                  ap=average_precision(full, roles["boundary"], keep))
                        stable_rows.append(used)
                    except Exception as exc:
                        rec.setdefault("stablepfn_errors", []).append(f"{type(exc).__name__}: {exc}")
            details.append(rec)
            print(f"  seed {seed} {label:<16} {world['family']:<16} {world['mech']}")

    best = add_best_alpha(board, "iamb")
    summary = board.summary()
    by_world = {}
    for label, _spec in WORLD_SPECS:
        sub = Scoreboard()
        for m, rows in board.rows.items():
            for key, row in rows.items():
                if key[0] == label:
                    sub.add(m, key, row, board.ap.get(m, {}).get(key))
        by_world[label] = sub.summary()
    payload = {
        "task": "classical_mb_vs_discovery",
        "what_is_measured": "Boundary overlap (parents, children, spouses). Orientation is not scored.",
        "seeds": seeds, "p": p, "n": n, "targets_per_world": per_world,
        "iamb_alphas": list(ALPHAS), "iamb_best_alpha": best,
        "checkpoint": None if model is None else str(args.checkpoint),
        "stablepfn": stable_note,
        "stablepfn_rows_used": sorted(set(stable_rows)),
        "summary": summary, "by_world": by_world, "worlds": details,
    }
    print_table("Markov boundary: IAMB vs StablePFN discovery vs AxialMB v2", summary)
    write_json(run_dir / "mb_classical.json", payload)
    return payload
