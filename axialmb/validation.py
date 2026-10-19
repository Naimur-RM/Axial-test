"""Frozen synthetic validation suite and the pre-benchmark gate.

The suite is a fixed list of seeds, so every call regenerates the same worlds.
None of the benchmark networks (bnlearn, Sachs, gas furnace) is in it.

Slices:

- ``continuous``, ``discrete``, ``mixed``: training families and mechanisms
- ``lattice``: the family held out of training, continuous and discrete
- ``bnlearn_like``: random DAGs with Dirichlet tables only, 2 to 5 states
- ``lagged``: vector autoregressions in the ``[current, lags]`` layout

``quick_validation`` is the cheap version used for checkpoint selection during
training. ``run_suite`` adds IAMB, PC, GES (and PCMCI-PC1 on ``lagged``) and
checks whether v2 beats the best of them on every slice.

    python -m axialmb.validation --checkpoint output/axialmb_v2/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .metrics import boundary_scores
from .prior import FAMILIES, graph_world, lagged_world

SUITE_SEED = 20261019
SLICES = ["continuous", "discrete", "mixed", "lattice", "bnlearn_like", "lagged"]
TRAIN_FAMILIES = [f for f in FAMILIES if f != "lattice"]


def suite_worlds(per_slice=8, p_range=(10, 25), n_range=(500, 2000), seed=SUITE_SEED, slices=SLICES):
    """Deterministic list of ``(slice, index, world)``."""
    out = []
    for s_idx, name in enumerate(SLICES):
        if name not in slices:
            continue
        for i in range(per_slice):
            rng = np.random.default_rng([seed, s_idx, i])
            p = int(rng.integers(p_range[0], p_range[1] + 1))
            n = int(rng.integers(n_range[0], n_range[1] + 1))
            if name == "lagged":
                world = lagged_world(rng, n, p_max=max(p_range[1], 12))
            elif name == "lattice":
                kind = "continuous" if i % 2 == 0 else "discrete"
                world = graph_world(rng, p, n, kind, family="lattice")
            elif name == "bnlearn_like":
                fam = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
                world = graph_world(rng, p, n, "discrete", family=fam, discrete_style="cpt")
            else:
                fam = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
                world = graph_world(rng, p, n, name, family=fam)
            out.append((name, i, world))
    return out


def _targets(world):
    return [i for i, r in enumerate(world["roles"]) if r is not None and r["boundary"]]


def _score_sets(world, sets):
    rows = []
    for i in _targets(world):
        r = world["roles"][i]
        rows.append(boundary_scores(sets.get(i, ()), r["parents"], r["children"], r["spouses"]))
    return rows


def axial_sets(model, world, device, n_views=8):
    from .infer import predict_boundaries

    rows = _targets(world)
    cand = None
    if world["kind"] == "lagged":
        cand = list(range(world["d"], world["X"].shape[1]))
    res = predict_boundaries(
        model, world["X"], world["codes"], world["is_cat"], world["n_states"],
        device=device, n_views=n_views, rows=rows, candidates=cand,
    )
    return res["boundaries"]


def quick_validation(model, cfg, device):
    """Mean boundary F1 over a small frozen subset, single view."""
    per = max(1, cfg.get("val_worlds", 12) // len(SLICES))
    n_hi = min(cfg["N_max"], 1000)
    p_hi = min(cfg["p_max"], 25)
    worlds = suite_worlds(per_slice=per, p_range=(min(10, p_hi), p_hi), n_range=(min(300, n_hi), n_hi),
                          seed=SUITE_SEED + 1)
    by_slice = {}
    for name, _i, world in worlds:
        sets = axial_sets(model, world, device, n_views=1)
        f1 = [r["f1"] for r in _score_sets(world, sets)]
        if f1:
            by_slice.setdefault(name, []).append(float(np.mean(f1)))
    out = {k: round(float(np.mean(v)), 4) for k, v in by_slice.items()}
    out["mean_f1"] = round(float(np.mean(list(out.values()))), 4) if out else 0.0
    return out


def _baselines(world, alpha=0.05):
    from experiments.baselines.causal import ges_boundaries, pc_boundaries, try_method
    from experiments.baselines.classical_mb import iamb

    x = np.where(world["is_cat"][None, :], world["codes"], world["X"]).astype(np.float64)
    p = x.shape[1]
    out = {}
    if world["kind"] == "lagged":
        from experiments.baselines.pcmci_pc1 import pc1

        d, lags = world["d"], world["lags"]
        found, _info = pc1(world["series"], tau_max=lags, alpha=alpha)
        out["pc1"] = {i: {d + j * lags + (ell - 1) for j, ell in found.get(i, ())} for i in range(d)}
        lagcols = list(range(d, p))
        sets = {}
        for i in range(d):
            sel = iamb(x[:, lagcols], x[:, i], alpha=alpha)
            sets[i] = {lagcols[j] for j in sel}
        out["iamb"] = sets
        return out
    sets = {}
    for i in _targets(world):
        keep = [j for j in range(p) if j != i]
        sel = iamb(x[:, keep], x[:, i], alpha=alpha)
        sets[i] = {keep[j] for j in sel}
    out["iamb"] = sets
    names = [f"v{j}" for j in range(p)]
    for key, fn in (("pc", lambda: pc_boundaries(x, names, alpha=alpha)),
                    ("ges", lambda: ges_boundaries(x, names))):
        b, info = try_method(fn)
        if b is not None:
            out[key] = {int(k[1:]): {int(v[1:]) for v in vals} for k, vals in b.items()}
    return out


def run_suite(model, device, per_slice=8, with_baselines=True, out_path=None, log=print):
    per_world = []
    table = {}
    for name, i, world in suite_worlds(per_slice=per_slice):
        t0 = time.time()
        methods = {"axialmb_v2": axial_sets(model, world, device)}
        if with_baselines:
            methods.update(_baselines(world))
        rec = {"slice": name, "index": i, "kind": world["kind"], "family": world["family"],
               "p": int(world["X"].shape[1]), "n": int(world["X"].shape[0])}
        for m, sets in methods.items():
            rows = _score_sets(world, sets)
            if rows:
                f1 = float(np.mean([r["f1"] for r in rows]))
                rec[m] = f1
                table.setdefault(name, {}).setdefault(m, []).append(f1)
        per_world.append(rec)
        log(f"  {name} {i}: " + " ".join(f"{m}={rec[m]:.3f}" for m in methods if m in rec)
            + f" ({time.time() - t0:.1f}s)")
    summary, gate = {}, {}
    for name, methods in table.items():
        summary[name] = {m: round(float(np.mean(v)), 4) for m, v in methods.items()}
        others = {m: v for m, v in summary[name].items() if m != "axialmb_v2"}
        if others:
            best_m = max(others, key=others.get)
            gate[name] = dict(
                ours=summary[name].get("axialmb_v2"), best_baseline=best_m,
                best_baseline_f1=others[best_m],
                passed=bool(summary[name].get("axialmb_v2", 0) >= others[best_m]),
            )
    payload = dict(suite_seed=SUITE_SEED, per_slice=per_slice, summary=summary, gate=gate,
                   all_passed=bool(gate) and all(g["passed"] for g in gate.values()),
                   worlds=per_world)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None):
    import torch

    from .train import load_checkpoint

    parser = argparse.ArgumentParser(description="AxialMB v2 synthetic validation gate")
    parser.add_argument("--checkpoint", default="output/axialmb_v2/checkpoints/best.pt")
    parser.add_argument("--out", default="output/axialmb_v2/validation.json")
    parser.add_argument("--per-slice", type=int, default=8)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _cfg = load_checkpoint(args.checkpoint, device=device)
    res = run_suite(model, device, per_slice=args.per_slice,
                    with_baselines=not args.no_baselines, out_path=args.out,
                    log=lambda s: print(s, flush=True))
    print(json.dumps({"summary": res["summary"], "gate": res["gate"], "all_passed": res["all_passed"]}, indent=2))


if __name__ == "__main__":
    main()
