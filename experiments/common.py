"""Shared loading, scoring, statistics, and result writing for the four evaluations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from axialmb.metrics import aggregate, boundary_scores

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0.001, 0.01, 0.05, 0.1)
OURS = "axialmb_v2"


def device_of(name: str) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_axial(path, device):
    from axialmb.train import load_checkpoint

    path = Path(path)
    if not path.is_file():
        return None, None
    return load_checkpoint(path, device=device)


def axial_predict(model, matrix, device, codes=None, is_cat=None, n_states=None,
                  rows=None, candidates=None, seed=0):
    """``(boundaries, prob)`` from ``predict_boundaries``; retries on CPU after a CUDA OOM."""
    from axialmb.infer import predict_boundaries

    kw = dict(rows=rows, candidates=candidates, seed=seed)
    try:
        res = predict_boundaries(model, matrix, codes, is_cat, n_states, device=device, **kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.to("cpu")
        res = predict_boundaries(model, matrix, codes, is_cat, n_states, device="cpu", **kw)
        model.to(device)
    return res["boundaries"], res["prob"]


def stable_predict(stable, x, y, device, seed=0):
    """StablePFN discovery head with an out-of-memory row cap.

    Returns ``(selected, scores, rows_used)``. ``rows_used`` is recorded so a
    capped run is visible in the output.
    """
    from .baselines.stablepfn_mb import discovery_boundary

    n = x.shape[0]
    for cap in (n, 1000, 500, 250):
        if cap > n:
            continue
        idx = np.arange(n) if cap == n else np.random.default_rng(seed).choice(n, cap, replace=False)
        try:
            sel, scores = discovery_boundary(stable, x[idx], y[idx], device=device)
            return sel, scores, int(cap)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("StablePFN ran out of GPU memory even at 250 rows")


def new_run_dir(root: str, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / name / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"saved {path}")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def score_target(predicted, roles) -> dict:
    return boundary_scores(predicted, roles["parents"], roles["children"], roles["spouses"])


def average_precision(scores, truth, candidates):
    """AUPRC of one target's partner ranking. ``None`` when the truth is empty."""
    from sklearn.metrics import average_precision_score

    cand = list(candidates)
    y = np.array([c in truth for c in cand], dtype=int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(average_precision_score(y, np.asarray([scores[c] for c in cand], dtype=float)))


def bootstrap_ci(values, n_boot=2000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


class Scoreboard:
    """Per-target scores keyed by ``(dataset, seed, target)`` for paired tests."""

    def __init__(self):
        self.rows = {}
        self.ap = {}

    def add(self, method, key, scores, ap=None):
        self.rows.setdefault(method, {})[key] = scores
        if ap is not None:
            self.ap.setdefault(method, {})[key] = ap

    def methods(self):
        return list(self.rows)

    def summary(self, ours=OURS):
        out = {}
        for m, by_key in self.rows.items():
            stats = aggregate(list(by_key.values()))
            stats["f1_ci95"] = bootstrap_ci([r["f1"] for r in by_key.values()])
            if m in self.ap:
                aps = list(self.ap[m].values())
                stats["auprc"] = float(np.mean(aps))
                stats["auprc_n"] = len(aps)
            out[m] = stats
        if ours in self.rows:
            for m in self.rows:
                if m != ours:
                    out[m]["vs_ours"] = paired_test(self.rows[ours], self.rows[m])
        return out


def paired_test(ours, other):
    """Wilcoxon signed-rank on per-target F1, ours minus other, shared targets only."""
    from scipy.stats import wilcoxon

    keys = sorted(set(ours) & set(other), key=str)
    if not keys:
        return None
    diff = np.array([ours[k]["f1"] - other[k]["f1"] for k in keys])
    res = dict(n=len(keys), mean_f1_diff=float(diff.mean()),
               wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), losses=int((diff < 0).sum()))
    if np.any(diff != 0):
        res["wilcoxon_p"] = float(wilcoxon(diff, zero_method="wilcox").pvalue)
    else:
        res["wilcoxon_p"] = 1.0
    return res


def best_alpha(board, prefix):
    """The alpha with the highest mean F1, reported as an oracle best case."""
    best, best_f1 = None, -1.0
    for a in ALPHAS:
        rows = board.rows.get(f"{prefix}@{a}")
        if rows:
            f1 = float(np.mean([r["f1"] for r in rows.values()]))
            if f1 > best_f1:
                best, best_f1 = a, f1
    return best


def add_best_alpha(board, prefix):
    a = best_alpha(board, prefix)
    if a is None:
        return None
    for key, row in board.rows[f"{prefix}@{a}"].items():
        board.add(f"{prefix}_best_alpha", key, row)
    return a


def print_table(title: str, summary: dict):
    cols = ["precision", "recall", "f1", "parent_recall", "child_recall", "spouse_recall", "auprc"]
    print(f"\n{title}")
    header = f"{'method':<22}" + "".join(f"{c:>14}" for c in cols) + f"{'p vs ours':>11}"
    print(header)
    print("-" * len(header))
    for name, stats in summary.items():
        if not isinstance(stats, dict) or "n" not in stats:
            print(f"{name:<22} {stats}")
            continue
        cells = []
        for col in cols:
            value = stats.get(col)
            cells.append(f"{value:14.3f}" if isinstance(value, float) else f"{'na':>14}")
        vs = stats.get("vs_ours") or {}
        pv = vs.get("wilcoxon_p")
        tail = f"{pv:11.3g}" if isinstance(pv, float) else f"{'':>11}"
        print(f"{name:<22}" + "".join(cells) + tail + f"   n={stats.get('n')}")
