"""Set scores against a Markov boundary.

Orientation is not scored. A feature counts as recovered when it appears in the
predicted boundary, whether or not anyone labeled it parent, child, or spouse.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def set_scores(predicted: Iterable, truth: Iterable) -> dict:
    pred, true = set(predicted), set(truth)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 1.0
    recall = tp / len(true) if true else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fdr = (len(pred) - tp) / len(pred) if pred else 0.0
    return dict(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fdr=float(fdr),
        tp=int(tp),
        n_pred=int(len(pred)),
        n_truth=int(len(true)),
    )


def _coverage(predicted: set, group: Iterable) -> float | None:
    group = set(group)
    if not group:
        return None
    return float(len(predicted & group) / len(group))


def boundary_scores(predicted: Iterable, parents, children, spouses) -> dict:
    """Score one target against parents, children, and spouses.

    ``parent_recall`` is the fraction of true parents that landed in the
    predicted boundary. It is not an arrow-direction accuracy.
    """
    pred = set(predicted)
    parents, children, spouses = set(parents), set(children), set(spouses)
    out = set_scores(pred, parents | children | spouses)
    out["parent_recall"] = _coverage(pred, parents)
    out["child_recall"] = _coverage(pred, children)
    out["spouse_recall"] = _coverage(pred, spouses)
    out["n_parents"] = int(len(parents))
    out["n_children"] = int(len(children))
    out["n_spouses"] = int(len(spouses))
    return out


def roles_from_parents(parents: Mapping) -> dict:
    """MB(t) = parents(t) ∪ children(t) ∪ co-parents of children(t)."""
    nodes = list(parents)
    children = {n: set() for n in nodes}
    for node, pas in parents.items():
        for p in pas:
            if p not in children:
                children[p] = set()
            children[p].add(node)
    out = {}
    for t in nodes:
        pa = set(parents.get(t, ()))
        ch = set(children.get(t, ()))
        spouses = set()
        for c in ch:
            spouses |= set(parents.get(c, ())) - {t}
        spouses -= pa | ch | {t}
        out[t] = dict(
            parents=pa,
            children=ch,
            spouses=spouses,
            boundary=pa | ch | spouses,
        )
    return out


def roles_from_labels(member, typ, orient) -> dict:
    """Recover roles from ``generate_world`` label vectors."""
    parents = set(np.flatnonzero(np.asarray(orient) == 0).tolist())
    children = set(np.flatnonzero(np.asarray(orient) == 1).tolist())
    spouses = set(np.flatnonzero(np.asarray(typ) == 2).tolist())
    return dict(parents=parents, children=children, spouses=spouses,
                boundary=parents | children | spouses)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    numeric = [k for k, v in rows[0].items() if _is_number(v) or v is None]
    # also pick up numeric keys that are None only on the first row
    keys = set(numeric)
    for row in rows:
        for k, v in row.items():
            if _is_number(v) or v is None:
                keys.add(k)
    out = {"n": len(rows)}
    for key in sorted(keys):
        vals = []
        for row in rows:
            v = row.get(key)
            if _is_number(v):
                vals.append(float(v))
        out[key] = float(np.mean(vals)) if vals else None
        out[f"{key}_n"] = len(vals)
    return out


def _is_number(v) -> bool:
    if isinstance(v, bool) or v is None:
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    return False
