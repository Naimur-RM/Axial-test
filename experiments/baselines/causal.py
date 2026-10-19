"""Classical causal discovery, scored only as a Markov boundary.

PC (constraint-based) and GES (score-based) return a CPDAG. We moralize the
discovered colliders and take the neighbors of each target. That set is
parents ∪ children ∪ spouses. Arrow direction is not a metric.
"""

from __future__ import annotations

import os

# pandas imports pyarrow, and on this machine pyarrow's DLLs stop JAX from loading
# if they are initialized first. AVICI needs JAX, so load it before pandas.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
try:
    import jax.sharding as _jax_sharding
except Exception:
    _jax_sharding = None

import numpy as np
import pandas as pd


def _is_discrete_matrix(data, max_card=20) -> bool:
    return all(len(np.unique(data[:, j])) <= max_card for j in range(data.shape[1]))


def boundaries_from_mixed(nodes, directed, undirected) -> dict:
    """Neighbors in the moral graph of a mixed graph.

    Undirected edges stay in the boundary as unresolved adjacencies.
    A spouse is added when two nodes both point into the same child.
    """
    adj = {n: set() for n in nodes}
    parents = {n: set() for n in nodes}
    for src, dst in directed:
        adj[src].add(dst)
        adj[dst].add(src)
        parents[dst].add(src)
    for a, b in undirected:
        adj[a].add(b)
        adj[b].add(a)
    children = {n: set() for n in nodes}
    for src, dst in directed:
        children[src].add(dst)
    out = {}
    for t in nodes:
        boundary = set(adj[t])
        for child in children[t]:
            boundary |= set(parents[child]) - {t}
        boundary.discard(t)
        out[t] = boundary
    return out


def _from_causallearn_graph(graph) -> tuple[set, set]:
    """causal-learn marks: graph[i, j] == -1 and graph[j, i] == 1 means i -> j."""
    p = graph.shape[0]
    directed, undirected = set(), set()
    for i in range(p):
        for j in range(i + 1, p):
            a, b = int(graph[i, j]), int(graph[j, i])
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                directed.add((i, j))
            elif a == 1 and b == -1:
                directed.add((j, i))
            else:
                undirected.add((i, j))
    return directed, undirected


def _from_pgmpy_pdag(pdag) -> tuple[list, set, set]:
    nodes = list(pdag.nodes())
    directed = set(tuple(e) for e in pdag.directed_edges)
    undirected = set()
    for a, b in pdag.undirected_edges:
        undirected.add((a, b) if isinstance(a, str) else (a, b))
    # undirected_edges may already be pairs; normalize order does not matter
    undirected = {(a, b) for a, b in undirected}
    return nodes, directed, undirected


def pc_boundaries(data, names=None, alpha=0.05, max_cond=3):
    """PC from causal-learn. Discrete matrices use the chi-square test."""
    from causallearn.search.ConstraintBased.PC import pc

    data = np.asarray(data)
    discrete = _is_discrete_matrix(data)
    if discrete:
        used = np.rint(data).astype(int)
        indep = "chisq"
    else:
        used = np.asarray(data, dtype=np.float64)
        indep = "fisherz"
    cg = pc(
        used, alpha=alpha, indep_test=indep, stable=True,
        show_progress=False, verbose=False, node_names=names,
        max_cond_vars=max_cond,
    )
    directed, undirected = _from_causallearn_graph(cg.G.graph)
    nodes = list(range(data.shape[1])) if names is None else list(names)
    # graph is indexed in column order, so map integer endpoints back to names
    if names is not None:
        directed = {(names[i], names[j]) for i, j in directed}
        undirected = {(names[i], names[j]) for i, j in undirected}
    return boundaries_from_mixed(nodes, directed, undirected), {
        "algorithm": "PC",
        "indep_test": indep,
        "alpha": alpha,
        "max_cond_requested": max_cond,
    }


def ges_boundaries(data, names):
    """GES from pgmpy. Gaussian BIC on continuous data, discrete BIC otherwise."""
    from pgmpy.estimators import GES

    frame = pd.DataFrame(np.asarray(data), columns=list(names))
    discrete = _is_discrete_matrix(np.asarray(data))
    if discrete:
        # Integer codes are treated as continuous by the Gaussian score.
        # Category dtype selects the discrete BIC.
        for col in frame.columns:
            frame[col] = frame[col].round().astype(int).astype("category")
        score = "bic-d"
    else:
        frame = frame.astype(float)
        score = "bic-g"
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        pdag = GES(frame).estimate(scoring_method=score)
    nodes, directed, undirected = _from_pgmpy_pdag(pdag)
    return boundaries_from_mixed(nodes, directed, undirected), {
        "algorithm": "GES",
        "score": score,
    }


def _patch_avici_imports():
    """Make the installed AVICI importable on current JAX and pyarrow.

    JAX 0.7 removed ``PositionalSharding``. Single-device inference does not use it.
    pyarrow removed the Plasma extension that AVICI imports while loading, even
    though pretrained inference never starts a Plasma store.
    """
    import os
    import sys
    import types

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if _jax_sharding is None:
        import jax.sharding as sharding
    else:
        sharding = _jax_sharding
    if "pyarrow._plasma" not in sys.modules:
        try:
            import pyarrow._plasma  # noqa: F401
        except Exception:
            plasma = types.ModuleType("pyarrow._plasma")
            plasma.ObjectID = object
            plasma.ObjectNotAvailable = type("ObjectNotAvailable", (Exception,), {})
            plasma.PlasmaBuffer = object
            plasma.PlasmaClient = object
            plasma.connect = lambda *args, **kwargs: None
            plasma.PlasmaObjectExists = type("PlasmaObjectExists", (Exception,), {})
            plasma.PlasmaObjectNotFound = type("PlasmaObjectNotFound", (Exception,), {})
            plasma.PlasmaStoreFull = type("PlasmaStoreFull", (Exception,), {})
            sys.modules["pyarrow._plasma"] = plasma

    if hasattr(sharding, "PositionalSharding"):
        return

    class PositionalSharding:
        def __init__(self, devices):
            self.devices = devices

        def reshape(self, shape):
            return self

    sharding.PositionalSharding = PositionalSharding


_AVICI_MODEL = None


def _avici_model():
    global _AVICI_MODEL
    if _AVICI_MODEL is not None:
        return _AVICI_MODEL
    _patch_avici_imports()
    import avici
    from pathlib import Path
    import shutil

    local = Path(__file__).resolve().parents[2] / "output" / "axialmb_evaluation" / "avici_cache" / "scm-v0"
    kwargs = local / "kwargs.json"
    checkpoint = local / "checkpoint_0300000.pkl"
    if not (kwargs.is_file() and checkpoint.is_file()):
        local.mkdir(parents=True, exist_ok=True)
        import subprocess
        for name in ("kwargs.json", "checkpoint_0300000.pkl"):
            url = f"https://huggingface.co/larslorch/avici/resolve/main/scm-v0/{name}"
            subprocess.run(
                ["curl.exe", "-L", "--fail", "-o", str(local / name), url],
                check=True,
            )
    # avici checks for kwargs.json in the parent of the checkpoint folder.
    parent_kwargs = local.parent / "kwargs.json"
    if not parent_kwargs.is_file():
        shutil.copyfile(kwargs, parent_kwargs)
    _AVICI_MODEL = avici.load_pretrained(checkpoint_dir=str(local), expects_counts=False)
    return _AVICI_MODEL


def avici_boundaries(data, names, threshold=0.5):
    """AVICI pretrained SCM graph, thresholded, then converted to boundaries.

    The package and its weights are optional. A missing install is reported
    by the caller; this function raises ImportError in that case.
    Weights are read from ``output/axialmb_evaluation/avici_cache/scm-v0`` when
    that folder already holds ``kwargs.json`` and the checkpoint. Hugging Face
    download is the fallback; Python's certificate store on this machine rejects
    that host, so the files are fetched with the system curl when missing.
    """
    model = _avici_model()
    x = np.asarray(data, dtype=np.float32)
    probs = np.asarray(model(x=x, shard_if_possible=False, devices="cpu"))
    edge = probs >= threshold  # edge[i, j] means i -> j
    np.fill_diagonal(edge, False)
    nodes = list(names)
    index = {name: i for i, name in enumerate(nodes)}
    directed = set()
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if edge[i, j]:
                directed.add((src, dst))
    return boundaries_from_mixed(nodes, directed, set()), {
        "algorithm": "AVICI",
        "threshold": threshold,
        "mean_edge_prob": float(probs.mean()),
        "_edge_prob": probs,
    }


def moral_score(edge_prob):
    """Pairwise boundary score from edge probabilities ``P[i, j] = P(i -> j)``.

    ``max(P_ij, P_ji, max_k P_ik P_jk)``: adjacent in either direction, or two
    parents of a shared child.
    """
    p = np.asarray(edge_prob, dtype=np.float64).copy()
    np.fill_diagonal(p, 0.0)
    spouse = (p[:, None, :] * p[None, :, :]).max(-1)
    score = np.maximum(np.maximum(p, p.T), spouse)
    np.fill_diagonal(score, 0.0)
    return score


def try_method(fn, *args, **kwargs):
    try:
        boundaries, info = fn(*args, **kwargs)
        info["status"] = "ok"
        return boundaries, info
    except Exception as exc:
        return None, {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
