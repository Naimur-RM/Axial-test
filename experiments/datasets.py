"""Synthetic worlds, bnlearn benchmarks, the real Sachs table, and time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from axialmb.metrics import roles_from_parents

DATA_DIR = Path(__file__).resolve().parents[1] / "axialmb" / "data"

# Box–Jenkins Series J. The input is gas rate; CO2 responds after a delay of
# about three samples. Own-lag order is not uniquely fixed by the book, so the
# scored claims are the delay and the exogeneity of the input, not a full DAG.
GAS_FURNACE_TAU = 5
GAS_FURNACE_INPUT_LAGS = {(0, 3), (0, 4), (0, 5)}
GAS_FURNACE_FORBIDDEN = {(1, lag) for lag in range(1, GAS_FURNACE_TAU + 1)}


def load_bnlearn(name):
    from pgmpy.example_models import load_model
    key = name if "/" in name else f"bnlearn/{name}"
    return load_model(key)


def network_roles(model) -> dict:
    nodes = list(model.nodes())
    parents = {n: set(model.get_parents(n)) for n in nodes}
    return roles_from_parents(parents)


def sample_bnlearn(name, n, seed=0):
    """Forward-sample the network's conditional probability tables.

    ``codes`` keeps the integer categories for discrete tests.
    ``standard`` is the z-scored matrix used by the axial pipeline.
    """
    from pgmpy.sampling import BayesianModelSampling

    model = load_bnlearn(name)
    frame = BayesianModelSampling(model).forward_sample(size=n, seed=seed, show_progress=False)
    names = list(frame.columns)
    codes = np.column_stack([
        frame[col].astype("category").cat.codes.to_numpy() for col in names
    ]).astype(np.float64)
    standard = (codes - codes.mean(0)) / (codes.std(0) + 1e-8)
    roles = network_roles(model)
    # roles are keyed by node name; also provide column order
    return dict(name=name, names=names, codes=codes, standard=standard.astype(np.float32), roles=roles)


def load_sachs_real():
    """Observational Sachs CD3/CD28 log-expression, scored against the consensus DAG."""
    path = DATA_DIR / "sachs_cd3cd28_log.csv"
    frame = pd.read_csv(path)
    names = list(frame.columns)
    values = frame.to_numpy(dtype=np.float64)
    standard = (values - values.mean(0)) / (values.std(0) + 1e-8)
    roles = network_roles(load_bnlearn("sachs"))
    missing = [n for n in names if n not in roles]
    if missing:
        raise RuntimeError(f"Sachs columns missing from the consensus graph: {missing}")
    return dict(
        name="sachs_cd3cd28_observational",
        names=names,
        codes=values,
        standard=standard.astype(np.float32),
        roles=roles,
        source=str(path),
        note="Consensus bnlearn/pgmpy sachs DAG. Observational cells, not CPT samples.",
    )


def var_process(d=5, lag=2, n=2000, rng=None, density=0.25, nonlinear=False,
                coef_lo=0.15, coef_hi=0.45, noise=1.0, burn=200):
    """Stable VAR. ``true[i]`` is the set of ``(j, lag)`` parents of series i."""
    rng = rng or np.random.default_rng(0)
    coef = [np.zeros((d, d)) for _ in range(lag)]
    true = {i: set() for i in range(d)}
    for i in range(d):
        for ell in range(lag):
            for j in range(d):
                if rng.random() < density:
                    coef[ell][i, j] = rng.uniform(coef_lo, coef_hi) * rng.choice([-1.0, 1.0])
                    true[i].add((j, ell + 1))
    for ell in range(lag):
        spectral = np.abs(np.linalg.eigvals(coef[ell])).max()
        if spectral > 0.6:
            coef[ell] *= 0.6 / spectral
    length = n + burn
    series = np.zeros((length, d))
    for t in range(lag, length):
        acc = np.zeros(d)
        for ell in range(lag):
            z = series[t - ell - 1]
            acc += coef[ell] @ (np.tanh(z) if nonlinear else z)
        series[t] = acc + rng.normal(0, noise, d)
    series = series[burn:]
    series = (series - series.mean(0)) / (series.std(0) + 1e-8)
    return series.astype(np.float32), true


def lagged_columns(d, lag):
    """``(series, lag)`` for each column of ``axialmb.prior.lagged_matrix`` after the ``d`` current ones."""
    return {d + j * lag + (ell - 1): (j, ell) for j in range(d) for ell in range(1, lag + 1)}


def lagged_design(series, target, lag, max_features=50):
    """Candidates are every ``(j, lag)``. Column index is ``j * L + (lag - 1)``."""
    n, d = series.shape
    if d * lag > max_features:
        raise ValueError(f"d*L={d * lag} exceeds the {max_features} feature cap")
    design = np.zeros((n - lag, d * lag), dtype=np.float32)
    for j in range(d):
        for ell in range(1, lag + 1):
            design[:, j * lag + (ell - 1)] = series[lag - ell: n - ell, j]
    y = series[lag:, target].astype(np.float32)
    return design, y


def parent_columns(true_parents, target, lag):
    return {j * lag + (ell - 1) for (j, ell) in true_parents[target] if ell <= lag}


def load_gas_furnace():
    frame = pd.read_csv(DATA_DIR / "gas_furnace.csv")
    values = frame.to_numpy(dtype=np.float64)
    standard = (values - values.mean(0)) / (values.std(0) + 1e-8)
    return dict(
        name="gas_furnace",
        names=list(frame.columns),
        series=standard.astype(np.float32),
        tau_max=GAS_FURNACE_TAU,
        input_lags_of_co2=GAS_FURNACE_INPUT_LAGS,
        forbidden_parents_of_gas=GAS_FURNACE_FORBIDDEN,
        note=(
            "Box-Jenkins Series J. Scored claims: CO2 depends on gas rate at lags "
            "3, 4 and 5, and past CO2 is not a parent of the gas input."
        ),
    )
