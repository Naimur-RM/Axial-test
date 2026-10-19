"""Training tiers.

``smoke`` checks that the pipeline runs. ``A`` fits a 6 GB laptop GPU and is
for development. ``C`` is the model to pretrain on a cloud GPU (L4 or A100).
"""

from __future__ import annotations

MAX_FEATURES = 50
MAX_MB = 25
MAX_STATES = 8

PRESETS = {
    "smoke": dict(
        d=32, heads=4, blocks=2, inducing=16, col_layers=1, steps=30,
        p_min=5, p_max=12, N_min=100, N_max=200,
        batch=4, lr=1e-3, warmup=5, log_every=5, ckpt_every=15, val_every=15, val_worlds=6,
    ),
    "A": dict(
        d=64, heads=4, blocks=4, inducing=32, col_layers=2, steps=6000,
        p_min=5, p_max=40, N_min=200, N_max=1000,
        batch=4, lr=6e-4, warmup=200, log_every=50, ckpt_every=250, val_every=500, val_worlds=24,
    ),
    "C": dict(
        d=192, heads=8, blocks=10, inducing=64, col_layers=3, steps=30000,
        p_min=5, p_max=50, N_min=200, N_max=2000,
        batch=8, lr=3e-4, warmup=1000, log_every=50, ckpt_every=500, val_every=1000, val_worlds=48,
    ),
}

COMMON = dict(
    dropout=0.0,
    amp=True,
    grad_checkpoint=True,
    holdout_family="lattice",
    kind_probs=[0.3, 0.3, 0.2, 0.2],  # continuous, discrete, mixed, lagged
    pos_weight=3.0,
    consistency_weight=0.1,
    role_weight=0.5,
    count_weight=1.0,
    workers=0,
)


def make_config(tier: str = "smoke", **overrides) -> dict:
    if tier not in PRESETS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(PRESETS)}")
    cfg = dict(COMMON)
    cfg.update(PRESETS[tier])
    cfg.update(overrides)
    cfg["tier"] = tier
    cfg["max_features"] = MAX_FEATURES
    cfg["max_mb"] = MAX_MB
    cfg["max_states"] = MAX_STATES
    return cfg
