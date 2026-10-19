"""Markov-boundary head of StablePFN, without the prediction head's forecast.

StablePFN was trained to mask features for stable prediction. This wrapper
reads only the discovery score and thresholds it at 0.5, which is the rule
used by ``StablePFN/experimenter.py``. It does not score orientation, because
the discovery head does not emit one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
STABLEPFN_ROOT = ROOT / "StablePFN"


def load_stablepfn(checkpoint_path, device="cpu"):
    # experimenter.py imports LimiX at import time. LimiX's own modules
    # import ``inference`` as a top-level package, so both the StablePFN
    # root and the LimiX directory have to be on the path before that import.
    limix_root = STABLEPFN_ROOT / "LimiX"
    for entry in (str(limix_root), str(STABLEPFN_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from experimenter import load_stablePFN

    model, _criterion = load_stablePFN(checkpoint_path, device=torch.device(device))
    model = model.to(torch.device(device))
    model.eval()
    return model


def discovery_boundary(model, x, y, threshold=0.5, device="cpu"):
    """Return feature indices whose discovery score exceeds ``threshold``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xs = StandardScaler().fit_transform(x).astype(np.float32)
    ys = StandardScaler().fit_transform(y.reshape(-1, 1)).astype(np.float32).reshape(-1)
    train_x = torch.tensor(xs, device=device).unsqueeze(1)
    train_y = torch.tensor(ys, device=device).unsqueeze(1)
    test_x = torch.zeros(1, 1, x.shape[1], device=device)
    with torch.no_grad():
        outputs = model(train_x=train_x, train_y=train_y, test_x=test_x)
    scores = outputs[1].detach().float().cpu().numpy().reshape(-1)
    selected = [int(i) for i in np.flatnonzero(scores > threshold)]
    return selected, scores
