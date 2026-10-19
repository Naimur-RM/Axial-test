"""AxialMB v2: amortized Markov-boundary set prediction.

A network pretrained on simulated worlds reads a whole data matrix and returns,
for every variable, a learned boundary size and the most probable partners.
The decision is the network's own output; no threshold is set after training.
"""

from .config import make_config
from .model import AxialMB, build_model

__all__ = ["AxialMB", "build_model", "make_config"]
