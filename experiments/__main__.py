"""Run the AxialMB v2 comparisons.

From the repository root:

    python -m axialmb.train --tier smoke
    python -m experiments all --checkpoint output/axialmb_v2/checkpoints/best.pt

``all`` runs four evaluations. Each one scores the Markov boundary
(parents, children, spouses). None of them scores arrow direction.

    mb           IAMB vs the StablePFN discovery head vs AxialMB v2
    boundary     PC, GES, and AVICI, reduced to the moral-graph boundary
    timeseries   synthetic VAR and Box-Jenkins gas furnace vs PCMCI-PC1
    tabular      six bnlearn networks plus the real Sachs observational table
"""

from __future__ import annotations

import os

# JAX must stay on CPU. It shares the process with the PyTorch models,
# and initializing both on the same 6 GB GPU runs out of memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import device_of, new_run_dir
from experiments.run_boundary import run as run_boundary
from experiments.run_mb import run as run_mb
from experiments.run_tabular import run as run_tabular
from experiments.run_timeseries import run as run_timeseries


def build_parser():
    parser = argparse.ArgumentParser(description="AxialMB v2 boundary comparisons")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("mb", "boundary", "timeseries", "tabular", "all"):
        command = sub.add_parser(name)
        command.add_argument("--quick", action="store_true", help="small run that checks the pipeline")
        command.add_argument("--checkpoint", default="output/axialmb_v2/checkpoints/best.pt")
        default_stable = ROOT / "StablePFN" / "checkpoints" / "stablepfn-regressor.pth"
        command.add_argument(
            "--stablepfn-checkpoint",
            default=str(default_stable) if default_stable.is_file() else "",
            help="StablePFN discovery checkpoint. Defaults to StablePFN/checkpoints/stablepfn-regressor.pth when that file exists.",
        )
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--seeds", type=int, default=5, help="independent repetitions per setting")
        command.add_argument("--device", default="")
        command.add_argument("--out", default="output/axialmb_v2/eval")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.device = device_of(args.device)
    tasks = {
        "mb": run_mb,
        "boundary": run_boundary,
        "timeseries": run_timeseries,
        "tabular": run_tabular,
    }
    selected = list(tasks) if args.cmd == "all" else [args.cmd]
    run_dir = new_run_dir(args.out, args.cmd)
    print(f"device {args.device}")
    print(f"results {run_dir}")
    if not Path(args.checkpoint).is_file():
        print(
            f"No AxialMB v2 checkpoint at {args.checkpoint}. "
            "Baselines still run. Train with: python -m axialmb.train --tier A"
        )
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        tasks[name](args, run_dir)
    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
