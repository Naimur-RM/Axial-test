# Running the comparisons

The library lives in `axialmb/`. The four evaluations live in `experiments/`.
Run both from the repository root.

Train a model first. `smoke` only checks that training works. Tiers `A` and `B`
match the notebook.

```bash
python -m axialmb.train --tier smoke
python -m axialmb.train --tier A
python -m axialmb.train --tier B --out axialmb_runs/checkpoints/axial_mb.pt
```

Then run one comparison, or all four:

```bash
python -m experiments mb --checkpoint axialmb_runs/checkpoints/axial_mb.pt --stablepfn-checkpoint PATH
python -m experiments boundary --checkpoint axialmb_runs/checkpoints/axial_mb.pt
python -m experiments timeseries --checkpoint axialmb_runs/checkpoints/axial_mb.pt
python -m experiments tabular --checkpoint axialmb_runs/checkpoints/axial_mb.pt
python -m experiments all --quick --checkpoint axialmb_runs/checkpoints/axial_mb.pt
```

`--quick` is a small pipeline check. Omit it for the larger setting.
`--alpha` is the level for IAMB, PC, and PCMCI-PC1 (default 0.05).
`--fdr` is the e-BH level for the axial certificate (default 0.10).

What each command measures:

- `mb`: IAMB versus the StablePFN discovery score versus the axial certificate, on synthetic worlds.
- `boundary`: PC, GES, and AVICI. Each graph is reduced to the moral-graph neighborhood (parents, children, spouses). Arrow direction is not a metric. AVICI runs only if the package is installed.
- `timeseries`: linear and nonlinear VARs, plus the Box–Jenkins gas furnace series, against PC1 (the parent screen inside PCMCI). The printed F1 gain is axial minus PC1.
- `tabular`: forward samples from the bnlearn networks, plus the real Sachs CD3/CD28 table scored against the consensus boundary.

StablePFN's discovery head needs its own checkpoint. Pass it with `--stablepfn-checkpoint`. Without that file the column is recorded as skipped.

Results are written under `axialmb_runs/<command>/<timestamp>/`.
