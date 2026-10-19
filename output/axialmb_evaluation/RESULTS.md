# AxialMB evaluation report

All artifacts from this run are in `output/axialmb_evaluation/`.

| Path | What it is |
|---|---|
| `checkpoints/axial_mb.pt` | Tier A weights and the fitted fusion stacker (about 0.8 MB) |
| `checkpoints/axial_mb.json` | Training summary |
| `checkpoints/axial_mb.pt.train` | Resume checkpoint written during training |
| `logs/train.log` | Training log |
| `logs/experiments.log` | Full evaluation log |
| `all/20261017_213049/mb_classical.json` | IAMB versus the axial certificate |
| `all/20261017_213049/boundary.json` | PC and GES, scored as Markov boundaries |
| `all/20261017_213049/timeseries.json` | Synthetic VARs and the gas furnace versus PCMCI-PC1 |
| `all/20261017_213049/tabular.json` | bnlearn samples and the real Sachs table |

Hardware: NVIDIA GeForce RTX 3050 Laptop, 6 GB. Tier A fits in about 0.5 GB. Tier B (`d=256`, 12 blocks) is the notebook's research-scale model and was not run on this GPU.

Two methods could not be executed:

- **StablePFN discovery head.** The repository has no StablePFN checkpoint. The column is recorded as skipped in every JSON file. Re-run with `--stablepfn-checkpoint PATH` once weights are available.
- **AVICI.** The `avici` package is not installed, so the pretrained causal-discovery network was not called.
- **Published PCMCI.** `tigramite` is not installed. PC1 was run with the included Fisher-z implementation of the same removal rule (`implementation: fisherz_pc1` in the JSON).

## What was trained

Tier A axial transformer: width 64, 4 heads, 4 blocks, 1,200 steps, AdamW at `2e-3`, batch 8. Worlds were drawn on the fly from eight graph families. `lattice` was held out of training. After training, a logistic fusion stacker was fit on 24 fresh worlds from the training families. It combines the transformer membership logit with the six statistical scores (marginal correlation, partial correlation given hubs, collider boost, correlation with hubs, correlation rank, and the precision-matrix row).

The run resumed a checkpoint that had already reached step 701 (same seed, same tier) and finished at step 1,200. Final training membership AUC on the last logged batch: **0.857**. Batch AUC during the last 500 steps sat mostly between 0.85 and 0.99. That number is a training-batch ranking score. The tables below are the held-out evaluations.

## How to read every table

Every method is scored on the **Markov boundary**, not on arrow direction.

For a target `t`:

`boundary(t) = parents(t) ∪ children(t) ∪ spouses(t)`

A spouse is another parent of a child of `t`. A feature counts as recovered when it appears in the predicted set. The method does not have to label it as parent, child, or spouse.

| Column | Meaning |
|---|---|
| precision | Fraction of the predicted set that is truly in the boundary |
| recall | Fraction of the true boundary that was predicted |
| f1 | Harmonic mean of precision and recall |
| fdr | Fraction of the predicted set that is not in the boundary. This is `1 - precision` |
| parent_recall | Fraction of true parents that appear in the predicted set |
| child_recall | Fraction of true children that appear in the predicted set |
| spouse_recall | Fraction of true spouses that appear in the predicted set |
| `na` | That role does not exist for the targets being averaged, so the recall is undefined |

Two thresholds are in force, and they are not the same:

- IAMB, PC, and PC1 use `alpha = 0.05`.
- The axial certificate uses e-BH at `fdr = 0.10`.

The axial number is a target for the false-discovery rate of the returned set. Empirical FDR can sit above 0.10 when the linear nuisance fit is wrong, which is the known failure mode on discrete or deterministic nodes.

PC and GES return a graph. The score uses the neighborhood of the target in the moral graph: skeleton neighbors, plus co-parents of children when a collider was oriented. That is how spouses enter the score. Arrow accuracy is never computed.

## 1. Classical Markov boundary versus the axial certificate

File: `all/20261017_213049/mb_classical.json`

Eight synthetic worlds, 18 variables including the target, 800 samples, target blanket size about 6. Families and mechanisms were random. IAMB uses Fisher-z partial correlation because these worlds are continuous.

| Method | Precision | Recall | F1 | FDR | Parent recall | Child recall | Spouse recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| IAMB | 0.926 | 0.958 | 0.941 | 0.074 | 1.000 | 1.000 | 0.857 |
| AxialMB | 1.000 | 0.896 | 0.936 | 0.000 | 1.000 | 0.917 | 0.786 |

Reading:

- On five of the eight worlds the two methods returned the same set, and that set was exactly the truth (`latent_confounder` / threshold, `modular` / threshold, `lattice` / heavy tail, `small_world` was almost exact for axial, `tree` / threshold).
- Axial's empirical FDR on this batch is 0, under the nominal 0.10. The misses are false negatives. The clearest one is `modular` / polynomial: truth had six members, axial returned three (`2, 10, 14`).
- IAMB's errors are the other kind: on `erdos_renyi` / MLP it added features 4 and 5, and on `collider_hub` / heavy tail it added feature 3. Spouse recall 0.857 means it still found most spouses.
- StablePFN's discovery head is absent. There is nothing to compare until a checkpoint is supplied.

## 2. Causal discovery, scored only as a boundary

File: `all/20261017_213049/boundary.json`

PC is causal-learn's stable PC. GES is pgmpy's score-based search (Gaussian BIC on continuous data, discrete BIC on bnlearn samples). AVICI was skipped.

### Synthetic collider-heavy worlds

Six worlds, family `collider_hub`, 12 variables, 1,000 samples, mechanisms multiplicative, heteroscedastic, or sigmoid.

| Method | Precision | Recall | F1 | FDR | Parent recall | Child recall | Spouse recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC | 0.925 | 0.767 | 0.815 | 0.075 | 1.000 | 1.000 | 0.583 |
| GES | 0.878 | 1.000 | 0.930 | 0.122 | 1.000 | 1.000 | 1.000 |
| AxialMB | 0.944 | 0.917 | 0.929 | 0.056 | 0.667 | 1.000 | 0.917 |

GES recovered every spouse in this batch and also every boundary member (recall 1). It paid for that with FDR 0.122. PC kept a cleaner set (FDR 0.075) and missed more spouses (spouse recall 0.583), which is what happens when a collider is left unoriented: the co-parent never enters the moral graph. Axial sits between them: spouse recall 0.917, FDR 0.056, and it missed some parents (parent recall 0.667). Parent recall here is coverage of the true parent set by the predicted boundary. It is not a count of correct arrows.

### bnlearn Asia, 2,000 forward samples

Eight nodes. The `either` node is a deterministic OR of `lung` and `tub`, which breaks faithfulness. Constraint methods, including this certificate, are expected to struggle there.

| Method | Precision | Recall | F1 | FDR | Parent recall | Child recall | Spouse recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC | 0.927 | 0.529 | 0.570 | 0.073 | 0.583 | 0.667 | 0.500 |
| GES | 1.000 | 0.833 | 0.850 | 0.000 | 0.833 | 0.833 | 1.000 |
| AxialMB | 0.750 | 0.425 | 0.538 | 0.250 | 0.417 | 0.583 | 0.500 |

GES is the strongest boundary estimator on this sample. Axial's FDR of 0.250 is above its nominal 0.10. That is the discrete, partly deterministic regime the verifier's linear nuisance does not cover well, even with the one-hot design used for discrete columns.

## 3. Time series versus PC1 of PCMCI

File: `all/20261017_213049/timeseries.json`

PC1 is the parent-screening stage of PCMCI. It deletes lagged links by partial correlation, conditioning on the strongest remaining parents. Contemporaneous links are not part of this stage. The axial model never saw a time series in training. Each series is turned into a lagged table, and the certificate is asked for the parent set of each variable.

F1 gain is axial F1 minus PC1 F1. A positive gain means the axial blanket matched the true lagged parents more closely.

### Synthetic VARs

Three trials each. Sparse settings: 5 series, lag 2, 2,000 steps, edge density 0.15. Denser setting: 6 series, lag 2, density 0.30.

| Setting | PC1 precision | PC1 recall | PC1 F1 | Axial precision | Axial recall | Axial F1 | F1 gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear, sparse | 0.903 | 1.000 | 0.931 | 1.000 | 1.000 | 1.000 | +0.069 |
| Nonlinear (`tanh`), sparse | 0.903 | 1.000 | 0.931 | 0.958 | 0.958 | 0.944 | +0.014 |
| Linear, denser | 0.963 | 0.968 | 0.959 | 0.975 | 1.000 | 0.986 | +0.027 |

What the gain is made of:

- On the linear sparse VARs both methods found every true parent (recall gain 0). PC1 also kept some extra lags (precision 0.903, FDR 0.097). Axial kept none of those extras, which is the whole +0.069 F1.
- On the nonlinear sparse VARs the gain shrinks to +0.014. Axial's recall is 0.042 below PC1, and its precision is 0.056 above PC1. Partial correlation still recovered the `tanh` parents at this sample size and density, so the nonlinear case did not open a large gap.
- On the denser linear VAR, axial recall is complete and PC1 recall is 0.968. F1 gain is +0.027.

### Real series: Box–Jenkins gas furnace

`axialmb/data/gas_furnace.csv`, 296 samples, two columns. Column 0 is input gas rate. Column 1 is outlet CO2. The book fixes two claims more tightly than it fixes the autoregressive order, so those two claims are the score:

- CO2 depends on gas rate at lags 3, 4, and 5.
- Past CO2 is not a parent of the gas input.

| Method | Delay-3/4/5 recall | Lags of gas recovered as parents of CO2 | CO2-into-gas violations | Parents selected for gas | Parents selected for CO2 |
|---|---:|---|---:|---|---|
| PC1 | 0.333 | lag 3 only | 0 | own lags 1 and 2 | gas lags 2 and 3, own lags 1 and 2 |
| AxialMB | 1.000 | lags 3, 4, and 5 | 0 | own lags 1 and 2 | gas lags 3, 4, and 5, own lags 1, 2, and 3 |

Both methods treat the gas rate as exogenous. Axial recovers the full delay window that PC1 truncates at lag 3. Axial also keeps CO2's own lag 3, which is outside the two scored claims. Delay recall gain versus PC1 is +0.667. Exogeneity violations stay at 0 for both.

## 4. Tabular benchmarks and the real Sachs table

File: `all/20261017_213049/tabular.json`

Benchmarks are 2,000 forward samples from the bnlearn conditional probability tables. Classical methods see the integer categories (chi-square PC, discrete BIC GES, G-test IAMB). The axial model sees the z-scored codes, and its verifier switches to a one-hot design when it detects discrete columns.

The real baseline is the Sachs CD3/CD28 observational table, 853 cells, 11 proteins, log-transformed (`axialmb/data/sachs_cd3cd28_log.csv`). It is scored against the consensus bnlearn/pgmpy Sachs graph. That graph's co-parents are already parents or children of the same node, so the spouse set is empty for every protein. Spouse recall is `na` on both the simulated Sachs network and the real table. Survey, Asia, and Child are the benchmarks that actually contain spouses.

### Survey (6 nodes)

| Method | Precision | Recall | F1 | FDR | Parent | Child | Spouse |
|---|---:|---:|---:|---:|---:|---:|---:|
| IAMB | 1.000 | 0.653 | 0.754 | 0.000 | 0.625 | 0.700 | 0.500 |
| PC | 0.778 | 0.653 | 0.660 | 0.222 | 0.625 | 0.700 | 0.500 |
| GES | 1.000 | 0.361 | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 |
| AxialMB | 1.000 | 0.472 | 0.600 | 0.000 | 0.500 | 0.500 | 0.500 |

IAMB is the best boundary estimator here. Axial matches GES on parents and children, and unlike GES it recovers half of the spouses, with no false inclusions.

### Asia (8 nodes, 2,000 samples)

Same pattern as the boundary section. GES F1 0.850. IAMB F1 0.771 with FDR 0. Axial F1 0.538 with FDR 0.250. Spouse recall is 0.750 for IAMB, 1.000 for GES, and 0.500 for axial.

### Sachs, simulated from the CPTs (11 nodes)

| Method | Precision | Recall | F1 | FDR | Parent | Child |
|---|---:|---:|---:|---:|---:|---:|
| IAMB | 0.886 | 1.000 | 0.927 | 0.114 | 1.000 | 1.000 |
| PC | 0.756 | 1.000 | 0.836 | 0.244 | 1.000 | 1.000 |
| GES | 0.932 | 0.901 | 0.915 | 0.068 | 0.907 | 0.948 |
| AxialMB | 0.924 | 0.741 | 0.792 | 0.076 | 0.722 | 0.895 |

On data drawn from the true tables, IAMB and GES recover almost the whole boundary. Axial's FDR stays near its nominal 0.10, and its recall is the part that lags, especially on parents.

### Child (20 nodes)

| Method | Precision | Recall | F1 | FDR | Parent | Child | Spouse |
|---|---:|---:|---:|---:|---:|---:|---:|
| IAMB | 0.565 | 0.907 | 0.661 | 0.435 | 0.895 | 0.854 | 0.595 |
| PC | 0.886 | 0.840 | 0.811 | 0.114 | 0.868 | 0.854 | 0.381 |
| GES | 1.000 | 0.968 | 0.982 | 0.000 | 0.974 | 0.981 | 0.881 |
| AxialMB | 0.921 | 0.818 | 0.852 | 0.079 | 0.816 | 0.893 | 0.190 |

This is the largest discrete network in the run. GES is nearly exact. Axial keeps FDR at 0.079 and recall at 0.818, so the certificate is behaving, but spouse recall falls to 0.190. IAMB finds more spouses (0.595) and also makes many false inclusions (FDR 0.435).

### Sachs, real observational cells

Consensus graph, 853 real cells, continuous measurements.

| Method | Precision | Recall | F1 | FDR | Parent | Child |
|---|---:|---:|---:|---:|---:|---:|
| IAMB | 0.955 | 0.496 | 0.620 | 0.045 | 0.426 | 0.581 |
| PC | 1.000 | 0.587 | 0.687 | 0.000 | 0.481 | 0.652 |
| GES | 1.000 | 0.496 | 0.626 | 0.000 | 0.426 | 0.581 |
| AxialMB | 0.909 | 0.449 | 0.550 | 0.091 | 0.370 | 0.533 |

Every method loses a large part of the consensus boundary on the real cells. The simulated Sachs table above was much easier, because those samples come from the same CPTs as the graph. The real table is one observational condition (CD3/CD28), and the consensus graph was built with interventions. PC is the best of the four on this table: precision 1, recall 0.587, F1 0.687. Axial is last on F1 (0.550), with FDR 0.091, close to its nominal 0.10.

## What this run supports

- On continuous synthetic blankets, the tier A certificate is conservative: FDR at or under 0.10, recall a little behind IAMB, spouses mostly recovered.
- On graphs, GES is the strongest boundary estimator in this run, especially on Asia and Child. PC misses spouses when colliders stay unoriented. Axial is competitive with GES on the synthetic collider worlds and behind GES on the discrete bnlearn networks.
- Against PC1, the gain is real and small on the synthetic VARs, and it comes from fewer extra lags rather than from a large recall advantage. On the gas furnace the gain is specific: axial returns the lag-3, lag-4, and lag-5 gas effect on CO2, and PC1 returns only lag 3. Neither method violates exogeneity of the input.
- On the real Sachs cells, classical PC is ahead of this tier A model.
- Spouse recovery is the weak point of the axial set on Child (recall 0.190).

## What was not run

- Tier B (width 256, 12 blocks, 20,000 steps).
- StablePFN's discovery head, until a checkpoint is passed.
- AVICI, until `avici` is installed.
- The tigramite implementation of PC1. The Fisher-z PC1 used here follows the same parent-removal rule and was checked on an AR(1), where it kept lag 1 and dropped lag 3.
