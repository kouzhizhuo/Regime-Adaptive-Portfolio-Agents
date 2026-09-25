# Router and meta-policy code

This folder holds the code for the utility-calibrated router, the three operating
points (O1, O2, O3), the exact turnover-penalized allocator, the meta-policy over
operating points, and the evaluation scripts that produce the paper's tables.
It runs on its own and does not import the `profinview/` package at the repository root.

## Requirements

Python 3.9 or later, with `numpy`, `pandas`, `scipy`, `scikit-learn` and `pytest`.
`lightgbm` is optional (one baseline only).

```bash
pip install numpy pandas scipy scikit-learn pytest
```

## Files

### Engine

| File | Contents |
|---|---|
| `experiment.py` | Causal feature construction (`make_features`) and the frozen ridge forecaster (`FrozenRidge`). |
| `modes.py` | The expert panel (trend, mean reversion, volatility-scaled momentum, cross-sectional momentum rank), the confidence gate, the router score `mu - lambda_risk * sigma^2 - gamma_cost * c`, Top-K selection, and the consensus rules. Expert constants for the experts table are defined here. |
| `run_modes.py` | Walk-forward runner. Produces one target-weight schedule per operating point (O1, O2, O3) with a shared risk model and allocator. |
| `allocator_turnover.py` | Exact mean-variance allocator with an L1 turnover penalty over the capped simplex, including the proximal step. |
| `allocator_exact.py` | Exact capped-simplex mean-variance solvers used by `allocator_turnover.py`. |
| `sim.py` | Daily simulator. Replays a target-weight schedule and charges proportional cost against drifted holdings. |
| `meta_policy.py` | Meta-policies over operating points (Hedge and fixed-share) and the do-no-harm validation gate. |
| `universes.py` | Rule-based universe builder and the named market-regime windows. |
| `baselines.py` | Classical portfolio baselines (equal weight, minimum variance, and others) as target-weight schedules. |
| `main_experiment.py` | Fold definitions, hyperparameter grids, and the two-stage validation search (allocator on O1, then router on O2). |

### Evaluation pipeline

Run in this order:

| Step | File | Output |
|---|---|---|
| 1 | `t1_F_cells.py` | Walk-forward operating points and baselines for every (universe, fold) cell, written to `results/`. |
| 2 | `t1_base_universes.py`, `t1_w2_router.py` | Named-universe cells and the frozen router design (`results/w2_frozen_F_default.json`). |
| 3 | `t1_final_eval.py` | Replay of every policy and baseline at every cost level, written to `final/`. |
| 4 | `t1_final_report.py` | Statistics and LaTeX tables from the replay files. |
| 5 | `t1_final_pairwise.py` | Paired tests of the deployed policy against each comparator, and model-confidence-set membership. |
| 6 | `t1_final_robust_tables.py` | Robustness tables (allocator risk-aversion grid, 4 vs 8 experts). |

### Statistical tests

| File | Contents |
|---|---|
| `acceptance.py` | Stationary block bootstrap for paired differences, Holm correction, SPA and related tests, and the trial-ledger count used for multiple-testing adjustment. |
| `trial_ledger_freeze.csv`, `.sha256` | Frozen list of every configuration evaluated, with its checksum. Read by `acceptance.py`. |

### Data

| File | Contents |
|---|---|
| `fetch_us_panel.py` | Downloads split- and dividend-adjusted daily closes and volumes (2005 onward) from one public source. Expects `data/tickers.txt` with one symbol per line. Writes `data/panel_close_us_2005.csv` and `data/panel_volume_us_2005.csv`. |

Data files are not included in the repository.

### Tests

| File | Checks |
|---|---|
| `test_allocator_turnover.py` | Allocator optimality and feasibility, checked against a general-purpose solver. |
| `test_acceptance.py` | Bootstrap and test functions on synthetic data with known answers. |

```bash
cd router
python -m pytest -q test_allocator_turnover.py test_acceptance.py   # 45 tests
```

Set `T2_FAST=1` to use fewer bootstrap resamples.

## Paths

By default, all paths are relative to this folder:

- `data/`: input price and volume panels
- `results/`: per-cell outputs
- `final/`: replay outputs and tables

To change them, set these environment variables:

- `ROUTER_DATA`: directory with `panel_close_us_2005.csv` and `panel_volume_us_2005.csv`
- `ROUTER_ROOT`: base directory used by `main_experiment.py` and `run_modes.py`

## Quick start

```bash
cd router
python fetch_us_panel.py             # needs data/tickers.txt
python t1_F_cells.py
python t1_base_universes.py
python t1_w2_router.py
python t1_final_eval.py
python t1_final_report.py
python t1_final_pairwise.py
python t1_final_robust_tables.py
```

Each script prints its options with `--help` when it accepts arguments. Bootstrap resamples use fixed seeds.
