# When to Route? Regime-Adaptive Meta-Policies for Hierarchical Portfolio Agents


A three-tier hierarchical investment agent with three operating points — **optimizer-only**, **consensus routing**, and **alpha-augmented** — plus a rolling adaptive meta-policy that selects among them based on market regime.

```
Tier 1 (Information)  →  Tier 2 (Routing & Aggregation)  →  Tier 3 (Allocation)
  Features, Sentiment       Router, Consensus, Calibration     Optimizer, Risk, Execution
                                        ↕
                              Adaptive Meta-Policy
                         (selects O1/O2/O3 per quarter)
```

## Structure

```
profinview/          Core framework
  agents/            Tier 1–3 agents (feature, router, forecaster, optimizer, etc.)
  backtest/          Backtest engine & plotting
  core/              Coordinator, message bus, schemas
  data/              Data loaders
  cli.py             CLI entry point
experiments/         Experiment scripts (Tables 1–6 in paper)
configs/             Example configurations
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Optimizer-only (O1)
python -m profinview.cli --data-dir data/sp500 --symbols "NVDA,MSFT,AAPL,GOOGL,AMZN,META" \
  --backtest --optimizer --test-start 2024-01-01 --test-end 2025-01-31

# Consensus routing (O2)
python -m profinview.cli --data-dir data/sp500 --symbols "NVDA,MSFT,AAPL,GOOGL,AMZN,META" \
  --backtest --optimizer --consensus --router-policy risk --router-top-k 3

# Alpha-augmented (O3)
python -m profinview.cli --data-dir data/sp500 --symbols "NVDA,MSFT,AAPL,GOOGL,AMZN,META" \
  --backtest --optimizer --alphas

# Full experiment suite
export SP500_DATA=/path/to/sp500_data
python experiments/nips_revision_experiments.py
```

Data (S&P 500 daily OHLCV JSON files) is not included — place your files in `data/sp500/` or set `--data-dir`.


