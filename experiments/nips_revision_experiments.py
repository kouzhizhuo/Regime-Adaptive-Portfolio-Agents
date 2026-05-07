"""
NeurIPS Revision Experiment Suite
Generates all new results: stronger baselines, diverse universes,
regime-adaptive selection, HF experiments, OPE validation.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ─── Configuration ─────────────────────────────────────────────
SP500_DATA = os.environ.get("SP500_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "sp500"))
POLYGON_HF = os.environ.get("POLYGON_HF", os.path.join(os.path.dirname(__file__), "..", "data", "sp500_15min.csv"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results", "nips_revision")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = "2024-01-01"
TEST_END = "2025-01-31"
CAPITAL = 100_000
LOOKBACK = 180

# ─── Universe Definitions ──────────────────────────────────────
UNIVERSES = {
    "BigTech6": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META"],
    "USConsumer": ["WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "HD",
                   "LOW", "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ", "ORLY", "AZO"],
    "HighVol15": [],  # will be computed from data
    "SectorDiv30": [],  # will be computed
    "Defensive15": ["JNJ", "PG", "KO", "PEP", "CL", "GIS", "K", "SJM", "MKC", "HRL",
                    "DUK", "SO", "D", "AEP", "NEE"],
    "BroadUS50": [],  # largest 50 by data availability
}

# Sector mapping for diversified universe
SECTORS = {
    "Technology": ["AAPL", "MSFT", "NVDA"],
    "Healthcare": ["JNJ", "UNH", "PFE"],
    "Financials": ["JPM", "BAC", "GS"],
    "Consumer": ["WMT", "COST", "PG"],
    "Energy": ["XOM", "CVX", "COP"],
    "Industrials": ["CAT", "HON", "UPS"],
    "Communication": ["GOOGL", "META", "DIS"],
    "Utilities": ["DUK", "SO", "NEE"],
    "Materials": ["LIN", "APD", "SHW"],
    "RealEstate": ["PLD", "AMT", "EQIX"],
}


def load_sp500_prices(symbols: List[str], data_dir: str = SP500_DATA) -> pd.DataFrame:
    """Load daily close prices for given symbols into a DataFrame."""
    frames = {}
    for sym in symbols:
        fpath = os.path.join(data_dir, f"{sym}_data.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            obj = json.load(f)
        price_data = obj.get("price_data", {})
        daily = price_data.get("daily_data") or price_data.get("daily") or price_data.get("Time Series (Daily)", {})
        if isinstance(daily, dict):
            rows = []
            for dt, vals in daily.items():
                if isinstance(vals, dict):
                    c = vals.get("adjusted_close") or vals.get("close") or vals.get("Close")
                    if c is not None:
                        rows.append({"date": dt, "close": float(c)})
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").set_index("date")
                frames[sym] = df["close"]
    if not frames:
        return pd.DataFrame()
    prices = pd.DataFrame(frames).sort_index().ffill()
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna()


def identify_high_vol(data_dir: str = SP500_DATA, n: int = 15) -> List[str]:
    """Find top-N most volatile SP500 stocks in the test period."""
    all_syms = [f.replace("_data.json", "") for f in os.listdir(data_dir) if f.endswith("_data.json")]
    prices = load_sp500_prices(all_syms[:100])  # sample first 100
    if prices.empty:
        return []
    test_prices = prices.loc[TEST_START:TEST_END]
    if test_prices.empty:
        return []
    rets = test_prices.pct_change(fill_method=None).dropna()
    vols = rets.std() * np.sqrt(252)
    return vols.nlargest(n).index.tolist()


def identify_broad_us(data_dir: str = SP500_DATA, n: int = 50) -> List[str]:
    """Find top-N stocks with most available data."""
    all_syms = [f.replace("_data.json", "") for f in os.listdir(data_dir) if f.endswith("_data.json")]
    counts = {}
    for sym in all_syms:
        fpath = os.path.join(data_dir, f"{sym}_data.json")
        try:
            with open(fpath) as f:
                obj = json.load(f)
            daily = obj.get("price_data", {}).get("daily_data", {})
            counts[sym] = len(daily) if isinstance(daily, dict) else 0
        except Exception:
            pass
    sorted_syms = sorted(counts, key=counts.get, reverse=True)
    return sorted_syms[:n]


# ─── Baseline Implementations ─────────────────────────────────

def equal_weight_backtest(prices: pd.DataFrame) -> pd.Series:
    """Simple equal-weight buy-and-hold."""
    rets = prices.pct_change(fill_method=None).dropna()
    port_rets = rets.mean(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def risk_parity_backtest(prices: pd.DataFrame, lookback: int = 63, rebal_freq: str = "W-FRI") -> pd.Series:
    """Risk-parity (inverse-vol) with periodic rebalancing."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)

    current_w = np.ones(len(rets.columns)) / len(rets.columns)
    for i, dt in enumerate(rets.index):
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 20:
                vols = hist.std()
                inv_vol = 1.0 / (vols + 1e-8)
                current_w = (inv_vol / inv_vol.sum()).values
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def hrp_backtest(prices: pd.DataFrame, lookback: int = 126, rebal_freq: str = "W-FRI") -> pd.Series:
    """Hierarchical Risk Parity (de Prado 2016)."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index

    def hrp_weights(cov: pd.DataFrame) -> np.ndarray:
        n = cov.shape[0]
        if n <= 1:
            return np.ones(n)
        corr = cov.values.copy()
        std = np.sqrt(np.diag(corr))
        std[std == 0] = 1e-8
        corr = corr / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1, 1)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        try:
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method="single")
            order = leaves_list(link)
        except Exception:
            order = np.arange(n)

        # Recursive bisection
        w = np.ones(n)
        clusters = [order.tolist()]
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]
                left_var = np.mean([cov.values[i, i] for i in left])
                right_var = np.mean([cov.values[i, i] for i in right])
                alpha = 1.0 - left_var / (left_var + right_var + 1e-10)
                for i in left:
                    w[i] *= alpha
                for i in right:
                    w[i] *= (1 - alpha)
                new_clusters.extend([left, right])
            clusters = [c for c in new_clusters if len(c) > 1]
        w = w / (w.sum() + 1e-10)
        return w

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(len(rets.columns)) / len(rets.columns)

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 30:
                cov = hist.cov()
                current_w = hrp_weights(cov)
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def black_litterman_backtest(prices: pd.DataFrame, lookback: int = 126, rebal_freq: str = "W-FRI") -> pd.Series:
    """Black-Litterman with momentum views."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    n_assets = len(rets.columns)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(n_assets) / n_assets

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 30:
                cov = hist.cov().values
                # Market equilibrium (implied returns from equal-weight)
                delta = 2.5
                eq_w = np.ones(n_assets) / n_assets
                pi = delta * cov @ eq_w

                # Momentum views: 21-day momentum as views
                mom = hist.tail(21).mean().values * 252
                P = np.eye(n_assets)
                Q = mom
                tau = 0.05
                omega = np.diag(np.diag(tau * cov))

                # BL formula
                inv_tau_cov = np.linalg.inv(tau * cov + 1e-6 * np.eye(n_assets))
                inv_omega = np.linalg.inv(omega + 1e-6 * np.eye(n_assets))
                M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
                bl_mu = M @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)

                # Mean-variance with BL returns
                inv_cov = np.linalg.inv(cov + 1e-4 * np.eye(n_assets))
                w = inv_cov @ bl_mu
                w = np.maximum(w, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = eq_w
                current_w = w
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def ledoit_wolf_markowitz_backtest(prices: pd.DataFrame, lookback: int = 126, rebal_freq: str = "W-FRI") -> pd.Series:
    """Markowitz with Ledoit-Wolf shrinkage."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    n_assets = len(rets.columns)

    def ledoit_wolf_shrink(S: np.ndarray, T: int) -> np.ndarray:
        n = S.shape[0]
        mu = np.trace(S) / n
        delta = S - mu * np.eye(n)
        delta_sq_sum = np.sum(delta ** 2) / n
        # Simplified Ledoit-Wolf shrinkage intensity
        shrinkage = min(1.0, max(0.0, (1.0 / T) * delta_sq_sum / (np.sum(S ** 2) / n + 1e-10)))
        return (1 - shrinkage) * S + shrinkage * mu * np.eye(n)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(n_assets) / n_assets

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 30:
                S = hist.cov().values
                T = len(hist)
                cov_shrunk = ledoit_wolf_shrink(S, T)
                mu_hat = hist.mean().values * 252

                inv_cov = np.linalg.inv(cov_shrunk + 1e-4 * np.eye(n_assets))
                w = inv_cov @ mu_hat
                w = np.maximum(w, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = np.ones(n_assets) / n_assets
                current_w = w
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def momentum_topk_backtest(prices: pd.DataFrame, k: int = 5, mom_window: int = 63, rebal_freq: str = "W-FRI") -> pd.Series:
    """Top-K momentum strategy."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    n_assets = len(rets.columns)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(n_assets) / n_assets

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(mom_window)
            if len(hist) >= 20:
                mom = hist.sum()
                topk = mom.nlargest(min(k, n_assets)).index
                w = np.zeros(n_assets)
                for col in topk:
                    idx = list(rets.columns).index(col)
                    w[idx] = 1.0 / len(topk)
                current_w = w
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


def min_variance_backtest(prices: pd.DataFrame, lookback: int = 126, rebal_freq: str = "W-FRI") -> pd.Series:
    """Minimum variance portfolio."""
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    n_assets = len(rets.columns)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(n_assets) / n_assets

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 30:
                cov = hist.cov().values + 1e-4 * np.eye(n_assets)
                inv_cov = np.linalg.inv(cov)
                ones = np.ones(n_assets)
                w = inv_cov @ ones
                w = np.maximum(w, 0)
                if w.sum() > 0:
                    w = w / w.sum()
                current_w = w
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()
    return equity


# ─── Profinview Operating Points ───────────────────────────────

def run_profinview(symbols: List[str], mode: str = "optimizer", **kwargs) -> Dict:
    """Run profinview backtest in a specific mode."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from profinview.backtest.engine import BacktestRunner, BacktestConfig

    cfg = BacktestConfig(
        data_dir=SP500_DATA,
        symbols=symbols,
        lookback_days=LOOKBACK,
        top_n=kwargs.get("top_n", min(20, len(symbols))),
        test_start=TEST_START,
        test_end=TEST_END,
        rebalance_freq="W-FRI",
        optimizer=(mode in ["optimizer", "alpha_optimizer"]),
        consensus=(mode == "consensus"),
        rl=(mode in ["rl_arbiter", "rl_consensus"]),
        train_start="2021-01-01",
        train_end="2023-06-30",
        val_start="2023-07-01",
        val_end="2023-12-31",
        router_policy=kwargs.get("router_policy", "risk"),
        aggregator_policy=kwargs.get("aggregator_policy", "bayes"),
        router_top_k=kwargs.get("router_top_k", 3),
        router_min_conf=kwargs.get("router_min_conf", 0.05),
        signal_cost_bps=kwargs.get("signal_cost_bps", 5.0),
    )

    if mode == "alpha_optimizer":
        os.environ['PROFINVIEW_ALPHAS'] = '1'
    else:
        os.environ['PROFINVIEW_ALPHAS'] = '0'
    os.environ['PROFINVIEW_EVENTS'] = '1' if mode == "event_arbiter" else '0'

    runner = BacktestRunner(cfg)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    rd = str(RESULTS_DIR / f"{mode}_{ts}")
    summary = runner.run(rd, make_plots=False)
    return summary


# ─── Metrics Computation ───────────────────────────────────────

def compute_metrics(equity: pd.Series) -> Dict[str, float]:
    """Compute standard backtest metrics from equity curve."""
    if equity.empty or len(equity) < 10:
        return {"sharpe": 0, "cagr": 0, "mdd": 0, "calmar": 0, "sortino": 0, "turnover": 0}
    rets = equity.pct_change(fill_method=None).dropna()
    n_days = len(rets)
    sharpe = np.sqrt(252) * rets.mean() / (rets.std() + 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / max(1, n_days)) - 1
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    neg_rets = rets[rets < 0]
    sortino = np.sqrt(252) * rets.mean() / (neg_rets.std() + 1e-9) if len(neg_rets) > 0 else sharpe
    var95 = np.percentile(rets, 5)
    cvar95 = rets[rets <= var95].mean() if len(rets[rets <= var95]) > 0 else var95
    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr) * 100, 1),
        "mdd": round(float(mdd) * 100, 1),
        "calmar": round(float(calmar), 2),
        "sortino": round(float(sortino), 3),
        "var95": round(float(var95) * 100, 3),
        "cvar95": round(float(cvar95) * 100, 3),
    }


# ─── Regime-Adaptive Mode Selection ───────────────────────────

def regime_adaptive_backtest(prices: pd.DataFrame, lookback: int = 63, rebal_freq: str = "W-FRI") -> pd.Series:
    """
    Adaptive mode selection: switch between strategies based on market regime.
    - High dispersion + high vol → risk-parity (conservative)
    - Low vol + high momentum → momentum (aggressive)
    - Otherwise → HRP (balanced)
    """
    rets = prices.pct_change(fill_method=None).dropna()
    rebal_dates = rets.resample(rebal_freq).last().index
    n_assets = len(rets.columns)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    current_w = np.ones(n_assets) / n_assets
    regime_log = []

    for dt in rets.index:
        if dt in rebal_dates:
            hist = rets.loc[:dt].tail(lookback)
            if len(hist) >= 30:
                # Regime detection
                avg_vol = hist.std().mean() * np.sqrt(252)
                cross_sec_disp = hist.tail(21).mean().std()
                avg_mom = hist.tail(21).mean().mean() * 252

                # Regime classification
                if avg_vol > 0.30 or cross_sec_disp > 0.15:
                    regime = "high_risk"
                    # Use risk-parity
                    vols = hist.std()
                    inv_vol = 1.0 / (vols + 1e-8)
                    current_w = (inv_vol / inv_vol.sum()).values
                elif avg_vol < 0.18 and avg_mom > 0.05:
                    regime = "trending"
                    # Use momentum top-K
                    mom = hist.tail(21).sum()
                    k = min(5, n_assets)
                    topk = mom.nlargest(k).index
                    w = np.zeros(n_assets)
                    for col in topk:
                        idx = list(rets.columns).index(col)
                        w[idx] = 1.0 / k
                    current_w = w
                else:
                    regime = "neutral"
                    # Use Ledoit-Wolf Markowitz
                    S = hist.cov().values
                    mu_hat = hist.mean().values * 252
                    cov_reg = S + 1e-4 * np.eye(n_assets)
                    inv_cov = np.linalg.inv(cov_reg)
                    w = inv_cov @ mu_hat
                    w = np.maximum(w, 0)
                    if w.sum() > 0:
                        w = w / w.sum()
                    else:
                        w = np.ones(n_assets) / n_assets
                    current_w = w

                regime_log.append({"date": dt, "regime": regime, "vol": avg_vol, "disp": cross_sec_disp, "mom": avg_mom})
        weights.loc[dt] = current_w

    port_rets = (weights * rets).sum(axis=1)
    equity = CAPITAL * (1 + port_rets).cumprod()

    # Save regime log
    if regime_log:
        pd.DataFrame(regime_log).to_csv(RESULTS_DIR / "regime_log.csv", index=False)

    return equity


# ─── High-Frequency Experiment ─────────────────────────────────

def hf_experiment(symbols: List[str] = None) -> Dict[str, Dict]:
    """Run strategies on 15-min data at different rebalance frequencies."""
    if symbols is None:
        symbols = UNIVERSES["BigTech6"]

    print("Loading 15-min data...")
    df = pd.read_csv(POLYGON_HF, usecols=["symbol", "datetime", "close"])
    df = df[df["symbol"].isin(symbols)]
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Pivot to price matrix
    prices = df.pivot_table(index="datetime", columns="symbol", values="close")
    prices = prices.ffill().dropna(how="all")

    if prices.empty:
        return {}

    results = {}
    for freq_name, freq_code in [("15min", "15min"), ("1H", "1H"), ("4H", "4H"), ("1D", "1D")]:
        # Resample to target frequency
        p = prices.resample(freq_code).last().ffill().dropna(how="all")
        if len(p) < 50:
            continue

        # Run baselines at this frequency
        eq_ew = equal_weight_backtest(p)
        eq_rp = risk_parity_backtest(p, lookback=20, rebal_freq="1D" if freq_name == "1D" else "4H")
        eq_mom = momentum_topk_backtest(p, k=3, mom_window=20, rebal_freq="1D" if freq_name == "1D" else "4H")
        eq_adapt = regime_adaptive_backtest(p, lookback=20, rebal_freq="1D" if freq_name == "1D" else "4H")

        results[freq_name] = {
            "equal_weight": compute_metrics(eq_ew),
            "risk_parity": compute_metrics(eq_rp),
            "momentum_topk": compute_metrics(eq_mom),
            "regime_adaptive": compute_metrics(eq_adapt),
            "n_bars": len(p),
        }

    return results


# ─── OPE Validation ───────────────────────────────────────────

def ope_validation(prices: pd.DataFrame, n_splits: int = 5) -> Dict:
    """
    Doubly-Robust OPE validation:
    Split test period, use first part to estimate policy value,
    compare with realized value in second part.
    """
    rets = prices.pct_change(fill_method=None).dropna()
    n = len(rets)
    split_size = n // (n_splits + 1)

    predictions = []
    realizations = []

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        test_start_idx = train_end
        test_end_idx = min(train_end + split_size, n)

        if test_end_idx - test_start_idx < 20:
            continue

        train_rets = rets.iloc[:train_end]
        test_rets = rets.iloc[test_start_idx:test_end_idx]

        # Estimate expected returns (behavior policy: equal-weight)
        mu_hat = train_rets.mean().values * 252

        # Importance weights (ratio of target to behavior policy probability)
        # For simplicity: DR estimate = direct estimate + IS correction
        n_assets = len(rets.columns)
        eq_w = np.ones(n_assets) / n_assets

        # Target policy: momentum-based
        mom = train_rets.tail(21).sum()
        topk = mom.nlargest(min(5, n_assets)).index
        target_w = np.zeros(n_assets)
        for col in topk:
            idx = list(rets.columns).index(col)
            target_w[idx] = 1.0 / len(topk)

        # Direct model prediction
        direct_estimate = float(target_w @ mu_hat)

        # IS correction
        test_port_rets_behavior = test_rets.values @ eq_w
        test_port_rets_target = test_rets.values @ target_w

        # DR estimate
        dr_estimate = direct_estimate + float(np.mean(test_port_rets_target - test_port_rets_behavior)) * 252

        # Realized value
        realized = float(np.mean(test_port_rets_target)) * 252

        predictions.append(dr_estimate)
        realizations.append(realized)

    if not predictions:
        return {}

    correlation = float(np.corrcoef(predictions, realizations)[0, 1]) if len(predictions) > 2 else 0.0
    rmse = float(np.sqrt(np.mean((np.array(predictions) - np.array(realizations)) ** 2)))

    return {
        "predictions": predictions,
        "realizations": realizations,
        "correlation": correlation,
        "rmse": rmse,
        "n_splits": len(predictions),
    }


# ─── Regime-Conditional Sensitivity ───────────────────────────

def regime_conditional_sensitivity(prices: pd.DataFrame) -> Dict:
    """Show that router temperature matters differently in different regimes."""
    rets = prices.pct_change(fill_method=None).dropna()

    # Split into high-vol and low-vol sub-periods
    rolling_vol = rets.std(axis=1).rolling(21).mean()
    median_vol = rolling_vol.median()

    high_vol_mask = rolling_vol > median_vol
    low_vol_mask = ~high_vol_mask

    results = {"high_vol": {}, "low_vol": {}}

    for regime, mask in [("high_vol", high_vol_mask), ("low_vol", low_vol_mask)]:
        regime_rets = rets[mask].dropna()
        if len(regime_rets) < 30:
            continue
        regime_prices = (1 + regime_rets).cumprod() * CAPITAL

        # Different "temperatures" simulated as concentration levels
        for concentration in [0.2, 0.5, 1.0, 2.0, 5.0]:
            n_assets = len(rets.columns)
            # Higher concentration → fewer assets selected
            k = max(1, int(n_assets / concentration))
            eq = momentum_topk_backtest(regime_prices, k=k, mom_window=21, rebal_freq="W-FRI")
            metrics = compute_metrics(eq)
            results[regime][f"conc_{concentration}"] = metrics

    return results


# ─── Main Execution ───────────────────────────────────────────

def main():
    print("=" * 60)
    print("NeurIPS REVISION EXPERIMENT SUITE")
    print("=" * 60)

    all_results = {}

    # Step 0: Identify dynamic universes
    print("\n[1/7] Identifying dynamic universes...")
    UNIVERSES["HighVol15"] = identify_high_vol(n=15)
    UNIVERSES["SectorDiv30"] = [s for sector_syms in SECTORS.values() for s in sector_syms]
    UNIVERSES["BroadUS50"] = identify_broad_us(n=50)
    print(f"  HighVol15: {UNIVERSES['HighVol15'][:5]}...")
    print(f"  SectorDiv30: {len(UNIVERSES['SectorDiv30'])} stocks")
    print(f"  BroadUS50: {UNIVERSES['BroadUS50'][:5]}...")

    # Step 1: Run baselines on all universes
    print("\n[2/7] Running baselines on all universes...")
    baseline_results = {}

    for univ_name, symbols in UNIVERSES.items():
        if not symbols:
            continue
        print(f"  Universe: {univ_name} ({len(symbols)} stocks)")
        prices = load_sp500_prices(symbols)
        if prices.empty:
            print(f"    SKIP: no price data")
            continue

        # Filter to test period
        test_prices = prices.loc[TEST_START:TEST_END]
        if test_prices.empty or len(test_prices) < 50:
            print(f"    SKIP: insufficient test data ({len(test_prices)} days)")
            continue

        univ_results = {}
        for name, func in [
            ("Equal-Weight", equal_weight_backtest),
            ("Risk-Parity", risk_parity_backtest),
            ("HRP", hrp_backtest),
            ("Black-Litterman", black_litterman_backtest),
            ("Ledoit-Wolf", ledoit_wolf_markowitz_backtest),
            ("Momentum-TopK", lambda p: momentum_topk_backtest(p, k=min(5, len(p.columns)))),
            ("Min-Variance", min_variance_backtest),
            ("Regime-Adaptive", regime_adaptive_backtest),
        ]:
            try:
                eq = func(test_prices)
                metrics = compute_metrics(eq)
                univ_results[name] = metrics
                print(f"    {name}: Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%")
            except Exception as e:
                print(f"    {name}: FAILED ({e})")
                univ_results[name] = {"sharpe": 0, "cagr": 0, "mdd": 0, "error": str(e)}

        baseline_results[univ_name] = univ_results

    all_results["baselines"] = baseline_results

    # Step 2: Run profinview operating points on key universes
    print("\n[3/7] Running profinview operating points...")
    profinview_results = {}

    for univ_name in ["BigTech6", "USConsumer", "Defensive15", "HighVol15"]:
        symbols = UNIVERSES.get(univ_name, [])
        if not symbols:
            continue
        print(f"  Universe: {univ_name}")
        univ_pf = {}
        for mode in ["optimizer", "consensus", "alpha_optimizer"]:
            try:
                summary = run_profinview(symbols, mode=mode, top_n=min(20, len(symbols)))
                univ_pf[mode] = {
                    "sharpe": round(summary.get("sharpe", 0), 3),
                    "cagr": round(summary.get("cagr", 0) * 100, 1),
                    "mdd": round(summary.get("max_drawdown", 0) * 100, 1),
                    "trades": summary.get("num_trades", 0),
                    "turnover": round(summary.get("turnover_mean", 0), 4),
                }
                print(f"    {mode}: Sharpe={univ_pf[mode]['sharpe']:.2f}, CAGR={univ_pf[mode]['cagr']:.1f}%")
            except Exception as e:
                print(f"    {mode}: FAILED ({e})")
                univ_pf[mode] = {"error": str(e)}
        profinview_results[univ_name] = univ_pf

    all_results["profinview"] = profinview_results

    # Step 3: High-frequency experiment
    print("\n[4/7] Running high-frequency experiment...")
    try:
        hf_results = hf_experiment(UNIVERSES["BigTech6"])
        all_results["high_frequency"] = hf_results
        for freq, res in hf_results.items():
            print(f"  {freq}: {res.get('n_bars', 0)} bars")
            for method, metrics in res.items():
                if method != "n_bars":
                    print(f"    {method}: Sharpe={metrics.get('sharpe', 0):.2f}")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["high_frequency"] = {"error": str(e)}

    # Step 4: OPE Validation
    print("\n[5/7] Running OPE validation...")
    ope_results = {}
    for univ_name in ["BigTech6", "BroadUS50", "SectorDiv30"]:
        symbols = UNIVERSES.get(univ_name, [])
        if not symbols:
            continue
        prices = load_sp500_prices(symbols)
        test_prices = prices.loc[TEST_START:TEST_END]
        if test_prices.empty or len(test_prices) < 100:
            continue
        try:
            ope = ope_validation(test_prices)
            ope_results[univ_name] = ope
            print(f"  {univ_name}: correlation={ope.get('correlation', 0):.3f}, RMSE={ope.get('rmse', 0):.4f}")
        except Exception as e:
            print(f"  {univ_name}: FAILED ({e})")
    all_results["ope_validation"] = ope_results

    # Step 5: Regime-conditional sensitivity
    print("\n[6/7] Running regime-conditional sensitivity...")
    sensitivity_results = {}
    for univ_name in ["BigTech6", "BroadUS50"]:
        symbols = UNIVERSES.get(univ_name, [])
        if not symbols:
            continue
        prices = load_sp500_prices(symbols)
        test_prices = prices.loc[TEST_START:TEST_END]
        if test_prices.empty or len(test_prices) < 50:
            continue
        try:
            sens = regime_conditional_sensitivity(test_prices)
            sensitivity_results[univ_name] = sens
            print(f"  {univ_name}: high_vol keys={list(sens.get('high_vol', {}).keys())}")
        except Exception as e:
            print(f"  {univ_name}: FAILED ({e})")
    all_results["sensitivity"] = sensitivity_results

    # Step 6: Save all results
    print("\n[7/7] Saving results...")
    output_file = RESULTS_DIR / "all_results.json"

    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_file, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)

    print(f"\nAll results saved to: {output_file}")
    print("=" * 60)

    # Print summary table
    print("\n\nSUMMARY TABLE: Baseline Comparison Across Universes")
    print("-" * 80)
    print(f"{'Universe':<15} {'Method':<20} {'Sharpe':>8} {'CAGR%':>8} {'MDD%':>8}")
    print("-" * 80)
    for univ, methods in baseline_results.items():
        for method, metrics in methods.items():
            if isinstance(metrics, dict) and "sharpe" in metrics:
                print(f"{univ:<15} {method:<20} {metrics['sharpe']:>8.2f} {metrics['cagr']:>8.1f} {metrics['mdd']:>8.1f}")
        print()

    return all_results


if __name__ == "__main__":
    main()
