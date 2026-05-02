from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
try:
    from scipy import stats as scistats
except Exception:
    scistats = None

# apply a clean style for clearer figures
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    pass


def save_plots(equity: pd.Series, baseline: pd.Series, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, label="Strategy")
    if baseline is not None and not baseline.empty:
        baseline.plot(ax=ax, label="Baseline", alpha=0.8)
    ax.set_title("Equity Curve vs Baseline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_vs_baseline.png", dpi=200)
    plt.close(fig)

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    fig, ax = plt.subplots(figsize=(10, 3))
    dd.plot(ax=ax, color="red")
    ax.set_title("Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(out_dir / "drawdown.png", dpi=200)
    plt.close(fig)


def save_plots_normalized(eq_norm: pd.Series, base_norm: pd.Series, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    eq_norm.plot(ax=ax, label="Strategy (Normalized)")
    if base_norm is not None and not base_norm.empty:
        base_norm.plot(ax=ax, label="Baseline (Normalized)", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_title("Cumulative Return (%) — Normalized at Test Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "normalized_returns.png", dpi=200)
    plt.close(fig)


def save_rolling_and_turnover(returns: pd.Series, out_dir: Path, window: int = 63, turnover: pd.Series | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if returns is not None and not returns.empty:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std(ddof=0)
        rolling_sharpe = np.sqrt(252) * (rolling_mean / (rolling_std + 1e-12))
        fig, ax = plt.subplots(figsize=(10, 3))
        rolling_sharpe.plot(ax=ax, color="purple")
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(f"Rolling Sharpe ({window}-day)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe")
        fig.tight_layout()
        fig.savefig(out_dir / "rolling_sharpe.png", dpi=200)
        plt.close(fig)

    if turnover is not None and not turnover.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        turnover.plot(ax=ax, color="teal")
        ax.set_title("Daily Turnover (sum |Δweights| / 2)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover")
        fig.tight_layout()
        fig.savefig(out_dir / "turnover.png", dpi=200)
        plt.close(fig)


def save_distribution_and_beta(strategy_rets: pd.Series, baseline_rets: Optional[pd.Series], out_dir: Path, window: int = 63) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if strategy_rets is not None and not strategy_rets.empty:
        # Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        strategy_rets.dropna().hist(ax=ax, bins=50, color="steelblue")
        ax.set_title("Daily Returns Histogram")
        ax.set_xlabel("Return")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / "returns_hist.png", dpi=200)
        plt.close(fig)

        # QQ plot
        if scistats is not None:
            fig = plt.figure(figsize=(4, 4))
            scistats.probplot(strategy_rets.dropna(), dist="norm", plot=plt)
            plt.title("QQ Plot (Strategy Returns vs Normal)")
            plt.tight_layout()
            plt.savefig(out_dir / "returns_qq.png", dpi=200)
            plt.close(fig)

    # Rolling beta vs baseline
    if baseline_rets is not None and not baseline_rets.empty and strategy_rets is not None and not strategy_rets.empty:
        sr = strategy_rets.reindex(baseline_rets.index).dropna()
        br = baseline_rets.dropna()
        idx = sr.index.intersection(br.index)
        sr = sr.loc[idx]
        br = br.loc[idx]
        if len(sr) > window + 5:
            cov = sr.rolling(window).cov(br)
            var = br.rolling(window).var()
            beta = cov / (var + 1e-12)
            fig, ax = plt.subplots(figsize=(10, 3))
            beta.plot(ax=ax, color="darkorange")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"Rolling Beta vs Baseline ({window}-day)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Beta")
            fig.tight_layout()
            fig.savefig(out_dir / "rolling_beta.png", dpi=150)
            plt.close(fig)


def save_weights_heatmap(weights_df: pd.DataFrame, out_dir: Path, top_k: int = 25) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if weights_df is None or weights_df.empty:
        return
    # select top_k symbols by average absolute weight
    avg_abs = weights_df.abs().mean(axis=0).sort_values(ascending=False)
    cols = list(avg_abs.head(top_k).index)
    W = weights_df[cols].copy()
    # plot heatmap
    fig, ax = plt.subplots(figsize=(max(8, top_k/2), 6))
    im = ax.imshow(W.T, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-W.abs().max().max(), vmax=W.abs().max().max())
    ax.set_title("Weights Heatmap (Top holdings)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Symbols")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Weight')
    fig.tight_layout()
    fig.savefig(out_dir / "weights_heatmap.png", dpi=200)
    plt.close(fig)


def save_rolling_accuracy(accuracy_series: pd.Series, out_dir: Path, window: int = 63) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if accuracy_series is None or accuracy_series.empty:
        return
    roll = accuracy_series.rolling(window).mean()
    fig, ax = plt.subplots(figsize=(10, 3))
    roll.plot(ax=ax, color="seagreen")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Directional Accuracy ({window}-day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_accuracy.png", dpi=200)
    plt.close(fig)


def save_calibration_curve(pred: pd.Series, realized: pd.Series, out_dir: Path, n_bins: int = 10) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pred is None or realized is None or pred.empty or realized.empty:
        return
    df = pd.DataFrame({"pred": pred, "real": realized}).dropna()
    if df.empty:
        return
    df = df.sort_values("pred")
    bins = pd.qcut(df["pred"], q=min(n_bins, max(2, len(df)//20)), duplicates='drop')
    grouped = df.groupby(bins)
    x = grouped["pred"].mean()
    y = grouped["real"].mean()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y, marker="o")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, linestyle="--", color="gray")
    ax.set_title("Calibration Curve (pred vs realized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Realized")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=200)
    plt.close(fig)


def save_regime_bars(regime: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not regime:
        return
    labels = []
    cagr_vals = []
    sharpe_vals = []
    if "low_vol_cagr" in regime and "high_vol_cagr" in regime:
        labels = ["Low vol", "High vol"]
        cagr_vals = [regime.get("low_vol_cagr", 0.0), regime.get("high_vol_cagr", 0.0)]
        sharpe_vals = [regime.get("low_vol_sharpe", 0.0), regime.get("high_vol_sharpe", 0.0)]
    else:
        return
    # CAGR bars
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, cagr_vals, color=["#4CAF50", "#F44336"]) 
    ax.set_title("Regime Performance — CAGR")
    ax.set_ylabel("CAGR")
    fig.tight_layout()
    fig.savefig(out_dir / "regime_cagr.png", dpi=200)
    plt.close(fig)
    # Sharpe bars
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, sharpe_vals, color=["#4CAF50", "#F44336"]) 
    ax.set_title("Regime Performance — Sharpe")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(out_dir / "regime_sharpe.png", dpi=200)
    plt.close(fig)
