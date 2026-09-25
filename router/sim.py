"""Replay a target-weight schedule into daily gross returns and turnover.

One simulator serves both the individual operating points and the meta-policy
mixture. That is deliberate: if the mixture were composed by averaging the modes'
*return series* it would silently get its mode switches for free, and the
comparison against a winner-take-all selector -- whose whole cost is that its
switches are discrete -- would be rigged. Here the mixture is a portfolio like
any other: it holds `sum_pi p(pi) w_pi`, it drifts, and it pays to move.

Causality contract, identical to `code_du/causal_runner_0908.run_paths`:

    returns through close t  are earned by the weights held at close t-1
    the portfolio starts in cash; the first rebalance is the first investment
    turnover at t             is measured against the *drifted* holdings at t
    no weight set at t        uses any price after t

Verified against the source harness in `test_sim.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def simulate(targets: pd.DataFrame, rets: pd.DataFrame, test_idx: pd.DatetimeIndex):
    """Replay `targets` (rebalance dates x assets) over `rets` (days x assets).

    Returns
    -------
    gross : np.ndarray (T,)   daily portfolio return before costs
    turn  : np.ndarray (T,)   one-way turnover charged on the rebalance day
    held  : pd.DataFrame      end-of-day weights, for diagnostics
    """
    cols = list(targets.columns)
    R = rets.loc[test_idx, cols].to_numpy(dtype=float)
    R = np.nan_to_num(R)
    n = len(cols)
    tgt_on = {d: targets.loc[d].to_numpy(dtype=float) for d in targets.index}

    w = np.zeros(n)
    invested = False
    gross = np.zeros(len(test_idx))
    turn = np.zeros(len(test_idx))
    held_rows = np.zeros((len(test_idx), n))

    for i, day in enumerate(test_idx):
        day_r = R[i]
        if invested:
            gross[i] = float(w @ day_r)
            w = w * (1.0 + day_r)
            tot = w.sum()
            if tot > 0:
                w = w / tot
        drifted = w.copy() if invested else np.zeros(n)

        if day in tgt_on:
            new_w = tgt_on[day]
            turn[i] = float(np.abs(new_w - drifted).sum())
            w = new_w.copy()
            invested = True
        held_rows[i] = w
    return gross, turn, pd.DataFrame(held_rows, index=test_idx, columns=cols)


def net_series(gross, turn, idx, bps):
    """Net daily return at a given one-way cost in basis points."""
    return pd.Series(gross - turn * bps / 1e4, index=idx)


def metrics(r: pd.Series, mar_daily: float = 0.0):
    """Annualised metrics. Matches `code_du/experiment._metrics` v2 exactly.

    Drawdown starts at wealth 1 so the entry cost counts; downside deviation is
    taken over all observations, not only the negative ones.
    """
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ("SR", "CAGR", "MDD", "Sortino", "Vol", "Calmar", "n_days")}
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    excess = r - mar_daily
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
    mdd = float(np.min(np.minimum(eq.to_numpy() / peak - 1.0, 0.0))) * 100
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if eq.iloc[-1] > 0 else np.nan
    return {
        "SR": float(mu / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
        "CAGR": cagr,
        "MDD": mdd,
        "Sortino": (float(excess.mean() / downside * np.sqrt(ANN))
                    if downside > 0 else np.nan),
        "Vol": float(sd * np.sqrt(ANN)) * 100,
        "Calmar": (cagr / abs(mdd) if np.isfinite(mdd) and abs(mdd) >= 0.5
                   else np.nan),
        "n_days": int(len(r)),
    }
