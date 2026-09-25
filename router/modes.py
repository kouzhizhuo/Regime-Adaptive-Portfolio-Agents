"""The three operating points, rebuilt on the verified causal harness.

Why this file exists rather than `profinview/backtest/engine.py`
----------------------------------------------------------------
The legacy engine cannot support the claims the paper makes about it:

  * it charges **no transaction cost at all** (`engine.py:395-470` moves cash by
    `qty * price` with no spread or impact term), while the paper reports a
    0-15 bps cost stress;
  * it trades **daily** on entry/exit thresholds, take-profit, stop-loss and an
    8-12% trailing stop, while Section 4 describes a weekly mean-variance
    allocator with an l1 turnover penalty;
  * its router saw exactly one expert on all 822 decisions.

So the operating points are re-expressed on `code_du`'s walk-forward engine,
which has an explicit causality contract, an exact capped-simplex allocator with
proven invariants, and a cost ladder that is actually charged.

Dimensional discipline (review question)
------------------------------------------
The reviewer is right that the old score added an expected return, a variance
and a cost that "are not naturally on the same scale". Every expert here emits a
mean in **h-day simple-return units**: a raw dimensionless score `s_m` is mapped
to `mu_m = s_m * sigma_h`, where `sigma_h` is the asset's own h-day return scale.
Then

    mu_m  [return]        sigma_m^2 [return^2]       c_m [return]
    lambda_risk [1/return]                           gamma_cost [dimensionless]

and `S_m = mu_m - lambda_risk*sigma_m^2 - gamma_cost*c_m` is in return units,
readable as the certainty equivalent of exponential utility (exactly so when
lambda_risk = gamma/2).

Expert panel
------------
Four experts, economically distinct, all causal, all reading only columns that
`experiment.make_features` already produces:

    trend    mom21 / sigma_21            level trend view (the legacy `heur`)
    meanrev  -z(mom5)                    contrarian; anti-correlated with trend
    volmom   mom63 / (vol21*sqrt(252))   risk-adjusted trend (re-ranks winners)
    xsrank   cross-sectional rank(mom63) purely relative, no absolute view

They are computed once for the whole panel as (dates x assets) frames, which is
both faster and easier to audit than the per-date object construction the legacy
engine used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0

EXPERT_NAMES = ["trend", "meanrev", "volmom", "xsrank"]

# Four further experts, used only by the M = 8 ablation. A reviewer asked how the
# approach scales with the number of agents; doubling the panel is the direct
# test. They are chosen to extend the same economic axes rather than to add
# near-duplicates: a slower trend, a long-horizon reversal, the low-volatility
# anomaly, and trend acceleration.
EXPERT_NAMES_EXT = EXPERT_NAMES + ["slowmom", "ltrev", "lowvol", "accel"]

# Per-expert predictive-variance multiplier and confidence multiplier. The first
# four are carried over unchanged from
# `profinview/agents/tier2_signal_experts.py` so the expert panel is the same
# economic object the NeurIPS submission described.
EXPERT_VAR_MULT = {"trend": 1.0, "meanrev": 1.5, "volmom": 1.0, "xsrank": 1.2,
                   "slowmom": 1.1, "ltrev": 1.6, "lowvol": 1.3, "accel": 1.4}
EXPERT_CONF_MULT = {"trend": 1.0, "meanrev": 0.9, "volmom": 1.0, "xsrank": 0.95,
                    "slowmom": 0.95, "ltrev": 0.85, "lowvol": 0.9, "accel": 0.88}

# Service-cost proxy in bps, per expert. A cross-sectional expert needs the whole
# panel and is the most expensive to serve; the level experts are cheap.
EXPERT_COST_BPS = {"trend": 2.0, "meanrev": 3.0, "volmom": 3.0, "xsrank": 5.0,
                   "slowmom": 2.0, "ltrev": 3.0, "lowvol": 4.0, "accel": 3.0}


# ------------------------------------------------------------------ experts
def build_expert_panel(px: pd.DataFrame, syms, zwin: int = 126):
    """Raw dimensionless expert scores as {name: DataFrame(dates x syms)}.

    Every column uses a trailing window ending at the row's own date, so row `t`
    is computable from prices <= t. `zwin` is the trailing window for the
    mean-reversion z-score.
    """
    p = px[list(syms)].astype(float)
    r = p.pct_change()

    mom5 = p.pct_change(5)
    mom21 = p.pct_change(21)
    mom63 = p.pct_change(63)
    vol21 = r.rolling(21).std()

    out = {}

    # trend: momentum standardised by its own risk, then squashed
    sig21 = vol21 * np.sqrt(21.0)
    out["trend"] = (mom21 / sig21.replace(0.0, np.nan)).clip(-3.0, 3.0) / 2.0

    # meanrev: negated trailing z-score of the 5-day return
    mu5 = mom5.rolling(zwin).mean()
    sd5 = mom5.rolling(zwin).std()
    out["meanrev"] = (-(mom5 - mu5) / sd5.replace(0.0, np.nan) / 2.0).clip(-1.5, 1.5)

    # volmom: 63-day momentum per unit of annualised vol
    ann_vol = vol21 * np.sqrt(ANN)
    out["volmom"] = (mom63 / ann_vol.replace(0.0, np.nan)).clip(-3.0, 3.0) / 2.0

    # xsrank: cross-sectional average rank of 63-day momentum, mapped to [-1, 1]
    rk = mom63.rank(axis=1, method="average")
    cnt = mom63.notna().sum(axis=1)
    out["xsrank"] = ((rk.sub(1.0)).div((cnt - 1.0).replace(0, np.nan), axis=0)
                     * 2.0 - 1.0).clip(-1.0, 1.0)

    # --- the four extension experts (M = 8 ablation only)
    mom126 = p.pct_change(126)
    mom252 = p.pct_change(252)
    vol63 = r.rolling(63).std()

    # slowmom: the same information ratio as volmom but at a slower horizon
    out["slowmom"] = (mom126 / (vol63 * np.sqrt(252.0)).replace(0.0, np.nan)
                      ).clip(-3.0, 3.0) / 2.0
    # ltrev: long-horizon reversal, the classic contrarian counterpart to trend
    mu252 = mom252.rolling(zwin).mean()
    sd252 = mom252.rolling(zwin).std()
    out["ltrev"] = (-(mom252 - mu252) / sd252.replace(0.0, np.nan) / 2.0).clip(-1.5, 1.5)
    # lowvol: the low-volatility anomaly, as a cross-sectional rank of -vol
    rkv = (-vol21).rank(axis=1, method="average")
    cntv = vol21.notna().sum(axis=1)
    out["lowvol"] = ((rkv.sub(1.0)).div((cntv - 1.0).replace(0, np.nan), axis=0)
                     * 2.0 - 1.0).clip(-1.0, 1.0)
    # accel: is the trend strengthening or fading
    out["accel"] = ((mom21 - mom63 / 3.0) / sig21.replace(0.0, np.nan)
                    ).clip(-3.0, 3.0) / 2.0

    for k in out:
        out[k] = out[k].replace([np.inf, -np.inf], np.nan)
    return out


def expert_messages(scores_at_t, sigma_h, horizon_days, names=None):
    """Assemble the message tuples the router consumes.

    Parameters
    ----------
    scores_at_t : dict name -> np.ndarray (n_assets,)  raw dimensionless scores
    sigma_h     : np.ndarray (n_assets,)  h-day return scale of each asset
    horizon_days: int

    Returns
    -------
    mu   (M, n) predicted h-day returns
    var  (M, n) predictive variance, return^2
    conf (M, n) confidence in [0, 1]
    cost (M, n) service cost in return units
    names list of M expert names
    """
    names = [k for k in (names or EXPERT_NAMES) if k in scores_at_t]
    M, n = len(names), len(sigma_h)
    mu = np.zeros((M, n))
    var = np.zeros((M, n))
    conf = np.zeros((M, n))
    cost = np.zeros((M, n))
    for j, name in enumerate(names):
        s = np.asarray(scores_at_t[name], dtype=float)
        finite = np.isfinite(s)
        s = np.where(finite, s, 0.0)
        # score -> return units
        mu[j] = s * sigma_h
        var[j] = np.maximum(sigma_h ** 2 * EXPERT_VAR_MULT[name], 1e-12)
        # a dimensionless confidence that falls as the posterior widens; an
        # expert with no information for this asset reports zero and the
        # router's min_conf gate drops it honestly.
        conf[j] = np.where(finite,
                           EXPERT_CONF_MULT[name] / (1.0 + 3.0 * np.sqrt(var[j])),
                           0.0)
        cost[j] = EXPERT_COST_BPS[name] / 1e4
    return mu, var, conf, cost, names


# ------------------------------------------------------------------- router
def route(mu, var, conf, cost, lam_risk, gam_cost, top_k, tau, min_conf=0.05,
          aggregator="precision"):
    """Utility-aware Top-K routing followed by aggregation. Vectorised over assets.

    Score (return units):  S_m = mu_m - lambda_risk * sigma_m^2 - gamma_cost * c_m

    Aggregation
    -----------
    `precision`  the exact posterior mean under conditionally independent
                 Gaussian signals of a common latent mean: omega_m ∝ q_m/sigma_m^2.
                 This is what "Bayesian aggregation" should have meant; reviewer
                 A reviewer correctly observed the submitted formula was not derived.
    `softmax`    the submitted utility-tilted rule, omega_m ∝ exp(S_m/tau),
                 retained as an ablation.

    In both cases the reported variance is the exact mixture variance (law of
    total variance): within-expert plus between-expert disagreement.

    Returns (mu_agg, var_agg, weights) with weights (M, n).
    """
    M, n = mu.shape
    S = mu - lam_risk * var - gam_cost * cost

    eligible = conf >= min_conf
    # If the gate would silence every expert for an asset, keep the most
    # confident one rather than emitting an arbitrary zero view.
    dead = ~eligible.any(axis=0)
    if dead.any():
        eligible[np.argmax(conf[:, dead], axis=0), np.where(dead)[0]] = True

    # Top-K by score among eligible experts, per asset.
    masked = np.where(eligible, S, -np.inf)
    k = int(min(top_k, M))
    keep = np.zeros_like(eligible)
    order = np.argsort(-masked, axis=0)[:k]
    np.put_along_axis(keep, order, True, axis=0)
    keep &= eligible

    if aggregator == "precision":
        raw = np.where(keep, conf / var, 0.0)
    elif aggregator == "softmax":
        z = np.where(keep, masked, -np.inf)
        z = z - np.max(np.where(keep, z, -np.inf), axis=0, keepdims=True)
        raw = np.where(keep, np.exp(z / max(tau, 1e-9)), 0.0)
    else:
        raise ValueError(f"unknown aggregator {aggregator!r}")

    tot = raw.sum(axis=0, keepdims=True)
    omega = np.divide(raw, tot, out=np.zeros_like(raw), where=tot > 0)
    # Asset with no surviving expert: fall back to an equal mixture of the
    # kept set so the column still sums to one.
    empty = (tot[0] <= 0)
    if empty.any():
        omega[:, empty] = keep[:, empty] / np.maximum(keep[:, empty].sum(axis=0), 1)

    mu_agg = (omega * mu).sum(axis=0)
    within = (omega * var).sum(axis=0)
    between = (omega * (mu - mu_agg[None, :]) ** 2).sum(axis=0)
    return mu_agg, within + between, omega


# ------------------------------------------------------- augmented features
def augmented_features(px, syms, base_feats):
    """O3's richer Tier-1 panel: base features plus alpha-style columns.

    All columns are trailing-window statistics, so row `t` uses prices <= t.
    Added relative to `experiment.make_features`:

        mom126, mom252   slower trend, the horizon the base panel omits
        xs_mom63         cross-sectional rank of 63-day momentum in [-1, 1]
        xs_vol21         cross-sectional rank of realised vol
        skew63, kurt63   higher moments of the return distribution
        vol_ratio        short/long volatility, a crude regime state
    """
    p = px[list(syms)].astype(float)
    r = p.pct_change()
    mom63 = p.pct_change(63)
    vol21 = r.rolling(21).std()
    vol63 = r.rolling(63).std()

    def _xs_rank(df):
        rk = df.rank(axis=1, method="average")
        cnt = df.notna().sum(axis=1)
        return (rk.sub(1.0)).div((cnt - 1.0).replace(0, np.nan), axis=0) * 2.0 - 1.0

    xs_mom = _xs_rank(mom63)
    xs_vol = _xs_rank(vol21)

    out = {}
    for s in syms:
        f = base_feats[s].copy()
        f["mom126"] = p[s].pct_change(126)
        f["mom252"] = p[s].pct_change(252)
        f["xs_mom63"] = xs_mom[s]
        f["xs_vol21"] = xs_vol[s]
        f["skew63"] = r[s].rolling(63).skew()
        f["kurt63"] = r[s].rolling(63).kurt()
        f["vol_ratio"] = vol21[s] / vol63[s].replace(0.0, np.nan)
        out[s] = f.replace([np.inf, -np.inf], np.nan)
    return out
