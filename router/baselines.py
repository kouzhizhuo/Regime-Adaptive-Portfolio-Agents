"""Classical portfolio baselines, emitted as target-weight schedules.

Every baseline returns the same object the operating points do -- a
(rebalance dates x assets) frame -- so all of them are replayed through the one
simulator in `sim.py`, on the same calendar, paying the same costs. That closes
two defects the audit found in the submitted tables: baseline weights formed at
`t` were applied to the return *at* `t` (a one-period look-ahead), and costs
were not charged on the baseline paths at all.

Black-Litterman is included because the abstract claimed seven baselines while
the table carried six, and BL was the one dropped -- despite often being the
strongest comparator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree

ANN = 252.0


def _cov(rets, day, window):
    """Annualised covariance from returns strictly before `day`."""
    hist = rets.loc[:day].iloc[-window:-1].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(hist) < 40:
        return None
    return np.cov(hist.to_numpy(), rowvar=False) * ANN


def _cap(w, cap):
    """Project onto the capped simplex by clip-and-redistribute (stable)."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if w.sum() <= 0:
        return np.full(len(w), 1.0 / len(w))
    w = w / w.sum()
    for _ in range(64):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        room = ~over
        if not room.any():
            return np.full(len(w), 1.0 / len(w))
        room_mass = w[room].sum()
        if room_mass <= 0:
            # Every uncapped name carries zero weight, so there is no
            # proportional rule to redistribute by; spread the excess evenly
            # rather than dividing by zero.
            w[room] += excess / room.sum()
        else:
            w[room] += excess * w[room] / room_mass
    return w / w.sum()


def equal_weight(rets, syms, rebal, **kw):
    n = len(syms)
    return pd.DataFrame(np.full((len(rebal), n), 1.0 / n), index=rebal, columns=syms)


def buy_and_hold(rets, syms, rebal, **kw):
    """Equal weight set once at the first rebalance, then left to drift."""
    n = len(syms)
    return pd.DataFrame(np.full((1, n), 1.0 / n), index=rebal[:1], columns=syms)


def min_variance(rets, syms, rebal, window=126, cap=0.35, **kw):
    rows, when = [], []
    for d in rebal:
        S = _cov(rets[syms], d, window)
        if S is None:
            continue
        inv = np.linalg.pinv(S + np.eye(len(syms)) * 1e-8)
        w = inv @ np.ones(len(syms))
        rows.append(_cap(w, cap))
        when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=syms)


def risk_parity(rets, syms, rebal, window=126, cap=0.35, **kw):
    """Inverse-volatility weighting: the standard naive risk-parity baseline."""
    rows, when = [], []
    for d in rebal:
        S = _cov(rets[syms], d, window)
        if S is None:
            continue
        vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        rows.append(_cap(1.0 / vol, cap))
        when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=syms)


def momentum(rets, syms, rebal, lookback=126, top=5, cap=0.35, px=None, **kw):
    """Equal weight over the top-`top` names by trailing total return."""
    rows, when = [], []
    for d in rebal:
        hist = rets.loc[:d, syms].iloc[-lookback:-1]
        if len(hist) < 40:
            continue
        score = (1 + hist).prod() - 1
        k = min(top, len(syms))
        win = score.sort_values(ascending=False).index[:k]
        w = pd.Series(0.0, index=syms)
        w[win] = 1.0 / k
        rows.append(_cap(w.to_numpy(), cap))
        when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=syms)


def _hrp_weights(S):
    """Lopez de Prado hierarchical risk parity on a covariance matrix."""
    n = S.shape[0]
    sd = np.sqrt(np.clip(np.diag(S), 1e-16, None))
    C = S / np.outer(sd, sd)
    C = np.clip(np.nan_to_num(C, nan=0.0), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - C) / 2.0, 0.0, None))
    # condensed distance for linkage
    iu = np.triu_indices(n, 1)
    Z = linkage(dist[iu], method="single")

    def leaves(node):
        if node.is_leaf():
            return [node.id]
        return leaves(node.get_left()) + leaves(node.get_right())

    order = leaves(to_tree(Z))

    def ivp(idx):
        d = np.clip(np.diag(S)[idx], 1e-16, None)
        w = 1.0 / d
        return w / w.sum()

    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            a, b = c[:half], c[half:]
            va = float(ivp(a) @ S[np.ix_(a, a)] @ ivp(a))
            vb = float(ivp(b) @ S[np.ix_(b, b)] @ ivp(b))
            alpha = 1.0 - va / (va + vb) if (va + vb) > 0 else 0.5
            w[a] *= alpha
            w[b] *= 1.0 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def hrp(rets, syms, rebal, window=126, cap=0.35, **kw):
    rows, when = [], []
    for d in rebal:
        S = _cov(rets[syms], d, window)
        if S is None:
            continue
        try:
            w = _hrp_weights(S)
        except Exception:
            continue
        rows.append(_cap(w, cap))
        when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=syms)


def black_litterman(rets, syms, rebal, window=126, cap=0.35, tau=0.05,
                    delta=2.0, views=None, **kw):
    """Black-Litterman with an inverse-volatility equilibrium prior.

    Without market caps in the panel, the neutral portfolio is taken to be
    inverse-volatility (stated in the paper, not hidden). Implied equilibrium
    returns are `pi = delta * Sigma @ w_eq`. Views, when supplied, are the
    operating point's own forecast with `P = I` and `Omega = diag(tau * Sigma)`;
    with no views BL returns the prior-implied optimum.
    """
    rows, when = [], []
    for d in rebal:
        S = _cov(rets[syms], d, window)
        if S is None:
            continue
        n = len(syms)
        vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        w_eq = (1.0 / vol) / (1.0 / vol).sum()
        pi = delta * S @ w_eq
        if views is not None and d in views.index:
            q = views.loc[d, syms].to_numpy(dtype=float)
            if np.all(np.isfinite(q)):
                Om = np.diag(np.clip(tau * np.diag(S), 1e-12, None))
                A = np.linalg.pinv(np.linalg.pinv(tau * S) + np.linalg.pinv(Om))
                pi = A @ (np.linalg.pinv(tau * S) @ pi + np.linalg.pinv(Om) @ q)
        w = np.linalg.pinv(delta * S + np.eye(n) * 1e-8) @ pi
        rows.append(_cap(w, cap))
        when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=syms)


REGISTRY = {
    "EqualWeight": equal_weight,
    "BuyAndHold": buy_and_hold,
    "MinVariance": min_variance,
    "RiskParity": risk_parity,
    "Momentum": momentum,
    "HRP": hrp,
    "BlackLitterman": black_litterman,
}


def run_all(rets, syms, rebal, **kw):
    out = {}
    for name, fn in REGISTRY.items():
        try:
            tg = fn(rets, syms, rebal, **kw)
        except Exception as e:  # a baseline that cannot be formed is reported, not faked
            print(f"    ! baseline {name}: {e}")
            continue
        if len(tg):
            out[name] = tg
    return out
