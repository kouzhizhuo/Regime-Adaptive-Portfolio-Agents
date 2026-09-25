"""Acceptance-test toolkit for the final router evaluation (T2b, 2026-09-23).

numpy / scipy / pandas only; runs under /usr/bin/python3 (3.9).  See README.md for
which PREREG rule (exp/T1/PREREG.md, B1-B3) each function serves.

Conventions
-----------
* "Utility" arrays are higher-is-better.  A router *beats* a comparator when the
  paired difference d_t = U_router,t - U_comp,t has positive mean.
* Panels: a pandas DataFrame indexed by block date with one column per cell
  (universe x fold) is reduced to one value per date by a NaN-aware cross-cell
  mean, and the bootstrap resamples *dates* (keeps cross-sectional dependence),
  exactly as T1's pilot does (t1_router_pilot.boot_p).
* One-sided bootstrap p-values use the (count + 1) / (B + 1) convention.
* All randomness goes through ``numpy.random.default_rng(seed)``.

References
----------
Politis & Romano (1994) JASA 89:1303-1313 (stationary bootstrap).
Holm (1979) Scand. J. Statist. 6:65-70.
Ledoit & Wolf (2008) J. Empirical Finance 15:850-859 (robust Sharpe-difference test).
Hansen (2005) JBES 23:365-380 (SPA).
Hansen, Lunde & Nason (2011) Econometrica 79:453-497 (MCS).
Bailey & Lopez de Prado (2012) J. Risk 15(2) (PSR); (2014) JPM 40(5):94-107 (DSR).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, Sequence[float]]
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- helpers
def pool_by_date(x: ArrayLike) -> np.ndarray:
    """Series/1-D -> float array; DataFrame (dates x cells) -> NaN-aware row mean.

    Rows that are all-NaN are dropped.
    """
    if isinstance(x, pd.DataFrame):
        v = x.mean(axis=1, skipna=True).to_numpy(dtype=float)
    else:
        v = np.asarray(x, dtype=float).ravel()
    return v[~np.isnan(v)]


def paired_diff(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Per-date pooled difference a - b, aligned on index/columns when pandas."""
    if isinstance(a, (pd.Series, pd.DataFrame)) and isinstance(b, (pd.Series, pd.DataFrame)):
        a, b = a.align(b, join="inner")
        return pool_by_date(a - b)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.ndim == 2:
        return pool_by_date(pd.DataFrame(a - b))
    d = a - b
    return d[~np.isnan(d)]


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) index matrix of Politis-Romano stationary-bootstrap draws.

    Block lengths are geometric with mean ``mean_block``; blocks wrap around.
    mean_block = 1 reduces to the iid bootstrap.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    new = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _boot_chunks(n, n_boot, mean_block, rng, max_cells=2_000_000):
    """Yield index blocks so that n_boot x n never exceeds ``max_cells`` ints."""
    step = max(1, max_cells // max(n, 1))
    done = 0
    while done < n_boot:
        k = min(step, n_boot - done)
        yield stationary_bootstrap_indices(n, k, mean_block, rng)
        done += k


def holm(pvals: Union[Sequence[float], Dict[str, float], pd.Series]):
    """Holm step-down adjusted p-values (same container type as the input)."""
    if isinstance(pvals, dict):
        s = pd.Series(pvals, dtype=float)
        return holm(s).to_dict()
    if isinstance(pvals, pd.Series):
        return pd.Series(holm(pvals.to_numpy()), index=pvals.index)
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def sharpe(r: ArrayLike, ddof: int = 1) -> float:
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def _hac_se_mean(x: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel HAC s.e. of the mean along the last axis (vectorised)."""
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    v = (xc * xc).mean(axis=-1)
    for j in range(1, lags + 1):
        v = v + 2 * (1 - j / (lags + 1)) * (xc[..., j:] * xc[..., :-j]).sum(axis=-1) / n
    return np.sqrt(np.maximum(v, 1e-300) / n)


# --------------------------------------------------------------------------- B1 / B3
@dataclass
class PairedResult:
    stat: str
    estimate: float
    p_value: float           # one-sided (alternative) or two-sided
    ci: tuple                # percentile CI of the estimate at level ``conf``
    n: int
    alternative: str
    n_boot: int
    mean_block: float
    extra: dict = field(default_factory=dict)


def paired_bootstrap(a: ArrayLike, b: Optional[ArrayLike] = None, *, stat: str = "mean",
                     alternative: str = "greater", n_boot: int = 5000, mean_block: float = 4.0,
                     conf: float = 0.90, seed: int = 0, periods_per_year: Optional[float] = None,
                     studentize: bool = False, hac_lags: Optional[int] = None) -> PairedResult:
    """Paired stationary bootstrap test of mean(a-b) or Sharpe(a)-Sharpe(b).

    stat="mean":  H0 E[a-b] <= 0 (alternative="greater").  ``a`` may already be the
                  difference series when ``b`` is None.  Panels are pooled by date first.
    stat="sharpe": H0 SR(a) <= SR(b); a, b are paired 1-D return series (no pooling).
    The null distribution is the bootstrap distribution of (theta* - theta_hat)
    (basic/centred bootstrap), matching T1's boot_p.  PREREG B1: mean_block=4,
    n_boot=5000, one-sided, then ``holm`` across the primary tests.

    studentize=True (stat="mean" only): compare t = mean/se_HAC with
    t* = (mean* - mean)/se_HAC(*) (Bartlett, ``hac_lags`` default ceil(mean_block)).
    With n of about 50 autocorrelated blocks, the unstudentized test over-rejects
    (see README, calibration table).  Report the studentized p alongside the
    PREREG p; do not replace it.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be greater|less|two-sided")
    rng = np.random.default_rng(seed)
    if stat == "mean":
        d = paired_diff(a, b) if b is not None else pool_by_date(a)
        n = len(d)
        est = float(d.mean())
        if studentize:
            lags = int(np.ceil(mean_block)) if hac_lags is None else int(hac_lags)
            se0 = float(_hac_se_mean(d, lags))
            tb = []
            for ix in _boot_chunks(n, n_boot, mean_block, rng):
                y = d[ix]
                tb.append((y.mean(axis=1) - est) / _hac_se_mean(y, lags))
            tb = np.concatenate(tb)
            t0 = est / se0
            if alternative == "greater":
                p = (np.sum(tb >= t0) + 1) / (n_boot + 1)
            elif alternative == "less":
                p = (np.sum(tb <= t0) + 1) / (n_boot + 1)
            else:
                p = (np.sum(np.abs(tb) >= abs(t0)) + 1) / (n_boot + 1)
            qlo, qhi = np.quantile(tb, [(1 - conf) / 2, 1 - (1 - conf) / 2])
            return PairedResult("mean_t", est, float(p), (est - qhi * se0, est - qlo * se0),
                                n, alternative, n_boot, mean_block,
                                extra={"t": t0, "se_hac": se0, "hac_lags": lags})
        boot = np.concatenate([d[ix].mean(axis=1)
                               for ix in _boot_chunks(n, n_boot, mean_block, rng)])
    elif stat == "sharpe":
        if b is None:
            raise ValueError("stat='sharpe' needs both a and b")
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            a, b = a.align(b, join="inner")
        x = np.c_[np.asarray(a, float), np.asarray(b, float)]
        x = x[~np.isnan(x).any(axis=1)]
        n = len(x)
        est = sharpe(x[:, 0]) - sharpe(x[:, 1])
        out = []
        for ix in _boot_chunks(n, n_boot, mean_block, rng):
            y = x[ix]                                   # (k, n, 2)
            sr = y.mean(axis=1) / y.std(axis=1, ddof=1)
            out.append(sr[:, 0] - sr[:, 1])
        boot = np.concatenate(out)
    else:
        raise ValueError("stat must be 'mean' or 'sharpe'")
    c = boot - est
    if alternative == "greater":
        p = (np.sum(c >= est) + 1) / (n_boot + 1)
    elif alternative == "less":
        p = (np.sum(c <= est) + 1) / (n_boot + 1)
    else:
        p = (np.sum(np.abs(c) >= abs(est)) + 1) / (n_boot + 1)
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    scale = np.sqrt(periods_per_year) if (stat == "sharpe" and periods_per_year) else 1.0
    return PairedResult(stat, est * scale, float(p), (float(lo * scale), float(hi * scale)),
                        n, alternative, n_boot, mean_block,
                        extra={"se_boot": float(boot.std(ddof=1) * scale)})


def b1_test(router: ArrayLike, comparators: Dict[str, ArrayLike], *, n_boot: int = 5000,
            mean_block: float = 4.0, seed: int = 0) -> pd.DataFrame:
    """PREREG B1/B3 helper: one-sided paired bootstrap of router vs each comparator.

    Returns a DataFrame with estimate, raw p and Holm-adjusted p across the given
    comparators.  (PREREG Holm is across *router tests*; to do that, collect the raw
    p of the primary comparison from each router and call ``holm`` on them.)
    """
    rows = {}
    for name, comp in comparators.items():
        r = paired_bootstrap(router, comp, n_boot=n_boot, mean_block=mean_block, seed=seed)
        rows[name] = dict(estimate=r.estimate, p=r.p_value, ci_lo=r.ci[0], ci_hi=r.ci[1], n=r.n)
    df = pd.DataFrame(rows).T
    df["p_holm"] = holm(df["p"].astype(float))
    return df


def b2_universe_wins(sr_router: Dict[str, float], sr_comp: Dict[str, float],
                     min_wins: int = 4) -> dict:
    """PREREG B2: Sharpe(router) > Sharpe(C*) in >= min_wins universe types."""
    keys = sorted(set(sr_router) & set(sr_comp))
    wins = [k for k in keys if sr_router[k] > sr_comp[k]]
    return dict(wins=len(wins), of=len(keys), passed=len(wins) >= min_wins, won=wins)


# --------------------------------------------------------------------------- Ledoit-Wolf 2008
def _sharpe_moments(x):
    """First and second raw moments of 2-column returns x, shape (n, 2)."""
    mu = x.mean(axis=-2)
    m2 = (x ** 2).mean(axis=-2)
    return mu, m2


def _lw_gradient(mu, m2):
    a, b = mu[..., 0], mu[..., 1]
    c, d = m2[..., 0], m2[..., 1]
    ga = c - a ** 2
    gb = d - b ** 2
    grad = np.stack([c / ga ** 1.5, -d / gb ** 1.5,
                     -0.5 * a / ga ** 1.5, 0.5 * b / gb ** 1.5], axis=-1)
    delta = a / np.sqrt(ga) - b / np.sqrt(gb)
    return delta, grad


def _hac_psi(y: np.ndarray, kernel: str = "parzen", bandwidth: Optional[float] = None):
    """HAC long-run covariance of the (n, 4) moment series y (demeaned inside)."""
    n = len(y)
    y = y - y.mean(axis=0)
    if bandwidth is None:
        bandwidth = 2.6614 * n ** 0.2 if kernel == "parzen" else 4 * (n / 100) ** (2 / 9)
    # einsum, not @: Accelerate-backed matmul on strided views raises spurious FP warnings
    psi = np.einsum("ti,tj->ij", y, y) / n
    L = int(np.floor(bandwidth))
    for j in range(1, max(L, 0) + 1):
        z = j / bandwidth
        if kernel == "parzen":
            w = 1 - 6 * z ** 2 + 6 * z ** 3 if z <= 0.5 else 2 * (1 - z) ** 3
        else:                                       # bartlett
            w = 1 - z
        g = np.einsum("ti,tj->ij", y[j:], y[:-j]) / n
        psi += w * (g + g.T)
    return psi * n / (n - 4)


def ledoit_wolf_sharpe(r1: ArrayLike, r2: ArrayLike, *, method: str = "boot",
                       n_boot: int = 4999, block: int = 5, seed: int = 0,
                       periods_per_year: Optional[float] = None) -> dict:
    """Ledoit & Wolf (2008) robust test of H0: SR1 == SR2 (two-sided).

    method="hac":  delta-method z test with a Parzen-kernel HAC covariance.
    method="boot": studentized circular block bootstrap (LW 2008, Sec. 3.2): HAC s.e.
                   for the observed statistic, block-sum s.e. for each resample;
                   recommended for small n.  p = (#{|d*-d|/s* >= |d|/s} + 1)/(B + 1).
    Returns dict(delta, se, p_value, method, n); delta is per period unless
    periods_per_year is given (then annualised, p unchanged).
    """
    if isinstance(r1, pd.Series) and isinstance(r2, pd.Series):
        r1, r2 = r1.align(r2, join="inner")
    x = np.c_[np.asarray(r1, float), np.asarray(r2, float)]
    x = x[~np.isnan(x).any(axis=1)]
    n = len(x)
    scale = np.sqrt(periods_per_year) if periods_per_year else 1.0
    if np.allclose(x[:, 0], x[:, 1], rtol=0, atol=1e-15):
        # identical series (e.g. the router *is* C*): the difference is exactly 0, not testable
        return dict(delta=0.0, se=0.0, p_value=1.0, method=method, n=n, degenerate=True)
    mu, m2 = _sharpe_moments(x)
    delta, grad = _lw_gradient(mu, m2)
    y = np.c_[x - mu, x ** 2 - m2]                  # moment series (n, 4)
    if method == "hac":
        psi = _hac_psi(y)
        se = float(np.sqrt(grad @ psi @ grad / n))
        p = float(2 * stats.norm.sf(abs(delta) / se))
    elif method == "boot":
        l = int(block)
        nb = n // l
        n_eff = nb * l
        def block_se(yy, gg):
            # yy (..., n_eff, 4) centred moments; block sums -> Psi
            s = yy[..., :n_eff, :].reshape(yy.shape[:-2] + (nb, l, 4)).sum(axis=-2) / np.sqrt(l)
            psi = np.einsum("...bi,...bj->...ij", s, s) / nb
            return np.sqrt(np.einsum("...i,...ij,...j->...", gg, psi, gg) / n_eff)
        se = float(np.sqrt(grad @ _hac_psi(y) @ grad / n))    # LW: HAC s.e. on the data,
        d_obs = abs(delta) / se                                # block s.e. inside the bootstrap
        rng = np.random.default_rng(seed)
        cnt = 0
        chunk = max(1, 2_000_000 // max(n_eff, 1))
        done = 0
        while done < n_boot:
            k = min(chunk, n_boot - done)
            st = rng.integers(0, n, size=(k, nb))
            ix = ((st[:, :, None] + np.arange(l)[None, None, :]) % n).reshape(k, n_eff)
            xb = x[ix]                                    # (k, n_eff, 2)
            mub = xb.mean(axis=1)
            m2b = (xb ** 2).mean(axis=1)
            db, gb = _lw_gradient(mub, m2b)
            yb = np.concatenate([xb - mub[:, None, :], xb ** 2 - m2b[:, None, :]], axis=-1)
            seb = block_se(yb, gb)
            cnt += int(np.sum(np.abs(db - delta) / seb >= d_obs))
            done += k
        p = (cnt + 1) / (n_boot + 1)
    else:
        raise ValueError("method must be 'hac' or 'boot'")
    scale = np.sqrt(periods_per_year) if periods_per_year else 1.0
    return dict(delta=float(delta * scale), se=float(se * scale), p_value=float(p),
                method=method, n=n)


# --------------------------------------------------------------------------- SPA
def _as_matrix(x) -> tuple:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(float), list(map(str, x.columns))
    if isinstance(x, dict):
        df = pd.DataFrame(x)
        return df.to_numpy(float), list(map(str, df.columns))
    a = np.asarray(x, float)
    if a.ndim == 1:
        a = a[:, None]
    return a, [f"m{i}" for i in range(a.shape[1])]


def spa_test(benchmark: ArrayLike, alternatives, *, n_boot: int = 5000,
             mean_block: float = 4.0, seed: int = 0) -> dict:
    """Hansen (2005) test for Superior Predictive Ability (higher utility = better).

    H0: no alternative beats the benchmark, max_k E[U_k - U_bench] <= 0.
    Router use: benchmark = C* (or EW); alternatives = every router configuration
    tried.  Rejection means some router beats the benchmark after accounting for
    the search over configurations.  ``benchmark`` has shape (n,); ``alternatives`` is
    an (n, K) array, a DataFrame or a dict of columns, sharing the same block dates.
    Returns the consistent (primary), lower and upper (= White reality check) p-values.
    """
    b = np.asarray(benchmark, float).ravel()
    A, names = _as_matrix(alternatives)
    if len(b) != len(A):
        raise ValueError("benchmark and alternatives must have the same length")
    ok = ~(np.isnan(b) | np.isnan(A).any(axis=1))
    d = A[ok] - b[ok, None]                          # (n, K) performance differentials
    n, K = d.shape
    dbar = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    bm = np.concatenate([d[ix].mean(axis=1)          # joint resampling of all K columns
                         for ix in _boot_chunks(n, n_boot, mean_block, rng)])   # (B, K)
    omega = np.sqrt(n) * bm.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t = np.sqrt(n) * dbar / omega
    T_obs = max(0.0, float(np.max(t)))
    thresh = -np.sqrt(2 * np.log(np.log(n))) if n > 15 else -np.inf
    g = {"lower": np.maximum(dbar, 0.0),
         "consistent": dbar * (t >= thresh),
         "upper": dbar}
    p = {}
    for key, gk in g.items():
        Tb = np.maximum(0.0, np.max(np.sqrt(n) * (bm - gk) / omega, axis=1))
        p[key] = float((np.sum(Tb >= T_obs) + 1) / (n_boot + 1)) if T_obs > 0 else 1.0
    return dict(stat=T_obs, p_consistent=p["consistent"], p_lower=p["lower"],
                p_upper=p["upper"], best=names[int(np.argmax(t))],
                t_stats=dict(zip(names, map(float, t))),
                mean_diff=dict(zip(names, map(float, dbar))), n=n)


# --------------------------------------------------------------------------- MCS
def model_confidence_set(losses, *, alpha: float = 0.10, method: str = "R",
                         n_boot: int = 5000, mean_block: float = 4.0, seed: int = 0,
                         higher_is_better: bool = False) -> dict:
    """Hansen, Lunde & Nason (2011) Model Confidence Set.

    ``losses``: an (n, M) array, DataFrame or dict of columns.  Pass utilities with
    higher_is_better=True and they are negated.  method="R" uses the range statistic
    T_R = max_ij |t_ij|; method="max" uses T_max = max_i t_i. (deviation from the
    set average).  Returns dict(included, pvalues, eliminated_order): ``pvalues``
    holds the MCS p-values and ``included`` lists the models whose p-value exceeds alpha.
    """
    L, names = _as_matrix(losses)
    L = L[~np.isnan(L).any(axis=1)]
    if higher_is_better:
        L = -L
    # exact duplicates (e.g. Deployed == O1) have zero-variance differences; test one copy and
    # give every copy that copy's p-value
    alias, keep = {}, []
    for j in range(L.shape[1]):
        k = next((i for i in keep if np.array_equal(L[:, i], L[:, j])), None)
        if k is None:
            keep.append(j)
        else:
            alias[names[j]] = names[k]
    if alias:
        res = model_confidence_set(pd.DataFrame(L[:, keep], columns=[names[i] for i in keep]), alpha=alpha,
                                   method=method, n_boot=n_boot, mean_block=mean_block, seed=seed)
        pv = {nm: res["pvalues"][alias.get(nm, nm)] for nm in names}
        return dict(included=[nm for nm in names if pv[nm] > alpha], pvalues=pv,
                    eliminated_order=res["eliminated_order"], alpha=alpha, method=method, n=res["n"],
                    duplicates=alias)
    n, M = L.shape
    rng = np.random.default_rng(seed)
    Lbar = L.mean(axis=0)
    Lb = np.concatenate([L[ix].mean(axis=1) for ix in _boot_chunks(n, n_boot, mean_block, rng)])
    alive = list(range(M))
    pvals: Dict[str, float] = {}
    order: List[str] = []
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        if method == "R":
            d = Lbar[a][:, None] - Lbar[a][None, :]                 # (m, m)
            db = Lb[:, a][:, :, None] - Lb[:, a][:, None, :]       # (B, m, m)
            var = ((db - d) ** 2).mean(axis=0)
            np.fill_diagonal(var, 1.0)
            tij = d / np.sqrt(var)
            T = np.max(np.abs(tij))
            Tb = np.max(np.abs(db - d) / np.sqrt(var), axis=(1, 2))
            worst = int(np.argmax(np.max(tij, axis=1)))
        elif method == "max":
            d = Lbar[a] - Lbar[a].mean()
            db = Lb[:, a] - Lb[:, a].mean(axis=1, keepdims=True)
            var = ((db - d) ** 2).mean(axis=0)
            ti = d / np.sqrt(var)
            T = np.max(ti)
            Tb = np.max((db - d) / np.sqrt(var), axis=1)
            worst = int(np.argmax(ti))
        else:
            raise ValueError("method must be 'R' or 'max'")
        p = float((np.sum(Tb >= T) + 1) / (n_boot + 1))
        running = max(running, p)
        k = alive.pop(worst)
        pvals[names[k]] = running
        order.append(names[k])
    pvals[names[alive[0]]] = 1.0
    included = [nm for nm in names if pvals[nm] > alpha]
    return dict(included=included, pvalues={nm: pvals[nm] for nm in names},
                eliminated_order=order, alpha=alpha, method=method, n=n)


# --------------------------------------------------------------------------- PSR / DSR
def probabilistic_sharpe(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0,
                         sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012): P[true SR > sr_benchmark].

    ``sr`` is per period (not annualised); ``kurt`` is non-excess (normal = 3).
    """
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(den)))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """SR_0 of Bailey & Lopez de Prado (2014), Eq. (2): the expected maximum Sharpe of
    n_trials unskilled strategies whose Sharpe estimates have variance var_trials."""
    if n_trials < 2:
        return 0.0
    N = float(n_trials)
    return float(np.sqrt(var_trials) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                                        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * np.e))))


def deflated_sharpe(returns: Optional[ArrayLike] = None, *, n_trials: int,
                    trial_sharpes: Optional[Sequence[float]] = None,
                    var_trials: Optional[float] = None, sr: Optional[float] = None,
                    n: Optional[int] = None, skew: Optional[float] = None,
                    kurt: Optional[float] = None) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Give either ``returns`` (per-period net returns of the reported strategy) or
    (sr, n, skew, kurt).  ``n_trials`` counts *every* configuration tried (routers x
    hyper-parameters x menus).  The dispersion of trial Sharpes is taken from
    ``trial_sharpes`` (per-period units, sample variance) or ``var_trials``.
    PREREG / T1 plan: a Sharpe claim needs DSR > 0.95.
    """
    if returns is not None:
        r = np.asarray(returns, float)
        r = r[~np.isnan(r)]
        n = len(r)
        sr = sharpe(r)
        skew = float(stats.skew(r, bias=False))
        kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    if sr is None or n is None:
        raise ValueError("need returns or (sr, n)")
    skew = 0.0 if skew is None else skew
    kurt = 3.0 if kurt is None else kurt
    if var_trials is None:
        if trial_sharpes is None:
            raise ValueError("need trial_sharpes or var_trials")
        var_trials = float(np.var(np.asarray(trial_sharpes, float), ddof=1))
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return dict(dsr=probabilistic_sharpe(sr, n, skew, kurt, sr0), sr0=sr0, sr=float(sr),
                psr0=probabilistic_sharpe(sr, n, skew, kurt, 0.0), n=int(n),
                skew=float(skew), kurt=float(kurt), n_trials=int(n_trials),
                var_trials=float(var_trials))


# --------------------------------------------------------------------------- PBO (CSCV)
def pbo_cscv(trials, *, n_splits: int = 16, metric: str = "sharpe", max_combinations: Optional[int] = None,
             seed: int = 0) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation
    (Bailey, Borwein, Lopez de Prado & Zhu 2017, J. Computational Finance 20(4)).

    ``trials``: (T, N) array or DataFrame of per-period performance (returns or per-block
    utilities) of the N configurations that were tried, on common dates.  The rows are
    cut into ``n_splits`` (even) contiguous blocks.  For every half/half split (IS = S/2
    blocks), the IS-best configuration n* is found by ``metric`` ("sharpe" or "mean").
    Its OOS relative rank is w = rank/(N+1), and lambda = logit(w).
    PBO = P[lambda <= 0], i.e. the IS winner lands at or below the OOS median.
    Also returns the OOS-on-IS degradation slope and P[OOS metric of n* < 0].
    ``max_combinations`` randomly subsamples the C(S, S/2) splits (seeded).
    """
    X, names = _as_matrix(trials)
    X = X[~np.isnan(X).any(axis=1)]
    T, N = X.shape
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be even and >= 2")
    if N < 2:
        raise ValueError("need at least 2 configurations")
    edges = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_splits)]
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if max_combinations is not None and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]

    # per-block sufficient statistics, so every split is O(N)
    cnt = np.array([len(b) for b in blocks], float)
    s1 = np.stack([X[b].sum(axis=0) for b in blocks])          # (S, N)
    s2 = np.stack([(X[b] ** 2).sum(axis=0) for b in blocks])

    def perf(idx):
        n = cnt[list(idx)].sum()
        m = s1[list(idx)].sum(axis=0) / n
        if metric == "mean":
            return m
        v = (s2[list(idx)].sum(axis=0) - n * m ** 2) / (n - 1)
        return m / np.sqrt(np.maximum(v, 1e-300))

    lam, is_best, oos_best = [], [], []
    allb = set(range(n_splits))
    for c in combos:
        pi = perf(c)
        po = perf(tuple(sorted(allb - set(c))))
        k = int(np.argmax(pi))
        rank = 1 + np.sum(po < po[k]) + 0.5 * (np.sum(po == po[k]) - 1)   # 1..N, ties averaged
        w = rank / (N + 1)
        lam.append(np.log(w / (1 - w)))
        is_best.append(pi[k])
        oos_best.append(po[k])
    lam = np.array(lam)
    is_best, oos_best = np.array(is_best), np.array(oos_best)
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if len(lam) > 2 and np.std(is_best) > 0 else float("nan")
    return dict(pbo=float(np.mean(lam <= 0)), lambda_median=float(np.median(lam)),
                degradation_slope=slope, p_oos_loss=float(np.mean(oos_best < 0)),
                n_combinations=len(lam), n_configs=N, n_periods=T, n_splits=n_splits, metric=metric)


def panel_from_long(df: pd.DataFrame, value_cols: Optional[Sequence[str]] = None,
                    date_col: str = "date", cell_col: str = "cell") -> Dict[str, pd.DataFrame]:
    """Long per-block file (one row per cell x date, one column per strategy/config) ->
    {name: DataFrame(date x cell)}, the input format of paired_bootstrap / final_report.

    Example: P = panel_from_long(pd.read_csv("w2_select_perblock_F_default.csv"));
    grid = pd.DataFrame({k: pool_by_date(v) for k, v in P.items() if k.startswith("W2a_")}).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    cols = value_cols or [c for c in d.columns if c not in (date_col, cell_col)]
    return {c: d.pivot(index=date_col, columns=cell_col, values=c).sort_index() for c in cols}


# --------------------------------------------------------------------------- trial ledger
LEDGER = Path(__file__).resolve().parent / "trial_ledger.csv"
# Snapshot taken when the coordinator approved the freeze (2026-09-23): N_test_seen = 194.
# final_report uses it by default so the approved TEST/holdout reads cannot inflate their own N.
LEDGER_FREEZE = Path(__file__).resolve().parent / "trial_ledger_freeze.csv"


def ledger_n(path: Union[str, Path] = LEDGER) -> dict:
    """N for the DSR from trial_ledger.csv (see build_ledger.py for definitions)."""
    L = pd.read_csv(path)
    L["duplicate_of"] = L["duplicate_of"].fillna("")
    c = L[L.role.isin(["router", "selector", "operating_point"]) & (L.duplicate_of == "")]
    w = c.config.apply(lambda s: json.loads(s).get("variants", 1) if isinstance(s, str) and s.startswith("{") else 1)
    seen = c.test_seen.astype(str).str.lower() == "true"
    H = L[L.role == "holdout"]
    touched = H[H.test_seen.astype(str).str.lower() == "true"].experiment.tolist()   # approved reads included
    return dict(N_test_seen=int(seen.sum()), N_total=int(w.sum()), rows=len(L),
                holdouts=H.experiment.tolist(), holdouts_touched=touched)


# --------------------------------------------------------------------------- final report
def _fmt(x, nd=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (bool, np.bool_)):
        return "pass" if x else "fail"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


_TEX_SUBS = (("p<0.05", "$p<0.05$"), (">= ", "$\\ge$ "), ("> 0.95", "$>0.95$"), ("< 0.2", "$<0.2$"),
             ("dU > 0", "$\\Delta U>0$"), ("dSR", "$\\Delta$SR"), ("dU", "$\\Delta U$"),
             ("C*", "$C^{\\ast}$"), ("SR >", "SR $>$"), ("%", "\\%"))


def _tex(s) -> str:
    """Make a table cell LaTeX-safe: relations in math mode, C*, Delta U, underscores."""
    s = str(s).replace("_", "\\_")
    for a, b in _TEX_SUBS:
        s = s.replace(a, b)
    return s


def final_report(utilities: Dict[str, ArrayLike], *, router: str = "router", cstar: str = "Cstar",
                 hedge: str = "Hedge", other_primary_p: Sequence[float] = (),
                 sr_by_universe: Optional[Dict[str, Dict[str, float]]] = None,
                 daily_returns: Optional[Dict[str, ArrayLike]] = None,
                 spa_alternatives=None, trials=None, trial_sharpes: Optional[Sequence[float]] = None,
                 n_trials: Optional[int] = None, ledger: Optional[Union[str, Path]] = None,
                 n_boot: int = 5000, seed: int = 0, periods_per_year: float = 252.0,
                 caption: str = "Acceptance tests for the frozen router.",
                 label: str = "tab:acceptance") -> dict:
    """One call that runs every acceptance test on a frozen design and returns a table.

    utilities : name -> per-block utility (Series by date, or date x cell DataFrame),
                TEST era.  Must contain ``router``, ``cstar`` and ``hedge``; every entry
                enters the MCS.
    other_primary_p : raw B1 p-values of the *other* router tests in the PREREG Holm family.
    sr_by_universe  : {"router": {type: SR}, "Cstar": {type: SR}} for B2.
    daily_returns   : name -> daily net returns (router and Cstar) for LW and DSR.
    spa_alternatives: per-block utilities of every router configuration tried (SPA);
                      defaults to the router alone.
    trials          : (dates x configs) per-block performance of the tried configs (PBO).
    trial_sharpes   : per-period Sharpes of the tried configs, in the units of
                      daily_returns (DSR variance).  n_trials defaults to N_test_seen
                      from the freeze snapshot trial_ledger_freeze.csv (194).
    Returns dict(table=DataFrame, latex=str, raw=dict).  B1-B3 follow exp/T1/PREREG.md.
    """
    U = utilities
    for k in (router, cstar, hedge):
        if k not in U:
            raise KeyError(f"utilities must contain {k!r}")
    raw, rows = {}, []

    b1 = paired_bootstrap(U[router], U[cstar], n_boot=n_boot, mean_block=4, seed=seed)
    fam = [b1.p_value] + list(other_primary_p)
    p_h = float(holm(fam)[0])
    b1s = paired_bootstrap(U[router], U[cstar], n_boot=n_boot, mean_block=4, seed=seed, studentize=True)
    d = paired_diff(U[router], U[cstar])
    acf1 = float(np.corrcoef(d[1:], d[:-1])[0, 1]) if len(d) > 2 else float("nan")
    raw.update(B1=b1, B1_studentized=b1s, acf1=acf1)
    rows.append(("B1 (PREREG)", "mean dU vs C*, block 4", b1.estimate, p_h,
                 f"Holm p<0.05 over {len(fam)}", p_h < 0.05))
    rows.append(("B1 studentized", "HAC-t bootstrap, block 4", b1s.estimate, b1s.p_value,
                 "report only", None))
    rows.append(("B1 diagnostic", "lag-1 ACF of dU", acf1, None, "report only", None))

    if sr_by_universe is not None:
        b2 = b2_universe_wins(sr_by_universe[router], sr_by_universe[cstar])
        raw["B2"] = b2
        rows.append(("B2 (PREREG)", "net SR > C* by universe type", f"{b2['wins']}/{b2['of']}", None,
                     f">= 4 of {b2['of']}", b2["passed"]))
    else:
        rows.append(("B2 (PREREG)", "net SR > C* by universe type", None, None, ">= 4 of 6", None))

    b3 = paired_bootstrap(U[router], U[hedge], n_boot=n_boot, mean_block=4, seed=seed)
    raw["B3"] = b3
    rows.append(("B3 (PREREG)", "mean dU vs Hedge", b3.estimate, b3.p_value, "dU > 0", b3.estimate > 0))

    if daily_returns is not None and router in daily_returns and cstar in daily_returns:
        lw = ledoit_wolf_sharpe(daily_returns[router], daily_returns[cstar], method="boot",
                                n_boot=min(n_boot, 4999), block=10, seed=seed,
                                periods_per_year=periods_per_year)
        raw["LW"] = lw
        rows.append(("Ledoit--Wolf", "dSR vs C* (ann.), two-sided", lw["delta"], lw["p_value"],
                     "report", None))

    alts = spa_alternatives if spa_alternatives is not None else {router: U[router]}
    if isinstance(alts, dict):
        alts = pd.DataFrame({k: pd.Series(pool_by_date(v) if isinstance(v, pd.DataFrame) else
                                          np.asarray(v, float)) for k, v in alts.items()})
    bench = pool_by_date(U[cstar]) if isinstance(U[cstar], pd.DataFrame) else np.asarray(U[cstar], float)
    A_ = alts.to_numpy(float) if isinstance(alts, pd.DataFrame) else np.asarray(alts, float)
    for L_ in (1, 4):
        spa = spa_test(bench, A_, n_boot=n_boot, mean_block=L_, seed=seed)
        raw[f"SPA_b{L_}"] = spa
        rows.append((f"SPA (block {L_})", f"{A_.shape[1] if A_.ndim > 1 else 1} configs vs C*",
                     spa["stat"], spa["p_consistent"], "p<0.05", spa["p_consistent"] < 0.05))

    mcs_in = pd.DataFrame({k: pd.Series(pool_by_date(v) if isinstance(v, pd.DataFrame) else
                                        np.asarray(v, float)) for k, v in U.items()})
    for L_ in (1, 4):
        mcs = model_confidence_set(mcs_in, alpha=0.10, higher_is_better=True, n_boot=n_boot,
                                   mean_block=L_, seed=seed)
        raw[f"MCS_b{L_}"] = mcs
        rows.append((f"MCS 90% (block {L_})", "router in set; size " + str(len(mcs["included"])),
                     mcs["pvalues"][router], None, "router retained", router in mcs["included"]))

    if ledger is None:
        ledger = LEDGER_FREEZE if LEDGER_FREEZE.exists() else LEDGER
    N = ledger_n(ledger) if n_trials is None and Path(ledger).exists() else None
    n_tr = n_trials if n_trials is not None else (N["N_test_seen"] if N else None)
    raw["ledger"] = N
    if daily_returns is not None and router in daily_returns and n_tr and trial_sharpes is not None:
        dsr = deflated_sharpe(daily_returns[router], n_trials=n_tr, trial_sharpes=trial_sharpes)
        raw["DSR"] = dsr
        rows.append(("Deflated Sharpe", f"N={n_tr} trials", dsr["dsr"], None, "> 0.95", dsr["dsr"] > 0.95))
        if N and n_trials is None:
            dsr2 = deflated_sharpe(daily_returns[router], n_trials=N["N_total"], trial_sharpes=trial_sharpes)
            raw["DSR_total"] = dsr2
            rows.append(("Deflated Sharpe", f"N={N['N_total']} (all tried)", dsr2["dsr"], None,
                         "sensitivity", None))
    else:
        rows.append(("Deflated Sharpe", f"N={n_tr}", None, None, "> 0.95",
                     None))

    if trials is not None:
        pbo = pbo_cscv(trials, n_splits=min(16, 2 * (len(np.asarray(trials)) // 8) or 2), metric="mean")
        raw["PBO"] = pbo
        rows.append(("PBO (CSCV)", f"{pbo['n_configs']} configs, S={pbo['n_splits']}", pbo["pbo"], None,
                     "< 0.2", pbo["pbo"] < 0.2))
    else:
        rows.append(("PBO (CSCV)", "--", None, None, "< 0.2", None))

    table = pd.DataFrame(rows, columns=["test", "statistic", "estimate", "p", "rule", "verdict"])
    body = "\n".join(
        f"{_tex(r.test)} & {_tex(r.statistic)} & {_fmt(r.estimate)} & {_fmt(r.p)} & {_tex(r.rule)} & "
        f"{'--' if r.verdict is None else ('pass' if r.verdict else 'fail')} \\\\"
        for r in table.itertuples())
    latex = (
        "\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{tabular}{@{}llrrll@{}}\n\\toprule\n"
        "Test & Statistic & Estimate & $p$ & Rule & Verdict \\\\\n\\midrule\n"
        + body +
        "\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\n{\\footnotesize\\raggedright "
        f"TEST era only; paired stationary bootstrap over block dates ({n_boot} resamples). "
        "B1--B3 as pre-registered; the studentized B1, SPA, MCS, DSR and PBO rows are "
        "supplementary controls. $\\Delta U$: per-block certainty-equivalent utility difference. "
        "$C^{\\ast}$: max(best fixed mode chosen on SELECT, EW).\\par}\n\\end{table}\n")
    return dict(table=table, latex=latex, raw=raw)


__all__ = ["pool_by_date", "paired_diff", "stationary_bootstrap_indices", "holm", "sharpe",
           "paired_bootstrap", "b1_test", "b2_universe_wins", "ledoit_wolf_sharpe",
           "spa_test", "model_confidence_set", "pbo_cscv", "ledger_n", "final_report", "panel_from_long", "probabilistic_sharpe",
           "expected_max_sharpe", "deflated_sharpe"]
