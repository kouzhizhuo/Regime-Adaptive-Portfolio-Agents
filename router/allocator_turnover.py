"""Exact turnover-penalised mean-variance allocator over the capped simplex.

    w_t = argmax_{w in W}  mu'w - (lam/2) w'Sigma w - kappa ||w - w_prev||_1
    W   = { w >= 0, 1'w = 1, w <= cap }

Why this is needed
------------------
Section 4 of the submitted paper states this objective, but the allocator that
produced its numbers solved the problem *without* the turnover term. The
consequence is not cosmetic: with a weekly forecast annualised by 252/h the mean
term dwarfs the risk term, the solution sits on a vertex of the capped simplex,
and the vertex changes most weeks -- 32 to 48x annual turnover, which at 25 bps
costs 8-12% a year.

A reviewer asked why the l1 distance between consecutive weights appears at
all, and why not l2. Under proportional transaction costs the l1 term *is* the
cost; it is not a regulariser chosen for tractability, and an l2 penalty would
correspond to no cost model anyone charges.

Solution method
---------------
Accelerated proximal gradient (FISTA) on the smooth part, with an **exact** prox
for the nonsmooth part. The prox

    min_w  (1/2)||w - v||^2 + kappa ||w - c||_1   s.t.  0 <= w <= cap, 1'w = 1

separates across coordinates given the multiplier theta on the sum constraint:

    w_i(theta) = clip( soft(v_i - theta; c_i, kappa), 0, cap ),
    soft(u; c, k) = c + sign(u - c) * max(|u - c| - k, 0)

`soft` is non-decreasing in u, so w_i(theta) is non-increasing in theta and
sum_i w_i(theta) is monotone; theta is recovered by bisection to machine
precision. This is the same structure as the projection in `allocator_exact`,
with soft-thresholding toward the previous holding replacing plain clipping, and
it reduces to that projection exactly when kappa = 0.

Pure numpy, O(n log(1/eps)) per prox, so it scales to the 30-120 name universes
that a 15-name basket cannot represent. `test_allocator_turnover.py` checks it
against SLSQP on random instances and checks the kappa = 0 reduction.
"""
from __future__ import annotations

import numpy as np

from allocator_exact import ExactMeanVariance, project_capped_simplex

_TOL = 1e-10


def _soft(u, c, k):
    """Soft-threshold `u` toward centre `c` with radius `k`."""
    d = u - c
    return c + np.sign(d) * np.maximum(np.abs(d) - k, 0.0)


def prox_turnover_simplex(v, c, kappa, cap, total=1.0, iters=100):
    """Exact prox of kappa*||.-c||_1 + indicator{0<=w<=cap, 1'w=total}."""
    v = np.asarray(v, dtype=float)
    c = np.asarray(c, dtype=float)
    n = v.shape[0]
    cap = max(cap, total / n)
    if kappa <= 0:
        return project_capped_simplex(v, cap, total)

    def w_of(theta):
        return np.clip(_soft(v - theta, c, kappa), 0.0, cap)

    # Bracket: theta large enough drives every coordinate to 0, small enough to cap.
    lo = float(np.min(v - c) - kappa - cap - 1.0)
    hi = float(np.max(v - c) + kappa + 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if w_of(mid).sum() > total:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    w = w_of(0.5 * (lo + hi))
    s = w.sum()
    # Residual normalisation guards against a degenerate bracket only; the
    # bisection above is already at 1e-10.
    return w if abs(s - total) < 1e-9 or s <= 0 else w * (total / s)


def prox_turnover_simplex_exact(v, c, kappa, cap, total=1.0):
    """Same prox as `prox_turnover_simplex`, solved by breakpoint search.

    T4 addition (speed only). sum_i w_i(theta) is piecewise linear and
    non-increasing in theta, with kinks where v_i - theta crosses one of
    {c_i - kappa, c_i + kappa, -kappa, kappa, cap - kappa, cap + kappa}. Evaluating
    the sum at every kink brackets the root between two adjacent kinks, where
    the sum is linear, so theta is recovered exactly by interpolation instead of
    by ~36 bisection steps. `test_t4_invariants.py` checks agreement with the
    bisection version to 1e-9.
    """
    v = np.asarray(v, dtype=float)
    c = np.asarray(c, dtype=float)
    n = v.shape[0]
    cap = max(cap, total / n)
    if kappa <= 0:
        return project_capped_simplex(v, cap, total)
    k = kappa
    offs = np.stack([c - k, c + k, np.full(n, -k), np.full(n, k),
                     np.full(n, cap - k), np.full(n, cap + k)], axis=1)
    th = np.unique((v[:, None] - offs).ravel())
    th = np.concatenate([[th[0] - 1.0], th, [th[-1] + 1.0]])
    U = v[None, :] - th[:, None]
    D = U - c[None, :]
    W = np.clip(c[None, :] + np.sign(D) * np.maximum(np.abs(D) - k, 0.0), 0.0, cap)
    tot = W.sum(axis=1)                        # non-increasing along th
    # first index where the sum drops to <= total
    j = int(np.searchsorted(-tot, -total, side="left"))
    if j <= 0:
        w = W[0]
    elif j >= len(th):
        w = W[-1]
    else:
        s0, s1 = tot[j - 1], tot[j]
        a = 0.0 if s0 == s1 else (s0 - total) / (s0 - s1)
        w = W[j - 1] + a * (W[j] - W[j - 1])   # linear between adjacent kinks
    s = w.sum()
    return w if abs(s - total) < 1e-9 or s <= 0 else w * (total / s)


PROX = prox_turnover_simplex_exact


class TurnoverMeanVariance(ExactMeanVariance):
    """ExactMeanVariance plus an exact l1 turnover penalty against `w_prev`."""

    name = "turnover"

    def __init__(self, risk_aversion=2.0, cap=0.35, kappa=0.0, iters=400):
        super().__init__(risk_aversion, cap)
        self.kappa = float(kappa)
        self.iters = int(iters)

    def __call__(self, mu, Sigma, w_prev=None):
        if self.kappa <= 0 or w_prev is None:
            return super().__call__(mu, Sigma)
        mu, S, cap = self._prep(mu, Sigma)
        n = len(mu)
        wp = np.asarray(w_prev, dtype=float)
        if wp.shape != (n,) or not np.all(np.isfinite(wp)) or not np.all(np.isfinite(mu)):
            return super().__call__(mu, Sigma)

        lam, kap = self.lam, self.kappa
        L = lam * float(np.linalg.eigvalsh(S)[-1]) + 1e-12   # Lipschitz const of grad f
        step = 1.0 / max(L, 1e-12)

        w = y = project_capped_simplex(wp, cap, 1.0)
        t = 1.0
        for _ in range(self.iters):
            grad = lam * (S @ y) - mu                        # grad of -mu'w + (lam/2)w'Sw
            w_new = PROX(y - step * grad, wp, kap * step, cap)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            if np.abs(w_new - w).max() < 1e-11:
                # T4 fix: two equal consecutive iterates are NOT optimality under
                # momentum -- the prox snaps coordinates exactly to w_prev, 0 or
                # cap, so w_new == w can occur at y != w away from the optimum
                # (tests/fixtures/fista_stall_*.npz). Stop only if the proximal-
                # gradient residual AT w vanishes; otherwise restart momentum.
                r = PROX(w_new - step * (lam * (S @ w_new) - mu), wp, kap * step, cap)
                if np.abs(r - w_new).max() < 1e-11:
                    w = w_new
                    break
                w, y, t = w_new, w_new, 1.0
                continue
            y = w_new + ((t - 1.0) / t_new) * (w_new - w)
            w, t = w_new, t_new
        self.n_calls += 1
        return w

    def objective(self, w, mu, Sigma, w_prev):
        mu, S, _ = self._prep(mu, Sigma)
        return float(mu @ w - 0.5 * self.lam * w @ S @ w
                     - self.kappa * np.abs(w - np.asarray(w_prev, float)).sum())


def build(risk_aversion=2.0, cap=0.35, kappa=0.0):
    return (TurnoverMeanVariance(risk_aversion, cap, kappa) if kappa > 0
            else ExactMeanVariance(risk_aversion, cap))
