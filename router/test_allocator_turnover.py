"""Correctness checks for the turnover-penalised allocator.

The allocator is the component the paper's Tier-3 claim rests on, so it is
checked against an independent solver rather than trusted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "iclr26/code_du"))

import allocator_turnover as AT
from allocator_exact import project_capped_simplex


def _rand(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    S = A @ A.T / n + np.eye(n) * 0.05
    mu = rng.normal(scale=0.3, size=n)
    wp = project_capped_simplex(rng.random(n), 0.35, 1.0)
    return mu, S, wp


def _slsqp(mu, S, wp, lam, kap, cap):
    n = len(mu)
    obj = lambda w: -(mu @ w - 0.5 * lam * w @ S @ w - kap * np.abs(w - wp).sum())
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    r = minimize(obj, wp.copy(), bounds=[(0.0, cap)] * n, constraints=cons,
                 method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return r.x


@pytest.mark.parametrize("n", [5, 15, 40])
@pytest.mark.parametrize("kap", [0.01, 0.05, 0.2])
def test_matches_slsqp(n, kap):
    """FISTA + exact prox reaches at least the objective SLSQP reaches."""
    lam, cap = 50.0, 0.35
    for seed in range(4):
        mu, S, wp = _rand(n, seed)
        a = AT.TurnoverMeanVariance(lam, cap, kap)
        w = a(mu, S, w_prev=wp)
        ref = _slsqp(mu, S, wp, lam, kap, cap)
        assert a.objective(w, mu, S, wp) >= a.objective(ref, mu, S, wp) - 1e-6


@pytest.mark.parametrize("n", [5, 20, 60])
def test_feasible(n):
    lam, cap, kap = 100.0, 0.20, 0.05
    mu, S, wp = _rand(n, 7)
    w = AT.TurnoverMeanVariance(lam, cap, kap)(mu, S, w_prev=wp)
    assert w.min() >= -1e-9
    assert w.max() <= max(cap, 1.0 / n) + 1e-8
    assert abs(w.sum() - 1.0) < 1e-8


def test_kappa_zero_reduces_to_projection():
    """kappa = 0 must reproduce the unpenalised exact allocator bit for bit."""
    mu, S, wp = _rand(12, 3)
    a0 = AT.build(50.0, 0.35, 0.0)
    a1 = AT.TurnoverMeanVariance(50.0, 0.35, 0.0)
    assert np.allclose(a0(mu, S), a1(mu, S, w_prev=wp), atol=1e-12)


def test_prox_reduces_to_projection():
    rng = np.random.default_rng(0)
    v, c = rng.normal(size=20), rng.random(20)
    assert np.allclose(AT.prox_turnover_simplex(v, c, 0.0, 0.3),
                       project_capped_simplex(v, 0.3, 1.0), atol=1e-12)


def test_turnover_is_monotone_in_kappa():
    """A larger penalty must not increase distance from the previous holding."""
    mu, S, wp = _rand(25, 11)
    prev = None
    for kap in [0.0, 0.01, 0.05, 0.2, 1.0]:
        w = AT.TurnoverMeanVariance(80.0, 0.3, kap)(mu, S, w_prev=wp) if kap > 0 \
            else AT.build(80.0, 0.3, 0.0)(mu, S)
        d = np.abs(w - wp).sum()
        if prev is not None:
            assert d <= prev + 1e-6
        prev = d
