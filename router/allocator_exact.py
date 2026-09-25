"""Exact solvers for the capped-simplex mean-variance decision (S4 fix).

The legacy `allocator.MeanVarianceAllocator` computes one Newton step from
the unconstrained optimum and then a (defective) projection. It does not solve

    w(mu) = argmax_w  mu'w - (lam/2) w'Sigma w    over  C = {w>=0, 1'w=1, w<=cap}

and violates the cap in most decisions (notes/S4_allocator_defect.md). This
module solves the stated problem exactly.

Method
------
1. Accelerated projected gradient (FISTA) on the strongly convex objective with
   the *exact* Euclidean projection onto C. The projection is a monotone 1-D
   root find in the shift theta: w_i = clip(v_i - theta, 0, cap), sum w = 1.
2. KKT polish: read the active set off the FISTA iterate, solve the
   equality-constrained QP on the free set in closed form, verify primal and
   dual feasibility. If the polished point passes KKT it is returned (machine
   precision); otherwise the FISTA iterate is returned (1e-9 level).

Both steps are pure numpy. n <= 30 in every universe, so cost is negligible.

Guarantees, asserted in `test_allocator_exact.py`:
    max(w) <= cap + 1e-9,  |sum(w) - 1| <= 1e-9,  w >= -1e-12
    w(mu + kappa*1) == w(mu) to 1e-8 for every kappa      (Proposition 1)
    w_lam(s*mu) == w_{lam/s}(mu) to 1e-8                   (Proposition 2)
    agrees with scipy SLSQP to 1e-5 on random instances
"""
import numpy as np

_CAP_TOL = 1e-9


# ----------------------------------------------------------------- projection
def project_capped_simplex(v, cap, total=1.0):
    """Exact Euclidean projection onto {w : 0 <= w <= cap, sum w = total}.

    w_i(theta) = clip(v_i - theta, 0, cap) is non-increasing in theta and
    piecewise linear with breakpoints at v_i and v_i - cap, so the root of
    sum_i w_i(theta) = total is found exactly by scanning the sorted
    breakpoints. Falls back to bisection only if the set is empty (cap*n <
    total), in which case the cap is relaxed to total/n.
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    if cap * n < total - 1e-12:
        cap = total / n
    bp = np.concatenate([v, v - cap])
    bp.sort()
    # sum w(theta) is decreasing; evaluate at breakpoints, locate the segment
    s = np.clip(v[None, :] - bp[:, None], 0.0, cap).sum(axis=1)
    # find last breakpoint with s >= total
    idx = np.nonzero(s >= total)[0]
    if len(idx) == 0:
        theta = bp[0]
    else:
        k = idx[-1]
        if k + 1 >= len(bp):
            theta = bp[k]
        else:
            t0, t1 = bp[k], bp[k + 1]
            s0, s1 = s[k], s[k + 1]
            if s0 == s1:
                theta = t0
            else:
                theta = t0 + (s0 - total) * (t1 - t0) / (s0 - s1)
    w = np.clip(v - theta, 0.0, cap)
    # exact renormalisation of the free block absorbs float error only
    d = total - w.sum()
    if abs(d) > 1e-13:
        free = (w > 0) & (w < cap)
        if free.any():
            w[free] += d / free.sum()
            w = np.clip(w, 0.0, cap)
    return w


def project_capped_box_budget(v, cap, gross):
    """Projection onto {0 <= w <= cap, sum w <= gross} (levered feasible set)."""
    w = np.clip(np.asarray(v, float), 0.0, cap)
    if w.sum() <= gross:
        return w
    return project_capped_simplex(v, cap, total=gross)


# ------------------------------------------------------------------- QP solve
def _fista(mu, Sigma, lam, proj, w0, iters=150, tol=1e-10):
    """Maximise mu'w - (lam/2) w'Sigma w over the set `proj` projects onto."""
    n = len(mu)
    L = lam * max(float(np.linalg.eigvalsh(Sigma)[-1]), 1e-12)
    step = 1.0 / L
    w = w0.copy()
    y = w.copy()
    t = 1.0
    for _ in range(iters):
        grad = mu - lam * (Sigma @ y)                # ascent direction
        w_new = proj(y + step * grad)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def _kkt_polish_simplex(mu, Sigma, lam, cap, w, tol=1e-7):
    """Closed-form solve on the active set read off `w`; verify KKT."""
    n = len(mu)
    Z = w <= tol                       # at zero
    C = w >= cap - tol                 # at cap
    F = ~(Z | C)
    if cap * n <= 1.0 + 1e-12:
        return None
    if not F.any():
        # every name at a bound: only feasible if counts match exactly
        wc = np.where(C, cap, 0.0)
        if abs(wc.sum() - 1.0) > 1e-9:
            return None
        # dual check with theta free: need exists theta with
        #   mu_i - lam (S wc)_i - theta <= 0 for Z, >= 0 for C
        g = mu - lam * (Sigma @ wc)
        if Z.any() and C.any() and g[Z].max() > g[C].min() + 1e-9:
            return None
        return wc
    SFF = Sigma[np.ix_(F, F)]
    mF = mu[F] - lam * (Sigma[np.ix_(F, C)] @ np.full(C.sum(), cap)) if C.any() else mu[F]
    try:
        Sinv_m = np.linalg.solve(SFF, mF)
        Sinv_1 = np.linalg.solve(SFF, np.ones(F.sum()))
    except np.linalg.LinAlgError:
        return None
    budget = 1.0 - cap * C.sum()
    # w_F = (1/lam) SFF^{-1} (mF - theta 1),  1'w_F = budget
    theta = (Sinv_m.sum() - lam * budget) / Sinv_1.sum()
    wF = (Sinv_m - theta * Sinv_1) / lam
    if (wF < -1e-10).any() or (wF > cap + 1e-10).any():
        return None
    out = np.where(C, cap, 0.0)
    out[F] = wF
    # dual feasibility: g_i = mu_i - lam (Sigma out)_i - theta
    g = mu - lam * (Sigma @ out) - theta
    if Z.any() and g[Z].max() > 1e-8:
        return None
    if C.any() and g[C].min() < -1e-8:
        return None
    out = np.clip(out, 0.0, cap)
    return out


class ExactMeanVariance:
    """Exact argmax of mu'w - (lam/2) w'Sigma w over the capped simplex.

    Drop-in for `allocator.MeanVarianceAllocator`: same constructor, same
    `__call__(mu, Sigma)`, `oracle`, `utility`.
    """

    name = "exact"

    def __init__(self, risk_aversion=2.0, cap=0.35):
        self.lam = float(risk_aversion)
        self.cap = float(cap)
        self.n_calls = 0
        self.n_polished = 0

    def _prep(self, mu, Sigma):
        mu = np.asarray(mu, dtype=float)
        n = mu.shape[0]
        S = np.asarray(Sigma, dtype=float)
        S = 0.5 * (S + S.T) + np.eye(n) * 1e-10
        cap = max(self.cap, 1.0 / n)
        return mu, S, cap

    def __call__(self, mu, Sigma):
        mu, S, cap = self._prep(mu, Sigma)
        n = len(mu)
        self.n_calls += 1
        if not np.all(np.isfinite(mu)):
            return np.full(n, 1.0 / n)
        proj = lambda v: project_capped_simplex(v, cap, 1.0)
        w0 = proj(mu / max(self.lam, 1e-12) / max(np.abs(mu).max(), 1e-12))
        w = w0
        for rounds, iters in enumerate((60, 150, 600, 3000)):
            w = _fista(mu, S, self.lam, proj, w, iters=iters)
            p = _kkt_polish_simplex(mu, S, self.lam, cap, w)
            if p is not None:
                self.n_polished += 1
                return p
        self.n_unpolished = getattr(self, "n_unpolished", 0) + 1
        return proj(w)

    def oracle(self, r, Sigma):
        return self(r, Sigma)

    def utility(self, w, r, Sigma):
        return float(w @ r - 0.5 * self.lam * w @ Sigma @ w)

    # The two methods below exist so that `du_cost.cost_aware_oracle` (WP5) can
    # run against the exact allocator. It was written against
    # `allocator.MeanVarianceAllocator`, which is a Newton step plus projection
    # and therefore exposes both pieces; this class solves the QP directly and
    # previously exposed neither, which is why the archived DU-TTT-C sweep only
    # ever ran on the legacy allocator.
    def unconstrained(self, mu, Sigma):
        """Unconstrained mean-variance optimum, same convention as the legacy
        allocator: Sigma^{-1} mu / lambda with the same 1e-8 ridge."""
        mu = np.asarray(mu, dtype=float)
        n = mu.shape[0]
        S = np.asarray(Sigma, dtype=float) + np.eye(n) * 1e-8
        try:
            return np.linalg.solve(S, mu) / self.lam
        except np.linalg.LinAlgError:
            return np.linalg.pinv(S) @ mu / self.lam

    def project(self, v):
        """Euclidean projection onto this allocator's feasible set."""
        v = np.asarray(v, dtype=float)
        return project_capped_simplex(v, max(self.cap, 1.0 / v.shape[0]), 1.0)


class ExactDiagRisk(ExactMeanVariance):
    """Decision map sees only diag(Sigma); utility still scored on full Sigma."""

    name = "exact_diag"

    def __call__(self, mu, Sigma):
        d = np.diag(np.asarray(Sigma, float)).copy()
        d[~np.isfinite(d)] = 1e-8
        d = np.maximum(d, 1e-8)
        return super().__call__(mu, np.diag(d))


class ExactLevered(ExactMeanVariance):
    """Feasible set {0 <= w <= cap, sum w <= gross}: level-SENSITIVE by design."""

    name = "exact_levered"

    def __init__(self, risk_aversion=2.0, cap=0.35, gross=1.0):
        super().__init__(risk_aversion, cap)
        self.gross = float(gross)

    def __call__(self, mu, Sigma):
        mu, S, cap = self._prep(mu, Sigma)
        n = len(mu)
        self.n_calls += 1
        if not np.all(np.isfinite(mu)):
            return np.zeros(n)
        proj = lambda v: project_capped_box_budget(v, self.cap, self.gross)
        w0 = proj(np.zeros(n))
        return proj(_fista(mu, S, self.lam, proj, w0))


def build(mode="simplex", impl="exact", risk_aversion=2.0, cap=0.35, gross=1.0):
    """Factory shared by experiment.run. `impl` in {"exact", "legacy"}."""
    if impl == "legacy":
        import allocator as LEG
        if mode == "levered":
            return LEG.LeveredAllocator(risk_aversion, cap, gross)
        if mode == "diag":
            return LEG.DiagRiskAllocator(risk_aversion, cap)
        return LEG.MeanVarianceAllocator(risk_aversion, cap)
    if mode == "levered":
        return ExactLevered(risk_aversion, cap, gross)
    if mode == "diag":
        return ExactDiagRisk(risk_aversion, cap)
    return ExactMeanVariance(risk_aversion, cap)


def jacobian(alloc, mu, Sigma, w=None, tol=1e-9):
    """Closed-form Jacobian dw/dmu of the exact capped-simplex map at mu.

    With F the free set at the solution,
        J_FF = (1/lam) [ S_FF^{-1} - S_FF^{-1} 1 1' S_FF^{-1} / (1' S_FF^{-1} 1) ],
    zero elsewhere. Symmetric PSD, J 1 = 0 (level invariance), piecewise
    constant in mu (depends on mu only through the active set).
    """
    mu = np.asarray(mu, float)
    n = len(mu)
    if w is None:
        w = alloc(mu, Sigma)
    cap = max(alloc.cap, 1.0 / n)
    F = (w > tol) & (w < cap - tol)
    J = np.zeros((n, n))
    k = int(F.sum())
    if k == 0:
        return J
    S = np.asarray(Sigma, float)
    SFF = S[np.ix_(F, F)] + np.eye(k) * 1e-10
    try:
        Sinv = np.linalg.inv(SFF)
    except np.linalg.LinAlgError:
        return J
    u = Sinv @ np.ones(k)
    JFF = (Sinv - np.outer(u, u) / max(u.sum(), 1e-12)) / alloc.lam
    J[np.ix_(F, F)] = JFF
    return J
