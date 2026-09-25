"""Meta-policies over the operating-point simplex, and the losses they minimise.

The submitted paper selected an operating mode by rolling-Sharpe argmax over a
one-quarter look-back. All three reviewers objected, and they pointed at the same
replacement. One reviewer: "the problem studied in this work appears to be an online
learning problem and a bandit formulation or an RL formulation would be a more
natural and defended alternative." Another: "instead of choosing one mode, maybe it
is worth exploring if a linear mixture should be used instead." A third reviewer asked why a
decision-focused framework was forgone.

This module implements that. Mode selection is prediction with expert advice over
`Delta(Pi)`, the loss is realised **net decision utility**, and the argmax rule
the paper shipped is recovered exactly as the degenerate `eta -> inf`, `alpha = 0`
corner of the same family -- so its empirical fragility becomes a prediction of
the theory rather than an unexplained negative result.

Guarantees (proved in the paper's appendix, standard results restated):

  Hedge with eta = sqrt(8 ln M / T):
      sum_t <p_t, l_t> - min_pi sum_t l_t(pi)  <=  sqrt(T ln M / 2)

  Fixed-share with switching rate alpha, against the best sequence of modes
  with at most k switches (Herbster & Warmuth 1998):
      R_T^k <= sqrt( T ( k ln M + k ln(T/k) + ln M ) / 2 )

  Argmax (eta -> inf, alpha = 0) admits no sublinear bound: Proposition 3 gives
  an instance on which it suffers regret linear in T.

Losses are scaled into [0, 1] by a *causal* running range: the scaling at time t
uses only utilities observed strictly before t. Using a global min/max would leak
the future into the normaliser, which is exactly the kind of quiet leak the
audit found elsewhere in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


# ------------------------------------------------------------------- losses
def mv_certainty_equivalent(r: pd.Series, lam: float = 2.0) -> float:
    """Annualised mean-variance certainty equivalent of a daily return block.

    Both terms are annualised. The submitted selector used `252*mean - Var/2` on
    daily returns, leaving the variance un-annualised at ~1e-4, so the risk term
    was numerically inert and the score was effectively raw return -- the likely
    mechanical cause of the eta-degeneracy reported in the draft.
    """
    if len(r) < 2:
        return np.nan
    return float(ANN * r.mean() - 0.5 * lam * (r.std(ddof=1) ** 2) * ANN)


def sortino_certainty_equivalent(r: pd.Series, lam: float = 2.0,
                                 mar_daily: float = 0.0) -> float:
    """Downside-deviation analogue, penalising only shortfall below MAR.

    A reviewer: the Sharpe ratio "essentially assumes the distribution of
    returns is symmetrical... other indicators, like the Sortino ratio, better
    handle skewed return distributions." This is that selector.
    """
    if len(r) < 2:
        return np.nan
    excess = r - mar_daily
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    return float(ANN * r.mean() - 0.5 * lam * (downside ** 2) * ANN)


def cvar_certainty_equivalent(r: pd.Series, lam: float = 2.0,
                              level: float = 0.95) -> float:
    """Mean minus a CVaR penalty: a coherent, tail-sensitive alternative."""
    if len(r) < 2:
        return np.nan
    q = np.quantile(r, 1.0 - level)
    tail = r[r <= q]
    cvar = float(-tail.mean()) if len(tail) else 0.0
    return float(ANN * r.mean() - lam * cvar * np.sqrt(ANN))


LOSSES = {
    "mv": mv_certainty_equivalent,
    "sortino": sortino_certainty_equivalent,
    "cvar": cvar_certainty_equivalent,
}


# ------------------------------------------------------------- the policies
def _causal_scale(u_hist: np.ndarray, u_now: np.ndarray):
    """Map utilities to losses in [0, 1] using only past observations.

    `u_hist` is (t, M) of utilities strictly before now. Before any history
    exists the block is uninformative and we emit 0.5 for every mode, which
    leaves the Hedge weights unchanged.
    """
    if u_hist.size == 0:
        return np.full_like(u_now, 0.5, dtype=float)
    lo, hi = np.nanmin(u_hist), np.nanmax(u_hist)
    lo, hi = min(lo, np.nanmin(u_now)), max(hi, np.nanmax(u_now))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(u_now, 0.5, dtype=float)
    return np.clip((hi - u_now) / (hi - lo), 0.0, 1.0)


def hedge(utilities: np.ndarray, eta: float | str | None = "adaptive",
          alpha: float = 0.0, argmax: bool = False):
    """Play a mixture over M modes; return the (T+1, M) matrix of played weights.

    Parameters
    ----------
    utilities : (T, M) realised utility of each mode over each scoring block.
                Row t is observed only *after* block t, so the weight played on
                block t may depend on rows < t only.
    eta       : learning rate.
                "adaptive" (default) uses the anytime schedule
                `eta_t = sqrt(8 ln M / t)`, which attains the same bound up to a
                constant without knowing the horizon (Cesa-Bianchi & Lugosi,
                Thm. 2.3). This matters for the paper's argument: the policy then
                carries *no* tuned constant at all, which is the cleanest
                possible answer to "the selection rule appears ad hoc".
                A float fixes eta; None uses the fixed-horizon sqrt(8 ln M / T).
    alpha     : fixed-share rate; 0 recovers plain Hedge.
    argmax    : play the vertex with the lowest cumulative loss (the submitted
                rule) instead of the exponential-weights mixture.

    Returns
    -------
    P : (T + 1, M) weights; row t is what is played on block t. The final row is
        the weight the policy would carry into the next block.
    """
    U = np.asarray(utilities, dtype=float)
    T, M = U.shape
    lnM = np.log(max(M, 2))
    adaptive = isinstance(eta, str) and eta == "adaptive"
    if eta is None:
        eta = np.sqrt(8.0 * lnM / max(T, 1))

    w = np.full(M, 1.0 / M)
    P = np.zeros((T + 1, M))
    Lcum = np.zeros(M)
    for t in range(T):
        if argmax:
            p = np.zeros(M)
            if t == 0:
                p[:] = 1.0 / M
            else:
                p[int(np.argmin(Lcum))] = 1.0
        else:
            p = w / w.sum()
        P[t] = p

        loss = _causal_scale(U[:t], U[t])
        loss = np.where(np.isfinite(loss), loss, 0.5)
        Lcum += loss
        eta_t = np.sqrt(8.0 * lnM / (t + 1)) if adaptive else eta
        w = w * np.exp(-eta_t * loss)
        w = w / w.sum()
        if alpha > 0:                      # fixed-share: leak mass to all modes
            w = (1.0 - alpha) * w + alpha / M
    P[T] = (np.eye(M)[int(np.argmin(Lcum))] if argmax else w / w.sum())
    return P


def regret(utilities: np.ndarray, P: np.ndarray):
    """Realised regret of the played mixture against the best fixed mode.

    Returned in utility units (not the [0,1] loss units the algorithm sees), so
    it is directly comparable with the tables.
    """
    U = np.asarray(utilities, dtype=float)
    T, M = U.shape
    played = np.nansum(U * P[:T], axis=1)
    best_fixed = np.nanmax(np.nansum(U, axis=0))
    return float(best_fixed - np.nansum(played))


def hedge_bound(T: int, M: int, k: int = 0):
    """The bound the paper states, in loss units, for T blocks and M modes.

    k = 0 gives the fixed-comparator Hedge bound; k > 0 the Herbster-Warmuth
    tracking bound against the best sequence with k switches.
    """
    if T <= 0:
        return np.nan
    if k <= 0:
        return float(np.sqrt(T * np.log(max(M, 2)) / 2.0))
    return float(np.sqrt(T * (k * np.log(max(M, 2)) + k * np.log(max(T / k, 1.0))
                              + np.log(max(M, 2))) / 2.0))


# --------------------------------------------------------------- scheduling
def scoring_blocks(idx: pd.DatetimeIndex, freq: str = "QE"):
    """Partition the test index into consecutive scoring blocks (quarters).

    Returns a list of (block_label, DatetimeIndex) in chronological order.
    """
    s = pd.Series(np.arange(len(idx)), index=idx)
    out = []
    for label, chunk in s.resample(freq):
        if len(chunk):
            out.append((label, idx[chunk.to_numpy()]))
    return out


def block_utilities(mode_returns: dict, blocks, loss_name="mv", lam=2.0):
    """(T, M) utility matrix: utility of each mode's *net* returns on each block."""
    fn = LOSSES[loss_name]
    names = list(mode_returns)
    U = np.full((len(blocks), len(names)), np.nan)
    for t, (_, days) in enumerate(blocks):
        for j, m in enumerate(names):
            U[t, j] = fn(mode_returns[m].reindex(days).dropna(), lam)
    return U, names


def mixture_targets(targets: dict, P: np.ndarray, blocks, names):
    """Deploy the mixture: at each rebalance, hold sum_pi p(pi) w_pi.

    The weight vector in force on a rebalance date is the one the policy played
    on the block that date falls in, which was fixed before the block began.
    """
    ref = targets[names[0]]
    rows, when = [], []
    for t, (_, days) in enumerate(blocks):
        dayset = set(days)
        p = P[t]
        for d in ref.index:
            if d in dayset:
                w = np.zeros(ref.shape[1])
                for j, m in enumerate(names):
                    if d in targets[m].index:
                        w = w + p[j] * targets[m].loc[d].to_numpy(dtype=float)
                tot = w.sum()
                rows.append(w / tot if tot > 0 else w)
                when.append(d)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(when), columns=ref.columns)
