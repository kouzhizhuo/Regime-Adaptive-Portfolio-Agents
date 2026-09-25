"""Walk-forward runner producing one target-weight schedule per operating point.

All three operating points share one allocator, one risk model and one rebalance
calendar, so they are comparable interventions on `f2` in the sense of
Definition 1 -- not three unrelated portfolios. Only the map from features to a
predicted mean differs:

    O1 optimizer-only   frozen ridge on the base Tier-1 panel
    O2 routed consensus four experts -> utility-aware Top-K router -> aggregation
    O3 alpha-augmented  frozen ridge on the augmented Tier-1 panel

The output is deliberately a *schedule of target weights*, not a return series,
so that `sim.simulate` can charge the meta-policy for its own mode switches.

Causality contract (inherited from code_du/causal_runner_0908):
    features at t      use prices <= t
    ridge              fitted on the train window with the unmatured tail purged
    Sigma at t         from returns strictly before t
    weights set at t   earn the return through close t+1
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# numpy's SIMD matmul loop on this build raises divide-by-zero / overflow /
# invalid on inputs that are entirely finite and well scaled -- the unused
# vector lanes set the FP status flags. Verified harmless here: for the ridge
# normal equations the `np.linalg.solve` result agrees with an independent
# `lstsq` to 3.2e-17 at condition number 56. Scoped to this exact message so a
# genuine numerical warning anywhere else still surfaces.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

ROOT = Path(os.environ.get("ROUTER_ROOT", Path(__file__).resolve().parent))
CODE_DU = Path(__file__).resolve().parent  # T1: vendored experiment.py/allocator_exact.py
if str(CODE_DU) not in sys.path:
    sys.path.insert(0, str(CODE_DU))

import experiment as E            # noqa: E402  FrozenRidge, make_features

import allocator_turnover as AT   # noqa: E402  exact MV + l1 turnover
import modes as M                 # noqa: E402

ANN = 252.0
MODES = ["O1", "O2", "O3"]

DEFAULT_CFG = dict(
    horizon=5,                 # label maturation, trading days
    rebal="W-FRI",
    cov_window=126,
    risk_aversion=2.0,
    kappa=0.0,                 # l1 turnover penalty; selected on validation
    cap=None,                  # None -> clip(5/N, 0.05, 0.35), scaled to breadth
    ridge_alpha=10.0,
    # router
    lam_risk=0.5,              # 1/return; re-tuned on validation, never inherited
    gam_cost=1.0,              # dimensionless
    top_k=3,
    tau=0.7,
    min_conf=0.05,
    aggregator="precision",
    experts=None,              # None -> M.EXPERT_NAMES (the 4-expert panel)
    min_names=5,
)


def _fit_ridge(feats, p, syms, cfg):
    """Frozen ridge, fitted on the train window with forward targets purged.

    Truncating the price panel before forming forward targets is what makes the
    last `h` training rows carry missing labels and be dropped, so no training
    target can reach past `train_end`.
    """
    fit_px = p.loc[:cfg["train_end"]]
    return E.FrozenRidge(alpha=cfg["ridge_alpha"]).fit(
        feats, fit_px, syms, cfg["train_start"], cfg["train_end"], cfg["horizon"])


def run_universe(px: pd.DataFrame, syms, cfg: dict):
    """Return dict with per-mode target-weight frames and shared context.

    Keys: targets {mode: DataFrame(rebal_dates x syms)}, rets, test_idx, syms,
    diag (router diagnostics).
    """
    cfg = {**DEFAULT_CFG, **cfg}
    h = int(cfg["horizon"])

    span = px.loc[cfg["train_start"]:cfg["test_end"]]
    cand = [c for c in syms if c in span.columns]
    p = span[cand].dropna(axis=1, how="any")
    syms = list(p.columns)
    if len(syms) < cfg["min_names"]:
        return None

    rets = p.pct_change()
    test_idx = p.loc[cfg["test_start"]:cfg["test_end"]].index
    if len(test_idx) < 60:
        return None

    rebal = pd.Series(1, index=test_idx).resample(cfg["rebal"]).last().index
    rebal = [d for d in rebal if d in set(test_idx)]
    if test_idx[0] not in rebal:
        rebal = [test_idx[0]] + rebal

    feats_base = E.build_panel_features(p, syms)
    feats_aug = M.augmented_features(p, syms, feats_base)
    ridge_base = _fit_ridge(feats_base, p, syms, cfg)
    ridge_aug = _fit_ridge(feats_aug, p, syms, cfg)

    experts = M.build_expert_panel(p, syms)
    vol21 = rets.rolling(21).std()

    cov_w = int(cfg["cov_window"])
    # A fixed 0.35 cap binds on 3 names out of 15 but never binds at N=50, so it
    # would silently mean different things across universes. Scaling it with
    # breadth keeps "how concentrated may the book get" constant.
    cap = cfg["cap"] if cfg.get("cap") else float(np.clip(5.0 / len(syms), 0.05, 0.35))
    cfg = dict(cfg, cap=cap)
    alloc = AT.build(cfg["risk_aversion"], cap, cfg["kappa"])

    rows = {m: {} for m in MODES}
    omega_log, nroute_log = [], []
    # Holdings each mode carries into the current rebalance, after drifting with
    # prices since it last traded. The turnover term is charged against these,
    # not against the previous target, matching how the simulator measures it.
    held = {m: None for m in MODES}
    last_day = {m: None for m in MODES}

    for day in rebal:
        hist = rets.loc[:day].iloc[-cov_w:-1]           # strictly before `day`
        hist = hist.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if len(hist) < 40:
            continue
        Sigma = np.cov(hist.to_numpy(), rowvar=False) * ANN

        # --- feature rows at `day`, base and augmented
        rb, ra, ok = [], [], True
        for s in syms:
            fb = feats_base[s].loc[:day]
            fa = feats_aug[s].loc[:day]
            if not len(fb) or not len(fa):
                ok = False
                break
            fb, fa = fb.iloc[-1], fa.iloc[-1]
            if fb.isna().any() or fa.isna().any():
                ok = False
                break
            rb.append(fb.to_numpy())
            ra.append(fa.to_numpy())
        if not ok:
            continue

        mu = {}
        mu["O1"] = np.array([ridge_base.predict_row(x) for x in rb])
        mu["O3"] = np.array([ridge_aug.predict_row(x) for x in ra])

        # --- O2: experts -> router -> aggregation
        sig_h = vol21.loc[day, syms].to_numpy(dtype=float) * np.sqrt(h)
        sig_h = np.where(np.isfinite(sig_h) & (sig_h > 0), sig_h, 0.02 * np.sqrt(h))
        scores = {k: experts[k].loc[day, syms].to_numpy(dtype=float) for k in experts}
        e_mu, e_var, e_conf, e_cost, names = M.expert_messages(
            scores, sig_h, h, names=cfg.get("experts"))
        mu_agg, _var_agg, omega = M.route(
            e_mu, e_var, e_conf, e_cost,
            cfg["lam_risk"], cfg["gam_cost"], cfg["top_k"], cfg["tau"],
            cfg["min_conf"], cfg["aggregator"])
        mu["O2"] = mu_agg
        omega_log.append(pd.Series(omega.mean(axis=1), index=names, name=day))
        nroute_log.append(float((omega > 1e-9).sum(axis=0).mean()))

        px_now = p.loc[day, syms].to_numpy(dtype=float)
        for m in MODES:
            v = mu[m]
            if not np.all(np.isfinite(v)):
                continue
            w_prev = None
            if held[m] is not None and last_day[m] is not None:
                factor = px_now / p.loc[last_day[m], syms].to_numpy(dtype=float)
                d = held[m] * factor
                tot = d.sum()
                w_prev = d / tot if tot > 0 else None
            w = alloc(v * (ANN / h), Sigma, w_prev=w_prev) if cfg["kappa"] > 0 \
                else alloc(v * (ANN / h), Sigma)
            rows[m][day] = w
            held[m], last_day[m] = w, day

    targets = {}
    for m in MODES:
        if not rows[m]:
            return None
        targets[m] = pd.DataFrame.from_dict(rows[m], orient="index", columns=syms
                                            ).sort_index()

    diag = {
        "n_assets": len(syms),
        "n_rebal": int(len(targets["O1"])),
        "router_expert_weight": (pd.concat(omega_log, axis=1).mean(axis=1).to_dict()
                                 if omega_log else {}),
        "router_n_routed": float(np.mean(nroute_log)) if nroute_log else np.nan,
        "n_experts": len(cfg.get("experts") or M.EXPERT_NAMES),
    }
    return {"targets": targets, "rets": rets, "test_idx": test_idx,
            "syms": syms, "diag": diag, "cfg": cfg}
