"""Walk-forward experiment: operating points, baselines, and meta-policies.

Protocol
--------
Rolling 3-year formation / 3-year evaluation, stepping three years at a time.
Every fold re-forms its universes by rule, refits the frozen forecasters, and
re-tunes the router on a validation slice carved from the END of its own
formation window. Nothing is selected on evaluation data.

    F1  form 2005-2007  eval 2008-2010     GFC, recovery
    F2  form 2008-2010  eval 2011-2013     Euro crisis / downgrade
    F3  form 2011-2013  eval 2014-2016     China/oil selloff
    F4  form 2014-2016  eval 2017-2019     Q4-2018
    F5  form 2017-2019  eval 2020-2022     COVID crash, 2022 bear
    F6  form 2020-2022  eval 2023-2026     AI rally

That is ~18.75 years of out-of-sample covering six drawdowns, against the 13-17
months of a single growth phase that the reviewers judged inadequate. The fold
boundaries are fixed by the calendar, not chosen -- a reviewer objected that "data
ranges and time windows change from experiment to experiment without
motivation", so one rule generates every window in the paper.

Outputs (analysis/):
    mode_returns.parquet      daily net returns, per fold/universe/mode/cost
    mode_targets.pkl          target weights, needed to charge the meta-policy
    characteristics.csv       regime descriptors per fold/universe
    router_selection.csv      the validation-chosen router config per cell
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B
import meta_policy as MP
import run_modes as RM
import sim as S
import universes as UNI

ROOT = Path(os.environ.get("ROUTER_ROOT", Path(__file__).resolve().parent))
ANALYSIS = ROOT / "analysis"
BPS_LADDER = [0.0, 10.0, 25.0, 50.0]
VAL_MONTHS = 12

US_FOLDS = [
    ("F1", "2005-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
    ("F2", "2008-01-01", "2010-12-31", "2011-01-01", "2013-12-31"),
    ("F3", "2011-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    ("F4", "2014-01-01", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("F5", "2017-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
    ("F6", "2020-01-01", "2022-12-31", "2023-01-01", "2026-09-30"),
]

# Only two folds are formable. Binance's USDT cross-section is 2 names over
# 2017-08..2018-12 and 5 over 2018..2019, which cannot support a cross-sectional
# portfolio study; the first window with a usable cross-section (15 names) is
# 2019-2020. Crypto evaluation therefore starts in 2021 and the 2018 bear is out
# of reach -- stated rather than papered over. What remains still spans the 2021
# mania, the LUNA/FTX collapse, the 2022 bear and the ETF era.
CRYPTO_FOLDS = [
    ("C1", "2019-01-01", "2020-12-31", "2021-01-01", "2022-12-31"),
    ("C2", "2021-01-01", "2022-12-31", "2023-01-01", "2026-09-20"),
]

# Two grids, searched on validation only, in two stages.
#
# Stage 1 fixes Tier 3. By Definition 3 the operating points share f_3 exactly --
# that is what makes them comparable interventions on f_2 rather than three
# different strategies -- so the allocator's hyperparameters must be chosen once
# per cell, not once per mode. They are selected on O1, the mode with no router,
# so the choice cannot be made to flatter the routing path.
#
# The range is set by the scale of the problem, not by inspection: the allocator
# receives an annualised forecast against an annualised covariance, so the
# variance-aversion that balances them is O(10^2) for equities, not O(1). The
# value of 2.0 carried by the submitted system left the mean term dominant, put
# the solution on a vertex of the capped simplex, and produced 30-48x annual
# turnover.
ALLOC_GRID = [
    dict(risk_aversion=lam, kappa=kap)
    for lam, kap in itertools.product([50.0, 100.0, 200.0, 400.0],
                                      [0.0, 0.02, 0.05])
]

# Stage 2 fixes Tier 2, with Tier 3 already frozen.
ROUTER_GRID = [
    dict(lam_risk=lr, top_k=k, tau=t, aggregator=agg)
    for lr, k, t, agg in itertools.product(
        [0.0, 0.5, 2.0], [2, 3], [0.7], ["precision", "softmax"])
]


def _val_split(train_start, train_end):
    te = pd.Timestamp(train_end)
    vs = te - pd.DateOffset(months=VAL_MONTHS) + pd.Timedelta(days=1)
    return str(vs.date()), str(te.date()), str((vs - pd.Timedelta(days=1)).date())


def _val_utility(px, syms, base, grid_point, mode, sel_bps):
    cfg = dict(base, **grid_point)
    try:
        out = RM.run_universe(px, syms, cfg)
    except Exception:
        return -np.inf
    if out is None:
        return -np.inf
    gr, tu, _ = S.simulate(out["targets"][mode], out["rets"], out["test_idx"])
    u = MP.mv_certainty_equivalent(S.net_series(gr, tu, out["test_idx"], sel_bps))
    return u if np.isfinite(u) else -np.inf


def select_config(px, syms, fold, sel_bps=25.0):
    """Two-stage validation selection: allocator on O1, then router on O2."""
    _, tr_s, tr_e, _, _ = fold
    val_s, val_e, sub_tr_e = _val_split(tr_s, tr_e)
    base = dict(train_start=tr_s, train_end=sub_tr_e,
                test_start=val_s, test_end=val_e)

    best_a, best_au = ALLOC_GRID[0], -np.inf
    for g in ALLOC_GRID:
        u = _val_utility(px, syms, base, g, "O1", sel_bps)
        if u > best_au:
            best_a, best_au = g, u

    base2 = dict(base, **best_a)
    best_r, best_ru = ROUTER_GRID[0], -np.inf
    for g in ROUTER_GRID:
        u = _val_utility(px, syms, base2, g, "O2", sel_bps)
        if u > best_ru:
            best_r, best_ru = g, u

    return {**best_a, **best_r}, float(best_au), float(best_ru)


def run(panel_path, folds, regimes, tag, top_liquid=200, vol_path=None,
        volume_is_notional=False, sizes=None):
    px = pd.read_csv(panel_path, index_col=0, parse_dates=True).sort_index()
    vol = (pd.read_csv(vol_path, index_col=0, parse_dates=True).sort_index()
           if vol_path else None)

    rows, targets_store, chars, selections = [], {}, [], []
    t0 = time.time()

    for fold in folds:
        fname, tr_s, tr_e, te_s, te_e = fold
        U = UNI.build_universes(px, vol, tr_s, tr_e, top_liquid=top_liquid,
                                volume_is_notional=volume_is_notional, sizes=sizes)
        if not U:
            print(f"[{fname}] no universes formable, skipping")
            continue
        print(f"[{fname}] form {tr_s}..{tr_e} eval {te_s}..{te_e} "
              f"-> {len(U)} universes")

        for uname, syms in U.items():
            g, au, ru = select_config(px, syms, fold)
            selections.append({"fold": fname, "universe": uname,
                               "val_utility_alloc": au, "val_utility_router": ru,
                               **g})
            cfg = dict(train_start=tr_s, train_end=tr_e,
                       test_start=te_s, test_end=te_e, **g)
            try:
                out = RM.run_universe(px, syms, cfg)
            except Exception as e:
                print(f"  ! {uname}: {e}")
                continue
            if out is None:
                print(f"  - {uname}: not formable")
                continue

            idx, rets = out["test_idx"], out["rets"]
            ch = UNI.characteristics(px, out["syms"], te_s, te_e)
            chars.append({"fold": fname, "universe": uname, **ch,
                          **{f"diag_{k}": v for k, v in out["diag"].items()
                             if not isinstance(v, dict)}})

            paths = dict(out["targets"])
            rebal = list(out["targets"]["O1"].index)
            paths.update(B.run_all(rets, out["syms"], rebal,
                                   window=out["cfg"].get("cov_window", 126),
                                   cap=out["cfg"]["cap"]))

            key = (fname, uname)
            targets_store[key] = {"targets": out["targets"], "syms": out["syms"],
                                  "test_idx": idx, "rets": rets[out["syms"]],
                                  "diag": out["diag"], "router": g}

            # Equal weight is the reference for the degeneracy check below.
            ew_g, ew_t, _ = S.simulate(paths["EqualWeight"], rets, idx) \
                if "EqualWeight" in paths else (None, None, None)
            ew_r = (S.net_series(ew_g, ew_t, idx, 0.0) if ew_g is not None else None)

            for name, tg in paths.items():
                gr, tu, _ = S.simulate(tg, rets, idx)
                # A large variance-aversion pushes any allocator toward 1/N, so a
                # Sharpe gain can be "became equal weight" rather than the method
                # working. Reporting the correlation makes that visible instead of
                # leaving it for a reviewer to suspect.
                corr_ew = (float(pd.Series(gr, index=idx).corr(ew_r))
                           if ew_r is not None else np.nan)
                for bps in BPS_LADDER:
                    r = S.net_series(gr, tu, idx, bps)
                    m = S.metrics(r)
                    rows.append({"fold": fname, "universe": uname, "strategy": name,
                                 "cost_bps": bps, "turnover_yr": float(tu.mean() * 252),
                                 "nnz": float((tg > 1e-8).sum(axis=1).mean()),
                                 "corr_equalweight": corr_ew, **m})
            print(f"  {uname:16s} n={out['diag']['n_assets']:3d} "
                  f"rebal={out['diag']['n_rebal']:3d} "
                  f"routed={out['diag']['router_n_routed']:.2f} "
                  f"({time.time()-t0:.0f}s)")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ANALYSIS / f"{tag}_strategy_metrics.csv", index=False)
    pd.DataFrame(chars).to_csv(ANALYSIS / f"{tag}_characteristics.csv", index=False)
    pd.DataFrame(selections).to_csv(ANALYSIS / f"{tag}_router_selection.csv", index=False)
    with open(ANALYSIS / f"{tag}_targets.pkl", "wb") as f:
        pickle.dump(targets_store, f)
    print(f"\nwrote {len(rows)} metric rows over {len(targets_store)} cells "
          f"in {time.time()-t0:.0f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=("us", "crypto"), default="us")
    a = ap.parse_args()
    if a.market == "us":
        run(ROOT / "data/panel_close_us_2005.csv", US_FOLDS, UNI.US_REGIMES, "us",
            top_liquid=200, vol_path=ROOT / "data/panel_volume_us_2005.csv")
    else:
        run(ROOT / "data/panel_close_crypto_2017.csv", CRYPTO_FOLDS,
            UNI.CRYPTO_REGIMES, "crypto", top_liquid=30,
            vol_path=ROOT / "data/panel_qvolume_crypto_2017.csv",
            volume_is_notional=True, sizes={"mega": 8, "basket": 10})


if __name__ == "__main__":
    main()
