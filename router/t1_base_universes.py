"""T1 re-test of the base paper's headline on the corrected engine.

Claim under test (base paper, abstract / Sec. 6.2): the routing signal
s = dispersion x (1 - mean corr), measured on pre-formation data, orders the
optimizer-only (O1) and routed-consensus (O2) operating points around 0.1:
s < 0.1 -> SR(O1) > SR(O2), s > 0.1 -> SR(O2) > SR(O1); on BigTech-6,
Consumer-20, High-Vol-15, Defensive-15 plus held-out Healthcare-10.

Protocol, as close to the base paper as the corrected engine permits:
  formation/train 2021-01-01 .. 2023-06-30, validation 2023-07-01 .. 2023-12-31,
  test 2024-01-01 .. 2025-01-31, weekly Friday rebalance, long-only, capped.
  Hyperparameters: the engine's two-stage validation search (allocator on O1,
  then router on O2), selected on the validation slice only; the final run refits
  the frozen ridge on 2021-01-01 .. 2023-12-31 (train + validation).
  Costs: 0 bps (the base paper effectively charged none) and 25 bps.
The prediction for each universe is recorded from formation-window data before
the test metrics are computed; nothing is tuned on the test window.

Usage:  python3 t1_base_universes.py [--window 2024] [--tag base]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import main_experiment as ME   # noqa: E402  (grids, _val_utility)
import meta_policy as MP       # noqa: E402
import run_modes as RM         # noqa: E402
import sim as S                # noqa: E402
import baselines as B          # noqa: E402
import universes as UNI        # noqa: E402

DATA = Path(os.environ.get("ROUTER_DATA", Path(__file__).resolve().parent / "data"))
OUT = HERE / "results"

FIXED = {
    "BigTech-6": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META"],
    "Consumer-20": ["WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
                    "HD", "LOW", "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ",
                    "ORLY", "AZO"],
    # nips_revision_experiments.py list (utilities + staples, matches the paper's
    # description). K (Kellanova) is absent from the Yahoo panel -> 14 names.
    "Defensive-15": ["JNJ", "PG", "KO", "PEP", "CL", "GIS", "K", "SJM", "MKC", "HRL",
                     "DUK", "SO", "D", "AEP", "NEE"],
    "Healthcare-10": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR",
                      "BMY"],
}
THRESH = 0.1


def highvol(px, vol, start, end, n=15, top_liquid=200):
    w, liquid = UNI._formation(px, vol, start, end)
    pool = [c for c in liquid.index[:top_liquid] if c in w.columns
            and c not in UNI.ETF_UNIVERSE]
    r = w[pool].pct_change().dropna(how="any")
    return list((r.std() * np.sqrt(252)).sort_values(ascending=False).index[:n])


def windows(year):
    """Base-paper calendar shifted to test year `year` (2024 = the paper)."""
    y = year
    return dict(form_start=f"{y-3}-01-01", val_start=f"{y-1}-07-01",
                sub_train_end=f"{y-1}-06-30", form_end=f"{y-1}-12-31",
                test_start=f"{y}-01-01", test_end=f"{y+1}-01-31")


def select(px, syms, W, sel_bps=25.0):
    base = dict(train_start=W["form_start"], train_end=W["sub_train_end"],
                test_start=W["val_start"], test_end=W["form_end"])
    best_a, bu = ME.ALLOC_GRID[0], -np.inf
    for g in ME.ALLOC_GRID:
        u = ME._val_utility(px, syms, base, g, "O1", sel_bps)
        if u > bu:
            best_a, bu = g, u
    base2 = dict(base, **best_a)
    best_r, ru = ME.ROUTER_GRID[0], -np.inf
    for g in ME.ROUTER_GRID:
        u = ME._val_utility(px, syms, base2, g, "O2", sel_bps)
        if u > ru:
            best_r, ru = g, u
    return {**best_a, **best_r}


def sr_diff_ci(a, b, n_boot=2000, block=10, seed=0):
    """Stationary block bootstrap CI for SR(a) - SR(b) on paired daily returns."""
    rng = np.random.default_rng(seed)
    x = np.c_[a.to_numpy(), b.to_numpy()]
    T = len(x)
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            if t and rng.random() < 1.0 / block:
                i = rng.integers(T)
            idx[t] = i
            i = (i + 1) % T
        y = x[idx]
        sr = y.mean(0) / y.std(0, ddof=1) * np.sqrt(252)
        out[k] = sr[0] - sr[1]
    return np.quantile(out, [0.025, 0.975])


def meta_quarterly(targets, rets, idx, bps):
    """Base-paper rule (argmax trailing-quarter utility) and Hedge, costed."""
    names = ["O1", "O2", "O3"]
    blocks = MP.scoring_blocks(idx, "QE")
    mode_r = {}
    for m in names:
        g, t, _ = S.simulate(targets[m], rets, idx)
        mode_r[m] = S.net_series(g, t, idx, bps)
    U, _ = MP.block_utilities(mode_r, blocks)
    out = {}
    for lab, kw in (("Argmax", dict(argmax=True)), ("Hedge", {})):
        P = MP.hedge(U, **kw)
        mt = MP.mixture_targets(targets, P, blocks, names)
        g, t, _ = S.simulate(mt, rets, idx)
        out[lab] = S.net_series(g, t, idx, bps)
    return out


def run(year, tag):
    px = pd.read_csv(DATA / "panel_close_us_2005.csv", index_col=0,
                     parse_dates=True).sort_index()
    vol = pd.read_csv(DATA / "panel_volume_us_2005.csv", index_col=0,
                      parse_dates=True).sort_index()
    W = windows(year)
    U = dict(FIXED)
    U["High-Vol-15"] = highvol(px, vol, W["form_start"], W["sub_train_end"])
    rows, preds, t0 = [], [], time.time()
    for uname, syms in U.items():
        syms = [s for s in syms if s in px.columns]
        g = select(px, syms, W)
        cfg = dict(train_start=W["form_start"], train_end=W["form_end"],
                   test_start=W["test_start"], test_end=W["test_end"], **g)
        out = RM.run_universe(px, syms, cfg)
        if out is None:
            print(f"{uname}: not formable"); continue
        idx, rets = out["test_idx"], out["rets"]
        # signal on the names that actually trade (full history over the span);
        # still formation-window prices only
        ch_f = UNI.characteristics(px, out["syms"], W["form_start"], W["form_end"])
        ch_t = UNI.characteristics(px, out["syms"], W["test_start"], W["test_end"])
        paths = dict(out["targets"])
        rebal = list(out["targets"]["O1"].index)
        bl = B.run_all(rets, out["syms"], rebal, window=out["cfg"]["cov_window"],
                       cap=out["cfg"]["cap"])
        paths["EqualWeight"] = bl["EqualWeight"]
        pred = "O2" if ch_f["routing_signal_legacy_avgvol"] > THRESH else "O1"
        pred_c = "O2" if ch_f["routing_signal"] > THRESH else "O1"
        rec = {"year": year, "universe": uname, "n": len(out["syms"]),
               "signal_legacy_form": ch_f["routing_signal_legacy_avgvol"],
               "signal_xs_form": ch_f["routing_signal"],
               "signal_legacy_test": ch_t.get("routing_signal_legacy_avgvol"),
               "signal_xs_test": ch_t.get("routing_signal"),
               "mean_corr_form": ch_f["mean_corr"],
               "pred_legacy": pred, "pred_xs": pred_c,
               "router_n_routed": out["diag"]["router_n_routed"],
               "router_w": json.dumps({k: round(v, 3) for k, v in
                                       out["diag"]["router_expert_weight"].items()}),
               **{f"sel_{k}": v for k, v in g.items()}}
        for bps in (0.0, 25.0):
            nets = {}
            for m, tg in paths.items():
                gr, tu, _ = S.simulate(tg, rets, idx)
                nets[m] = S.net_series(gr, tu, idx, bps)
                if bps == 0.0:
                    rec[f"turn_{m}"] = float(tu.mean() * 252)
            nets.update(meta_quarterly(out["targets"], rets, idx, bps))
            for m, r in nets.items():
                mm = S.metrics(r)
                rec[f"SR_{m}_{int(bps)}"] = mm["SR"]
                rec[f"CAGR_{m}_{int(bps)}"] = mm["CAGR"]
                rec[f"MDD_{m}_{int(bps)}"] = mm["MDD"]
            lo, hi = sr_diff_ci(nets["O2"], nets["O1"])
            rec[f"dSR_O2mO1_{int(bps)}"] = rec[f"SR_O2_{int(bps)}"] - rec[f"SR_O1_{int(bps)}"]
            rec[f"dSR_ci_lo_{int(bps)}"], rec[f"dSR_ci_hi_{int(bps)}"] = lo, hi
            rec[f"winner_{int(bps)}"] = "O2" if rec[f"dSR_O2mO1_{int(bps)}"] > 0 else "O1"
        rows.append(rec)
        print(f"{year} {uname:14s} n={rec['n']:2d} sL={rec['signal_legacy_form']:.3f} "
              f"sX={rec['signal_xs_form']:.3f} pred={pred} "
              f"SR O1/O2/O3/EW(25)={rec['SR_O1_25']:.2f}/{rec['SR_O2_25']:.2f}/"
              f"{rec['SR_O3_25']:.2f}/{rec['SR_EqualWeight_25']:.2f} "
              f"dSR={rec['dSR_O2mO1_25']:+.2f} [{rec['dSR_ci_lo_25']:+.2f},"
              f"{rec['dSR_ci_hi_25']:+.2f}] routed={rec['router_n_routed']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}_{year}.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2024")
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    for y in [int(v) for v in a.years.split(",")]:
        run(y, a.tag)
