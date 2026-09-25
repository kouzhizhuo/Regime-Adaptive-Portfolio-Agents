"""Frozen-design F evaluation, step 1: replay (coordinator approval 2026-09-23; PREREG §F, §H).

For every cell in the given target pickles this computes, once, the net daily
returns and per-quarter utilities of
    Deployed  (D1 decision in w2_frozen_F_default.json; here = C*_F)
    Hedge     (the frozen W2a config, "not deployed" diagnostic)
    W2a_0..7  (the full frozen grid, for PBO / SPA)
    Argmax    (eta -> inf corner, ablation)
    O1 O2 O3  EW  RiskParity  MinVariance  HRP  Momentum
at 0/10/25/50 bps under the MV certainty equivalent. At 25 bps it also does the
Sortino and CVaR utilities; the meta-policies are re-run under each loss, while the
deployed policy stays the frozen one. Black-Litterman is excluded (identical to
RiskParity without views; T4).

    python3 t1_final_eval.py --cells A.pkl[,B.pkl] --tag us_test
Outputs in ../final/: <tag>_perblock.csv (long: cell,date,bps,loss,strategies...),
<tag>_daily.pkl, <tag>_cell_metrics.csv.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B      # noqa: E402
import meta_policy as MP   # noqa: E402
import sim as S            # noqa: E402
import t1_w2_router as W   # noqa: E402

OUT = HERE / "final"
FROZEN = HERE / "results" / "w2_frozen_F_default.json"
MODES = ["O1", "O2", "O3"]
BASE = ["RiskParity", "MinVariance", "HRP", "Momentum"]
LADDER = [0.0, 10.0, 25.0, 50.0]
LOSSES = ["mv", "sortino", "cvar"]


def block_u(net, blocks, loss):
    fn = MP.LOSSES[loss]
    return np.array([fn(net.reindex(d).dropna()) for _, d in blocks])


def mix(targets, P, blocks, names, rets, idx):
    Pfull = np.vstack([P, P[-1:]])
    mt = MP.mixture_targets({m: targets[m] for m in names}, Pfull, blocks, names)
    return S.simulate(mt, rets, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="")      # e.g. F3,F4,F5,F6; empty = all cells
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{a.tag}_perblock.csv").exists():
        sys.exit("already evaluated: approved runs happen once")
    fr = json.load(open(FROZEN))
    assert fr["modes"] == MODES
    fam = fr["families"]["W2a"]
    deployed = fam["config"] if fam["deployed"] else None
    cstar = fr["C_star"]
    hcfg = fam["config"]
    grid = W.grids(["W2a"], [0.0], [0.0])["W2a"]
    store = {}
    for p in a.cells.split(","):
        # optional market prefix, "file.pkl:JP" -> keys (fold, "JP:<universe>"); needed when
        # several markets share (fold, universe) names
        path, _, pre = p.partition(":")
        d = pickle.load(open(path, "rb"))
        if pre:
            d = {(k[0], f"{pre}:{k[1]}"): v for k, v in d.items()}
        assert not set(d) & set(store), "duplicate cell keys across pickles"
        store.update(d)
    if a.folds:
        keep = set(a.folds.split(","))
        store = {k: v for k, v in store.items() if k[0] in keep}
    t0 = time.time()
    long_rows, daily, mrows = [], {}, []
    for key, c in store.items():
        tg = {m: c["targets"][m] for m in MODES}
        rets, idx = c["rets"], c["test_idx"]
        syms = list(tg["O1"].columns)
        rebal = list(tg["O1"].index)
        tg["EW"] = pd.DataFrame(1.0 / len(syms), index=tg["O1"].index, columns=syms)
        cap = float(c.get("diag", {}).get("cap", np.clip(5.0 / len(syms), 0.05, 0.35)))
        bl = B.run_all(rets, syms, rebal, window=126, cap=cap)
        for b in BASE:
            if b in bl:
                tg[b] = bl[b]
        blocks = MP.scoring_blocks(idx, "QE")
        gt = {m: S.simulate(t, rets, idx)[:2] for m, t in tg.items()}
        for loss in LOSSES:
            for bps in (LADDER if loss == "mv" else [25.0]):
                net = {m: S.net_series(g, tu, idx, bps) for m, (g, tu) in gt.items()}
                U = {m: block_u(n, blocks, loss) for m, n in net.items()}
                Um = np.c_[[U[m] for m in MODES]].T
                turn = {m: float(tu.mean() * 252) for m, (g, tu) in gt.items()}
                # meta-policies on this cost/loss
                metas = {}
                for i, cfg in enumerate(grid):
                    P = MP.hedge(Um, eta=cfg["eta"], alpha=cfg["alpha"])[: len(blocks)]
                    metas[f"W2a_{i}"] = P
                    if cfg["eta"] == hcfg["eta"] and float(cfg["alpha"]) == float(hcfg["alpha"]):
                        metas["Hedge"] = P
                metas["Argmax"] = MP.hedge(Um, argmax=True)[: len(blocks)]
                for name, P in metas.items():
                    g, tu = mix(tg, P, blocks, MODES, rets, idx)[:2]
                    net[name] = S.net_series(g, tu, idx, bps)
                    U[name] = block_u(net[name], blocks, loss)
                    turn[name] = float(tu.mean() * 252)
                # deployed policy (frozen): D1 -> C*, else the frozen Hedge config
                dep = cstar if deployed is None else "Hedge"
                net["Deployed"], U["Deployed"], turn["Deployed"] = net[dep], U[dep], turn[dep]
                for t, (lab, _) in enumerate(blocks):
                    long_rows.append(dict(cell=str(key), date=lab, bps=bps, loss=loss,
                                          **{m: U[m][t] for m in U}))
                if loss == "mv":
                    for m, n in net.items():
                        mm = S.metrics(n)
                        mrows.append(dict(cell=str(key), fold=key[0], universe=key[1], bps=bps,
                                          strategy=m, turnover_yr=turn[m], **mm))
                    if bps == 25.0:
                        daily[str(key)] = pd.DataFrame(net)
        print(key, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(long_rows).to_csv(OUT / f"{a.tag}_perblock.csv", index=False)
    pd.DataFrame(mrows).to_csv(OUT / f"{a.tag}_cell_metrics.csv", index=False)
    pickle.dump(daily, open(OUT / f"{a.tag}_daily.pkl", "wb"))
    print("done", len(store), "cells", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
