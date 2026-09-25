"""Generate frozen-design F cells (PREREG.md §F): 6 rule-built + 5 named universes
per fold, sibling two-stage validation selection, fixed allocator.

Writes results/F_cells_<folds>.pkl in the us_targets format
    (fold, universe) -> {targets{O1,O2,O3}, syms, test_idx, rets, diag, router}
plus results/F_selection_<folds>.csv. Computes NO strategy outcome (no
simulation on the evaluation window).

    python3 t1_F_cells.py --folds F1,F2 --workers 6
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = Path(os.environ.get("ROUTER_DATA", Path(__file__).resolve().parent / "data"))
OUT = HERE / "results"
NAMED = {
    "BigTech-6": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META"],
    "Consumer-20": ["WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW",
                    "TJX", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ", "ORLY", "AZO"],
    "Defensive-15": ["JNJ", "PG", "KO", "PEP", "CL", "GIS", "K", "SJM", "MKC", "HRL",
                     "DUK", "SO", "D", "AEP", "NEE"],
    "Healthcare-10": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY"],
}
_PX = _VOL = None


def _load():
    global _PX, _VOL
    if _PX is None:
        _PX = pd.read_csv(DATA / "panel_close_us_2005.csv", index_col=0, parse_dates=True).sort_index()
        _VOL = pd.read_csv(DATA / "panel_volume_us_2005.csv", index_col=0, parse_dates=True).sort_index()
    return _PX, _VOL


def universes_for(fold):
    import universes as UNI
    import t1_base_universes as TB
    px, vol = _load()
    _, tr_s, tr_e, _, te_e = fold
    U = UNI.build_universes(px, vol, tr_s, tr_e, top_liquid=200)
    span = px.loc[tr_s:te_e]
    for name, syms in NAMED.items():
        ok = [s for s in syms if s in span.columns and span[s].notna().all()]
        if len(ok) >= 5:
            U[f"N:{name}"] = ok
    U["N:High-Vol-15"] = TB.highvol(px, vol, tr_s, tr_e)
    return U


def one_cell(args):
    fold, uname, syms = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import main_experiment as ME
    import run_modes as RM
    px, _ = _load()
    fname, tr_s, tr_e, te_s, te_e = fold
    t0 = time.time()
    g, au, ru = ME.select_config(px, syms, fold)
    cfg = dict(train_start=tr_s, train_end=tr_e, test_start=te_s, test_end=te_e, **g)
    out = RM.run_universe(px, syms, cfg)
    sel = {"fold": fname, "universe": uname, "val_utility_alloc": au,
           "val_utility_router": ru, "secs": time.time() - t0, **g}
    if out is None:
        return (fname, uname), None, sel
    rec = {"targets": out["targets"], "syms": out["syms"], "test_idx": out["test_idx"],
           "rets": out["rets"][out["syms"]], "diag": out["diag"], "router": g}
    return (fname, uname), rec, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import main_experiment as ME
    want = a.folds.split(",")
    folds = [f for f in ME.US_FOLDS if f[0] in want]
    jobs = []
    for f in folds:
        for u, s in universes_for(f).items():
            jobs.append((f, u, s))
    print(f"{len(jobs)} cells", flush=True)
    store, sels = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        for key, rec, sel in ex.map(one_cell, jobs):
            sels.append(sel)
            if rec is not None:
                store[key] = rec
            print(key, "ok" if rec else "not formable", f"{sel['secs']:.0f}s", flush=True)
    tag = "_".join(want)
    pickle.dump(store, open(OUT / f"F_cells_{tag}.pkl", "wb"))
    pd.DataFrame(sels).to_csv(OUT / f"F_selection_{tag}.csv", index=False)
    print("wrote", len(store), "cells")


if __name__ == "__main__":
    main()
