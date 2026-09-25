"""W2: router over improved operating points + EW, with do-no-harm shrinkage,
hysteresis and the D1 deployment rule.  Protocol: ../PREREG.md, section W2.

Two phases, enforced in code:

  select  reads only the SELECT pickle, picks C* and the best config per family,
          applies D1, and writes results/w2_frozen_<tag>.json.  Refuses to run
          if the frozen file already exists (no re-selection).
  test    requires the frozen file, reads both pickles (SELECT cells supply the
          causal training history for the contextual router), evaluates the
          frozen policies once, and writes results/w2_test_<tag>.csv.

Usage:
  python3 t1_w2_router.py select --sel PKL --modes O1w,O2w,O3w,EW --tag w1
  python3 t1_w2_router.py test   --sel PKL --test PKL --modes ... --tag w1
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meta_policy as MP   # noqa: E402
import sim as S            # noqa: E402

OUT = HERE / "results"
BPS = 25.0
FEATS = ["disp", "corr", "vol", "sig", "mom", "du1", "du4"]
ETAS = ["adaptive", 0.5, 2.0, 8.0]
ALPHAS = [0.0, 0.05]
RIDGE = [10.0, 100.0, 1000.0]
TEMPS = [0.01, 0.03, 0.1]
RHOS = [0.0, 0.25, 0.5, 0.75]
HYST = [0.0, 0.1, 0.25]


# ------------------------------------------------------------------ data
def load_cells(path, modes):
    store = pickle.load(open(path, "rb"))
    cells = {}
    for key, c in store.items():
        tg = dict(c["targets"])
        ref = tg[[m for m in modes if m != "EW"][0]]
        if "EW" not in tg:
            tg["EW"] = pd.DataFrame(1.0 / ref.shape[1], index=ref.index, columns=ref.columns)
        tg = {m: tg[m] for m in modes}
        rets, idx = c["rets"], c["test_idx"]
        blocks = MP.scoring_blocks(idx, "QE")
        U, net = {}, {}
        for m, t in tg.items():
            g, tu, _ = S.simulate(t, rets, idx)
            net[m] = S.net_series(g, tu, idx, BPS)
            U[m] = np.array([MP.mv_certainty_equivalent(net[m].reindex(d).dropna())
                             for _, d in blocks])
        R = rets.replace([np.inf, -np.inf], np.nan)
        feats = []
        for _, d in blocks:
            h = R.loc[:d[0]].iloc[:-1].iloc[-63:].dropna(how="any")
            C = np.corrcoef(h.to_numpy(), rowvar=False)
            np.fill_diagonal(C, np.nan)
            disp = float(h.std(axis=1).mean() * np.sqrt(252))
            corr = float(np.nanmean(C))
            ew = h.mean(axis=1)
            feats.append(dict(disp=disp, corr=corr, vol=float(ew.std() * np.sqrt(252)),
                              sig=disp * (1 - corr), mom=float((1 + ew).prod() - 1)))
        cells[key] = dict(targets=tg, rets=rets, idx=idx, blocks=blocks, U=U, net=net,
                          feats=pd.DataFrame(feats))
    return cells


def univ(key):
    return str(key[1]).split("-")[0]


# -------------------------------------------------------------- policies
def post(Praw, anchor, rho, h):
    """Shrink toward the C* vertex, then apply hysteresis on the played mixture."""
    P = (1 - rho) * Praw + rho * anchor[None, :]
    out = P.copy()
    for t in range(1, len(P)):
        out[t] = P[t] if np.abs(P[t] - out[t - 1]).sum() > h else out[t - 1]
    return out


def hedge_raw(c, modes, eta, alpha):
    Um = np.c_[[c["U"][m] for m in modes]].T
    return MP.hedge(Um, eta=eta, alpha=alpha)[: len(Um)]


def long_table(cells, modes):
    rows = []
    for key, c in cells.items():
        Um = np.c_[[c["U"][m] for m in modes]].T
        rel = Um - Um.mean(axis=1, keepdims=True)
        for t, (_, d) in enumerate(c["blocks"]):
            f = c["feats"].iloc[t].to_dict()
            for j, m in enumerate(modes):
                rows.append(dict(cell=key, t=t, start=d[0], end=d[-1], mode=m, y=rel[t, j],
                                 du1=rel[t - 1, j] if t else 0.0,
                                 du4=rel[max(0, t - 4):t, j].mean() if t else 0.0, **f))
    return pd.DataFrame(rows)


def ridge_predictions(L, modes, alpha, eval_cells, min_train=40):
    """y_hat[(cell, t, mode)] from a pooled ridge fit on blocks ending before t starts."""
    pred = {}
    ev = L[L["cell"].isin(eval_cells)]
    for s in sorted(ev["start"].unique()):
        tr = L[L["end"] < s]
        cur = ev[ev["start"] == s]
        for m in modes:
            trm, cm = tr[tr["mode"] == m], cur[cur["mode"] == m]
            if not len(cm):
                continue
            if len(trm) < min_train:
                vals = np.zeros(len(cm))
            else:
                X, y = trm[FEATS].to_numpy(), trm["y"].to_numpy()
                mu, sd = X.mean(0), X.std(0) + 1e-12
                Z = (X - mu) / sd
                b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
                vals = ((cm[FEATS].to_numpy() - mu) / sd) @ b + y.mean()
            for (cell, t), v in zip(cm[["cell", "t"]].itertuples(index=False), vals):
                pred[(cell, t, m)] = float(v)
    return pred


def ctx_raw(c, key, modes, pred, temp):
    T = len(c["blocks"])
    P = np.zeros((T, len(modes)))
    for t in range(T):
        y = np.array([pred.get((key, t, m), 0.0) for m in modes])
        z = np.exp((y - y.max()) / temp)
        P[t] = z / z.sum()
    return P


def deploy(c, modes, P):
    Pfull = np.vstack([P, P[-1:]])
    mt = MP.mixture_targets(c["targets"], Pfull, c["blocks"], modes)
    g, tu, _ = S.simulate(mt, c["rets"], c["idx"])
    net = S.net_series(g, tu, c["idx"], BPS)
    U = np.array([MP.mv_certainty_equivalent(net.reindex(d).dropna()) for _, d in c["blocks"]])
    return dict(net=net, U=U, turn=float(tu.mean() * 252))


def pooled(res):
    return float(np.mean(np.concatenate([r["U"] for r in res.values()])))


def run_family(cells, modes, anchor, fam, cfg, pred_cache, L):
    res = {}
    for key, c in cells.items():
        if fam == "W2a":
            Praw = hedge_raw(c, modes, cfg["eta"], cfg["alpha"])
        else:
            pk = cfg["ridge"]
            Praw = ctx_raw(c, key, modes, pred_cache[pk], cfg["temp"])
        res[key] = deploy(c, modes, post(Praw, anchor, cfg["rho"], cfg["h"]))
    return res


def grids(families=("W2a", "W2b"), rhos=RHOS, hyst=HYST):
    ga = [dict(eta=e, alpha=a, rho=r, h=h) for e, a, r, h in
          itertools.product(ETAS, ALPHAS, rhos, hyst)]
    gb = [dict(ridge=a, temp=t, rho=r, h=h) for a, t, r, h in
          itertools.product(RIDGE, TEMPS, rhos, hyst)]
    return {k: v for k, v in {"W2a": ga, "W2b": gb}.items() if k in families}


# ------------------------------------------------------------ statistics
def boot_p(x, reps=5000, block=4, seed=0):
    x = np.asarray(x, float)
    T = len(x)
    rng = np.random.default_rng(seed)
    obs, xc, cnt = x.mean(), x - x.mean(), 0
    for _ in range(reps):
        idx, i = [], rng.integers(T)
        for t in range(T):
            if t and rng.random() < 1 / block:
                i = rng.integers(T)
            idx.append(i)
            i = (i + 1) % T
        cnt += xc[idx].mean() >= obs
    return (cnt + 1) / (reps + 1)


def by_date(cells, res):
    rows = [dict(cell=k, date=lab, U=res[k]["U"][t])
            for k, c in cells.items() for t, (lab, _) in enumerate(c["blocks"])]
    return pd.DataFrame(rows).set_index(["cell", "date"]).U


def sharpe_by_type(cells, res):
    d = {}
    for k in cells:
        d.setdefault(univ(k), []).append(S.metrics(res[k]["net"])["SR"])
    return {u: float(np.mean(v)) for u, v in d.items()}


# ----------------------------------------------------------------- phases
def phase_select(a, modes):
    frozen = OUT / f"w2_frozen_{a.tag}.json"
    if frozen.exists():
        sys.exit(f"{frozen} exists: selection is frozen; refusing to re-select")
    cells = load_cells(a.sel, modes)
    fixed = {m: {k: dict(U=c["U"][m], net=c["net"][m]) for k, c in cells.items()} for m in modes}
    usel = {m: pooled(fixed[m]) for m in modes}
    cstar = max(usel, key=usel.get)
    anchor = np.eye(len(modes))[modes.index(cstar)]
    L = long_table(cells, modes)
    pred_cache = ({al: ridge_predictions(L, modes, al, list(cells)) for al in RIDGE}
                  if "W2b" in a.families else {})
    rec = {"tag": a.tag, "modes": modes, "C_star": cstar, "U_sel_fixed": usel, "families": {},
           "grid": {"families": a.families, "rhos": a.rhos, "hyst": a.hyst}}
    rows, perblock = [], {m: by_date(cells, fixed[m]) for m in modes}
    for fam, grid in grids(a.families.split(","), [float(x) for x in a.rhos.split(",")],
                           [float(x) for x in a.hyst.split(",")]).items():
        best, bu = None, -np.inf
        for ci, cfg in enumerate(grid):
            res = run_family(cells, modes, anchor, fam, cfg, pred_cache, L)
            u = pooled(res)
            rows.append(dict(family=fam, cfg_id=f"{fam}_{ci}",
                             **{k: str(v) for k, v in cfg.items()}, U_sel=u))
            # per-block utilities (dates x cells) for PBO / SPA (exp/T2/acceptance.py)
            perblock[f"{fam}_{ci}"] = by_date(cells, res)
            key = (u, cfg["rho"], cfg["h"])
            if best is None or key > (bu, best["rho"], best["h"]):
                best, bu = cfg, u
        deployed = bu > usel[cstar]
        rec["families"][fam] = dict(config=best, U_sel=bu, U_sel_Cstar=usel[cstar],
                                    deployed=bool(deployed),
                                    status="deployed" if deployed else "not deployed (D1): C* deployed")
        print(fam, rec["families"][fam], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"w2_select_grid_{a.tag}.csv", index=False)
    pd.DataFrame(perblock).to_csv(OUT / f"w2_select_perblock_{a.tag}.csv")
    json.dump(rec, open(frozen, "w"), indent=1, default=str)
    print("frozen ->", frozen, "| C* =", cstar, {m: round(v, 4) for m, v in usel.items()})


def phase_test(a, modes):
    frozen = OUT / f"w2_frozen_{a.tag}.json"
    if not frozen.exists():
        sys.exit("run the select phase first")
    rec = json.load(open(frozen))
    assert rec["modes"] == modes, "menu differs from the frozen one"
    sel_cells = load_cells(a.sel, modes)
    cells = load_cells(a.test, modes)
    cstar = rec["C_star"]
    anchor = np.eye(len(modes))[modes.index(cstar)]
    L = long_table({**sel_cells, **cells}, modes)
    fixed = {m: {k: dict(U=c["U"][m], net=c["net"][m]) for k, c in cells.items()} for m in modes}
    hedge = {k: deploy(c, modes, hedge_raw(c, modes, "adaptive", 0.0)) for k, c in cells.items()}
    if "EW" in modes:
        ew = fixed["EW"]
    else:
        ew = {}
        for k, c in cells.items():
            ref = c["targets"][modes[0]]
            t = pd.DataFrame(1.0 / ref.shape[1], index=ref.index, columns=ref.columns)
            g, tu, _ = S.simulate(t, c["rets"], c["idx"])
            n = S.net_series(g, tu, c["idx"], BPS)
            ew[k] = dict(net=n, U=np.array([MP.mv_certainty_equivalent(n.reindex(d).dropna())
                                            for _, d in c["blocks"]]))
    argm = {k: deploy(c, modes, MP.hedge(np.c_[[c["U"][m] for m in modes]].T, argmax=True)[:len(c["blocks"])])
            for k, c in cells.items()}
    comps = {"C*": fixed[cstar], "EW": ew, "Hedge": hedge, "Argmax": argm}
    comps[modes[0]] = fixed[modes[0]]
    out = []
    for fam, f in rec["families"].items():
        cfg = f["config"]
        cfg = {k: (v if k == "eta" and v == "adaptive" else float(v)) for k, v in cfg.items()}
        pc = {}
        if fam == "W2b":
            pc[cfg["ridge"]] = ridge_predictions(L, modes, cfg["ridge"], list(cells))
        res = run_family(cells, modes, anchor, fam, cfg, pc, L)
        deployed = f["deployed"]
        live = res if deployed else fixed[cstar]
        r = dict(family=fam, status=f["status"], U=pooled(live),
                 hypothetical_U_not_deployed=None if deployed else pooled(res),
                 turn=float(np.mean([x.get("turn", np.nan) for x in live.values()])))
        srr = sharpe_by_type(cells, live)
        for cn, cres in comps.items():
            d = (by_date(cells, live) - by_date(cells, cres)).groupby(level="date").mean()
            r[f"dU_vs_{cn}"] = float(d.mean())
            r[f"p_vs_{cn}"] = boot_p(d.to_numpy())
            src = sharpe_by_type(cells, cres)
            r[f"SR_wins_vs_{cn}"] = int(sum(srr[u] > src[u] for u in srr))
        r["SR_by_type"] = json.dumps({u: round(v, 3) for u, v in srr.items()})
        out.append(r)
        print(r, flush=True)
    R = pd.DataFrame(out)
    R["p_primary"] = R[["p_vs_C*", "p_vs_EW"]].max(axis=1)
    m = len(R)
    order = R["p_primary"].sort_values().index
    run, adj = 0.0, {}
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * R.at[j, "p_primary"]))
        adj[j] = run
    R["p_holm"] = pd.Series(adj)
    ref = {n: sharpe_by_type(cells, c) for n, c in {**fixed, "EW": ew, "Hedge": hedge, "Argmax": argm}.items()}
    pd.DataFrame(ref).to_csv(OUT / f"w2_test_reference_SR_{a.tag}.csv")
    R.to_csv(OUT / f"w2_test_{a.tag}.csv", index=False)
    print(pd.DataFrame(ref).round(3))
    print(R.drop(columns=["SR_by_type"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "test"])
    ap.add_argument("--sel", required=True)
    ap.add_argument("--test")
    ap.add_argument("--modes", default="O1w,O2w,O3w,EW")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--families", default="W2a,W2b")
    ap.add_argument("--rhos", default=",".join(map(str, RHOS)))
    ap.add_argument("--hyst", default=",".join(map(str, HYST)))
    a = ap.parse_args()
    modes = a.modes.split(",")
    (phase_select if a.phase == "select" else phase_test)(a, modes)
