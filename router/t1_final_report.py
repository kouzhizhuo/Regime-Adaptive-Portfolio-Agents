"""Frozen-design F evaluation, step 2: statistics and LaTeX tables from the replay files.

    python3 t1_final_report.py --tag us_test [--select-tag select_parity] [--label US]

Reads ../final/<tag>_perblock.csv, <tag>_daily.pkl and <tag>_cell_metrics.csv, which
t1_final_eval.py writes. Writes ../final/<tag>_acceptance_{deployed,hedge}.{csv,tex},
<tag>_tables.tex and <tag>_summary.json. No simulation happens here.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIN = HERE / "final"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance as A   # noqa: E402

GRID = [f"W2a_{i}" for i in range(8)]
SHOW = ["Deployed", "Hedge", "Argmax", "O1", "O2", "O3", "EW", "RiskParity", "MinVariance", "HRP", "Momentum"]


def utype(cell):
    u = eval(cell)[1] if cell.startswith("(") else cell
    if u.startswith("N:"):
        return u
    pre, _, base = u.rpartition(":")          # "JP:MegaLiquid-50" -> "JP:MegaLiquid"
    return (pre + ":" if pre else "") + base.split("-")[0]


def panels(pb, bps=25.0, loss="mv"):
    d = pb[(pb.bps == bps) & (pb.loss == loss)].drop(columns=["bps", "loss"])
    return A.panel_from_long(d)


def pooled_daily(daily, name):
    return pd.concat([df[name] for df in daily.values()], axis=1).mean(axis=1).sort_index()


def sr_by_type(cm, strat, bps=25.0, rule_only=True):
    x = cm[(cm.bps == bps) & (cm.strategy == strat)].copy()
    x["t"] = x.cell.apply(utype)
    if rule_only:
        x = x[~x.t.str.startswith("N:")]
    return x.groupby("t").SR.mean().to_dict()


def _esc(x):
    """Escape underscores, except inside math-mode labels."""
    x = str(x)
    return x if "$" in x else x.replace("_", r"\_")


def tex_table(df, title, note, label, fmt="{:.2f}"):
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{title}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & ".join([""] + [_esc(c) for c in cols]) + r" \\", r"\midrule"]
    for i, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                 ("--" if isinstance(v, float) else str(v)) for v in r.to_numpy()]
        lines.append(" & ".join([_esc(i)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\par\vspace{{2pt}}\footnotesize {note}", r"\end{table}"]
    return "\n".join(lines)


def acc_latex(table, orig):
    """Rebuild the acceptance table body from the (post-processed) table, keeping T2's layout.
    The note's C* definition is corrected for F: C*_F is the best fixed operating point on SELECT."""
    def verdict(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return "pass" if bool(v) else "fail"
    body = "\n".join(f"{A._tex(r.test)} & {A._tex(r.statistic)} & {A._fmt(r.estimate)} & {A._fmt(r.p)} & "
                     f"{A._tex(r.rule)} & {verdict(r.verdict)} \\\\" for r in table.itertuples())
    head, _, rest = orig.partition("\\midrule\n")
    _, _, tail = rest.partition("\n\\bottomrule")
    tail = tail.replace("$C^{\\ast}$: max(best fixed mode chosen on SELECT, EW).",
                        "$C^{\\ast}$: the best fixed operating point on SELECT (O1).")
    return head + "\\midrule\n" + body + "\n\\bottomrule" + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--select-tag", default="select_parity")
    ap.add_argument("--label", default="US 2014+")
    ap.add_argument("--holdout", action="store_true")   # PREREG §H: H-B3 needs p < 0.05
    a = ap.parse_args()
    pb = pd.read_csv(FIN / f"{a.tag}_perblock.csv")
    cm = pd.read_csv(FIN / f"{a.tag}_cell_metrics.csv")
    daily = pickle.load(open(FIN / f"{a.tag}_daily.pkl", "rb"))
    P = panels(pb)
    grid_test = pd.DataFrame({k: A.pool_by_date(P[k]) for k in GRID}, index=P["O1"].index)
    if a.select_tag == "none":
        # holdout: every fold is out of sample, so PBO runs on the holdout panel alone
        grid_all = grid_test
    else:
        sel = panels(pd.read_csv(FIN / f"{a.select_tag}_perblock.csv"))
        grid_all = pd.concat([pd.DataFrame({k: A.pool_by_date(sel[k]) for k in GRID}, index=sel["O1"].index),
                              grid_test]).sort_index()
        grid_all = grid_all[~grid_all.index.duplicated()]
    dr = {k: pooled_daily(daily, k) for k in ["Deployed", "Hedge", "O1", "EW"] + GRID}
    tsr = [A.sharpe(dr[k].to_numpy()) for k in GRID]
    U = {k: P[k] for k in ["Deployed", "O1", "Hedge", "EW", "O2", "O3", "Argmax"]}
    out, summ = {}, {"tag": a.tag, "n_cells": len(daily), "n_dates": int(len(P["O1"].index))}
    for rname, cname in (("Deployed", "O1"), ("Hedge", "O1")):
        # router, C* and B3 comparator; the router's own column is not duplicated
        UU = {k: v for k, v in U.items() if k not in (rname, cname)}
        UU["router"], UU["Cstar"] = P[rname], P[cname]
        hk = "Hedge" if rname != "Hedge" else "Argmax"   # B3 ref; for Hedge-as-router use Argmax
        rep = A.final_report(UU, hedge=hk, sr_by_universe={"router": sr_by_type(cm, rname), "Cstar": sr_by_type(cm, cname)},
                             daily_returns={"router": dr[rname], "Cstar": dr[cname]},
                             spa_alternatives=grid_test, trials=grid_all, trial_sharpes=tsr,
                             caption=f"Acceptance tests, {a.label}: {'deployed policy (D1 gate)' if rname=='Deployed' else 'ungated Hedge (not deployed)'}.",
                             label=f"tab:acc_{a.tag}_{rname.lower()}")
        if rname == "Deployed" and np.allclose(A.pool_by_date(P[rname]), A.pool_by_date(P[cname])):
            # D1 deployed C* itself: router-vs-C* statistics are 0 by construction. B1/B2 keep their
            # (failing) values; LW and MCS, which need two distinct series, are shown as degenerate.
            t = rep["table"]
            m = t["test"].str.contains("B1|B2")
            t.loc[m, "rule"] = t.loc[m, "rule"] + " [router = C*]"
            m = t["test"].str.contains("Ledoit|MCS")
            t.loc[m, "estimate"] = "-- (degenerate)"
            t.loc[m, "p"] = None
            t.loc[m, "verdict"] = None
            t.loc[m, "rule"] = "router = C*"
            t.loc[t["test"].str.contains("MCS"), "statistic"] = "router in 90% set"
            t["verdict"] = t["verdict"].astype(object)
        if rname == "Hedge":
            rep["table"].loc[rep["table"]["test"].str.startswith("B3"), "statistic"] = "mean dU vs Argmax (B3 ref)"
        if a.holdout:
            # H-B3 (PREREG §H) requires dU > 0 AND p < 0.05; the US B3 (P-router B3) is dU > 0 only
            t = rep["table"]; m = t["test"].str.startswith("B3")
            t.loc[m, "rule"] = "dU > 0, p<0.05 (H-B3)"
            t.loc[m, "verdict"] = [(e > 0) and (p < 0.05) for e, p in zip(t.loc[m, "estimate"], t.loc[m, "p"])]
            rep["latex"] = rep["latex"].replace("$\\Delta U>0$ &", "$\\Delta U>0$, $p<0.05$ &")
        rep["latex"] = acc_latex(rep["table"], rep["latex"])
        if a.holdout:
            rep["latex"] = rep["latex"].replace("TEST era only;", "Held-out sample, all folds out of sample;")
        rep["table"].to_csv(FIN / f"{a.tag}_acceptance_{rname.lower()}.csv", index=False)
        (FIN / f"{a.tag}_acceptance_{rname.lower()}.tex").write_text(rep["latex"])
        out[rname] = rep
        print(rname, "\n", rep["table"].to_string(index=False))
    # extra pre-registered comparisons (vs EW, vs Hedge) for both
    extra = {}
    for r in ("Deployed", "Hedge"):
        for c in ("EW", "O1", "Hedge", "Argmax"):
            if r == c:
                continue
            res = A.paired_bootstrap(P[r], P[c], n_boot=5000, mean_block=4, seed=0)
            st = A.paired_bootstrap(P[r], P[c], n_boot=5000, mean_block=4, seed=0, studentize=True)
            extra[f"{r}_vs_{c}"] = dict(dU=res.estimate, p=res.p_value, p_stud=st.p_value)
    summ["pairwise"] = extra
    print(json.dumps(extra, indent=1))
    # tables
    strat = [s for s in SHOW if s in cm.strategy.unique()]
    x = cm[cm.bps == 25.0].copy()
    x["type"] = x.cell.apply(utype)
    t1 = x.pivot_table(index="strategy", columns="type", values="SR", aggfunc="mean").reindex(strat)
    t1["mean"] = x.groupby("strategy").SR.mean().reindex(strat)
    t1m = x.groupby("strategy")[["SR", "CAGR", "MDD", "turnover_yr"]].mean().reindex(strat)
    lad = cm.pivot_table(index="strategy", columns="bps", values="SR", aggfunc="mean").reindex(strat)
    lossrows = {}
    for loss in ("mv", "sortino", "cvar"):
        Pl = panels(pb, 25.0, loss)
        lossrows[loss] = {s: float(np.nanmean(A.pool_by_date(Pl[s]))) for s in strat if s in Pl}
    lt = pd.DataFrame(lossrows)
    tex = [tex_table(t1, f"Net Sharpe by universe, {a.label}",
                     "25 bps one-way; mean over folds; Deployed = D1-gated meta-policy (holds $C^\\ast$=O1).",
                     f"tab:{a.tag}_sr"),
           tex_table(t1m, f"Pooled cell means, {a.label}",
                     "25 bps; CAGR and MDD in \\%; turnover in multiples of NAV per year.",
                     f"tab:{a.tag}_pooled"),
           tex_table(lad, f"Cost ladder, {a.label}", "Mean net Sharpe over cells at each one-way cost (bps).",
                     f"tab:{a.tag}_cost"),
           tex_table(lt, f"Utility robustness, {a.label}",
                     "Mean per-quarter certainty equivalent at 25 bps; meta-policies re-run under each utility.",
                     f"tab:{a.tag}_loss", fmt="{:.4f}")]
    (FIN / f"{a.tag}_tables.tex").write_text("\n\n".join(tex))
    t1.to_csv(FIN / f"{a.tag}_sr_by_type.csv"); t1m.to_csv(FIN / f"{a.tag}_pooled.csv")
    lad.to_csv(FIN / f"{a.tag}_cost_ladder.csv"); lt.to_csv(FIN / f"{a.tag}_loss_robustness.csv")
    summ["sr_by_type"] = t1.round(4).to_dict(); summ["pooled"] = t1m.round(4).to_dict()
    json.dump(summ, open(FIN / f"{a.tag}_summary.json", "w"), indent=1, default=float)
    print(t1.round(3).to_string()); print(t1m.round(3).to_string()); print(lad.round(3).to_string()); print(lt.round(4).to_string())


if __name__ == "__main__":
    main()
