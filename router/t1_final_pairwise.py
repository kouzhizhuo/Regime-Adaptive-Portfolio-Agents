"""Frozen F: paired tests of Deployed vs each comparator under every utility, plus MCS membership.
    python3 t1_final_pairwise.py --tag us_test"""
import argparse, json, sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import acceptance as A
from pathlib import Path
FIN = Path(__file__).resolve().parent / "final"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
pb = pd.read_csv(FIN / f"{a.tag}_perblock.csv")
rows, mcs = [], {}
for loss in ["mv", "sortino", "cvar"]:
    P = A.panel_from_long(pb[(pb.bps == 25) & (pb.loss == loss)].drop(columns=["bps", "loss"]))
    for c in ["EW", "RiskParity", "MinVariance", "HRP", "Momentum", "Hedge", "Argmax", "O2", "O3"]:
        r = A.paired_bootstrap(P["Deployed"], P[c], n_boot=5000, mean_block=4, seed=0)
        s = A.paired_bootstrap(P["Deployed"], P[c], n_boot=5000, mean_block=4, seed=0, studentize=True)
        rows.append(dict(loss=loss, comparator=c, dU=r.estimate, p=r.p_value, p_stud=s.p_value))
    M = pd.DataFrame({k: A.pool_by_date(P[k]) for k in ["Deployed", "Hedge", "Argmax", "O2", "O3", "EW",
                                                      "RiskParity", "MinVariance", "HRP", "Momentum"]})
    mcs[loss] = {b: A.model_confidence_set(M, higher_is_better=True, mean_block=b, seed=0)["included"] for b in (1, 4)}
R = pd.DataFrame(rows); R.to_csv(FIN / f"{a.tag}_pairwise.csv", index=False)
json.dump(mcs, open(FIN / f"{a.tag}_mcs.json", "w"), indent=1)
print(R.round(4).to_string(index=False)); print(mcs)
