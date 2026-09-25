"""Frozen F robustness (T4 targets, evaluated once on F3-F6): lambda_alloc grid and 4 vs 8 experts."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_final_report import tex_table
FIN = Path(__file__).resolve().parent / "final"
S = ["Deployed", "Hedge", "Argmax", "O1", "O2", "O3", "EW"]
def load(tag):
    m = pd.read_csv(FIN / f"{tag}_cell_metrics.csv"); return m[m.bps == 25.0]
base = load("us_test")
rows = {}
for tag, lab in [("us_test", "selected (F)"), ("robust_lam50", "$\\lambda_{\\text{alloc}}=50$"), ("robust_lam100", "$\\lambda_{\\text{alloc}}=100$"),
                 ("robust_lam200", "$\\lambda_{\\text{alloc}}=200$"), ("robust_lam400", "$\\lambda_{\\text{alloc}}=400$")]:
    m = load(tag); g = m.groupby("strategy")
    rows[lab] = {**{s: g.SR.mean()[s] for s in S}, "turn_O1": g.turnover_yr.mean()["O1"]}
L = pd.DataFrame(rows).T
e8 = load("robust_experts8")
a = base[base.strategy == "O2"].set_index("cell"); b = e8[e8.strategy == "O2"].set_index("cell").loc[a.index]
w = stats.wilcoxon(b.SR, a.SR); wt = stats.wilcoxon(b.turnover_yr, a.turnover_yr)
E = pd.DataFrame({"M=4 (F)": [a.SR.mean(), a.turnover_yr.mean(), np.nan],
                  "M=8": [b.SR.mean(), b.turnover_yr.mean(), np.nan]},
                 index=["O2 net SR", "O2 turnover/yr", "_"]).drop("_")
d8 = {s: e8.groupby("strategy").SR.mean()[s] for s in ["Deployed", "Hedge", "Argmax", "O2"]}
extra = dict(sr_wilcoxon_p=w.pvalue, turn_wilcoxon_p=wt.pvalue, cells_O2_SR_up=int((b.SR > a.SR).sum()),
             cells_turn_up=int((b.turnover_yr > a.turnover_yr).sum()), n=len(a), meta_with_M8=d8)
L.to_csv(FIN / "robust_lambda.csv"); E.to_csv(FIN / "robust_experts.csv")
pd.Series(extra).to_json(FIN / "robust_experts_tests.json", indent=1)
tex = tex_table(L.rename(columns={"turn_O1": "O1 turnover"}), "Allocator risk aversion, US 2014+",
                "Net Sharpe at 25 bps, mean over 44 cells; $\\lambda_{\\text{alloc}}$ fixed per row, $\\kappa$ and router as selected; Deployed = D1-gated policy ($C^\\ast$=O1).",
                "tab:robust_lambda") + "\n\n" + tex_table(E, "Number of experts, US 2014+",
                f"O2 over 44 cells at 25 bps; Wilcoxon signed-rank p = {w.pvalue:.3f} (Sharpe), {wt.pvalue:.1e} (turnover); turnover higher with M=8 in {extra['cells_turn_up']}/44 cells.",
                "tab:robust_experts")
(FIN / "robust_tables.tex").write_text(tex)
print(L.round(3).to_string()); print(E.round(3).to_string()); print(extra)
