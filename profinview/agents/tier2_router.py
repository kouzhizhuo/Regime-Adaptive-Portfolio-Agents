from __future__ import annotations
from typing import List, Dict
import numpy as np
from .tier2_bus import SignalMessage

class RouterParams:
    def __init__(self, min_capacity: float = 1e5, min_conf: float = 0.1, top_k: int = 3, lambda_risk: float = 1.0, gamma_cost: float = 1.0, tau: float = 0.5, delta_div: float = 0.0):
        self.min_capacity = min_capacity
        self.min_conf = min_conf
        self.top_k = top_k
        self.lambda_risk = lambda_risk
        self.gamma_cost = gamma_cost
        self.tau = tau
        self.delta_div = delta_div

class RiskAwareRouter:
    def __init__(self, params: RouterParams | None = None) -> None:
        self.params = params or RouterParams()

    def route(self, signals: List[SignalMessage]) -> Dict[str, float]:
        cands = [s for s in signals if s.capacity >= self.params.min_capacity and s.conf >= self.params.min_conf]
        if not cands:
            return {}
        utilities: List[float] = []
        for s in cands:
            mu, var = s.dist.mean, s.dist.var
            U = mu - self.params.lambda_risk * var - self.params.gamma_cost * (s.cost_bps / 1e4)
            utilities.append(U)
        idx = np.argsort(utilities)[::-1][: self.params.top_k]
        top = [cands[i] for i in idx]
        scores = np.array([utilities[i] for i in idx], dtype=float)
        # softmax with temperature
        if self.params.tau <= 0:
            w = np.zeros_like(scores)
            if len(scores) > 0:
                w[0] = 1.0
        else:
            z = (scores - scores.max()) / self.params.tau
            e = np.exp(z)
            w = e / e.sum()
        return {top[i].agent_id: float(w[i]) for i in range(len(top))}


class DiversityRouter:
    """
    Encourages diversification across experts by adding a diversity bonus for
    distinct regime_tag groups. Still cost- and risk-aware.
    """
    def __init__(self, params: RouterParams | None = None) -> None:
        self.params = params or RouterParams(delta_div=0.1)

    def route(self, signals: List[SignalMessage]) -> Dict[str, float]:
        cands = [s for s in signals if s.capacity >= self.params.min_capacity and s.conf >= self.params.min_conf]
        if not cands:
            return {}
        # base utility
        base_utils: List[float] = []
        for s in cands:
            mu, var = s.dist.mean, s.dist.var
            U = mu - self.params.lambda_risk * var - self.params.gamma_cost * (s.cost_bps / 1e4)
            base_utils.append(U)
        # pick top_k greedily with diversity bonus
        selected: List[int] = []
        used_regimes: Dict[str, int] = {}
        remaining = set(range(len(cands)))
        while remaining and len(selected) < self.params.top_k:
            best_i = None
            best_score = -1e18
            for i in list(remaining):
                r = cands[i].regime_tag or "default"
                div_bonus = self.params.delta_div * (0 if used_regimes.get(r, 0) > 0 else 1)
                score = base_utils[i] + div_bonus
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            r = cands[best_i].regime_tag or "default"
            used_regimes[r] = used_regimes.get(r, 0) + 1
            remaining.remove(best_i)
        if not selected:
            return {}
        scores = np.array([base_utils[i] for i in selected], dtype=float)
        if self.params.tau <= 0:
            w = np.zeros_like(scores)
            if len(scores) > 0:
                w[0] = 1.0
        else:
            z = (scores - scores.max()) / self.params.tau
            e = np.exp(z)
            w = e / e.sum()
        return {cands[selected[i]].agent_id: float(w[i]) for i in range(len(selected))}


def make_router(policy: str, params: RouterParams | None = None):
    """Factory to construct router by policy name.

    Supported policies: "risk" (default), "diversity".
    """
    name = (policy or "risk").lower()
    if name in {"risk", "softmax", "riskaware"}:
        return RiskAwareRouter(params)
    if name in {"div", "diversity"}:
        return DiversityRouter(params)
    # fallback
    return RiskAwareRouter(params)
