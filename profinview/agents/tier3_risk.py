from __future__ import annotations
from typing import Dict, Any
import math

class RiskAgent:
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        orders = state.get("orders", [])
        md = state.get("market_data", {})
        exposure = 0.0
        for o in orders:
            price = float(o.get("price", 0.0))
            exposure += price * float(o.get("quantity", 0.0))
        # crude risk score: sqrt(num positions) scaled by exposure
        risk_score = math.sqrt(max(1, len(orders))) * (exposure / max(1.0, state.get("capital", 100000.0)))
        return {"risk_summary": {"exposure": exposure, "positions": len(orders), "risk_score": risk_score}}
