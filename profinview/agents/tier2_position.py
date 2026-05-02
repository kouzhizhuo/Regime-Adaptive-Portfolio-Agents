from __future__ import annotations
from typing import Dict, Any
import pandas as pd

class PositionAgent:
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, pd.DataFrame] = state["features"]
        signals = state["signals"]
        capital: float = float(state.get("capital", 100000.0))
        max_weight = float(state.get("max_weight", 0.05))
        # inverse volatility weights for BUY picks
        inv_vol: Dict[str, float] = {}
        for sym, df in features.items():
            if signals.get(sym, {}).get("side") != "BUY":
                continue
            vol = df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02
            inv_vol[sym] = 1.0 / max(1e-4, vol)
        if not inv_vol:
            return {"position_sizes": {}}
        # Convert to weights with cap
        total = sum(inv_vol.values())
        weights = {s: min(max_weight, inv_vol[s] / total) for s in inv_vol}
        # Renormalize if needed
        wsum = sum(weights.values())
        if wsum > 0:
            weights = {s: w / wsum for s, w in weights.items()}
        # Dollar sizes
        sizes = {s: float(capital * w) for s, w in weights.items()}
        return {"position_sizes": sizes}
