from __future__ import annotations
from typing import Dict, Any
import pandas as pd

class AnalysisAgent:
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, pd.DataFrame] = state["features"]
        analysis: Dict[str, Dict[str, float]] = {}
        for sym, df in features.items():
            ret = df["close"].pct_change().dropna()
            analysis[sym] = {
                "mean_return": float(ret.mean()),
                "volatility": float(ret.std()),
                "rsi_last": float(df["rsi_14"].iloc[-1]),
                "macd_hist_last": float((df["macd"] - df["macd_signal"]).iloc[-1]),
            }
        return {"analysis": analysis}
