from __future__ import annotations
from typing import Dict, Any
import pandas as pd


class MASignalAgent:
    """
    Simple moving-average based signal.

    Score heuristic:
    - +1.0 if EMA50 > EMA200 and close > EMA50
    - +0.5 if EMA50 > EMA200
    - -0.5 if EMA50 < EMA200
    - -1.0 if EMA50 < EMA200 and close < EMA50
    """

    def register_handlers(self, bus):
        self.bus = bus

    def _score_symbol(self, df: pd.DataFrame) -> float:
        ema50 = df.get("ema_50", df["close"]).iloc[-1]
        ema200 = df.get("ema_200", df["close"]).iloc[-1]
        close = df["close"].iloc[-1]
        if ema50 > ema200 and close > ema50:
            return 1.0
        if ema50 > ema200:
            return 0.5
        if ema50 < ema200 and close < ema50:
            return -1.0
        return -0.5

    def propose(self, payload: Dict[str, Any]) -> float:
        df: pd.DataFrame = payload["df"]
        return float(self._score_symbol(df))

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, pd.DataFrame] = state["features"]
        signals: Dict[str, Dict[str, Any]] = {}
        for sym, df in features.items():
            s = float(self._score_symbol(df))
            side = "BUY" if s > 0 else "HOLD"
            signals[sym] = {"score": s, "side": side}
            if hasattr(self, "bus"):
                self.bus.publish("signal:score", {"symbol": sym, "score": s})
        return {"signals": signals}





