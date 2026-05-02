from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import math

class SignalAgent:
    def register_handlers(self, bus):
        self.bus = bus
        # subscribe example for future extensions
        bus.subscribe("market:update", lambda msg: None)

    def _score_symbol(self, df: pd.DataFrame) -> float:
        close = df["close"].iloc[-1]
        ema50 = df.get("ema_50", df["close"]).iloc[-1]
        ema200 = df.get("ema_200", df["close"]).iloc[-1]
        rsi = df.get("rsi_14", pd.Series([50])).iloc[-1]
        macd_hist = (df.get("macd", df["close"]).iloc[-1] - df.get("macd_signal", df["close"]).iloc[-1])
        mom21 = df.get("mom_21", pd.Series([0.0])).iloc[-1]
        mom63 = df.get("mom_63", pd.Series([0.0])).iloc[-1]
        mom126 = df.get("mom_126", pd.Series([0.0])).iloc[-1]
        vol21 = df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02
        trend = 0.0
        trend += 0.5 if close > ema50 else -0.5
        trend += 0.5 if close > ema200 else -0.5
        trend += 0.3 if ema50 > ema200 else -0.3
        momentum = 0.6 * mom21 + 0.3 * mom63 + 0.1 * mom126
        rsi_adj = -0.2 if rsi > 80 else ( -0.1 if rsi > 70 else (0.1 if 45 <= rsi <= 70 else -0.1))
        macd_adj = 0.2 if macd_hist > 0 else -0.1
        vol_penalty = - min(0.5, max(0.0, vol21))
        score = 1.5 * momentum + trend + rsi_adj + macd_adj + vol_penalty
        return float(score)

    def propose(self, payload: Dict[str, Any]) -> float:
        df: pd.DataFrame = payload["df"]
        return self._score_symbol(df)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, pd.DataFrame] = state["features"]
        sentiment: Dict[str, float] = state.get("sentiment", {})
        signals: Dict[str, Dict[str, Any]] = {}
        for sym, df in features.items():
            s = self._score_symbol(df)
            s += 0.1 * float(sentiment.get(sym, 0.0))
            side = "BUY" if s > 0 else "HOLD"
            signals[sym] = {"score": float(s), "side": side}
            if hasattr(self, "bus"):
                self.bus.publish("signal:score", {"symbol": sym, "score": s})
        return {"signals": signals}
