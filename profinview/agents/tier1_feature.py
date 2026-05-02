from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

class FeatureAgent:
    def register_handlers(self, bus):
        pass

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        df = df.copy()
        # Core indicators
        df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
        df["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
        df["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
        macd = MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["rsi_14"] = RSIIndicator(close, window=14).rsi()
        # Volatility using ATR and returns std
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr_14"] = atr.average_true_range()
        ret = close.pct_change()
        df["vol_21"] = ret.rolling(21).std()
        # Momentum returns
        df["mom_21"] = close.pct_change(21)
        df["mom_63"] = close.pct_change(63)
        df["mom_126"] = close.pct_change(126)
        # Cleanup
        df = df.ffill().bfill()
        return df

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        md = state["market_data"]
        features = {sym: self._compute_indicators(df) for sym, df in md.items()}
        return {"features": features}
