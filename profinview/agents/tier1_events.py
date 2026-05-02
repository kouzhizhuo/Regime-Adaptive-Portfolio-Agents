from __future__ import annotations
from typing import Dict, Any
import pandas as pd

class EventsAgent:
    def __init__(self, gap_thr: float = 0.02, vol_mult: float = 2.0, boost: float = 0.05):
        self.gap_thr = gap_thr
        self.vol_mult = vol_mult
        self.boost = boost

    def event_boost(self, df: pd.DataFrame) -> float:
        # Assume df has columns: open, close, volume
        if df.shape[0] < 2:
            return 0.0
        dfl = df.copy()
        dfl = dfl.reset_index(drop=True)
        last = dfl.iloc[-1]
        prev = dfl.iloc[-2]
        gap = 0.0
        if 'open' in dfl.columns and 'close' in dfl.columns:
            if prev['close']:
                gap = float((last['open'] - prev['close']) / prev['close'])
        vol_spike = 0.0
        if 'volume' in dfl.columns:
            base = dfl['volume'].tail(21).median()
            if base:
                vol_spike = float(last['volume'] / base)
        boost = 0.0
        if gap >= self.gap_thr:
            boost += self.boost
        elif gap <= -self.gap_thr:
            boost -= self.boost
        if vol_spike >= self.vol_mult:
            boost += 0.5 * self.boost
        return float(boost)
