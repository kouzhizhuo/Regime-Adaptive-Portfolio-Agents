from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

class AlphaAgent:
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out['close']
        if 'volume' in out.columns:
            vol = out['volume']
        else:
            vol = pd.Series(np.nan, index=out.index)
        out['ret_1'] = close.pct_change(1)
        out['ret_5'] = close.pct_change(5)
        out['ret_20'] = close.pct_change(20)
        out['vol_ratio_5'] = vol / vol.rolling(5).mean()
        out['price_range_5'] = (out.get('high', close).rolling(5).max() - out.get('low', close).rolling(5).min()) / close
        out = out.ffill().bfill()
        return out
