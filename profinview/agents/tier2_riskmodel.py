from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

class RiskModelAgent:
    def __init__(self, window: int = 63, shrink: float = 0.1):
        self.window = window
        self.shrink = shrink

    def estimate_cov(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # align last window closes
        df_map = {}
        for sym, df in prices.items():
            s = df["close"].tail(self.window + 1).pct_change().dropna().reset_index(drop=True)
            df_map[sym] = s
        if not df_map:
            return pd.DataFrame()
        R = pd.DataFrame(df_map)
        S = np.cov(R.values, rowvar=False)
        # shrink to diagonal
        diag = np.diag(np.diag(S))
        Sigma = (1 - self.shrink) * S + self.shrink * diag
        return pd.DataFrame(Sigma, index=R.columns, columns=R.columns)
