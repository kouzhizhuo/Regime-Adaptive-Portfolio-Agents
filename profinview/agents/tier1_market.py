from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
from profinview.data.loader import get_loader_for_dir

class MarketDataAgent:
    def __init__(self, data_dir: str) -> None:
        self.loader = get_loader_for_dir(data_dir)

    def register_handlers(self, bus):
        pass

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symbols: List[str] = state["symbols"]
        lookback_days: int = state.get("lookback_days", 180)
        raw = self.loader.load(symbols)
        sliced: Dict[str, pd.DataFrame] = {}
        for sym, df in raw.items():
            if lookback_days and len(df) > lookback_days:
                sliced[sym] = df.iloc[-lookback_days:].reset_index(drop=True)
            else:
                sliced[sym] = df
        return {"market_data": sliced}
