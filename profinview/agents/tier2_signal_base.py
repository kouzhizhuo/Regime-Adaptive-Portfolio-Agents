from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from .tier2_bus import Dist

class BaseSignalAgent:
    agent_id: str = "base"

    def predict_dist(self, df: pd.DataFrame) -> Dist:
        raise NotImplementedError

    def estimate_costs(self, asset_id: str, size: float) -> float:
        return 0.0

    def capacity(self, asset_id: str) -> float:
        return 1e9
