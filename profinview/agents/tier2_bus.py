from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class Dist:
    mean: float
    var: float
    skew: float = 0.0
    kurt: float = 3.0

@dataclass
class SignalMessage:
    agent_id: str
    asset_id: str
    horizon: str
    ts: pd.Timestamp
    dist: Dist
    conf: float
    cost_bps: float
    capacity: float
    regime_tag: str
    explanation_ref: Optional[str] = None
    features_hash: Optional[str] = None
