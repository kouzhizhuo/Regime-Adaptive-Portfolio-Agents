from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Bar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

class TimeSeries(BaseModel):
    symbol: str
    bars: List[Bar] = Field(default_factory=list)

class Message(BaseModel):
    sender: str
    receiver: str
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class Signal(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    strength: float
    reason: str

class PositionSizing(BaseModel):
    symbol: str
    size: float
    risk_score: float

class Order(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
