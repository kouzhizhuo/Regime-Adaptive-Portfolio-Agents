from __future__ import annotations
from typing import Dict, Any, List, Tuple

class StrategyAgent:
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        signals = state["signals"]
        sizes = state.get("position_sizes", {})
        ranked: List[Tuple[str, float]] = sorted(
            [(s, v["score"]) for s, v in signals.items() if v["side"] == "BUY"],
            key=lambda x: x[1], reverse=True
        )
        picks = [sym for sym, _ in ranked[: self.top_n]]
        intents = [{"symbol": sym, "side": "BUY", "size": sizes.get(sym, 0.0)} for sym in picks]
        return {"intents": intents}
