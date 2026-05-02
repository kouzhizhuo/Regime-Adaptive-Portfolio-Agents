from __future__ import annotations
from typing import Dict, Any, List

class ExecutionAgent:
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intents: List[Dict[str, Any]] = state.get("intents", [])
        last_prices = {s: float(df["close"].iloc[-1]) for s, df in state["market_data"].items()}
        orders = []
        for intent in intents:
            sym = intent["symbol"]
            size = float(intent.get("size", 0.0))
            price = last_prices.get(sym)
            qty = size / price if price else 0.0
            if qty > 0:
                orders.append({"symbol": sym, "side": "BUY", "quantity": round(qty, 4), "price": price})
        return {"orders": orders}
