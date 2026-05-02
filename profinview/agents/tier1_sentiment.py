from __future__ import annotations
import os
from typing import Dict, Any
import requests

API_URL = "https://api3.xhub.chat/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a financial analyst. Given a ticker symbol, return a sentiment score between -1 and 1 "
    "reflecting overall recent news sentiment. Respond ONLY with a float value."
)

class SentimentAgent:
    def __init__(self) -> None:
        self.api_key = os.getenv("XHUB_API_KEY")

    def register_handlers(self, bus):
        pass

    def _query_sentiment(self, symbol: str) -> float:
        if not self.api_key:
            return 0.0
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Symbol: {symbol}"},
            ],
        }
        try:
            resp = requests.post(API_URL, json=body, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            return float(text)
        except Exception:
            return 0.0

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symbols = state["symbols"]
        sentiments = {s: self._query_sentiment(s) for s in symbols}
        return {"sentiment": sentiments}
