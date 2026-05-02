from __future__ import annotations
from typing import Callable, Dict, List, Any
from collections import defaultdict

class Message:
    def __init__(self, topic: str, payload: dict):
        self.topic = topic
        self.payload = payload

class MessageBus:
    def __init__(self) -> None:
        self.subscribers: Dict[str, List[Callable[[Message], Any]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Message], Any]) -> None:
        self.subscribers[topic].append(handler)

    def publish(self, topic: str, payload: dict) -> List[Any]:
        msg = Message(topic, payload)
        responses: List[Any] = []
        for handler in self.subscribers.get(topic, []):
            responses.append(handler(msg))
        return responses

    def broadcast(self, topics: List[str], payload: dict) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        for t in topics:
            out[t] = self.publish(t, payload)
        return out

class SharedMemory:
    def __init__(self) -> None:
        self.store: Dict[str, Any] = {}

    def write(self, key: str, value: Any) -> None:
        self.store[key] = value

    def read(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

class WeightedArbiter:
    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or {}

    def decide(self, proposals: Dict[str, float]) -> float:
        if not proposals:
            return 0.0
        # normalize weights
        keys = list(proposals.keys())
        w = [self.weights.get(k, 1.0) for k in keys]
        s = sum(w)
        if s <= 0:
            w = [1.0 for _ in keys]
            s = len(keys)
        w = [x / s for x in w]
        vals = [proposals[k] for k in keys]
        return float(sum(wi * vi for wi, vi in zip(w, vals)))
