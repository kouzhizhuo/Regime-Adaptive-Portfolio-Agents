from __future__ import annotations
from typing import Dict, Any, List
from rich import print
from .bus import MessageBus, SharedMemory, WeightedArbiter

class Coordinator:
    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.bus = MessageBus()
        self.memory = SharedMemory()
        self.registry: Dict[str, Any] = {}
        self.arbiter = WeightedArbiter(weights=weights)

    def register(self, name: str, agent: Any) -> None:
        self.registry[name] = agent
        if hasattr(agent, "register_handlers"):
            agent.register_handlers(self.bus)

    def run_pipeline(self, steps: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context)
        for step in steps:
            agent = self.registry.get(step)
            if not agent:
                raise ValueError(f"Agent '{step}' not registered")
            output = agent.execute(state)
            if output:
                state.update(output)
        return state

    def run_phase(self, publishers: List[str], topic: str, payload: Dict[str, Any]) -> float:
        proposals: Dict[str, float] = {}
        for name in publishers:
            agent = self.registry.get(name)
            if not agent or not hasattr(agent, "propose"):
                continue
            proposals[name] = float(agent.propose(payload))
        decision = self.arbiter.decide(proposals)
        self.memory.write(topic, {"proposals": proposals, "decision": decision})
        return decision
