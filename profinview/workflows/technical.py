from __future__ import annotations
from typing import Dict, Any, List
from rich import print

from profinview.core.coordinator import Coordinator
from profinview.agents.tier1_market import MarketDataAgent
from profinview.agents.tier1_feature import FeatureAgent
from profinview.agents.tier1_sentiment import SentimentAgent
from profinview.agents.tier2_analysis import AnalysisAgent
from profinview.agents.tier2_signal import SignalAgent
from profinview.agents.tier2_position import PositionAgent
from profinview.agents.tier3_strategy import StrategyAgent
from profinview.agents.tier3_execution import ExecutionAgent
from profinview.agents.tier3_risk import RiskAgent


def run_technical_workflow(data_dir: str, symbols: List[str], lookback_days: int = 180, verbose: bool = True) -> Dict[str, Any]:
    coord = Coordinator()

    # Register agents by tier
    coord.register("market", MarketDataAgent(data_dir))
    coord.register("feature", FeatureAgent())
    coord.register("sentiment", SentimentAgent())

    coord.register("analysis", AnalysisAgent())
    coord.register("signal", SignalAgent())
    coord.register("position", PositionAgent())

    coord.register("strategy", StrategyAgent(top_n=5))
    coord.register("execution", ExecutionAgent())
    coord.register("risk", RiskAgent())

    steps = [
        "market",
        "feature",
        "sentiment",
        "analysis",
        "signal",
        "position",
        "strategy",
        "execution",
        "risk",
    ]

    context = {
        "symbols": symbols,
        "lookback_days": lookback_days,
        "capital": 100000.0,
        "risk_per_trade": 0.01,
    }

    result = coord.run_pipeline(steps, context)

    if verbose:
        summary = {
            "signals": result.get("signals"),
            "orders": result.get("orders"),
            "risk": result.get("risk_summary"),
        }
        print({"summary": summary})

    return result
