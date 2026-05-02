from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from .tier2_bus import SignalMessage, Dist

class BayesianAggregator:
    def aggregate(self, msgs: List[SignalMessage], weights: Dict[str, float]) -> Dist:
        if not msgs:
            return Dist(mean=0.0, var=1.0)
        # normalize weights on provided agents
        w = np.array([weights.get(m.agent_id, 0.0) for m in msgs], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        means = np.array([m.dist.mean for m in msgs], dtype=float)
        vars_ = np.array([m.dist.var for m in msgs], dtype=float)
        # aggregate mean
        mu = float(np.dot(w, means))
        # aggregate variance: weighted sum of vars + inter-expert disagreement (shrinked)
        var_intra = float(np.dot(w, vars_))
        var_disagree = float(np.dot(w, (means - mu) ** 2))
        var = var_intra + 0.5 * var_disagree
        return Dist(mean=mu, var=max(var, 1e-6))


class PrecisionWeightedAggregator:
    """
    Aggregates experts by precision (1/variance). Experts with lower uncertainty
    receive higher weights. Falls back to equal weighting if variances are invalid.
    """
    def aggregate(self, msgs: List[SignalMessage], weights: Dict[str, float]) -> Dist:
        if not msgs:
            return Dist(mean=0.0, var=1.0)
        vars_ = np.array([max(1e-6, float(m.dist.var)) for m in msgs], dtype=float)
        prec = 1.0 / vars_
        # optional external weights act as priors
        w_ext = np.array([float(weights.get(m.agent_id, 1.0)) for m in msgs], dtype=float)
        w = prec * w_ext
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        means = np.array([float(m.dist.mean) for m in msgs], dtype=float)
        mu = float(np.dot(w, means))
        # posterior variance ~ 1/sum(prec), adjusted by disagreement
        var_intra = float(1.0 / max(1e-6, prec.sum()))
        var_disagree = float(np.dot(w, (means - mu) ** 2))
        var = var_intra + 0.25 * var_disagree
        return Dist(mean=mu, var=max(var, 1e-6))


class MedianAggregator:
    """
    Robust aggregation by median of means; variance estimated from MAD.
    """
    def aggregate(self, msgs: List[SignalMessage], weights: Dict[str, float]) -> Dist:
        if not msgs:
            return Dist(mean=0.0, var=1.0)
        means = np.array([float(m.dist.mean) for m in msgs], dtype=float)
        mu = float(np.median(means))
        # Median Absolute Deviation as robust spread -> convert to variance proxy
        mad = float(np.median(np.abs(means - mu)))
        var = (1.4826 * mad) ** 2 if not np.isnan(mad) else 1.0
        return Dist(mean=mu, var=max(var, 1e-6))


def make_aggregator(policy: str):
    """Factory for aggregator by policy name.

    Supported: "bayes" (default), "precision", "median".
    """
    name = (policy or "bayes").lower()
    if name in {"bayes", "bayesian", "default"}:
        return BayesianAggregator()
    if name in {"precision", "prec", "pw"}:
        return PrecisionWeightedAggregator()
    if name in {"median", "med"}:
        return MedianAggregator()
    return BayesianAggregator()
