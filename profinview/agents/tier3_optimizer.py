from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class OptimizerAgent:
    def __init__(
        self,
        risk_aversion: float = 5.0,
        max_weight: float = 0.05,
        shrink_lambda: float = 1e-4,
        turnover_penalty: float = 1.0,
        use_mu_zscore: bool = True,
    ):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.shrink_lambda = shrink_lambda
        self.turnover_penalty = turnover_penalty
        self.use_mu_zscore = use_mu_zscore

    def _prepare_mu(self, mu: Dict[str, float], symbols: List[str]) -> np.ndarray:
        m = np.array([mu[s] for s in symbols], dtype=float)
        if self.use_mu_zscore:
            mu_mean = m.mean()
            mu_std = m.std() + 1e-9
            m = (m - mu_mean) / mu_std
        # clip extreme negatives to avoid pathological shorts (we do long-only)
        m = np.maximum(m, 0.0)
        return m

    def _project_caps(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, self.max_weight)
        s = w.sum()
        if s > 0:
            w = w / s
        return w

    def optimize(
        self,
        mu: Dict[str, float],
        cov: pd.DataFrame,
        top_n: Optional[int] = None,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if cov is None or cov.empty or not mu:
            return {}
        symbols = list(mu.keys())
        # select top_n by raw mu for universe reduction
        if top_n and len(symbols) > top_n:
            symbols = sorted(symbols, key=lambda s: mu[s], reverse=True)[: top_n]
        m = self._prepare_mu(mu, symbols)
        Sigma = cov.loc[symbols, symbols].values
        # regularize covariance more strongly for stability
        n = Sigma.shape[0]
        Sigma_reg = Sigma + (self.shrink_lambda + 1e-6) * np.eye(n)
        # mean-variance closed-form direction
        inv = np.linalg.inv(Sigma_reg + (self.risk_aversion * 1e-6) * np.eye(n))
        w_raw = inv @ m
        w_raw = self._project_caps(w_raw)
        # turnover-aware blend with current weights
        if current_weights:
            w_curr = np.array([max(0.0, current_weights.get(s, 0.0)) for s in symbols], dtype=float)
            if w_curr.sum() > 0:
                w_curr = w_curr / w_curr.sum()
            k = max(0.0, self.turnover_penalty)
            w = (w_raw + k * w_curr) / (1.0 + k)
            w = self._project_caps(w)
        else:
            w = w_raw
        return {s: float(wi) for s, wi in zip(symbols, w)}
