from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class ForecasterAgent:
    def __init__(self, feature_names: List[str] | None = None, ridge_alpha: float = 10.0, horizon: int = 5):
        self.feature_names = feature_names or [
            "mom_21", "mom_63", "mom_126", "rsi_14", "macd", "macd_signal", "sma_20", "ema_50", "ema_200", "vol_21", "atr_14"
        ]
        self.alpha = ridge_alpha
        self.horizon = horizon
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def _row_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        vals = []
        row = df.iloc[idx]
        for name in self.feature_names:
            vals.append(float(row[name]) if name in df.columns else 0.0)
        return np.array(vals, dtype=float)

    def fit(self, features_full: Dict[str, pd.DataFrame], train_start: str | None, train_end: str | None) -> None:
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        for sym, df in features_full.items():
            sdf = df
            if train_start or train_end:
                sdf = sdf[(sdf["date"] >= (pd.to_datetime(train_start).date() if train_start else sdf["date"].min())) & (sdf["date"] <= (pd.to_datetime(train_end).date() if train_end else sdf["date"].max()))]
            if len(sdf) <= self.horizon + 200:
                continue
            sdf = sdf.reset_index(drop=True)
            close = sdf["close"].values
            future = np.roll(close, -self.horizon)
            future[-self.horizon:] = np.nan
            ret_fwd = future / close - 1.0
            for i in range(200, len(sdf) - self.horizon):
                feats = self._row_features(sdf, i)
                y = ret_fwd[i]
                if np.isnan(y):
                    continue
                X_list.append(feats)
                y_list.append(float(y))
        if not X_list:
            self.coef_ = np.zeros(len(self.feature_names))
            self.intercept_ = 0.0
            return
        X = np.vstack(X_list)
        y = np.array(y_list)
        n = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n)
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = float(y.mean())

    def predict_mu(self, window_features: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        mu: Dict[str, float] = {}
        for sym, df in window_features.items():
            row = df.iloc[-1]
            feats = np.array([float(row.get(name, 0.0)) for name in self.feature_names])
            if self.coef_ is None:
                mu[sym] = 0.0
            else:
                mu[sym] = float(self.intercept_ + feats @ self.coef_)
        return mu
