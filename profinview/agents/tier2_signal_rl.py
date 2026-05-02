from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class RLSignalAgent:
    def __init__(self, feature_names: List[str] | None = None, ridge_alpha: float = 1.0):
        self.feature_names = feature_names or [
            "mom_21", "mom_63", "mom_126", "rsi_14", "macd", "macd_signal", "sma_20", "ema_50", "ema_200", "vol_21", "atr_14"
        ]
        self.alpha = ridge_alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.bus = None

    def register_handlers(self, bus):
        self.bus = bus

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        vals = []
        for name in self.feature_names:
            if name in df.columns:
                vals.append(float(df[name].iloc[-1]))
            else:
                vals.append(0.0)
        return np.array(vals, dtype=float)

    def fit(self, symbol_to_df: Dict[str, pd.DataFrame], lookahead: int = 5) -> None:
        X_list: List[np.ndarray] = []
        y_list: List[float] = []
        for sym, df in symbol_to_df.items():
            if len(df) < 250:
                continue
            close = df["close"].values
            future = np.roll(close, -lookahead)
            future[-lookahead:] = np.nan
            ret_fwd = future / close - 1.0
            for i in range(len(df) - lookahead - 200, len(df) - lookahead):
                if i <= 0:
                    continue
                row = df.iloc[: i]
                feats = self._extract_features(row)
                y = float(ret_fwd[i]) if not np.isnan(ret_fwd[i]) else None
                if y is None:
                    continue
                X_list.append(feats)
                y_list.append(y)
        if not X_list:
            self.coef_ = np.zeros(len(self.feature_names))
            self.intercept_ = 0.0
            return
        X = np.vstack(X_list)
        y = np.array(y_list)
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = float(y.mean())

    def predict_score(self, df: pd.DataFrame) -> float:
        feats = self._extract_features(df)
        if self.coef_ is None:
            return 0.0
        score = float(self.intercept_ + feats @ self.coef_)
        vol = float(df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02)
        return score - 0.5 * vol

    def propose(self, payload: Dict[str, Any]) -> float:
        df: pd.DataFrame = payload["df"]
        s = self.predict_score(df)
        if self.bus:
            self.bus.publish("signal_rl:score", {"symbol": payload.get("symbol"), "score": s})
        return s

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, pd.DataFrame] = state["features"]
        signals: Dict[str, Dict[str, Any]] = {}
        for sym, df in features.items():
            s = self.predict_score(df)
            side = "BUY" if s > 0 else "HOLD"
            signals[sym] = {"score": float(s), "side": side}
            if self.bus:
                self.bus.publish("signal_rl:score", {"symbol": sym, "score": s})
        return {"signals": signals}
