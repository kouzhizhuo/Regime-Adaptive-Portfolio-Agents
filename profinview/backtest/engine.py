from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

from profinview.agents.tier1_feature import FeatureAgent
from profinview.agents.tier1_sentiment import SentimentAgent
from profinview.agents.tier2_signal import SignalAgent
from profinview.agents.tier2_position import PositionAgent
from profinview.agents.tier3_strategy import StrategyAgent
from profinview.agents.tier2_signal_rl import RLSignalAgent
from profinview.core.coordinator import Coordinator
from profinview.agents.tier2_forecaster import ForecasterAgent
from profinview.agents.tier2_riskmodel import RiskModelAgent
from profinview.agents.tier3_optimizer import OptimizerAgent
from profinview.agents.tier2_bus import SignalMessage, Dist
from profinview.agents.tier2_router import RiskAwareRouter, RouterParams, make_router
from profinview.agents.tier2_auction import BayesianAggregator, make_aggregator
from profinview.agents.tier1_events import EventsAgent
from profinview.agents.tier2_alpha import AlphaAgent

from profinview.data.loader import SP500JsonLoader, get_loader_for_dir


@dataclass
class BacktestConfig:
    data_dir: str
    symbols: List[str]
    lookback_days: int = 180
    capital: float = 100000.0
    risk_per_trade: float = 0.01
    top_n: int = 5
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    rebalance_freq: str = "W-FRI"
    rl: bool = False
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    entry_thresh: float = 0.05
    exit_thresh: float = -0.05
    take_profit: float = 0.15
    stop_loss: float = 0.08
    max_weight_per_symbol: float = 0.08
    optimizer: bool = False
    consensus: bool = False
    router_policy: str = "risk"
    aggregator_policy: str = "bayes"
    router_top_k: int = 3
    router_min_conf: float = 0.05
    signal_cost_bps: float = 5.0
    consensus_blend: float = 1.0


class BacktestRunner:
    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg
        self.loader = get_loader_for_dir(cfg.data_dir)
        self.feature_agent = FeatureAgent()
        self.sentiment_agent = SentimentAgent()
        self.signal_agent = SignalAgent()
        self.position_agent = PositionAgent()
        self.strategy_agent = StrategyAgent(top_n=cfg.top_n)

    def _prepare_data(self) -> Dict[str, pd.DataFrame]:
        raw = self.loader.load(self.cfg.symbols)
        for sym, df in raw.items():
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date
            raw[sym] = df
        return raw

    def _common_dates(self, md: Dict[str, pd.DataFrame]) -> List[str]:
        date_sets = []
        for df in md.values():
            date_sets.append(set(df["date"].tolist()))
        common = set.intersection(*date_sets) if date_sets else set()
        return sorted(str(d) for d in common)

    def _baseline_buy_and_hold(self, md: Dict[str, pd.DataFrame], index: List[str]) -> pd.Series:
        aligned = {}
        for sym, df in md.items():
            s = df.set_index(df["date"].astype(str))["close"].reindex(index).ffill()
            aligned[sym] = s
        prices = pd.DataFrame(aligned, index=index).dropna(how="all")
        if prices.empty:
            return pd.Series([], dtype=float)
        weights = np.repeat(1.0 / prices.shape[1], prices.shape[1])
        start_prices = prices.iloc[0]
        shares = (self.cfg.capital * weights) / start_prices.values
        portfolio_values = (prices * shares).sum(axis=1)
        portfolio_values.index = [str(i) for i in portfolio_values.index]
        return portfolio_values

    def run(self, results_dir: str, make_plots: bool = True) -> Dict[str, Any]:
        md_full = self._prepare_data()
        all_dates = sorted({d for df in md_full.values() for d in df["date"].tolist()})
        if len(all_dates) <= self.cfg.lookback_days:
            raise ValueError("Insufficient data for lookback window")

        # Robustness controls (env-driven): seed, prediction noise, mu scaling, missingness
        try:
            seed_env = os.getenv('PROFINVIEW_SEED', '')
            seed_val = int(seed_env) if seed_env not in {None, ''} else 0
        except Exception:
            seed_val = 0
        if seed_val:
            try:
                np.random.seed(seed_val)
            except Exception:
                pass
        try:
            noise_std = float(os.getenv('PROFINVIEW_ROBUST_NOISE_STD', '0.0') or '0.0')
        except Exception:
            noise_std = 0.0
        try:
            mu_scale = float(os.getenv('PROFINVIEW_ROBUST_MU_SCALE', '1.0') or '1.0')
        except Exception:
            mu_scale = 1.0
        try:
            drop_pct = float(os.getenv('PROFINVIEW_ROBUST_DROP_PCT', '0.0') or '0.0')
            drop_pct = max(0.0, min(1.0, drop_pct))
        except Exception:
            drop_pct = 0.0

        # Precompute features
        features_full: Dict[str, pd.DataFrame] = {}
        for sym, df in md_full.items():
            fdf = self.feature_agent._compute_indicators(df)
            # optional alpha expansion
            if os.getenv('PROFINVIEW_ALPHAS','0') == '1':
                fdf = AlphaAgent().compute(fdf)
            features_full[sym] = fdf

        # RL training
        rl_agent: RLSignalAgent | None = None
        if self.cfg.rl:
            rl_agent = RLSignalAgent()
            train_data: Dict[str, pd.DataFrame] = {}
            for sym, fdf in features_full.items():
                sdf = fdf
                if self.cfg.train_start or self.cfg.train_end:
                    sdf = sdf[(sdf["date"] >= (pd.to_datetime(self.cfg.train_start).date() if self.cfg.train_start else sdf["date"].min())) & (sdf["date"] <= (pd.to_datetime(self.cfg.train_end).date() if self.cfg.train_end else sdf["date"].max()))]
                train_data[sym] = sdf.reset_index(drop=True)
            rl_agent.fit(train_data, lookahead=5)

        cash = self.cfg.capital
        positions: Dict[str, float] = {s: 0.0 for s in self.cfg.symbols}
        avg_cost: Dict[str, float] = {s: 0.0 for s in self.cfg.symbols}
        peak_price: Dict[str, float] = {s: 0.0 for s in self.cfg.symbols}
        equity_curve = []
        weights_history: List[Dict[str, Any]] = []
        predictions_history: List[Dict[str, Any]] = []
        consensus_diag: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        sentiments = {s: 0.0 for s in self.cfg.symbols}
        daily_meta: List[Dict[str, Any]] = []

        # Coordinator for arbitration
        coord = Coordinator(weights={"rl": 0.6, "heur": 0.4})
        if rl_agent:
            coord.register("rl", rl_agent)
        coord.register("heur", self.signal_agent)

        # weekly flags for optimizer rebalances
        idx_dates = pd.to_datetime(all_dates)
        weekly_flags = pd.Series(1, index=idx_dates).resample("W-FRI").last().index.date
        weekly_set = set(weekly_flags)

        # set up optimizer agents
        alpha_mode = os.getenv('PROFINVIEW_ALPHAS','0') == '1'
        if alpha_mode:
            feat_names = [
                "mom_21","mom_63","mom_126","rsi_14","macd","macd_signal","sma_20","ema_50","ema_200","vol_21","atr_14",
                "ret_1","ret_5","ret_20","vol_ratio_5","price_range_5"
            ]
            forecaster = ForecasterAgent(feature_names=feat_names)
        else:
            forecaster = ForecasterAgent()
        riskmodel = RiskModelAgent()
        optimizer = OptimizerAgent(risk_aversion=5.0, max_weight=self.cfg.max_weight_per_symbol)
        if self.cfg.optimizer or self.cfg.consensus:
            forecaster.fit(features_full, self.cfg.train_start, self.cfg.train_end)
        router = make_router(self.cfg.router_policy, RouterParams(min_conf=self.cfg.router_min_conf, top_k=self.cfg.router_top_k, lambda_risk=0.5, gamma_cost=0.2, tau=0.7, delta_div=0.1))
        aggregator = make_aggregator(self.cfg.aggregator_policy)

        ev_agent = EventsAgent()
        for idx in range(self.cfg.lookback_days, len(all_dates)):
            today = all_dates[idx]
            window_features: Dict[str, pd.DataFrame] = {}
            last_prices: Dict[str, float] = {}
            for sym, fdf in features_full.items():
                window = fdf[fdf["date"] <= today].tail(self.cfg.lookback_days)
                if window.empty:
                    continue
                window_features[sym] = window.reset_index(drop=True)
                last_prices[sym] = float(window["close"].iloc[-1])
            if not window_features:
                continue

            portfolio_equity = cash + sum(positions[s] * last_prices.get(s, 0.0) for s in positions)
            equity_curve.append({"date": str(today), "equity": float(portfolio_equity)})

            # Combined signals via arbiter
            signals: Dict[str, Dict[str, Any]] = {}
            for sym, df in window_features.items():
                payload = {"df": df, "symbol": sym}
                proposals: Dict[str, float] = {}
                if rl_agent is not None:
                    proposals["rl"] = rl_agent.propose(payload)
                proposals["heur"] = self.signal_agent.propose(payload)
                score = coord.arbiter.decide(proposals)
                if os.getenv('PROFINVIEW_EVENTS','0') == '1':
                    score += ev_agent.event_boost(features_full[sym])
                # Apply robustness: scale, noise, and optional missingness
                score = (float(score) * mu_scale) + (np.random.normal(0.0, noise_std) if noise_std > 0 else 0.0)
                if drop_pct > 0.0 and (np.random.rand() < drop_pct):
                    # simulate missing prediction by skipping this symbol
                    continue
                signals[sym] = {"score": float(score)}
            # record predictions (per-symbol scores) for diagnostics
            predictions_history.append({"date": str(today), "preds": {s: float(v["score"]) for s, v in signals.items()}})

            # Optimizer/consensus mode: trade only on rebalance days; skip event-driven entries entirely
            if (self.cfg.optimizer or self.cfg.consensus):
                if today not in weekly_set:
                    # record weights and move on without trading
                    pe2 = cash + sum(positions.get(s, 0.0) * last_prices.get(s, 0.0) for s in positions)
                    if pe2 > 0:
                        w_today = {s: (positions.get(s, 0.0) * last_prices.get(s, 0.0)) / pe2 for s in positions}
                    else:
                        w_today = {s: 0.0 for s in positions}
                    weights_history.append({"date": str(today), "weights": w_today})
                    continue

            # If optimizer mode and weekly day: compute target weights and trade towards targets
            if (self.cfg.optimizer or self.cfg.consensus) and today in weekly_set and window_features:
                if self.cfg.consensus:
                    # build expert messages per symbol
                    msgs_per_sym = {}
                    for sym, df in window_features.items():
                        l = []
                        # heuristic
                        heur_score = self.signal_agent.propose({"df": df, "symbol": sym})
                        vol = float(df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02)
                        var = max(1e-6, vol * vol)
                        conf_h = float(1.0 / (1.0 + 3.0 * (var ** 0.5)))
                        l.append(SignalMessage(
                            agent_id="heur", asset_id=sym, horizon="1w", ts=pd.to_datetime(today),
                            dist=Dist(mean=float(heur_score), var=float(var)),
                            conf=conf_h, cost_bps=self.cfg.signal_cost_bps, capacity=1e7, regime_tag="default"
                        ))
                        if rl_agent is not None:
                            rl_score = rl_agent.propose({"df": df, "symbol": sym})
                            conf_r = float(1.0 / (1.0 + 3.0 * (var ** 0.5)))
                            l.append(SignalMessage(
                                agent_id="rl", asset_id=sym, horizon="1w", ts=pd.to_datetime(today),
                                dist=Dist(mean=float(rl_score), var=float(var)),
                                conf=conf_r, cost_bps=self.cfg.signal_cost_bps, capacity=1e7, regime_tag="default"
                            ))
                        # route to get weights then aggregate
                        weights = router.route(l)
                        consensus = aggregator.aggregate(l, weights)
                        msgs_per_sym[sym] = consensus
                        consensus_diag.append({
                            "date": str(today),
                            "symbol": sym,
                            "router": self.cfg.router_policy,
                            "aggregator": self.cfg.aggregator_policy,
                            "weights": {m.agent_id: float(weights.get(m.agent_id, 0.0)) for m in l},
                            "mu": float(consensus.mean),
                            "var": float(consensus.var),
                        })
                    # Apply robustness: scale, noise, and optional missingness on consensus mu
                    mu = {}
                    blend = float(self.cfg.consensus_blend)
                    for sym, d in msgs_per_sym.items():
                        if drop_pct > 0.0 and (np.random.rand() < drop_pct):
                            continue
                        agg_mu = (float(d.mean) * mu_scale) + (np.random.normal(0.0, noise_std) if noise_std > 0 else 0.0)
                        if blend < 1.0:
                            base_mu = forecaster.predict_mu({sym: window_features[sym]}).get(sym, agg_mu)
                            agg_mu = blend * agg_mu + (1.0 - blend) * float(base_mu)
                        mu[sym] = agg_mu
                else:
                    mu = forecaster.predict_mu(window_features)
                    # Apply robustness for forecaster path as well
                    if isinstance(mu, dict):
                        mu2 = {}
                        for sym, v in mu.items():
                            if drop_pct > 0.0 and (np.random.rand() < drop_pct):
                                continue
                            vv = (float(v) * mu_scale) + (np.random.normal(0.0, noise_std) if noise_std > 0 else 0.0)
                            mu2[sym] = vv
                        mu = mu2
                cov = riskmodel.estimate_cov({s: features_full[s] for s in window_features})
                # regime-aware risk aversion and caps
                vols = [float(df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02) for df in window_features.values()]
                avg_vol = float(np.mean(vols)) if vols else 0.02
                if avg_vol > 0.02:
                    ra = 8.0
                    cap = 0.03
                else:
                    ra = 2.0
                    cap = 0.04
                opt = OptimizerAgent(risk_aversion=ra, max_weight=cap, turnover_penalty=2.0)
                curr_w = None
                weights = opt.optimize(mu, cov, top_n=self.strategy_agent.top_n, current_weights=curr_w)
                # translate weights to dollar targets
                targets = {s: float(portfolio_equity * w) for s, w in weights.items()}
                # sell/trim others
                for sym, qty in list(positions.items()):
                    if qty <= 0:
                        continue
                    price = last_prices.get(sym)
                    tgt = targets.get(sym, 0.0)
                    cur = qty * price if price else 0.0
                    diff = cur - tgt
                    if diff > 1e-6 and price:
                        sell_qty = diff / price
                        positions[sym] -= sell_qty
                        if positions[sym] <= 1e-8:
                            positions[sym] = 0.0
                            avg_cost[sym] = 0.0
                        cash += sell_qty * price
                        trades.append({"date": str(today), "symbol": sym, "side": "SELL", "price": float(price), "quantity": float(sell_qty)})
                # buy to targets
                for sym, tgt in targets.items():
                    price = last_prices.get(sym)
                    cur = positions.get(sym, 0.0) * price if price else 0.0
                    diff = tgt - cur
                    if price and diff > 1e-6 and cash > 0:
                        buy = min(diff, cash)
                        qty = buy / price
                        prev_qty = positions.get(sym, 0.0)
                        new_qty = prev_qty + qty
                        new_cost = (prev_qty * avg_cost.get(sym, price) + qty * price) / new_qty if new_qty > 0 else 0.0
                        positions[sym] = new_qty
                        avg_cost[sym] = new_cost
                        cash -= buy
                        trades.append({"date": str(today), "symbol": sym, "side": "BUY", "price": float(price), "quantity": float(qty)})
                # record end-of-day weights and continue
                pe2 = cash + sum(positions.get(s, 0.0) * last_prices.get(s, 0.0) for s in positions)
                if pe2 > 0:
                    w_today = {s: (positions.get(s, 0.0) * last_prices.get(s, 0.0)) / pe2 for s in positions}
                else:
                    w_today = {s: 0.0 for s in positions}
                weights_history.append({"date": str(today), "weights": w_today})
                continue

            # regime-aware thresholds
            vols_today = [float(df.get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02) for df in window_features.values()]
            avg_vol = float(np.mean(vols_today)) if vols_today else 0.02
            daily_meta.append({"date": str(today), "avg_vol": avg_vol})
            entry_th = self.cfg.entry_thresh if avg_vol <= 0.02 else max(self.cfg.entry_thresh * 1.5, self.cfg.entry_thresh + 0.02)
            exit_th = self.cfg.exit_thresh if avg_vol <= 0.02 else min(self.cfg.exit_thresh * 1.5, self.cfg.exit_thresh - 0.02)
            trail_pct = 0.08 if avg_vol <= 0.02 else 0.12

            # Exits first: score below exit threshold or TP/SL or trailing stop
            for sym, qty in list(positions.items()):
                if qty <= 0:
                    continue
                price = last_prices.get(sym)
                if not price:
                    continue
                entry = avg_cost.get(sym, price)
                pnl = (price - entry) / entry if entry else 0.0
                score = signals.get(sym, {"score": 0.0}).get("score", 0.0)
                # trailing peak update
                peak_price[sym] = max(peak_price.get(sym, 0.0), float(price))
                if (score <= exit_th) or (pnl >= self.cfg.take_profit) or (pnl <= -self.cfg.stop_loss) or (price <= peak_price.get(sym, price) * (1.0 - trail_pct)):
                    sell_qty = qty
                    positions[sym] = 0.0
                    avg_cost[sym] = 0.0
                    peak_price[sym] = 0.0
                    cash += sell_qty * price
                    trades.append({"date": str(today), "symbol": sym, "side": "SELL", "price": float(price), "quantity": float(sell_qty)})

            # Entries next: top-N candidates above entry threshold
            ranked_syms = sorted(signals.keys(), key=lambda s: signals[s]["score"], reverse=True)
            candidates = [s for s in ranked_syms if signals[s]["score"] >= entry_th]
            if self.strategy_agent.top_n:
                candidates = candidates[: self.strategy_agent.top_n]

            # compute current weights
            weights_now = {s: (positions.get(s, 0.0) * last_prices.get(s, 0.0)) / portfolio_equity if portfolio_equity > 0 else 0.0 for s in positions}
            for sym in candidates:
                price = last_prices.get(sym)
                if not price or cash <= 0:
                    continue
                cur_weight = weights_now.get(sym, 0.0)
                # Kelly-like fraction from score and variance proxy
                vol = float(window_features[sym].get("vol_21", pd.Series([0.02])).iloc[-1] or 0.02)
                var = max(1e-6, vol * vol)
                edge = max(0.0, float(signals[sym]["score"]))
                kelly = min(self.cfg.max_weight_per_symbol, 0.1 * edge / (var ** 0.5))
                add_weight = max(0.0, min(self.cfg.max_weight_per_symbol - cur_weight, kelly))
                if add_weight <= 0:
                    continue
                buy_dollars = portfolio_equity * add_weight
                buy_dollars = min(buy_dollars, cash)
                if buy_dollars <= 0:
                    continue
                qty = buy_dollars / price
                prev_qty = positions.get(sym, 0.0)
                new_qty = prev_qty + qty
                new_cost = (prev_qty * avg_cost.get(sym, price) + qty * price) / new_qty if new_qty > 0 else 0.0
                positions[sym] = new_qty
                avg_cost[sym] = new_cost
                peak_price[sym] = max(peak_price.get(sym, 0.0), float(price))
                cash -= buy_dollars
                trades.append({"date": str(today), "symbol": sym, "side": "BUY", "price": float(price), "quantity": float(qty)})

            # record end-of-day weights
            pe2 = cash + sum(positions.get(s, 0.0) * last_prices.get(s, 0.0) for s in positions)
            if pe2 > 0:
                w_today = {s: (positions.get(s, 0.0) * last_prices.get(s, 0.0)) / pe2 for s in positions}
            else:
                w_today = {s: 0.0 for s in positions}
            weights_history.append({"date": str(today), "weights": w_today})

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        test_start = self.cfg.test_start
        test_end = self.cfg.test_end
        if test_start or test_end:
            equity_df = equity_df.loc[test_start: test_end]
        index = list(equity_df.index)
        baseline = self._baseline_buy_and_hold(md_full, index)

        def normalize_to_zero(series: pd.Series) -> pd.Series:
            if series.empty:
                return series
            base = series.iloc[0]
            if base == 0:
                return series
            return (series / base - 1.0) * 100.0

        eq = equity_df["equity"].astype(float)
        eq_norm = normalize_to_zero(eq)
        base_norm = normalize_to_zero(baseline.astype(float))

        rets = eq.pct_change(fill_method=None).dropna()
        base_eq = baseline.astype(float)
        base_rets = base_eq.pct_change(fill_method=None).dropna()
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(1, len(rets))) - 1 if len(rets) > 0 else 0.0
        sharpe = np.sqrt(252) * (rets.mean() / (rets.std() + 1e-9)) if len(rets) > 2 else 0.0
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        mdd = float(drawdown.min()) if not drawdown.empty else 0.0

        # Build weights DataFrame for heatmap
        weights_df = pd.DataFrame()
        if weights_history:
            wh_index = [w["date"] for w in weights_history]
            all_syms = sorted({sym for w in weights_history for sym in w["weights"].keys()})
            W = pd.DataFrame(0.0, index=wh_index, columns=all_syms)
            for row in weights_history:
                d = row["date"]
                for s, v in row["weights"].items():
                    W.at[d, s] = float(v)
            weights_df = W.sort_index()
        base_cagr = (base_eq.iloc[-1] / base_eq.iloc[0]) ** (252 / max(1, len(base_rets))) - 1 if len(base_rets) > 0 else 0.0
        base_sharpe = np.sqrt(252) * (base_rets.mean() / (base_rets.std() + 1e-9)) if len(base_rets) > 2 else 0.0
        base_mdd = float(((base_eq - base_eq.cummax()) / base_eq.cummax()).min()) if not base_eq.empty else 0.0

        # compute turnover from weights_history
        turnover_series = pd.Series(dtype=float)
        if weights_history:
            # build DataFrame of weights
            wh_index = [w["date"] for w in weights_history]
            all_syms = sorted({sym for w in weights_history for sym in w["weights"].keys()})
            W = pd.DataFrame(0.0, index=wh_index, columns=all_syms)
            for row in weights_history:
                d = row["date"]
                for s, v in row["weights"].items():
                    W.at[d, s] = float(v)
            W = W.sort_index()
            turnover_series = (W.diff().abs().sum(axis=1) / 2.0).fillna(0.0)

        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(out_dir / "equity_curve.csv")
        pd.Series(baseline, name="baseline").to_csv(out_dir / "baseline_curve.csv")
        pd.DataFrame(trades).to_csv(out_dir / "trades_all.csv", index=False)
        pd.concat([eq_norm.rename("strategy_norm_%") , base_norm.rename("baseline_norm_%")], axis=1).to_csv(out_dir / "normalized_returns.csv")
        if not turnover_series.empty:
            turnover_series.rename("turnover").to_csv(out_dir / "turnover.csv")
        # Build directional accuracy and calibration inputs
        price_map: Dict[str, pd.Series] = {}
        for sym, df in md_full.items():
            s = df.set_index(df["date"].astype(str))["close"].astype(float)
            price_map[sym] = s
        acc_series = pd.Series(dtype=float)
        pred_all: List[float] = []
        real_all: List[float] = []
        for entry in predictions_history:
            d = entry["date"]
            preds: Dict[str, float] = entry["preds"]
            vals = []
            for sym, p in preds.items():
                ps = price_map.get(sym)
                if ps is None or d not in ps.index:
                    continue
                try:
                    idx_loc = ps.index.get_loc(d)
                except Exception:
                    continue
                if idx_loc is None or idx_loc == len(ps.index) - 1:
                    continue
                r = float(ps.iloc[idx_loc + 1] / ps.iloc[idx_loc] - 1.0)
                vals.append(1.0 if p * r > 0 else 0.0)
                pred_all.append(p)
                real_all.append(r)
            if vals:
                acc_series.at[d] = float(np.mean(vals))
        if not acc_series.empty:
            acc_series.rename("directional_accuracy").to_csv(out_dir / "directional_accuracy.csv")
        if pred_all and real_all:
            pd.DataFrame({"pred": pred_all, "real": real_all}).to_csv(out_dir / "pred_vs_real.csv", index=False)
        if consensus_diag:
            pd.DataFrame(consensus_diag).to_csv(out_dir / "consensus_diag.csv", index=False)
        # Additional robustness metrics
        turnover_mean = float(turnover_series.mean()) if not turnover_series.empty else float("nan")
        var95 = float(np.percentile(rets.dropna(), 5)) if not rets.empty else float("nan")
        cvar95 = float(rets[rets <= np.percentile(rets.dropna(), 5)].mean()) if len(rets.dropna()) > 0 else float("nan")
        regime_metrics: Dict[str, float] = {}
        if daily_meta:
            regime_df = pd.DataFrame(daily_meta).set_index("date")
            if not regime_df.empty and not rets.empty:
                r = rets.copy()
                r.index = [str(i) for i in r.index]
                both = r.to_frame("ret").join(regime_df, how="inner")
                if not both.empty:
                    thr = float(both["avg_vol"].median())
                    low = both[both["avg_vol"] <= thr]["ret"]
                    high = both[both["avg_vol"] > thr]["ret"]
                    if not low.empty:
                        regime_metrics["low_vol_cagr"] = float((1.0 + low).prod() ** (252 / max(1, len(low))) - 1)
                        regime_metrics["low_vol_sharpe"] = float(np.sqrt(252) * (low.mean() / (low.std() + 1e-9)))
                    if not high.empty:
                        regime_metrics["high_vol_cagr"] = float((1.0 + high).prod() ** (252 / max(1, len(high))) - 1)
                        regime_metrics["high_vol_sharpe"] = float(np.sqrt(252) * (high.mean() / (high.std() + 1e-9)))

        summary = {
            "test_start": test_start,
            "test_end": test_end,
            "final_equity": float(eq.iloc[-1]) if not eq.empty else self.cfg.capital,
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
            "baseline_final": float(base_eq.iloc[-1]) if not base_eq.empty else float("nan"),
            "baseline_cagr": float(base_cagr),
            "baseline_sharpe": float(base_sharpe),
            "baseline_max_drawdown": float(base_mdd),
            "num_trades": int(len(trades)),
            "turnover_mean": turnover_mean,
            "var_95": var95,
            "cvar_95": cvar95,
            "seed": int(seed_val),
            "robust_noise_std": float(noise_std),
            "robust_mu_scale": float(mu_scale),
            "robust_drop_pct": float(drop_pct),
        }
        pd.Series(summary).to_json(out_dir / "summary.json")
        if regime_metrics:
            pd.Series(regime_metrics).to_csv(out_dir / "regime_metrics.csv")

        if make_plots:
            try:
                from .plotting import (
                    save_plots_normalized,
                    save_rolling_and_turnover,
                    save_distribution_and_beta,
                    save_weights_heatmap,
                    save_rolling_accuracy,
                    save_calibration_curve,
                    save_regime_bars,
                )
                save_plots_normalized(eq_norm, base_norm, out_dir)
                save_rolling_and_turnover(rets, out_dir, window=63, turnover=turnover_series)
                save_distribution_and_beta(rets, base_rets, out_dir, window=63)
                if not weights_df.empty:
                    save_weights_heatmap(weights_df, out_dir, top_k=25)
                if not acc_series.empty:
                    save_rolling_accuracy(acc_series, out_dir, window=63)
                if pred_all and real_all:
                    import pandas as _pd
                    save_calibration_curve(_pd.Series(pred_all), _pd.Series(real_all), out_dir, n_bins=10)
                # regime bars if available
                try:
                    import json as _json
                    s = _json.loads((out_dir / "summary.json").read_text())
                    # attempt to read regime csv too
                    reg = {}
                    rp = out_dir / "regime_metrics.csv"
                    if rp.exists():
                        import pandas as _pd2
                        sr = _pd2.read_csv(rp, index_col=0).squeeze()
                        reg = {k: float(v) for k, v in sr.to_dict().items()}
                    save_regime_bars(reg, out_dir)
                except Exception:
                    pass
            except Exception:
                pass
        return summary
