"""Walk-forward harness that scores a TTA adapter on BOTH objectives at once.

For every rebalance we record
  - the forecast error the adapter's calibrated output achieves, and
  - the realised utility of the decision that output induced.

Recording both on the identical decision path is the whole point: it lets us
ask whether an MSE improvement buys a decision improvement.

Causality contract
------------------
  features at t    use prices <= t
  frozen model     all forward training targets end within the train window
  Sigma at t       estimated from returns <= t
  adapter update   consumes only labels matured at or before t
  return ending t  earned by the portfolio held before close t
  turnover at t    measured against the drifted holdings at close t
"""
import numpy as np
import pandas as pd

ANN = 252.0
METRICS_VERSION = "initial_wealth_downside_v2"
LEGACY_METRICS_VERSION = "legacy_v1"


# ------------------------------------------------------------------ features
def make_features(px, sym):
    """Causal per-asset feature frame. Every column at row t uses prices <= t."""
    p = px[sym].astype(float)
    r = p.pct_change()
    f = pd.DataFrame(index=p.index)
    f["mom5"] = p.pct_change(5)
    f["mom21"] = p.pct_change(21)
    f["mom63"] = p.pct_change(63)
    f["vol21"] = r.rolling(21).std()
    f["vol63"] = r.rolling(63).std()
    f["ma_ratio"] = p / p.rolling(50).mean() - 1.0
    f["rsi_prox"] = r.rolling(14).apply(
        lambda x: (x > 0).mean() if len(x) else np.nan, raw=True)
    f["dd"] = p / p.rolling(126).max() - 1.0
    return f


def build_panel_features(px, syms):
    return {s: make_features(px, s) for s in syms}


# ---------------------------------------------------------------- forecaster
class FrozenRidge:
    """Pooled cross-sectional ridge on standardised features. Fit once, frozen."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.coef_ = None
        self.mu_ = None
        self.sd_ = None
        self.feature_second_moment_ = None
        self.target_std_ = None

    def fit(self, feats, px, syms, start, end, horizon):
        X, y = [], []
        for s in syms:
            f = feats[s].loc[start:end]
            fwd = (px[s].shift(-horizon) / px[s] - 1.0).loc[start:end]
            d = pd.concat([f, fwd.rename("y")], axis=1).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(d):
                X.append(d.drop(columns="y").values)
                y.append(d["y"].values)
        if not X:
            return self
        X = np.vstack(X)
        y = np.concatenate(y)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu_) / self.sd_
        # Training-only scales for feature-space trust regions. These are
        # diagnostics/FX inputs; the established ridge solve below is unchanged.
        self.feature_second_moment_ = Xs.T @ Xs / len(Xs)
        self.target_std_ = float(np.std(y))
        n_f = Xs.shape[1]
        self.coef_ = np.linalg.solve(
            Xs.T @ Xs + self.alpha * np.eye(n_f), Xs.T @ y)
        return self

    def predict_row(self, x):
        if self.coef_ is None:
            return 0.0
        xs = (x - self.mu_) / self.sd_
        return float(xs @ self.coef_)


class FrozenGBM:
    """Stronger frozen forecaster: gradient-boosted trees on the same features.

    Exists to test whether the objective contrast survives when less
    miscalibration is left for a calibration layer to repair. Same fit-once,
    freeze-forever contract as FrozenRidge; identical feature set and window.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=15,
                 min_child_samples=40, seed=0):
        self.kw = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                       num_leaves=num_leaves, min_child_samples=min_child_samples,
                       random_state=seed, verbose=-1, n_jobs=1)
        self.model = None

    def fit(self, feats, px, syms, start, end, horizon):
        X, y = [], []
        for s in syms:
            f = feats[s].loc[start:end]
            fwd = (px[s].shift(-horizon) / px[s] - 1.0).loc[start:end]
            d = pd.concat([f, fwd.rename("y")], axis=1).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(d):
                X.append(d.drop(columns="y").values)
                y.append(d["y"].values)
        if not X:
            return self
        X, y = np.vstack(X), np.concatenate(y)
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(**self.kw).fit(X, y)
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(
                max_iter=self.kw["n_estimators"],
                learning_rate=self.kw["learning_rate"],
                max_leaf_nodes=self.kw["num_leaves"],
                random_state=self.kw["random_state"]).fit(X, y)
        return self

    def predict_row(self, x):
        if self.model is None:
            return 0.0
        return float(self.model.predict(np.asarray(x).reshape(1, -1))[0])


# ------------------------------------------------------------------- runner
def run(px, syms, cfg):
    """Run one portfolio using the shared chronological engine.

    New calls purge training targets, start in cash, earn returns on prior
    holdings, and charge turnover against drifted holdings. ``run_legacy`` is
    the explicit historical reproduction entrypoint.
    """
    # Local import avoids a circular import: the shared engine imports this
    # module's unchanged feature builders and frozen forecaster classes.
    import causal_runner_0908 as runner

    cfg = runner._normalise_cfg(cfg)
    out = runner.run_paths(px, cfg.get("universe_name", "experiment"), syms, cfg)
    if out is None:
        return None
    gross, turn, idx, aux = out
    s = pd.Series(gross - turn * cfg.get("cost_bps", 0.0) / 1e4, index=idx)
    result = {**_metrics(s, cfg.get("metrics_version", METRICS_VERSION),
                        cfg.get("sortino_mar_daily", 0.0)), **aux}
    if cfg.get("save_returns"):
        result["returns"] = s
    return result


def run_legacy(px, syms, cfg):
    """Reproduce the old experiment chronology and metrics, excluding the oracle.

    Results from this explicit legacy entrypoint are not causal evidence.
    Unlike the historical costed runner, this old path did not renormalise
    levered risky holdings. That distinction is preserved for reproduction.
    """
    import causal_runner_0908 as runner
    return run(px, syms, runner.legacy_config(cfg, source="experiment"))


def _metrics(r, metrics_version=METRICS_VERSION, sortino_mar_daily=0.0):
    """Annualized metrics from daily simple returns, with an explicit version.

    v2 starts drawdown at wealth 1 (so entry costs count) and computes downside
    deviation over ALL observations: sqrt(mean(min(R_t - MAR, 0)^2)). MAR is a
    daily minimum acceptable return, zero by default. SR and CAGR retain their
    established formulas. ``legacy_v1`` reproduces the old MDD and Sortino.
    """
    if metrics_version not in (METRICS_VERSION, LEGACY_METRICS_VERSION):
        raise ValueError(f"unknown metrics_version {metrics_version!r}")
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    yrs = len(r) / ANN
    if metrics_version == LEGACY_METRICS_VERSION:
        if sortino_mar_daily != 0.0:
            raise ValueError("legacy_v1 metrics support only MAR = 0")
        dn = r[r < 0].std(ddof=1)
        sortino = float(mu / dn * np.sqrt(ANN)) if dn and dn > 0 else np.nan
        mdd = float((eq / eq.cummax() - 1).min()) * 100
    else:
        excess = r - sortino_mar_daily
        downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
        sortino = (float(excess.mean() / downside * np.sqrt(ANN))
                   if downside > 0 else np.nan)
        peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
        mdd = float(np.min(np.minimum(eq.to_numpy() / peak - 1.0, 0.0))) * 100
    return {
        "SR": float(mu / sd * np.sqrt(ANN)) if sd > 0 else np.nan,
        "CAGR": float(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 and eq.iloc[-1] > 0 else np.nan,
        "MDD": mdd,
        "Sortino": sortino,
        "vol": float(sd * np.sqrt(ANN)) * 100,
        "n_days": len(r),
        "metrics_version": metrics_version,
        "sortino_mar_daily": float(sortino_mar_daily),
    }


def _metrics_legacy(r):
    """Historical metric convention for explicitly labelled reproduction."""
    return _metrics(r, LEGACY_METRICS_VERSION)
