"""Unit tests for acceptance.py on synthetic data with known answers.

Run:  cd router0923/exp/T2 && /usr/bin/python3 -m unittest -v test_acceptance
Monte-Carlo size/power tests use fixed seeds; tolerance bands are wide enough for
their MC error (300 reps -> s.e. of a 5% rate ~ 1.3 points).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acceptance as A  # noqa: E402

FAST = os.environ.get("T2_FAST", "0") == "1"
MC = 150 if FAST else 300


def ar1(rng, n, phi, k=1, mu=0.0, sd=1.0):
    e = rng.standard_normal((n + 50, k)) * sd * np.sqrt(1 - phi ** 2)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[50:] + mu


def t1_boot_p(x, reps, block, seed):
    """Verbatim logic of exp/T1/code/t1_router_pilot.boot_p (reference)."""
    T = len(x)
    rng = np.random.default_rng(seed)
    obs = x.mean()
    xc = x - obs
    cnt = 0
    for _ in range(reps):
        idx, i = [], rng.integers(T)
        for t in range(T):
            if t and rng.random() < 1 / block:
                i = rng.integers(T)
            idx.append(i)
            i = (i + 1) % T
        cnt += xc[idx].mean() >= obs
    return (cnt + 1) / (reps + 1)


class TestHelpers(unittest.TestCase):
    def test_holm_known(self):
        adj = A.holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
        d = A.holm({"a": 0.2, "b": 0.01})
        self.assertAlmostEqual(d["b"], 0.02)
        self.assertAlmostEqual(d["a"], 0.2)

    def test_stationary_bootstrap_block_length_and_uniformity(self):
        rng = np.random.default_rng(1)
        n, B, L = 200, 2000, 4.0
        ix = A.stationary_bootstrap_indices(n, B, L, rng)
        self.assertEqual(ix.shape, (B, n))
        cont = (ix[:, 1:] == (ix[:, :-1] + 1) % n)
        # P(continue) = 1 - 1/L, plus the 1/n chance that a fresh start is the next index
        self.assertAlmostEqual(cont.mean(), 1 - 1 / L + (1 / L) / n, delta=0.01)
        counts = np.bincount(ix.ravel(), minlength=n) / ix.size
        self.assertLess(np.abs(counts * n - 1).max(), 0.1)

    def test_pool_by_date_panel(self):
        df = pd.DataFrame({"u1": [1.0, 2.0, np.nan], "u2": [3.0, np.nan, np.nan]},
                          index=pd.date_range("2014-03-31", periods=3, freq="QE"))
        np.testing.assert_allclose(A.pool_by_date(df), [2.0, 2.0])
        d = A.paired_diff(df, df * 0)
        np.testing.assert_allclose(d, [2.0, 2.0])

    def test_matches_t1_boot_p(self):
        rng = np.random.default_rng(7)
        x = ar1(rng, 48, 0.3).ravel() * 0.03 + 0.004
        mine = A.paired_bootstrap(x, n_boot=3000, mean_block=4, seed=0).p_value
        ref = t1_boot_p(x, 3000, 4, 0)
        self.assertLess(abs(mine - ref), 0.02, (mine, ref))


class TestPairedBootstrap(unittest.TestCase):
    def _rate(self, mu, stat="mean", n=60, phi=0.2, studentize=False):
        rng = np.random.default_rng(11 if mu == 0 else 12)
        rej = 0
        for r in range(MC):
            if stat == "mean":
                d = ar1(rng, n, phi).ravel() + mu
                p = A.paired_bootstrap(d, n_boot=499, mean_block=4, seed=r,
                                       studentize=studentize).p_value
            else:
                z = rng.standard_normal((n, 2))
                a = 0.01 * (z[:, 0]) + mu
                b = 0.01 * (0.6 * z[:, 0] + 0.8 * z[:, 1])
                p = A.paired_bootstrap(a, b, stat="sharpe", n_boot=499, mean_block=5,
                                       seed=r).p_value
            rej += p < 0.05
        return rej / MC

    def test_mean_size(self):
        # iid: close to nominal.  AR(1) phi=0.2, n=60: the PREREG (unstudentized) test is
        # mildly liberal (~0.08-0.10 in calibrate.py); the studentized one is nearer 0.05.
        self.assertTrue(0.015 <= self._rate(0.0, phi=0.0) <= 0.10)
        self.assertTrue(0.015 <= self._rate(0.0) <= 0.13)
        self.assertTrue(0.015 <= self._rate(0.0, studentize=True) <= 0.10)

    def test_mean_power(self):
        self.assertGreater(self._rate(0.45), 0.80)
        self.assertGreater(self._rate(0.45, studentize=True), 0.75)

    def test_sharpe_size_and_power(self):
        self.assertTrue(0.015 <= self._rate(0.0, "sharpe", n=500) <= 0.10)
        self.assertGreater(self._rate(0.0015, "sharpe", n=500), 0.80)

    def test_b1_helper_holm(self):
        rng = np.random.default_rng(3)
        n = 48
        base = rng.standard_normal(n)
        router = base + 1.0
        df = A.b1_test(router, {"EW": base, "fixed": base + 0.9, "Hedge": base + 1.2},
                       n_boot=999)
        self.assertLess(df.loc["EW", "p_holm"], 0.01)
        self.assertGreater(df.loc["Hedge", "p"], 0.5)
        self.assertTrue((df["p_holm"] >= df["p"]).all())

    def test_b2(self):
        r = A.b2_universe_wins({"a": 1, "b": 1, "c": 1, "d": 1, "e": 0, "f": 0},
                               {k: 0.5 for k in "abcdef"})
        self.assertEqual(r["wins"], 4)
        self.assertTrue(r["passed"])


class TestLedoitWolf(unittest.TestCase):
    def test_hac_se_matches_iid_formula(self):
        rng = np.random.default_rng(5)
        n, rho = 20000, 0.5
        z = rng.standard_normal((n, 2))
        x1 = 0.10 + z[:, 0]
        x2 = 0.05 + rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]
        out = A.ledoit_wolf_sharpe(x1, x2, method="hac")
        s1, s2 = 0.10, 0.05
        v = (2 * (1 - rho) + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / n
        self.assertAlmostEqual(out["se"] / np.sqrt(v), 1.0, delta=0.1)

    def _rate(self, dsr, method, reps, n=500):
        rng = np.random.default_rng(21 if dsr == 0 else 22)
        rej = 0
        for r in range(reps):
            x = ar1(rng, n, 0.1, k=2)
            x[:, 1] = 0.7 * x[:, 0] + np.sqrt(1 - 0.49) * x[:, 1]
            x[:, 0] += 0.05 + dsr
            x[:, 1] += 0.05
            out = A.ledoit_wolf_sharpe(x[:, 0], x[:, 1], method=method, n_boot=199,
                                       block=5, seed=r)
            rej += out["p_value"] < 0.05
        return rej / reps

    def test_hac_size_power(self):
        self.assertTrue(0.02 <= self._rate(0.0, "hac", MC) <= 0.10)
        self.assertGreater(self._rate(0.12, "hac", MC), 0.80)

    def test_boot_size_power(self):
        reps = 100 if FAST else 200
        self.assertTrue(0.01 <= self._rate(0.0, "boot", reps) <= 0.10)
        self.assertGreater(self._rate(0.12, "boot", reps), 0.75)


class TestDegenerate(unittest.TestCase):
    def test_lw_identical_series(self):
        r = np.random.default_rng(0).standard_normal(500)
        out = A.ledoit_wolf_sharpe(r, r.copy(), method="boot", n_boot=199)
        self.assertEqual(out["p_value"], 1.0)
        self.assertEqual(out["delta"], 0.0)

    def test_mcs_duplicate_columns(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal((60, 3)) + [0.0, 0.0, -1.5]
        U = pd.DataFrame(np.c_[base, base[:, 0]], columns=["O1", "EW", "bad", "Deployed"])
        r = A.model_confidence_set(U, higher_is_better=True, n_boot=999)
        self.assertEqual(r["duplicates"], {"Deployed": "O1"})
        self.assertEqual(r["pvalues"]["Deployed"], r["pvalues"]["O1"])
        self.assertIn("Deployed", r["included"])
        self.assertNotIn("bad", r["included"])


class TestSPA(unittest.TestCase):
    def _sim(self, mus, reps, n=80, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for r in range(reps):
            bench = rng.standard_normal(n)
            alts = bench[:, None] * 0.5 + rng.standard_normal((n, len(mus))) + np.array(mus)
            res = A.spa_test(bench, alts, n_boot=499, mean_block=2, seed=r)
            out.append(res)
        return out

    def test_ordering_and_size_least_favourable(self):
        res = self._sim([0.0] * 5, MC, seed=31)
        for r in res:
            self.assertLessEqual(r["p_lower"], r["p_consistent"] + 1e-12)
            self.assertLessEqual(r["p_consistent"], r["p_upper"] + 1e-12)
        size = np.mean([r["p_consistent"] < 0.05 for r in res])
        self.assertTrue(0.015 <= size <= 0.10, size)

    def test_consistent_less_conservative_with_poor_alternatives(self):
        # one null alternative plus four clearly worse ones: RC (upper) is conservative.
        res = self._sim([0.0, -1.0, -1.0, -1.0, -1.0], MC, seed=32)
        size_c = np.mean([r["p_consistent"] < 0.05 for r in res])
        size_u = np.mean([r["p_upper"] < 0.05 for r in res])
        self.assertTrue(size_c <= 0.10)
        self.assertGreaterEqual(size_c, size_u)

    def test_power(self):
        res = self._sim([0.0, 0.0, 0.0, 0.0, 0.5], MC, seed=33)
        power = np.mean([r["p_consistent"] < 0.05 for r in res])
        self.assertGreater(power, 0.80)
        self.assertEqual(pd.Series([r["best"] for r in res]).mode()[0], "m4")


class TestMCS(unittest.TestCase):
    def _sim(self, shifts, reps, method, n=80, seed=0):
        rng = np.random.default_rng(seed)
        res = []
        for r in range(reps):
            common = rng.standard_normal((n, 1))
            L = common + rng.standard_normal((n, len(shifts))) + np.array(shifts)
            res.append(A.model_confidence_set(L, alpha=0.10, method=method,
                                              n_boot=499, mean_block=2, seed=r))
        return res

    def test_equal_models_retained(self):
        for method in ("R", "max"):
            res = self._sim([0, 0, 0, 0], MC, method, seed=41)
            full = np.mean([len(r["included"]) == 4 for r in res])
            self.assertGreater(full, 0.82, (method, full))

    def test_bad_model_eliminated_best_kept(self):
        for method in ("R", "max"):
            res = self._sim([0, 0, 0, 0.8], MC, method, seed=42)
            elim = np.mean(["m3" not in r["included"] for r in res])
            self.assertGreater(elim, 0.85, (method, elim))
            kept = np.mean([all(m in r["included"] for m in ("m0", "m1", "m2")) for r in res])
            self.assertGreater(kept, 0.75, (method, kept))

    def test_utilities_flag_and_pvalue_monotone(self):
        rng = np.random.default_rng(4)
        U = pd.DataFrame(rng.standard_normal((60, 3)) + [0.0, 0.0, -1.0],
                         columns=["O1", "EW", "bad"])
        r = A.model_confidence_set(U, higher_is_better=True, n_boot=999)
        self.assertNotIn("bad", r["included"])
        ps = [r["pvalues"][m] for m in r["eliminated_order"]]
        self.assertEqual(ps, sorted(ps))


class TestDSR(unittest.TestCase):
    def test_paper_numerical_example(self):
        # Bailey & Lopez de Prado (2014), numerical example: annual SR 2.5, 5y daily
        # (T=1250), N=100 trials, annual V[SR]=0.5, skew -3, kurt 10 -> DSR ~ 0.9004.
        out = A.deflated_sharpe(sr=2.5 / np.sqrt(250), n=1250, skew=-3.0, kurt=10.0,
                                n_trials=100, var_trials=0.5 / 250)
        self.assertAlmostEqual(out["sr0"], 0.1132, places=3)
        self.assertAlmostEqual(out["dsr"], 0.9004, places=3)

    def test_psr_and_monotonicity(self):
        self.assertAlmostEqual(A.probabilistic_sharpe(0.0, 100), 0.5)
        d = [A.deflated_sharpe(sr=0.1, n=500, n_trials=k, var_trials=0.002)["dsr"]
             for k in (1, 2, 10, 100, 1000)]
        self.assertEqual(d, sorted(d, reverse=True))
        self.assertAlmostEqual(d[0], A.probabilistic_sharpe(0.1, 500))

    def test_from_returns_calibration(self):
        # Under H0 (true SR 0, one trial) PSR(0) is ~Uniform -> P(PSR>0.95) ~ 5%.
        rng = np.random.default_rng(9)
        hits = [A.deflated_sharpe(rng.standard_normal(250), n_trials=1,
                                  var_trials=0.0)["dsr"] > 0.95 for _ in range(1000)]
        self.assertTrue(0.03 <= np.mean(hits) <= 0.07)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPBO(unittest.TestCase):
    def test_pure_noise_near_half(self):
        # E[PBO] = 0.5 under pure noise; a single sample is very dispersed (10-90%: ~0.25-0.78),
        # so average over 60 samples (s.e. ~0.02).
        vals = [A.pbo_cscv(np.random.default_rng(100 + s).standard_normal((480, 50)), n_splits=10)["pbo"]
                for s in range(60)]
        self.assertTrue(0.43 <= np.mean(vals) <= 0.57, np.mean(vals))

    def test_genuine_edge_low_pbo(self):
        X = np.random.default_rng(1).standard_normal((480, 50))
        X[:, 7] += 0.3                                   # one config with a real edge
        r = A.pbo_cscv(X, n_splits=10)
        self.assertLess(r["pbo"], 0.05)
        self.assertEqual(r["n_combinations"], 252)      # C(10, 5)

    def test_overfit_selection_high_pbo(self):
        # configs that are good in one half of the sample are bad in the other (regime flip)
        rng = np.random.default_rng(2)
        X = rng.standard_normal((400, 40)) * 0.5
        sign = np.where(np.arange(400)[:, None] < 200, 1.0, -1.0)
        X += sign * np.linspace(-0.3, 0.3, 40)[None, :]
        r = A.pbo_cscv(X, n_splits=8)
        self.assertGreater(r["pbo"], 0.5)
        self.assertLess(r["degradation_slope"], 0)

    def test_subsample_and_errors(self):
        X = np.random.default_rng(3).standard_normal((160, 5))
        r = A.pbo_cscv(X, n_splits=16, max_combinations=500)
        self.assertEqual(r["n_combinations"], 500)
        with self.assertRaises(ValueError):
            A.pbo_cscv(X, n_splits=5)


class TestPanelFromLong(unittest.TestCase):
    def test_roundtrip(self):
        long = pd.DataFrame({"cell": ["a", "b", "a", "b"], "date": ["2008-03-31"] * 2 + ["2008-06-30"] * 2,
                             "O1": [1.0, 3.0, 2.0, 4.0], "W2a_0": [0.0, 1.0, 1.0, 1.0]})
        P = A.panel_from_long(long)
        self.assertEqual(set(P), {"O1", "W2a_0"})
        np.testing.assert_allclose(A.pool_by_date(P["O1"]), [2.0, 3.0])
        self.assertEqual(list(P["O1"].columns), ["a", "b"])


class TestFinalReport(unittest.TestCase):
    def test_runs_and_detects_planted_edge(self):
        rng = np.random.default_rng(8)
        dates = pd.date_range("2014-03-31", periods=48, freq="QE")
        cells = [f"c{i}" for i in range(6)]
        base = pd.DataFrame(rng.normal(0.02, 0.03, (48, 6)), index=dates, columns=cells)
        U = {"Cstar": base, "EW": base + rng.normal(0, 0.005, (48, 6)),
             "Hedge": base + rng.normal(-0.002, 0.01, (48, 6)),
             "router": base + 0.012 + rng.normal(0, 0.01, (48, 6))}
        grid = pd.DataFrame({f"cfg{k}": A.pool_by_date(base + rng.normal(0, 0.01, (48, 6)))
                             for k in range(10)})
        grid["router"] = A.pool_by_date(U["router"])
        daily = {"router": pd.Series(rng.normal(0.0008, 0.01, 3000)),
                 "Cstar": pd.Series(rng.normal(0.0004, 0.01, 3000))}
        out = A.final_report(U, other_primary_p=[0.5, 0.8, 0.9],
                             sr_by_universe={"router": dict(zip("abcdef", [1, 1, 1, 1, 1, 0])),
                                             "Cstar": dict(zip("abcdef", [.5] * 6))},
                             daily_returns=daily, spa_alternatives=grid, trials=grid,
                             trial_sharpes=rng.normal(0.02, 0.01, 50), n_boot=999)
        t = out["table"].set_index("test")
        self.assertTrue(t.loc["B1 (PREREG)", "verdict"])
        self.assertTrue(t.loc["B2 (PREREG)", "verdict"])
        self.assertTrue(t.loc["B3 (PREREG)", "verdict"])
        self.assertIn("\\begin{tabular}", out["latex"])
        self.assertIn("\\toprule", out["latex"])
        self.assertEqual(out["latex"].count("\\\\"), len(t) + 1)   # one row end per row + header
        self.assertEqual(out["raw"]["ledger"]["N_test_seen"], A.ledger_n(A.LEDGER_FREEZE)["N_test_seen"])
        self.assertEqual(out["raw"]["ledger"]["N_test_seen"], 194)

    def test_null_router_fails(self):
        rng = np.random.default_rng(9)
        base = pd.Series(rng.normal(0.02, 0.03, 48))
        U = {"Cstar": base, "Hedge": base + rng.normal(0, 0.01, 48),
             "router": base + rng.normal(0, 0.01, 48)}
        out = A.final_report(U, n_boot=499, n_trials=188)
        t = out["table"].set_index("test")
        self.assertFalse(t.loc["B1 (PREREG)", "verdict"])
