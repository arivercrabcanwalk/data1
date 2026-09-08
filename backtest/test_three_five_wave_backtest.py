import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

MODULE_PATH = os.path.join(os.path.dirname(__file__), "three_five_wave_backtest.py")
spec = importlib.util.spec_from_file_location("strategy", MODULE_PATH)
strategy = importlib.util.module_from_spec(spec)
sys.modules["strategy"] = strategy
spec.loader.exec_module(strategy)


class StrategyV2Tests(unittest.TestCase):
    def test_big_bar_uses_previous_close_not_candle_body(self):
        prev = pd.Series({"close": 10.0, "high": 10.1})
        row = pd.Series({"open": 10.5, "high": 11.1, "close": 11.0, "ret": 0.10, "close_pos": 0.85, "amount": 200.0, "amt_ma10": 100.0})
        anchor = strategy.compute_bigbar_anchor(row, prev)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["start"], 10.0)

    def test_non_gap_big_bar_uses_open_as_anchor(self):
        prev = pd.Series({"close": 10.0, "high": 10.2})
        row = pd.Series({"open": 10.05, "high": 11.0, "close": 10.8, "ret": 0.08, "close_pos": 0.82, "amount": 150.0, "amt_ma10": 100.0})
        anchor = strategy.compute_bigbar_anchor(row, prev)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["start"], 10.05)

    def test_wave_divergence_arms_before_reversal_day(self):
        n = 37
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        close = np.full(n, 100.0)
        high = np.full(n, 100.5)
        low = np.full(n, 99.5)
        ret = np.zeros(n)
        open_ = close.copy()
        amount = np.full(n, 100.0)
        amt_ma10 = np.full(n, 100.0)
        close_pos = np.full(n, 0.60)
        dif = np.full(n, -0.5)
        hist = np.full(n, -0.2)
        neg_area = np.full(n, 1.0)

        close[20] = 90.0
        open_[20] = 91.0
        high[20] = 91.0
        low[20] = 89.5
        dif[20] = -2.0
        hist[20] = -1.0
        neg_area[20] = 5.0
        for i in range(21, 30):
            close[i] = 97.0
            open_[i] = 96.5
            high[i] = 98.0
            low[i] = 96.0
        close[35] = 88.0
        open_[35] = 89.0
        high[35] = 89.2
        low[35] = 87.5
        dif[35] = -1.0
        hist[35] = -0.4
        neg_area[35] = 3.0
        ret[35] = -0.02
        close_pos[35] = 0.30

        close[36] = 91.6
        open_[36] = 88.5
        high[36] = 92.0
        low[36] = 88.2
        ret[36] = close[36] / close[35] - 1
        close_pos[36] = 0.90
        amount[36] = 120.0
        dif[36] = -0.8
        hist[36] = -0.1

        g = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "amount": amount,
            "amt_ma10": amt_ma10,
            "ret": ret,
            "prev_close": np.r_[np.nan, close[:-1]],
            "close_pos": close_pos,
            "dif": dif,
            "hist": hist,
            "neg_hist_area5": neg_area,
        })
        div = strategy.wave_divergence_candidate(g, 35)
        self.assertIsNotNone(div)
        self.assertEqual(div["kind"], "wave3")
        signals, diag = strategy.detect_signals_one(g)
        self.assertGreaterEqual(diag["wave3_armed"], 1)
        self.assertGreaterEqual(diag["wave3_confirmed"], 1)
        self.assertIn("wave3_div_reversal", set(signals["signal_type"]))
        sig = signals.loc[signals["signal_type"] == "wave3_div_reversal"].iloc[0]
        self.assertEqual(pd.Timestamp(sig["date"]), pd.Timestamp(dates[36]))

    def test_early_big_bar_can_create_pullback_before_macd_ready(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        g = pd.DataFrame({
            "date": dates,
            "open": [10, 10, 10.7, 10.5, 10.3, 10.15, 10.2, 10.3, 10.4, 10.5],
            "high": [10.1, 10.9, 10.9, 10.7, 10.5, 10.3, 10.4, 10.5, 10.6, 10.7],
            "low": [9.9, 9.95, 10.5, 10.3, 10.1, 10.0, 10.0, 10.1, 10.2, 10.3],
            "close": [10, 10.8, 10.7, 10.5, 10.3, 10.12, 10.25, 10.35, 10.45, 10.55],
            "amount": [100, 200, 100, 90, 80, 80, 90, 100, 100, 100],
            "amt_ma10": [100] * 10,
            "ret": [np.nan, 0.08, -0.0093, -0.0187, -0.0190, -0.0175, 0.0128, 0.0098, 0.0097, 0.0096],
            "prev_close": [np.nan, 10, 10.8, 10.7, 10.5, 10.3, 10.12, 10.25, 10.35, 10.45],
            "close_pos": [0.5, 0.85, 0.5, 0.5, 0.5, 0.55, 0.62, 0.62, 0.62, 0.62],
            "dif": [np.nan] * 10,
            "hist": [np.nan] * 10,
            "neg_hist_area5": [np.nan] * 10,
        })
        signals, diag = strategy.detect_signals_one(g)
        self.assertGreaterEqual(diag["anchors_created"], 1)
        self.assertGreaterEqual(diag["right_pullback_setups"], 1)
        self.assertIn("right_pullback", set(signals["signal_type"]))

    def test_intraday_confirmation_waits_for_stability(self):
        ts = pd.date_range("2026-07-01 09:30", periods=40, freq="min")
        prices = np.linspace(10.0, 10.12, 40)
        prices[:5] = [10.0, 9.96, 9.92, 9.90, 9.91]
        prices[5:] = np.linspace(9.92, 10.12, 35)
        x = pd.DataFrame({
            "code": ["000001"] * 40,
            "datetime": ts,
            "open": prices - 0.002,
            "high": prices + 0.01,
            "low": prices - 0.01,
            "close": prices,
            "volume": [1000] * 40,
            "amount": [10000] * 40,
        })
        signal = pd.Series({"stop_anchor": 9.85})
        status = pd.Series({"is_st": False, "is_star_st": False, "is_new_listing_initial": False, "is_suspended": False, "price_limit_up_pct": 10.0})
        confirm, reason = strategy.find_intraday_entry(x, signal, 10.0, status)
        self.assertEqual(reason, "confirmed")
        self.assertIsNotNone(confirm)
        self.assertGreaterEqual(pd.Timestamp(confirm["entry_time"]).time(), strategy.parse_hhmm("09:45"))
        self.assertGreaterEqual(confirm["bars_since_low"], strategy.CFG["intraday"]["stable_bars_after_low"])


if __name__ == "__main__":
    unittest.main()
