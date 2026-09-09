from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from v4_shared import (
    V4CFG,
    add_market_modes,
    annotate_regular_signals,
    causal_find_left_probe_entry,
    _fundamental_left_probe_pass,
)
from v4_engine import v4_risk_sized_shares


class V4NoteIntegrationTests(unittest.TestCase):
    def test_market_mode_prefix_is_invariant_to_future_change(self):
        dates = pd.date_range("2026-03-02", periods=12, freq="B")
        base = pd.DataFrame({
            "date": dates,
            "benchmark_ret": [0.001, -0.002, 0.003, 0.001, -0.001, 0.002, 0.001, -0.002, 0.001, 0.0, 0.001, -0.001],
            "market_score": [50.0] * 12,
            "pct_above_ma5": [0.50] * 12,
        })
        a = add_market_modes(base)
        changed = base.copy()
        changed.loc[10:, "benchmark_ret"] = 0.20
        changed.loc[10:, "market_score"] = 100.0
        b = add_market_modes(changed)
        self.assertListEqual(a.loc[:9, "market_mode"].tolist(), b.loc[:9, "market_mode"].tolist())
        np.testing.assert_allclose(a.loc[:9, "benchmark_ret10"].fillna(0), b.loc[:9, "benchmark_ret10"].fillna(0))

    def test_risk_multiplier_shrinks_position(self):
        full, full_cost, dist1 = v4_risk_sized_shares(1_000_000, 1_000_000, 800_000, 100.0, 95.0, 1.0, 1.0)
        half, half_cost, dist2 = v4_risk_sized_shares(1_000_000, 1_000_000, 800_000, 100.0, 95.0, 0.5, 0.5)
        self.assertEqual(dist1, dist2)
        self.assertGreater(full, 0)
        self.assertLess(half, full)
        self.assertLess(half_cost, full_cost)

    def test_left_probe_waits_then_executes_next_bar_open(self):
        ts = pd.date_range("2026-08-03 09:30", periods=18, freq="min")
        rows = []
        for i, t in enumerate(ts):
            if i < 5:
                close = 99.5 - 0.6 * i
                low = close - 0.2
            elif i == 5:
                close = 96.5
                low = 96.4
            else:
                close = 96.5 + 0.12 * (i - 5)
                low = close - 0.05
            rows.append({
                "datetime": t,
                "open": close - 0.03,
                "high": close + 0.08,
                "low": low,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            })
        x = pd.DataFrame(rows)
        status = pd.Series({"is_st": False, "is_star_st": False, "is_new_listing_initial": False, "is_suspended": False})
        confirm, reason = causal_find_left_probe_entry(x, 100.0, status)
        self.assertIsNotNone(confirm, reason)
        self.assertGreater(confirm["execution_time"], confirm["confirmation_time"])
        idx = x.index[x["datetime"] == confirm["execution_time"]][0]
        self.assertAlmostEqual(confirm["raw_price"], float(x.loc[idx, "open"]), places=10)

    def test_range_support_gate_requires_adjustment_and_core(self):
        date = pd.Timestamp("2026-07-10")
        signal = pd.DataFrame([{
            "code": "000001", "date": date, "signal_type": "pingbu_qingyun", "profile": "strict",
            "stop_anchor": 10.0, "candidate_score": 80.0, "absolute_gate_pass": True,
        }])
        daily = pd.DataFrame([{
            "code": "000001", "date": date, "ret": -0.01, "rs5_rank": 0.8, "rs20_rank": 0.8,
            "liquidity_rank": 0.8, "atr10_pct": 0.02, "height60": 0.20, "down_days5": 3.0,
            "days_since_high20": 5.0, "technical_core_proxy": 0.75, "second_bigbar_support20": 9.5,
            "bigbar_count20_v4": 2.0, "market_cap_yuan": np.nan, "fundamental_score": np.nan,
            "has_real_order": np.nan, "industry_leader_score": np.nan,
        }])
        market = pd.DataFrame([{
            "date": date, "market_mode": "range", "benchmark_ret5": 0.0, "benchmark_ret10": 0.01,
        }])
        good = annotate_regular_signals(signal, daily, market, False)
        self.assertTrue(bool(good.loc[0, "note_gate_pass"]))
        daily.loc[0, "down_days5"] = 1.0
        bad = annotate_regular_signals(signal, daily, market, False)
        self.assertFalse(bool(bad.loc[0, "note_gate_pass"]))

    def test_left_probe_fundamental_gate_is_not_fabricated(self):
        self.assertTrue(bool(V4CFG["note_rules"]["fundamental_required_for_production_left_probe"]))
        missing = pd.Series({"has_real_order": np.nan, "fundamental_score": np.nan, "industry_leader_score": np.nan})
        self.assertFalse(_fundamental_left_probe_pass(missing))
        order = pd.Series({"has_real_order": True, "fundamental_score": np.nan, "industry_leader_score": np.nan})
        self.assertTrue(_fundamental_left_probe_pass(order))

    def test_speculative_ambush_is_disabled_without_required_data(self):
        self.assertFalse(bool(V4CFG["note_rules"]["speculative_ambush_enabled"]))


if __name__ == "__main__":
    unittest.main()
