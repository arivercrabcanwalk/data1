import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import v3_backtest as v3


class V3CausalityTests(unittest.TestCase):
    def test_intraday_confirmation_executes_next_bar_open(self):
        ts = pd.date_range('2026-06-01 09:30', periods=31, freq='min')
        rows = []
        for i, t in enumerate(ts):
            close = 100.0 + 0.08 * i
            rows.append({
                'code': '000001', 'datetime': t,
                'open': close - 0.03, 'high': close + 0.05,
                'low': 100.0 if i == 0 else close - 0.06,
                'close': close, 'volume': 1000.0, 'amount': 100000.0,
            })
        x = pd.DataFrame(rows)
        x.loc[x['datetime'] == pd.Timestamp('2026-06-01 09:45'), 'high'] = 110.0
        signal = pd.Series({'stop_anchor': 99.5})
        status = pd.Series({
            'is_st': False, 'is_star_st': False, 'is_new_listing_initial': False,
            'is_suspended': False, 'price_limit_up_pct': 0.10,
        })
        out, reason = v3.causal_find_intraday_entry(x, signal, 100.0, status, 'balanced')
        self.assertIsNotNone(out, reason)
        self.assertEqual(pd.Timestamp(out['confirmation_time']), pd.Timestamp('2026-06-01 09:45'))
        self.assertEqual(pd.Timestamp(out['execution_time']), pd.Timestamp('2026-06-01 09:46'))
        expected_open = float(x.loc[x['datetime'] == pd.Timestamp('2026-06-01 09:46'), 'open'].iloc[0])
        self.assertAlmostEqual(out['raw_price'], expected_open)
        self.assertLess(out['entry_day_post_high'], 110.0)

    def test_risk_sizing_shrinks_when_stop_is_wider(self):
        narrow, narrow_cost, _ = v3.risk_sized_shares(1_000_000, 1_000_000, 800_000, 100.0, 95.0)
        wide, wide_cost, _ = v3.risk_sized_shares(1_000_000, 1_000_000, 800_000, 100.0, 90.0)
        self.assertGreater(narrow, wide)
        self.assertGreater(narrow_cost, wide_cost)
        self.assertEqual(narrow % 100, 0)
        self.assertEqual(wide % 100, 0)

    def test_static_sector_map_is_ignored_by_default(self):
        path = Path(v3.STATIC_SECTOR_PATH)
        history = Path(v3.SECTOR_HISTORY_PATH)
        if history.exists():
            history.unlink()
        path.write_text('code,sector\n000001,test\n', encoding='utf-8')
        try:
            daily = pd.DataFrame({'date': [pd.Timestamp('2026-03-02')], 'code': ['000001']})
            out, enabled, mode = v3.add_point_in_time_sector_features(daily)
            self.assertFalse(enabled)
            self.assertEqual(mode, 'static_map_ignored_to_avoid_historical_lookahead')
            self.assertTrue(out['sector_rank'].isna().all())
        finally:
            if path.exists():
                path.unlink()

    def test_existing_position_is_rebalanced_when_previous_day_cap_falls_to_zero(self):
        dates = pd.to_datetime(['2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05'])
        daily = pd.DataFrame({
            'date': dates,
            'code': ['000001'] * 4,
            'open': [100, 100, 100, 100],
            'high': [101, 102, 101, 101],
            'low': [99, 99, 99, 99],
            'close': [100, 101, 100, 100],
            'rs20_rank': [0.7, 0.7, 0.4, 0.4],
            'rs5_rank': [0.7, 0.7, 0.4, 0.4],
            'ma5': [99, 99, 101, 101],
        })
        exec_px = pd.DataFrame({
            'date': dates, 'code': ['000001'] * 4,
            'first_open': [100, 100, 100, 100],
            'first_close': [100, 100, 100, 100], 'last_close': [100, 101, 100, 100],
            'is_st': [False]*4, 'is_star_st': [False]*4,
            'is_new_listing_initial': [False]*4, 'is_suspended': [False]*4,
            'price_limit_up_pct': [0.10]*4, 'price_limit_down_pct': [0.10]*4,
        })
        market = pd.DataFrame({
            'date': dates,
            'market_score': [70, 20, 20, 70],
            'max_exposure': [0.80, 0.0, 0.0, 0.80],
            'benchmark_ret': [0.0, 0.0, 0.0, 0.0],
        })
        entries = pd.DataFrame([{
            'profile': 'balanced', 'signal_type': 'pingbu_qingyun', 'entry_confirmed': True,
            'entry_date': dates[1], 'code': '000001', 'raw_price': 100.0,
            'execution_time': pd.Timestamp('2026-03-03 10:00'),
            'entry_day_post_high': 102.0, 'entry_day_post_low': 99.5,
            'stop_anchor': 95.0, 'signal_date': dates[0], 'candidate_score': 80.0,
        }])
        curve, trades, open_pos, _, stats = v3.run_simulation(
            daily, exec_px, entries, market, 'balanced', ['pingbu_qingyun'], dates[0], dates[-1]
        )
        self.assertTrue((trades['exit_reason'] == 'exposure_rebalance').any())
        day3 = curve[curve['date'] == dates[2]].iloc[0]
        self.assertAlmostEqual(float(day3['applied_exposure_cap']), 0.0)
        self.assertAlmostEqual(float(day3['actual_exposure']), 0.0)
        self.assertEqual(len(open_pos), 0)

    def test_realized_and_unrealized_are_reported_separately(self):
        curve = pd.DataFrame({'date': pd.to_datetime(['2026-08-30','2026-08-31']), 'equity': [1_000_000, 1_100_000]})
        trades = pd.DataFrame([{'pnl': 40_000.0, 'net_return_pct': 0.04, 'signal_type':'pingbu_qingyun', 'exit_reason':'time_stop', 'mfe_pct':0.08, 'mae_pct':-0.02, 'hold_days':5}])
        open_pos = pd.DataFrame([{'unrealized_pnl': 60_000.0}])
        market = pd.DataFrame({'date': pd.to_datetime(['2026-08-30','2026-08-31']), 'benchmark_ret':[0.0,0.0]})
        st = v3.compute_stats(curve, trades, open_pos, market, 1_000_000)
        self.assertEqual(st['realized_pnl'], 40_000.0)
        self.assertEqual(st['unrealized_pnl'], 60_000.0)
        self.assertAlmostEqual(st['total_return'], 0.10)

    def test_warmup_report_does_not_pretend_march_is_fully_warmed(self):
        dates = pd.bdate_range('2026-03-02', periods=40)
        daily = pd.DataFrame({'date': dates, 'code': ['000001']*40})
        signals = pd.DataFrame(columns=['date','signal_type'])
        w = v3.warmup_diagnostics(daily, signals)
        self.assertFalse(w['warmup_complete_at_development_start'])
        self.assertEqual(w['prior_trading_days_available_before_development_start'], 0)

    def test_selection_objective_penalizes_tiny_sample(self):
        tiny = {'trades': 2, 'total_return': 0.50, 'max_drawdown': -0.05}
        adequate = {'trades': 20, 'total_return': 0.15, 'max_drawdown': -0.08}
        self.assertLess(v3.selection_objective(tiny), v3.selection_objective(adequate))


if __name__ == '__main__':
    unittest.main()
