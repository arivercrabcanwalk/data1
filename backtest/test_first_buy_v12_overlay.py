import unittest

from first_buy_v12_overlay import (
    classify_expansion_lane,
    confirm_panic_repair,
    confirm_qualified_probe_add,
    first_buy_semantics_ok,
    qualified_probe_add_watch,
)


def base(**kw):
    row = {
        "prior_streak": 5,
        "is_first_break": True,
        "is_limit_up": False,
        "is_secondary_break": False,
        "market_height_gap": 0,
        "event_ret": 0.05,
        "close_vwap": 1.01,
        "close_loc": 0.70,
        "d1_gap": 0.09,
        "d1_tradable": True,
        "vol_prev_ratio": 1.6,
        "vol_med10": 3.0,
        "amount": 1_500_000_000,
        "range": 0.10,
        "prior_theme_leader": False,
        "prior_sector_rank": 2,
        "event_touched_upper": False,
        "v11_qualified_probe": False,
    }
    row.update(kw)
    return row


class TestV12Overlay(unittest.TestCase):
    def test_semantic_guard_rejects_non_first_break(self):
        self.assertFalse(first_buy_semantics_ok(base(is_first_break=False)))
        self.assertFalse(first_buy_semantics_ok(base(is_secondary_break=True)))
        self.assertFalse(first_buy_semantics_ok(base(prior_streak=3)))

    def test_height_high_gap(self):
        self.assertEqual(
            classify_expansion_lane(base()),
            "HIGH_GAP_HEIGHT_5PLUS",
        )

    def test_high_gap_requires_tradability(self):
        self.assertIsNone(classify_expansion_lane(base(d1_tradable=False)))

    def test_theme_four_requires_touch(self):
        row = base(
            prior_streak=4,
            market_height_gap=4,
            prior_theme_leader=True,
            event_touched_upper=True,
        )
        self.assertEqual(classify_expansion_lane(row), "HIGH_GAP_THEME_4")
        self.assertIsNone(classify_expansion_lane({**row, "event_touched_upper": False}))

    def test_super_high_absorption(self):
        row = base(
            prior_streak=9,
            d1_gap=0.04,
            event_ret=0.0,
            close_loc=0.90,
            close_vwap=1.03,
        )
        self.assertEqual(classify_expansion_lane(row), "SUPER_HIGH_ABSORPTION")
        self.assertIsNone(classify_expansion_lane({**row, "close_loc": 0.70}))

    def test_panic_needs_causal_confirmation(self):
        row = base(
            d1_gap=-0.09,
            event_ret=0.06,
            close_loc=0.80,
            close_vwap=1.02,
        )
        self.assertEqual(classify_expansion_lane(row), "PANIC_REPAIR_5PLUS_WATCH")
        self.assertTrue(confirm_panic_repair(row, 23))
        self.assertFalse(confirm_panic_repair(row, 61))
        self.assertFalse(confirm_panic_repair(row, None))

    def test_true_red_6_7_rejects_low_open(self):
        row = base(
            prior_streak=6,
            event_ret=-0.03,
            close_vwap=0.98,
            d1_gap=0.02,
        )
        self.assertEqual(classify_expansion_lane(row), "TRUE_RED_6_7_HEIGHT_OVERRIDE")
        self.assertIsNone(classify_expansion_lane({**row, "d1_gap": -0.01}))

    def test_probe_add_never_creates_candidate(self):
        row = base(
            d1_gap=-0.01,
            event_ret=0.03,
            v11_qualified_probe=True,
        )
        self.assertTrue(qualified_probe_add_watch(row))
        self.assertTrue(confirm_qualified_probe_add(row, 32))
        self.assertFalse(confirm_qualified_probe_add(row, 75))


if __name__ == "__main__":
    unittest.main()
