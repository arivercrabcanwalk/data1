from __future__ import annotations

from typing import Any, Mapping, Optional


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def _b(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    return bool(value)


def strict_explosion(row: Mapping[str, Any]) -> bool:
    vol_prev = _f(row, "vol_prev_ratio")
    vol_med10 = _f(row, "vol_med10")
    return vol_prev >= 1.50 or (vol_med10 >= 2.00 and vol_prev >= 0.80)


def soft_explosion(row: Mapping[str, Any]) -> bool:
    vol_prev = _f(row, "vol_prev_ratio")
    vol_med10 = _f(row, "vol_med10")
    amount = _f(row, "amount")
    day_range = _f(row, "range")
    return (
        vol_prev >= 1.35
        or (vol_med10 >= 2.00 and vol_prev >= 0.70)
        or (amount >= 1_000_000_000 and day_range >= 0.08 and vol_prev >= 0.65)
    )


def first_buy_semantics_ok(row: Mapping[str, Any]) -> bool:
    # Mandatory semantic guard: this project accepts only the first non-limit-up
    # day immediately following a corrected >=4 limit-up streak.
    return (
        int(_f(row, "prior_streak")) >= 4
        and _b(row, "is_first_break")
        and not _b(row, "is_limit_up")
        and not _b(row, "is_secondary_break")
    )


def classify_expansion_lane(row: Mapping[str, Any]) -> Optional[str]:
    """
    Return a V12 expansion lane. None means the row must fall back to the
    unchanged V10/V11 mainline or be rejected.

    This function deliberately does not implement the V10/V11 mainline.
    It only implements narrowly-scoped expansion lanes.
    """
    if not first_buy_semantics_ok(row):
        return None

    streak = int(_f(row, "prior_streak"))
    height_gap = _f(row, "market_height_gap", 99)
    event_ret = _f(row, "event_ret")
    close_vwap = _f(row, "close_vwap")
    close_loc = _f(row, "close_loc")
    d1_gap = _f(row, "d1_gap", 99)
    d1_tradable = _b(row, "d1_tradable")

    # A. 5+ board market-height leader with a tradable near-limit D+1 open.
    if (
        streak >= 5
        and height_gap <= 1
        and soft_explosion(row)
        and event_ret >= 0
        and close_vwap >= 0.98
        and close_loc >= 0.35
        and 0.08 <= d1_gap <= 0.102
        and d1_tradable
    ):
        return "HIGH_GAP_HEIGHT_5PLUS"

    # B. 4-board thematic leader. Require a touch of the upper limit on the
    # divergence day so a generic high-open continuation cannot enter.
    prior_theme_core = _b(row, "prior_theme_leader") or _f(row, "prior_sector_rank", 999) <= 1
    if (
        streak == 4
        and prior_theme_core
        and _b(row, "event_touched_upper")
        and strict_explosion(row)
        and event_ret >= 0
        and close_vwap >= 0.98
        and 0.08 <= d1_gap <= 0.102
        and d1_tradable
    ):
        return "HIGH_GAP_THEME_4"

    # C. Super-high leader. This is an experimental lane and should be
    # position-sized separately by the execution layer.
    if (
        streak >= 8
        and height_gap <= 1
        and soft_explosion(row)
        and event_ret >= -0.02
        and close_loc >= 0.80
        and close_vwap >= 1.02
        and -0.03 <= d1_gap <= 0.08
    ):
        return "SUPER_HIGH_ABSORPTION"

    # D. Panic repair is not an open buy. The event becomes a watch only;
    # execution requires a separate causal intraday confirmation.
    if (
        streak >= 5
        and height_gap <= 1
        and soft_explosion(row)
        and event_ret >= 0.03
        and close_loc >= 0.60
        and close_vwap >= 0.98
        and -0.102 <= d1_gap <= -0.04
    ):
        return "PANIC_REPAIR_5PLUS_WATCH"

    # E. 6-7 board true-red market-height override. Low-open D+1 is excluded.
    if (
        6 <= streak <= 7
        and height_gap <= 1
        and strict_explosion(row)
        and -0.08 <= event_ret < 0
        and close_vwap >= 0.95
        and 0 <= d1_gap <= 0.08
    ):
        return "TRUE_RED_6_7_HEIGHT_OVERRIDE"

    return None


def confirm_panic_repair(
    row: Mapping[str, Any],
    reclaim_minutes: Optional[int],
) -> bool:
    return (
        classify_expansion_lane(row) == "PANIC_REPAIR_5PLUS_WATCH"
        and reclaim_minutes is not None
        and 0 <= int(reclaim_minutes) <= 60
    )


def qualified_probe_add_watch(row: Mapping[str, Any]) -> bool:
    """
    Position upgrade only. It never creates a new candidate.
    """
    return (
        _b(row, "v11_qualified_probe")
        and first_buy_semantics_ok(row)
        and _f(row, "event_ret") >= 0
        and -0.02 <= _f(row, "d1_gap") < 0
    )


def confirm_qualified_probe_add(
    row: Mapping[str, Any],
    reclaim_minutes: Optional[int],
) -> bool:
    return (
        qualified_probe_add_watch(row)
        and reclaim_minutes is not None
        and 0 <= int(reclaim_minutes) <= 60
    )
