from __future__ import annotations

from typing import Any

import pandas as pd

import v3_engine as v3e
from v4_engine import run_v4_simulation
from v4_shared import V4CFG, BASE_PROFILE


def flat_stats(st: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return": st.get("total_return"),
        "max_drawdown": st.get("max_drawdown"),
        "trades": st.get("trades"),
        "win_rate": st.get("win_rate"),
        "profit_factor": st.get("profit_factor"),
        "sharpe": st.get("sharpe"),
        "calmar": st.get("calmar"),
        "benchmark_total_return": st.get("benchmark_total_return"),
        "excess_return": st.get("excess_return"),
        "realized_pnl": st.get("realized_pnl"),
        "unrealized_pnl": st.get("unrealized_pnl"),
    }


def left_probe_research_market(market: pd.DataFrame) -> pd.DataFrame:
    """Independent left-side risk book from the note: max 50% exposure even in a down-cycle.

    This is intentionally separate from the right-side 0/30/50/80 market cap. Otherwise a
    wave-3 left probe is structurally impossible in the exact risk-off regime where the note
    describes using it. Production still requires point-in-time fundamental confirmation.
    """
    m = market.copy()
    m["max_exposure"] = float(V4CFG["note_rules"]["left_probe_total_exposure_cap"])
    return m


def historical_ablation_suite(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    v4_entries: pd.DataFrame,
    v3_entries: pd.DataFrame,
    market: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descriptive only. The best row is never fed back into production policy selection."""
    rows: list[dict[str, Any]] = []
    _, _, _, _, v3_base = v3e.run_simulation(
        daily, exec_px, v3_entries, market, BASE_PROFILE,
        ["wave5_exhaust_reversal", "pingbu_qingyun"], start, end, keep_fills=False
    )
    rows.append({"variant": "v3_frozen_baseline", **flat_stats(v3_base)})

    variants = [
        ("v4_fixed", dict()),
        ("v4_no_range_impulse_exit", {"enable_range_impulse_exit": False}),
        ("v4_no_high_position_resize", {"enable_high_position_resize": False}),
        ("v4_no_right_pullback", {"enable_right_pullback": False}),
    ]
    stats_cache: dict[str, Any] = {"v3_frozen_baseline": v3_base}
    for name, kwargs in variants:
        _, _, _, _, st = run_v4_simulation(
            daily, exec_px, v4_entries, market, start, end, keep_fills=False, **kwargs
        )
        rows.append({"variant": name, **flat_stats(st)})
        stats_cache[name] = st

    raw_only = v4_entries[v4_entries["signal_type"] == "wave3_probe_raw"].copy() if len(v4_entries) else pd.DataFrame()
    left_market = left_probe_research_market(market)
    _, _, _, _, raw_stats = run_v4_simulation(
        daily, exec_px, raw_only, left_market, start, end,
        allow_technical_raw_wave3=True, keep_fills=False
    )
    rows.append({"variant": "raw_wave3_technical_only_research", **flat_stats(raw_stats)})
    stats_cache["raw_wave3_technical_only_research"] = raw_stats

    df = pd.DataFrame(rows)
    numeric = df["total_return"].astype(float)
    historical_best = str(df.loc[numeric.idxmax(), "variant"]) if len(df) else None
    audit = {
        "fixed_policy": V4CFG["fixed_policy"],
        "policy_selected_from_this_ablation": False,
        "historical_best_variant_for_information_only": historical_best,
        "left_probe_uses_independent_half_exposure_research_book": True,
        "warning": "March-August and the August examples in the note were already seen by the human research process. These ablations are explanatory historical replay only and must not be used to retune V4 after seeing their results.",
    }
    return df, audit


def monthly_fixed_policy_review(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    entries: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month_start in pd.to_datetime([
        "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"
    ]):
        month_end = month_start + pd.offsets.MonthEnd(0)
        _, _, _, _, st = run_v4_simulation(daily, exec_px, entries, market, month_start, month_end)
        rows.append({"month": month_start.strftime("%Y-%m"), **flat_stats(st)})
    return pd.DataFrame(rows)


def continuous_monthly_returns(curve: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    c = curve.sort_values("date").set_index("date")
    eq = c["equity"].resample("ME").last()
    first_base = pd.Series([1_000_000.0], index=[c.index.min() - pd.Timedelta(days=1)])
    eq2 = pd.concat([first_base, eq])
    rets = eq2.pct_change().iloc[1:]
    m = market.set_index("date")["benchmark_ret"].copy()
    bm = (1 + m.fillna(0)).groupby(m.index.to_period("M")).prod() - 1
    rows = []
    for dt, r in rets.items():
        p = dt.to_period("M")
        br = float(bm.get(p, float("nan")))
        rows.append({
            "month": str(p),
            "strategy_return": float(r),
            "benchmark_return": br,
            "excess_return": float(r - br) if pd.notna(br) else float("nan"),
        })
    return pd.DataFrame(rows)
