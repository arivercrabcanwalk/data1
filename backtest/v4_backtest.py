from __future__ import annotations

import copy
import json
import os

import pandas as pd

import three_five_wave_backtest as v2
import v3_engine as v3e
import v3_shared as v3
from v4_shared import (
    OUT, V4CFG, BASE_PROFILE,
    add_market_modes, add_v4_stock_features, add_point_in_time_fundamentals,
    annotate_regular_signals, detect_raw_wave3_probes, precompute_v4_entries,
)
from v4_engine import run_v4_simulation
from v4_research import (
    historical_ablation_suite, monthly_fixed_policy_review, continuous_monthly_returns,
    left_probe_research_market,
)


def main() -> None:
    print("V4 loading March-August minute data")
    daily, exec_px, minute_map = v2.load_daily_from_minute()

    print("V4 adding causal stock indicators")
    original_cfg = copy.deepcopy(v2.CFG)
    try:
        v2.CFG = v3.profile_v2_cfg(BASE_PROFILE)
        daily = v2.add_all_stock_indicators(daily)
    finally:
        v2.CFG = original_cfg

    market = add_market_modes(v3.compute_market_features_v3(daily))
    daily, sector_available, sector_mode = v3.add_point_in_time_sector_features(daily)
    daily = add_v4_stock_features(daily)
    daily, fundamental_available, fundamental_mode = add_point_in_time_fundamentals(daily)
    print("sector mode", sector_mode)
    print("fundamental mode", fundamental_mode)

    print("V4 detecting frozen strict V3 signals once")
    regular, signal_diag = v3.detect_profile_signals(daily, market, sector_available, BASE_PROFILE)
    # V3 scoring already carries these ranks. Drop them before merging the richer V4 daily row
    # so pandas does not create rs5_rank_x/rs5_rank_y style duplicate columns.
    regular = regular.drop(columns=["rs5_rank", "rs20_rank", "liquidity_rank"], errors="ignore")
    regular = annotate_regular_signals(regular, daily, market, fundamental_available)
    raw_wave3 = detect_raw_wave3_probes(daily, market, fundamental_available)
    print("regular signals", len(regular), "raw wave3 probes", len(raw_wave3))

    print("V4 precomputing strict causal entries")
    v3_entries = v3.precompute_entries(daily, exec_px, minute_map, regular)
    regular_v4_entries = v3_entries[v3_entries["note_gate_pass"] == True].copy() if len(v3_entries) else pd.DataFrame()
    raw_entries = precompute_v4_entries(daily, exec_px, minute_map, pd.DataFrame(), raw_wave3)
    v4_entries = pd.concat([regular_v4_entries, raw_entries], ignore_index=True, sort=False) if len(raw_entries) else regular_v4_entries.copy()

    start = pd.Timestamp("2026-03-02")
    end = pd.Timestamp("2026-08-31")
    print("V4 running preregistered fixed policy; no result-driven policy selection")
    curve, trades, open_positions, fills, stats = run_v4_simulation(
        daily, exec_px, v4_entries, market, start, end, keep_fills=True
    )
    intraday = v3e.replay_intraday_equity(daily, minute_map, fills, start, end) if len(fills) else pd.DataFrame()
    if len(intraday):
        stats["intraday_max_drawdown"] = float(intraday["drawdown"].min())
        stats["intraday_final_equity_replay"] = float(intraday["equity"].iloc[-1])
    else:
        stats["intraday_max_drawdown"] = None
        stats["intraday_final_equity_replay"] = None

    ablation, ablation_audit = historical_ablation_suite(
        daily, exec_px, v4_entries, v3_entries, market, start, end
    )
    monthly_independent = monthly_fixed_policy_review(daily, exec_px, v4_entries, market)
    monthly_continuous = continuous_monthly_returns(curve, market)

    raw_only = v4_entries[v4_entries["signal_type"] == "wave3_probe_raw"].copy() if len(v4_entries) else pd.DataFrame()
    left_market = left_probe_research_market(market)
    _, raw_trades, raw_open, _, raw_stats = run_v4_simulation(
        daily, exec_px, raw_only, left_market, start, end,
        allow_technical_raw_wave3=True, keep_fills=False
    )

    stats["policy_preregistered_before_backtest"] = True
    stats["policy_selected_from_march_august_results"] = False
    stats["historical_review_only"] = bool(V4CFG["historical_review_only"])
    stats["human_seen_data_warning"] = (
        "The new note itself contains August examples and March-August has already been repeatedly inspected. "
        "V4 March-August results are historical replay, not pristine out-of-sample evidence. No V4 rule is changed after reading this run."
    )
    stats["sector_filter_enabled"] = bool(sector_available)
    stats["sector_mode"] = sector_mode
    stats["fundamental_filter_enabled"] = bool(fundamental_available)
    stats["fundamental_mode"] = fundamental_mode
    stats["speculative_ambush_mode"] = {
        "enabled": bool(V4CFG["note_rules"]["speculative_ambush_enabled"]),
        "status": "disabled: note describes this as largely random capital-game behavior and repository lacks point-in-time market-cap/earnings labels",
    }
    stats["left_wave3_mode"] = {
        "production_enabled": bool(fundamental_available),
        "production_requires_point_in_time_fundamental_confirmation": bool(V4CFG["note_rules"]["fundamental_required_for_production_left_probe"]),
        "separate_risk_book_exposure_cap": float(V4CFG["note_rules"]["left_probe_total_exposure_cap"]),
        "technical_only_research_stats": raw_stats,
    }
    stats["lookahead_controls_v4"] = {
        "daily_mode_and_signal_known_only_after_close": True,
        "entries_execute_on_next_day_after_completed_minute_confirmation": True,
        "left_probe_waits_for_observed_sharp_drop_then_stabilization": True,
        "left_probe_uses_independent_half_exposure_book_in_down_cycle": True,
        "range_big_rise_exit_uses_prior_completed_daily_bar_then_next_open": True,
        "market_exposure_uses_previous_close": True,
        "static_sector_or_fundamental_snapshot_backfill_forbidden": True,
        "march_august_result_not_used_to_select_v4_policy": True,
    }
    stats["note_integration"] = {
        "mode_first": True,
        "range_support_requires_3_to_15_day_adjustment_and_core_proxy": True,
        "range_large_rise_exits_next_open": True,
        "support_break_is_hard_exit": True,
        "high_position_high_volatility_reduces_risk_and_may_use_lower_second_bigbar_support": True,
        "raw_wave3_left_probe_is_half_risk_and_requires_point_in_time_fundamental_confirmation_for_production": True,
        "pure_speculative_ambush_not_promoted_to_production": True,
    }
    stats["ablation_audit"] = ablation_audit
    stats["signal_counts"] = regular["signal_type"].value_counts().to_dict() if len(regular) else {}
    stats["note_gated_signal_counts"] = regular.loc[regular["note_gate_pass"], "signal_type"].value_counts().to_dict() if len(regular) else {}
    stats["raw_wave3_probe_count"] = int(len(raw_wave3))
    stats["raw_wave3_technical_gate_count"] = int(raw_wave3["technical_gate_pass"].sum()) if len(raw_wave3) else 0
    stats["raw_wave3_production_gate_count"] = int(raw_wave3["production_gate_pass"].sum()) if len(raw_wave3) else 0

    curve.to_csv(os.path.join(OUT, "equity_curve.csv"), index=False)
    trades.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    open_positions.to_csv(os.path.join(OUT, "open_positions.csv"), index=False)
    fills.to_csv(os.path.join(OUT, "fills.csv"), index=False)
    regular.to_csv(os.path.join(OUT, "regular_signals.csv"), index=False)
    raw_wave3.to_csv(os.path.join(OUT, "raw_wave3_probes.csv"), index=False)
    v4_entries.to_csv(os.path.join(OUT, "entries.csv"), index=False)
    v3_entries.to_csv(os.path.join(OUT, "v3_strict_entries_for_baseline.csv"), index=False)
    signal_diag.to_csv(os.path.join(OUT, "signal_diagnostics.csv"), index=False)
    market.to_csv(os.path.join(OUT, "market_modes.csv"), index=False)
    ablation.to_csv(os.path.join(OUT, "historical_ablation.csv"), index=False)
    monthly_independent.to_csv(os.path.join(OUT, "monthly_fixed_policy_independent_cash.csv"), index=False)
    monthly_continuous.to_csv(os.path.join(OUT, "monthly_continuous.csv"), index=False)
    raw_trades.to_csv(os.path.join(OUT, "raw_wave3_technical_only_trades.csv"), index=False)
    raw_open.to_csv(os.path.join(OUT, "raw_wave3_technical_only_open.csv"), index=False)
    if len(intraday):
        intraday.to_csv(os.path.join(OUT, "intraday_equity_curve.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
