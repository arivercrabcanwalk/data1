from __future__ import annotations

import copy
import json
import os

import pandas as pd

import three_five_wave_backtest as v2
from v3_shared import *  # re-exported for tests and research notebooks
from v3_engine import *  # re-exported for tests and research notebooks
from v3_research import *  # re-exported for tests and research notebooks

def main() -> None:
    print("V3 loading source minute data")
    daily, exec_px, minute_map = v2.load_daily_from_minute()
    print("V3 adding stock indicators")
    original_cfg = copy.deepcopy(v2.CFG)
    try:
        v2.CFG = profile_v2_cfg("balanced" if "balanced" in PROFILES else next(iter(PROFILES)))
        daily = v2.add_all_stock_indicators(daily)
    finally:
        v2.CFG = original_cfg
    market = compute_market_features_v3(daily)
    daily, sector_available, sector_mode = add_point_in_time_sector_features(daily)
    print("sector mode", sector_mode)
    signals, signal_diag = detect_all_profiles(daily, market, sector_available)
    print("signals total", len(signals), "gated", int(signals["absolute_gate_pass"].sum()) if len(signals) else 0)
    print("precomputing strictly causal minute entries")
    entries = precompute_entries(daily, exec_px, minute_map, signals)
    selected_profile, selected_modules, research_matrix, selection = research_select_profile_and_modules(
        daily, exec_px, entries, market
    )
    print("selected profile", selected_profile, "modules", selected_modules)
    walk_forward = monthly_walk_forward(daily, exec_px, entries, market)
    curve, trades, open_positions, fills, full_stats = run_simulation(
        daily, exec_px, entries, market, selected_profile, selected_modules,
        pd.Timestamp(CFG["evaluation"]["development_start"]),
        pd.Timestamp(CFG["evaluation"]["test_end"]),
        keep_fills=True,
    )
    tcurve, ttrades, topen, _, test_stats = run_simulation(
        daily, exec_px, entries, market, selected_profile, selected_modules,
        pd.Timestamp(CFG["evaluation"]["test_start"]),
        pd.Timestamp(CFG["evaluation"]["test_end"]),
        keep_fills=False,
    )
    intraday_curve = replay_intraday_equity(
        daily, minute_map, fills,
        pd.Timestamp(CFG["evaluation"]["development_start"]),
        pd.Timestamp(CFG["evaluation"]["test_end"]),
    )
    if len(intraday_curve):
        full_stats["intraday_max_drawdown"] = float(intraday_curve["drawdown"].min())
        full_stats["intraday_final_equity_replay"] = float(intraday_curve["equity"].iloc[-1])
    else:
        full_stats["intraday_max_drawdown"] = None
        full_stats["intraday_final_equity_replay"] = None
    full_stats["strategy_version"] = CFG["version"]
    full_stats["selected_profile"] = selected_profile
    full_stats["selected_modules"] = selected_modules
    full_stats["selection"] = selection
    full_stats["algorithmic_test_stats"] = test_stats
    full_stats["sector_filter_enabled"] = bool(sector_available)
    full_stats["sector_mode"] = sector_mode
    full_stats["warmup"] = warmup_diagnostics(daily, signals)
    full_stats["lookahead_controls"] = {
        "daily_signal_uses_close_then_next_day_only": True,
        "minute_confirmation_executes_next_bar_open": True,
        "entry_mfe_mae_begin_at_execution_bar": True,
        "market_exposure_cap_uses_previous_close_and_rebalances_existing_positions": True,
        "point_in_time_sector_required": bool(CFG["integrity"]["require_point_in_time_sector_map"]),
        "static_sector_map_used": sector_mode == "static_map_explicitly_allowed",
        "algorithmic_test_used_for_selection": False,
    }
    full_stats["overfit_controls"] = {
        "profiles_predeclared": list(PROFILES.keys()),
        "profile_count": len(PROFILES),
        "full_grid_search": False,
        "development_period": [CFG["evaluation"]["development_start"], CFG["evaluation"]["development_end"]],
        "validation_period": [CFG["evaluation"]["validation_start"], CFG["evaluation"]["validation_end"]],
        "algorithmic_test_period": [CFG["evaluation"]["test_start"], CFG["evaluation"]["test_end"]],
        "human_seen_test_warning": bool(CFG["evaluation"]["human_seen_test_warning"]),
    }

    curve.to_csv(os.path.join(OUT, "equity_curve.csv"), index=False)
    trades.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    open_positions.to_csv(os.path.join(OUT, "open_positions.csv"), index=False)
    fills.to_csv(os.path.join(OUT, "fills.csv"), index=False)
    signals.to_csv(os.path.join(OUT, "signals_all_profiles.csv"), index=False)
    entries.to_csv(os.path.join(OUT, "entry_confirmations_all_profiles.csv"), index=False)
    signal_diag.to_csv(os.path.join(OUT, "signal_diagnostics_all_profiles.csv"), index=False)
    market.to_csv(os.path.join(OUT, "market_regime_and_benchmark.csv"), index=False)
    research_matrix.to_csv(os.path.join(OUT, "research_matrix.csv"), index=False)
    walk_forward.to_csv(os.path.join(OUT, "walk_forward_monthly.csv"), index=False)
    if len(intraday_curve):
        intraday_curve.to_csv(os.path.join(OUT, "intraday_equity_curve.csv"), index=False)
    if len(tcurve):
        tcurve.to_csv(os.path.join(OUT, "algorithmic_test_equity.csv"), index=False)
    if len(ttrades):
        ttrades.to_csv(os.path.join(OUT, "algorithmic_test_trades.csv"), index=False)
    if len(topen):
        topen.to_csv(os.path.join(OUT, "algorithmic_test_open_positions.csv"), index=False)
    with open(os.path.join(OUT, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(full_stats, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(full_stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
