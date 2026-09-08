from __future__ import annotations

import math
from typing import Any

import pandas as pd

from v3_shared import CFG, PROFILES, SIGNAL_TYPES
from v3_engine import run_simulation

def selection_objective(stats: dict[str, Any]) -> float:
    min_trades = int(CFG["evaluation"]["min_development_trades"])
    trades = int(stats.get("trades", 0))
    if trades < min_trades:
        return -1e9
    dd = abs(float(stats.get("max_drawdown", 0.0)))
    ret = float(stats.get("total_return", 0.0))
    sample_penalty = min(1.0, math.sqrt(trades / 20.0))
    return (ret / max(dd, 0.05)) * sample_penalty


def research_select_profile_and_modules(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    entries: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[str, list[str], pd.DataFrame, dict[str, Any]]:
    ev = CFG["evaluation"]
    ds, de = pd.Timestamp(ev["development_start"]), pd.Timestamp(ev["development_end"])
    vs, ve = pd.Timestamp(ev["validation_start"]), pd.Timestamp(ev["validation_end"])
    ts, te = pd.Timestamp(ev["test_start"]), pd.Timestamp(ev["test_end"])
    matrix: list[dict[str, Any]] = []
    dev_stats_by_profile: dict[str, dict[str, Any]] = {}
    val_stats_by_profile: dict[str, dict[str, Any]] = {}

    # Phase A: development and validation only. Test-period performance is not read here.
    for profile_name in PROFILES:
        _, _, _, _, dev = run_simulation(daily, exec_px, entries, market, profile_name, SIGNAL_TYPES, ds, de)
        _, _, _, _, val = run_simulation(daily, exec_px, entries, market, profile_name, SIGNAL_TYPES, vs, ve)
        dev_stats_by_profile[profile_name] = dev
        val_stats_by_profile[profile_name] = val
        matrix.extend([
            {"profile": profile_name, "module": "combined_all", "period": "development", **flat_stats(dev)},
            {"profile": profile_name, "module": "combined_all", "period": "validation", **flat_stats(val)},
        ])
    ranked = sorted(PROFILES, key=lambda p: (-selection_objective(dev_stats_by_profile[p]), p))
    selected = None
    for p in ranked:
        val = val_stats_by_profile[p]
        if (
            int(val.get("trades", 0)) >= int(ev["min_validation_trades"])
            and float(val.get("total_return", 0.0)) >= float(ev["validation_min_return"])
            and float(val.get("max_drawdown", 0.0)) >= float(ev["validation_max_drawdown"])
        ):
            selected = p
            break
    validation_fallback = False
    if selected is None:
        selected = "strict" if "strict" in PROFILES else ranked[0]
        validation_fallback = True

    # Phase B: choose modules from development + validation only.
    selected_modules: list[str] = []
    module_dev_cache: dict[str, dict[str, Any]] = {}
    module_val_cache: dict[str, dict[str, Any]] = {}
    for module in SIGNAL_TYPES:
        _, _, _, _, dev = run_simulation(daily, exec_px, entries, market, selected, [module], ds, de)
        _, _, _, _, val = run_simulation(daily, exec_px, entries, market, selected, [module], vs, ve)
        module_dev_cache[module] = dev
        module_val_cache[module] = val
        matrix.extend([
            {"profile": selected, "module": module, "period": "development", **flat_stats(dev)},
            {"profile": selected, "module": module, "period": "validation", **flat_stats(val)},
        ])
        dev_pf = dev.get("profit_factor")
        dev_pf_num = float(dev_pf) if isinstance(dev_pf, (int, float)) else (99.0 if dev_pf == "inf" else 0.0)
        if (
            int(dev.get("trades", 0)) >= int(ev["module_min_development_trades"])
            and dev_pf_num >= float(ev["module_min_development_profit_factor"])
            and float(dev.get("total_return", 0.0)) >= float(ev["module_min_development_return"])
            and int(val.get("trades", 0)) >= int(ev["module_min_validation_trades"])
            and float(val.get("total_return", 0.0)) >= float(ev["module_min_validation_return"])
            and float(val.get("max_drawdown", 0.0)) >= float(ev["validation_max_drawdown"])
        ):
            selected_modules.append(module)
    if not selected_modules:
        def module_obj(st: dict[str, Any]) -> float:
            trades = int(st.get("trades", 0))
            if trades < int(ev["module_min_development_trades"]):
                return -1e9
            dd = abs(float(st.get("max_drawdown", 0.0)))
            ret = float(st.get("total_return", 0.0))
            return (ret / max(dd, 0.05)) * min(1.0, math.sqrt(trades / 10.0))
        selected_modules = [sorted(SIGNAL_TYPES, key=lambda m: (-module_obj(module_dev_cache[m]), m))[0]]

    # Phase C: only after profile/modules are frozen do we evaluate the algorithmic test period.
    for profile_name in PROFILES:
        _, _, _, _, tst = run_simulation(daily, exec_px, entries, market, profile_name, SIGNAL_TYPES, ts, te)
        matrix.append({"profile": profile_name, "module": "combined_all", "period": "algorithmic_test", **flat_stats(tst)})
    for module in SIGNAL_TYPES:
        _, _, _, _, tst = run_simulation(daily, exec_px, entries, market, selected, [module], ts, te)
        matrix.append({"profile": selected, "module": module, "period": "algorithmic_test", **flat_stats(tst)})

    matrix_df = pd.DataFrame(matrix)
    selection = {
        "selected_profile": selected,
        "selected_modules": selected_modules,
        "profile_rank_from_development_only": ranked,
        "profile_objective_development": {p: selection_objective(dev_stats_by_profile[p]) for p in PROFILES},
        "validation_fallback_used": validation_fallback,
        "algorithmic_test_not_read_until_after_selection_frozen": True,
        "algorithmic_test_not_used_for_selection": True,
        "human_seen_test_warning": bool(ev.get("human_seen_test_warning", True)),
        "note": "August is excluded from programmatic selection, but it has been seen by the human research process in earlier iterations, so it is not a pristine untouched out-of-sample test.",
    }
    return selected, selected_modules, matrix_df, selection


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
    }


def monthly_walk_forward(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    entries: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    dates = pd.Series(sorted(pd.Timestamp(d) for d in daily["date"].unique()))
    rows: list[dict[str, Any]] = []
    for month_start in pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"]):
        month_end = month_start + pd.offsets.MonthEnd(0)
        train_end = month_start - pd.Timedelta(days=1)
        train_start = dates.iloc[0]
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for profile_name in PROFILES:
            _, _, _, _, train = run_simulation(daily, exec_px, entries, market, profile_name, SIGNAL_TYPES, train_start, train_end)
            candidates.append((selection_objective(train), profile_name, train))
        _, chosen, train_stats = sorted(candidates, key=lambda z: (-z[0], z[1]))[0]
        _, _, _, _, month_stats = run_simulation(daily, exec_px, entries, market, chosen, SIGNAL_TYPES, month_start, month_end)
        rows.append({
            "month": month_start.strftime("%Y-%m"),
            "profile_selected_using_prior_data_only": chosen,
            "train_end": train_end,
            "train_objective": selection_objective(train_stats),
            **flat_stats(month_stats),
        })
    return pd.DataFrame(rows)


def warmup_diagnostics(daily: pd.DataFrame, signals: pd.DataFrame) -> dict[str, Any]:
    dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    min_hist = int(CFG["integrity"]["wave_min_history"])
    first_wave_eligible = dates[min_hist] if len(dates) > min_hist else None
    wave_signals = signals[signals["signal_type"].isin(["wave3_div_reversal", "wave5_exhaust_reversal"])] if not signals.empty else pd.DataFrame()
    first_wave_signal = pd.Timestamp(wave_signals["date"].min()) if len(wave_signals) else None
    dev_start = pd.Timestamp(CFG["evaluation"]["development_start"])
    prior_obs = sum(1 for d in dates if d < dev_start)
    return {
        "dataset_first_date": str(dates[0].date()) if dates else None,
        "requested_development_start": str(dev_start.date()),
        "prior_trading_days_available_before_development_start": prior_obs,
        "wave_min_history_required": min_hist,
        "warmup_complete_at_development_start": bool(prior_obs >= min_hist),
        "first_wave_structurally_eligible_date": str(first_wave_eligible.date()) if first_wave_eligible else None,
        "first_actual_wave_signal_date": str(first_wave_signal.date()) if first_wave_signal is not None else None,
        "warning": "January-February data are not present in the repository; March wave statistics therefore begin only after genuine in-repository warm-up. No synthetic or future data are used to backfill this gap." if prior_obs < min_hist else None,
    }
