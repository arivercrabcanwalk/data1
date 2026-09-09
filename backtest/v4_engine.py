from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import three_five_wave_backtest as v2
import v3_engine as v3e
import v3_shared as v3s
from v4_shared import V4CFG, V4_PRIORITY, BASE_PROFILE

START_CASH = v3s.START_CASH
COMMISSION = v3s.COMMISSION
SELL_STAMP = v3s.SELL_STAMP
SLIPPAGE = v3s.SLIPPAGE
MAX_POSITIONS = v3s.MAX_POSITIONS
MAX_HOLD_DAYS = v3s.MAX_HOLD_DAYS


@dataclass
class V4Position:
    code: str
    shares: float
    entry_date: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    entry_cost: float
    stop_anchor: float
    initial_stop: float
    trailing_stop: float
    signal_type: str
    signal_date: pd.Timestamp
    profile: str
    candidate_score: float
    trade_mode: str
    entry_market_mode: str
    risk_multiplier: float
    high_position_unstable: bool
    hold_days: int = 0
    highest_close: float = 0.0
    highest_high: float = 0.0
    lowest_low: float = math.inf


def v4_risk_sized_shares(
    equity: float,
    cash: float,
    remaining_exposure: float,
    entry_price: float,
    initial_stop: float,
    risk_multiplier: float,
    position_fraction_multiplier: float = 1.0,
) -> tuple[float, float, float]:
    stop_distance = float(entry_price - initial_stop)
    if stop_distance <= 0 or equity <= 0 or cash <= 0 or remaining_exposure <= 0:
        return 0.0, 0.0, stop_distance
    r = v3s.CFG["risk"]
    risk_budget = equity * float(r["risk_per_trade"]) * max(0.0, float(risk_multiplier))
    risk_shares = math.floor((risk_budget / stop_distance) / 100.0) * 100.0
    position_cap = equity * float(r["max_position_fraction"]) * max(0.0, float(position_fraction_multiplier))
    alloc_cap = min(cash, remaining_exposure, position_cap)
    cap_shares = math.floor((alloc_cap / (entry_price * (1 + COMMISSION))) / 100.0) * 100.0
    shares = min(risk_shares, cap_shares)
    cost = shares * entry_price * (1 + COMMISSION)
    return shares, cost, stop_distance


def sell_v4_position(
    pos: V4Position,
    date: pd.Timestamp,
    er: pd.Series,
    reason: str,
    cash: float,
    trades: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> float:
    raw = float(er["first_open"])
    px = raw * (1 - SLIPPAGE)
    proceeds = pos.shares * px * (1 - COMMISSION - SELL_STAMP)
    cash += proceeds
    pnl = proceeds - pos.entry_cost
    trades.append({
        "code": pos.code,
        "shares": pos.shares,
        "signal_type": pos.signal_type,
        "trade_mode": pos.trade_mode,
        "entry_market_mode": pos.entry_market_mode,
        "profile": pos.profile,
        "signal_date": pos.signal_date,
        "entry_date": pos.entry_date,
        "entry_time": pos.entry_time,
        "entry_price": pos.entry_price,
        "entry_cost": pos.entry_cost,
        "candidate_score": pos.candidate_score,
        "risk_multiplier": pos.risk_multiplier,
        "high_position_unstable": pos.high_position_unstable,
        "initial_stop": pos.initial_stop,
        "final_trailing_stop": pos.trailing_stop,
        "exit_date": date,
        "exit_time": pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30),
        "exit_price": px,
        "exit_reason": reason,
        "hold_days": pos.hold_days,
        "pnl": pnl,
        "net_return_pct": pnl / pos.entry_cost if pos.entry_cost > 0 else np.nan,
        "gross_return_pct": px / pos.entry_price - 1,
        "mfe_pct": pos.highest_high / pos.entry_price - 1,
        "mae_pct": pos.lowest_low / pos.entry_price - 1,
    })
    fills.append({
        "timestamp": pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30),
        "date": date,
        "code": pos.code,
        "side": "SELL",
        "shares": pos.shares,
        "price": px,
        "cash_delta": proceeds,
        "reason": reason,
    })
    return cash


def run_v4_simulation(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    entries: pd.DataFrame,
    market: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    *,
    enable_range_impulse_exit: bool = True,
    enable_high_position_resize: bool = True,
    enable_right_pullback: bool = True,
    allow_technical_raw_wave3: bool = False,
    keep_fills: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily = daily.sort_values(["date", "code"])
    exec_px = exec_px.sort_values(["date", "code"])
    all_dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    sim_dates = [d for d in all_dates if (start_date is None or d >= pd.Timestamp(start_date)) and (end_date is None or d <= pd.Timestamp(end_date))]
    date_index = {d: i for i, d in enumerate(all_dates)}
    by_day = {pd.Timestamp(d): x.set_index("code") for d, x in daily.groupby("date")}
    ex_day = {pd.Timestamp(d): x.set_index("code") for d, x in exec_px.groupby("date")}
    market_by_date = market.set_index("date")

    e = entries[entries["entry_confirmed"] == True].copy() if len(entries) else pd.DataFrame()
    if len(e):
        if not enable_right_pullback:
            e = e[e["signal_type"] != "right_pullback"]
        regular = e[e["signal_type"] != "wave3_probe_raw"]
        raw = e[e["signal_type"] == "wave3_probe_raw"].copy()
        if len(raw):
            if allow_technical_raw_wave3:
                raw = raw[raw["technical_gate_pass"] == True]
            else:
                raw = raw[raw["production_gate_pass"] == True]
        e = pd.concat([regular, raw], ignore_index=True, sort=False)
        entries_by_date = {pd.Timestamp(d): x.copy() for d, x in e.groupby("entry_date")}
    else:
        entries_by_date = {}

    cash = START_CASH
    positions: dict[str, V4Position] = {}
    cooldown_until: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    exposure_breaches: list[dict[str, Any]] = []
    r = v3s.CFG["risk"]
    n = V4CFG["note_rules"]

    for date in sim_dates:
        day = by_day[date]
        ex = ex_day.get(date)
        if ex is None:
            continue
        idx = date_index[date]
        prev_date = all_dates[idx - 1] if idx > 0 else None
        prev = by_day.get(prev_date) if prev_date is not None else None
        applied_cap = float(market_by_date.loc[prev_date]["max_exposure"]) if prev_date is not None and prev_date in market_by_date.index else 0.0
        applied_market_mode = str(market_by_date.loc[prev_date].get("market_mode", "mixed")) if prev_date is not None and prev_date in market_by_date.index else "mixed"

        if prev is not None:
            for code in list(positions.keys()):
                if code not in prev.index or code not in ex.index:
                    continue
                pos = positions[code]
                pr = prev.loc[code]
                er = ex.loc[code]
                reason = None
                if float(pr["close"]) < pos.initial_stop:
                    reason = "support_break"
                elif (
                    enable_range_impulse_exit
                    and pos.trade_mode == "range_support"
                    and pd.notna(pr.get("ret", np.nan))
                    and float(pr["ret"]) >= float(n["big_rise_exit_daily_return"])
                ):
                    reason = "range_big_rise_next_open"
                elif float(pr["close"]) < pos.trailing_stop:
                    reason = "trailing_stop"
                elif pos.hold_days >= MAX_HOLD_DAYS:
                    reason = "time_stop"
                if reason and v3s.tradable_sell(er, float(pr["close"])):
                    cash = sell_v4_position(pos, date, er, reason, cash, trades, fills)
                    if reason == "support_break":
                        cooldown_until[code] = idx + int(v3s.CFG["cooldown_days_after_stop"])
                    del positions[code]

        current_values = {
            code: pos.shares * (float(ex.loc[code]["first_open"]) if code in ex.index else pos.entry_price)
            for code, pos in positions.items()
        }
        equity_open = cash + sum(current_values.values())
        target_value = max(0.0, equity_open * applied_cap)
        tolerance = float(r["exposure_rebalance_tolerance"])
        while positions and sum(current_values.values()) > target_value + equity_open * tolerance:
            candidates: list[tuple[float, str]] = []
            for code, pos in positions.items():
                if prev is None or code not in prev.index or code not in ex.index:
                    continue
                if not v3s.tradable_sell(ex.loc[code], float(prev.loc[code]["close"])):
                    continue
                strength = v3e.position_hold_strength(pos, prev.loc[code])
                candidates.append((strength, code))
            if not candidates:
                exposure_breaches.append({
                    "date": date,
                    "applied_cap": applied_cap,
                    "actual_open_exposure": sum(current_values.values()) / max(equity_open, 1.0),
                    "reason": "cannot_reduce_suspended_or_limit_down",
                })
                break
            _, code = sorted(candidates, key=lambda z: (z[0], z[1]))[0]
            pos = positions[code]
            cash = sell_v4_position(pos, date, ex.loc[code], "exposure_rebalance", cash, trades, fills)
            del positions[code]
            current_values.pop(code, None)
            equity_open = cash + sum(current_values.values())
            target_value = max(0.0, equity_open * applied_cap)

        cands = entries_by_date.get(date, pd.DataFrame()).copy()
        if not cands.empty and applied_cap > 0:
            cands["priority"] = cands["signal_type"].map(V4_PRIORITY).fillna(9)
            cands["rank_score"] = cands["v4_rank_score"].fillna(cands.get("candidate_score", 0))
            cands = cands.sort_values(["rank_score", "priority", "code"], ascending=[False, True, True], kind="mergesort")
            cands = cands.drop_duplicates("code", keep="first")
            for _, sig in cands.iterrows():
                code = str(sig["code"])
                reason = None
                if code in positions:
                    reason = "already_held"
                elif idx <= cooldown_until.get(code, -1):
                    reason = "cooldown_after_stop"
                elif code not in ex.index or code not in day.index:
                    reason = "missing_execution_row"
                elif len(positions) >= MAX_POSITIONS:
                    reason = "max_positions"
                if reason:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": reason})
                    continue

                invested_open = sum(
                    p.shares * (float(ex.loc[c]["first_open"]) if c in ex.index else p.entry_price)
                    for c, p in positions.items()
                )
                equity_pre = cash + invested_open
                entry_cap = applied_cap
                if str(sig.get("trade_mode", "")) == "left_probe":
                    entry_cap = min(entry_cap, float(n["left_probe_total_exposure_cap"]))
                remaining_exposure = max(0.0, equity_pre * entry_cap - invested_open)
                if remaining_exposure < float(r["min_trade_value"]):
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "portfolio_exposure_limit"})
                    continue

                raw_price = float(sig["raw_price"])
                px = raw_price * (1 + SLIPPAGE)
                stop_anchor = float(sig.get("effective_stop_anchor", sig.get("stop_anchor", np.nan)))
                if not enable_high_position_resize and str(sig["signal_type"]) != "wave3_probe_raw":
                    stop_anchor = float(sig.get("stop_anchor", stop_anchor))
                initial_stop = stop_anchor * (1 - float(r["break_tolerance"]))
                risk_multiplier = float(sig.get("risk_multiplier", 1.0))
                if not enable_high_position_resize and str(sig["signal_type"]) != "wave3_probe_raw":
                    risk_multiplier = 1.0
                position_mult = risk_multiplier if bool(sig.get("high_position_unstable", False)) and enable_high_position_resize else 1.0
                shares, cost, stop_distance = v4_risk_sized_shares(
                    equity_pre, cash, remaining_exposure, px, initial_stop, risk_multiplier, position_mult
                )
                if stop_distance <= 0:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "invalid_stop_distance"})
                    continue
                if shares < 100 or cost < float(r["min_trade_value"]) or cost > cash + 1e-6:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "risk_sized_allocation_too_small"})
                    continue

                cash -= cost
                pos = V4Position(
                    code=code,
                    shares=shares,
                    entry_date=date,
                    entry_time=pd.Timestamp(sig["execution_time"]),
                    entry_price=px,
                    entry_cost=cost,
                    stop_anchor=stop_anchor,
                    initial_stop=initial_stop,
                    trailing_stop=initial_stop,
                    signal_type=str(sig["signal_type"]),
                    signal_date=pd.Timestamp(sig["signal_date"]),
                    profile=BASE_PROFILE,
                    candidate_score=float(sig.get("v4_rank_score", sig.get("candidate_score", 0.0))),
                    trade_mode=str(sig.get("trade_mode", "unrouted")),
                    entry_market_mode=str(sig.get("market_mode", applied_market_mode)),
                    risk_multiplier=risk_multiplier,
                    high_position_unstable=bool(sig.get("high_position_unstable", False)),
                    hold_days=0,
                    highest_close=max(px, float(day.loc[code]["close"])),
                    highest_high=max(px, float(sig["entry_day_post_high"])),
                    lowest_low=min(px, float(sig["entry_day_post_low"])),
                )
                positions[code] = pos
                fills.append({
                    "timestamp": pd.Timestamp(sig["execution_time"]),
                    "date": date,
                    "code": code,
                    "side": "BUY",
                    "shares": shares,
                    "price": px,
                    "cash_delta": -cost,
                    "reason": str(sig["signal_type"]),
                    "trade_mode": pos.trade_mode,
                })

        equity = cash
        invested_close = 0.0
        for code, pos in positions.items():
            if code in day.index:
                rr = day.loc[code]
                close = float(rr["close"])
                value = pos.shares * close
                equity += value
                invested_close += value
                pos.highest_close = max(pos.highest_close, close)
                if date > pos.entry_date:
                    pos.highest_high = max(pos.highest_high, float(rr["high"]))
                    pos.lowest_low = min(pos.lowest_low, float(rr["low"]))
                if pos.highest_close / pos.entry_price - 1 >= float(r["trail_activate_gain"]):
                    pos.trailing_stop = max(pos.initial_stop, pos.highest_close * (1 - float(r["trail_drawdown"])))
                pos.hold_days += 1
            else:
                equity += pos.shares * pos.entry_price
                invested_close += pos.shares * pos.entry_price
        next_cap = float(market_by_date.loc[date]["max_exposure"]) if date in market_by_date.index else np.nan
        curve.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "positions": len(positions),
            "actual_exposure": invested_close / max(equity, 1.0),
            "applied_exposure_cap": applied_cap,
            "next_day_exposure_cap": next_cap,
            "market_score": float(market_by_date.loc[date]["market_score"]) if date in market_by_date.index else np.nan,
            "market_mode": str(market_by_date.loc[date].get("market_mode", "mixed")) if date in market_by_date.index else "mixed",
        })

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    fills_df = pd.DataFrame(fills)
    rejects_df = pd.DataFrame(rejects)
    final_date = sim_dates[-1] if sim_dates else None
    open_rows: list[dict[str, Any]] = []
    if final_date is not None:
        final_day = by_day[final_date]
        for code, pos in positions.items():
            mark = float(final_day.loc[code]["close"]) if code in final_day.index else pos.entry_price
            market_value = pos.shares * mark
            open_rows.append({
                "code": code,
                "signal_type": pos.signal_type,
                "trade_mode": pos.trade_mode,
                "entry_market_mode": pos.entry_market_mode,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "entry_time": pos.entry_time,
                "entry_price": pos.entry_price,
                "shares": pos.shares,
                "entry_cost": pos.entry_cost,
                "mark_date": final_date,
                "mark_price": mark,
                "market_value": market_value,
                "unrealized_pnl": market_value - pos.entry_cost,
                "unrealized_return_pct": market_value / pos.entry_cost - 1 if pos.entry_cost > 0 else np.nan,
                "initial_stop": pos.initial_stop,
                "trailing_stop": pos.trailing_stop,
                "risk_multiplier": pos.risk_multiplier,
                "high_position_unstable": pos.high_position_unstable,
                "mfe_pct": pos.highest_high / pos.entry_price - 1,
                "mae_pct": pos.lowest_low / pos.entry_price - 1,
                "hold_days": pos.hold_days,
            })
    open_df = pd.DataFrame(open_rows)
    stats = v3e.compute_stats(curve_df, trades_df, open_df, market, START_CASH)
    stats.update({
        "strategy_version": V4CFG["version"],
        "fixed_policy": V4CFG["fixed_policy"],
        "base_profile": BASE_PROFILE,
        "exposure_breach_days": int(len(exposure_breaches)),
        "rejected_entries": int(len(rejects_df)),
        "enable_range_impulse_exit": bool(enable_range_impulse_exit),
        "enable_high_position_resize": bool(enable_high_position_resize),
        "enable_right_pullback": bool(enable_right_pullback),
        "allow_technical_raw_wave3": bool(allow_technical_raw_wave3),
    })
    if len(rejects_df):
        stats["rejections_by_reason"] = rejects_df.groupby("reason").size().sort_values(ascending=False).to_dict()
    if len(trades_df):
        stats["by_trade_mode"] = trades_df.groupby("trade_mode").agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
            avg_return=("net_return_pct", "mean"),
            total_pnl=("pnl", "sum"),
        ).reset_index().to_dict("records")
        stats["by_entry_market_mode"] = trades_df.groupby("entry_market_mode").agg(
            trades=("pnl", "size"),
            avg_return=("net_return_pct", "mean"),
            total_pnl=("pnl", "sum"),
        ).reset_index().to_dict("records")
    return curve_df, trades_df, open_df, fills_df if keep_fills else pd.DataFrame(), stats
