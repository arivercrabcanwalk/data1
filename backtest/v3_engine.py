from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

import three_five_wave_backtest as v2
from v3_shared import (
    CFG, START_CASH, COMMISSION, SELL_STAMP, SLIPPAGE, MAX_POSITIONS, MAX_HOLD_DAYS,
    SIGNAL_PRIORITY, SIGNAL_TYPES, tradable_sell, risk_sized_shares,
)

@dataclass
class Position:
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
    hold_days: int = 0
    highest_close: float = 0.0
    highest_high: float = 0.0
    lowest_low: float = math.inf


def period_mask(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None, col: str = "date") -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= pd.to_datetime(df[col]) >= pd.Timestamp(start)
    if end is not None:
        mask &= pd.to_datetime(df[col]) <= pd.Timestamp(end)
    return mask


def position_hold_strength(pos: Position, prev_row: pd.Series) -> float:
    rs20 = float(prev_row.get("rs20_rank", 0.5)) if pd.notna(prev_row.get("rs20_rank", np.nan)) else 0.5
    rs5 = float(prev_row.get("rs5_rank", 0.5)) if pd.notna(prev_row.get("rs5_rank", np.nan)) else 0.5
    trend = 1.0 if float(prev_row["close"]) >= float(prev_row.get("ma5", prev_row["close"])) else 0.0
    pnl = float(prev_row["close"] / pos.entry_price - 1)
    pnl_component = float(np.clip((pnl + 0.10) / 0.30, 0.0, 1.0))
    return 0.45 * rs20 + 0.25 * rs5 + 0.20 * trend + 0.10 * pnl_component


def sell_position(
    pos: Position,
    date: pd.Timestamp,
    er: pd.Series,
    reason: str,
    cash: float,
    trades: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> tuple[float, bool]:
    raw = float(er["first_open"])
    px = raw * (1 - SLIPPAGE)
    proceeds = pos.shares * px * (1 - COMMISSION - SELL_STAMP)
    cash += proceeds
    pnl = proceeds - pos.entry_cost
    trades.append({
        "code": pos.code,
        "shares": pos.shares,
        "signal_type": pos.signal_type,
        "profile": pos.profile,
        "signal_date": pos.signal_date,
        "entry_date": pos.entry_date,
        "entry_time": pos.entry_time,
        "entry_price": pos.entry_price,
        "entry_cost": pos.entry_cost,
        "candidate_score": pos.candidate_score,
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
    return cash, True


def compute_stats(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    market: pd.DataFrame,
    start_cash: float,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if curve.empty:
        return {
            "start_cash": start_cash, "final_equity": start_cash, "total_return": 0.0,
            "max_drawdown": 0.0, "trades": 0, "profit_factor": None,
        }
    curve = curve.copy().sort_values("date")
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1
    total_return = float(curve["equity"].iloc[-1] / start_cash - 1)
    max_dd = float(curve["drawdown"].min())
    realized = float(trades["pnl"].sum()) if len(trades) else 0.0
    unrealized = float(open_positions["unrealized_pnl"].sum()) if len(open_positions) else 0.0
    wins = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if len(trades) else 0.0
    losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum()) if len(trades) else 0.0
    pf = wins / losses if losses > 0 else (math.inf if wins > 0 else np.nan)
    daily_ret = curve["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std(ddof=1)) if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    sortino = float(np.sqrt(252) * daily_ret.mean() / downside.std(ddof=1)) if len(downside) > 1 and downside.std(ddof=1) > 0 else np.nan
    n_days = max(1, len(curve))
    ann_ret = float((1 + total_return) ** (252 / n_days) - 1) if 1 + total_return > 0 else -1.0
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan
    m = market[(market["date"] >= curve["date"].min()) & (market["date"] <= curve["date"].max())].copy()
    bench_total = float((1 + m["benchmark_ret"].fillna(0)).prod() - 1) if len(m) else np.nan
    merged = curve[["date", "equity"]].merge(m[["date", "benchmark_ret"]], on="date", how="left")
    sr = merged["equity"].pct_change()
    br = merged["benchmark_ret"].fillna(0)
    valid = pd.DataFrame({"s": sr, "b": br}).dropna()
    beta = float(valid["s"].cov(valid["b"]) / valid["b"].var()) if len(valid) > 2 and valid["b"].var() > 0 else np.nan
    alpha = float((valid["s"] - beta * valid["b"]).mean() * 252) if len(valid) > 2 and np.isfinite(beta) else np.nan
    stats.update({
        "start_cash": float(start_cash),
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": total_return,
        "multiple": 1 + total_return,
        "max_drawdown": max_dd,
        "annualized_return": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "benchmark_total_return": bench_total,
        "excess_return": total_return - bench_total if np.isfinite(bench_total) else np.nan,
        "beta_vs_equal_weight_all_a": beta,
        "alpha_annualized_vs_equal_weight_all_a": alpha,
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else None,
        "avg_trade_return": float(trades["net_return_pct"].mean()) if len(trades) else None,
        "median_trade_return": float(trades["net_return_pct"].median()) if len(trades) else None,
        "profit_factor": float(pf) if np.isfinite(pf) else ("inf" if pf == math.inf else None),
        "realized_pnl": realized,
        "realized_return_on_start_cash": realized / start_cash,
        "unrealized_pnl": unrealized,
        "unrealized_return_on_start_cash": unrealized / start_cash,
        "open_positions": int(len(open_positions)),
    })
    if len(trades):
        stats["by_signal"] = trades.groupby("signal_type").agg(
            trades=("pnl", "size"),
            win_rate=("pnl", lambda x: float((x > 0).mean())),
            avg_return=("net_return_pct", "mean"),
            total_pnl=("pnl", "sum"),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        ).reset_index().to_dict("records")
        stats["by_exit"] = trades.groupby("exit_reason").agg(
            trades=("pnl", "size"), avg_return=("net_return_pct", "mean"), total_pnl=("pnl", "sum")
        ).reset_index().to_dict("records")
        fast = trades[trades["hold_days"] <= 3]
        stats["hold_3d_or_less"] = {
            "trades": int(len(fast)),
            "win_rate": float((fast["pnl"] > 0).mean()) if len(fast) else None,
            "total_pnl": float(fast["pnl"].sum()) if len(fast) else 0.0,
        }
    return stats


def run_simulation(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    entries: pd.DataFrame,
    market: pd.DataFrame,
    profile_name: str,
    allowed_signals: Iterable[str],
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    keep_fills: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    allowed = set(allowed_signals)
    daily = daily.sort_values(["date", "code"])
    exec_px = exec_px.sort_values(["date", "code"])
    all_dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    sim_dates = [d for d in all_dates if (start_date is None or d >= pd.Timestamp(start_date)) and (end_date is None or d <= pd.Timestamp(end_date))]
    date_index = {d: i for i, d in enumerate(all_dates)}
    by_day = {pd.Timestamp(d): x.set_index("code") for d, x in daily.groupby("date")}
    ex_day = {pd.Timestamp(d): x.set_index("code") for d, x in exec_px.groupby("date")}
    market_by_date = market.set_index("date")
    if entries.empty:
        entries_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    else:
        e = entries[(entries["profile"] == profile_name) & (entries["signal_type"].isin(allowed)) & (entries["entry_confirmed"] == True)].copy()
        entries_by_date = {pd.Timestamp(d): x.copy() for d, x in e.groupby("entry_date")}
    cash = START_CASH
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    exposure_breaches: list[dict[str, Any]] = []
    r_cfg = CFG["risk"]
    for date in sim_dates:
        day = by_day[date]
        ex = ex_day.get(date)
        if ex is None:
            continue
        idx = date_index[date]
        prev_date = all_dates[idx - 1] if idx > 0 else None
        prev = by_day.get(prev_date) if prev_date is not None else None
        applied_cap = float(market_by_date.loc[prev_date]["max_exposure"]) if prev_date is not None and prev_date in market_by_date.index else 0.0
        if prev is not None:
            for code in list(positions.keys()):
                if code not in prev.index or code not in ex.index:
                    continue
                pos = positions[code]
                pr = prev.loc[code]
                er = ex.loc[code]
                reason = None
                if float(pr["close"]) < pos.initial_stop:
                    reason = "anchor_break"
                elif float(pr["close"]) < pos.trailing_stop:
                    reason = "trailing_stop"
                elif pos.hold_days >= MAX_HOLD_DAYS:
                    reason = "time_stop"
                if reason and tradable_sell(er, float(pr["close"])):
                    cash, _ = sell_position(pos, date, er, reason, cash, trades, fills)
                    if reason == "anchor_break":
                        cooldown_until[code] = idx + int(CFG["cooldown_days_after_stop"])
                    del positions[code]
        current_values = {
            code: pos.shares * (float(ex.loc[code]["first_open"]) if code in ex.index else pos.entry_price)
            for code, pos in positions.items()
        }
        equity_open = cash + sum(current_values.values())
        target_value = max(0.0, equity_open * applied_cap)
        tolerance = float(r_cfg["exposure_rebalance_tolerance"])
        while positions and sum(current_values.values()) > target_value + equity_open * tolerance:
            candidates = []
            for code, pos in positions.items():
                if prev is None or code not in prev.index or code not in ex.index:
                    continue
                er = ex.loc[code]
                if not tradable_sell(er, float(prev.loc[code]["close"])):
                    continue
                strength = position_hold_strength(pos, prev.loc[code])
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
            er = ex.loc[code]
            cash, _ = sell_position(pos, date, er, "exposure_rebalance", cash, trades, fills)
            del positions[code]
            current_values.pop(code, None)
            equity_open = cash + sum(current_values.values())
            target_value = max(0.0, equity_open * applied_cap)
        cands = entries_by_date.get(date, pd.DataFrame()).copy()
        if not cands.empty and applied_cap > 0:
            cands["priority"] = cands["signal_type"].map(SIGNAL_PRIORITY).fillna(9)
            cands = cands.sort_values(["candidate_score", "priority", "code"], ascending=[False, True, True], kind="mergesort")
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
                remaining_exposure = max(0.0, equity_pre * applied_cap - invested_open)
                if remaining_exposure < float(r_cfg["min_trade_value"]):
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "portfolio_exposure_limit"})
                    continue
                raw_price = float(sig["raw_price"])
                px = raw_price * (1 + SLIPPAGE)
                initial_stop = float(sig["stop_anchor"]) * (1 - float(r_cfg["break_tolerance"]))
                shares, cost, stop_distance = risk_sized_shares(
                    equity_pre, cash, remaining_exposure, px, initial_stop
                )
                if stop_distance <= 0:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "invalid_stop_distance"})
                    continue
                if shares < 100 or cost < float(r_cfg["min_trade_value"]) or cost > cash + 1e-6:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "risk_sized_allocation_too_small"})
                    continue
                cash -= cost
                pos = Position(
                    code=code,
                    shares=shares,
                    entry_date=date,
                    entry_time=pd.Timestamp(sig["execution_time"]),
                    entry_price=px,
                    entry_cost=cost,
                    stop_anchor=float(sig["stop_anchor"]),
                    initial_stop=initial_stop,
                    trailing_stop=initial_stop,
                    signal_type=str(sig["signal_type"]),
                    signal_date=pd.Timestamp(sig["signal_date"]),
                    profile=profile_name,
                    candidate_score=float(sig["candidate_score"]),
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
                if pos.highest_close / pos.entry_price - 1 >= float(r_cfg["trail_activate_gain"]):
                    pos.trailing_stop = max(
                        pos.initial_stop,
                        pos.highest_close * (1 - float(r_cfg["trail_drawdown"])),
                    )
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
                "profile": pos.profile,
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
                "mfe_pct": pos.highest_high / pos.entry_price - 1,
                "mae_pct": pos.lowest_low / pos.entry_price - 1,
                "hold_days": pos.hold_days,
            })
    open_df = pd.DataFrame(open_rows)
    stats = compute_stats(curve_df, trades_df, open_df, market, START_CASH)
    stats["profile"] = profile_name
    stats["allowed_signals"] = sorted(allowed)
    stats["exposure_breach_days"] = int(len(exposure_breaches))
    stats["rejected_entries"] = int(len(rejects_df))
    if len(rejects_df):
        stats["rejections_by_reason"] = rejects_df.groupby("reason").size().sort_values(ascending=False).to_dict()
    return curve_df, trades_df, open_df, fills_df if keep_fills else pd.DataFrame(), stats


def replay_intraday_equity(
    daily: pd.DataFrame,
    minute_map: dict[pd.Timestamp, str],
    fills: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    fills = fills.copy().sort_values(["timestamp", "side"], ascending=[True, False])
    fills_by_date = {pd.Timestamp(d): x.copy() for d, x in fills.groupby("date")}
    dates = [d for d in sorted(pd.Timestamp(x) for x in daily["date"].unique()) if start_date <= d <= end_date]
    cash = START_CASH
    holdings: dict[str, float] = {}
    last_price: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    peak = START_CASH
    for date in dates:
        day_fills = fills_by_date.get(date, pd.DataFrame())
        codes = set(holdings)
        if not day_fills.empty:
            codes |= set(day_fills["code"].astype(str))
        if not codes or date not in minute_map:
            equity = cash + sum(holdings[c] * last_price.get(c, 0.0) for c in holdings)
            peak = max(peak, equity)
            rows.append({"datetime": date + pd.Timedelta(hours=15), "equity": equity, "drawdown": equity / peak - 1})
            continue
        x = v2.load_intraday_for_codes(minute_map[date], sorted(codes))
        if x.empty:
            continue
        bars_by_ts = {pd.Timestamp(ts): g for ts, g in x.groupby("datetime")}
        timestamps = sorted(set(bars_by_ts) | set(pd.to_datetime(day_fills["timestamp"])) if not day_fills.empty else set(bars_by_ts))
        fill_groups = {pd.Timestamp(ts): g for ts, g in day_fills.groupby("timestamp")} if not day_fills.empty else {}
        for ts in timestamps:
            if ts in fill_groups:
                fg = fill_groups[ts].copy()
                fg["side_order"] = fg["side"].map({"SELL": 0, "BUY": 1}).fillna(9)
                for _, f in fg.sort_values("side_order").iterrows():
                    code = str(f["code"])
                    shares = float(f["shares"])
                    cash += float(f["cash_delta"])
                    if f["side"] == "BUY":
                        holdings[code] = holdings.get(code, 0.0) + shares
                        last_price[code] = float(f["price"])
                    else:
                        holdings[code] = max(0.0, holdings.get(code, 0.0) - shares)
                        last_price[code] = float(f["price"])
                        if holdings[code] <= 1e-9:
                            holdings.pop(code, None)
            if ts in bars_by_ts:
                for _, r in bars_by_ts[ts].iterrows():
                    last_price[str(r["code"])] = float(r["close"])
            equity = cash + sum(shares * last_price.get(code, 0.0) for code, shares in holdings.items())
            peak = max(peak, equity)
            rows.append({
                "datetime": ts,
                "equity": equity,
                "cash": cash,
                "positions": len(holdings),
                "drawdown": equity / peak - 1,
            })
    return pd.DataFrame(rows)
