from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from datetime import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

import three_five_wave_backtest as v2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "backtest", "results_v3")
V3_CONFIG_PATH = os.path.join(ROOT, "backtest", "v3_config.json")
PROFILES_PATH = os.path.join(ROOT, "backtest", "v3_profiles.json")
SECTOR_HISTORY_PATH = os.path.join(ROOT, "backtest", "sector_map_history.csv")
STATIC_SECTOR_PATH = os.path.join(ROOT, "backtest", "sector_map.csv")
os.makedirs(OUT, exist_ok=True)

SIGNAL_TYPES = [
    "wave3_div_reversal",
    "wave5_exhaust_reversal",
    "pingbu_qingyun",
    "right_pullback",
]
SIGNAL_PRIORITY = {
    "wave5_exhaust_reversal": 0,
    "wave3_div_reversal": 1,
    "pingbu_qingyun": 2,
    "right_pullback": 3,
}


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_json(V3_CONFIG_PATH)
PROFILES = load_json(PROFILES_PATH)["profiles"]
START_CASH = float(CFG["start_cash"])
COMMISSION = float(CFG["commission"])
SELL_STAMP = float(CFG["sell_stamp"])
SLIPPAGE = float(CFG["slippage"])
MAX_POSITIONS = int(CFG["max_positions"])
MAX_HOLD_DAYS = int(CFG["max_hold_days"])


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def profile_v2_cfg(profile_name: str) -> dict[str, Any]:
    base = copy.deepcopy(v2.DEFAULT_CONFIG)
    base["start_cash"] = START_CASH
    base["max_positions"] = MAX_POSITIONS
    base["commission"] = COMMISSION
    base["sell_stamp"] = SELL_STAMP
    base["slippage"] = SLIPPAGE
    base["max_hold_days"] = MAX_HOLD_DAYS
    base["cooldown_days_after_stop"] = int(CFG["cooldown_days_after_stop"])
    base = deep_merge(base, PROFILES[profile_name])
    r = CFG["risk"]
    base["risk"].update({
        "market_score_flat": r["market_score_flat"],
        "market_score_low": r["market_score_low"],
        "market_score_mid": r["market_score_mid"],
        "exposure_flat": r["exposure_flat"],
        "exposure_low": r["exposure_low"],
        "exposure_mid": r["exposure_mid"],
        "exposure_high": r["exposure_high"],
    })
    return base


def parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def pct_decimal(value: float) -> float:
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def compute_market_features_v3(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, d in daily.groupby("date", sort=True):
        valid_ret = d["ret"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "date": pd.Timestamp(date),
            "breadth_up": float((valid_ret > 0).mean()) if len(valid_ret) else 0.5,
            "breadth_3pct": float((valid_ret > 0.03).mean()) if len(valid_ret) else 0.0,
            "median_ret": float(valid_ret.median()) if len(valid_ret) else 0.0,
            "benchmark_ret": float(valid_ret.mean()) if len(valid_ret) else 0.0,
            "pct_above_ma5": float((d["close"] > d["ma5"]).mean()) if d["ma5"].notna().any() else 0.5,
            "total_amount": float(d["amount"].sum()),
        })
    m = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    m["breadth_up_3d"] = m["breadth_up"].rolling(3, min_periods=1).mean()
    m["amount_ma10"] = m["total_amount"].rolling(10, min_periods=3).mean()
    m["amount_ratio"] = m["total_amount"] / m["amount_ma10"].replace(0, np.nan)
    breadth_component = ((m["breadth_up_3d"] - 0.35) / 0.30 * 35).clip(0, 35)
    trend_component = ((m["pct_above_ma5"] - 0.30) / 0.40 * 30).clip(0, 30)
    median_component = ((m["median_ret"] + 0.015) / 0.03 * 20).clip(0, 20)
    amount_component = ((m["amount_ratio"] - 0.75) / 0.50 * 15).clip(0, 15).fillna(7.5)
    m["market_score"] = (breadth_component + trend_component + median_component + amount_component).clip(0, 100)
    r = CFG["risk"]
    m["max_exposure"] = np.select(
        [
            m["market_score"] < float(r["market_score_flat"]),
            m["market_score"] < float(r["market_score_low"]),
            m["market_score"] < float(r["market_score_mid"]),
        ],
        [float(r["exposure_flat"]), float(r["exposure_low"]), float(r["exposure_mid"])],
        default=float(r["exposure_high"]),
    )
    m["benchmark_equity"] = START_CASH * (1 + m["benchmark_ret"].fillna(0)).cumprod()
    return m


def add_point_in_time_sector_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    daily = daily.copy()
    require_pit = bool(CFG["integrity"].get("require_point_in_time_sector_map", True))
    allow_static = bool(CFG["integrity"].get("allow_static_sector_map", False))
    if os.path.exists(SECTOR_HISTORY_PATH):
        s = pd.read_csv(SECTOR_HISTORY_PATH, dtype={"code": str})
        required = {"date", "code", "sector"}
        if not required.issubset(s.columns):
            raise ValueError("sector_map_history.csv must contain date,code,sector")
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        s["code"] = v2.normalize_code(s["code"])
        s = s[["date", "code", "sector"]].dropna().drop_duplicates(["date", "code"])
        daily = daily.merge(s, on=["date", "code"], how="left")
        known = daily.dropna(subset=["sector"]).copy()
        if known.empty:
            daily["sector_rank"] = np.nan
            return daily, False, "point_in_time_map_empty"
        sector_daily = known.groupby(["sector", "date"], as_index=False).agg(
            sector_ret=("ret", "mean"),
            sector_breadth=("ret", lambda x: float((x > 0).mean())),
            sector_amount=("amount", "sum"),
        ).sort_values(["sector", "date"])
        sector_daily["sector_mom5"] = sector_daily.groupby("sector")["sector_ret"].transform(
            lambda x: (1 + x.fillna(0)).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1
        )
        sector_daily["sector_rank"] = sector_daily.groupby("date")["sector_mom5"].rank(pct=True, method="average")
        daily = daily.merge(
            sector_daily[["sector", "date", "sector_rank"]],
            on=["sector", "date"], how="left"
        )
        return daily, True, "point_in_time"
    if os.path.exists(STATIC_SECTOR_PATH) and allow_static and not require_pit:
        s = pd.read_csv(STATIC_SECTOR_PATH, dtype={"code": str})
        if not {"code", "sector"}.issubset(s.columns):
            raise ValueError("sector_map.csv must contain code,sector")
        s["code"] = v2.normalize_code(s["code"])
        daily = daily.merge(s[["code", "sector"]].drop_duplicates("code"), on="code", how="left")
        known = daily.dropna(subset=["sector"]).copy()
        sector_daily = known.groupby(["sector", "date"], as_index=False).agg(sector_ret=("ret", "mean"))
        sector_daily["sector_mom5"] = sector_daily.groupby("sector")["sector_ret"].transform(
            lambda x: (1 + x.fillna(0)).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1
        )
        sector_daily["sector_rank"] = sector_daily.groupby("date")["sector_mom5"].rank(pct=True)
        daily = daily.merge(sector_daily[["sector", "date", "sector_rank"]], on=["sector", "date"], how="left")
        return daily, True, "static_map_explicitly_allowed"
    daily["sector"] = pd.NA
    daily["sector_rank"] = np.nan
    if os.path.exists(STATIC_SECTOR_PATH):
        return daily, False, "static_map_ignored_to_avoid_historical_lookahead"
    return daily, False, "no_point_in_time_sector_map"


def detect_profile_signals(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    sector_available: bool,
    profile_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_cfg = copy.deepcopy(v2.CFG)
    try:
        v2.CFG = profile_v2_cfg(profile_name)
        signals, diag = v2.detect_all_signals(daily, market, sector_available)
    finally:
        v2.CFG = original_cfg
    if not signals.empty:
        signals["profile"] = profile_name
    diag["profile"] = profile_name
    return signals, diag


def detect_all_profiles(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    sector_available: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_signals: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    for profile_name in PROFILES:
        print("detecting profile", profile_name)
        s, d = detect_profile_signals(daily, market, sector_available, profile_name)
        if not s.empty:
            all_signals.append(s)
        all_diag.append(d)
    signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    diag = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
    return signals, diag


def is_bad_status(r: pd.Series) -> bool:
    for c in ["is_st", "is_star_st", "is_new_listing_initial"]:
        if c in r and pd.notna(r[c]) and bool(r[c]):
            return True
    if "is_suspended" in r:
        if pd.isna(r["is_suspended"]):
            return True
        if bool(r["is_suspended"]):
            return True
    return False


def tradable_sell(r: pd.Series, prev_close: float) -> bool:
    if "is_suspended" in r:
        if pd.isna(r["is_suspended"]) or bool(r["is_suspended"]):
            return False
    px = float(r["first_open"])
    if not np.isfinite(px) or px <= 0:
        return False
    lim = r.get("price_limit_down_pct", np.nan)
    if pd.notna(lim) and prev_close > 0:
        limit_down = prev_close * (1 - abs(pct_decimal(lim)))
        if px <= limit_down * 1.001:
            return False
    return True


def causal_find_intraday_entry(
    x: pd.DataFrame,
    signal: pd.Series,
    prev_close: float,
    status_row: pd.Series,
    profile_name: str,
) -> tuple[dict[str, Any] | None, str]:
    """Confirm on completed bar i, execute only at next bar i+1 OPEN."""
    if x.empty:
        return None, "no_intraday_data"
    if is_bad_status(status_row):
        return None, "bad_status"
    pcfg = profile_v2_cfg(profile_name)["intraday"]
    start_t = parse_hhmm(str(pcfg["not_before"]))
    end_t = parse_hhmm(str(pcfg["not_after"]))
    x = x.sort_values("datetime").reset_index(drop=True).copy()
    if len(x) < 2:
        return None, "insufficient_intraday_bars"
    day_open = float(x.iloc[0]["open"])
    if prev_close <= 0 or day_open / prev_close - 1 > float(pcfg["max_open_gap"]):
        return None, "open_gap_too_high"
    lim = status_row.get("price_limit_up_pct", np.nan)
    limit_up = prev_close * (1 + pct_decimal(lim)) if pd.notna(lim) else np.inf
    stop_anchor = float(signal["stop_anchor"])
    running_low = math.inf
    running_low_idx = -1
    cum_pv = 0.0
    cum_vol = 0.0
    lookback = int(pcfg["lookback_bars_for_flow"])
    for i in range(len(x) - 1):
        r = x.iloc[i]
        ts = pd.Timestamp(r["datetime"])
        t = ts.time()
        low = float(r["low"])
        close = float(r["close"])
        vol = max(float(r["volume"]), 0.0)
        if low < running_low:
            running_low = low
            running_low_idx = i
        cum_pv += close * vol
        cum_vol += vol
        vwap = cum_pv / cum_vol if cum_vol > 0 else close
        if running_low < stop_anchor * (1 - float(pcfg["anchor_intraday_break_tolerance"])):
            return None, "intraday_anchor_break"
        if t < start_t:
            continue
        if t > end_t:
            break
        if close >= limit_up * 0.999:
            continue
        if close / prev_close - 1 > float(pcfg["max_chase"]):
            continue
        rebound = close / running_low - 1 if running_low > 0 else 0.0
        bars_since_low = i - running_low_idx
        if rebound < float(pcfg["rebound_from_low_min"]):
            continue
        if bars_since_low < int(pcfg["stable_bars_after_low"]):
            continue
        if close < vwap * (1 - float(pcfg["vwap_tolerance"])):
            continue
        lb = x.iloc[max(0, i - lookback + 1): i + 1]
        up_amt = float(lb.loc[lb["close"] >= lb["open"], "amount"].sum())
        down_amt = float(lb.loc[lb["close"] < lb["open"], "amount"].sum())
        flow_ratio = up_amt / max(down_amt, 1.0)
        if flow_ratio < float(pcfg["up_down_amount_ratio_min"]):
            continue
        nxt = x.iloc[i + 1]
        exec_time = pd.Timestamp(nxt["datetime"])
        raw_price = float(nxt["open"])
        if not np.isfinite(raw_price) or raw_price <= 0:
            continue
        if raw_price >= limit_up * 0.999:
            continue
        if raw_price / prev_close - 1 > float(pcfg["max_chase"]):
            continue
        if raw_price < stop_anchor * (1 - float(pcfg["anchor_intraday_break_tolerance"])):
            return None, "next_bar_open_broke_anchor"
        rest = x.iloc[i + 1:]
        return {
            "confirmation_time": ts,
            "execution_time": exec_time,
            "raw_price": raw_price,
            "intraday_rebound": float(rebound),
            "bars_since_low": int(bars_since_low),
            "flow_ratio": float(flow_ratio),
            "entry_day_post_high": float(rest["high"].max()),
            "entry_day_post_low": float(rest["low"].min()),
        }, "confirmed_next_bar_open"
    return None, "no_intraday_confirmation"


def precompute_entries(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    minute_map: dict[pd.Timestamp, str],
    signals: pd.DataFrame,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    prev_date = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
    ex_day = {pd.Timestamp(d): x.set_index("code") for d, x in exec_px.groupby("date")}
    by_day = {pd.Timestamp(d): x.set_index("code") for d, x in daily.groupby("date")}
    gated = signals[signals["absolute_gate_pass"] == True].copy()
    gated["entry_date"] = gated["date"].map(next_date)
    gated = gated.dropna(subset=["entry_date"])
    rows: list[dict[str, Any]] = []
    for entry_date, cands in gated.groupby("entry_date", sort=True):
        entry_date = pd.Timestamp(entry_date)
        signal_date = prev_date.get(entry_date)
        if signal_date is None or entry_date not in ex_day or entry_date not in minute_map:
            continue
        codes = sorted(set(cands["code"].astype(str)))
        intraday = v2.load_intraday_for_codes(minute_map[entry_date], codes)
        groups = {c: x for c, x in intraday.groupby("code")} if not intraday.empty else {}
        ex = ex_day[entry_date]
        prev = by_day[signal_date]
        for _, sig in cands.iterrows():
            code = str(sig["code"])
            base = sig.to_dict()
            base["signal_date"] = pd.Timestamp(sig["date"])
            base["entry_date"] = entry_date
            if code not in ex.index or code not in prev.index:
                base.update({"entry_confirmed": False, "entry_reject_reason": "missing_execution_row"})
                rows.append(base)
                continue
            confirm, reason = causal_find_intraday_entry(
                groups.get(code, pd.DataFrame()), sig, float(prev.loc[code]["close"]), ex.loc[code], str(sig["profile"])
            )
            if confirm is None:
                base.update({"entry_confirmed": False, "entry_reject_reason": reason})
            else:
                base.update({"entry_confirmed": True, "entry_reject_reason": "", **confirm})
            rows.append(base)
        print("precomputed entries", entry_date.date(), len(cands))
    return pd.DataFrame(rows)


def risk_sized_shares(
    equity: float, cash: float, remaining_exposure: float, entry_price: float, initial_stop: float
) -> tuple[float, float, float]:
    """Return (shares, cost, stop_distance); wider stops mechanically get smaller size."""
    r_cfg = CFG["risk"]
    stop_distance = float(entry_price - initial_stop)
    if stop_distance <= 0 or equity <= 0 or cash <= 0 or remaining_exposure <= 0:
        return 0.0, 0.0, stop_distance
    risk_budget = equity * float(r_cfg["risk_per_trade"])
    risk_shares = math.floor((risk_budget / stop_distance) / 100.0) * 100.0
    position_cap = equity * float(r_cfg["max_position_fraction"])
    alloc_cap = min(cash, remaining_exposure, position_cap)
    cap_shares = math.floor((alloc_cap / (entry_price * (1 + COMMISSION))) / 100.0) * 100.0
    shares = min(risk_shares, cap_shares)
    cost = shares * entry_price * (1 + COMMISSION)
    return shares, cost, stop_distance
