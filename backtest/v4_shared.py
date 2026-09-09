from __future__ import annotations

import copy
import json
import math
import os
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

import three_five_wave_backtest as v2
import v3_shared as v3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "backtest", "results_v4")
V4_CONFIG_PATH = os.path.join(ROOT, "backtest", "v4_config.json")
FUNDAMENTAL_HISTORY_PATH = os.path.join(ROOT, "backtest", "fundamental_history.csv")
os.makedirs(OUT, exist_ok=True)

with open(V4_CONFIG_PATH, "r", encoding="utf-8") as f:
    V4CFG: dict[str, Any] = json.load(f)

BASE_PROFILE = str(V4CFG["base_profile"])
V4_SIGNAL_TYPES = [
    "wave5_exhaust_reversal",
    "pingbu_qingyun",
    "right_pullback",
    "wave3_probe_raw",
]
V4_PRIORITY = {
    "wave5_exhaust_reversal": 0,
    "pingbu_qingyun": 1,
    "right_pullback": 2,
    "wave3_probe_raw": 3,
}


def parse_hhmm(value: str) -> time:
    hh, mm = str(value).split(":")
    return time(int(hh), int(mm))


def _compound_window(x: np.ndarray) -> float:
    return float(np.prod(1.0 + np.nan_to_num(x, nan=0.0)) - 1.0)


def add_market_modes(market: pd.DataFrame) -> pd.DataFrame:
    """Causal market-mode labels: each row uses that date and earlier observations only."""
    m = market.sort_values("date").copy()
    m["benchmark_ret5"] = m["benchmark_ret"].rolling(5, min_periods=3).apply(_compound_window, raw=True)
    m["benchmark_ret10"] = m["benchmark_ret"].rolling(10, min_periods=5).apply(_compound_window, raw=True)
    c = V4CFG["market_mode"]

    def classify(r: pd.Series) -> str:
        score = float(r.get("market_score", 50.0))
        ret5 = float(r.get("benchmark_ret5", 0.0)) if pd.notna(r.get("benchmark_ret5", np.nan)) else 0.0
        ret10 = float(r.get("benchmark_ret10", 0.0)) if pd.notna(r.get("benchmark_ret10", np.nan)) else 0.0
        above = float(r.get("pct_above_ma5", 0.5)) if pd.notna(r.get("pct_above_ma5", np.nan)) else 0.5
        if score < float(c["risk_off_score_max"]) or ret5 <= float(c["risk_off_ret5_max"]):
            return "risk_off"
        if (
            score >= float(c["trend_up_score_min"])
            and ret5 >= float(c["trend_up_ret5_min"])
            and above >= float(c["trend_up_pct_above_ma5_min"])
        ):
            return "trend_up"
        if (
            abs(ret10) <= float(c["range_abs_ret10_max"])
            and float(c["range_score_min"]) <= score <= float(c["range_score_max"])
            and float(c["range_pct_above_ma5_min"]) <= above <= float(c["range_pct_above_ma5_max"])
        ):
            return "range"
        return "mixed"

    m["market_mode"] = m.apply(classify, axis=1)
    return m


def _days_since_window_high(s: pd.Series, window: int = 20) -> pd.Series:
    return s.rolling(window, min_periods=5).apply(
        lambda x: float(len(x) - 1 - int(np.argmax(x))), raw=True
    )


def _bigbar_support_features(g: pd.DataFrame) -> tuple[list[float], list[float]]:
    g = g.reset_index(drop=True)
    count20: list[float] = []
    second_support: list[float] = []
    anchors: list[tuple[int, float]] = []
    big_min = float(v2.DEFAULT_CONFIG["anchor"]["big_bar_min_return"])
    close_pos_min = float(v2.DEFAULT_CONFIG["anchor"]["big_bar_close_pos"])
    gap_tol = float(v2.DEFAULT_CONFIG["anchor"]["gap_vs_prev_high"])
    for i in range(len(g)):
        row = g.iloc[i]
        if i > 0 and pd.notna(row.get("ret", np.nan)) and float(row["ret"]) >= big_min and float(row.get("close_pos", 0.0)) >= close_pos_min:
            prev = g.iloc[i - 1]
            gap = float(row["open"]) > float(prev["high"]) * (1 + gap_tol)
            start = float(prev["close"]) if gap else float(row["open"])
            anchors.append((i, start))
        anchors = [(j, a) for j, a in anchors if i - j <= 20]
        count20.append(float(len(anchors)))
        if len(anchors) >= 2:
            last_two = anchors[-2:]
            second_support.append(float(min(a for _, a in last_two)))
        else:
            second_support.append(np.nan)
    return count20, second_support


def add_v4_stock_features(daily: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for code, g0 in daily.groupby("code", sort=False):
        g = g0.sort_values("date").copy().reset_index(drop=True)
        prev_close = g["close"].shift(1)
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - prev_close).abs(),
            (g["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        g["atr10_pct"] = tr.rolling(10, min_periods=5).mean() / g["close"].replace(0, np.nan)
        g["low60"] = g["low"].rolling(60, min_periods=20).min()
        g["height60"] = g["close"] / g["low60"].replace(0, np.nan) - 1
        g["down_days5"] = (g["ret"].fillna(0) <= 0).rolling(5, min_periods=3).sum()
        g["days_since_high20"] = _days_since_window_high(g["close"], 20)
        g["range10_pct"] = g["high"].rolling(10, min_periods=5).max() / g["low"].rolling(10, min_periods=5).min().replace(0, np.nan) - 1
        counts, supports = _bigbar_support_features(g)
        g["bigbar_count20_v4"] = counts
        g["second_bigbar_support20"] = supports
        g["code"] = str(code)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else daily.copy()
    big_component = (out["bigbar_count20_v4"].fillna(0) / 3.0).clip(0, 1)
    out["technical_core_proxy"] = (
        0.40 * out["rs20_rank"].fillna(0.5)
        + 0.25 * out["liquidity_rank"].fillna(0.0)
        + 0.20 * out["rs5_rank"].fillna(0.5)
        + 0.15 * big_component
    ).clip(0, 1)
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def add_point_in_time_fundamentals(daily: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    """Only exact point-in-time records are accepted; no current snapshot is backfilled."""
    d = daily.copy()
    cols = ["market_cap_yuan", "fundamental_score", "has_real_order", "industry_leader_score"]
    if not os.path.exists(FUNDAMENTAL_HISTORY_PATH):
        for c in cols:
            d[c] = np.nan
        return d, False, "no_point_in_time_fundamental_history"
    f = pd.read_csv(FUNDAMENTAL_HISTORY_PATH, dtype={"code": str})
    if not {"date", "code"}.issubset(f.columns):
        raise ValueError("fundamental_history.csv must contain at least date,code")
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["code"] = v2.normalize_code(f["code"])
    keep = ["date", "code"] + [c for c in cols if c in f.columns]
    f = f[keep].drop_duplicates(["date", "code"])
    d = d.merge(f, on=["date", "code"], how="left")
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    available = bool(d[cols].notna().any(axis=None))
    return d, available, "point_in_time_exact_date" if available else "point_in_time_file_has_no_usable_values"


def _fundamental_left_probe_pass(r: pd.Series) -> bool:
    order = r.get("has_real_order", np.nan)
    fscore = r.get("fundamental_score", np.nan)
    leader = r.get("industry_leader_score", np.nan)
    order_pass = pd.notna(order) and bool(order)
    score_pass = pd.notna(fscore) and pd.notna(leader) and float(fscore) >= 0.70 and float(leader) >= 0.60
    return bool(order_pass or score_pass)


def annotate_regular_signals(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    market: pd.DataFrame,
    fundamental_available: bool,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    cols = [
        "code", "date", "ret", "rs5_rank", "rs20_rank", "liquidity_rank", "atr10_pct", "height60",
        "down_days5", "days_since_high20", "technical_core_proxy", "second_bigbar_support20",
        "bigbar_count20_v4", "market_cap_yuan", "fundamental_score", "has_real_order", "industry_leader_score",
    ]
    s = signals.merge(daily[cols], on=["code", "date"], how="left")
    s = s.merge(market[["date", "market_mode", "benchmark_ret5", "benchmark_ret10"]], on="date", how="left")
    n = V4CFG["note_rules"]
    adjust = (
        s["days_since_high20"].between(float(n["pullback_min_days_from_20d_high"]), float(n["pullback_max_days_from_20d_high"]), inclusive="both")
        & (s["down_days5"].fillna(0) >= float(n["pullback_down_days_in_last5_min"]))
    )
    core = s["technical_core_proxy"].fillna(0) >= float(n["technical_core_proxy_min"])
    s["adjustment_pass"] = adjust
    s["core_proxy_pass"] = core
    s["high_position_unstable"] = (
        (s["height60"].fillna(0) >= float(n["high_position_height60_min"]))
        & (s["atr10_pct"].fillna(0) >= float(n["high_volatility_atr10_pct_min"]))
    )
    s["risk_multiplier"] = np.where(s["high_position_unstable"], float(n["high_position_risk_multiplier"]), 1.0)
    s["effective_stop_anchor"] = s["stop_anchor"].astype(float)
    use_lower = s["high_position_unstable"] & s["second_bigbar_support20"].notna()
    s.loc[use_lower, "effective_stop_anchor"] = np.minimum(
        s.loc[use_lower, "stop_anchor"].astype(float),
        s.loc[use_lower, "second_bigbar_support20"].astype(float),
    )
    s["trade_mode"] = "unrouted"
    ping = s["signal_type"] == "pingbu_qingyun"
    right = s["signal_type"] == "right_pullback"
    wave5 = s["signal_type"] == "wave5_exhaust_reversal"
    s.loc[ping & (s["market_mode"] == "trend_up"), "trade_mode"] = "trend_support"
    s.loc[ping & s["market_mode"].isin(["range", "mixed"]), "trade_mode"] = "range_support"
    s.loc[right & s["market_mode"].isin(["range", "mixed"]), "trade_mode"] = "range_support"
    s.loc[wave5, "trade_mode"] = "exhaustion_reversal"
    s["note_gate_pass"] = False
    s.loc[ping & adjust & core & (s["market_mode"] != "risk_off"), "note_gate_pass"] = True
    s.loc[right & adjust & core & s["market_mode"].isin(["range", "mixed"]), "note_gate_pass"] = True
    s.loc[wave5 & (s["technical_core_proxy"].fillna(0) >= float(n["wave5_core_proxy_min"])), "note_gate_pass"] = True
    s["fundamental_left_probe_pass"] = s.apply(_fundamental_left_probe_pass, axis=1) if fundamental_available else False
    s["v4_rank_score"] = s["candidate_score"].astype(float) + (s["technical_core_proxy"].fillna(0.5) - 0.5) * 10.0
    s.loc[s["high_position_unstable"], "v4_rank_score"] -= 3.0
    s["v4_rank_score"] = s["v4_rank_score"].clip(0, 100)
    return s


def detect_raw_wave3_probes(
    daily: pd.DataFrame,
    market: pd.DataFrame,
    fundamental_available: bool,
) -> pd.DataFrame:
    """Research-only raw wave-3 divergence probes; production requires point-in-time fundamentals."""
    original_cfg = copy.deepcopy(v2.CFG)
    rows: list[dict[str, Any]] = []
    market_lookup = market.set_index("date")
    n = V4CFG["note_rules"]
    try:
        v2.CFG = v3.profile_v2_cfg(BASE_PROFILE)
        for code, g0 in daily.groupby("code", sort=False):
            g = g0.sort_values("date").reset_index(drop=True).copy()
            for i in range(len(g)):
                div = v2.wave_divergence_candidate(g, i)
                if div is None or str(div.get("kind")) != "wave3":
                    continue
                r = g.iloc[i]
                date = pd.Timestamp(r["date"])
                if date not in market_lookup.index:
                    continue
                m = market_lookup.loc[date]
                tech_pass = bool(
                    float(m.get("market_score", 100.0)) <= float(n["left_probe_market_score_max"])
                    and float(r.get("rs20_rank", 0.0) if pd.notna(r.get("rs20_rank", np.nan)) else 0.0) >= float(n["left_probe_rs20_rank_min"])
                    and float(r.get("technical_core_proxy", 0.0) if pd.notna(r.get("technical_core_proxy", np.nan)) else 0.0) >= float(n["left_probe_core_proxy_min"])
                )
                fund_pass = _fundamental_left_probe_pass(r) if fundamental_available else False
                rows.append({
                    "code": str(code),
                    "date": date,
                    "signal_date": date,
                    "signal_type": "wave3_probe_raw",
                    "profile": BASE_PROFILE,
                    "stop_anchor": float(div["divergence_low"]),
                    "effective_stop_anchor": float(div["divergence_low"]),
                    "candidate_score": float(np.clip(65.0 + 20.0 * float(r.get("technical_core_proxy", 0.0)), 0, 100)),
                    "v4_rank_score": float(np.clip(65.0 + 20.0 * float(r.get("technical_core_proxy", 0.0)), 0, 100)),
                    "trade_mode": "left_probe",
                    "risk_multiplier": float(n["left_probe_risk_multiplier"]),
                    "technical_gate_pass": tech_pass,
                    "fundamental_left_probe_pass": fund_pass,
                    "production_gate_pass": bool(tech_pass and fund_pass and fundamental_available),
                    "market_mode": str(m.get("market_mode", "mixed")),
                    "market_score": float(m.get("market_score", np.nan)),
                    "rs20_rank": float(r.get("rs20_rank", np.nan)),
                    "technical_core_proxy": float(r.get("technical_core_proxy", np.nan)),
                    "divergence_low": float(div["divergence_low"]),
                    "divergence_strength": float(div["divergence_strength"]),
                })
    finally:
        v2.CFG = original_cfg
    return pd.DataFrame(rows)


def causal_find_left_probe_entry(
    x: pd.DataFrame,
    prev_close: float,
    status_row: pd.Series,
) -> tuple[dict[str, Any] | None, str]:
    """Wait for a sharp next-day drop, then stabilization; execute only next minute open."""
    if x.empty:
        return None, "no_intraday_data"
    if v3.is_bad_status(status_row):
        return None, "bad_status"
    n = V4CFG["note_rules"]
    start_t = parse_hhmm(n["left_probe_not_before"])
    end_t = parse_hhmm(n["left_probe_not_after"])
    x = x.sort_values("datetime").reset_index(drop=True).copy()
    if len(x) < 2 or prev_close <= 0:
        return None, "insufficient_intraday_bars"
    running_low = math.inf
    running_low_idx = -1
    sharp_seen = False
    for i in range(len(x) - 1):
        r = x.iloc[i]
        ts = pd.Timestamp(r["datetime"])
        low = float(r["low"])
        close = float(r["close"])
        if low < running_low:
            running_low = low
            running_low_idx = i
        if running_low / prev_close - 1 <= float(n["left_probe_sharp_drop_from_prev_close"]):
            sharp_seen = True
        if ts.time() < start_t:
            continue
        if ts.time() > end_t:
            break
        if not sharp_seen:
            continue
        rebound = close / running_low - 1 if running_low > 0 else 0.0
        bars_since_low = i - running_low_idx
        if rebound < float(n["left_probe_rebound_from_low_min"]):
            continue
        if bars_since_low < int(n["left_probe_stable_bars_after_low"]):
            continue
        nxt = x.iloc[i + 1]
        raw_price = float(nxt["open"])
        if not np.isfinite(raw_price) or raw_price <= 0:
            continue
        rest = x.iloc[i + 1:]
        return {
            "confirmation_time": ts,
            "execution_time": pd.Timestamp(nxt["datetime"]),
            "raw_price": raw_price,
            "entry_day_post_high": float(rest["high"].max()),
            "entry_day_post_low": float(rest["low"].min()),
            "entry_stop_anchor": float(running_low),
            "intraday_rebound": float(rebound),
            "bars_since_low": int(bars_since_low),
        }, "left_probe_sharp_drop_then_stable"
    return None, "no_left_probe_confirmation"


def precompute_v4_entries(
    daily: pd.DataFrame,
    exec_px: pd.DataFrame,
    minute_map: dict[pd.Timestamp, str],
    regular_signals: pd.DataFrame,
    raw_wave3: pd.DataFrame,
) -> pd.DataFrame:
    dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    by_day = {pd.Timestamp(d): x.set_index("code") for d, x in daily.groupby("date")}
    ex_day = {pd.Timestamp(d): x.set_index("code") for d, x in exec_px.groupby("date")}
    parts: list[pd.DataFrame] = []
    if not regular_signals.empty:
        reg = regular_signals[(regular_signals["absolute_gate_pass"] == True) & (regular_signals["note_gate_pass"] == True)].copy()
        reg["production_gate_pass"] = True
        parts.append(reg)
    if not raw_wave3.empty:
        raw = raw_wave3[raw_wave3["technical_gate_pass"] == True].copy()
        parts.append(raw)
    if not parts:
        return pd.DataFrame()
    cands = pd.concat(parts, ignore_index=True, sort=False)
    cands["entry_date"] = cands["date"].map(next_date)
    cands = cands.dropna(subset=["entry_date"])
    rows: list[dict[str, Any]] = []
    for entry_date, group in cands.groupby("entry_date", sort=True):
        entry_date = pd.Timestamp(entry_date)
        if entry_date not in ex_day or entry_date not in minute_map:
            continue
        signal_dates = sorted(pd.Timestamp(d) for d in group["date"].unique())
        codes = sorted(set(group["code"].astype(str)))
        intraday = v2.load_intraday_for_codes(minute_map[entry_date], codes)
        groups = {c: x for c, x in intraday.groupby("code")} if not intraday.empty else {}
        ex = ex_day[entry_date]
        for _, sig in group.iterrows():
            code = str(sig["code"])
            signal_date = pd.Timestamp(sig["date"])
            base = sig.to_dict()
            base["signal_date"] = signal_date
            base["entry_date"] = entry_date
            if code not in ex.index or signal_date not in by_day or code not in by_day[signal_date].index:
                base.update({"entry_confirmed": False, "entry_reject_reason": "missing_execution_row"})
                rows.append(base)
                continue
            prev_close = float(by_day[signal_date].loc[code]["close"])
            if str(sig["signal_type"]) == "wave3_probe_raw":
                confirm, reason = causal_find_left_probe_entry(groups.get(code, pd.DataFrame()), prev_close, ex.loc[code])
                if confirm is not None:
                    base["effective_stop_anchor"] = float(confirm["entry_stop_anchor"])
            else:
                confirm, reason = v3.causal_find_intraday_entry(
                    groups.get(code, pd.DataFrame()), sig, prev_close, ex.loc[code], BASE_PROFILE
                )
            if confirm is None:
                base.update({"entry_confirmed": False, "entry_reject_reason": reason})
            else:
                base.update({"entry_confirmed": True, "entry_reject_reason": "", **confirm})
            rows.append(base)
    return pd.DataFrame(rows)
