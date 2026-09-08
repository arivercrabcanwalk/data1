from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "backtest", "results")
CONFIG_PATH = os.path.join(ROOT, "backtest", "v2_config.json")
SECTOR_MAP_PATH = os.path.join(ROOT, "backtest", "sector_map.csv")
os.makedirs(OUT, exist_ok=True)

DEFAULT_CONFIG: dict[str, Any] = {
    "version": "v2.0-causal-state-machine",
    "start_cash": 1_000_000.0,
    "max_positions": 3,
    "commission": 0.0003,
    "sell_stamp": 0.0005,
    "slippage": 0.0010,
    "max_hold_days": 12,
    "cooldown_days_after_stop": 5,
    "wave": {
        "min_history": 30,
        "min_gap_days": 5,
        "max_gap_days": 45,
        "rebound_min": 0.055,
        "price_new_low_pct": 0.005,
        "dif_improvement_frac": 0.05,
        "reversal_min_return": 0.03,
        "reversal_close_pos": 0.70,
        "reversal_amount_ratio_min": 0.80,
        "arm_expiry_days": 3,
        "five_wave_area_ratio_max": 0.90
    },
    "anchor": {
        "big_bar_min_return": 0.07,
        "big_bar_close_pos": 0.70,
        "max_age_days": 60,
        "min_age_days": 3,
        "break_tolerance": 0.005,
        "near_support_above": 0.03,
        "near_support_below": 0.005,
        "pullback_daily_ret_min": -0.015,
        "pullback_daily_ret_max": 0.03,
        "pullback_close_pos_min": 0.45,
        "pullback_amount_vs_anchor_max": 0.85,
        "pullback_amount_vs_ma10_max": 1.05,
        "gap_vs_prev_high": 0.002,
        "duplicate_setup_cooldown_days": 5
    },
    "pingbu": {
        "min_age_days": 3,
        "max_age_days": 25,
        "midpoint_break_tolerance": 0.005,
        "near_platform_above": 0.035,
        "near_platform_below": 0.005
    },
    "intraday": {
        "not_before": "09:45",
        "not_after": "14:30",
        "max_open_gap": 0.03,
        "max_chase": 0.03,
        "anchor_intraday_break_tolerance": 0.005,
        "rebound_from_low_min": 0.010,
        "stable_bars_after_low": 10,
        "vwap_tolerance": 0.002,
        "up_down_amount_ratio_min": 0.90,
        "lookback_bars_for_flow": 10
    },
    "scoring": {
        "min_score_no_sector": 70.0,
        "min_score_with_sector": 75.0,
        "sector_rank_min": 0.55,
        "right_side_rs20_rank_min": 0.55,
        "liquidity_rank_min": 0.20
    },
    "risk": {
        "trail_drawdown": 0.08,
        "trail_min_hold_days": 3,
        "market_score_flat": 35.0,
        "market_score_low": 45.0,
        "market_score_mid": 60.0,
        "exposure_flat": 0.0,
        "exposure_low": 0.30,
        "exposure_mid": 0.50,
        "exposure_high": 0.80
    }
}


def load_config() -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        for key, value in user_cfg.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


CFG = load_config()
START_CASH = float(CFG["start_cash"])
MAX_POSITIONS = int(CFG["max_positions"])
COMMISSION = float(CFG["commission"])
SELL_STAMP = float(CFG["sell_stamp"])
SLIPPAGE = float(CFG["slippage"])
MAX_HOLD_DAYS = int(CFG["max_hold_days"])


def pct_decimal(value: float) -> float:
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def normalize_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def load_daily_from_minute() -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, str]]:
    minute_files = sorted(glob.glob(os.path.join(ROOT, "2026-*", "part-*", "date=*", "minute1.parquet")))
    status_files = sorted(glob.glob(os.path.join(ROOT, "2026-*", "part-*", "date=*", "daily_stock_status.parquet")))
    if not minute_files:
        raise RuntimeError("No minute parquet files found")

    bars = []
    exec_rows = []
    minute_map: dict[pd.Timestamp, str] = {}
    for i, fp in enumerate(minute_files, 1):
        x = pd.read_parquet(fp, columns=["code", "datetime", "open", "high", "low", "close", "volume", "amount"])
        x["code"] = normalize_code(x["code"])
        x["datetime"] = pd.to_datetime(x["datetime"])
        x = x.sort_values(["code", "datetime"])
        date = pd.Timestamp(x["datetime"].dt.normalize().iloc[0])
        minute_map[date] = fp
        g = x.groupby("code", sort=False)
        d = g.agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"), amount=("amount", "sum"), first_open=("open", "first"),
            first_close=("close", "first"), last_close=("close", "last")
        ).reset_index()
        d["date"] = date
        bars.append(d[["code", "date", "open", "high", "low", "close", "volume", "amount"]])
        exec_rows.append(d[["code", "date", "first_open", "first_close", "last_close"]])
        if i % 20 == 0:
            print(f"aggregated {i}/{len(minute_files)} minute files")

    daily = pd.concat(bars, ignore_index=True)
    exec_px = pd.concat(exec_rows, ignore_index=True)

    statuses = []
    for fp in status_files:
        s = pd.read_parquet(fp)
        s["code"] = normalize_code(s["code"])
        if "date" not in s.columns:
            date_str = fp.split("date=")[-1].split(os.sep)[0]
            s["date"] = pd.Timestamp(date_str)
        else:
            s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        keep = [c for c in ["code", "date", "is_st", "is_star_st", "is_new_listing_initial", "is_suspended", "price_limit_up_pct", "price_limit_down_pct", "market_board", "security_status"] if c in s.columns]
        statuses.append(s[keep])
    status = pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame(columns=["code", "date"])

    daily["code"] = normalize_code(daily["code"])
    exec_px["code"] = normalize_code(exec_px["code"])
    if not status.empty:
        status["code"] = normalize_code(status["code"])
    exec_px = exec_px.merge(status, on=["code", "date"], how="left")
    return daily.sort_values(["code", "date"]), exec_px.sort_values(["code", "date"]), minute_map


def add_stock_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["prev_close"] = g["close"].shift(1)
    g["ret"] = g["close"] / g["prev_close"] - 1
    g["body_ret"] = g["close"] / g["open"] - 1
    g["dif"] = ema(g["close"], 12) - ema(g["close"], 26)
    g["dea"] = ema(g["dif"], 9)
    g["hist"] = (g["dif"] - g["dea"]) * 2
    g["neg_hist_area5"] = (-g["hist"].clip(upper=0)).rolling(5, min_periods=3).sum()
    g["amt_ma10"] = g["amount"].rolling(10, min_periods=5).mean()
    g["vol_ma10"] = g["volume"].rolling(10, min_periods=5).mean()
    g["ma5"] = g["close"].rolling(5, min_periods=5).mean()
    g["ma10"] = g["close"].rolling(10, min_periods=8).mean()
    g["range"] = (g["high"] - g["low"]).replace(0, np.nan)
    g["close_pos"] = ((g["close"] - g["low"]) / g["range"]).clip(0, 1)
    g["drawdown20"] = g["close"] / g["close"].rolling(20, min_periods=10).max() - 1
    g["ret5"] = g["close"].pct_change(5)
    g["ret20"] = g["close"].pct_change(20)
    g["recent_big_bars20"] = (g["ret"] >= float(CFG["anchor"]["big_bar_min_return"])).rolling(20, min_periods=1).sum()
    return g


def add_all_stock_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for code, g in daily.groupby("code", sort=False):
        out = add_stock_indicators(g)
        out["code"] = str(code)
        parts.append(out)
    daily = pd.concat(parts, ignore_index=True)
    daily["rs5_rank"] = daily.groupby("date")["ret5"].rank(pct=True, method="average")
    daily["rs20_rank"] = daily.groupby("date")["ret20"].rank(pct=True, method="average")
    daily["liquidity_rank"] = daily.groupby("date")["amount"].rank(pct=True, method="average")
    return daily.sort_values(["code", "date"])


def compute_market_features(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, d in daily.groupby("date", sort=True):
        valid_ret = d["ret"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "date": pd.Timestamp(date),
            "breadth_up": float((valid_ret > 0).mean()) if len(valid_ret) else 0.5,
            "breadth_3pct": float((valid_ret > 0.03).mean()) if len(valid_ret) else 0.0,
            "median_ret": float(valid_ret.median()) if len(valid_ret) else 0.0,
            "pct_above_ma5": float((d["close"] > d["ma5"]).mean()) if d["ma5"].notna().any() else 0.5,
            "total_amount": float(d["amount"].sum())
        })
    m = pd.DataFrame(rows).sort_values("date")
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
        [m["market_score"] < float(r["market_score_flat"]), m["market_score"] < float(r["market_score_low"]), m["market_score"] < float(r["market_score_mid"])],
        [float(r["exposure_flat"]), float(r["exposure_low"]), float(r["exposure_mid"])],
        default=float(r["exposure_high"])
    )
    return m


def load_sector_map() -> pd.DataFrame | None:
    if not os.path.exists(SECTOR_MAP_PATH):
        return None
    s = pd.read_csv(SECTOR_MAP_PATH, dtype={"code": str})
    if not {"code", "sector"}.issubset(s.columns):
        raise ValueError("backtest/sector_map.csv must contain columns: code,sector")
    s = s[["code", "sector"]].dropna().drop_duplicates("code")
    s["code"] = normalize_code(s["code"])
    return s


def add_sector_features(daily: pd.DataFrame, sector_map: pd.DataFrame | None) -> tuple[pd.DataFrame, bool]:
    daily = daily.copy()
    if sector_map is None or sector_map.empty:
        daily["sector"] = pd.NA
        daily["sector_rank"] = np.nan
        return daily, False
    daily = daily.merge(sector_map, on="code", how="left")
    known = daily.dropna(subset=["sector"]).copy()
    if known.empty:
        daily["sector_rank"] = np.nan
        return daily, False
    sector_daily = known.groupby(["sector", "date"], as_index=False).agg(
        sector_ret=("ret", "mean"), sector_breadth=("ret", lambda x: float((x > 0).mean())), sector_amount=("amount", "sum")
    ).sort_values(["sector", "date"])
    sector_daily["sector_mom5"] = sector_daily.groupby("sector")["sector_ret"].transform(
        lambda x: (1 + x.fillna(0)).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1
    )
    sector_daily["sector_rank"] = sector_daily.groupby("date")["sector_mom5"].rank(pct=True, method="average")
    daily = daily.merge(sector_daily[["sector", "date", "sector_rank"]], on=["sector", "date"], how="left")
    return daily, True


def compute_bigbar_anchor(row: pd.Series, prev: pd.Series | None) -> dict[str, float] | None:
    a = CFG["anchor"]
    if prev is None or not np.isfinite(row.get("ret", np.nan)):
        return None
    if float(row["ret"]) < float(a["big_bar_min_return"]):
        return None
    if float(row.get("close_pos", 0.0)) < float(a["big_bar_close_pos"]):
        return None
    gap = float(row["open"]) > float(prev["high"]) * (1 + float(a["gap_vs_prev_high"]))
    start = float(prev["close"]) if gap else float(row["open"])
    midpoint = (float(row["open"]) + float(row["close"])) / 2.0
    return {
        "start": start,
        "midpoint": midpoint,
        "gap": float(gap),
        "daily_return": float(row["ret"]),
        "amount": float(row["amount"]),
        "amount_ratio": float(row["amount"] / max(float(row.get("amt_ma10", row["amount"])), 1.0))
    }


def prior_low_candidate(g: pd.DataFrame, i: int, min_gap: int | None = None, max_gap: int | None = None):
    w = CFG["wave"]
    min_gap = int(w["min_gap_days"] if min_gap is None else min_gap)
    max_gap = int(w["max_gap_days"] if max_gap is None else max_gap)
    lo = max(0, i - max_gap)
    hi = i - min_gap
    if hi <= lo:
        return None
    hist = g.iloc[lo:hi]
    if len(hist) < 5:
        return None
    valid = hist["close"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    j = int(valid.idxmin())
    if j >= i - 2:
        return None
    mid = g.iloc[j + 1:i]
    if mid.empty:
        return None
    rebound = float(mid["high"].max() / g.iloc[j]["close"] - 1)
    if rebound < float(w["rebound_min"]):
        return None
    return j, rebound


def wave_divergence_candidate(g: pd.DataFrame, i: int) -> dict[str, Any] | None:
    w = CFG["wave"]
    if i < int(w["min_history"]):
        return None
    row = g.iloc[i]
    if not np.isfinite(row.get("dif", np.nan)) or not np.isfinite(row.get("hist", np.nan)):
        return None
    cand = prior_low_candidate(g, i)
    if cand is None:
        return None
    j, rebound = cand
    p = g.iloc[j]
    price_new_low = float(row["close"]) < float(p["close"]) * (1 - float(w["price_new_low_pct"]))
    dif_div = np.isfinite(p.get("dif", np.nan)) and float(row["dif"]) > float(p["dif"]) + float(w["dif_improvement_frac"]) * max(abs(float(p["dif"])), 0.01)
    hist_div = np.isfinite(p.get("hist", np.nan)) and float(row["hist"]) > float(p["hist"])
    if not (price_new_low and dif_div and hist_div):
        return None
    divergence_strength = (float(row["dif"]) - float(p["dif"])) / max(abs(float(p["dif"])), 0.05)
    kind = "wave3"
    earlier = prior_low_candidate(g.iloc[:j + 1].copy(), j, min_gap=4, max_gap=int(w["max_gap_days"])) if j >= 12 else None
    if earlier is not None:
        k, rebound2 = earlier
        pk = g.iloc[k]
        price_stair = float(row["close"]) < float(p["close"]) < float(pk["close"])
        dif5_better = np.isfinite(pk.get("dif", np.nan)) and float(row["dif"]) > float(p["dif"])
        area_now = float(row.get("neg_hist_area5", np.nan))
        area_j = float(p.get("neg_hist_area5", np.nan))
        area_weaker = np.isfinite(area_now) and np.isfinite(area_j) and area_now <= area_j * float(w["five_wave_area_ratio_max"])
        if price_stair and dif5_better and area_weaker:
            kind = "wave5"
            divergence_strength += 0.50 + float(rebound2)
    return {
        "kind": kind,
        "divergence_i": i,
        "prior_low_i": j,
        "divergence_low": float(row["low"]),
        "divergence_close": float(row["close"]),
        "rebound": float(rebound),
        "divergence_strength": float(divergence_strength),
        "expiry_i": i + int(w["arm_expiry_days"])
    }


def is_reversal_bar(row: pd.Series) -> bool:
    w = CFG["wave"]
    amount_ratio = float(row["amount"] / max(float(row.get("amt_ma10", row["amount"])), 1.0))
    return bool(
        np.isfinite(row.get("ret", np.nan))
        and float(row["ret"]) >= float(w["reversal_min_return"])
        and float(row.get("close_pos", 0.0)) >= float(w["reversal_close_pos"])
        and amount_ratio >= float(w["reversal_amount_ratio_min"])
        and float(row["close"]) > float(row.get("prev_close", row["close"]))
    )


def detect_signals_one(g: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    g = g.sort_values("date").reset_index(drop=True).copy()
    a_cfg = CFG["anchor"]
    p_cfg = CFG["pingbu"]
    signals = []
    anchors = []
    arms: dict[str, dict[str, Any] | None] = {"wave3": None, "wave5": None}
    diag = {
        "wave3_armed": 0, "wave5_armed": 0, "wave3_confirmed": 0, "wave5_confirmed": 0,
        "right_pullback_setups": 0, "pingbu_setups": 0, "anchors_created": 0, "anchors_invalidated": 0
    }
    for i in range(len(g)):
        row = g.iloc[i]
        prev = g.iloc[i - 1] if i > 0 else None
        for anchor in anchors:
            if not anchor["valid"] or i <= anchor["i"]:
                continue
            if float(row["close"]) < float(anchor["start"]) * (1 - float(a_cfg["break_tolerance"])):
                anchor["valid"] = False
                diag["anchors_invalidated"] += 1
        for kind in ["wave5", "wave3"]:
            arm = arms[kind]
            if arm is None:
                continue
            if i > int(arm["expiry_i"]):
                arms[kind] = None
                continue
            if i <= int(arm["divergence_i"]):
                continue
            if float(row["low"]) < float(arm["divergence_low"]) * 0.99:
                continue
            if is_reversal_bar(row):
                rev_anchor = compute_bigbar_anchor(row, prev)
                if rev_anchor is None:
                    gap = prev is not None and float(row["open"]) > float(prev["high"]) * (1 + float(a_cfg["gap_vs_prev_high"]))
                    stop_anchor = float(prev["close"]) if gap and prev is not None else float(row["open"])
                else:
                    stop_anchor = float(rev_anchor["start"])
                structure = (62.0 if kind == "wave3" else 70.0) + min(12.0, max(0.0, float(arm["divergence_strength"]) * 8.0))
                signals.append({
                    "date": pd.Timestamp(row["date"]),
                    "signal_type": "wave3_div_reversal" if kind == "wave3" else "wave5_exhaust_reversal",
                    "stop_anchor": stop_anchor,
                    "anchor_date": pd.Timestamp(row["date"]),
                    "structure_score": min(82.0, structure),
                    "source_i": i,
                    "divergence_date": pd.Timestamp(g.iloc[int(arm["divergence_i"])]["date"]),
                    "prior_low_date": pd.Timestamp(g.iloc[int(arm["prior_low_i"])]["date"]),
                    "anchor_strength": float(row["ret"]),
                    "volume_quality": min(1.5, float(row["amount"] / max(float(row.get("amt_ma10", row["amount"])), 1.0)))
                })
                diag[f"{kind}_confirmed"] += 1
                arms[kind] = None
        div = wave_divergence_candidate(g, i)
        if div is not None:
            kind = str(div["kind"])
            current = arms.get(kind)
            if current is None or int(div["divergence_i"]) >= int(current["divergence_i"]):
                arms[kind] = div
                diag[f"{kind}_armed"] += 1
        for anchor in anchors:
            if not anchor["valid"]:
                continue
            age = i - int(anchor["i"])
            if age < int(a_cfg["min_age_days"]) or age > int(a_cfg["max_age_days"]):
                continue
            if i - int(anchor.get("last_setup_i", -999)) < int(a_cfg["duplicate_setup_cooldown_days"]):
                continue
            dist = float(row["close"] / anchor["start"] - 1)
            pullback_ret_ok = float(a_cfg["pullback_daily_ret_min"]) <= float(row.get("ret", 0.0)) <= float(a_cfg["pullback_daily_ret_max"])
            stable = pullback_ret_ok and float(row.get("close_pos", 0.0)) >= float(a_cfg["pullback_close_pos_min"])
            amount_vs_anchor = float(row["amount"] / max(float(anchor["amount"]), 1.0))
            amount_vs_ma10 = float(row["amount"] / max(float(row.get("amt_ma10", row["amount"])), 1.0))
            volume_shrink = amount_vs_anchor <= float(a_cfg["pullback_amount_vs_anchor_max"]) and amount_vs_ma10 <= float(a_cfg["pullback_amount_vs_ma10_max"])
            if -float(a_cfg["near_support_below"]) <= dist <= float(a_cfg["near_support_above"]) and stable and volume_shrink:
                volume_quality = max(0.0, 1.0 - amount_vs_anchor)
                structure = 48.0 + min(10.0, float(anchor["daily_return"]) * 100.0 - 7.0) + min(8.0, volume_quality * 20.0)
                signals.append({
                    "date": pd.Timestamp(row["date"]), "signal_type": "right_pullback", "stop_anchor": float(anchor["start"]),
                    "anchor_date": pd.Timestamp(anchor["date"]), "structure_score": min(70.0, structure), "source_i": i,
                    "divergence_date": pd.NaT, "prior_low_date": pd.NaT, "anchor_strength": float(anchor["daily_return"]),
                    "volume_quality": float(volume_quality)
                })
                anchor["last_setup_i"] = i
                diag["right_pullback_setups"] += 1
            if int(p_cfg["min_age_days"]) <= age <= int(p_cfg["max_age_days"]):
                after = g.iloc[int(anchor["i"]) + 1:i + 1]
                if len(after) >= int(p_cfg["min_age_days"]):
                    midpoint_held = float(after["close"].min()) >= float(anchor["midpoint"]) * (1 - float(p_cfg["midpoint_break_tolerance"]))
                    platform_low = float(after["low"].min())
                    pdist = float(row["close"] / platform_low - 1)
                    near_platform = -float(p_cfg["near_platform_below"]) <= pdist <= float(p_cfg["near_platform_above"])
                    if midpoint_held and near_platform and stable and volume_shrink:
                        structure = 50.0 + min(10.0, float(anchor["daily_return"]) * 100.0 - 7.0) + min(8.0, max(0.0, 1.0 - amount_vs_anchor) * 20.0)
                        signals.append({
                            "date": pd.Timestamp(row["date"]), "signal_type": "pingbu_qingyun", "stop_anchor": platform_low,
                            "anchor_date": pd.Timestamp(anchor["date"]), "structure_score": min(72.0, structure), "source_i": i,
                            "divergence_date": pd.NaT, "prior_low_date": pd.NaT, "anchor_strength": float(anchor["daily_return"]),
                            "volume_quality": float(max(0.0, 1.0 - amount_vs_anchor))
                        })
                        anchor["last_setup_i"] = i
                        diag["pingbu_setups"] += 1
        big = compute_bigbar_anchor(row, prev)
        if big is not None:
            anchors.append({
                "i": i, "date": pd.Timestamp(row["date"]), "start": float(big["start"]), "midpoint": float(big["midpoint"]),
                "daily_return": float(big["daily_return"]), "amount": float(big["amount"]), "amount_ratio": float(big["amount_ratio"]),
                "valid": True, "last_setup_i": -999
            })
            diag["anchors_created"] += 1
        anchors = [a for a in anchors if i - int(a["i"]) <= int(a_cfg["max_age_days"]) + 5]
    if not signals:
        return pd.DataFrame(), diag
    out = pd.DataFrame(signals).sort_values(["date", "structure_score", "signal_type"], ascending=[True, False, True])
    return out, diag


def score_signals(signals: pd.DataFrame, daily: pd.DataFrame, market: pd.DataFrame, sector_available: bool) -> pd.DataFrame:
    if signals.empty:
        return signals
    lookup_cols = ["code", "date", "rs5_rank", "rs20_rank", "liquidity_rank", "drawdown20", "recent_big_bars20", "sector_rank"]
    scored = signals.merge(daily[lookup_cols], on=["code", "date"], how="left")
    scored = scored.merge(market[["date", "market_score", "max_exposure"]], on="date", how="left")
    rs20 = scored["rs20_rank"].fillna(0.50)
    liq = scored["liquidity_rank"].fillna(0.0)
    market_component = scored["market_score"].fillna(50.0) / 100.0 * 10.0
    liq_component = liq * 5.0
    rs_component = rs20 * 12.0
    sector_component = scored["sector_rank"].fillna(0.50) * (8.0 if sector_available else 0.0)
    wave_mask = scored["signal_type"].isin(["wave3_div_reversal", "wave5_exhaust_reversal"])
    scored["candidate_score"] = scored["structure_score"] + market_component + liq_component + sector_component
    scored.loc[~wave_mask, "candidate_score"] += rs_component[~wave_mask]
    scored.loc[wave_mask, "candidate_score"] += (-scored.loc[wave_mask, "drawdown20"].fillna(0).clip(-0.40, 0) * 20.0)
    scored["candidate_score"] = scored["candidate_score"].clip(0, 100)
    scored["score_threshold"] = float(CFG["scoring"]["min_score_with_sector"] if sector_available else CFG["scoring"]["min_score_no_sector"])
    scored["score_pass"] = scored["candidate_score"] >= scored["score_threshold"]
    scored["liquidity_pass"] = scored["liquidity_rank"].fillna(0) >= float(CFG["scoring"]["liquidity_rank_min"])
    scored["sector_pass"] = True if not sector_available else scored["sector_rank"].fillna(0) >= float(CFG["scoring"]["sector_rank_min"])
    scored["rs_pass"] = True
    right_mask = scored["signal_type"].isin(["right_pullback", "pingbu_qingyun"])
    scored.loc[right_mask, "rs_pass"] = scored.loc[right_mask, "rs20_rank"].fillna(0) >= float(CFG["scoring"]["right_side_rs20_rank_min"])
    scored["absolute_gate_pass"] = scored["score_pass"] & scored["liquidity_pass"] & scored["sector_pass"] & scored["rs_pass"] & (scored["max_exposure"].fillna(0) > 0)
    return scored


def detect_all_signals(daily: pd.DataFrame, market: pd.DataFrame, sector_available: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_signals = []
    diag_rows = []
    for code, g in daily.groupby("code", sort=False):
        s, diag = detect_signals_one(g)
        if not s.empty:
            s["code"] = str(code)
            all_signals.append(s)
        diag["code"] = str(code)
        diag_rows.append(diag)
    signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows)
    if not signals.empty:
        signals = score_signals(signals, daily, market, sector_available)
    return signals, diag_df


def is_bad_status(r: pd.Series) -> bool:
    for c in ["is_st", "is_star_st", "is_new_listing_initial"]:
        if c in r and pd.notna(r[c]) and bool(r[c]):
            return True
    if "is_suspended" in r and pd.notna(r["is_suspended"]) and bool(r["is_suspended"]):
        return True
    return False


def tradable_sell(r: pd.Series, prev_close: float) -> bool:
    if "is_suspended" in r and pd.notna(r["is_suspended"]) and bool(r["is_suspended"]):
        return False
    px = float(r["first_open"])
    if not np.isfinite(px) or px <= 0:
        return False
    lim = r.get("price_limit_down_pct", np.nan)
    if pd.notna(lim) and prev_close > 0 and px <= prev_close * (1 - abs(pct_decimal(lim)) + 0.001):
        return False
    return True


def load_intraday_for_codes(fp: str, codes: list[str]) -> pd.DataFrame:
    cols = ["code", "datetime", "open", "high", "low", "close", "volume", "amount"]
    if not codes:
        return pd.DataFrame(columns=cols)
    try:
        x = pd.read_parquet(fp, columns=cols, filters=[("code", "in", codes)])
    except Exception:
        x = pd.read_parquet(fp, columns=cols)
    x["code"] = normalize_code(x["code"])
    x = x[x["code"].isin(codes)].copy()
    x["datetime"] = pd.to_datetime(x["datetime"])
    return x.sort_values(["code", "datetime"])


def find_intraday_entry(x: pd.DataFrame, signal: pd.Series, prev_close: float, status_row: pd.Series) -> tuple[dict[str, Any] | None, str]:
    if x.empty:
        return None, "no_intraday_data"
    if is_bad_status(status_row):
        return None, "bad_status"
    cfg = CFG["intraday"]
    start_t = parse_hhmm(str(cfg["not_before"]))
    end_t = parse_hhmm(str(cfg["not_after"]))
    x = x.sort_values("datetime").reset_index(drop=True).copy()
    day_open = float(x.iloc[0]["open"])
    if prev_close <= 0 or day_open / prev_close - 1 > float(cfg["max_open_gap"]):
        return None, "open_gap_too_high"
    lim = status_row.get("price_limit_up_pct", np.nan)
    limit_up = prev_close * (1 + pct_decimal(lim)) if pd.notna(lim) else np.inf
    stop_anchor = float(signal["stop_anchor"])
    running_low = math.inf
    running_low_idx = -1
    cum_pv = 0.0
    cum_vol = 0.0
    lookback = int(cfg["lookback_bars_for_flow"])
    for i, r in x.iterrows():
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
        if running_low < stop_anchor * (1 - float(cfg["anchor_intraday_break_tolerance"])):
            return None, "intraday_anchor_break"
        if t < start_t:
            continue
        if t > end_t:
            break
        if close >= limit_up * 0.999:
            continue
        if close / prev_close - 1 > float(cfg["max_chase"]):
            continue
        rebound = close / running_low - 1 if running_low > 0 else 0.0
        bars_since_low = i - running_low_idx
        if rebound < float(cfg["rebound_from_low_min"]):
            continue
        if bars_since_low < int(cfg["stable_bars_after_low"]):
            continue
        if close < vwap * (1 - float(cfg["vwap_tolerance"])):
            continue
        lb = x.iloc[max(0, i - lookback + 1):i + 1]
        up_amt = float(lb.loc[lb["close"] >= lb["open"], "amount"].sum())
        down_amt = float(lb.loc[lb["close"] < lb["open"], "amount"].sum())
        flow_ratio = up_amt / max(down_amt, 1.0)
        if flow_ratio < float(cfg["up_down_amount_ratio_min"]):
            continue
        rest = x.iloc[i:]
        return {
            "raw_price": close, "entry_time": ts, "intraday_rebound": rebound, "bars_since_low": int(bars_since_low),
            "flow_ratio": flow_ratio, "entry_day_post_high": float(rest["high"].max()), "entry_day_post_low": float(rest["low"].min())
        }, "confirmed"
    return None, "no_intraday_confirmation"


@dataclass
class Position:
    code: str
    shares: float
    entry_date: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    entry_cost: float
    stop_anchor: float
    anchor_date: pd.Timestamp
    signal_type: str
    signal_date: pd.Timestamp
    candidate_score: float
    market_exposure: float
    hold_days: int = 0
    highest_close: float = 0.0
    highest_high: float = 0.0
    lowest_low: float = math.inf


def run_backtest(daily: pd.DataFrame, exec_px: pd.DataFrame, minute_map: dict[pd.Timestamp, str], signals: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily = daily.sort_values(["date", "code"])
    exec_px = exec_px.sort_values(["date", "code"])
    dates = sorted(pd.Timestamp(d) for d in daily["date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}
    by_day = {pd.Timestamp(d): x.set_index("code") for d, x in daily.groupby("date")}
    ex_day = {pd.Timestamp(d): x.set_index("code") for d, x in exec_px.groupby("date")}
    market_by_date = market.set_index("date")
    signals_by_date = {pd.Timestamp(d): x.copy() for d, x in signals.groupby("date")} if not signals.empty else {}
    cash = START_CASH
    positions: dict[str, Position] = {}
    cooldown_until: dict[str, int] = {}
    trades = []
    curve = []
    rejects = []
    pending_entries = pd.DataFrame()
    prev_date = None
    priority = {"wave5_exhaust_reversal": 0, "wave3_div_reversal": 1, "pingbu_qingyun": 2, "right_pullback": 3}
    for date in dates:
        day = by_day[date]
        ex = ex_day.get(date)
        if ex is None:
            continue
        idx = date_index[date]
        if prev_date is not None:
            prev = by_day[prev_date]
            for code in list(positions.keys()):
                if code not in prev.index or code not in ex.index:
                    continue
                pos = positions[code]
                pr = prev.loc[code]
                er = ex.loc[code]
                exit_reason = None
                if float(pr["close"]) < pos.stop_anchor * (1 - float(CFG["anchor"]["break_tolerance"])):
                    exit_reason = "anchor_break"
                elif pos.hold_days >= MAX_HOLD_DAYS:
                    exit_reason = "time_stop"
                elif pos.hold_days >= int(CFG["risk"]["trail_min_hold_days"]) and float(pr["close"]) < pos.highest_close * (1 - float(CFG["risk"]["trail_drawdown"])):
                    exit_reason = "trail_8pct"
                if exit_reason and tradable_sell(er, float(pr["close"])):
                    raw = float(er["first_open"])
                    px = raw * (1 - SLIPPAGE)
                    proceeds = pos.shares * px * (1 - COMMISSION - SELL_STAMP)
                    cash += proceeds
                    pnl = proceeds - pos.entry_cost
                    trades.append({
                        "code": code, "signal_type": pos.signal_type, "signal_date": pos.signal_date, "anchor_date": pos.anchor_date,
                        "entry_date": pos.entry_date, "entry_time": pos.entry_time, "entry_price": pos.entry_price,
                        "candidate_score": pos.candidate_score, "market_exposure": pos.market_exposure, "exit_date": date,
                        "exit_price": px, "exit_reason": exit_reason, "hold_days": pos.hold_days, "pnl": pnl,
                        "gross_return_pct": px / pos.entry_price - 1, "net_return_pct": pnl / pos.entry_cost if pos.entry_cost > 0 else np.nan,
                        "mfe_pct": pos.highest_high / pos.entry_price - 1, "mae_pct": pos.lowest_low / pos.entry_price - 1
                    })
                    if exit_reason == "anchor_break":
                        cooldown_until[code] = idx + int(CFG["cooldown_days_after_stop"])
                    del positions[code]
        if prev_date is not None and not pending_entries.empty:
            max_exposure = float(market_by_date.loc[prev_date]["max_exposure"]) if prev_date in market_by_date.index else 0.0
            candidates = pending_entries.copy()
            candidates["priority"] = candidates["signal_type"].map(priority).fillna(9)
            candidates = candidates.sort_values(["candidate_score", "priority", "code"], ascending=[False, True, True], kind="mergesort")
            filtered_codes = []
            filtered_rows = []
            for _, sig in candidates.iterrows():
                code = str(sig["code"])
                reason = None
                if not bool(sig.get("absolute_gate_pass", False)):
                    reason = "absolute_gate_fail"
                elif code in positions:
                    reason = "already_held"
                elif idx <= cooldown_until.get(code, -1):
                    reason = "cooldown_after_stop"
                elif code not in ex.index or code not in day.index:
                    reason = "missing_execution_row"
                if reason:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": reason, "score": float(sig.get("candidate_score", np.nan))})
                    continue
                filtered_codes.append(code)
                filtered_rows.append(sig)
            intraday = load_intraday_for_codes(minute_map[date], sorted(set(filtered_codes))) if filtered_codes and date in minute_map else pd.DataFrame()
            intraday_groups = {c: x for c, x in intraday.groupby("code")} if not intraday.empty else {}
            current_values = 0.0
            for code, pos in positions.items():
                current_values += pos.shares * (float(ex.loc[code]["first_open"]) if code in ex.index else pos.entry_price)
            equity_pre = cash + current_values
            target_invested = equity_pre * max_exposure
            invest_capacity = max(0.0, target_invested - current_values)
            for sig in filtered_rows:
                if len(positions) >= MAX_POSITIONS or invest_capacity <= 0:
                    rejects.append({"date": date, "code": str(sig["code"]), "signal_type": sig["signal_type"], "reason": "portfolio_exposure_limit", "score": float(sig["candidate_score"])})
                    continue
                code = str(sig["code"])
                er = ex.loc[code]
                pr = by_day[prev_date].loc[code]
                confirm, reason = find_intraday_entry(intraday_groups.get(code, pd.DataFrame()), sig, float(pr["close"]), er)
                if confirm is None:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": reason, "score": float(sig["candidate_score"])})
                    continue
                slots = MAX_POSITIONS - len(positions)
                alloc = min(cash, invest_capacity, target_invested / max(MAX_POSITIONS, 1), cash / max(slots, 1))
                if alloc < 10_000:
                    rejects.append({"date": date, "code": code, "signal_type": sig["signal_type"], "reason": "allocation_too_small", "score": float(sig["candidate_score"])})
                    continue
                px = float(confirm["raw_price"]) * (1 + SLIPPAGE)
                shares = alloc / (px * (1 + COMMISSION))
                cost = shares * px * (1 + COMMISSION)
                if shares <= 0 or cost > cash + 1e-6:
                    continue
                cash -= cost
                positions[code] = Position(
                    code=code, shares=shares, entry_date=date, entry_time=pd.Timestamp(confirm["entry_time"]), entry_price=px,
                    entry_cost=cost, stop_anchor=float(sig["stop_anchor"]), anchor_date=pd.Timestamp(sig["anchor_date"]),
                    signal_type=str(sig["signal_type"]), signal_date=prev_date, candidate_score=float(sig["candidate_score"]),
                    market_exposure=max_exposure, hold_days=0, highest_close=float(day.loc[code]["close"]),
                    highest_high=max(px, float(confirm["entry_day_post_high"])), lowest_low=min(px, float(confirm["entry_day_post_low"]))
                )
                invest_capacity -= cost
        equity = cash
        for code, pos in positions.items():
            if code in day.index:
                r = day.loc[code]
                c = float(r["close"])
                equity += pos.shares * c
                pos.highest_close = max(pos.highest_close, c)
                if date > pos.entry_date:
                    pos.highest_high = max(pos.highest_high, float(r["high"]))
                    pos.lowest_low = min(pos.lowest_low, float(r["low"]))
                pos.hold_days += 1
            else:
                equity += pos.shares * pos.entry_price
        mrow = market_by_date.loc[date] if date in market_by_date.index else pd.Series({"market_score": np.nan, "max_exposure": np.nan})
        curve.append({
            "date": date, "equity": equity, "cash": cash, "positions": len(positions),
            "market_score": float(mrow.get("market_score", np.nan)), "max_exposure": float(mrow.get("max_exposure", np.nan))
        })
        pending_entries = signals_by_date.get(date, pd.DataFrame()).copy()
        prev_date = date
    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    rejects_df = pd.DataFrame(rejects)
    if not curve_df.empty:
        curve_df["peak"] = curve_df["equity"].cummax()
        curve_df["drawdown"] = curve_df["equity"] / curve_df["peak"] - 1
        total_return = float(curve_df["equity"].iloc[-1] / START_CASH - 1)
        max_dd = float(curve_df["drawdown"].min())
    else:
        total_return = np.nan
        max_dd = np.nan
    stats: dict[str, Any] = {
        "strategy_version": CFG["version"], "start_cash": START_CASH,
        "final_equity": float(curve_df["equity"].iloc[-1]) if not curve_df.empty else None,
        "total_return": total_return, "multiple": float(1 + total_return) if np.isfinite(total_return) else None,
        "max_drawdown": max_dd, "trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()) if len(trades_df) else None,
        "avg_trade_return": float(trades_df["net_return_pct"].mean()) if len(trades_df) else None,
        "median_trade_return": float(trades_df["net_return_pct"].median()) if len(trades_df) else None,
        "rejected_entries": int(len(rejects_df)), "open_positions": int(len(positions))
    }
    if len(trades_df):
        stats["by_signal"] = trades_df.groupby("signal_type").agg(
            trades=("pnl", "size"), win_rate=("pnl", lambda x: float((x > 0).mean())), avg_return=("net_return_pct", "mean"),
            total_pnl=("pnl", "sum"), avg_mfe=("mfe_pct", "mean"), avg_mae=("mae_pct", "mean")
        ).reset_index().to_dict("records")
        stats["by_exit"] = trades_df.groupby("exit_reason").agg(
            trades=("pnl", "size"), avg_return=("net_return_pct", "mean"), total_pnl=("pnl", "sum")
        ).reset_index().to_dict("records")
        fast = trades_df[trades_df["hold_days"] <= 3]
        slow = trades_df[trades_df["hold_days"] >= 4]
        stats["hold_3d_or_less"] = {"trades": int(len(fast)), "win_rate": float((fast["pnl"] > 0).mean()) if len(fast) else None, "total_pnl": float(fast["pnl"].sum()) if len(fast) else 0.0}
        stats["hold_4d_or_more"] = {"trades": int(len(slow)), "win_rate": float((slow["pnl"] > 0).mean()) if len(slow) else None, "total_pnl": float(slow["pnl"].sum()) if len(slow) else 0.0}
    if len(rejects_df):
        stats["rejections_by_reason"] = rejects_df.groupby("reason").size().sort_values(ascending=False).to_dict()
    return curve_df, trades_df, rejects_df, stats


def main() -> None:
    print("loading minute data")
    daily, exec_px, minute_map = load_daily_from_minute()
    print("adding stock indicators")
    daily = add_all_stock_indicators(daily)
    market = compute_market_features(daily)
    sector_map = load_sector_map()
    daily, sector_available = add_sector_features(daily, sector_map)
    print("sector filter", "enabled" if sector_available else "disabled (no backtest/sector_map.csv)")
    print("detecting causal signals")
    signals, signal_diag = detect_all_signals(daily, market, sector_available)
    print("signals", len(signals), "gated", int(signals["absolute_gate_pass"].sum()) if not signals.empty else 0)
    curve, trades, rejects, stats = run_backtest(daily, exec_px, minute_map, signals, market)
    curve.to_csv(os.path.join(OUT, "equity_curve.csv"), index=False)
    trades.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    rejects.to_csv(os.path.join(OUT, "entry_rejections.csv"), index=False)
    signals.to_csv(os.path.join(OUT, "signals.csv"), index=False)
    signal_diag.to_csv(os.path.join(OUT, "signal_diagnostics.csv"), index=False)
    market.to_csv(os.path.join(OUT, "market_regime.csv"), index=False)
    if not curve.empty:
        monthly = curve.set_index("date")["equity"].resample("ME").last().reset_index()
        monthly["monthly_return"] = monthly["equity"].pct_change()
        monthly.to_csv(os.path.join(OUT, "monthly_equity.csv"), index=False)
    stats["sector_filter_enabled"] = bool(sector_available)
    stats["signal_counts"] = signals["signal_type"].value_counts().to_dict() if not signals.empty else {}
    stats["gated_signal_counts"] = signals.loc[signals["absolute_gate_pass"], "signal_type"].value_counts().to_dict() if not signals.empty else {}
    stats["signal_diagnostics_totals"] = signal_diag.drop(columns=["code"]).sum(numeric_only=True).astype(int).to_dict()
    stats["config"] = CFG
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
