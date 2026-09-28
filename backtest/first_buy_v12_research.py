#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtest" / "results_first_buy_v12"
OUT.mkdir(parents=True, exist_ok=True)

MONTHS = {f"2026-{m:02d}" for m in range(3, 9)}
MAIN_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
TICK = 0.01


def date_from_path(path: str) -> str:
    m = re.search(r"date=(\d{4}-\d{2}-\d{2})", path)
    if not m:
        raise ValueError(path)
    return m.group(1)


def round_half_up_cent(x):
    # A-share reference prices are rounded to fen; avoid banker's rounding.
    return np.floor(np.asarray(x, dtype=float) * 100.0 + 0.5000001) / 100.0


def read_status(path: str):
    sp = path.replace("minute1.parquet", "daily_stock_status.parquet")
    if not os.path.exists(sp):
        return None
    try:
        s = pd.read_parquet(sp)
    except Exception:
        return None
    if "code" not in s.columns:
        return None
    s["code"] = s["code"].astype(str).str.zfill(6)
    keep = ["code"]
    for c in ["market_board", "is_st", "is_star_st", "is_new_listing_initial",
              "security_status", "price_limit_up_pct", "price_limit_down_pct"]:
        if c in s.columns:
            keep.append(c)
    return s[keep].drop_duplicates("code")


def aggregate_day(path: str):
    d = date_from_path(path)
    df = pd.read_parquet(path, columns=["code", "datetime", "open", "high", "low", "close", "volume", "amount"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].str.startswith(MAIN_PREFIXES)]
    if df.empty:
        return pd.DataFrame()
    # Files are already stock/time ordered. Still sort to make first/last deterministic.
    df = df.sort_values(["code", "datetime"], kind="stable")
    g = df.groupby("code", sort=False)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        bars=("datetime", "size"),
    ).reset_index()
    out["date"] = d
    v = out["amount"] / out["volume"].replace(0, np.nan)
    # Some sources express volume in lots, others in shares.
    out["vwap"] = np.where(v / out["close"] > 20, v / 100.0, v)

    s = read_status(path)
    if s is not None:
        out = out.merge(s, on="code", how="left")
        bad = pd.Series(False, index=out.index)
        for c in ["is_st", "is_star_st", "is_new_listing_initial"]:
            if c in out.columns:
                bad |= out[c].fillna(False).astype(bool)
        out = out[~bad].copy()
    return out


def load_daily():
    paths = []
    for p in glob.glob(str(ROOT / "2026-*" / "part-*" / "date=*" / "minute1.parquet")):
        d = date_from_path(p)
        if d[:7] in MONTHS:
            paths.append(p)
    paths.sort(key=date_from_path)
    print(f"V12|daily_files={len(paths)}")
    chunks = []
    for i, p in enumerate(paths, 1):
        x = aggregate_day(p)
        chunks.append(x)
        if i % 10 == 0 or i == len(paths):
            print(f"V12|aggregated={i}/{len(paths)} date={date_from_path(p)} rows={len(x)}")
    daily = pd.concat(chunks, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    return daily, {date_from_path(p): p for p in paths}


def build_features(daily: pd.DataFrame):
    g = daily.groupby("code", group_keys=False)
    for c in ["close", "volume", "amount"]:
        daily[f"prev_{c}"] = g[c].shift(1)
    daily["prev_date"] = g["date"].shift(1)

    # Use status-table price limit percentage if usable, otherwise standard main-board 10%.
    pct = pd.Series(0.10, index=daily.index, dtype=float)
    if "price_limit_up_pct" in daily.columns:
        raw = pd.to_numeric(daily["price_limit_up_pct"], errors="coerce")
        raw = np.where(raw > 1.0, raw / 100.0, raw)
        valid = pd.Series(raw, index=daily.index).between(0.05, 0.15)
        pct.loc[valid] = pd.Series(raw, index=daily.index).loc[valid]
    daily["limit_pct"] = pct
    daily["upper_calc"] = round_half_up_cent(daily["prev_close"] * (1.0 + daily["limit_pct"]))
    # Correct LU classification from price itself. This fixes rows where stored LU metadata is stale/wrong.
    daily["lu_calc"] = (
        daily["prev_close"].notna()
        & (daily["close"] >= daily["upper_calc"] - 0.0051)
    )

    # Consecutive LU streak on each stock's observed trading sessions; suspension gaps do not break it.
    streak = np.zeros(len(daily), dtype=np.int16)
    for _, idx in daily.groupby("code").indices.items():
        s = 0
        for pos in idx:
            if bool(daily.at[pos, "lu_calc"]):
                s += 1
            else:
                s = 0
            streak[pos] = s
    daily["streak_calc"] = streak
    daily["prior_streak"] = g["streak_calc"].shift(1).fillna(0).astype(int)

    daily["ret"] = daily["close"] / daily["prev_close"] - 1.0
    daily["range"] = (daily["high"] - daily["low"]) / daily["prev_close"]
    daily["gap"] = daily["open"] / daily["prev_close"] - 1.0
    daily["vol_prev_ratio"] = daily["volume"] / daily["prev_volume"].replace(0, np.nan)
    daily["amount_prev_ratio"] = daily["amount"] / daily["prev_amount"].replace(0, np.nan)

    med10 = (
        daily.groupby("code")["volume"]
        .rolling(10, min_periods=5)
        .median()
        .reset_index(level=0, drop=True)
    )
    daily["vol_med10"] = daily["volume"] / med10.groupby(daily["code"]).shift(1).replace(0, np.nan)

    daily["close_loc"] = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"]).replace(0, np.nan)
    daily["close_vwap"] = daily["close"] / daily["vwap"].replace(0, np.nan)

    # Market height on each date and how far the candidate is from that height on D-1.
    market_height = daily.groupby("date")["streak_calc"].max().rename("market_height")
    daily = daily.merge(market_height, on="date", how="left")
    mh_prev = market_height.rename("market_height_prev")
    daily = daily.merge(mh_prev, left_on="prev_date", right_index=True, how="left")
    daily["height_gap"] = daily["market_height_prev"] - daily["prior_streak"]

    # Entry day = next observed trading session for this stock.
    for c in ["date", "open", "high", "low", "close", "upper_calc"]:
        daily[f"d1_{c}"] = daily.groupby("code")[c].shift(-1)
    for h in [1, 3, 5]:
        daily[f"h{h}_close"] = daily.groupby("code")["close"].shift(-h)

    # Exact convention: H3 = close of the 3rd session starting with entry day, versus D+1 entry price.
    daily["d1_gap"] = daily["d1_open"] / daily["close"] - 1.0
    return daily


def first_break_universe(df: pd.DataFrame):
    e = df[
        (df["prior_streak"] >= 4)
        & (~df["lu_calc"])
        & df["d1_open"].notna()
        & (df["amount"] >= 3e8)
        & (df["range"] >= 0.06)
    ].copy()
    # only first break after a live streak; the prior_streak definition guarantees this is not a later break
    e["explosive_strict"] = (
        (e["vol_prev_ratio"] >= 1.5)
        | ((e["vol_med10"] >= 2.0) & (e["vol_prev_ratio"] >= 0.80))
    )
    e["explosive_soft"] = (
        (e["vol_prev_ratio"] >= 1.35)
        | ((e["vol_med10"] >= 2.0) & (e["vol_prev_ratio"] >= 0.70))
        | ((e["amount"] >= 1e9) & (e["range"] >= 0.08) & (e["vol_prev_ratio"] >= 0.65))
    )
    e["elite_height"] = e["height_gap"] <= 1
    e["core_height"] = e["height_gap"] <= 2
    e["support"] = (e["ret"] >= -0.08) & (e["close_vwap"] >= 0.95)
    return e


def add_forward_metrics(x: pd.DataFrame, daily: pd.DataFrame):
    if x.empty:
        return x
    x = x.copy()
    # default entry is D+1 open
    x["entry_price"] = x["d1_open"]
    x["entry_time"] = "OPEN"
    x["entry_mode"] = "open"
    x["tradable_at_open"] = (
        (x["d1_open"] < x["d1_upper_calc"] - 0.0051)
        | (x["d1_low"] < x["d1_upper_calc"] - 0.0051)
    )
    x["h1_ret"] = x["h1_close"] / x["entry_price"] - 1
    x["h3_ret"] = x["h3_close"] / x["entry_price"] - 1
    x["h5_ret"] = x["h5_close"] / x["entry_price"] - 1

    # H3 MAE/MFE over entry day + following two stock trading sessions.
    look = daily[["code", "date", "low", "high"]].copy()
    grouped = {c: z.reset_index(drop=True) for c, z in look.groupby("code")}
    maes, mfes = [], []
    for r in x.itertuples():
        z = grouped.get(r.code)
        if z is None:
            maes.append(np.nan); mfes.append(np.nan); continue
        arr = z.index[z["date"] == r.d1_date].to_numpy()
        if len(arr) == 0:
            maes.append(np.nan); mfes.append(np.nan); continue
        i = int(arr[0]); w = z.iloc[i:i+3]
        if w.empty:
            maes.append(np.nan); mfes.append(np.nan); continue
        maes.append(w["low"].min() / r.entry_price - 1)
        mfes.append(w["high"].max() / r.entry_price - 1)
    x["h3_mae"] = maes
    x["h3_mfe"] = mfes
    return x


def minute_panic_entry(row, path_map):
    # Causal intraday confirmation: on D+1, within first 60 minutes price must reclaim
    # the observation-day close. Enter at the first minute close >= event close.
    d = pd.Timestamp(row["d1_date"]).strftime("%Y-%m-%d")
    path = path_map.get(d)
    if not path:
        return None
    m = pd.read_parquet(path, columns=["code", "datetime", "open", "high", "low", "close", "amount"])
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[m["code"] == row["code"]].sort_values("datetime").head(60)
    if m.empty:
        return None
    target = float(row["close"])
    hit = m[m["close"] >= target * 0.998]
    if hit.empty:
        return None
    z = hit.iloc[0]
    return {
        "entry_price": float(z["close"]),
        "entry_time": str(z["datetime"]),
        "entry_mode": "reclaim_event_close_60m",
    }


def select_lanes(e: pd.DataFrame, daily: pd.DataFrame, path_map):
    common_strict = e["support"] & e["explosive_strict"] & e["core_height"]
    common_soft_elite = e["support"] & e["explosive_soft"] & e["elite_height"]

    lanes = {}

    lanes["STANDARD"] = e[
        common_strict
        & e["prior_streak"].between(4, 7)
        & e["d1_gap"].between(0.0, 0.08, inclusive="both")
    ].copy()

    lanes["SMALL_LOW_FULL"] = e[
        common_strict
        & e["prior_streak"].between(4, 7)
        & e["elite_height"]
        & (e["ret"] >= 0.0)
        & (e["close_loc"] >= 0.35)
        & e["d1_gap"].between(-0.02, 0.0, inclusive="left")
    ].copy()

    lanes["HIGH_GAP"] = e[
        common_soft_elite
        & e["prior_streak"].between(4, 7)
        & (e["ret"] >= 0.0)
        & (e["close_loc"] >= 0.40)
        & (e["close_vwap"] >= 0.98)
        & e["d1_gap"].between(0.08, 0.102, inclusive="both")
    ].copy()
    # Exclude true one-price next-day limit boards: no executable trade.
    lanes["HIGH_GAP"] = lanes["HIGH_GAP"][
        (lanes["HIGH_GAP"]["d1_low"] < lanes["HIGH_GAP"]["d1_upper_calc"] - 0.0051)
        | (lanes["HIGH_GAP"]["d1_open"] < lanes["HIGH_GAP"]["d1_upper_calc"] - 0.0051)
    ].copy()

    lanes["SUPER_HIGH"] = e[
        e["explosive_soft"]
        & e["elite_height"]
        & (e["prior_streak"] >= 8)
        & (e["ret"] >= -0.02)
        & (e["close_loc"] >= 0.45)
        & (e["close_vwap"] >= 0.98)
        & e["d1_gap"].between(-0.03, 0.08, inclusive="both")
    ].copy()

    panic = e[
        e["explosive_soft"]
        & e["elite_height"]
        & (e["prior_streak"] >= 5)
        & (e["ret"] >= 0.03)
        & (e["close_loc"] >= 0.60)
        & (e["close_vwap"] >= 0.98)
        & e["d1_gap"].between(-0.102, -0.04, inclusive="both")
    ].copy()
    if not panic.empty:
        entries = []
        for _, row in panic.iterrows():
            conf = minute_panic_entry(row, path_map)
            entries.append(conf)
        panic["_conf"] = entries
        panic = panic[panic["_conf"].notna()].copy()
        if not panic.empty:
            panic["entry_price"] = panic["_conf"].map(lambda x: x["entry_price"])
            panic["entry_time"] = panic["_conf"].map(lambda x: x["entry_time"])
            panic["entry_mode"] = panic["_conf"].map(lambda x: x["entry_mode"])
        panic = panic.drop(columns=["_conf"], errors="ignore")
    lanes["PANIC_REVERSAL"] = panic

    frames = []
    for name, x in lanes.items():
        if x.empty:
            continue
        x = add_forward_metrics(x, daily)
        # preserve panic confirmed entry after generic metric construction
        if name == "PANIC_REVERSAL":
            entries = []
            for _, row in x.iterrows():
                conf = minute_panic_entry(row, path_map)
                entries.append(conf)
            ok = [v is not None for v in entries]
            x = x.loc[ok].copy()
            entries = [v for v in entries if v is not None]
            if len(x):
                x["entry_price"] = [v["entry_price"] for v in entries]
                x["entry_time"] = [v["entry_time"] for v in entries]
                x["entry_mode"] = [v["entry_mode"] for v in entries]
                x["h1_ret"] = x["h1_close"] / x["entry_price"] - 1
                x["h3_ret"] = x["h3_close"] / x["entry_price"] - 1
                x["h5_ret"] = x["h5_close"] / x["entry_price"] - 1
                # recompute H3 MAE/MFE from daily lows/highs against confirmed entry
                grouped = {c: z.reset_index(drop=True) for c, z in daily[["code","date","low","high"]].groupby("code")}
                maes, mfes = [], []
                for r in x.itertuples():
                    z = grouped[r.code]
                    arr = z.index[z["date"] == r.d1_date].to_numpy()
                    if len(arr):
                        w = z.iloc[int(arr[0]):int(arr[0])+3]
                        maes.append(w["low"].min()/r.entry_price-1)
                        mfes.append(w["high"].max()/r.entry_price-1)
                    else:
                        maes.append(np.nan); mfes.append(np.nan)
                x["h3_mae"] = maes; x["h3_mfe"] = mfes
        x["lane"] = name
        frames.append(x)
    if not frames:
        return pd.DataFrame(), lanes
    allx = pd.concat(frames, ignore_index=True)
    # If an event qualifies for multiple lanes, use a deterministic lane priority.
    prio = {"PANIC_REVERSAL":0, "SUPER_HIGH":1, "HIGH_GAP":2, "SMALL_LOW_FULL":3, "STANDARD":4}
    allx["_prio"] = allx["lane"].map(prio)
    allx = allx.sort_values(["code", "date", "_prio"]).drop_duplicates(["code", "date"], keep="first")
    return allx.drop(columns="_prio"), lanes


def summarize(x: pd.DataFrame, label: str):
    if x.empty:
        return {"label": label, "trades": 0}
    z = x[x["h3_ret"].notna()]
    return {
        "label": label,
        "trades": int(len(x)),
        "h3_complete": int(len(z)),
        "h3_win_rate": float((z["h3_ret"] > 0).mean()) if len(z) else None,
        "h3_mean": float(z["h3_ret"].mean()) if len(z) else None,
        "h3_median": float(z["h3_ret"].median()) if len(z) else None,
        "h3_worst": float(z["h3_ret"].min()) if len(z) else None,
        "h3_mae_mean": float(z["h3_mae"].mean()) if len(z) else None,
        "h3_mfe_mean": float(z["h3_mfe"].mean()) if len(z) else None,
        "h5_mean": float(x["h5_ret"].mean(skipna=True)),
    }


def evaluate_neighborhood(e, daily):
    # Daily-open-only neighborhood tests to avoid tuning to one exact threshold.
    rows = []
    for height_gap in [0, 1, 2]:
        for soft_floor in [0.70, 0.75, 0.80]:
            soft = (
                (e["vol_prev_ratio"] >= 1.35)
                | ((e["vol_med10"] >= 2.0) & (e["vol_prev_ratio"] >= soft_floor))
            )
            base = e[
                (e["height_gap"] <= height_gap)
                & soft
                & (e["ret"] >= 0)
                & (e["close_vwap"] >= 0.98)
                & (e["close_loc"] >= 0.40)
                & (e["prior_streak"] >= 4)
                & e["d1_gap"].between(-0.02, 0.102, inclusive="both")
                & ((e["d1_open"] < e["d1_upper_calc"] - 0.0051) | (e["d1_low"] < e["d1_upper_calc"] - 0.0051))
            ].copy()
            base = add_forward_metrics(base, daily)
            s = summarize(base, f"hg={height_gap},vf={soft_floor}")
            rows.append({
                "height_gap": height_gap,
                "vol_floor": soft_floor,
                **{k:v for k,v in s.items() if k != "label"}
            })
    return pd.DataFrame(rows)


def main():
    daily, path_map = load_daily()
    daily = build_features(daily)
    e = first_break_universe(daily)
    print(f"V12|first_break_universe={len(e)}")
    print("V12|prior_streak_counts=" + json.dumps(e["prior_streak"].value_counts().sort_index().to_dict(), ensure_ascii=False))

    allx, lanes = select_lanes(e, daily, path_map)

    lane_summaries = []
    if not allx.empty:
        for lane, z in allx.groupby("lane"):
            lane_summaries.append(summarize(z, lane))
    total_summary = summarize(allx, "V12_BALANCED")

    monthly = []
    if not allx.empty:
        y = allx.copy()
        y["month"] = y["d1_date"].dt.strftime("%Y-%m")
        for m, z in y.groupby("month"):
            s = summarize(z, m)
            monthly.append(s)

    nb = evaluate_neighborhood(e, daily)

    cols = [
        "code","date","d1_date","lane","prior_streak","market_height_prev","height_gap",
        "ret","range","vol_prev_ratio","vol_med10","amount","vwap","close_vwap","close_loc",
        "d1_gap","entry_price","entry_time","entry_mode","h1_ret","h3_ret","h5_ret","h3_mae","h3_mfe"
    ]
    if not allx.empty:
        allx[cols].to_csv(OUT/"trades.csv", index=False)
    e.to_csv(OUT/"first_break_universe.csv", index=False)
    nb.to_csv(OUT/"neighborhood.csv", index=False)

    summary = {
        "period": ["2026-03-01", "2026-08-31"],
        "horizon_definition": "H3 = close of 3rd trading session starting with D+1 entry day, divided by actual entry price",
        "universe_first_breaks": int(len(e)),
        "total": total_summary,
        "lanes": lane_summaries,
        "monthly": monthly,
        "rules": {
            "first_break": "prior streak >=4, current non-limit-up, amount>=300m, range>=6%",
            "limit_up": "recomputed from close versus rounded prior_close*(1+limit_pct)",
            "standard": "strict explosion, height_gap<=2, streak4-7, D1 gap[0,8%]",
            "small_low_full": "strict explosion, height_gap<=1, event_ret>=0, close_loc>=0.35, D1 gap[-2,0)",
            "high_gap": "soft explosion, height_gap<=1, event_ret>=0, close_loc>=0.40, close/vwap>=0.98, D1 gap[8,10.2%], not one-price LU",
            "super_high": "soft explosion, height_gap<=1, prior_streak>=8, event_ret>=-2%, close_loc>=0.45, D1 gap[-3,8%]",
            "panic_reversal": "soft explosion, height_gap<=1, prior_streak>=5, event_ret>=3%, close_loc>=0.60, D1 gap[-10.2,-4%], enter only after first-60m reclaim of event close",
        },
    }
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("V12_SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("V12_SUMMARY_JSON_END")

    if not allx.empty:
        show = allx[cols].copy()
        for c in ["ret","range","vol_prev_ratio","vol_med10","close_vwap","close_loc","d1_gap","h1_ret","h3_ret","h5_ret","h3_mae","h3_mfe"]:
            show[c] = show[c].round(4)
        print("V12_TRADES_CSV_BEGIN")
        print(show.to_csv(index=False))
        print("V12_TRADES_CSV_END")

    print("V12_NEIGHBORHOOD_CSV_BEGIN")
    print(nb.to_csv(index=False))
    print("V12_NEIGHBORHOOD_CSV_END")


if __name__ == "__main__":
    main()
