#!/usr/bin/env python3
from __future__ import annotations

import json
import pandas as pd
import numpy as np

from first_buy_v12_research import load_daily, build_features, first_break_universe

TICK = 0.01


def minute_reclaim(row, path_map, minutes=60, target_mult=0.998):
    d = pd.Timestamp(row["d1_date"]).strftime("%Y-%m-%d")
    path = path_map.get(d)
    if not path:
        return None
    m = pd.read_parquet(path, columns=["code","datetime","open","high","low","close","amount"])
    m["code"] = m["code"].astype(str).str.zfill(6)
    m = m[m["code"] == row["code"]].sort_values("datetime").head(minutes)
    if m.empty:
        return None
    target = float(row["close"]) * target_mult
    hit = m[m["close"] >= target]
    if hit.empty:
        return None
    z = hit.iloc[0]
    return float(z["close"]), str(z["datetime"])


def with_open_metrics(x):
    x = x.copy()
    x["entry_price"] = x["d1_open"]
    x["entry_time"] = "OPEN"
    x["h1_ret"] = x["h1_close"] / x["entry_price"] - 1
    x["h3_ret"] = x["h3_close"] / x["entry_price"] - 1
    x["h5_ret"] = x["h5_close"] / x["entry_price"] - 1
    return x


def with_reclaim_metrics(x, path_map, minutes):
    rows = []
    for _, r in x.iterrows():
        hit = minute_reclaim(r, path_map, minutes=minutes)
        if hit is None:
            continue
        z = r.copy()
        z["entry_price"], z["entry_time"] = hit
        z["h1_ret"] = z["h1_close"] / z["entry_price"] - 1
        z["h3_ret"] = z["h3_close"] / z["entry_price"] - 1
        z["h5_ret"] = z["h5_close"] / z["entry_price"] - 1
        rows.append(z)
    if not rows:
        return pd.DataFrame(columns=list(x.columns)+["entry_price","entry_time","h1_ret","h3_ret","h5_ret"])
    return pd.DataFrame(rows)


def summary(x, label):
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
        "h5_mean": float(x["h5_ret"].mean(skipna=True)) if len(x) else None,
    }


def describe_trades(x):
    if x.empty:
        return []
    cols = ["code","date","d1_date","prior_streak","height_gap","ret","close_loc","close_vwap",
            "vol_prev_ratio","vol_med10","d1_gap","entry_price","entry_time","h1_ret","h3_ret","h5_ret"]
    out = []
    for r in x[cols].to_dict("records"):
        for k,v in list(r.items()):
            if isinstance(v, pd.Timestamp):
                r[k] = v.strftime("%Y-%m-%d")
            elif isinstance(v, (float,np.floating)) and not pd.isna(v):
                r[k] = round(float(v), 5)
            elif pd.isna(v):
                r[k] = None
        out.append(r)
    return out


def main():
    daily, path_map = load_daily()
    daily = build_features(daily)
    e = first_break_universe(daily)
    e["touched_upper"] = e["high"] >= e["upper_calc"] - 0.0051
    e["d1_tradable"] = (
        (e["d1_open"] < e["d1_upper_calc"] - 0.0051)
        | (e["d1_low"] < e["d1_upper_calc"] - 0.0051)
    )

    strict = e["support"] & e["explosive_strict"]
    soft = e["support"] & e["explosive_soft"]
    elite = e["height_gap"] <= 1

    variants = {}

    # High-gap continuation: 5+ board may qualify on height alone.
    hg_base = soft & elite & (e["ret"] >= 0) & (e["close_vwap"] >= 0.98) & e["d1_gap"].between(0.08,0.102,inclusive="both") & e["d1_tradable"]
    variants["HG_5PLUS"] = with_open_metrics(e[hg_base & (e["prior_streak"] >= 5) & (e["close_loc"] >= 0.35)])
    variants["HG_5PLUS_OR_4_TOUCH"] = with_open_metrics(e[hg_base & (
        ((e["prior_streak"] >= 5) & (e["close_loc"] >= 0.35))
        | ((e["prior_streak"] == 4) & e["touched_upper"] & (e["close_loc"] >= 0.30))
    )])
    variants["HG_5PLUS_OR_4_TOUCH_STRONGVOL"] = with_open_metrics(e[hg_base & (
        ((e["prior_streak"] >= 5) & (e["close_loc"] >= 0.40))
        | ((e["prior_streak"] == 4) & e["touched_upper"] & (e["vol_prev_ratio"] >= 1.5) & (e["close_loc"] >= 0.30))
    )])

    # Super-high boards: test a conceptual height cutoff, not a ticker exception.
    sh_base = soft & elite & (e["ret"] >= -0.02) & (e["close_loc"] >= 0.45) & (e["close_vwap"] >= 0.98) & e["d1_gap"].between(-0.03,0.08,inclusive="both")
    for min_streak in [8,9,10]:
        variants[f"SUPER_{min_streak}PLUS"] = with_open_metrics(e[sh_base & (e["prior_streak"] >= min_streak)])

    # Small low-open: do not promote blindly. Require a causal first-hour reclaim of event close.
    sl_base = strict & elite & e["prior_streak"].between(4,7) & (e["ret"] >= 0) & (e["close_loc"] >= 0.35) & (e["close_vwap"] >= 0.98) & e["d1_gap"].between(-0.02,0.0,inclusive="left")
    for mins in [30,60,90]:
        variants[f"SMALL_LOW_RECLAIM_{mins}M"] = with_reclaim_metrics(e[sl_base], path_map, mins)

    # Panic lane: only fresh 5+ market-height leaders with a strong positive divergence day.
    for ret_floor in [0.02,0.03,0.04]:
        for mins in [30,60,90]:
            p = e[
                soft & elite
                & (e["prior_streak"] >= 5)
                & (e["ret"] >= ret_floor)
                & (e["close_loc"] >= 0.60)
                & (e["close_vwap"] >= 0.98)
                & e["d1_gap"].between(-0.102,-0.04,inclusive="both")
            ]
            variants[f"PANIC_R{int(ret_floor*100)}_{mins}M"] = with_reclaim_metrics(p, path_map, mins)

    # Same lane with 4-board admission only when the break day touched the upper limit.
    p4 = e[
        soft & elite
        & (
            (e["prior_streak"] >= 5)
            | ((e["prior_streak"] == 4) & e["touched_upper"])
        )
        & (e["ret"] >= 0.03)
        & (e["close_loc"] >= 0.60)
        & (e["close_vwap"] >= 0.98)
        & e["d1_gap"].between(-0.102,-0.04,inclusive="both")
    ]
    variants["PANIC_5PLUS_OR_4_TOUCH_60M"] = with_reclaim_metrics(p4, path_map, 60)

    report = []
    for name,x in variants.items():
        report.append({"summary": summary(x,name), "trades": describe_trades(x)})

    print("V12_REFINE_JSON_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("V12_REFINE_JSON_END")


if __name__ == "__main__":
    main()
