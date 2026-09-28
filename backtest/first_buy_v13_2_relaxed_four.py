#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import first_buy_v13_walkforward as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"backtest"/"results_first_buy_v13_2"
OUT.mkdir(parents=True,exist_ok=True)


def prepare():
    daily,path_map=base.load_daily()
    daily=base.build_features(daily)
    daily=base.attach_d1_amount(daily)
    daily=base.add_history_features(daily)
    e=base.build_relaxed_events(daily)
    e=base.add_m30_features(e,path_map)
    e=base.add_fixed_targets(e)
    e=base.precompute_exit_returns(e,daily)
    return e


def strict_explosion(e):
    return (e["vol_prev_ratio"]>=1.5)|((e["vol_med10"]>=2.0)&(e["vol_prev_ratio"]>=0.8))


def lane(e, amount_min=3e8, range_min=.06, close_vwap_min=.995, close_loc_min=.55):
    z=e[
        (e["prior_streak"]==4)
        & (e["height_gap"]<=2)
        & (e["amount"]>=amount_min)
        & (e["range"]>=range_min)
        & strict_explosion(e)
        & (e["event_ret"]>=0)
        & (e["close_vwap"]>=close_vwap_min)
        & (e["close_loc"]>=close_loc_min)
        & (e["d1_gap"]>=0)
        & (e["d1_gap"]<.08)
        & e["d1_tradable"]
    ].copy()
    return z


def metrics(z,col):
    r=z[col].dropna()
    return {
        "n":int(len(r)),
        "win_rate":float((r>0).mean()) if len(r) else None,
        "mean":float(r.mean()) if len(r) else None,
        "median":float(r.median()) if len(r) else None,
        "worst":float(r.min()) if len(r) else None,
        "best":float(r.max()) if len(r) else None,
    }


def main():
    e=prepare()
    z=lane(e)

    hold={h:metrics(z,f"ret_open_H{h}") for h in [2,3,5]}
    monthly={}
    z["month"]=z["date"].dt.strftime("%Y-%m")
    for m,g in z.groupby("month"):
        monthly[m]=metrics(g,"ret_open_H2")

    neigh=[]
    for amount in [3e8,5e8,8e8,1e9,1.5e9]:
      for rng in [.05,.06,.075,.09]:
       for cv in [.99,.995,1.0,1.005]:
        for cl in [.50,.55,.60,.65]:
          q=lane(e,amount,rng,cv,cl)
          s=metrics(q,"ret_open_H2")
          neigh.append({
            "amount_min":amount,"range_min":rng,"close_vwap_min":cv,"close_loc_min":cl,
            **s,
            "codes":";".join(q["code"].astype(str).tolist())
          })
    ndf=pd.DataFrame(neigh)
    ndf.to_csv(OUT/"neighborhood.csv",index=False)

    trades=z[[
        "code","date","d1_date","prior_streak","height_gap","ret20","event_ret","range",
        "vol_prev_ratio","vol_med10","amount","close_vwap","close_loc","d1_gap",
        "ret_open_H2","ret_open_H3","ret_open_H5"
    ]].copy()
    trades.to_csv(OUT/"trades.csv",index=False)

    stable=ndf[
        (ndf["range_min"].between(.06,.075))
        & (ndf["close_vwap_min"].between(.995,1.0))
        & (ndf["close_loc_min"].between(.55,.60))
    ]
    payload={
        "version":"V13.2_RESEARCH_CANDIDATE",
        "lane":"RELAXED_4_NORMAL_ABSORPTION_FAST",
        "rule":{
            "prior_streak":4,
            "height_gap_max":2,
            "amount_min":300000000,
            "range_min":.06,
            "strict_explosion":True,
            "event_ret_min":0,
            "close_vwap_min":.995,
            "close_loc_min":.55,
            "d1_gap":[0,.08],
            "entry":"D+1 open",
            "exit":"close of second trading session starting with entry day (H2)",
            "explicitly_not_hard_gates":["ret20","sector_rank","followers","leader","shock_base"]
        },
        "hold_comparison":hold,
        "monthly_h2":monthly,
        "neighborhood_stability":{
            "tested_rows":int(len(stable)),
            "rows_with_same_trade_count_as_center":int((stable["n"]==len(z)).sum()),
            "rows_100pct_h2_in_seen_history":int((stable["win_rate"]==1.0).sum()),
            "min_win_rate":float(stable["win_rate"].min()),
            "min_mean":float(stable["mean"].min()),
        },
        "trades":trades.assign(
            date=trades["date"].dt.strftime("%Y-%m-%d"),
            d1_date=trades["d1_date"].dt.strftime("%Y-%m-%d")
        ).to_dict("records")
    }
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V13_2_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_2_END")


if __name__=="__main__":
    main()
