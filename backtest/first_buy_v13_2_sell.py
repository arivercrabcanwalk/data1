#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import first_buy_v13_walkforward as base
from first_buy_v13_2_relaxed_four import prepare, lane

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"backtest"/"results_first_buy_v13_2_sell"
OUT.mkdir(parents=True,exist_ok=True)


def minute_rows(path,code):
    cols=["code","datetime","open","high","low","close"]
    try:
        m=pd.read_parquet(path,columns=cols,filters=[("code","==",code)])
    except Exception:
        m=pd.read_parquet(path,columns=cols)
        m["code"]=m["code"].astype(str).str.zfill(6)
        m=m[m["code"]==code]
    if m.empty:return m
    m["code"]=m["code"].astype(str).str.zfill(6)
    return m.sort_values("datetime")


def simulate(row,path_map,tp,sl,hold_days=2,trail_activate=None,trail_drawdown=None):
    entry=float(row["d1_open"])
    if not np.isfinite(entry) or entry<=0:return None
    # derive next stock trading dates from event row's shifted daily dates is unavailable here,
    # so use sorted market dates starting at d1_date.
    start=pd.Timestamp(row["d1_date"]).strftime("%Y-%m-%d")
    dates=sorted(d for d in path_map if d>=start)[:hold_days]
    peak=entry
    for d in dates:
        m=minute_rows(path_map[d],str(row["code"]).zfill(6))
        if m.empty:continue
        for z in m.itertuples():
            px=float(z.close)
            ret=px/entry-1
            peak=max(peak,px)
            if sl is not None and ret<=sl:
                return {"ret":ret,"exit_time":str(z.datetime),"reason":"SL"}
            if tp is not None and ret>=tp:
                return {"ret":ret,"exit_time":str(z.datetime),"reason":"TP"}
            if trail_activate is not None and peak/entry-1>=trail_activate:
                draw=px/peak-1
                if draw<=-trail_drawdown:
                    return {"ret":ret,"exit_time":str(z.datetime),"reason":"TRAIL"}
    # Fallback to H2 close already computed from exact stock sessions.
    return {"ret":float(row["ret_open_H2"]),"exit_time":"H2_CLOSE","reason":"TIME"}


def met(vals):
    if not vals:return {"n":0}
    r=np.asarray(vals,float)
    return {"n":int(len(r)),"win_rate":float((r>0).mean()),"mean":float(r.mean()),
            "median":float(np.median(r)),"worst":float(r.min()),"best":float(r.max())}


def main():
    # prepare() returns event features; reload path_map separately.
    _,path_map=base.load_daily()
    e=prepare()
    z=lane(e)
    configs=[]
    for tp in [None,.06,.08,.10,.12,.15]:
      for sl in [None,-.04,-.06,-.08]:
        vals=[]; details=[]
        for _,r in z.iterrows():
          q=simulate(r,path_map,tp,sl,2)
          vals.append(q["ret"])
          details.append({"code":r["code"],"event":r["date"].strftime("%Y-%m-%d"),**q})
        configs.append({"type":"fixed","tp":tp,"sl":sl,**met(vals),"details":details})
    for act in [.06,.08,.10,.12]:
      for dd in [.03,.04,.05,.06]:
        vals=[];details=[]
        for _,r in z.iterrows():
          q=simulate(r,path_map,None,-.08,2,act,dd)
          vals.append(q["ret"])
          details.append({"code":r["code"],"event":r["date"].strftime("%Y-%m-%d"),**q})
        configs.append({"type":"trail","activate":act,"drawdown":dd,"sl":-.08,**met(vals),"details":details})

    # Objective favors expectancy and win rate but penalizes worse left tail.
    for q in configs:
        if q["n"]:
            q["utility"]=100*q["mean"]+8*(q["win_rate"]-.5)+30*min(0,q["worst"])
        else:q["utility"]=-1e9
    ranked=sorted(configs,key=lambda q:q["utility"],reverse=True)
    payload={
      "lane":"RELAXED_4_NORMAL_ABSORPTION_FAST",
      "baseline_h2":met(z["ret_open_H2"].dropna().tolist()),
      "best":ranked[0],
      "top10":ranked[:10],
      "discipline":"minute-close causal triggers only; no intraday high hindsight; H2 close fallback"
    }
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([{k:v for k,v in q.items() if k!="details"} for q in ranked]).to_csv(OUT/"sell_grid.csv",index=False)
    print("V13_2_SELL_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_2_SELL_END")


if __name__=="__main__":main()
