#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import first_buy_v13_walkforward as base
from first_buy_v13_2_relaxed_four import prepare, lane

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"backtest"/"results_first_buy_v13_2_sell_fast"
OUT.mkdir(parents=True,exist_ok=True)


def met(vals):
    r=np.asarray(vals,float)
    return {"n":int(len(r)),"win_rate":float((r>0).mean()),"mean":float(r.mean()),
            "median":float(np.median(r)),"worst":float(r.min()),"best":float(r.max())}


def load_paths():
    _,paths=base.load_daily()
    return paths


def build_paths(z,path_map):
    market_dates=sorted(path_map)
    cache={}
    for _,r in z.iterrows():
        code=str(r["code"]).zfill(6)
        start=pd.Timestamp(r["d1_date"]).strftime("%Y-%m-%d")
        dates=[d for d in market_dates if d>=start][:2]
        points=[]
        for d in dates:
            p=path_map[d]
            cols=["code","datetime","close"]
            try:
                m=pd.read_parquet(p,columns=cols,filters=[("code","==",code)])
            except Exception:
                m=pd.read_parquet(p,columns=cols)
                m["code"]=m["code"].astype(str).str.zfill(6)
                m=m[m["code"]==code]
            if not m.empty:
                m["code"]=m["code"].astype(str).str.zfill(6)
                for q in m.sort_values("datetime").itertuples():
                    points.append((str(q.datetime),float(q.close)))
        cache[(code,pd.Timestamp(r["date"]).strftime("%Y-%m-%d"))]=points
    return cache


def simulate(points,entry,h2,tp=None,sl=None,activate=None,drawdown=None):
    peak=entry
    for ts,px in points:
        ret=px/entry-1
        peak=max(peak,px)
        if sl is not None and ret<=sl:return ret,ts,"SL"
        if tp is not None and ret>=tp:return ret,ts,"TP"
        if activate is not None and peak/entry-1>=activate and px/peak-1<=-drawdown:
            return ret,ts,"TRAIL"
    return h2,"H2_CLOSE","TIME"


def main():
    e=prepare(); z=lane(e)
    path_map=load_paths()
    pcache=build_paths(z,path_map)
    configs=[]

    def run_fixed(tp,sl):
        vals=[];detail=[]
        for _,r in z.iterrows():
            key=(str(r["code"]).zfill(6),r["date"].strftime("%Y-%m-%d"))
            q=simulate(pcache[key],float(r["d1_open"]),float(r["ret_open_H2"]),tp,sl)
            vals.append(q[0]);detail.append({"code":key[0],"event":key[1],"ret":q[0],"exit_time":q[1],"reason":q[2]})
        return vals,detail

    for tp in [None,.06,.08,.10,.12,.15]:
      for sl in [None,-.04,-.06,-.08]:
        vals,d=run_fixed(tp,sl)
        configs.append({"type":"fixed","tp":tp,"sl":sl,**met(vals),"details":d})

    for act in [.06,.08,.10,.12]:
      for dd in [.03,.04,.05,.06]:
        vals=[];detail=[]
        for _,r in z.iterrows():
            key=(str(r["code"]).zfill(6),r["date"].strftime("%Y-%m-%d"))
            q=simulate(pcache[key],float(r["d1_open"]),float(r["ret_open_H2"]),None,-.08,act,dd)
            vals.append(q[0]);detail.append({"code":key[0],"event":key[1],"ret":q[0],"exit_time":q[1],"reason":q[2]})
        configs.append({"type":"trail","activate":act,"drawdown":dd,"sl":-.08,**met(vals),"details":detail})

    for q in configs:
        q["utility"]=100*q["mean"]+8*(q["win_rate"]-.5)+30*min(0,q["worst"])
    ranked=sorted(configs,key=lambda q:q["utility"],reverse=True)
    payload={"baseline_h2":met(z["ret_open_H2"].tolist()),"best":ranked[0],"top10":ranked[:10],
             "discipline":"causal minute-close triggers; cached once per trade; H2 close fallback"}
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V13_2_SELL_FAST_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_2_SELL_FAST_END")
if __name__=="__main__":main()
