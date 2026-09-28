#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import first_buy_v15_observation as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtest" / "results_first_buy_v15_cv"
OUT.mkdir(parents=True, exist_ok=True)

MONTHS = ["2026-03","2026-04","2026-05","2026-06","2026-07","2026-08"]
OUTER_TESTS = ["2026-06","2026-07","2026-08"]

PROFILES = {
    "QUALITY": {"min_share":0.12,"min_win":0.58,"min_mean":0.005,"w_mean":120.0,"w_win":14.0,"w_n":0.18,"w_worst":55.0},
    "BALANCED": {"min_share":0.22,"min_win":0.54,"min_mean":0.000,"w_mean":105.0,"w_win":11.0,"w_n":0.58,"w_worst":42.0},
    "ACTIVE": {"min_share":0.32,"min_win":0.50,"min_mean":0.000,"w_mean":88.0,"w_win":8.5,"w_n":0.95,"w_worst":32.0},
}

def candidate_configs():
    for pool in ["mid","wide","ultra","max","all3plus"]:
        for qo in [0.55,0.65,0.75,0.85]:
            for qw in [0.25,0.40,0.55]:
                if qw >= qo:
                    continue
                for qc in [0.50,0.65,0.80]:
                    for spec in base.EXIT_SPECS:
                        yield pool,qo,qw,qc,spec

def fold_for_config(events, train_months, test_month, cfg):
    pool,qo,qw,qc,spec=cfg
    trn=events[events["month"].isin(train_months)].copy()
    tst=events[events["month"]==test_month].copy()
    tr,te,os,cs,ots,cts,ocoef,ccoef=base.score_bundle(trn,tst,pool)
    if len(tr)<8 or len(te)==0:
        return None
    trades=base.apply_hybrid(te,os,cs,ots,cts,qo,qw,qc,spec)
    return {
        "metrics":base.metrics(trades),"trades":trades,"candidates":int(len(te)),
        "ocoef":ocoef,"ccoef":ccoef,
    }

def historical_cv(events, cutoff_month, cfg):
    ci=MONTHS.index(cutoff_month)
    val_months=MONTHS[2:ci]
    out=[]
    for vm in val_months:
        vi=MONTHS.index(vm)
        r=fold_for_config(events,MONTHS[:vi],vm,cfg)
        if r is not None:
            out.append((vm,r))
    return out

def aggregate_cv(folds):
    parts=[]; month_stats=[]
    for month,r in folds:
        if not r["trades"].empty:
            z=r["trades"].copy(); z["cv_month"]=month; parts.append(z)
        month_stats.append((month,r["metrics"]))
    alltr=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    agg=base.metrics(alltr)
    valid=[m for _,m in month_stats if m["mean"] is not None]
    agg["folds"]=len(valid)
    agg["positive_months"]=sum(1 for m in valid if m["mean"]>0)
    agg["nonnegative_months"]=sum(1 for m in valid if m["mean"]>=0)
    agg["worst_month_mean"]=min((m["mean"] for m in valid),default=None)
    return agg,month_stats

def utility(m,p):
    if m["trades"]==0 or m["mean"] is None or not m["folds"]:
        return -1e9
    stability=m["nonnegative_months"]/m["folds"]
    worst=min(0.0,m["worst"])
    month_floor=min(0.0,m["worst_month_mean"]) if m["worst_month_mean"] is not None else -1.0
    return (
        p["w_mean"]*m["mean"] + p["w_win"]*(m["win_rate"]-0.5)
        + p["w_n"]*math.log1p(m["trades"]) + p["w_worst"]*worst
        + 3.0*stability + 25.0*month_floor
    )

def select_config(events,test_month,profile_name):
    p=PROFILES[profile_name]
    best=None
    for cfg in candidate_configs():
        folds=historical_cv(events,test_month,cfg)
        if not folds:
            continue
        agg,month_stats=aggregate_cv(folds)
        total_candidates=sum(r["candidates"] for _,r in folds)
        min_trades=max(2,math.ceil(p["min_share"]*total_candidates))
        if agg["trades"]<min_trades:
            continue
        if agg["win_rate"]<p["min_win"] or agg["mean"]<p["min_mean"]:
            continue
        if agg["folds"]>=2 and agg["nonnegative_months"]<math.ceil(agg["folds"]/2):
            continue
        u=utility(agg,p)
        if best is None or u>best[0]:
            best=(u,cfg,agg,month_stats)
    return best

def run(events):
    results={}
    for profile_name in PROFILES:
        folds=[]; alltr=[]
        for tm in OUTER_TESTS:
            selected=select_config(events,tm,profile_name)
            if selected is None:
                folds.append({"test_month":tm,"selected":None})
                continue
            u,cfg,cvagg,cvmonths=selected
            ti=MONTHS.index(tm)
            test=fold_for_config(events,MONTHS[:ti],tm,cfg)
            if test is None:
                folds.append({"test_month":tm,"selected":None})
                continue
            pool,qo,qw,qc,spec=cfg
            trades=test["trades"]
            if not trades.empty:
                z=trades.copy(); z["test_month"]=tm; z["profile"]=profile_name; alltr.append(z)
            folds.append({
                "test_month":tm,"pool":pool,"q_open":qo,"q_watch":qw,"q_confirm":qc,"exit":spec.name,
                "selection_utility":u,"prior_cv":cvagg,
                "prior_cv_months":[{"month":m,**met} for m,met in cvmonths],
                "test_metrics":test["metrics"],"test_candidates":test["candidates"],
                "top_open_coef":sorted(test["ocoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:8],
                "top_confirm_coef":sorted(test["ccoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:8],
            })
        combined=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
        results[profile_name]={
            "combined_metrics":base.metrics(combined),"folds":folds,
            "trades":combined.to_dict("records") if not combined.empty else [],
        }
    return results

def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,list): return [safe(x) for x in v]
    if isinstance(v,tuple): return [safe(x) for x in v]
    if isinstance(v,pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v,float) and math.isnan(v): return None
    return v

def main():
    daily,path_map=base.load_daily()
    daily=base.build_features(daily)
    daily=base.attach_d1_amount(daily)
    daily=base.add_history_features(daily)
    events=base.build_relaxed_events(daily)
    events=base.add_m30_features(events,path_map)
    events=base.add_fixed_targets(events)
    events=base.precompute_exit_returns(events,daily)
    payload=safe({
        "method":"V15 all-3plus observation pool + causal Day2 execution + multi-month pseudo-OOS configuration selection",
        "observation_diagnostics":base.observation_diagnostics(daily, events),
        "pool_diagnostics":base.pool_diagnostics(events),
        "profiles":run(events),
    })
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V15_CV_SUMMARY_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V15_CV_SUMMARY_END")

if __name__=="__main__":
    main()
