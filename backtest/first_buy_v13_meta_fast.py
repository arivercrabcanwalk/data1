#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import first_buy_v13_walkforward as base
import first_buy_v13_meta_walkforward as meta

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"backtest"/"results_first_buy_v13_meta_fast"
OUT.mkdir(parents=True,exist_ok=True)
MONTHS=meta.MONTHS
OUTER_TESTS=meta.OUTER_TESTS

def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,list): return [safe(x) for x in v]
    if isinstance(v,tuple): return [safe(x) for x in v]
    if isinstance(v,pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v,float) and math.isnan(v): return None
    return v

def prepare():
    daily,path_map=base.load_daily()
    daily=base.build_features(daily)
    daily=base.attach_d1_amount(daily)
    daily=base.add_history_features(daily)
    e=base.build_relaxed_events(daily)
    e=base.add_m30_features(e,path_map)
    e=base.add_fixed_targets(e)
    e=base.precompute_exit_returns(e,daily)
    e=meta.add_meta(e)
    return daily,e

def make_cache(e):
    cache={}
    # Each validation/test month always trains on all strictly earlier months.
    for tm in MONTHS[2:]:
        ti=MONTHS.index(tm)
        train=e[e["month"].isin(MONTHS[:ti])].copy()
        test=e[e["month"]==tm].copy()
        for pool in ["mid","wide","ultra"]:
            tr=base.subset_pool(train,pool); te=base.subset_pool(test,pool)
            if len(tr)<8 or len(te)==0:
                continue
            os,ocoef=base.fit_score(tr,te,meta.OPEN_FEATURES,"h3_ret_open")
            cs,ccoef=base.fit_score(tr[tr["m30_entry"].notna()],te,meta.CONFIRM_FEATURES,"h3_ret_confirm")
            ots,_=base.fit_score(tr,tr,meta.OPEN_FEATURES,"h3_ret_open")
            ctr=tr[tr["m30_entry"].notna()].copy()
            cts,_=base.fit_score(ctr,ctr,meta.CONFIRM_FEATURES,"h3_ret_confirm")
            cache[(tm,pool)]=dict(tr=tr,te=te,os=os,cs=cs,ots=ots,cts=cts,ocoef=ocoef,ccoef=ccoef)
    return cache

def eval_cfg(cache,tm,cfg):
    pool,qo,qw,qc,spec=cfg
    b=cache.get((tm,pool))
    if b is None: return None
    t=base.apply_hybrid(b["te"],b["os"],b["cs"],b["ots"],b["cts"],qo,qw,qc,spec)
    return {"metrics":base.metrics(t),"trades":t,"candidates":len(b["te"]),
            "ocoef":b["ocoef"],"ccoef":b["ccoef"]}

def cfgs():
    for pool in ["mid","wide","ultra"]:
      for qo in [0.60,0.70,0.80,0.88]:
       for qw in [0.35,0.50,0.60]:
        if qw>=qo: continue
        for qc in [0.55,0.70,0.82]:
         for spec in base.EXIT_SPECS:
          yield (pool,qo,qw,qc,spec)

def aggregate(parts):
    alltr=[]; stats=[]
    for m,r in parts:
        if not r["trades"].empty:
            z=r["trades"].copy(); z["cv_month"]=m; alltr.append(z)
        stats.append((m,r["metrics"]))
    z=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
    a=base.metrics(z)
    a["folds"]=len(stats)
    a["positive_months"]=sum(1 for _,x in stats if x["mean"] is not None and x["mean"]>0)
    a["nonnegative_months"]=sum(1 for _,x in stats if x["mean"] is not None and x["mean"]>=0)
    return a,stats

def util(m,p):
    if m["trades"]==0 or m["mean"] is None:return -1e9
    stability=m["positive_months"]/m["folds"] if m["folds"] else 0
    return p["w_mean"]*m["mean"]+p["w_win"]*(m["win_rate"]-.5)+p["w_n"]*math.log1p(m["trades"])+p["w_worst"]*min(0,m["worst"])+2.5*stability

def choose(cache,test_month,profile_name):
    p=meta.PROFILE[profile_name]
    ti=MONTHS.index(test_month)
    vals=MONTHS[2:ti]
    best=None
    for cfg in cfgs():
        parts=[]
        for vm in vals:
            r=eval_cfg(cache,vm,cfg)
            if r:parts.append((vm,r))
        if not parts:continue
        agg,stats=aggregate(parts)
        total_candidates=sum(r["candidates"] for _,r in parts)
        min_trades=max(2,math.ceil(p["min_share"]*total_candidates))
        if agg["trades"]<min_trades:continue
        if agg["win_rate"]<p["min_win"] or agg["mean"]<p["min_mean"]:continue
        if agg["folds"]>=2 and agg["nonnegative_months"]<math.ceil(agg["folds"]/2):continue
        u=util(agg,p)
        if best is None or u>best[0]:best=(u,cfg,agg,stats)
    return best

def run(e,cache):
    out={}
    for pname in meta.PROFILE:
        folds=[]; trades=[]
        for tm in OUTER_TESTS:
            sel=choose(cache,tm,pname)
            if sel is None:
                folds.append({"test_month":tm,"selected":None});continue
            u,cfg,cvagg,cvstats=sel
            tst=eval_cfg(cache,tm,cfg)
            pool,qo,qw,qc,spec=cfg
            if not tst["trades"].empty:
                z=tst["trades"].copy();z["test_month"]=tm;z["profile"]=pname;trades.append(z)
            folds.append({"test_month":tm,"pool":pool,"q_open":qo,"q_watch":qw,"q_confirm":qc,"exit":spec.name,
                          "selection_utility":u,"prior_cv":cvagg,
                          "prior_cv_months":[{"month":m,**x} for m,x in cvstats],
                          "test_metrics":tst["metrics"],"test_candidates":tst["candidates"],
                          "top_open_coef":sorted(tst["ocoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:10],
                          "top_confirm_coef":sorted(tst["ccoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:10]})
        allx=pd.concat(trades,ignore_index=True) if trades else pd.DataFrame()
        out[pname]={"combined_metrics":base.metrics(allx),"folds":folds,
                    "trades":allx.to_dict("records") if not allx.empty else []}
    return out

def main():
    daily,e=prepare()
    cache=make_cache(e)
    result=run(e,cache)
    payload=safe({"method":"cached metadata-aware nested walk-forward",
                  "metadata_coverage":{"ultra":int(e["pool_ultra"].sum()),
                                       "ultra_with_meta":int(e.loc[e["pool_ultra"],"meta_missing"].eq(0).sum())},
                  "profiles":result})
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V13_META_FAST_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_META_FAST_END")
if __name__=="__main__":main()
