#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

import first_buy_v13_walkforward as base
import first_buy_v13_meta_walkforward as meta

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"backtest"/"results_first_buy_v13_stratified"
OUT.mkdir(parents=True,exist_ok=True)
RESEARCH_REV = "V13.1b"\nMONTHS=["2026-03","2026-04","2026-05","2026-06","2026-07","2026-08"]


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


def add_scores(e):
    x=e.copy()
    s=np.zeros(len(x),dtype=float)
    # Height itself can partially self-certify leadership.
    s += np.where(x["prior_streak"]>=8,3,np.where(x["prior_streak"]>=6,2,np.where(x["prior_streak"]>=5,1,0)))
    s += np.where(x["height_gap"]<=0,2,np.where(x["height_gap"]<=1,1,np.where(x["height_gap"]>=4,-1,0)))
    # Theme position is a prior, not a hard veto.
    s += np.where(x["prior_leader_flag"]>0,2,0)
    s += np.where(x["shock_flag"]>0,1,0)
    s += np.where(x["prior_sector_rank"]<=1,2,np.where(x["prior_sector_rank"]<=3,1,np.where(x["prior_sector_rank"]>=8,-1,0)))
    s += np.where(x["prior_sector_lu3"]>=10,2,np.where(x["prior_sector_lu3"]>=5,1,np.where(x["prior_sector_lu3"]<3,-1,0)))
    # Single-stock / tiny themes need height or event evidence instead of being auto-core.
    isolated=(x["sector_n"].fillna(0)<=2)&(x["prior_sector_lu3"].fillna(0)<3)
    s += np.where(isolated,-1,0)
    x["identity_score"]=s

    q=np.zeros(len(x),dtype=float)
    q += np.where(x["event_ret"]>=0.03,2,np.where(x["event_ret"]>=0,1,np.where(x["event_ret"]<-0.08,-2,0)))
    q += np.where(x["close_vwap"]>=1.01,2,np.where(x["close_vwap"]>=0.98,1,np.where(x["close_vwap"]<0.95,-2,0)))
    q += np.where(x["close_loc"]>=0.70,2,np.where(x["close_loc"]>=0.45,1,np.where(x["close_loc"]<0.20,-1,0)))
    q += np.where(x["touched_upper"]>0,1,0)
    q += np.where((x["vol_prev_ratio"]>=1.5)|((x["vol_med10"]>=2)&(x["vol_prev_ratio"]>=0.8)),1,0)
    x["event_score"]=q

    m=np.zeros(len(x),dtype=float)
    m += np.where(x["m30_reclaim_event"]>=1,2,0)
    m += np.where(x["m30_close_vwap"]>=0.005,2,np.where(x["m30_close_vwap"]>=0,1,-1))
    m += np.where(x["m30_close_loc"]>=0.70,2,np.where(x["m30_close_loc"]>=0.50,1,np.where(x["m30_close_loc"]<0.30,-1,0)))
    m += np.where(x["m30_ret_open"]>=0.02,1,np.where(x["m30_ret_open"]<=-0.03,-1,0))
    x["confirm_score"]=m

    x["gap_regime"]=pd.cut(
        x["d1_gap"],bins=[-0.103,-0.04,0,0.08,0.103],
        labels=["PANIC","LOW","NORMAL","HIGH"],include_lowest=True,right=False
    ).astype("object")
    return x


def ret_col(mode,hold):
    return f"ret_{'open' if mode=='OPEN' else 'confirm'}_H{hold}"


def select_rows(df,regime,identity_min,event_min,confirm_min,hold):
    z=df[(df["pool_ultra"])&(df["gap_regime"]==regime)].copy()
    z=z[(z["identity_score"]>=identity_min)&(z["event_score"]>=event_min)]
    mode="OPEN" if regime in ("NORMAL","HIGH") else "M30"
    if mode=="OPEN":
        z=z[z["d1_tradable"]]
    else:
        z=z[z["m30_entry"].notna()&(z["confirm_score"]>=confirm_min)]
    col=ret_col(mode,hold)
    z=z[z[col].notna()].copy()
    z["ret_rule"]=z[col]
    z["entry_mode_rule"]=mode
    z["hold_rule"]=hold
    return z


def stats(z):
    if len(z)==0:return {"trades":0,"win_rate":None,"mean":None,"median":None,"worst":None,"sum":0}
    r=z["ret_rule"].astype(float)
    return {"trades":int(len(r)),"win_rate":float((r>0).mean()),"mean":float(r.mean()),
            "median":float(r.median()),"worst":float(r.min()),"sum":float(r.sum())}


def utility(st, candidate_n):
    if st["trades"]==0 or st["mean"] is None:return -1e9
    coverage=st["trades"]/max(candidate_n,1)
    return 110*st["mean"] + 11*(st["win_rate"]-.5) + 1.5*math.log1p(st["trades"]) + 35*min(0,st["worst"]) + 2*coverage


def configurations(regime):
    ids=range(1,9)
    evs=range(0,7)
    confs=[0] if regime in ("NORMAL","HIGH") else range(1,8)
    holds=[2,3,5]
    for i in ids:
      for e in evs:
       for c in confs:
        for h in holds:
         yield (i,e,c,h)


def choose_for_regime(history,regime):
    candidates=history[(history["pool_ultra"])&(history["gap_regime"]==regime)]
    if len(candidates)<3:return None
    best=None
    for cfg in configurations(regime):
        z=select_rows(history,regime,*cfg)
        st=stats(z)
        # Avoid a rule that gets its score from a single lucky ticket.
        if st["trades"]<max(2,math.ceil(0.12*len(candidates))):continue
        if st["mean"]<=0:continue
        # Wins are a soft target, but below 50% is not acceptable for the learned overlay.
        if st["win_rate"]<0.50:continue
        u=utility(st,len(candidates))
        if best is None or u>best[0]:
            best=(u,cfg,st)
    return best


def walkforward(e):
    test_months=["2026-06","2026-07","2026-08"]
    alltr=[]; folds=[]
    for tm in test_months:
        ti=MONTHS.index(tm)
        hist=e[e["month"].isin(MONTHS[:ti])].copy()
        test=e[e["month"]==tm].copy()
        month_parts=[]
        regime_detail=[]
        for regime in ["PANIC","LOW","NORMAL","HIGH"]:
            chosen=choose_for_regime(hist,regime)
            if chosen is None:
                regime_detail.append({"regime":regime,"selected":None})
                continue
            u,cfg,hst=chosen
            z=select_rows(test,regime,*cfg)
            if len(z):
                z=z.copy();z["test_month"]=tm;z["regime_rule"]=regime
                month_parts.append(z);alltr.append(z)
            regime_detail.append({"regime":regime,"config":{"identity_min":cfg[0],"event_min":cfg[1],"confirm_min":cfg[2],"hold":cfg[3]},
                                  "history_metrics":hst,"test_metrics":stats(z),"utility":u,
                                  "test_candidates":int(((test["pool_ultra"])&(test["gap_regime"]==regime)).sum())})
        month=pd.concat(month_parts,ignore_index=True) if month_parts else pd.DataFrame()
        folds.append({"test_month":tm,"regimes":regime_detail,"combined_test":stats(month)})
    allx=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
    return folds,allx


def bucket_table(e):
    rows=[]
    z=e[e["pool_ultra"]].copy()
    z["identity_bin"]=pd.cut(z["identity_score"],[-99,1,3,5,7,99],labels=["<=1","2-3","4-5","6-7","8+"])
    z["streak_bin"]=pd.cut(z["prior_streak"],[3,4,5,7,99],labels=["4","5","6-7","8+"])
    for keys,g in z.groupby(["streak_bin","identity_bin","gap_regime"],observed=True):
        for mode,col in [("OPEN","h3_ret_open"),("M30","h3_ret_confirm")]:
            h=g[g[col].notna()]
            if len(h)==0:continue
            r=h[col]
            rows.append({"streak_bin":str(keys[0]),"identity_bin":str(keys[1]),"gap_regime":keys[2],"mode":mode,
                         "n":int(len(h)),"win_rate":float((r>0).mean()),"mean":float(r.mean()),
                         "median":float(r.median()),"worst":float(r.min())})
    return pd.DataFrame(rows)


def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,list):return [safe(x) for x in v]
    if isinstance(v,pd.Timestamp):return v.strftime("%Y-%m-%d")
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return None if np.isnan(v) else float(v)
    if isinstance(v,float) and math.isnan(v):return None
    return v


def main():
    daily,e=prepare()
    e=add_scores(e)
    folds,trades=walkforward(e)
    buckets=bucket_table(e)
    payload=safe({
      "method":"V13.1 stratified leader prior: broad ULTRA pool, interpretable identity/event/confirmation scores, regime-specific walk-forward entry and hold",
      "candidate_count":int(e["pool_ultra"].sum()),
      "metadata_coverage":int(e.loc[e["pool_ultra"],"meta_missing"].eq(0).sum()),
      "walkforward_folds":folds,
      "combined":stats(trades),
      "trades":trades[["code","date","d1_date","test_month","regime_rule","identity_score","event_score","confirm_score","entry_mode_rule","hold_rule","ret_rule","prior_streak","height_gap","sector_rank","prior_sector_rank","prior_sector_lu3","leader_flag","prior_leader_flag","shock_flag","d1_gap"]].to_dict("records") if len(trades) else []
    })
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    e.to_csv(OUT/"scored_events.csv",index=False)
    buckets.to_csv(OUT/"bucket_diagnostics.csv",index=False)
    print("V13_STRATIFIED_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_STRATIFIED_END")


if __name__=="__main__":main()
