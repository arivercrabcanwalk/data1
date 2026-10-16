#!/usr/bin/env python3
"""R81 Phase2: frozen predecessor successfully-recaptured reset memory.
Historical causal tracking, no current or later episode outcome in features.
Account proxy replayer imported from R81 W04 with exact frozen R79 parity.
"""
from pathlib import Path
from collections import defaultdict
import json,runpy
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent.parent
O=R/"w06_rearm_memory";L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
H=Path.home()/"Documents/stock_data_windows_mirror/daily/tdx_parquet"
assert (R/"PHASE2_PRIOR_REARM_MEMORY_FROZEN.md").exists()
mother=pd.read_csv(R/"w02_population/R81_fullmarket_reset_episodes.csv",dtype={"code":str})
codes=set(mother.code)
files=sorted(H.glob("date=*/daily_ohlcv.parquet"))
files=[p for p in files if "2025-09-01"<=p.parent.name[5:]<="2026-10-16"]
days=[p.parent.name[5:] for p in files];di={d:i for i,d in enumerate(days)}
records=[]
for j,p in enumerate(files):
 d=pd.read_parquet(p,columns=["code","date","name","board","close","amount","volume"])
 d.code=d.code.astype(str).str.zfill(6)
 d=d[d.code.isin(codes)]
 records.append(d)
 if j%65==0:print("REARM_DAILY_READ",p.parent.name,len(d),flush=True)
d=pd.concat(records,ignore_index=True).sort_values(["code","date"])
grp=d.groupby("code",sort=False)
d["prior_close"]=grp.close.shift(1)
d["upclose"]=d.close.div(d.prior_close).sub(1).ge(.095) & d.close.div(d.prior_close).sub(1).le(.115)
d["upclose"]&=d.board.eq("主板")&~d.name.fillna("").str.contains("ST|退",case=False,regex=True)
d["upclose"]&=d.amount.gt(0)&d.volume.gt(0)
upset={c:set(z.loc[z.upclose,"date"]) for c,z in d.groupby("code",sort=False)}
event_groups={c:list(z.sort_values("date").itertuples(index=False)) for c,z in mother.groupby("code")}
out=[]
for code,rs in event_groups.items():
 for i,r in enumerate(rs):
  now=int(di[r.date]);previous=[]
  # Only earlier reset events that had recovered to a positive limit-ish day
  # within 1,2,3 complete exchange sessions prior to current reset day.
  for p in rs[:i]:
   pday=int(di[p.date])
   if now-pday>30:continue
   recapture_dates=[days[z] for z in range(pday+1,min(pday+4,now+1))
           if days[z] in upset[code]]
   if recapture_dates:previous.append((p.date,recapture_dates[0]))
  # If an up-day could be credited to 2 nearby reset dates, it counts once.
  seen=set();recaptures=[]
  for pdate,rebday in sorted(previous,key=lambda x:x[0]):
   if rebday not in seen:
    seen.add(rebday);recaptures.append((pdate,rebday))
  record={"code":code,"reset_date":r.date,"reset_uid":r.reset_uid,"past_reset_count30":sum(0<int(now-di[p.date])<=30 for p in rs[:i]),
     "prior_successful_recapture_distinct":len(recaptures),
     "LOOP1":len(recaptures)>=1,"LOOP2":len(recaptures)>=2,
     "past_success_evidence":";".join(a+">"+b for a,b in recaptures)}
  # Every recapture date MUST be <= current reset date (strict causality).
  assert all(rebday<=r.date and pdate<r.date for pdate,rebday in recaptures)
  out.append(record)
mem=pd.DataFrame(out)
assert len(mem)==len(mother)
assert mem[["code","reset_date"]].duplicated().sum()==0
mem.to_csv(O/"R81_PRIOR_SUCCESS_MEMORY_EACH_RESET.csv",index=False)
print("R81_MEMORY_STOCKDAYS",len(mem),"loop1",int(mem.LOOP1.sum()),"loop2",int(mem.LOOP2.sum()),flush=True)
print("R81_601811_ASOF_LOOP",mem[mem.code=="601811"].to_dict("records"),flush=True)
# The complete shared account replayer is re-run idempotently. It can generate
# no production orders (research-only csv; no network calls to broker).
context=runpy.run_path(str(R/"w04_portfolio/replay_account.py"))
oldsrc=context["oldsrc"]; prev=context["prev"]; replay=context["replay"]; B=context["B"]; comp=context["comp"]
original_all_new=context["all_new"]
x=original_all_new.merge(mem,on=["code","reset_date"],how="left",validate="m:1")
assert x.LOOP1.notna().all()
assert ((x.LOOP2)&(~x.LOOP1)).sum()==0
reports=[];accepted=[];rejected=[]
cols=B.columns.intersection(x.columns)
for loop in ["LOOP1","LOOP2"]:
 for gate in ["BASE","P4","P6"]:
  for branch in ["BOTH","AM_REASSERT","PM_RECAPTURE"]:
   for exit_variant in ["BASE_T3","T1_WEAK"]:
    z=x[x[loop]&x[gate]&x.variant.eq(exit_variant)]
    if branch!="BOTH":z=z[z.branch.eq(branch)]
    # signal arrival sort; no post-hoc selection using per-trade returns.
    z=z.sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"])
    new=z[cols].copy()
    ac,rj=replay(pd.concat([oldsrc,prev,new],ignore_index=True))
    only=ac[ac.source.eq("R81")]
    closed=only[~only.censored]
    oldok=ac[ac.source.ne("R81")]
    removed=B[~B.refid.isin(oldok.refid)]
    newcontrib=float((only.ret*only.weight).sum()*100)
    displaced=float((oldok.ret*oldok.weight).sum()*100-comp)
    assert abs(float((ac.ret*ac.weight).sum()*100)-comp-newcontrib-displaced)<1e-9
    obj={"loop":loop,"gate":gate,"branch":branch,"exit_variant":exit_variant,
       "signals_pre_capacity":len(new),"new_accepted":len(only),"closed":len(closed),
       "new_wins":int(closed.ret.gt(0).sum()),"winpct":100*closed.ret.gt(0).mean() if len(closed) else None,
       "avg_single_pct":100*closed.ret.mean() if len(closed) else None,
       "MAE10":int(only.mae.le(-.10).sum()),"final_loss10":int(closed.ret.le(-.10).sum()),
       "censored":int(only.censored.sum()),"unique_codes":only.code.nunique(),
       "months":closed.entry_dt.dt.to_period("M").nunique(),"no_original_R50_A_lost":int(((removed.source=="R50")&(removed.method=="A")).sum())==0,
       "old_R50_A_removed":int(((removed.source=="R50")&(removed.method=="A")).sum()),
       "old_R50_B_removed":int(((removed.source=="R50")&(removed.method=="B")).sum()),
       "old_R78_R79_removed":int(removed.source.isin(["R78","R79"]).sum()),
       "profit_proxy_new_pp":newcontrib,"displacement_pp":displaced,
       "delta_vs_frozen_R79_pp":newcontrib+displaced}
    reports.append(obj);tag=loop+"_"+gate+"_"+branch+"_"+exit_variant
    for y in only.to_dict("records"):
     accepted.append({"tag":tag,"code":y["code"],"entry_dt":y["entry_dt"],"exit_dt":y["exit_dt"],
       "ret_net_pct":y["ret"]*100,"MAE_pct":y["mae"]*100,"censored":y["censored"],"weight":y["weight"]})
    for y in rj[rj.source.eq("R81")].to_dict("records"):
     rejected.append({"tag":tag,"code":y["code"],"entry_dt":y["entry_dt"],"why":y.get("reason")})
T=pd.DataFrame(reports).sort_values("delta_vs_frozen_R79_pp",ascending=False)
T.to_csv(O/"R81_LOOP1_LOOP2_36_FROZEN_PIT_ACCOUNT_SCENARIOS.csv",index=False)
pd.DataFrame(accepted).to_csv(O/"R81_LOOP_ACCOUNT_ACCEPTED_DETAILS.csv",index=False)
pd.DataFrame(rejected).to_csv(O/"R81_LOOP_ACCOUNT_REJECTED.csv",index=False)
q={"status":"R81_LOOP_MEMORY_ALL_36_SCENARIOS_RAN","reset_days":len(mem),
   "reset_days_with_loop1":int(mem.LOOP1.sum()),"reset_days_with_loop2":int(mem.LOOP2.sum()),
   "R81_signals_with_loop1":int(x.loc[x.variant.eq("BASE_T3"),"LOOP1"].sum()),
   "R81_signals_with_loop2":int(x.loc[x.variant.eq("BASE_T3"),"LOOP2"].sum()),
   "baseline_R79_exact_profit_proxy_pp":comp,"scenarios":len(T),
   "all_36_reported_not_only_winners":True,
   "no_trading_result_used_to_create_prior_loop":True,
   "upclose_are_unofficial_price_proxy":True,
   "historical_multiple_comparison_bias":True,"study_after_viewing_R81_loss":True,
   "real_OOS":False,"no_production_modifications":True}
(O/"qa.json").write_text(json.dumps(q,ensure_ascii=False,indent=2))
print("R81_REARM_MEMORY_QA",json.dumps(q,ensure_ascii=False),flush=True)
print("R81_REARM_TOP12",T.head(12).round(4).to_string(index=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]