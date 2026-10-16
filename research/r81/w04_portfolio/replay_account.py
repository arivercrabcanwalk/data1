#!/usr/bin/env python3
"""R81 W04 unified historic accepted-order capacity-PnL PROXY, frozen old base.
Uses old R50 all intents + R79 frozen accepted 33; new entries compete, A orders
are not deleted from the input; report any collisions. No real broker NAV.
"""
from pathlib import Path
import json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent.parent
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
O=ROOT/"w04_portfolio"
oldfolder=L/"r60_capacity_stress_20261008"
old=pd.concat([pd.read_csv(oldfolder/"deploy_pilot_accepted.csv",dtype={"code":str}),
               pd.read_csv(oldfolder/"deploy_pilot_rejected.csv",dtype={"code":str})],ignore_index=True)
assert len(old)==76 and old.uid.is_unique
oldsrc=pd.DataFrame({"code":old.code,"name":old.name,"entry_dt":pd.to_datetime(old.entry_dt),
    "exit_dt":pd.to_datetime(old.exit_dt),"method":old.method,"weight":old.weight.astype(float),
    "ret":old.ret.astype(float),"priority":old.priority.astype(int),"score1":old.score1.astype(float),
    "score2":old.score2.astype(float),"source":"R50","refid":old.uid,
    "censored":False,"route":old.route,"mae":np.nan,"id_group":"LEGACY"})
older=pd.read_csv(L/"r79_volume_quality_20261016/w14_phase3/accepted_trades.csv",dtype={"code":str})
older=older[older.variant.eq("P4_B_H_E_D_G")].copy()
assert len(older)==33 and older[["code","entry"]].duplicated().sum()==0
prev=pd.DataFrame({"code":older.code,"name":older.name,"entry_dt":pd.to_datetime(older.entry),
    "exit_dt":pd.to_datetime(older.exit),"method":older.source.map(lambda x: str(x)+"_B"),
    "weight":older.weight.astype(float),"ret":older.ret.astype(float),"priority":older.source.map({"R78":3,"R79":4}),
    "score1":0.,"score2":0.,"source":older.source,
    "refid":older.source+"|"+older.code+"|"+older.entry.astype(str)+"|"+older.route,
    "censored":older.censored.astype(bool),"route":older.route,"mae":older.mae.astype(float),
    "id_group":"OLD_RESEARCH_ACCEPTED_ONLY"})
assert prev.refid.is_unique
H=pd.read_csv(O/"R81_trades_with_frozen_PIT.csv",dtype={"code":str})
assert H.groupby(["code","buy_date","entry_time","branch"]).size().eq(2).all()
# Nonterminal missing exit minute data is censorship; state occupied through
# data end and do not count an imaginary unrealized profit in PnL.
all_new=H.copy()
all_new["entry_dt"]=pd.to_datetime(all_new.buy_date+" "+all_new.entry_time)
all_new["exit_dt"]=pd.to_datetime(all_new.exit_date.fillna("2026-09-23")+" "+all_new.exit_time.fillna("15:00"))
all_new.loc[all_new.status.eq("RIGHT_CENSORED"),"exit_dt"]=pd.Timestamp("2026-09-23 15:00:00")
all_new["ret"]=all_new.return_net_pct.fillna(0)/100
all_new["weight"]=all_new.entry_weight
all_new["priority"]=5
all_new["score1"]=0.
all_new["score2"]=0.
all_new["source"]="R81"
all_new["method"]="R81_B"
all_new["censored"]=all_new.status.eq("RIGHT_CENSORED")
all_new["mae"]=all_new.MAE_pct/100
all_new["refid"]=all_new.code+"|"+all_new.buy_date+"|"+all_new.entry_time+"|"+all_new.branch+"|"+all_new.variant
all_new["route"]=all_new.branch
all_new["id_group"]="RECURRENT_RESET_PRICE_CORE"
assert all_new.refid.is_unique
def replay(orders):
 sort=orders.sort_values(["entry_dt","priority","score1","score2","code","refid"],
       ascending=[True,True,False,False,True,True])
 live=[];accept=[];reject=[]
 for x in sort.to_dict("records"):
  live=[v for v in live if v["exit_dt"]>x["entry_dt"]]
  same=[v for v in live if v["code"]==x["code"]]
  intended=float(x["weight"])
  if x["source"]!="R50" and same:
   x["reason"]="same_ticker_occupied";reject.append(x);continue
  if x["source"]=="R50" and same:
   if any(v["source"]=="R50" for v in same):
    x["reason"]="R50_original_same_ticker";reject.append(x);continue
   x["weight"]=max(0.,min(float(x["weight"]),
     (.125 if x["method"]=="A" else .05)-sum(float(v["weight"]) for v in same)))
   if x["weight"]<.001:
    x["reason"]="legacy_capacity_full";reject.append(x);continue
  cap=.125 if x["method"]=="A" or any(v["method"]=="A" for v in same) else .05
  if sum(v["weight"] for v in same)+x["weight"]>cap+1e-8:
   x["reason"]="per_ticker_cap";reject.append(x);continue
  tickers={v["code"] for v in live}|{x["code"]}
  gross=sum(v["weight"] for v in live)
  bsum=sum(v["weight"] for v in live if v["method"]!="A")
  reason=None
  if len(tickers)>5:reason="max_5_stocks"
  elif gross+x["weight"]>.55+1e-8:reason="max_gross55"
  elif x["method"]!="A" and bsum+x["weight"]>.15+1e-8:reason="max_B15"
  elif x["source"]!="R50" and (len(tickers)>4 or gross+x["weight"]>.425+1e-8):
   reason="reserve_original_A_headroom"
  if reason:
   x["reason"]=reason;reject.append(x);continue
  x["original_weight"]=intended;x["reduced_weight"]=intended-x["weight"]
  accept.append(x);live.append(x)
 return pd.DataFrame(accept),pd.DataFrame(reject)
# Original baseline has the same R60+R79 order timing/risk as existing R79.
B,BR=replay(pd.concat([oldsrc,prev],ignore_index=True))
comp=float((B.ret*B.weight).sum()*100)
print("R81_BASELINE_REPLAY",len(B),"source",B.source.value_counts().to_dict(),
      "contrib_proxy_pp",comp,"rejected",len(BR),flush=True)
expected=57.01407796050304
# Fail closed on any baseline drift, rather than comparing incomparable accounts.
if abs(comp-expected)>.02 or len(B)!=88 or len(B[B.source=="R50"])!=55:
 diag={"status":"INCOMPARABLE_BASELINE_STOP","expected_profit_proxy_pp":expected,
       "observed_profit_proxy_pp":comp,"observed_count":len(B),
       "observed_source_counts":B.source.value_counts().to_dict()}
 (O/"R81_account_baseline_failure.json").write_text(json.dumps(diag,ensure_ascii=False,indent=2))
 raise SystemExit("R81_BASELINE_MISMATCH_"+json.dumps(diag,ensure_ascii=False))
oldaccepted=set(B.refid)
styles=[]
for g in ("BASE","P4","P6","STRICT_LIVE_CORE"):
 for b in ("BOTH","AM_REASSERT","PM_RECAPTURE"):
  for v in ("BASE_T3","T1_WEAK"):
   styles.append((g,b,v))
results=[];outacc=[];outrej=[]
for gate,branch,variant in styles:
 t=all_new[(all_new[gate])&(all_new.variant==variant)]
 if branch!="BOTH":t=t[t.branch==branch]
 t=t.sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"],keep="first")
 cols=B.columns.intersection(all_new.columns)
 new=t[cols].copy()
 merged=pd.concat([oldsrc,prev,new],ignore_index=True)
 acc,rej=replay(merged)
 newa=acc[acc.source=="R81"]
 olda=acc[acc.source!="R81"]
 closed=newa[~newa.censored];bad=newa[newa.mae.le(-.10)]
 skipped=B[~B.refid.isin(olda.refid)]
 unchanged=B[B.refid.isin(olda.refid)]
 # Exact baseline weighted-profit-proxy delta = newly accepted + old changes.
 total_profit=float((acc.ret*acc.weight).sum()*100)
 old_deficit=float((olda.ret*olda.weight).sum()*100-(B.ret*B.weight).sum()*100)
 incr=float((newa.ret*newa.weight).sum()*100)
 assert abs(total_profit-comp-(incr+old_deficit))<1e-8
 outcome={"gate":gate,"branch":branch,"exit_variant":variant,
   "precapacity_new_distinct_intents":len(new),
   "accepted_total":len(acc),"new_accepted":len(newa),"new_closed":len(closed),
   "new_winners":int(closed.ret.gt(0).sum()),
   "new_win_rate_pct":float(100*closed.ret.gt(0).mean()) if len(closed) else None,
   "new_avg_closed_pct":float(100*closed.ret.mean()) if len(closed) else None,
   "new_MAE10":len(bad),"new_loss10":int(closed.ret.le(-.10).sum()),
   "new_censored":int(newa.censored.sum()),
   "new_distinct_codes":newa.code.nunique(),
   "new_monthly_acceptance":newa.groupby(newa.entry_dt.dt.strftime("%Y-%m")).size().to_dict(),
   "old_R50_A_removed":int(((skipped.source=="R50")&(skipped.method=="A")).sum()),
   "old_R50_B_removed":int(((skipped.source=="R50")&(skipped.method=="B")).sum()),
   "old_R78_removed":int(skipped.source.eq("R78").sum()),
   "old_R79_removed":int(skipped.source.eq("R79").sum()),
   "old_R50_A_reduced":int(((olda.source=="R50")&(olda.method=="A")&olda.reduced_weight.gt(0.00001)).sum()),
   "new_weighted_profit_proxy_pp":incr,
   "old_displacement_proxy_pp":old_deficit,
   "total_profit_proxy_pp":total_profit,
   "delta_vs_R79_baseline_proxy_pp":total_profit-comp,
   "R50_original_orders_not_deleted_from_candidate_input":True,
   "profit_metric_not_live_NAV":True}
 results.append(outcome)
 tag=gate+"_"+branch+"_"+variant
 for r in newa.to_dict("records"):
  outacc.append({"tag":tag,"code":r["code"],"name":r["name"],"entry_dt":r["entry_dt"],
    "exit_dt":r["exit_dt"],"return_net_pct":r["ret"]*100,"MAE_pct":r["mae"]*100,
    "weight":r["weight"],"censored":r["censored"],"route":r["route"]})
 for r in rej[rej.source=="R81"].to_dict("records"):
  outrej.append({"tag":tag,"code":r["code"],"entry_dt":r["entry_dt"],"reason":r["reason"]})
print("R81_ACCOUNT_SCENARIOS",len(results),flush=True)
res=pd.DataFrame(results).sort_values("delta_vs_R79_baseline_proxy_pp",ascending=False)
res.to_csv(O/"R81_24_ACCOUNT_OVERLAY_SCENARIOS.csv",index=False)
pd.DataFrame(outacc).to_csv(O/"R81_accepted_new_trade_details.csv",index=False)
pd.DataFrame(outrej).to_csv(O/"R81_rejected_new_opportunities.csv",index=False)
B.to_csv(O/"R81_frozen_old_baseline_replayed.csv",index=False)
q={"status":"R81_JOINT_CAPACITY_PROXY_REPLAY_PASS",
"baseline_trades":len(B),"baseline_R50":int(B.source.eq("R50").sum()),
"baseline_R78":int(B.source.eq("R78").sum()),"baseline_R79":int(B.source.eq("R79").sum()),
"baseline_proxy_pp":comp,"baseline_exact_R79_parity":abs(comp-expected)<=.02,
"counterfactual_scenarios":len(res),"all_original_R50_orders_in_input":len(oldsrc),
"R79_accepted_not_future_generated":len(prev),
"R81_event_signals_not_same_stock_same_day_double_counted":True,
"original_A_priority_by_order_clock_preserved":True,
"no_T1_intraday_sale":True,"same_account_gross55_B15_max5_perstock":True,
"realistic_100_share_and_1pct_minute_participation_is_previous_study_verified":True,
"FULL_CASH_NAV_REPLAY":False,"ex_dividend_and_broker_queue_verified":False,
"source_PIT_curated_incomplete":True,"not_deployable":True}
(O/"account_qa.json").write_text(json.dumps(q,ensure_ascii=False,indent=2))
print("R81_ACCOUNT_QA",json.dumps(q,ensure_ascii=False),flush=True)
print("R81_ACCOUNT_TOP_SCENARIOS",res.head(20).drop(columns="new_monthly_acceptance").round(4).to_string(index=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]