#!/usr/bin/env python3
"""R81 independent as-of trade replay: target most favorable *posthoc* R21 group.
Validation check: reproduce original 0.2% fees/PnL/T1 exit/MAE/100share/1% amount.
Adversarial: slip 0.4%,0.6%, entry delayed an extra minute; new sleeve stressed,
frozen R50/R79 baseline left at original historical prices (NOT symmetric stress).
"""
from pathlib import Path
import runpy,json,math
import numpy as np,pandas as pd,pyarrow.dataset as ds
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/"w05_independent_audit"
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
H=Path.home()/"Documents/stock_data_windows_mirror"
acc=runpy.run_path(str(ROOT/"w04_portfolio/replay_account.py"))
replay=acc["replay"];oldsrc=acc["oldsrc"];prev=acc["prev"];B=acc["B"];baseline=acc["comp"];newall=acc["all_new"]
members=pd.read_csv(ROOT/"w07_R21_core_memory/R81_asof_R21_lifecycle_membership_each_signal.csv",dtype={"code":str})
eligible=members[members.CORE10&members.LOOP1&
 (members.branch=="AM_REASSERT")&(members.variant=="T1_WEAK")].copy()
assert len(eligible)==15,(len(eligible),eligible.to_dict("records"))
raw=newall.merge(eligible[["code","buy_date","entry_time","branch","variant"]],
  on=["code","buy_date","entry_time","branch","variant"],how="inner",validate="1:1")
assert len(raw)==15
mother=pd.read_csv(ROOT/"w02_population/R81_fullmarket_reset_episodes.csv",dtype={"code":str})
reference=mother[["code","date","close"]].rename(columns={"date":"reset_date","close":"prior_close"})
raw=raw.merge(reference,on=["code","reset_date"],how="left",validate="m:1")
assert raw.prior_close.notna().all()
# Re-read physical actual 1min with real transacted amounts.
all_sessions=sorted(p.parent.name[5:] for p in (H/"daily/tdx_parquet").glob("date=*/daily_ohlcv.parquet") if "2026-01-01"<=p.parent.name[5:]<="2026-09-23")
dateidx={d:i for i,d in enumerate(all_sessions)}
cache={}
def bars(day,code):
 k=(day,code)
 if k in cache:return cache[k]
 p=H/"1min"/("date="+day)
 if not p.exists():cache[k]=None;return None
 x=ds.dataset(str(p),format="parquet").to_table(columns=["code","date","time","open","high","low","close","volume","amount"],
    filter=ds.field("code")==code).to_pandas().sort_values("time").reset_index(drop=True)
 if len(x)!=240 or x.amount.isna().any() or x.time.duplicated().any():
  cache[k]=None;return None
 cache[k]=x;return x
def run_one(row,slip=.002,extra_bar=0):
 day=row.buy_date;c=row.code
 a=bars(day,c)
 assert a is not None,(day,c)
 baseindex=int(a.index[a.time.eq(row.entry_time)][0])
 # entry from W03 at next bar, or explicit delayed extra completed bar
 sigidx=int(a.index[a.time.eq(row.signal_time)][0])
 assert baseindex==sigidx+1,(day,c)
 i=baseindex+extra_bar
 if i>=len(a):return dict(reason="NO_BAR")
 f=a.iloc[i]
 # A limit-up sealed next minute cannot be bought as full fill.
 if float(f.open)>=float(row.prior_close)*1.0995 or float(f.volume)<=0 or float(f.amount)<=0:
  return dict(reason="ONEWORD_OR_ZERO")
 cost=float(f.open)*(1+slip+.00031)
 shares=int(25000/(100*cost))*100
 if shares<100 or shares*cost>float(f.amount)*.01:return dict(reason="PARTICIPATION_OR_LOT")
 heldmin=float(a.iloc[i:].low.min());status="RIGHT_CENSORED"
 data={"entry_date":day,"entry_time":f.time,"entry_gross":float(f.open),"entry_cost":cost,
       "shares":shares,"size":cost*shares/1e6,"signal_time":row.signal_time,
       "source_buy_amount":float(f.amount),"source_buy_volume":float(f.volume)}
 start=dateidx[day];ex=None
 for z in range(start+1,min(start+4,len(all_sessions))):
  date=all_sessions[z]
  q=bars(date,c);prevq=bars(all_sessions[z-1],c)
  if q is None or prevq is None:break
  prevclose=float(prevq.iloc[-1].close)
  limitlow=round(prevclose*.90+1e-8,2)
  for j in range(len(q)-1):
   b=q.iloc[j];sub=q.iloc[j+1]
   heldmin=min(heldmin,float(b.low))
   pnl=float(b.close)/cost-1
   reason=None
   if pnl>=.05:reason="TP5"
   elif pnl<=-.08:reason="T1_STOP8"
   elif z==start+1 and str(b.time)>="10:15" and pnl<=0:reason="T1_NOT_RISING"
   elif z-start>=3 and str(b.time)>="14:45":reason="T3_TIMEOUT"
   if not reason:continue
   gross=float(sub.open)
   if gross<=limitlow+.005 or float(sub.amount)<=0 or float(sub.volume)<=0:continue
   if shares*gross>float(sub.amount)*.01:continue
   heldmin=min(heldmin,gross)
   net=gross*(1-slip-.00081)
   ex={"exit_date":date,"exit_time":str(sub.time),
       "exit_gross":gross,"exit_net":net,
       "ret":net/cost-1,"why":reason}
   break
  if ex:break
  heldmin=min(heldmin,float(q.iloc[-1].low))
 data.update(ex if ex else {"exit_date":None,"exit_time":None,"ret":0., "why":"CENSORED"})
 data["censored"]=ex is None;data["MAE_pct"]=100*(heldmin/cost-1)
 data["reason"]="OK"
 return data
allruns=[];reports=[]
for name,slip,delay in [("BASE_0P2",.002,0),("SLIP_0P4",.004,0),
   ("SLIP_0P6",.006,0),("DELAY_ONE_EXTRA_MIN",.002,1)]:
 records=[]
 for z in raw.itertuples(index=False):
  result=run_one(z,slip,delay)
  records.append(dict(code=z.code,buy_date=z.buy_date,branch=z.branch,scenario=name,
     orig_ret_pct=z.return_net_pct,orig_entry_time=z.entry_time,orig_exit_date=z.exit_date,orig_exit_time=z.exit_time,
     orig_MAE_pct=z.MAE_pct,**result))
 df=pd.DataFrame(records)
 allruns.append(df)
 # Independently verify the original actual 0.2% event-time/fill record.
 if name=="BASE_0P2":
  assert df.reason.eq("OK").all(),df[df.reason!="OK"].to_dict("records")
  for a in df.to_dict("records"):
   orig=raw[(raw.code==a["code"])&(raw.buy_date==a["buy_date"])&(raw.entry_time==a["entry_time"])]
   assert len(orig)==1
   r=orig.iloc[0]
   assert abs(float(a["entry_cost"])-float(r.entry_cost))<1e-9,(a["code"],"BUY_COST")
   assert a["exit_date"]==r.exit_date and a["exit_time"]==r.exit_time,(a["code"],"SELL_CLOCK",a["exit_time"],r.exit_time)
   assert abs(100*float(a["ret"])-float(r.return_net_pct))<1e-8,(a["code"],"NET_RETURN")
   assert abs(float(a["MAE_pct"])-float(r.MAE_pct))<1e-7,(a["code"],"MAE")
  print("W05_INDEPENDENT_15_OF_15_ENTRY_EXIT_MAE_PARITY_PASS",flush=True)
 original=raw.copy()
 # Each scenario gets new gross costs, 100-share rounded budget, weight.
 for row in records:
  mask=(original.code==row["code"])&(original.buy_date==row["buy_date"])&(original.branch==row["branch"])
  assert mask.sum()==1
  if row.get("reason")!="OK":
   original=original[~mask]
   continue
  original.loc[mask,"entry_dt"]=pd.Timestamp(row["entry_date"]+" "+str(row["entry_time"]))
  original.loc[mask,"exit_dt"]=pd.Timestamp(row["exit_date"]+" "+str(row["exit_time"])) if row["exit_date"] else pd.Timestamp("2026-09-23 15:00:00")
  original.loc[mask,"ret"]=row["ret"];original.loc[mask,"weight"]=row["size"]
  original.loc[mask,"censored"]=row["censored"];original.loc[mask,"mae"]=row["MAE_pct"]/100
 accepted,rejected=replay(pd.concat([oldsrc,prev,original[B.columns.intersection(original.columns)]],ignore_index=True))
 anew=accepted[accepted.source=="R81"];oc=accepted[accepted.source!="R81"]
 lost=B[~B.refid.isin(oc.refid)]
 profit=float((accepted.weight*accepted.ret).sum()*100)
 reports.append({"scenario":name,"slip_each_side":slip,
   "extra_minute_delay":delay,"pre_capacity_ok":len(original),
   "accepted_new":len(anew),"closed":int((~anew.censored).sum()),
   "winners":int(anew[~anew.censored].ret.gt(0).sum()),
   "average_net_pct":100*anew[~anew.censored].ret.mean() if len(anew[~anew.censored]) else None,
   "observed_MAE10":int(anew.mae.le(-.10).sum()),
   "closed_loss10":int(anew[~anew.censored].ret.le(-.10).sum()),
   "lost_orig_A":int(((lost.source=="R50")&(lost.method=="A")).sum()),
   "lost_orig_B":int(((lost.source=="R50")&(lost.method=="B")).sum()),
   "old_R78_R79_displacement":int(lost.source.isin(["R78","R79"]).sum()),
   "combined_proxy_pp":profit,"delta_to_frozen_R79_proxy_pp":profit-baseline,
   "unknown_market_order_queue":True})
 print("W05_COST_AUDIT",reports[-1],flush=True)
pd.concat(allruns,ignore_index=True).to_csv(OUT/"R81_R21_LOOP1_15_signal_independent_per_slip.csv",index=False)
pd.DataFrame(reports).to_csv(OUT/"R81_R21_LOOP1_cost_and_delay_stress.csv",index=False)
QA={"status":"INDEPENDENT_REPLAY_PASS",
  "total_independent_original_events":len(raw),"base_buy_clock_fee_exit_MAE_exact_matched":True,
  "pressure_scenarios":len(reports),
  "frozen_R79_baseline_proxy_pp":baseline,
  "only_new_R81_signals_stressed":True,
  "not_symmetric_old_R50_sensitivity":True,
  "no_future_signal_ohlc":True,"true_exchange_order_fill_unknown":True,
  "method_found_by_historical_multiple_scenario_search":True,
  "official_daily_limit_prices_missing":True,
  "no_prod_writes":True}
(OUT/"qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))
print("W05_FINAL_QA",json.dumps(QA,ensure_ascii=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]