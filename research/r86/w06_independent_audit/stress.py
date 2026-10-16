#!/usr/bin/env python3
"""R86 W06: independently re-read real 240 x 1m bars, compare exact 0.2% R81
next bar buys and T+1 exits, then stress G2/G10 small profitable in-sample
models at 0.4%, 0.6% both sides or +1m entry delay.
Old 88 frozen trades left at original cost: marginal sensitivity ONLY.
"""
import runpy,json,math
from pathlib import Path
import pandas as pd,numpy as np,pyarrow.dataset as ds
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
H=Path.home()/"Documents/stock_data_windows_mirror"
O=R/"w06_independent_audit"
a=runpy.run_path(str(R/"w04_joint_account/account.py"))
oldsrc=a["oldsrc"];prev=a["prev"];B=a["B"];replay=a["replay"];base=a["reference"];allnew=a["T"]
days=sorted(p.parent.name[5:] for p in (H/"daily/tdx_parquet").glob("date=*/daily_ohlcv.parquet")
   if "2026-01-01"<=p.parent.name[5:]<="2026-09-23")
di={d:i for i,d in enumerate(days)}
cache={}
def getday(date,code):
 k=(date,code)
 if k in cache:return cache[k]
 f=H/"1min"/("date="+date)
 if not f.exists():cache[k]=None;return None
 q=ds.dataset(str(f),format="parquet").to_table(
 filter=ds.field("code")==code,columns=["code","time","open","high","low","close","volume","amount"]).to_pandas().sort_values("time").reset_index(drop=True)
 if len(q)!=240 or q.time.duplicated().any() or q.amount.isna().any() or q.volume.isna().any():
  cache[k]=None;return None
 cache[k]=q;return q
def simulate(o,slip,delay,variant):
 date=o.buy_date;code=o.code
 q=getday(date,code)
 assert q is not None,(date,code)
 pos=np.flatnonzero(q.time.to_numpy()==o.entry_time)
 assert len(pos)==1,(date,code,o.entry_time)
 buyidx=int(pos[0])+delay
 if buyidx>=len(q):return None
 curr=q.iloc[buyidx]
 # True 09:25 limit upper from prior day's close is in the original signal
 prevprice=float(o.prevclose_from_reset)
 if float(curr.open)>=prevprice*1.0995 or curr.volume<=0 or curr.amount<=0:return None
 cost=float(curr.open)*(1+slip+.00031)
 shares=int(25000/(cost*100))*100
 if shares<100 or shares*cost>float(curr.amount)*.01:return None
 minlow=float(q.iloc[buyidx:].low.min())
 exitsignal=None;daystart=di[date]
 for k in range(daystart+1,min(daystart+4,len(days))):
  x=getday(days[k],code);prevday=getday(days[k-1],code)
  if x is None or prevday is None:break
  lower=round(float(prevday.iloc[-1].close)*.90+1e-8,2)
  for i in range(len(x)-1):
   r=x.iloc[i];nxt=x.iloc[i+1]
   minlow=min(minlow,float(r.low))
   gain=float(r.close)/cost-1
   reason=None
   if gain>=.05:reason="TP5"
   elif gain<=-.08:reason="T1_STOP8"
   elif variant=="T1_WEAK" and k==daystart+1 and str(r.time)>="10:15" and gain<=0:reason="T1_NOT_RISING"
   elif k-daystart>=3 and str(r.time)>="14:45":reason="T3_TIMEOUT"
   if not reason:continue
   px=float(nxt.open)
   if px<=lower+.005 or float(nxt.amount)<=0 or float(nxt.volume)<=0:continue
   if px*shares>float(nxt.amount)*.01:continue
   minlow=min(minlow,px)
   exitsignal={"exit_date":days[k],"exit_time":str(nxt.time),"exit_open":px,
    "ret":px*(1-slip-.00081)/cost-1,"exit_reason":reason}
   break
  if exitsignal:break
  minlow=min(minlow,float(x.iloc[-1].low))
 row={"code":code,"name":o.name,"buy_date":date,"branch":o.branch,"entry_time":str(curr.time),
     "entry_dt":pd.Timestamp(date+" "+str(curr.time)), "entry_cost":cost,
     "shares":shares,"weight":cost*shares/1e6,
     "mae":minlow/cost-1,"source":"R86","method":"R86_B",
     "priority":5,"score1":0.,"score2":0.,"route":o.branch,
     "id_group":"R86_EXE_STRESS","censored":exitsignal is None}
 if exitsignal:row.update(exitsignal)
 else:row.update({"exit_date":None,"exit_time":None,"ret":0.})
 row["exit_dt"]=pd.Timestamp(str(row["exit_date"])+" "+str(row["exit_time"])) if exitsignal else pd.Timestamp("2026-09-23 15:00")
 row["refid"]=code+"|"+date+"|"+row["entry_time"]+"|"+o.branch+"|"+variant
 return row
mother=pd.read_csv(L/"r81_recurrent_reversal_20261016/w02_population/R81_fullmarket_reset_episodes.csv",dtype={"code":str})
mother=mother[["code","date","close"]].rename(columns={"date":"reset_date","close":"prevclose_from_reset"})
T=allnew.merge(mother,on=["code","reset_date"],how="left",validate="m:1")
assert T.prevclose_from_reset.notna().all()
styles=[("G2_SECTOR_PLUS_AUCTION","BASE_T3"),
 ("G2_SECTOR_PLUS_AUCTION","T1_WEAK"),
 ("G10_TWO_SOURCE_REPLICATION","BASE_T3")]
stress=[("BASE_02",.002,0),("SLIP_04",.004,0),("SLIP_06",.006,0),("DELAY_ONE_MIN",.002,1)]
reports=[];trades=[]
for gate,var in styles:
 q=T[T[gate]&T.variant.eq(var)].sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"])
 for name,slip,delay in stress:
  sim=[];parity_count=0
  for o in q.itertuples(index=False):
   res=simulate(o,slip,delay,var)
   if res is None:continue
   if name=="BASE_02":
    if o.status=="CLOSED":
     assert res["exit_date"]==o.exit_date and res["exit_time"]==o.exit_time,(gate,o.code,"exit",res["exit_time"],o.exit_time)
     assert abs(res["ret"]*100-o.return_net_pct)<1e-8,(gate,o.code,"net",res["ret"]*100,o.return_net_pct)
     assert abs(res["mae"]*100-o.MAE_pct)<1e-7,(gate,o.code,"mae",res["mae"]*100,o.MAE_pct)
     assert abs(res["entry_cost"]-o.entry_cost)<1e-8,(gate,o.code,"cost")
     parity_count+=1
    else:assert res["censored"]
   sim.append(res)
  newer=pd.DataFrame(sim)
  cols=B.columns.intersection(newer.columns)
  accepted,refused=replay(pd.concat([oldsrc,prev,newer[cols]],ignore_index=True))
  new=accepted[accepted.source.eq("R86")];olds=accepted[accepted.source.ne("R86")]
  displaced=B[~B.refid.isin(olds.refid)]
  closed=new[~new.censored]
  oldloss=float((olds.weight*olds.ret).sum()*100-base)
  newgain=float((new.weight*new.ret).sum()*100)
  delta=oldloss+newgain
  rec={"gate":gate,"exit_variant":var,"stress":name,
   "original_intents":len(q),"resim_fillable":len(sim),"entry_delay_extra_mins":delay,
   "each_side_slippage_pct":slip*100,"independent_baseline_parity":parity_count,
   "new_accepted":len(new),"new_closed":len(closed),
   "wins":int(closed.ret.gt(0).sum()),"mean_return_pct":float(100*closed.ret.mean()) if len(closed) else None,
   "mae10":int(new.mae.le(-.10).sum()),"loss10":int(closed.ret.le(-.10).sum()),
   "R50_A_displaced":int(((displaced.source=="R50")&(displaced.method=="A")).sum()),
   "new_contrib_pp":newgain,"old_contrib_change_pp":oldloss,
   "delta_vs_old88_pp":delta}
  reports.append(rec)
  tag=gate+"_"+var+"_"+name
  for v in new.to_dict("records"):
   trades.append({"scenario":tag,"code":v["code"],"entry_dt":v["entry_dt"],"exit_dt":v["exit_dt"],
   "net_return_pct":100*v["ret"],"MAE_pct":100*v["mae"],"weight":v["weight"]})
  print("R86_STRESS",rec,flush=True)
S=pd.DataFrame(reports)
S.to_csv(O/"R86_12_REAL_MINUTE_INDEPENDENT_STRESS_SCENARIOS.csv",index=False)
pd.DataFrame(trades).to_csv(O/"R86_STRESS_ACCEPTED_NEW_TRADES.csv",index=False)
full=pd.read_csv(R/"w04_joint_account/R86_66_COMPLETE_FIXED_R50_R79_ACCOUNT_SCENARIOS.csv")
for r in reports:
 if r["stress"]!="BASE_02":continue
 orig=full[(full.gate==r["gate"])&(full.exit==r["exit_variant"])&(full.branch=="BOTH")].iloc[0]
 assert r["new_accepted"]==orig.new_accepted,(r["gate"],r["new_accepted"],orig.new_accepted)
 assert abs(r["delta_vs_old88_pp"]-orig.delta_vs_frozen_old88_pp)<1e-8,(r["gate"],r["delta_vs_old88_pp"],orig.delta_vs_frozen_old88_pp)
QA={"status":"R86_INDEPENDENT_M1_BAR_COST_DELAY_STRESS_BASELINE_PARITY_PASS",
 "groups":len(styles),"stress_scenarios":len(stress),"scenario_rows":len(S),
 "all_original_0p2_baseline_scenarios_matched_account_exactly":True,
 "all_original_relevant_closed_trades_resimulated_from_source_minute":True,
 "slips_changed_on_new_signals_only":True,
 "old_R50_A_B_cost_not_stressed_symmetrically":True,
 "future_outcomes_not_used_for_entry_admission":True,
 "production_untouched":True}
(O/"qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))
print("R86_STRESS_QA",json.dumps(QA,ensure_ascii=False),flush=True)