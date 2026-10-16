#!/usr/bin/env python3
"""R81 W03, raw 1-minute full-market and next-session research, strict evidence.
Method already frozen in R81 RESEARCH_CONTRACT. Not product deployment.
"""
from pathlib import Path
from collections import defaultdict
import json,math
import pandas as pd,numpy as np,pyarrow.dataset as ds
R=Path(__file__).resolve().parent.parent
H=Path.home()/"Documents/stock_data_windows_mirror"
O=R/"w03_causal_signal"
mother=pd.read_csv(R/"w02_population/R81_fullmarket_reset_episodes.csv",dtype={"code":str})
assert len(mother)>1000
days=sorted(p.name[5:] for p in (H/"daily/tdx_parquet").iterdir()
            if p.is_dir() and p.name.startswith("date=") and "2026-01-01"<=p.name[5:]<="2026-09-23")
calendar={d:i for i,d in enumerate(days)}
# True source 1minute(amount) coverage starts Jan29 and is reliable Feb-Sep23.
ev=mother[mother.date.between("2026-02-01","2026-09-22")].copy()
ev["buy_date"]=ev.date.map(lambda x:days[calendar[x]+1] if x in calendar and calendar[x]+1<len(days) else None)
ev=ev[ev.buy_date.notna()].copy()
# For each T-1 reset, read only the same ticker's T and T+3 session minutes.
needed=defaultdict(set)
for z in ev.itertuples(index=False):
 i=calendar[z.buy_date]
 for day in days[i:min(i+4,len(days))]:needed[day].add(z.code)
print("R81_W03_SOURCE_DAYS",len(needed),"RESET_EVENTS",len(ev),flush=True)
cache={}
qlist=[]
def gdata(d,codes):
 path=H/"1min"/("date="+d)
 if not path.exists():return {}
 x=ds.dataset(str(path),format="parquet").to_table(
      columns=["code","date","time","open","high","low","close","volume","amount"],
      filter=ds.field("code").isin(sorted(codes))).to_pandas()
 if x.empty:return {}
 x["code"]=x.code.astype(str).str.zfill(6)
 return {str(c):z.sort_values("time").reset_index(drop=True) for c,z in x.groupby("code")}
for n,(day,codes) in enumerate(sorted(needed.items())):
 got=gdata(day,codes)
 for c in codes:
  z=got.get(c)
  if z is None:qlist.append({"date":day,"code":c,"reason":"NO_TICKER_MINUTES"});continue
  if len(z)!=240 or z.time.duplicated().any() or z.time.iloc[0]!="09:31":
   qlist.append({"date":day,"code":c,"reason":"INCOMPLETE_TAPE"});continue
  if z.amount.isna().any() or z.volume.isna().any() or z.volume.lt(0).any():
   qlist.append({"date":day,"code":c,"reason":"MISSING_ACTUAL_MINUTE_AMOUNT"});continue
  # Historical aggregated minute amount actually uses shares, verified unit dimension.
  ratio=z.loc[z.volume.gt(0),"amount"].sum()/max(z.loc[z.volume.gt(0),"volume"].sum(),1)
  if not(z.close.min()*.70<ratio<z.close.max()*1.30):
   qlist.append({"date":day,"code":c,"reason":"PRICE_VOLUME_AMOUNT_UNIT_MISMATCH"});continue
  cache[(day,c)]=z
  qlist.append({"date":day,"code":c,"reason":"VERIFIED_240MIN_AMOUNT"})
 if (n+1)%25==0:print("R81_W03_STAGED",day,n+1,len(cache),flush=True)
Q=pd.DataFrame(qlist)
Q.to_csv(O/"source_verification.csv",index=False)
print("SOURCE_QUALITY",Q.reason.value_counts().to_dict(),flush=True)
# Signals use complete bars only; AM and PM are exclusive if an earlier route has entered.
def scan(x,prev,which):
 vol=0.;amt=0.;low=float("inf");high=0.
 hit=[]
 for i,row in enumerate(x.itertuples(index=False)):
  tm=str(row.time);low=min(low,float(row.low));high=max(high,float(row.high))
  vol+=float(row.volume);amt+=float(row.amount)
  if i<25 or i+1>=len(x):continue
  if which=="AM_REASSERT" and not("09:46"<=tm<="11:00"):continue
  if which=="PM_RECAPTURE" and not("13:20"<=tm<="14:25"):continue
  price=float(row.close);clv=(price-low)/(high-low) if high>low else .5
  if price<prev*1.02:continue
  last2=x.iloc[i-1:i+1].close
  if not bool(last2.ge(prev).all()):continue
  if which=="AM_REASSERT":
   cond=price>float(x.iloc[i-10:i].close.max()) and clv>=.60
  else:
   amlow=float(x.loc[x.time.le("11:30"),"low"].min())
   cond=(amlow<=prev and price>float(x.iloc[i-20:i].close.max()) and clv>=.65
       and price>=amt/max(vol,1))
  if not cond:continue
  following=x.iloc[i+1]
  if float(following.open)>=prev*1.0995:continue
  if float(following.volume)<=0 or float(following.amount)<=0:continue
  entry=float(following.open)*(1+.002+.00031)
  shares=100*int(25000/(entry*100))
  if shares<100:continue
  filled=shares*entry
  if filled>float(following.amount)*.01:continue
  return {"signal_time":tm,"signal_price":price,"clv":clv,
          "signal_vwap":amt/max(vol,1),"entry_time":str(following.time),
          "entry_open":float(following.open),"entry_cost":entry,
          "entry_size":filled,"shares":shares,"entry_bar_amount":float(following.amount)}
 return None
raw=[]
for j,z in enumerate(ev.itertuples(index=False)):
 x=cache.get((z.buy_date,z.code))
 if x is None:continue
 for branch in ("AM_REASSERT","PM_RECAPTURE"):
  s=scan(x,float(z.close),branch)
  if s:
   a={"code":z.code,"name":z.name,"reset_date":z.date,"buy_date":z.buy_date,
      "reset_type":z.reset_type,"leader_proxy":z.leader_type,
      "streak_prior20_max":int(z.streak_prior20_max),
      "since_board":int(z.since_board),"reset_pct":float(z.max_drop_pct),
      "prior_breadth":float(z.prior_breadth) if pd.notna(z.prior_breadth) else np.nan,
      "branch":branch}|s
   raw.append(a)
 if (j+1)%300==0:print("R81_W03_SCAN_PROGRESS",j+1,"signals",len(raw),flush=True)
S=pd.DataFrame(raw)
assert len(S), "No causal signals"
assert S[["code","buy_date","branch"]].duplicated().sum()==0
S.sort_values(["buy_date","entry_time","code","branch"],inplace=True)
S.to_csv(O/"raw_r81_2clock_signals.csv",index=False)
# Causal T+1 exit variants, no hindsight. Capture no-fill and all right-censor events.
def simulate(pos,variant):
 cost=pos.entry_cost;d=pos.buy_date;c=pos.code
 b=cache[(d,c)];buyidx=int(b.index[b.time.eq(pos.entry_time)][0])
 held_minlow=float(b.iloc[buyidx:].low.min())
 start=calendar[d];bad_days=[];sell=None
 for k in range(start+1,min(start+4,len(days))):
  day=days[k];q=cache.get((day,c))
  prior_day_bar=cache.get((days[k-1],c))
  if q is None or prior_day_bar is None:bad_days.append(day);break
  # Per-session price-limit reference. Never freeze the entry day's prior close.
  actual_prior_close=float(prior_day_bar.iloc[-1].close)
  estimated_lower=round(actual_prior_close*.90+1e-8,2)
  can_weak=(k==start+1)
  for i in range(len(q)-1):
   row=q.iloc[i];following=q.iloc[i+1]
   held_minlow=min(held_minlow,float(row.low))
   pct=float(row.close)/cost-1
   why=""
   if pct>=.05:why="TP5"
   elif pct<=-.08:why="T1_STOP8"
   elif variant=="T1_WEAK" and can_weak and str(row.time)>="10:15" and pct<=.00:why="T1_NOT_RISING"
   elif k-start>=3 and str(row.time)>="14:45":why="T3_TIMEOUT"
   if not why:continue
   # All fills at the NEXT completed minute, no ideal intraminute price assumptions.
   sellopen=float(following.open)
   if float(following.amount)<=0 or float(following.volume)<=0:continue
   if sellopen<=estimated_lower+.005:
    continue
   if pos.shares*sellopen>float(following.amount)*.01:continue
   held_minlow=min(held_minlow,sellopen)
   sell={"exit_date":day,"exit_time":str(following.time),"exit_price_gross":sellopen,
     "exit_price_net":sellopen*(1-.002-.00081),"exit_reason":why}
   break
  if sell:break
  held_minlow=min(held_minlow,float(q.iloc[-1].low))
 state={"code":c,"name":pos.name,"reset_date":pos.reset_date,
        "buy_date":d,"entry_time":pos.entry_time,"entry_price_gross":pos.entry_open,
        "entry_cost":cost,"shares":int(pos.shares),"entry_weight":pos.entry_size/1e6,
        "branch":pos.branch,"reset_type":pos.reset_type,
        "leader_streak":pos.streak_prior20_max,"prior_breadth":pos.prior_breadth,
        "reset_pct":pos.reset_pct,"variant":variant,
        "MAE_pct":100*(held_minlow/cost-1),"status":"CLOSED" if sell else "RIGHT_CENSORED",
        "missing_exit_days":",".join(bad_days)}
 if sell:
  state.update(sell);state["return_net_pct"]=100*(sell["exit_price_net"]/cost-1)
 return state
res=[]
for i,rec in enumerate(S.itertuples(index=False)):
 for variant in ("T1_WEAK","BASE_T3"):
  res.append(simulate(rec,variant))
 if (i+1)%90==0:print("R81_W03_EXIT_PROGRESS",i+1,flush=True)
T=pd.DataFrame(res)
T.to_csv(O/"raw_r81_independent_exits.csv",index=False)
G=T[T.status=="CLOSED"].groupby(["branch","variant"]).agg(
 count=("return_net_pct","size"),win=("return_net_pct",lambda s:int(s.gt(0).sum())),
 netmean=("return_net_pct","mean"),mae10=("MAE_pct",lambda s:int(s.le(-10).sum())),
 loss10=("return_net_pct",lambda s:int(s.le(-10).sum()))).reset_index()
G.to_csv(O/"raw_r81_branch_comparison.csv",index=False)
meta={"status":"W03_COMPLETED_NOT_PROMOTED","mother_reset_events":len(ev),
 "quality_verified_stockdays":int(Q.reason.eq("VERIFIED_240MIN_AMOUNT").sum()),
 "quality_rejected_stockdays":int(Q.reason.ne("VERIFIED_240MIN_AMOUNT").sum()),
 "raw_signal_events":len(S),"raw_closed_exit_scenario_records":int(T.status.eq("CLOSED").sum()),
 "raw_censored_scenario_records":int(T.status.eq("RIGHT_CENSORED").sum()),
 "signals_are_not_account_trades":True,"no_old_A_B_overlay":True,
 "before2026_09_23_real_minute_amount_only":True,
 "historical_signal_identity_price_proxy_not_true_theme_cert":True,
 "no_production_touch":True}
(O/"qa.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print("R81_W03_DONE",json.dumps(meta,ensure_ascii=False),flush=True)
print("R81_W03_SUMMARY",G.round(4).to_string(index=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]