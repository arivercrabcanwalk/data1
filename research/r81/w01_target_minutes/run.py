#!/usr/bin/env python3
"""R81 W01: 601811 3 episode causal completed-minute, next-minute, T+1 research.
Exact observed market data. Partial data => label PRICE-ONLY, not executable truth.
"""
import json,math
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.dataset as ds
H=Path.home()/"Documents/stock_data_windows_mirror"
ROOT=Path(__file__).resolve().parent.parent
O=ROOT/"w01_target_minutes"
TARGET="601811"
RESET_DAY={"2026-09-29":"2026-09-28","2026-10-09":"2026-10-08","2026-10-14":"2026-10-13"}
NEXT_DAY={"2026-09-29":"2026-09-30","2026-10-09":"2026-10-12","2026-10-14":"2026-10-15"}
def m1(day):
 p=H/"1min"/("date="+day)
 assert p.exists(),p
 x=ds.dataset(str(p),format="parquet").to_table(filter=ds.field("code")==TARGET,
         columns=["code","date","time","open","high","low","close","volume","amount"]).to_pandas().sort_values("time").reset_index(drop=True)
 assert len(x)==240 and x.time.is_unique and x.time.iloc[0]=="09:31" and x.time.iloc[-1]=="15:00",(day,len(x))
 assert x.volume.ge(0).all() and x.open.gt(0).all()
 return x
def d1(day):
 p=H/"daily/tdx_parquet"/("date="+day)/"daily_ohlcv.parquet"
 x=pd.read_parquet(p,columns=["code","date","name","open","high","low","close","volume","amount"])
 z=x[x.code.astype(str).str.zfill(6)==TARGET]
 assert len(z)==1
 return z.iloc[0]
def profile(x,j):
 p=x.iloc[:j+1];cur=x.iloc[j];seenhi=float(p.high.max());seenlo=float(p.low.min())
 clv=(float(cur.close)-seenlo)/(seenhi-seenlo) if seenhi>seenlo else .5
 vwap=None
 if p.amount.notna().all() and p.volume.gt(0).any():
  vwap=float(p.amount.sum()/p.volume.sum())
 return {"time":str(cur.time),"close":float(cur.close),"clv":clv,"low_sofar":seenlo,
         "high_sofar":seenhi,"prefix_vwap":vwap,"volume_sofar":float(p.volume.sum()),
         "amount_available":vwap is not None}
def first_signal(x,prev,branch):
 for j in range(25,len(x)-1):
  t=str(x.iloc[j].time);p=profile(x,j);px=p["close"]
  above2=(px>=prev*1.02)
  h10=float(x.iloc[j-10:j].close.max())
  h20=float(x.iloc[j-20:j].close.max())
  last2=list(x.iloc[j-1:j+1].close)
  # All tests only use fully completed data at j, not today's final bar.
  cond2=(last2[0]>=prev and last2[1]>=prev)
  if branch=="AM_REASSERT":
   if not ("09:46"<=t<="11:00"):continue
   ok=above2 and px>h10 and cond2 and p["clv"]>=.60
  else:
   if not ("13:20"<=t<="14:25"):continue
   if float(x.loc[x.time.lt("11:30"),"low"].min())>prev:continue
   # Prefix-unavailable historical VWAP cannot certify this branch
   ok=above2 and px>h20 and cond2 and p["clv"]>=.65
   if p["prefix_vwap"] is not None:ok=ok and px>=p["prefix_vwap"]
  if not ok:continue
  fill=x.iloc[j+1]
  if float(fill.open)>=prev*1.0995 or float(fill.volume)<=0:continue
  cost=float(fill.open)*(1+.002+.00031)
  res={"code":TARGET,"branch":branch,"reset_day":RESET_DAY[str(x.iloc[0].date)],
    "signal_date":str(x.iloc[0].date),"signal_time":t,"signal_px":px,
    "signal_clv":round(p["clv"],5),"signal_vwap":p["prefix_vwap"],
    "signal_amt_available":p["amount_available"],
    "entry_time":str(fill.time),"entry_open":float(fill.open),
    "entry_cost":cost,
    "entry_volume_shares":float(fill.volume),
    "entry_min_amount":float(fill.amount) if pd.notna(fill.amount) else None,
    "entry_status":"REAL_AMOUNT_VERIFIED" if p["amount_available"] and pd.notna(fill.amount) else "PRICE_ONLY_UNVERIFIED"}
  return res
 return None
def exit_t1(sig,x,day):
 c=sig["entry_cost"]
 for j in range(len(x)-1):
  p=x.iloc[j]
  if float(p.close)/c-1<.05:continue
  f=x.iloc[j+1]
  price=float(f.open)*(1-.002-.00081)
  return {"exit_date":day,"exit_signal_time":str(p.time),"exit_time":str(f.time),
          "exit_open":float(f.open),"exit_net_px":price,"net_return_pct":100*(price/c-1),
          "exit_amount_available":bool(pd.notna(f.amount)),"exit_amt":float(f.amount) if pd.notna(f.amount) else None,
          "exit_volume_shares":float(f.volume)}
 return {"exit_date":day,"exit_signal_time":"NO_TP5","exit_time":"14:50",
         "exit_open":float(x[x.time.eq("14:50")].iloc[0].close),
         "exit_net_px":float(x[x.time.eq("14:50")].iloc[0].close)*(1-.002-.00081),
         "net_return_pct":100*(float(x[x.time.eq("14:50")].iloc[0].close)*(1-.002-.00081)/c-1),
         "exit_amount_available":False,"exit_amt":None}
rows=[];daily=[]
for date,reset in RESET_DAY.items():
 a=d1(reset);b=d1(date);nextd=NEXT_DAY[date];c=d1(nextd)
 dayret=100*(float(a.close)/float(d1({"2026-09-28":"2026-09-24","2026-10-08":"2026-09-30","2026-10-13":"2026-10-12"}[reset]).close)-1)
 assert dayret<=-5,(date,dayret)
 x=m1(date);z=m1(nextd)
 assert abs(float(x.close.iloc[-1])/float(b.close)-1)<1e-5
 minute_amt=float(x.amount.sum()) if x.amount.notna().all() else None
 volume_err=abs(float(x.volume.sum()/float(b.volume)-1))
 amount_err=abs(minute_amt/float(b.amount)-1) if minute_amt is not None else None
 daily.append({"reset":reset,"reset_pct":dayret,"signal":date,"signal_close":b.close,
   "signal_pct":100*(float(b.close)/float(a.close)-1),
   "next_date":nextd,"next_close":c.close,
   "intraminute_volume_err":volume_err,"intraminute_amount_err":amount_err,
   "minutes_amount_complete":minute_amt is not None})
 for branch in ("AM_REASSERT","PM_RECAPTURE"):
  signal=first_signal(x,float(a.close),branch)
  if not signal:rows.append({"code":TARGET,"signal_date":date,"branch":branch,"status":"NO_SIGNAL"});continue
  exitrow=exit_t1(signal,z,nextd)
  # adverse lowest since and including the first minute after entry, and up to actual exit minute
  now=x[x.time.ge(signal["entry_time"])]
  till=z[z.time.le(exitrow["exit_time"])]
  minlow=min(float(now.low.min()),float(till.low.min()))
  row=signal|exitrow|{"status":"PRICE_ONLY_PROXY" if not(signal["entry_status"]=="REAL_AMOUNT_VERIFIED" and exitrow["exit_amount_available"]) else "AMOUNT_FILLED",
     "observed_MAE_pct":100*(minlow/signal["entry_cost"]-1),"amount_daily_diff":amount_err,
     "buy_1pct_participation_verified":signal["entry_min_amount"] is not None and signal["entry_min_amount"]*.01>=25000.0}
  rows.append(row)
  print("R81_TARGET_SIGNAL",json.dumps(row,ensure_ascii=False,default=str),flush=True)
# The dangerous first 09:32 limit touch should NOT be counted as a quality confirmation.
early=m1("2026-10-09")
chase=float(early[early.time.eq("09:33")].iloc[0].open)
intraday_after=early[early.time.ge("09:33")]
danger=100*(intraday_after.low.min()/chase-1)
qa={"title":"601811 Three Real Reset Episodes","asof":"2026-10-16",
    "status":"CROSS_CHECKED_DAILY_AND_MINUTE_PRICE_ONLY",
    "reset_days":daily,
    "target_signal_count":sum(v.get("status")!="NO_SIGNAL" for v in rows),
    "real_amount_entry_count":sum(v.get("entry_status")=="REAL_AMOUNT_VERIFIED" for v in rows),
    "all_three_trigger_dates_all_amount":False,
    "10_09_early_touch_20_91_and_subsequent_day_drawdown_pct":danger,
    "data_provenance":"local_tdx_1min raw OHLCV and exact daily; no minute amount substitution",
    "no_hindsight_entries":True,"not_actual_order_fills":True,
    "new_rules_developed_after_observing_this_stock":True,
    "R50_original_not_changed":True}
pd.DataFrame(rows).to_csv(O/"R81_601811_causal_price_minute_casebook.csv",index=False)
(O/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2,default=str))
print("R81_TARGET_QA",json.dumps(qa,ensure_ascii=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]