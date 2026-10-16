#!/usr/bin/env python3
"""R77 independent three-branch second-buy signals and T+1 minute exits.
Development May-Jun; July-Aug already human-viewed, only descriptive checks.
A+B real account NOT modeled; no deployment/promotion from these results.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json,math
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=Path("r77_out");OUT.mkdir(exist_ok=True)
idx=pd.read_csv(OUT/"R77_asof_eligible_stockdays.csv",dtype={"code":str})
assert not idx.duplicated(["date","code"]).any()
idx=idx[idx.date.between("2026-04-01","2026-08-27")]
byday={day:g for day,g in idx.groupby("date")}
files=sorted(ROOT.glob("2026-*/part-*/date=*/minute1.parquet"))
files={p.parent.name[5:]:p for p in files}
dates=sorted(files)
master=pd.read_csv(OUT/"R77_asof_daily_master.csv",usecols=["code","date","close"],dtype={"code":str})
master=master.sort_values(["code","date"])
master["day_prior_close"]=master.groupby("code",sort=False).close.shift(1)
prev_close_map={(z.code,z.date):float(z.day_prior_close)
                for z in master.itertuples(index=False) if np.isfinite(z.day_prior_close)}
del master
expected=["09:31","11:30","13:01","15:00"]
# Strict universe already identified exclusively from T-1. Observe today's
# minute coverage only to decide whether live signal was legitimately observable.
paths=["REPAIR_REBREAK","STEADY_CONTROL","PM_RETAKE"]
signals=[];logs=[];active=[];trades=[];sid=0
def pct(a,b):return float(a/b-1)
def signal_frame(q,ev):
    q=q.sort_values("datetime").copy()
    ts=pd.to_datetime(q.datetime)
    q["tm"]=ts.dt.strftime("%H:%M")
    if q.tm.duplicated().any():return [],"duplicates"
    # Source convention may use 09:30/13:00 rather than 09:31/13:01.
    # Do not silently shift; require 240 distinct minutes and known TDX policy.
    if len(q)!=240:return [],"minute_count"
    if q.amount.isna().any() or q.volume.isna().any():return [],"null_volume"
    if q.open.le(0).any() or q.close.le(0).any():return [],"bad_prices"
    prior=float(ev.prior_close);day_low=np.inf;day_high=0.
    amount=0.;volume=0.;above_count=0;pm_count=0;pm_above=0
    buf=[];first_attack=None;candidate={}
    for i,r in enumerate(q.itertuples(index=False)):
        tm=r.tm
        day_low=min(day_low,float(r.low));day_high=max(day_high,float(r.high))
        amount+=float(r.amount);volume+=float(r.volume)
        # units check relative to price: each q.volume unit may be shares OR
        # hands; infer from prefix ratio only if stable, else reject.
        ratio=float(r.amount)/(float(r.volume)*float(r.close)) if r.volume>0 else np.nan
        buf.append({"time":tm,"px":float(r.close),"lo":float(r.low),"hi":float(r.high),
                    "amount":float(r.amount),"vol":float(r.volume)})
        if i<10:continue
        valid_ratios=[x["amount"]/(x["vol"]*x["px"]) for x in buf if x["vol"]>0 and x["amount"]>0]
        unit=float(np.median(valid_ratios)) if valid_ratios else np.nan
        if not(np.isfinite(unit) and (.95<=unit<=1.05 or 95<=unit<=105)):continue
        vwap=amount/(volume*unit)
        clv=(r.close-day_low)/(day_high-day_low) if day_high>day_low else .5
        above=r.close>=vwap
        above_count+=int(above)
        if tm>="13:00":pm_count+=1;pm_above+=int(above)
        change=pct(r.close,prior)
        prior10=max(x["px"] for x in buf[-11:-1])
        prior5=max(x["px"] for x in buf[-6:-1])
        recent=buf[-3:]
        consecutive=len(recent)==3 and all(x["px"]>=vwap*.995 for x in recent)
        if first_attack is None and change>=.025 and above and clv>=.62:
            first_attack=float(r.close)
        cond={}
        if "09:45"<=tm<="14:30" and len(buf)>=20:
            # Sell-side failed to defend a prior swing high; meaningful new
            # higher high rather than merely restoring one MA/VWAP threshold.
            cond["REPAIR_REBREAK"]=(r.close>=max(float(ev.prior_day_high),float(ev.prior_ma5))
                and r.close>prior10 and change>=.025 and clv>=.65
                and consecutive and ev.drawdown_from_peak<=-.05)
        if "09:50"<=tm<="14:25" and first_attack is not None:
            cond["STEADY_CONTROL"]=(change>=.03 and change<=.095
                and r.close>=first_attack*1.005 and clv>=.70 and above
                and consecutive and (r.close>=prior10))
        if "13:25"<=tm<="14:45" and pm_count>=24:
            cond["PM_RETAKE"]=(pm_above/pm_count>=.80 and above and clv>=.65
               and pct(r.close,prior)>=.005 and pct(r.close,prior)<=.065
               and r.close>prior5 and consecutive)
        for name,ok in cond.items():
            if name not in candidate and bool(ok) and i+1<len(q):
                nxt=q.iloc[i+1]
                # Earliest price is next bar OPEN, not current bar close.
                if float(nxt.volume)<=0 or float(nxt.amount)<=0:continue
                # Must avoid sealed limit-up and unsupported slip effects.
                if float(nxt.open)>=prior*1.098:continue
                entry=float(nxt.open)*(1+.002+.00031)
                candidate[name]={"branch":name,"signal_time":tm,
                    "entry_time":str(nxt.tm),"signal_price":float(r.close),
                    "entry_px_gross":float(nxt.open),"entry_cost":entry,
                    "signal_vwap":float(vwap),"signal_clv":float(clv),
                    "stock":str(ev.code),"date":str(ev.date),
                    "regime_breadth":float(ev.prior_breadth)}
    return list(candidate.values()),None

def sell_position(pos,rows,date,session_day,day_prior_close):
    # T+1: absolutely no same-day exit. TP5, SL12, time max3 days.
    if session_day<=pos["entry_day"]:return False
    q=rows.sort_values("datetime")
    for i,r in enumerate(q.itertuples(index=False)):
        c=float(r.close);tm=pd.Timestamp(r.datetime).strftime("%H:%M")
        # MAE uses only actually-held completed minutes, not later same-day lows.
        pos["observed_mae"]=min(pos["observed_mae"],pct(float(r.low),pos["entry_cost"]))
        pnl=pct(c,pos["entry_cost"])
        reason=None
        if pnl>=.05:reason="TP5"
        elif pnl<=-.12:reason="SL12"
        elif session_day-pos["entry_day"]>=3 and tm>="14:45":reason="T3"
        if not reason:continue
        if i+1>=len(q):
            return False
        nxt=q.iloc[i+1]
        if float(nxt.volume)<=0 or float(nxt.amount)<=0:
            return False
        # Limit down cannot guarantee liquidation.
        if not np.isfinite(day_prior_close) or float(day_prior_close)<=0:
            return False
        # The price-limit reference is the PRIOR CLOSE OF THIS EXIT DAY,
        # never the frozen prior close of the original entry day.
        if float(nxt.open)<=float(day_prior_close)*.902:
            return False
        net=float(nxt.open)*(1-.002-.00081)
        pos["observed_mae"]=min(pos["observed_mae"],pct(float(nxt.open),pos["entry_cost"]))
        pos.update(exit_date=date,exit_time=pd.Timestamp(nxt.datetime).strftime("%H:%M"),
          exit_net_px=net,net_return=net/pos["entry_cost"]-1,
          exit_reason=reason,status="CLOSED")
        return True
    return False

for dayi,date in enumerate(dates):
    if dayi==0:continue
    leaders=byday.get(date)
    relevant=set(leaders.code) if leaders is not None else set()
    relevant|={p["stock"] for p in active}
    if not relevant:continue
    f=files[date]
    df=pd.read_parquet(f,columns=["code","datetime","open","high","low","close","volume","amount"])
    df.code=df.code.astype(str).str.zfill(6)
    df=df[df.code.isin(relevant)]
    grouped={c:g.sort_values("datetime").copy() for c,g in df.groupby("code")}
    for p in list(active):
        if p["stock"] not in grouped:continue
        q=grouped[p["stock"]]
        if sell_position(p,q,date,dayi,prev_close_map.get((p["stock"],date),float("nan"))):
            active.remove(p);trades.append(p)
    if leaders is None:continue
    day_intents=[]
    for z in leaders.itertuples(index=False):
        if z.code not in grouped:continue
        if not np.isfinite(z.prior_breadth):continue
        ss,err=signal_frame(grouped[z.code],z)
        logs.append({"date":date,"code":z.code,"reason":err or "OK","signals":len(ss)})
        for item in ss:
            item.update(name="UNKNOWN_NAME",leader_tier=z.leader_tier,
                prior_close=float(z.prior_close),entry_day=dayi,
                observed_mae=0.,status="OPEN")
            signals.append(dict(item))
            day_intents.append(dict(item))
    # Causal order: sort by completed signal time, rather than by code.
    # One live position per ticker, even if multiple routes fire.
    for item in sorted(day_intents,key=lambda z:(z["signal_time"],z["entry_time"],z["stock"],z["branch"])):
        if any(p["stock"]==item["stock"] for p in active):continue
        if len(active)>=5:continue
        q=grouped[item["stock"]]
        after_entry=q[pd.to_datetime(q.datetime).dt.strftime("%H:%M").ge(item["entry_time"])]
        if after_entry.empty:continue
        px=item["entry_cost"]
        n_shares=int(math.floor(25000./px/100.))*100
        if n_shares<100:continue
        item["shares"]=n_shares
        item["entry_notional_weight"]=n_shares*px/1_000_000.
        # First entry-minute open is genuine purchase; subsequent minute lows
        # are adverse only while still holding and before a sell signal.
        item["observed_mae"]=min(0.,float(after_entry.low.min()/px-1.))
        active.append(item)
    if dayi%12==0:print("EXECUTED",date,"intents",len(signals),"closed",len(trades),flush=True)
for p in active:
    p["status"]="RIGHT_CENSORED"
    p["last_known_date"]=dates[-1]
    trades.append(p)
si=pd.DataFrame(signals);tr=pd.DataFrame(trades)
si.to_csv(OUT/"R77_new_branch_signals.csv",index=False)
tr.to_csv(OUT/"R77_new_branch_trade_details.csv",index=False)
pd.DataFrame(logs).to_csv(OUT/"R77_signal_data_quality.csv",index=False)
report=[]
for (branch,period),g in tr.groupby(["branch",tr.date.str[:7]]):
    q=g[g.status=="CLOSED"]
    report.append({"branch":branch,"month":period,"accepted":len(g),
        "closed":len(q),"wins":int(q.net_return.gt(0).sum()),
        "MAE10":int(q.observed_mae.le(-.10).sum()),"loss10":int(q.net_return.le(-.10).sum()),
        "mean_net_pct":float(100*q.net_return.mean()) if len(q) else None,
        "nominal_single_trade_pnl_pct":float(100*q.net_return.sum()) if len(q) else 0.})
summary=pd.DataFrame(report)
summary.to_csv(OUT/"R77_monthly_quality.csv",index=False)
qa={"status":"R77_NEW_SECOND_BUY_EXPLORATORY_COMPLETED","intents":len(si),"executed":len(tr),
    "closed":int(tr.status.eq("CLOSED").sum()),
    "right_censored":int(tr.status.eq("RIGHT_CENSORED").sum()),
    "single_active_ticker_guard":True,
    "chronological_signal_priority":True,
    "100_share_buy_lots":True,
    "MAE_t1_only_observed_before_exit":True,
    "limit_down_reference_updated_each_exit_date":True,
    "valid_asof_1minute_signals":True,
    "full_account_A_B_overlay_executed":False,
    "historically_viewed_months_not_blind":True,
    "portfolio_nav_verified":False,
    "not_eligible_for_production_deployment":True}
(OUT/"R77_exec_qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R77_EXEC_QA",json.dumps(qa),flush=True)
