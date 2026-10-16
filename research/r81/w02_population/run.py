#!/usr/bin/env python3
"""R81 W02 all-market prior-day core-memory + >=5% reset episode catalogue.
Labels constructed on the reset day close. Never use next day's price for identity.
This is historical price-core proxy, NOT true theme+leader certification.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd,numpy as np
H=Path.home()/"Documents/stock_data_windows_mirror/daily/tdx_parquet"
O=Path(__file__).resolve().parent
allfiles=sorted(H.glob("date=*/daily_ohlcv.parquet"))
allfiles=[p for p in allfiles if "2025-09-01"<=p.parent.name[5:]<="2026-10-16"]
assert len(allfiles)>=250,len(allfiles)
def one(p):
 d=pd.read_parquet(p,columns=["code","date","name","board","open","high","low","close","volume","amount"])
 d["code"]=d.code.astype(str).str.zfill(6)
 return d
with ThreadPoolExecutor(max_workers=6) as ex:xs=list(ex.map(one,allfiles))
d=pd.concat(xs,ignore_index=True).sort_values(["code","date"]).reset_index(drop=True)
d["close"]=pd.to_numeric(d.close,errors="coerce");d["high"]=pd.to_numeric(d.high,errors="coerce")
d["low"]=pd.to_numeric(d.low,errors="coerce")
d["eligible"]=d.board.eq("主板") & d.name.fillna("").astype(str).str.contains(r"ST|退",case=False,regex=True).eq(False)
d["eligible"]&=d.close.gt(0) & d.volume.gt(0) & d.amount.gt(0)
cal=sorted(d.date.unique());di={x:i for i,x in enumerate(cal)}
d["di"]=d.date.map(di)
g=d.groupby("code",sort=False)
d["prevclose"]=g.close.shift(1)
d["ret"]=d.close/d.prevclose-1
# Historical proxy of a MAIN-BOARD actually 10% closing limit; no official per-session price limits.
expected_up=(d.prevclose*1.10+0.000001).round(2)
d["lu_proxy"]=d.eligible & d.prevclose.gt(0) & d.close.ge(expected_up-.025) & d.close.le(expected_up+.015)
breaks=(~d.lu_proxy).groupby(d.code).cumsum()
d["streak"]=d.lu_proxy.astype("int16").groupby([d.code,breaks]).cumsum()
g=d.groupby("code",sort=False)
# Latest recognized price-core identity at reset-close may have been formed days ago.
d["streak_prior20_max"]=g.streak.transform(lambda x:x.shift(1).rolling(20,min_periods=15).max()).fillna(0)
d["streak_prior30_max"]=g.streak.transform(lambda x:x.shift(1).rolling(30,min_periods=20).max()).fillna(0)
d["last_board_di"]=d.di.where(d.lu_proxy).groupby(d.code).ffill().groupby(d.code).shift(1)
d["since_board"]=d.di-d.last_board_di
# Note previous valid day and volume prior close are known by today's close.
d["prior_close_valid"]=g.eligible.shift(1).fillna(False)
d["prior20_high"]=g.high.transform(lambda x:x.shift(1).rolling(20,min_periods=15).max())
d["prev3close"]=g.close.shift(3)
d["prev_volume"]=g.volume.shift(1)
d["prev_amount"]=g.amount.shift(1)
breadth=d[(d.eligible)&d.date.between("2026-01-01","2026-10-16")].groupby("date").agg(up=("ret",lambda x:float((x>0).mean())),n=("code","size"))
breadth["prior_up"]=breadth.up.shift(1)
d=d.merge(breadth[["prior_up"]],left_on="date",right_index=True,how="left")
# RESET = confirmed date T close after positive leader stage, no future use:
d["price_core"]=d.streak_prior20_max.ge(3)&d.since_board.between(1,15)
d["reset_direct"]=(d.ret.between(-.105,-.05))
# Expanded 2-3day falls only when T's drawdown cumulative>=8% and not already direct,
# require at least a red (daily decline >=2%) and no positive board that day.
d["reset_multi"]=~d.reset_direct & d.ret.le(-.02) & (d.close/d.prev3close-1).le(-.08) & ~d.lu_proxy
d["candidate"]=d.eligible & d.prior_close_valid & d.price_core & d.prev_amount.ge(20_000_000) & (d.reset_direct | d.reset_multi)
# This is a known-time price-core eligibility certificate, not 09:25 industry money-rank.
m=d[d.candidate & d.date.between("2026-01-01","2026-10-16")].copy()
assert len(m)>60,len(m)
m["reset_type"]=np.where(m.reset_direct,"DIRECT_DROP5","MULTI3_DROP8")
m["reset_uid"]=m.code+":"+m.date
m["leader_type"]="PRICE_CORE_PROXY_3BOARD"
m["true_theme_certified"]=False
m["max_drop_pct"]=100*m.ret
m["prior_breadth"]=m.prior_up
# Feature as known at each close; remember ticker identity can survive next re-reset.
keep=["reset_uid","date","code","name","leader_type","true_theme_certified","reset_type","max_drop_pct",
 "close","prevclose","streak_prior20_max","streak_prior30_max","since_board","prior20_high",
 "prev_volume","prev_amount","prior_breadth","di"]
m=m[keep]
m.to_csv(O/"R81_fullmarket_reset_episodes.csv",index=False,encoding="utf-8-sig")
m.groupby([m.date.str[:7],"reset_type"]).agg(n=("code","size"),unique=("code","nunique")).to_csv(O/"R81_episodes_by_month.csv")
c=m.groupby("code").agg(episodes=("reset_uid","size"),names=("name","last"),
    first_reset=("date","min"),last_reset=("date","max")).sort_values("episodes",ascending=False)
c.to_csv(O/"R81_repeated_reset_leaders.csv")
target=m[m.code=="601811"][["date","name","reset_type","max_drop_pct","streak_prior20_max","since_board"]]
print("R81_TARGET_EPISODE",target.to_dict("records"),flush=True)
print("R81_MOTHER_MONTH",m.groupby(m.date.str[:7]).size().to_dict(),flush=True)
qa={"status":"R81_ASOF_MOTHER_READY","all_daily_rows":len(d),"sessions":len(cal),
 "price_core_reset_events":len(m),"unique_stocks":m.code.nunique(),
 "multi_reset_stock_count":int(c.episodes.ge(2).sum()),"multi_reset_event_count":int(c.loc[c.episodes.ge(2),"episodes"].sum()),
 "target_601811_drops":target.to_dict("records"),"max_train_analogs_date":"2026-09-23",
 "historical_theme_certified":False,"official_exchange_limits_verified":False,
 "ex_date_adjustment_verified":False,"status_historic_names_available_from_dated_TDX":True,
 "uses_future_outcomes_for_candidate_identity":False,"does_not_write_R50":True}
(O/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R81_MOTHER_QA",json.dumps(qa,ensure_ascii=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]