#!/usr/bin/env python3
"""R77 full-market daily *as-of* leader/reset catalogue. Standalone, research-only.
Never scans First Buy branch signals and never modifies any deployed R50 service.
"""
from pathlib import Path
import json, hashlib, os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
WORK=Path("r77_out");WORK.mkdir(exist_ok=True)
files=sorted(ROOT.glob("2026-*/part-*/date=*/minute1.parquet"))
assert len(files)>=100, f"expected 2026 Mar-Aug minute parquet; got {len(files)}"
daily=[]
coverage=[]
for j,f in enumerate(files):
 day=f.parent.name.split("=")[-1]
 if not ("2026-03-01"<=day<="2026-08-31"):continue
 x=pd.read_parquet(f,columns=["code","datetime","open","high","low","close","volume","amount"])
 x["code"]=x.code.astype(str).str.zfill(6)
 x=x.sort_values(["code","datetime"])
 g=x.groupby("code",sort=False,observed=True)
 z=g.agg(opn=("open","first"),hi=("high","max"),lo=("low","min"),
         close=("close","last"),vol=("volume","sum"),amt=("amount","sum"),
         bars=("close","size")).reset_index()
 z["date"]=day;z["di"]=j
 status=f.with_name("daily_stock_status.parquet")
 assert status.exists(),status
 s=pd.read_parquet(status)
 s["code"]=s.code.astype(str).str.zfill(6)
 keep=["code","is_st","is_star_st","is_new_listing_initial","is_suspended","price_limit_up_pct","market_board"]
 s=s[[k for k in keep if k in s.columns]]
 assert not s.code.duplicated().any()
 z=z.merge(s,on="code",how="left",validate="one_to_one")
 z["main_board"]=z.code.str.match(r"^(60[0-7]|000|001|002|003)")
 z["status_ok"]=(z.is_st.eq(False)&z.is_star_st.eq(False)&z.is_new_listing_initial.eq(False))
 # A full 240 nonzero minute tape is affirmative evidence the security actually traded,
 # even where the repo's supplemental halt feed has a nullable unknown.
 z["traded_verified"]=z.bars.eq(240)&z.vol.gt(0)&z.amt.gt(0)&z.close.gt(0)
 z["eligible_board"]=z.main_board&z.status_ok&z.traded_verified&(~z.is_suspended.eq(True))
 daily.append(z)
 coverage.append({"date":day,"stocks_with_240_bars":int(z.traded_verified.sum()),
                  "main_regular_valid":int(z.eligible_board.sum()),
                  "status_unknown_st":int(z.is_st.isna().sum())})
 if j%12==0:print("DAILY_AGG",day,"count",len(z),"valid",int(z.eligible_board.sum()),flush=True)
del x,z,s
d=pd.concat(daily,ignore_index=True).sort_values(["code","date"]).reset_index(drop=True)
g=d.groupby("code",sort=False,observed=True)
d["prior_close"]=g.close.shift(1)
d["priorprior_close"]=g.close.shift(2)
d["daily_ret"]=d.close/d.prior_close-1
lim=pd.to_numeric(d.price_limit_up_pct,errors="coerce").astype(float)
# Unknown historical price-limit rule MUST NOT be guessed as normal 10%.
# Price limits expressed as 0.10 or 10 are normalized; unknown stays NaN.
lim_norm=np.where(lim.gt(1),lim/100.,lim)
limit_known=pd.Series(lim_norm,index=d.index).between(.095,.105)
d["lu"]=(d.daily_ret.ge(.095)&d.daily_ret.le(.115)&
         limit_known&d.eligible_board).fillna(False)
breaks=(~d.lu).groupby(d.code).cumsum()
d["streak"]=d.lu.astype("int16").groupby([d.code,breaks]).cumsum()
g=d.groupby("code",sort=False,observed=True)
d["prior20_count"]=g.lu.transform(lambda x:x.shift(1).rolling(20,min_periods=20).sum())
d["prior20_maxrun"]=g.streak.transform(lambda x:x.shift(1).rolling(20,min_periods=20).max())
d["prior20_high"]=g.hi.transform(lambda x:x.shift(1).rolling(20,min_periods=20).max())
d["prior_ma5"]=g.close.transform(lambda x:x.shift(1).rolling(5,min_periods=5).mean())
d["last_lu_di"]=d.di.where(d.lu).groupby(d.code).ffill()
d["since_lu"]=d.di-g.last_lu_di.shift(1)
d["prior_day_high"]=g.hi.shift(1)
d["prev_day_valid"]=g.eligible_board.shift(1).fillna(False)
d["drawdown_from_peak"]=d.prior_close/d.prior20_high-1
# Strict multi-board public-market leadership; formed entirely before the buy date.
d["leader_strict"]=(d.prior20_maxrun>=3)
d["leader_2board"]=(d.prior20_maxrun>=2)&(d.prior20_count>=3)
d["leader_eligible"]=(d.eligible_board&d.prev_day_valid&
     (d.leader_strict|d.leader_2board)&d.since_lu.between(2,12)&
     d.drawdown_from_peak.between(-.28,-.035)&
     d.prior_close.notna()&d.priorprior_close.notna())
# All market regime measures are lagged by a complete session; not sector ground truth.
breadth=d[d.eligible_board].groupby("date").agg(
    broad_up=("daily_ret",lambda x:float(x.gt(0).mean())),
    strong_count=("daily_ret",lambda x:int(x.ge(.05).sum())),
    limit_count=("lu","sum")).sort_index()
breadth["prior_breadth"]=breadth.broad_up.shift(1)
breadth["prior_limit_count"]=breadth.limit_count.shift(1)
b=breadth[["prior_breadth","prior_limit_count"]]
e=d[d.leader_eligible].merge(b,left_on="date",right_index=True,how="left")
e["prior_day_push"]=(e.prior_day_high/e.priorprior_close-1>=.05)
e["leader_tier"]=np.where(e.leader_strict,"STRICT_3BOARD","SECOND_2BOARD")
e=e[["date","di","code","prior_close","priorprior_close","prior_ma5","prior20_high",
  "drawdown_from_peak","since_lu","leader_tier","prior_day_push","prior_day_high","prior_breadth","prior_limit_count",
  "price_limit_up_pct"]]
# Genuine 20-session in-repository warm-up can already finish in late March.
# These rows are preserved for audit; trade engine separately starts April.
assert e.date.ge("2026-03-23").all() if len(e) else True
assert e[["date","code"]].duplicated().sum()==0
d[["code","date","di","close","hi","lo","vol","amt","bars","eligible_board","lu",
    "prior_close","prior20_count","prior20_maxrun"]].to_csv(WORK/"R77_asof_daily_master.csv",index=False)
e.to_csv(WORK/"R77_asof_eligible_stockdays.csv",index=False)
pd.DataFrame(coverage).to_csv(WORK/"R77_daily_data_quality.csv",index=False)
summ={"status":"R77_FULLMARKET_DAILY_PASS","days":len(files),"daily_stockrows":len(d),
  "unique_stockcodes":d.code.nunique(),"eligible_stockdays":len(e),
  "eligible_unique_codes":e.code.nunique(),"eligible_by_month":e.groupby(e.date.str[:7]).size().to_dict(),
  "main_board_status_strict":True,"source_status_uncertainty_after_2026_07_30":True,
  "new_issuer_ST_filled_from_existing_repo_metadata_not_contemporary_exchange_record":True,
  "future_price_based_label_in_identity":False,
  "no_r50_production_writes":True,
  "require_nonfuture_sector_source_before_real_topic_study":True}
(WORK/"R77_daily_qa.json").write_text(json.dumps(summ,ensure_ascii=False,indent=2))
print("R77_DAILY_PASS",json.dumps(summ,ensure_ascii=False),flush=True)
