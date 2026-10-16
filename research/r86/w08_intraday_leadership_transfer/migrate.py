#!/usr/bin/env python3
"""R86 phase3 sector price strength migration: 09:25 REAL -> completed 1m signal.
All prior sector group membership frozen at previous T-1 EOD (R21 reconstructible).
No same-day final prices, no post-signal minutes, no future labels for gates.
Posthoc hypotheses inspired by positive known 601811, MUST NOT market as OOS.
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd,numpy as np,pyarrow.dataset as ds,json
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
H=Path.home()/"Documents/stock_data_windows_mirror/1min"
O=R/"w08_intraday_leadership_transfer"
assert (R/"PHASE3_AUCTION_TO_SIGNAL_LEADERSHIP_MIGRATION_POSTHOC.md").exists()
S=pd.read_csv(L/"r81_recurrent_reversal_20261016/w03_causal_signal/raw_r81_2clock_signals.csv",dtype={"code":str})
assert len(S)==563
D=pd.read_parquet(L/"r21_true_height_rebuild_20260927/daily_r21.parquet",
 columns=["date","code","close","board","eligible","family_theme","family_known",
 "last_core_family","last_core_date","last_core_lane","core_age","captured_at_ts"])
D.code=D.code.astype(str).str.zfill(6)
D=D[D.date.isin(S.reset_date)].copy()
assert not D[["date","code"]].duplicated().any()
D=D.rename(columns={"date":"reset_date","close":"prior_close"})
mapday=S[["reset_date","buy_date"]].drop_duplicates()
assert not mapday.reset_date.duplicated().any()
D=D.merge(mapday,on="reset_date",how="left",validate="m:1")
D["member"]=D.family_known.fillna(False).astype(bool)&D.family_theme.notna()&D.board.eq("主板")&D.eligible.fillna(False).astype(bool)&D.prior_close.gt(0)
A=pd.read_parquet(R/"w01_canonical/R86_REPAIRED_FULL_MARKET_0925.parquet",
 columns=["date","code","r86_auction_price","r86_auction_amount_yuan","r86_lag10_ratio"])
A.code=A.code.astype(str).str.zfill(6)
A=A.rename(columns={"date":"buy_date"})
D=D.merge(A,on=["buy_date","code"],how="left",validate="m:1")
D["auction_gap_pct"]=100*(D.r86_auction_price/D.prior_close-1)
lookup={(r.reset_date,r.code):r for r in D.itertuples(index=False)}
group={k:z for k,z in D[D.member&D.r86_auction_price.gt(0)].groupby(["reset_date","family_theme"])}
from collections import defaultdict
need=defaultdict(set)
for s in S.itertuples(index=False):
 current=lookup.get((s.reset_date,s.code))
 if current is None:continue
 family=current.family_theme if bool(current.family_known) and isinstance(current.family_theme,str) else None
 if family is None:
  stamp=current.last_core_date
  if (isinstance(stamp,str) and stamp<=s.reset_date
     and str(current.last_core_lane) in ("FAMILY_STAGE","ABS_HEIGHT")
     and 0<=float(current.core_age)<=20
     and isinstance(current.last_core_family,str)):
   family=current.last_core_family
 if family is None:continue
 g=group.get((s.reset_date,family))
 if g is None:continue
 need[s.buy_date].update(g.code.astype(str).tolist())
 need[s.buy_date].add(s.code)
print("R86_MIGRATION_TAPE_DAYS",len(need),"needs_stockday_pairs",sum(len(v) for v in need.values()),flush=True)
quotes={}
datefail=[]
for n,(day,codes) in enumerate(sorted(need.items())):
 path=H/("date="+day)
 if not path.exists():datefail.append(day);continue
 times=S[S.buy_date.eq(day)].signal_time.unique().tolist()
 try:
  z=ds.dataset(str(path),format="parquet").to_table(
   columns=["code","time","close"],
   filter=ds.field("code").isin(sorted(codes))&ds.field("time").isin(times)).to_pandas()
 except Exception as e:
  raise RuntimeError(f"Tape source problem at {day}") from e
 z.code=z.code.astype(str).str.zfill(6)
 assert not z[["code","time"]].duplicated().any(),day
 for k,price in zip(zip(z.code,z.time),z.close):
  quotes[(day,k[0],k[1])]=float(price)
 if (n+1)%30==0:print("R86_MIGRATION_TAPES",day,"days",n+1,"quotes",len(quotes),flush=True)
rows=[];match=0
for sig in S.itertuples(index=False):
 k=(sig.reset_date,sig.code)
 d=lookup.get(k)
 family=None;familysource="NO_PRIOR_KNOWN_FAMILY";core20=False
 if d is not None:
  core20=(isinstance(d.last_core_date,str) and d.last_core_date<=sig.reset_date
      and str(d.last_core_lane) in ["FAMILY_STAGE","ABS_HEIGHT"]
      and pd.notna(d.core_age) and 0<=d.core_age<=20)
  if bool(d.family_known) and isinstance(d.family_theme,str):
   family=d.family_theme;familysource="CURRENT_T_MINUS1_FAMILY"
  elif core20 and isinstance(d.last_core_family,str):
   family=d.last_core_family;familysource="LAST_CERTIFIED_CORE_FAMILY_FALLBACK"
 out={"code":sig.code,"name":sig.name,"reset_date":sig.reset_date,
  "buy_date":sig.buy_date,"branch":sig.branch,"signal_time":sig.signal_time,
  "entry_time":sig.entry_time,"family_used":family,"family_source":familysource,
  "historical_core_20":bool(core20),
  "asof_auction_lead_pctpts":np.nan,"completed_signal_lead_pctpts":np.nan,
  "leadership_change_pctpts":np.nan,"own_gap_0925_pct":np.nan,
  "peer_breadth_positive_at_signal":np.nan,"signal_top_peer_percentile":np.nan,
  "matched_peers_in_both_times":0,
  "signal_m1_crosschecks_R81":False}
 prev=d.prior_close if d is not None else np.nan
 if d is None or pd.isna(prev) or not family:
  rows.append(out);continue
 own_auction=float(d.auction_gap_pct) if pd.notna(d.auction_gap_pct) else np.nan
 own_sig=float(quotes.get((sig.buy_date,sig.code,sig.signal_time),np.nan))
 if np.isfinite(own_sig):
  assert abs(own_sig-sig.signal_price)<1e-7,(sig.code,sig.buy_date,sig.signal_time,own_sig,sig.signal_price)
  out["signal_m1_crosschecks_R81"]=True;match+=1
 if not np.isfinite(own_sig) or not np.isfinite(own_auction):
  rows.append(out);continue
 g=group.get((sig.reset_date,family))
 if g is None:
  rows.append(out);continue
 peers=g[g.code.ne(sig.code)].copy()
 peers["curr_completed"]=peers.code.map(lambda c:quotes.get((sig.buy_date,c,sig.signal_time),np.nan))
 # Baseline at 09:25 and minute must compare exact same peer members.
 peers=peers[peers.curr_completed.notna()&peers.auction_gap_pct.notna()&peers.prior_close.gt(0)]
 count=len(peers)
 out["matched_peers_in_both_times"]=count
 if count<4:rows.append(out);continue
 peers["minute_gain_pct"]=100*(peers.curr_completed/peers.prior_close-1)
 own_minute_ret=100*(own_sig/prev-1)
 alead=own_auction-float(peers.auction_gap_pct.median())
 mlead=own_minute_ret-float(peers.minute_gain_pct.median())
 out.update({"asof_auction_lead_pctpts":alead,"completed_signal_lead_pctpts":mlead,
  "leadership_change_pctpts":mlead-alead,
  "own_gap_0925_pct":own_auction,
  "peer_breadth_positive_at_signal":float(peers.minute_gain_pct.gt(0).mean()),
  "signal_top_peer_percentile":float(peers.minute_gain_pct.le(own_minute_ret).mean())})
 rows.append(out)
F=pd.DataFrame(rows)
assert len(F)==len(S) and F[["code","buy_date","branch","entry_time"]].duplicated().sum()==0
valid=F.matched_peers_in_both_times.ge(4)
rotate=valid&F.completed_signal_lead_pctpts.ge(2)&F.leadership_change_pctpts.ge(2)&F.signal_top_peer_percentile.ge(.75)
strategies={
 "RAW_R81_ALL":pd.Series(True,index=F.index),
 "ROTATE_FROM_WEAK":rotate&F.asof_auction_lead_pctpts.le(0),
 "ROTATE_ANY":rotate,
 "ROTATE_WITH_BREADTH":rotate&F.peer_breadth_positive_at_signal.ge(.40),
 "ROTATE_CORE20":rotate&F.historical_core_20,
 "STICKY_CORE_ONLY":rotate&F.historical_core_20&F.family_source.eq("LAST_CERTIFIED_CORE_FAMILY_FALLBACK"),
 "ABSOLUTE_LEADER_CONTROL":valid&F.completed_signal_lead_pctpts.ge(2)&F.signal_top_peer_percentile.ge(.75)
}
for name,state in strategies.items():F[name]=state.fillna(False).astype(bool)
assert F.RAW_R81_ALL.sum()==563 and F.signal_m1_crosschecks_R81.sum()>420
F.to_csv(O/"R86_563_AUCTION_TO_COMPLETED_MINUTE_SECTOR_LEADERSHIP_MIGRATION.csv",index=False)
QA={"status":"R86_PHASE3_563_CROSS_MARKET_ASOF_MINUTE_LEADERSHIP_MIGRATION_READY",
 "source_signals":len(F),"days_read":len(need),"needed_stockday_pairs":sum(len(x) for x in need.values()),
 "matched_original_complete_minute_closes":int(F.signal_m1_crosschecks_R81.sum()),
 "same_peer_both_time_rows":int(valid.sum()),
 "last_core_family_sticky_fallback_rows":int(F.family_source.eq("LAST_CERTIFIED_CORE_FAMILY_FALLBACK").sum()),
 "all_strategy_counts":{key:int(F[key].sum()) for key in strategies},
 "failure_dates":datefail,
 "all_tape_time_equals_completed_original_signal":True,
 "all_peers_exclude_self":True,
 "none_of_the_minutes_after_original_signal_used":True,
 "conditional_hypothesis_from_known_Xinhua_case_POSTHOC":True,
 "actual_family_capture_at_prior_time_not_proven":True,
 "no_production_changes":True}
(O/"qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))
print("R86_MIGRATION_QA",json.dumps(QA,ensure_ascii=False),flush=True)
print("R86_MIGRATION_MONTH",F.groupby(F.buy_date.str[:7])[list(strategies)].sum().to_string(),flush=True)