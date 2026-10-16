#!/usr/bin/env python3
"""R86 W02 asof T-1 RECONSTRUCTED theme membership, 09:25 real same-day auction.
Self EXCLUDED from every peer median/positive-breadth/rank. No future prices
in features. Historical capture timestamp controls separately reported.
"""
from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
OUT=R/"w02_sector_peers"
A=pd.read_parquet(R/"w01_canonical/R86_REPAIRED_FULL_MARKET_0925.parquet")
A.code=A.code.astype(str).str.zfill(6)
A=A.rename(columns={"date":"buy_date"})
assert not A[["buy_date","code"]].duplicated().any()
B=pd.read_csv(L/"r84_auction_retake_20261016/w07_shadow_oct/R84_R83_RESET_589_auction_and_future_label_separated.csv",
  dtype={"code":str})
S=pd.read_csv(L/"r81_recurrent_reversal_20261016/w03_causal_signal/raw_r81_2clock_signals.csv",dtype={"code":str})
mapdata=pd.concat([B[["reset_date","auction_day"]].rename(columns={"auction_day":"buy_date"}),
 S[["reset_date","buy_date"]]],ignore_index=True).drop_duplicates()
assert not mapdata.reset_date.duplicated().any(),mapdata[mapdata.reset_date.duplicated()].to_dict("records")[:8]
forward=dict(zip(mapdata.reset_date,mapdata.buy_date))
R21=pd.read_parquet(L/"r21_true_height_rebuild_20260927/daily_r21.parquet",
 columns=["date","code","name","board","eligible","close","family_theme","family_known",
          "captured_at_ts","last_core_date","last_core_lane","core_age","family_rank","family_size"])
R21.code=R21.code.astype(str).str.zfill(6)
D=R21[R21.date.isin(mapdata.reset_date)].copy()
del R21
D["reset_date"]=D.date
D["buy_date"]=D.reset_date.map(forward)
assert D.buy_date.notna().all()
D=D.merge(A[["buy_date","code","r86_auction_price","r86_auction_lots","r86_auction_amount_yuan",
       "r86_lag10_ratio","r86_lag10_known","r86_strict_cross","r86_live_capture_proven","r86_source_type"]],
  on=["buy_date","code"],how="left",validate="1:1")
D["auction_gap_pct"]=100*(D.r86_auction_price/D.close-1)
D["auction_eligible"]=D.board.eq("主板")&D.eligible.fillna(False).astype(bool)&D.close.gt(0)&D.r86_auction_price.gt(0)
D["R21_label_eligible"]=D.family_known.fillna(False).astype(bool)&D.family_theme.notna()
D["captured_local_date"]=pd.to_datetime(D.captured_at_ts,unit="s",utc=True,errors="coerce").dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
D["signal_freeze"]=pd.to_datetime(D.buy_date+" 09:25:00")
D["R21_original_capture_asof"]=D.captured_local_date.le(D.signal_freeze)
D["group_valid"]=D.auction_eligible&D.R21_label_eligible
K=D[D.group_valid].copy()
print("R86_PIT_MEMBER_FULL",len(D),"groups_members",len(K),"families",K.family_theme.nunique(),flush=True)
groups={(dt,t):g.sort_values("code") for (dt,t),g in K.groupby(["buy_date","family_theme"],sort=False)}
globalby={dt:g for dt,g in D[D.auction_eligible].groupby("buy_date")}
V=pd.concat([
 B[["reset_date","auction_day","code"]].rename(columns={"auction_day":"buy_date"}),
 S[["reset_date","buy_date","code"]]
 ],ignore_index=True).drop_duplicates(["reset_date","buy_date","code"])
cand=V.merge(D.drop(columns=["date"]),on=["reset_date","buy_date","code"],how="left",validate="1:1")
assert len(cand)==len(V),(len(cand),len(V))
qa_missing=[];rows=[]
for z in cand.itertuples(index=False):
 out={"code":z.code,"reset_date":z.reset_date,"buy_date":z.buy_date,
      "sector_from_R21":z.family_theme if isinstance(z.family_theme,str) else None,
      "theme_member_known":bool(z.R21_label_eligible) if pd.notna(z.R21_label_eligible) else False,
      "label_source_R21_prior_date_exists":pd.notna(z.close),
      "own_R21_theme_captured_at_trade":bool(z.R21_original_capture_asof) if pd.notna(z.R21_original_capture_asof) else False,
      "own_0925_source":z.r86_source_type if isinstance(z.r86_source_type,str) else "MISSING",
      "own_0925_dual":bool(z.r86_strict_cross) if pd.notna(z.r86_strict_cross) else False,
      "gap_pct":float(z.auction_gap_pct) if pd.notna(z.auction_gap_pct) else np.nan,
      "own_auction_amount_yuan":float(z.r86_auction_amount_yuan) if pd.notna(z.r86_auction_amount_yuan) else np.nan,
      "own_lag10_ratio":float(z.r86_lag10_ratio) if pd.notna(z.r86_lag10_ratio) else np.nan,
      "sector_eligible_peers":0,"sector_peer_median_gap":np.nan,"sector_peer_positive_fraction":np.nan,
      "sector_peer_median_ratio10":np.nan,"sector_peer_ratio10_valid":0,
      "sector_gap_lead_pp":np.nan,"sector_open_gap_percentile":np.nan,
      "strict_peer_captured_before_0925":0,"all_peer_R21_captured_at_trade":False,
      "market_eligible_tickers":0,"market_median_gap":np.nan,"market_positive_fraction":np.nan}
 grp=groups.get((z.buy_date,z.family_theme)) if isinstance(z.family_theme,str) else None
 if grp is not None:
  others=grp[grp.code.ne(z.code)].copy()
  if len(others):
   out["sector_eligible_peers"]=len(others)
   out["sector_peer_median_gap"]=float(others.auction_gap_pct.median())
   out["sector_peer_positive_fraction"]=float(others.auction_gap_pct.gt(0).mean())
   ratios=others.r86_lag10_ratio.dropna()
   out["sector_peer_ratio10_valid"]=len(ratios)
   if len(ratios):out["sector_peer_median_ratio10"]=float(ratios.median())
   if pd.notna(z.auction_gap_pct):
    out["sector_gap_lead_pp"]=float(z.auction_gap_pct)-out["sector_peer_median_gap"]
    out["sector_open_gap_percentile"]=float(others.auction_gap_pct.le(z.auction_gap_pct).mean())
   out["strict_peer_captured_before_0925"]=int(others.R21_original_capture_asof.sum())
   out["all_peer_R21_captured_at_trade"]=bool(others.R21_original_capture_asof.all())
 market=globalby.get(z.buy_date)
 if market is not None:
  others=market[market.code.ne(z.code)]
  out["market_eligible_tickers"]=len(others)
  if len(others):
   out["market_median_gap"]=float(others.auction_gap_pct.median())
   out["market_positive_fraction"]=float(others.auction_gap_pct.gt(0).mean())
 rows.append(out)
P=pd.DataFrame(rows)
assert not P[["code","reset_date","buy_date"]].duplicated().any()
P.to_csv(OUT/"R86_auction_relative_peer_features_every_reset.csv",index=False,encoding="utf-8-sig")
merged=S.merge(P,on=["code","reset_date","buy_date"],how="left",validate="m:1")
assert len(merged)==563
MEM=pd.read_csv(L/"r81_recurrent_reversal_20261016/w07_R21_core_memory/R81_asof_R21_lifecycle_membership_each_signal.csv",dtype={"code":str})
MEM=MEM[MEM.variant.eq("BASE_T3")][["code","buy_date","branch","entry_time","CORE10","LOOP1"]]
assert not MEM[["code","buy_date","branch","entry_time"]].duplicated().any()
merged=merged.merge(MEM,on=["code","buy_date","branch","entry_time"],how="left",validate="1:1")
FAC=pd.read_csv(L/"r82_causal_retake_20261016/w02_signal_features/R82_563_causal_signal_state_factors.csv",dtype={"code":str})
merged=merged.merge(FAC[["code","buy_date","branch","entry_time","H1_REAL_ACCUMULATION"]],
  on=["code","buy_date","branch","entry_time"],how="left",validate="1:1")
assert merged.CORE10.notna().all() and merged.H1_REAL_ACCUMULATION.notna().all()
amt=merged.own_auction_amount_yuan.ge(3e6)
npeers=merged.sector_eligible_peers.ge(4)
g1=npeers&amt&merged.sector_gap_lead_pp.ge(.5)&merged.sector_peer_positive_fraction.ge(.50)
g2=g1&merged.sector_open_gap_percentile.ge(.75)&merged.own_lag10_ratio.ge(merged.sector_peer_median_ratio10)&merged.sector_peer_ratio10_valid.ge(3)
g3=npeers&amt&merged.gap_pct.between(-6,0)&merged.sector_gap_lead_pp.ge(.8)&merged.own_lag10_ratio.ge(.5)
g4=npeers&amt&merged.sector_gap_lead_pp.ge(.5)&merged.sector_peer_positive_fraction.ge(.60)
rules={
 "G0_ALL":np.ones(len(merged),dtype=bool),
 "G1_SECTOR_LEAD":g1,"G2_SECTOR_PLUS_AUCTION":g2,
 "G3_LOW_OPEN_LEAD":g3,"G4_BREADTH_FIRST":g4,
 "G5_CORE_REAL":(g1|g3)&merged.CORE10,
 "G6_REPEATED_LEADER":(g1|g3)&merged.LOOP1,
 "G7_SECTOR_AND_INTRADAY":g2&merged.H1_REAL_ACCUMULATION,
 "G8_CAPTURE_STRICT":(g1|g3)&merged.own_R21_theme_captured_at_trade&merged.strict_peer_captured_before_0925.ge(4),
 "G9_MARKET_CONTROL":amt&merged.gap_pct.ge(merged.market_median_gap+.5)&merged.market_positive_fraction.ge(.5),
 "G10_TWO_SOURCE_REPLICATION":(g1|g3)&merged.own_0925_dual
}
for k,v in rules.items():merged[k]=v.fillna(False).astype(bool) if hasattr(v,"fillna") else v
gates=list(rules)
assert all(str(merged[c].dtype)=="bool" for c in gates)
assert merged.G0_ALL.sum()==563
merged.to_csv(OUT/"R86_563_intraday_signals_and_fixed_relative_auction_gates.csv",index=False)
late=P[P.buy_date.isin(["2026-09-29","2026-10-09","2026-10-14"]) &P.code.eq("601811")]
late.to_csv(OUT/"R86_601811_3_reversal_peer_visibility_audit.csv",index=False)
qa={"status":"R86_T_MINUS1_THEME_MEMBERSHIP_AND_REAL0925_PEER_AUDIT",
 "reset_uniques":len(P),"raw_research_signals":len(merged),
 "sector_T_minus1_from_R21_only":True,"R21_theme_historical_backfilled_flagged":True,
 "peers_explicitly_exclude_own_stock":True,"perstock_priorclose_is_reset_date":True,
 "r81_signals_with_prior_R21_family":int(merged.theme_member_known.sum()),
 "r81_signals_with_four_tradeable_peers":int(merged.sector_eligible_peers.ge(4).sum()),
 "r81_signals_with_prior_R21_live_capture":int(merged.own_R21_theme_captured_at_trade.sum()),
 "r81_signals_with_four_live_captured_peers":int(merged.strict_peer_captured_before_0925.ge(4).sum()),
 "signal_by_gate":{k:int(merged[k].sum()) for k in gates},
 "Xinhua_R21_Oct_after_Sep23_missing":bool(late.label_source_R21_prior_date_exists.fillna(False).sum()<3),
 "many_historical_auction_files_backfilled":True,"no_production_write":True}
(OUT/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R86_SECTOR_QA",json.dumps(qa,ensure_ascii=False),flush=True)
print("R86_GATES_MONTHS",merged.groupby(merged.buy_date.str[:7])[gates].sum().to_string(),flush=True)
print("R86_XINHUA_PEERS",late[["reset_date","buy_date","sector_from_R21","gap_pct","sector_gap_lead_pp","sector_eligible_peers","own_lag10_ratio","label_source_R21_prior_date_exists"]].to_string(index=False),flush=True)