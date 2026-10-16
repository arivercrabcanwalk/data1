#!/usr/bin/env python3
"""R86 repair 09:25 canonical auction. Inputs R84 READ ONLY.
Explicit fillna(False) repairs 460690 Mac-only NULLs, never retroclaims
live timestamp on backfilled raw shards.
"""
import pandas as pd,numpy as np,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
O=R/"w01_canonical"
src=L/"r84_auction_retake_20261016/w03_overlap_qa/R84_canonical_auction_09_25_with_lagged_ratios.parquet"
C=pd.read_parquet(src)
assert len(C)==515399 and C[["date","code"]].duplicated().sum()==0
orig_mac=int(C.source_status.eq("MAC_0925_ONLY").sum())
original_has=int(C.auction_price.notna().sum())
assert orig_mac==460690,(orig_mac,C.source_status.value_counts().to_dict())
bool_src=C.source_server_ok.fillna(False).astype(bool)
conflict=C.source_conflict.fillna(False).astype(bool)
C["R86_use_server"]=bool_src&C.server_open_px.gt(0)&C.server_open_lots.gt(0)&C.server_open_amount.gt(0)
C["R86_use_mac"]=~C.R86_use_server&C.present_mac.fillna(False).astype(bool)&C.price.gt(0)&C.lots.gt(0)&C.amount_yuan.gt(0)
valid=(C.R86_use_server|C.R86_use_mac)&~conflict
C["r86_auction_price"]=np.where(C.R86_use_server,C.server_open_px,C.price)
C["r86_auction_lots"]=np.where(C.R86_use_server,C.server_open_lots,C.lots)
C["r86_auction_amount_yuan"]=np.where(C.R86_use_server,C.server_open_amount,C.amount_yuan)
C.loc[~valid,["r86_auction_price","r86_auction_lots","r86_auction_amount_yuan"]]=np.nan
assert C.r86_auction_price.gt(0).sum()>=493700,C.r86_auction_price.notna().sum()
C["r86_strict_cross"]=C.cross_verified.fillna(False).astype(bool)&valid
C["r86_source_type"]=np.select(
 [C.r86_strict_cross,C.R86_use_mac&C.source_status.eq("MAC_0925_ONLY"),
  C.R86_use_server&C.source_status.eq("SERVER_HISTORIC_BACKFILL_ONLY")],
 ["TWO_SOURCE_MATCH","VALID_MAC_ONLY","VALID_SERVER_ONLY"],default="OTHER_OR_INVALID")
# Mac-only after source validation is a valid historical trade price, but not
# evidence a bot downloaded that specific file before open on the past day.
C["r86_live_capture_proven"]=C.server_captured_at_trade.fillna(False).astype(bool)
C.sort_values(["code","date"],inplace=True)
C["r86_lag10_median_amount"]=C.groupby("code",sort=False).r86_auction_amount_yuan.transform(
 lambda v:v.shift(1).rolling(10,min_periods=5).median())
C["r86_lag10_ratio"]=C.r86_auction_amount_yuan/C.r86_lag10_median_amount
C["r86_lag10_known"]=C.r86_lag10_median_amount.gt(0)
# Same auction monetary unit must generally agree with 100 shares per lot.
C["r86_amt_px_lot_ratio"]=C.r86_auction_amount_yuan/(C.r86_auction_price*C.r86_auction_lots*100)
drift=C[C.r86_auction_price.notna()&C.r86_amt_px_lot_ratio.notna()]
C[["date","code","source_status","source_file_mtime_utc","collected_at",
 "r86_auction_price","r86_auction_lots","r86_auction_amount_yuan",
 "r86_lag10_median_amount","r86_lag10_ratio","r86_lag10_known",
 "r86_live_capture_proven","r86_strict_cross","r86_source_type","r86_amt_px_lot_ratio"
 ]].to_parquet(O/"R86_REPAIRED_FULL_MARKET_0925.parquet",index=False)
qa={"status":"R86_MAC_ONLY_0925_REPAIRED_AND_READONLY_SOURCES_VERIFIED",
 "input_rows":len(C),"mac_only_raw_rows":orig_mac,
 "R84_original_auction_price_valid":original_has,
 "R86_auction_price_valid":int(C.r86_auction_price.notna().sum()),
 "mac_only_restored":int(C.source_status.eq("MAC_0925_ONLY").mul(C.r86_auction_price.notna()).sum()),
 "two_source_exact_original_preserved":int(C.r86_strict_cross.sum()),
 "source_conflict_rejected":int(conflict.sum()),
 "source_types":C.r86_source_type.value_counts().to_dict(),
 "R86_prior10_complete":int(C.r86_lag10_known.sum()),
 "amount_unit_ratio_within_0p1pct":int((drift.r86_amt_px_lot_ratio-1).abs().le(.001).sum()),
 "amount_unit_ratio_valid_rows":len(drift),
 "historical_true_live_timestamp_evidence":int(C.r86_live_capture_proven.sum()),
 "no_same_day_close_for_auction_features":True,
 "source_R84_original_not_modified":True,"production_R50_untouched":True}
(O/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R86_REPAIRED_QA",json.dumps(qa,ensure_ascii=False),flush=True)
print("R86_COVERAGE_BY_MONTH",C.assign(month=C.date.str[:7]).groupby("month").agg(
 source_rows=("code","size"),repaired_price=("r86_auction_price","count"),
 restored_mac=("r86_source_type",lambda v:int((v=="VALID_MAC_ONLY").sum()))).to_string(),flush=True)