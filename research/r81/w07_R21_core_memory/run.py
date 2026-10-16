#!/usr/bin/env python3
"""R81 Phase3 formal original R21 certified leader-memory vs pure price identity.
Frozen all 108 scenarios in prior contract. In sample, not new OOS.
"""
from pathlib import Path
import runpy,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent
O=ROOT/"w07_R21_core_memory";L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
assert (ROOT/"PHASE3_R21_PURE_CORE_MEMORY_FROZEN.md").exists()
x=runpy.run_path(str(ROOT/"w04_portfolio/replay_account.py"))
oldsrc=x["oldsrc"];prev=x["prev"]; B=x["B"];replay=x["replay"];base=x["comp"];raw=x["all_new"]
R=pd.read_parquet(L/"r21_true_height_rebuild_20260927/daily_r21.parquet",columns=[
  "code","date","core_age","last_core_date","last_core_lane","family_stage_validation","abs_height_validation"])
R.code=R.code.astype(str).str.zfill(6)
assert not R.duplicated(["code","date"]).any()
R=R.rename(columns={"date":"reset_date"})
Z=raw.merge(R,on=["code","reset_date"],how="left",validate="m:1")
assert not Z.duplicated(["code","reset_date","entry_time","branch","variant"]).any()
lanes={"ABS_HEIGHT","FAMILY_STAGE"}
asof=pd.to_datetime(Z.last_core_date,errors="coerce").le(pd.to_datetime(Z.reset_date))
lane=Z.last_core_lane.isin(lanes)
Z["CORE10"]=asof&lane&Z.core_age.between(0,10)
Z["CORE20"]=asof&lane&Z.core_age.between(0,20)
assert not (Z.CORE10&~Z.CORE20).any()
MEM=pd.read_csv(ROOT/"w06_rearm_memory/R81_PRIOR_SUCCESS_MEMORY_EACH_RESET.csv",dtype={"code":str})
Z=Z.merge(MEM[["code","reset_date","LOOP1","LOOP2","past_success_evidence"]],
          on=["code","reset_date"],how="left",validate="m:1")
assert Z.LOOP1.notna().all() and Z.CORE10.notna().all()
Z[["code","reset_date","buy_date","branch","entry_time","variant",
   "CORE10","CORE20","LOOP1","LOOP2","P4","P6","last_core_date","last_core_lane",
   "core_age","past_success_evidence","status","return_net_pct","MAE_pct"]].to_csv(
    O/"R81_asof_R21_lifecycle_membership_each_signal.csv",index=False)
print("W07_CANDIDATE_COUNTS",len(Z)//2,"CORE10",int(Z.loc[Z.variant.eq("BASE_T3"),"CORE10"].sum()),
      "CORE20",int(Z.loc[Z.variant.eq("BASE_T3"),"CORE20"].sum()),flush=True)
cols=B.columns.intersection(Z.columns)
output=[];trades=[]
for co in ["CORE10","CORE20"]:
 for topic in ["BASE","P4","P6"]:
  for direction in ["BOTH","AM_REASSERT","PM_RECAPTURE"]:
   for exitstyle in ["BASE_T3","T1_WEAK"]:
    for loop in ["ANY","LOOP1","LOOP2"]:
     src=Z[Z[co]&Z[topic]&Z.variant.eq(exitstyle)]
     if direction!="BOTH":src=src[src.branch.eq(direction)]
     if loop!="ANY":src=src[src[loop]]
     # No future profit or MAE dedup. Earliest valid entry after signal.
     src=src.sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"])
     ac,rj=replay(pd.concat([oldsrc,prev,src[cols]],ignore_index=True))
     fresh=ac[ac.source.eq("R81")];closed=fresh[~fresh.censored]
     oldok=ac[ac.source.ne("R81")];oldlost=B[~B.refid.isin(oldok.refid)]
     newp=float((fresh.ret*fresh.weight).sum()*100)
     oldp=float((oldok.ret*oldok.weight).sum()*100-base)
     delta=float((ac.ret*ac.weight).sum()*100-base)
     assert abs(newp+oldp-delta)<1e-9
     out={"CORE":co,"topic":topic,"branch":direction,"exit":exitstyle,"loop":loop,
          "pre_capacity":len(src),"new_accepted":len(fresh),"closed":len(closed),
          "winners":int(closed.ret.gt(0).sum()),
          "win_rate_pct":100*closed.ret.gt(0).mean() if len(closed) else None,
          "mean_net_closed_pct":100*closed.ret.mean() if len(closed) else None,
          "MAE10":int(fresh.mae.le(-.10).sum()),"realized_loss10":int(closed.ret.le(-.10).sum()),
          "censored":int(fresh.censored.sum()),"unique_tickers":fresh.code.nunique(),
          "old_A_lost":int(((oldlost.source=="R50")&(oldlost.method=="A")).sum()),
          "old_B_lost":int(((oldlost.source=="R50")&(oldlost.method=="B")).sum()),
          "old_R78_R79_lost":int(oldlost.source.isin(["R78","R79"]).sum()),
          "old_reduction_profit_delta_pp":oldp,
          "new_profit_pp":newp,"total_profit_delta_pp":delta,
          "accepted_months":int(fresh.entry_dt.dt.to_period("M").nunique()),
          "accepted_month_hist":fresh.entry_dt.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()}
     output.append(out)
     tag=co+"_"+topic+"_"+direction+"_"+exitstyle+"_"+loop
     for z in fresh.to_dict("records"):
      trades.append({"variant":tag,"code":z["code"],"name":z["name"],
                     "entry_dt":z["entry_dt"],"exit_dt":z["exit_dt"],
                     "net_return_pct":z["ret"]*100,"mae_pct":z["mae"]*100,
                     "weight":z["weight"],"censored":z["censored"]})
D=pd.DataFrame(output).sort_values("total_profit_delta_pp",ascending=False)
assert len(D)==108,(len(D),D.shape)
D.to_csv(O/"R81_R21_AUTHENTIC_CORE_108_SCENARIO_MATRIX.csv",index=False)
pd.DataFrame(trades).to_csv(O/"R81_R21_AUTHENTIC_CORE_ACCEPTED_ALL.csv",index=False)
Q={"status":"R81_R21_CORE_ASOF_CERTIFICATION_STUDY_COMPLETED",
 "base_return_proxy":base,"CORE10_signal_count":int(Z.loc[Z.variant.eq("BASE_T3"),"CORE10"].sum()),
 "CORE20_signal_count":int(Z.loc[Z.variant.eq("BASE_T3"),"CORE20"].sum()),
 "source_R21_core_date_not_in_future":bool(asof[Z.core_age.notna()].all()),
 "stage_labels_genuinely_from_R21_original":True,
 "prior_PIT_no_same_current_day_outcomes":True,
 "all_108_scenarios_exhaustively_reported":True,
 "strict_OOS_proof":False,
 "original_core_data_ends":"2026-09-23",
 "study_after_R81_negative_history_viewed":True,
 "old_R50_A_B_unmodified":True,"no_deploy":True}
(O/"qa.json").write_text(json.dumps(Q,ensure_ascii=False,indent=2))
print("W07_R21_QA",json.dumps(Q,ensure_ascii=False),flush=True)
print("W07_SCENARIO_BEST20",D.head(20).drop(columns="accepted_month_hist").round(4).to_string(index=False),flush=True)
print("W07_SCENARIO_WORST5",D.tail(5).drop(columns="accepted_month_hist").round(4).to_string(index=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]