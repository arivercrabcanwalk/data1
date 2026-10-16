#!/usr/bin/env python3
"""R86 phase3 frozen 7 hypothesis rules x R81 2 exits x AM/PM/BOTH
=42 old R50+R79 capacity scenarios +168 segment trade outcomes.
Sector peer prices only at previous EOD and completed signal minute.
"""
from pathlib import Path
import json,runpy
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
O=R/"w08_intraday_leadership_transfer"
env=runpy.run_path(str(R/"w04_joint_account/account.py"))
oldsrc=env["oldsrc"];prev=env["prev"];B=env["B"];replay=env["replay"];base=env["reference"];T=env["T"]
F=pd.read_csv(O/"R86_563_AUCTION_TO_COMPLETED_MINUTE_SECTOR_LEADERSHIP_MIGRATION.csv",dtype={"code":str})
keys=["code","buy_date","branch","entry_time"]
hypotheses=["RAW_R81_ALL","ROTATE_FROM_WEAK","ROTATE_ANY","ROTATE_WITH_BREADTH","ROTATE_CORE20","STICKY_CORE_ONLY","ABSOLUTE_LEADER_CONTROL"]
assert len(F)==563 and all(a in F for a in hypotheses)
Q=T.merge(F[keys+hypotheses],on=keys,how="left",validate="m:1")
assert len(Q)==1126 and Q.RAW_R81_ALL.notna().all()
scenarios=[];accepted=[];independent=[]
for gate in hypotheses:
 for v in ["BASE_T3","T1_WEAK"]:
  for direction in ["BOTH","AM_REASSERT","PM_RECAPTURE"]:
   t=Q[Q[gate]&Q.variant.eq(v)]
   if direction!="BOTH":t=t[t.branch.eq(direction)]
   # Independent intent quality: same physical episode may produce AM&PM,
   # account only accepts earliest timed signal, never selects hindsight best.
   for period in ["FEB_JUN","JUL_AUG","SEP_TO23","ALL_PREVIOUSLY_SEEN"]:
    if period=="FEB_JUN":q=t[t.buy_date.le("2026-06-30")]
    elif period=="JUL_AUG":q=t[t.buy_date.between("2026-07-01","2026-08-31")]
    elif period=="SEP_TO23":q=t[t.buy_date.ge("2026-09-01")]
    else:q=t
    closed=q[q.status.eq("CLOSED")]
    independent.append({"gate":gate,"exit":v,"branch":direction,"period":period,
      "intents":len(q),"closed":len(closed),"wins":int(closed.return_net_pct.gt(0).sum()),
      "winrate_pct":100*closed.return_net_pct.gt(0).mean() if len(closed) else np.nan,
      "mean_net_pct":float(closed.return_net_pct.mean()) if len(closed) else np.nan,
      "MAE10":int(closed.MAE_pct.le(-10).sum()),"realized_loss10":int(closed.return_net_pct.le(-10).sum()),
      "censored":int(q.status.ne("CLOSED").sum()),"months":q.buy_date.str[:7].nunique(),
      "by_month":q.buy_date.str[:7].value_counts().sort_index().to_dict()})
   q=t.sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"],keep="first")
   cols=B.columns.intersection(q.columns)
   acc,rej=replay(pd.concat([oldsrc,prev,q[cols]],ignore_index=True))
   new=acc[acc.source=="R86"];prior=acc[acc.source!="R86"]
   closed=new[~new.censored]
   displaced=B[~B.refid.isin(prior.refid)]
   newprofit=float((new.weight*new.ret).sum()*100)
   olddelta=float((prior.weight*prior.ret).sum()*100-base)
   delta=float((acc.weight*acc.ret).sum()*100-base)
   assert abs(newprofit+olddelta-delta)<1e-8
   scenarios.append({"gate":gate,"exit":v,"branch":direction,
      "pre_capacity_stockday_intents":len(q),"new_accepted":len(new),"new_closed":len(closed),
      "winners":int(closed.ret.gt(0).sum()),
      "winrate_pct":100*closed.ret.gt(0).mean() if len(closed) else np.nan,
      "mean_net_pct":100*closed.ret.mean() if len(closed) else np.nan,
      "MAE10":int(new.mae.le(-.1).sum()),"loss10":int(closed.ret.le(-.1).sum()),
      "censored":int(new.censored.sum()),
      "old_A_displaced":int(((displaced.source=="R50")&(displaced.method=="A")).sum()),
      "old_B_displaced":int(((displaced.source=="R50")&(displaced.method=="B")).sum()),
      "old_R78_R79_displaced":int(displaced.source.isin(["R78","R79"]).sum()),
      "new_contrib_proxy_pp":newprofit,"old_displacement_proxy_pp":olddelta,
      "total_contrib_proxy_pp":base+delta,"delta_vs_frozen_old88_pp":delta,
      "monthly_new_trades":new.entry_dt.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()})
   key=gate+"_"+v+"_"+direction
   for x in new.to_dict("records"):
    accepted.append({"variant":key,"code":x["code"],"name":x["name"],
       "entry_dt":x["entry_dt"],"exit_dt":x["exit_dt"],"net_pct":100*x["ret"],
       "MAE_pct":100*x["mae"],"weight":x["weight"],"censored":x["censored"]})
df=pd.DataFrame(scenarios).sort_values("delta_vs_frozen_old88_pp",ascending=False)
assert len(df)==42 and len(independent)==168
df.to_csv(O/"R86_42_ROTATION_REGISTERED_R50_R79_JOINT_ACCOUNT.csv",index=False)
pd.DataFrame(independent).to_csv(O/"R86_168_ROTATION_MONTHLY_INDEPENDENT_TRADES.csv",index=False)
pd.DataFrame(accepted).to_csv(O/"R86_ROTATION_42_ACCEPTED_TRADE_DETAILS.csv",index=False)
main=df[(df.gate=="RAW_R81_ALL")&(df.exit=="BASE_T3")&(df.branch=="BOTH")].iloc[0]
old= pd.read_csv(R/"w04_joint_account/R86_66_COMPLETE_FIXED_R50_R79_ACCOUNT_SCENARIOS.csv")
old=old[(old.gate=="G0_ALL")&(old.exit=="BASE_T3")&(old.branch=="BOTH")].iloc[0]
assert main.new_accepted==old.new_accepted and abs(main.delta_vs_frozen_old88_pp-old.delta_vs_frozen_old88_pp)<1e-8
QA={"status":"R86_ROTATION_42_ALL_REGISTERED_R50_R79_PARITY_RESEARCH",
 "all_hypothesis_scenarios":len(df),"segment_rows":len(independent),
 "R81_baseline_R50_R79_parity":True,"old88_contribution_proxy":base,
 "all_original_563_R81_signal_universe_kept":True,
 "minutes_only_completed_signal_lookback":True,
 "source_R21_family_capture_for_history_not_real_time_proven":True,
 "rules_created_after_viewing_known_Xinhua_3_positive_cases":True,
 "non_blind_historical":True,"no_prod_writes":True}
(O/"account_qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))
print("R86_ROTATE_ACCOUNT_QA",json.dumps(QA,ensure_ascii=False),flush=True)
print("R86_ROTATE_ACCOUNT_BOTH_BASE",df[(df.exit=="BASE_T3")&(df.branch=="BOTH")].drop(columns="monthly_new_trades").round(4).to_string(index=False),flush=True)
print("R86_ROTATE_ACCOUNT_TOP12",df.head(12).drop(columns="monthly_new_trades").round(4).to_string(index=False),flush=True)