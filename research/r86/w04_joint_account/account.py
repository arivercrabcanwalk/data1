#!/usr/bin/env python3
"""R86 isolate old account simulation using function-only AST import from R81.
Import replay() only, never run R81 module -> NO writes outside R86.
All old accepted/rejected orders remain in input, no hindsight stock deletion.
"""
import ast,json
from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
OUT=R/"w04_joint_account"
# Extract and execute the ORIGINAL UNMODIFIED R81 capacity rule function only.
source=L/"r81_recurrent_reversal_20261016/w04_portfolio/replay_account.py"
tree=ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="replay"]
assert len(nodes)==1
scope={"pd":pd}
exec(compile(ast.Module(body=nodes,type_ignores=[]),filename=str(source),mode="exec"),scope)
replay=scope["replay"]
oldfolder=L/"r60_capacity_stress_20261008"
old=pd.concat([pd.read_csv(oldfolder/"deploy_pilot_accepted.csv",dtype={"code":str}),
 pd.read_csv(oldfolder/"deploy_pilot_rejected.csv",dtype={"code":str})],ignore_index=True)
assert len(old)==76 and old.uid.is_unique
oldsrc=pd.DataFrame({"code":old.code,"name":old.name,"entry_dt":pd.to_datetime(old.entry_dt),
 "exit_dt":pd.to_datetime(old.exit_dt),"method":old.method,"weight":old.weight.astype(float),
 "ret":old.ret.astype(float),"priority":old.priority.astype(int),"score1":old.score1.astype(float),
 "score2":old.score2.astype(float),"source":"R50","refid":old.uid,
 "censored":False,"route":old.route,"mae":np.nan,"id_group":"LEGACY"})
older=pd.read_csv(L/"r79_volume_quality_20261016/w14_phase3/accepted_trades.csv",dtype={"code":str})
older=older[older.variant.eq("P4_B_H_E_D_G")].copy()
assert len(older)==33 and older[["code","entry"]].duplicated().sum()==0
prev=pd.DataFrame({"code":older.code,"name":older.name,"entry_dt":pd.to_datetime(older.entry),
 "exit_dt":pd.to_datetime(older.exit),"method":older.source.map(lambda x:str(x)+"_B"),
 "weight":older.weight.astype(float),"ret":older.ret.astype(float),
 "priority":older.source.map({"R78":3,"R79":4}),"score1":0.,"score2":0.,
 "source":older.source,
 "refid":older.source+"|"+older.code+"|"+older.entry.astype(str)+"|"+older.route,
 "censored":older.censored.astype(bool),"route":older.route,"mae":older.mae.astype(float),
 "id_group":"OLD_RESEARCH_ACCEPTED_ONLY"})
assert prev.refid.is_unique
B,_=replay(pd.concat([oldsrc,prev],ignore_index=True))
reference=float((B.ret*B.weight).sum()*100)
assert len(B)==88 and int(B.source.eq("R50").sum())==55
assert abs(reference-57.01407796050304)<1e-8,reference
T=pd.read_csv(R/"w03_price_signals/R86_all_1126_trade_outcomes_fixed_gates.csv",dtype={"code":str})
F=pd.read_csv(R/"w02_sector_peers/R86_563_intraday_signals_and_fixed_relative_auction_gates.csv",dtype={"code":str})
rules=sorted(c for c in F.columns if c.startswith("G") and c[1].isdigit() and "_" in c)
assert len(rules)==11
assert len(T)==1126 and T.refid.notna().sum()==0 if "refid" in T else len(T)==1126
T["entry_dt"]=pd.to_datetime(T.buy_date+" "+T.entry_time)
T["exit_dt"]=pd.to_datetime(T.exit_date.fillna("2026-09-23")+" "+T.exit_time.fillna("15:00"))
T.loc[T.status.eq("RIGHT_CENSORED"),"exit_dt"]=pd.Timestamp("2026-09-23 15:00")
T["ret"]=T.return_net_pct.fillna(0)/100
T["weight"]=T.entry_weight;T["priority"]=5
T["score1"]=0.;T["score2"]=0.
T["source"]="R86";T["method"]="R86_B"
T["censored"]=T.status.ne("CLOSED")
T["mae"]=T.MAE_pct/100
T["refid"]=T.code+"|"+T.buy_date+"|"+T.entry_time+"|"+T.branch+"|"+T.variant
T["route"]=T.branch;T["id_group"]="R86_SECTOR_RELATIVE"
assert T.refid.is_unique
rows=[];accept=[];reject=[]
for gate in rules:
 for exitvariant in ("BASE_T3","T1_WEAK"):
  for direction in ("BOTH","AM_REASSERT","PM_RECAPTURE"):
   q=T[T[gate]&T.variant.eq(exitvariant)]
   if direction!="BOTH":q=q[q.branch.eq(direction)]
   q=q.sort_values(["entry_dt","code","branch"]).drop_duplicates(["code","buy_date"],keep="first")
   order_cols=B.columns.intersection(q.columns)
   account=pd.concat([oldsrc,prev,q[order_cols]],ignore_index=True)
   taken,rej=replay(account)
   recent=taken[taken.source=="R86"];prior=taken[taken.source!="R86"]
   baseline_lost=B[~B.refid.isin(prior.refid)]
   closed=recent[~recent.censored]
   newprofit=float((recent.weight*recent.ret).sum()*100)
   old_delta=float((prior.weight*prior.ret).sum()*100-reference)
   total=float((taken.weight*taken.ret).sum()*100)
   assert abs(total-reference-newprofit-old_delta)<1e-9
   outcome={"gate":gate,"exit":exitvariant,"branch":direction,
    "raw_unique_day_intents":len(q),"new_accepted":len(recent),"new_closed":len(closed),
    "new_wins":int(closed.ret.gt(0).sum()),
    "winrate_pct":100*closed.ret.gt(0).mean() if len(closed) else np.nan,
    "avg_net_pct":100*closed.ret.mean() if len(closed) else np.nan,
    "new_mae10":int(recent.mae.le(-.1).sum()),"new_loss10":int(closed.ret.le(-.1).sum()),
    "new_censored":int(recent.censored.sum()),
    "old_A_displaced":int(((baseline_lost.source=="R50")&(baseline_lost.method=="A")).sum()),
    "old_B_displaced":int(((baseline_lost.source=="R50")&(baseline_lost.method=="B")).sum()),
    "old_R78_R79_displaced":int(baseline_lost.source.isin(["R78","R79"]).sum()),
    "old_reduced_A":int(((prior.source=="R50")&(prior.method=="A")&prior.reduced_weight.gt(1e-5)).sum()),
    "new_contribution_pp":newprofit,"old_contribution_delta_pp":old_delta,
    "combined_weighted_contribution_pp":total,
    "delta_vs_frozen_old88_pp":total-reference,
    "months":int(recent.entry_dt.dt.to_period("M").nunique()),
    "monthly_count":recent.entry_dt.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()}
   rows.append(outcome)
   key=gate+"_"+exitvariant+"_"+direction
   for t in recent.to_dict("records"):
    accept.append({"scenario":key,"code":t["code"],"name":t["name"],
     "entry_dt":t["entry_dt"],"exit_dt":t["exit_dt"],"return_net_pct":100*t["ret"],
     "MAE_pct":100*t["mae"],"weight":t["weight"],"censored":t["censored"],
     "weighted_contribution_pp":100*t["ret"]*t["weight"]})
   for t in rej[rej.source=="R86"].to_dict("records"):
    reject.append({"scenario":key,"code":t["code"],"entry_dt":t["entry_dt"],"reason":t.get("reason")})
Z=pd.DataFrame(rows).sort_values("delta_vs_frozen_old88_pp",ascending=False)
assert len(Z)==66
par=Z[(Z.gate=="G0_ALL")&(Z.exit=="BASE_T3")&(Z.branch=="BOTH")].iloc[0]
# R84 raw model's no-auction pricing: same underlying opportunities and
# same exact official old 88 account; restored historical quotes should not
# change this completely unfiltered reference.
oldr84=pd.read_csv(L/"r84_auction_retake_20261016/w05_joint_account/R84_AU0_AU9_60_frozen_joint_account_comparisons.csv")
p=oldr84[(oldr84.gate=="AU0_ALL")&(oldr84.variant=="BASE_T3")&(oldr84.branch=="BOTH")].iloc[0]
assert int(par.new_accepted)==int(p.new_accepted),(par.new_accepted,p.new_accepted)
assert abs(par.delta_vs_frozen_old88_pp-p.delta_vs_old_88_pp)<1e-9
Z.to_csv(OUT/"R86_66_COMPLETE_FIXED_R50_R79_ACCOUNT_SCENARIOS.csv",index=False)
pd.DataFrame(accept).to_csv(OUT/"R86_ACCOUNT_ACCEPTED_NEW_ALL.csv",index=False)
pd.DataFrame(reject).to_csv(OUT/"R86_ACCOUNT_REJECTED_NEW_ALL.csv",index=False)
B.to_csv(OUT/"R86_FROZEN_OLD_88_ACCOUNT.csv",index=False)
QA={"status":"R86_ALL_66_COMPARABLE_R50_PLUS_R79_CAPACITY_ACCOUNT_QA_PASS",
 "baseline_accepted":len(B),"old_R50":int(B.source.eq("R50").sum()),
 "old_R78":int(B.source.eq("R78").sum()),"old_R79":int(B.source.eq("R79").sum()),
 "baseline_88_weighted_profit_proxy_pp":reference,
 "G0_exact_parity_with_old_R84":True,"all_11_gates_tested":True,
 "branches_exits_tested":66,"all_original_A_intents_kept":True,
 "single_same_ticker_and_A_priority":True,"B_limit_15percent":True,
 "gross_55percent_5_stock":True,"broker_NAV_simulation":False,
 "strict_OOS":False,"source_R81_and_R84_untouched":True,
 "production_R50_not_modified":True}
(OUT/"qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))
print("R86_ACCOUNT_QA",json.dumps(QA,ensure_ascii=False),flush=True)
print("R86_ACCOUNT_BOTH_BASE_T3",Z[(Z.exit=="BASE_T3")&(Z.branch=="BOTH")].drop(columns="monthly_count").round(4).to_string(index=False),flush=True)
print("R86_ACCOUNT_TOP12",Z.head(12).drop(columns="monthly_count").round(4).to_string(index=False),flush=True)