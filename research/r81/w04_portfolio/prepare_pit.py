#!/usr/bin/env python3
"""R81 W04: only pre-existing R78/R79 09:25 PIT P4/P6 gates on new recurrent resets.
No factor threshold tuned on R81 returns; reports raw vs PIT; fail closed on no data.
"""
from pathlib import Path
import json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent.parent
O=ROOT/"w04_portfolio"
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
S=pd.read_csv(ROOT/"w03_causal_signal/raw_r81_2clock_signals.csv",dtype={"code":str})
T=pd.read_csv(ROOT/"w03_causal_signal/raw_r81_independent_exits.csv",dtype={"code":str})
assert len(S)>400
P=pd.read_parquet(L/"r3_loophole_closure_20260926/results/pit_theme_features.parquet")
P.code=P.code.astype(str).str.zfill(6)
assert not P.duplicated(["code","date"]).any()
keep=["code","date","prev_theme","theme_n","theme_open_pos","prior_sector_rank",
      "prev_followers","prev_core_valid","prev_core_age",
      "prior_sector_lu3","prev_family_rank2","prev_height_edge"]
X=S.merge(P[keep],left_on=["code","buy_date"],right_on=["code","date"],how="left",validate="m:1")
X["P4"]=X.theme_n.ge(5)&X.theme_open_pos.ge(.50)&X.prior_sector_rank.le(3)&X.prev_followers.ge(2)
X["P6"]=X.theme_n.ge(8)&X.theme_open_pos.ge(.45)&X.prior_sector_rank.le(2)&X.prev_followers.ge(3)
X["PIT_avail"]=X.theme_n.notna()&X.theme_open_pos.notna()&X.prior_sector_rank.notna()
X["STRICT_LIVE_CORE"]=X.prev_core_valid.fillna(False).astype(bool)
X["BASE"]=True
X.drop(columns="date").to_csv(O/"R81_signals_with_prior_PIT.csv",index=False)
k=["code","buy_date","branch","entry_time"]
assert not X.duplicated(k).any()
M=X.merge(T,on=k,how="inner",validate="1:m",suffixes=("_signal",""))
assert len(M)==len(T),(len(M),len(T))
assert M.groupby(k).variant.nunique().eq(2).all()
assert (M.groupby(k).size()==2).all()
M.to_csv(O/"R81_trades_with_frozen_PIT.csv",index=False)
closed=M[M.status=="CLOSED"].copy()
closed["month"]=closed.buy_date.str[:7]
summary=[]
for gate in ("BASE","P4","P6","STRICT_LIVE_CORE"):
 for v in ("BASE_T3","T1_WEAK"):
  x=closed[(closed[gate])&(closed.variant==v)]
  for br in ("AM_REASSERT","PM_RECAPTURE","BOTH"):
   z=x if br=="BOTH" else x[x.branch==br]
   summary.append({"gate":gate,"exit":v,"branch":br,"accepted_without_portfolio":len(z),
                   "closed_wins":int(z.return_net_pct.gt(0).sum()),
                   "win_rate_pct":100*z.return_net_pct.gt(0).mean() if len(z) else None,
                   "avg_net_pct":float(z.return_net_pct.mean()) if len(z) else None,
                   "mae10":int(z.MAE_pct.le(-10).sum()),"loss10":int(z.return_net_pct.le(-10).sum()),
                   "unique_stocks":z.code.nunique(),"months":z.month.nunique(),
                   "per_month":z.groupby("month").size().to_dict()})
pd.DataFrame(summary).to_csv(O/"R81_PIT_gate_unportfolioed_results.csv",index=False)
F=M[M.variant=="BASE_T3"].copy()
F["at_risk_over10"]=F.MAE_pct.le(-10)
F["large_final_loss"]=F.return_net_pct.le(-10)
F.to_csv(O/"R81_PIT_trade_forensics.csv",index=False)
qa={"status":"W04_PRIOR_REGISTERED_PIT_REPLAY_DONE","inputs_raw_signals":len(S),
    "merged_scenario_rows":len(M),"PIT_available":int(X.PIT_avail.sum()),
    "P4_signals":int(X.P4.sum()),"P6_signals":int(X.P6.sum()),
    "actual_strict_core_true":int(X.STRICT_LIVE_CORE.sum()),
    "P4_and_P6_are_prior_preexisting_gates":True,
    "PIT_project_curated_not_full_market_true_leadership":True,
    "do_not_promote_on_seen_data":True,
    "2026_Oct_09_theme_never_verified":True,
    "no_production_change":True}
(O/"PIT_QA.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R81_PIT_QA",json.dumps(qa,ensure_ascii=False),flush=True)
print("R81_PIT_GATE_SUMMARY",pd.DataFrame(summary)[lambda z:(z.exit=="BASE_T3")&(z.branch=="BOTH")].drop(columns="per_month").to_string(index=False),flush=True)

[executed on device: zhanghaodeMacBook-Air-3.local (3d784e4c-30b2-405d-b841-683054ab2ad9)]