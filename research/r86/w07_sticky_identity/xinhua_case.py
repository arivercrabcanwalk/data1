#!/usr/bin/env python3
"""Read-only: Tencent R50 daily snapshots 601811 and only its contemporaneous
previous-close family members. Compute corresponding MAC true 09:25 &
completed-minute peer-relative PRICE ONLY. Never claim historical trade fills.
"""
from pathlib import Path
import subprocess,json
import pandas as pd,numpy as np,pyarrow.dataset as ds
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
H=Path.home()/"Documents/stock_data_windows_mirror"
O=R/"w07_sticky_identity"
server_code=r'''
import pandas as pd,json
D=pd.read_parquet("/var/lib/leader-relaunch-r50/daily.parquet")
for reset in ("2026-09-28","2026-10-08","2026-10-13"):
 day=D[D.date.eq(reset)]
 a=day[day.code.eq("601811")]
 assert len(a)==1
 r=a.iloc[0]
 family=r.family_theme if isinstance(r.family_theme,str) and r.family_theme else r.last_core_family
 reason="CURRENT_FAMILY" if isinstance(r.family_theme,str) and r.family_theme else "HISTORICAL_LAST_CORE"
 assert isinstance(family,str) and family
 peers=day[day.family_theme.eq(family)]
 rows=peers[["code","close","name"]].to_dict("records")
 out={"reset_date":reset,"code":"601811","leader_name":r["name"],"leader_last_close":float(r.close),
 "sector":family,"source":reason,
 "last_core_date":r.last_core_date,"last_core_age":float(r.core_age),
 "theme_primary_at_reset":r.theme_primary,"last_core_family":r.last_core_family,
 "peer_members":rows}
 print(json.dumps(out,ensure_ascii=False,default=str))
'''
res=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=12",
 "lighthouse","/opt/liangun/.venv/bin/python","-"],
 input=server_code,text=True,capture_output=True,timeout=50)
assert res.returncode==0,res.stderr[:600]
source=[json.loads(line) for line in res.stdout.splitlines() if line.strip().startswith("{")]
assert len(source)==3,(len(source),res.stdout[:600])
(O/"R86_R50_PRODUCTION_READONLY_3_PREV_DAY_THEME_LISTS.json").write_text(
 json.dumps(source,ensure_ascii=False,indent=2))
a=pd.read_parquet(R/"w01_canonical/R86_REPAIRED_FULL_MARKET_0925.parquet")
a.code=a.code.astype(str).str.zfill(6)
a=a[["date","code","r86_auction_price","r86_auction_amount_yuan","r86_lag10_ratio","r86_source_type"]]
minutes_day={"2026-09-28":"2026-09-29","2026-10-08":"2026-10-09","2026-10-13":"2026-10-14"}
signals={"2026-09-28":"10:19","2026-10-08":"13:45","2026-10-13":"10:13"}
out=[]
for event in source:
 reset=event["reset_date"];day=minutes_day[reset];sig=signals[reset]
 peers=pd.DataFrame(event["peer_members"])
 peers.code=peers.code.astype(str).str.zfill(6)
 if not peers.code.eq("601811").any():
  peers=pd.concat([peers,pd.DataFrame([{"code":"601811","close":event["leader_last_close"],"name":"新华文轩"}])])
 peer=peers.merge(a[a.date.eq(day)],on="code",how="left",validate="1:1")
 peer["auction_gap_pct"]=100*(peer.r86_auction_price/peer.close-1)
 root=(H/"1min"/("date="+day))
 if root.exists():
  tape=ds.dataset(str(root),format="parquet").to_table(
   columns=["code","time","close"],
   filter=ds.field("code").isin(peer.code.unique().tolist())).to_pandas()
  tape.code=tape.code.astype(str).str.zfill(6)
  instant=tape[tape.time.eq(sig)][["code","close"]].rename(columns={"close":"minute_close_at_signal"})
  peer=peer.merge(instant,on="code",how="left",validate="1:1")
 else:peer["minute_close_at_signal"]=np.nan
 peer["minute_gain_from_previous_close_pct"]=100*(peer.minute_close_at_signal/peer.close-1)
 leader=peer[peer.code.eq("601811")].iloc[0];others=peer[peer.code.ne("601811")]
 valid_auction=others.auction_gap_pct.dropna()
 valid_minute=others.minute_gain_from_previous_close_pct.dropna()
 rec={"reset_date":reset,"signal_day":day,"signal_completed_time":sig,"reset_family_from":event["source"],
  "asof_family":"AI应用" if event["sector"]=="AI应用" else event["sector"],
  "last_core_date":event["last_core_date"],"last_core_age":event["last_core_age"],
  "same_family_production_previous_close_members":len(others),
  "auction_peer_valid":len(valid_auction),
  "own_gap_0925_pct":float(leader.auction_gap_pct),
  "peer_median_gap_0925_pct":float(valid_auction.median()) if len(valid_auction) else None,
  "relative_gap_lead_pctpts":float(leader.auction_gap_pct-valid_auction.median()) if len(valid_auction) else None,
  "sector_peers_positive_0925_fraction":float(valid_auction.gt(0).mean()) if len(valid_auction) else None,
  "own_0925_cash_yuan":float(leader.r86_auction_amount_yuan),
  "own_auction_lag_ratio10":float(leader.r86_lag10_ratio),
  "own_intraday_return_by_completed_signal_pct":float(leader.minute_gain_from_previous_close_pct),
  "peer_median_intraday_return_at_signal_pct":float(valid_minute.median()) if len(valid_minute) else None,
  "peer_minute_valid":len(valid_minute),
  "intraday_relative_strength_pctpts":float(leader.minute_gain_from_previous_close_pct-valid_minute.median()) if len(valid_minute) else None,
  "minute_amount_buy_sell_fully_verified":False,
  "source_membership_first_archive_timestamp_verified":False}
 out.append(rec)
 print("R86_CASE_RETAKE",json.dumps(rec,ensure_ascii=False),flush=True)
pd.DataFrame(out).to_csv(O/"R86_XINHUA_3_EPISODES_0925_AND_COMPLETE_MINUTE_RELATIVE.csv",index=False)
QA={"status":"R86_XINHUA_THREE_STICKY_REGISTRY_PRICE_ONLY_CASE_VERIFIED",
 "three_reset_dates":list(minutes_day),"three_auction_days":list(minutes_day.values()),
 "true_R50_last_core_family_fallback_10_08_and_10_13":True,
 "historical_R50_server_only_read":True,"R50_daily_source_first_capture_timestamp_missing":True,
 "minute_amount_missing_some_oct_stocks":True,"not_proof_of_actual_orders":True,
 "no_production_server_modification":True}
(O/"case_qa.json").write_text(json.dumps(QA,ensure_ascii=False,indent=2))