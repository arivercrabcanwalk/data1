from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
T=pd.read_parquet(ROOT/'v19_results/b4p_panic_timing_table.parquet'); T['date']=pd.to_datetime(T.date)
D=pd.read_parquet(ROOT/'v19_results/v19_panic_decision_tree_trades.parquet'); D['date']=pd.to_datetime(D.date)
C=pd.read_parquet(ROOT/'v19_results/v18_v19_research_stack.parquet'); C['date']=pd.to_datetime(C.date)
S=json.loads((ROOT/'v19_results/v19_summary.json').read_text())
F=json.loads((ROOT/'v19_results/next_unseen_freeze_v19.json').read_text())

assert len(T)==32

open_lane=D[D.lane.eq('B4P_PANIC_CAPITULATION_OPEN')]
m5_lane=D[D.lane.eq('B4P_PANIC_DEEP_IMPULSE5')]

# New direct-open research branch.
assert len(open_lane)==9
assert len(open_lane[open_lane.date<='2026-06-30'])==6
assert len(open_lane[open_lane.date>='2026-07-01'])==3
assert (open_lane.t1_ret>0).mean()>0.70
assert open_lane.t1_ret.mean()>0.08
assert open_lane.t1_ret.min()>-.02
assert (open_lane.d1_close_vwap<=.965).all()
assert open_lane.entry_time.eq('OPEN').all()

# Waiting does not improve this exact historical subset.
wait=S['open_lane']['wait_comparison']
assert wait['open_ret']['all_2026']['mean'] > wait['ret_after_m1']['all_2026']['mean']
assert wait['open_ret']['all_2026']['mean'] > wait['ret_after_m3']['all_2026']['mean']
assert wait['open_ret']['all_2026']['mean'] > wait['ret_after_m5']['all_2026']['mean']
assert wait['open_ret']['all_2026']['worst'] > wait['ret_after_m1']['all_2026']['worst']
assert wait['open_ret']['all_2026']['worst'] > wait['ret_after_m3']['all_2026']['worst']

# Threshold neighborhood is not a single-point cliff.
nb={round(x['threshold'],4):x for x in S['open_lane']['threshold_neighborhood']}
for th in [0.9625,0.965,0.9675]:
    x=nb[th]
    assert x['train_jan_jun']['mean']>0
    assert x['pseudo_oos_jul_sep']['mean']>0

# 5-minute fallback remains only for events outside the open lane.
assert len(m5_lane)==2
assert (m5_lane.d1_close_vwap>.965).all()
assert (m5_lane.d2_gap<=-.03).all()
assert (m5_lane.m5_ret_from_open>=.03).all()
assert m5_lane.m5_above_vwap.all()

# Combined B4P PANIC decision tree.
assert len(D)==11
assert (D.t1_ret<=-.03).sum()==0
assert D.t1_ret.mean()>0.07

# Combined V18+V19 research stack has no duplicate events.
assert len(C)==41
assert not C.event_id.duplicated().any()

# Latest Snow Dragon classification is frozen before next unseen session.
latest={x['code']:x for x in F['latest_watchlist']}
snow=latest['603949']
assert snow['plan']=='WATCH_V19_OPEN_CAPITULATION'
assert snow['day1_close_vwap']<=.965
assert F['data_through']=='2026-09-30'
assert F['no_retroactive_changes_after_next_session_seen'] is True

print('V19_TESTS_OK')