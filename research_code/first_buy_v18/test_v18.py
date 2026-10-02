from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
D=pd.read_parquet(ROOT/'v17_data/daily.parquet'); D['date']=pd.to_datetime(D.date)
E=pd.read_parquet(ROOT/'v17_results/events.parquet'); E['date']=pd.to_datetime(E.date)
C17=pd.read_parquet(ROOT/'v17_results/candidate_v17_trades.parquet')
L=pd.read_parquet(ROOT/'v18_results/lane_trades.parquet'); L['date']=pd.to_datetime(L.date)
X=pd.read_parquet(ROOT/'v18_results/extra_minute_primitives.parquet')
S=pd.read_parquet(ROOT/'v18_results/frozen_stack_trades.parquet')
summary=json.loads((ROOT/'v18_results/v18_summary.json').read_text())
case=json.loads((ROOT/'v18_results/casebook_summary_v18.json').read_text())
freeze=json.loads((ROOT/'v18_results/next_unseen_freeze.json').read_text())

# Price-basis reset guard is frozen and the known -38% ex-rights-like jump cannot become a D2 action.
assert int(D.price_reset.sum())==125
r=D[(D.code=='001388')&(D.date==pd.Timestamp('2026-07-17'))]
assert len(r)==1 and bool(r.iloc[0].price_reset)
e=E[(E.code=='001388')&(E.date==pd.Timestamp('2026-07-16'))]
assert len(e)==1 and bool(e.iloc[0].d2_price_reset)
assert not X.event_id.eq(e.iloc[0].event_id).any()

# V18 research lanes never promote R2 recycled two-board events.
assert not L.bucket.eq('R2').any()

# Minute-confirmed lanes are causal: signal minute is complete before the next-minute fill.
minute=L[L.entry_time.astype(str)!='OPEN']
assert (minute.entry_time.astype(str)>minute.signal_time.astype(str)).all()

# Frozen V17 CORE is carried byte-for-byte at the event/lane level, not re-fit in V18.
core17=set(C17.loc[C17.tier.eq('CORE'),'event_id'])
core18=set(L.loc[L.tier.eq('CORE_FROZEN'),'event_id'])
assert core17==core18 and len(core18)==19

# Snapshot counts for V18 structural lanes.
expected={
    'CORE_RED_4P_REPAIR45':11,
    'PANIC_3_LIQ_RECLAIM45':8,
    'POS_4P_VOLUME_RECLAIM35':11,
    'B3_NORMAL_DEEP_IMPULSE3':5,
    'B4P_PANIC_DEEP_IMPULSE5':5,
    'B3_NORMAL_FLUSH_REBOUND2':2,
    'POS_4P_DEEP_ABSORBED_OPEN':8,
}
counts=L.groupby('lane').size().to_dict()
for lane,n in expected.items():
    assert counts.get(lane)==n,(lane,counts.get(lane),n)

# Temporally-confirmed experimental lane must have both early-history and later-period observations.
pos=L[L.lane.eq('POS_4P_VOLUME_RECLAIM35')]
assert len(pos[pos.date<='2026-06-30'])==7
assert len(pos[pos.date>='2026-07-01'])==4
assert pos[pos.date<='2026-06-30'].t1_ret.mean()>0
assert pos[pos.date>='2026-07-01'].t1_ret.mean()>0

# Frozen research stack is CORE + the single temporally-confirmed experimental lane.
assert len(S)==30
assert set(S.tier.unique())=={'CORE_FROZEN','TEMPORALLY_CONFIRMED_EXPERIMENTAL'}

# Casebook is diagnostic only; V18 expands structural explanation without being used for rule selection.
assert case['casebook_events']==31
assert case['v17_candidate_events']==6
assert case['v17_or_v18_any_research_union_events']==12
assert case['new_events_explained_by_v18_structures']==6
assert case['unexplained_events']==19

# Freeze future validation before any post-2026-09-30 session is seen.
assert summary['data_through']=='2026-09-30'
assert freeze['data_through']=='2026-09-30'
assert freeze['no_retroactive_rule_change_after_next_session_seen'] is True
latest={x['code']:x for x in freeze['latest_watchlist']}
assert latest['002912']['plan']=='OBSERVE_ONLY'
assert latest['603949']['plan']=='WATCH_SMALL_N_B4P_PANIC_DEEP_IMPULSE5'

print('V18_TESTS_OK')