from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
B=pd.read_parquet(ROOT/'v20_results/v20_49_trade_stack.parquet').copy();B['date']=pd.to_datetime(B.date)
V=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy();V['date']=pd.to_datetime(V.date)
S=json.loads((ROOT/'v21_results/v21_summary.json').read_text())

assert len(B)==49
assert len(V)==52
assert not V.event_id.duplicated().any()

r=V.final_ret
assert int((r>0).sum())==44
assert abs(float((r>0).mean())-0.8461538461538461)<1e-12
assert abs(float(r.mean())-0.07088381843990328)<1e-12
assert abs(float(r.median())-0.061803943260857794)<1e-12
assert abs(float(r.min())-(-0.03685033911077619))<1e-12
assert abs(float(r.sum())-3.6859585588749706)<1e-12

# Exactly three new non-overlap entry events versus V20.
new=set(V.event_id.astype(str))-set(B.event_id.astype(str))
assert len(new)==3
N=V[V.event_id.astype(str).isin(new)]
assert set(N.name)=={'宏柏新材','豪尔赛','澳弘电子'}
assert N.lane.eq('NORMAL_B4P_DOUBLE_WASHOUT_M5').all()

# Exit overlays are causal: signal minute precedes next-open execution.
assert V.loc[V.final_exit_rule.eq('T1_M2_OPEN_IF_M1_FROM_ENTRY_GT_8PCT'),'t1_signal_m1'].gt(.08).all()
assert V.loc[V.final_exit_rule.eq('T1_M6_OPEN_IF_M5_VS_T1_OPEN_GT_0.5PCT'),'t1_m5_vs_open'].gt(.005).all()
assert V.loc[V.final_exit_rule.eq('T1_M4_OPEN_IF_M3_VS_T1_OPEN_GT_5PCT'),'t1_m3_vs_open'].gt(.05).all()
assert V.loc[V.final_exit_rule.eq('T1_M4_OPEN_STOP_IF_M3_FROM_ENTRY_LT_-2PCT'),'t1_signal_m3'].lt(-.02).all()
assert V.loc[V.final_exit_rule.eq('T1_M4_OPEN_STOP_IF_M3_VS_T1_OPEN_LT_-1.5PCT'),'t1_m3_vs_open'].lt(-.015).all()

# User hard constraint: final bundle improves count, mean, win rate and worst trade in all time cuts.
def m(df,col):
    x=df[col]
    return len(x),(x>0).mean(),x.mean(),x.min()
for label,b,v in [
    ('all',B,V),
    ('train',B[B.date<='2026-06-30'],V[V.date<='2026-06-30']),
    ('pseudo',B[B.date>='2026-07-01'],V[V.date>='2026-07-01'])
]:
    bn,bw,bm,bworst=m(b,'trade_ret')
    vn,vw,vm,vworst=m(v,'final_ret')
    assert vn>bn,label
    assert vw>=bw,label
    assert vm>=bm,label
    assert vworst>=bworst,label

# Stress remains positive and robust.
st=S['stress']
assert st['double_cost_all']['win_rate']>.82
assert st['double_cost_all']['mean']>.065
assert st['double_cost_all']['worst']>-.043
assert st['drop_top1_all']['mean']>.065
assert st['drop_top2_all']['mean']>.063
assert st['drop_top3_all']['mean']>.061

# Strict post-52 search found no additional candidate that passed all gates.
post=json.loads((ROOT/'v21_results/post52_candidates.json').read_text())
assert post==[]

print('V21_TESTS_OK')