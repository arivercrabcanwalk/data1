from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=pd.read_parquet(ROOT/'v19_results/v18_v19_research_stack.parquet').copy()
BASE['date']=pd.to_datetime(BASE.date)
BASE['trade_ret']=BASE['ret']
LANE=pd.read_parquet(ROOT/'v20_results/v20_expansion_lane_trades.parquet').copy()
LANE['date']=pd.to_datetime(LANE.date)
STACK=pd.read_parquet(ROOT/'v20_results/v20_49_trade_stack.parquet').copy()
STACK['date']=pd.to_datetime(STACK.date)
S=json.loads((ROOT/'v20_results/v20_expansion_summary.json').read_text())

def m(df):
    r=df.trade_ret
    return len(r),(r>0).mean(),r.mean(),r.min()

# Frozen counts.
assert len(BASE)==41
assert len(LANE)==8
assert len(STACK)==49
assert not STACK.event_id.duplicated().any()

# Every V20 expansion event obeys the frozen causal rule.
assert LANE.branch.eq('NORMAL_RED').all()
assert LANE.bucket.eq('B4P').all()
assert (LANE.amount>=1.6e9).all()
assert (LANE.close_loc<=.50).all()
assert (LANE.m3_ret.abs()>=.01).all()
assert (LANE.m3_ret<=.03).all()
assert LANE.entry_time.eq('M4_OPEN').all()
assert (~LANE.d2_price_reset.fillna(False)).all()
assert (~LANE.exit_t1_reset.fillna(False)).all()
assert LANE.entry_below_upper.all()

# The added lane itself remains positive in both time slices.
tr=LANE[LANE.date<='2026-06-30']
te=LANE[LANE.date>='2026-07-01']
assert len(tr)==5 and (tr.trade_ret>0).all() and tr.trade_ret.mean()>.06
assert len(te)==3 and (te.trade_ret>0).all() and te.trade_ret.mean()>.09

# User's hard requirement: adding trades may not degrade return quality.
for label,base,stack in [
    ('all',BASE,STACK),
    ('train',BASE[BASE.date<='2026-06-30'],STACK[STACK.date<='2026-06-30']),
    ('pseudo',BASE[BASE.date>='2026-07-01'],STACK[STACK.date>='2026-07-01'])
]:
    bn,bw,bmean,bworst=m(base)
    sn,sw,smean,sworst=m(stack)
    assert sn>bn,label
    assert smean>=bmean,label
    assert sw>=bw,label
    assert sworst>=bworst,label

# Stress.
stress=S['lane']['stress']
assert stress['double_cost_all']['win_rate']==1.0
assert stress['double_cost_all']['mean']>.069
assert stress['drop_top1_all']['mean']>.06
assert stress['drop_top2_all']['mean']>.05

# Threshold neighborhood must not be a single-point knife edge.
nb=S['neighborhood']
for amount in [1.5e9,1.6e9,1.7e9,1.8e9]:
    xs=[x for x in nb if x['amount_min']==amount and x['close_loc_max']==.5 and x['m3_abs_min']==.01 and x['m3_upper_max']==.03]
    assert len(xs)==1
    x=xs[0]['all_2026']
    assert x['mean']>.058
    assert x['win_rate']>.75
    assert x['worst']>-.05

print('V20_EXPANSION_TESTS_OK')