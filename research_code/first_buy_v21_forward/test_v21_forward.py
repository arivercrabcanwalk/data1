# -*- coding: utf-8 -*-
"""Unit tests for freeze discipline and causal V21 day2/T1 decisions."""
import math
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import v21_forward_engine as f

def bars(price=10.0):
    times=[f'{9+(30+i)//60:02d}:{(30+i)%60:02d}' for i in range(1,121)]
    times += [f'{13+i//60:02d}:{i%60:02d}' for i in range(1,121)]
    assert len(times)==240 and times[0]=='09:31' and times[-1]=='15:00'
    s=pd.DataFrame({'time':times,'open':[price]*240,'high':[price]*240,
                    'low':[price]*240,'close':[price]*240,'volume':[100000]*240,
                    'amount':[price*100000]*240})
    return s

def row(branch='NORMAL_RED',bucket='B4P',event_type='PRIMARY_3PLUS',height=4,amt=2e9):
    return pd.Series({'event_id':'600123_20261008_g0','code':'600123','name':'合成测试',
      'date':pd.Timestamp('2026-10-08'),'event_type':event_type,
      'branch':branch,'bucket':bucket,'prior_streak':height,
      'close':10.,'amount':amt,'close_loc':.30,'close_vwap':.97,
      'gap':0.,'vol_prev':2.,'d2_price_reset':False})

def run():
    assert f.CONFIG['historical_data_end']=='2026-09-30'
    assert f.START_DAY1=='2026-10-08'
    # Frozen PANIC close/VWAP capitulation opens at D2 opening match, not M2.
    p=row(branch='PANIC')
    p['close_vwap']=.96
    s=bars();s.loc[0,'open']=9.5;s.loc[0,'low']=9.5
    ent,why=f.choose_entry(p,s)
    assert ent and ent['lane']=='B4P_PANIC_CAPITULATION_OPEN' and ent['entry_idx']==0

    # Non-PANIC B4P M3 down displacement: entry at actual M4 open, no future-bar information.
    p=row()
    s=bars()
    for i,x in enumerate([9.9,9.8,9.75]):
        s.loc[i,['close','low','amount']]=[x,x,x*100000]
    s.loc[3,['open','low','close','amount']]=[9.75,9.75,9.75,9.75*100000]
    e,why=f.choose_entry(p,s,.9)
    assert e and e['lane']=='NORMAL_B4P_LIQUID_M3'
    assert e['entry_idx']==3 and e['entry_price']==9.75
    # Changes to all post-entry prices cannot alter the Day2 entry decision.
    future=s.copy()
    future.loc[8:239,['open','high','low','close']]=11.0
    future.loc[8:239,'amount']=1100000.
    e2,why=f.choose_entry(p,future,.9)
    assert e2 and e2['lane']==e['lane'] and e2['entry_price']==e['entry_price']

    # RECYCLE_2PLUS rearmed >=3 boards is allowed in core, but barred from new LM3/W5 expansions.
    p['event_type']='RECYCLE_2PLUS'
    eb,why=f.choose_entry(p,s,.2)
    assert eb is None or eb['lane'] not in ('NORMAL_B4P_LIQUID_M3','NORMAL_B4P_DOUBLE_WASHOUT_M5')
    p['prior_streak']=2
    assert f.choose_entry(p,s,.2)[0] is None

    # Weak PM double washout triggers after M5, even if M3 liquid lane is not eligible.
    p=row(amt=1e9)
    s=bars()
    for i in range(5):
        px=10-.03*(i+1)
        s.loc[i,['close','low','amount']]=[px,px,px*100000]
    s.loc[5,['open','low','close','amount']]=[9.85,9.85,9.85,985000]
    w,reason=f.choose_entry(p,s,.2)
    assert w and w['lane']=='NORMAL_B4P_DOUBLE_WASHOUT_M5' and w['entry_idx']==5
    assert f.choose_entry(p,s,.5)[0] is None or f.choose_entry(p,s,.5)[0]['lane']!='NORMAL_B4P_DOUBLE_WASHOUT_M5'

    # T1 first-minute causal take-profit; sell at next bar open.
    trade={'lane':'B4P_PANIC_CAPITULATION_OPEN','entry_price':10.}
    s=bars()
    s.loc[0,['close','high','amount']]=[10.95,10.95,1095000.]
    s.loc[1,['open','high','low','close','amount']]=[10.93,10.93,10.93,10.93,1093000.]
    z,reason=f.choose_exit(trade,s)
    assert z and z['exit_rule']=='CAP_M1_PROFIT_GT_8PCT'
    assert abs(z['final_ret']-(10.93/10-1-f.COST))<1e-12

    # Freeze archive never rewrites on matching input and rejects contradictory updates.
    old=f.OUT
    with tempfile.TemporaryDirectory() as temp:
        f.OUT=Path(temp)
        created=f.frozen_snapshot('day1','2026-10-08',[{'code':'600123'}])
        assert created==f.frozen_snapshot('day1','2026-10-08',[{'code':'600123'}])
        rejected=False
        try:f.frozen_snapshot('day1','2026-10-08',[{'code':'600456'}])
        except RuntimeError:rejected=True
        assert rejected
    f.OUT=old
    print('FORWARD_UNIT_TESTS_OK 7 groups including no-lookahead + immutable snapshots')

if __name__=='__main__':
    run()