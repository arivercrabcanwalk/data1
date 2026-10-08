# -*- coding: utf-8 -*-
"""Isolated tests; no Feishu calls, no changes to old apps."""
import datetime as dt
import math
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
import longtou_runtime as a
from longtou_core import entry

TZ=ZoneInfo('Asia/Shanghai')
def stub(day,p,prev):
    return SimpleNamespace(code=str(p.code),name=str(p['name']),open=float(p.open),
        high=float(p.high),low=float(p.low),price=float(p.close),
        volume=float(p.shares)/100,amount=float(p.amount),
        prev_close=float(prev.close),server_time=day.replace('-','')+'150001',
        source='fixture')

def minute_test():
    day=dt.date(2026,10,9)
    auction={'time':'09:30','price':10.,'cum_volume':1000.,'cum_amount':1000000.}
    mins=[auction];vol=1000.;amt=1000000.
    for minute,price in [(31,9.90),(32,9.80),(33,9.70)]:
        vol+=300;amt+=300*100*price
        mins.append({'time':f'09:{minute:02d}','price':price,'cum_volume':vol,'cum_amount':amt})
    z=pd.DataFrame(mins)
    t1=dt.datetime(2026,10,9,9,30,12,tzinfo=TZ)
    s=a.provider_bars(z.iloc[:1],t1,10.,10.0)
    assert len(s)==1 and s.iloc[0].time=='09:31' and pd.isna(s.iloc[0].close)
    t2=dt.datetime(2026,10,9,9,33,12,tzinfo=TZ)
    s=a.provider_bars(z,t2,10.,9.70)
    assert len(s)==4 and s.time.tolist()==['09:31','09:32','09:33','09:34']
    assert float(s.iloc[0].volume)==30000., 'AUCTION_VOLUME_MUST_NOT_BE_DOUBLE_COUNTED'
    assert float(s.iloc[0].amount)==(300.*100.*9.90)
    malformed=z.copy();malformed.loc[2,'cum_amount']*=10000
    assert a.provider_bars(malformed,t2,10.,9.70) is None

    assert math.isnan(s.iloc[-1]['low']) and s.iloc[-1]['open']==9.70
    r=pd.Series(dict(event_id='600123_20261008_g0',code='600123',name='合成样本',
           date=pd.Timestamp('2026-10-08'),close=10.,amount=3e9,close_loc=.25,
           close_vwap=.97,branch='NORMAL_RED',bucket='B4P',prior_streak=4,
           event_type='PRIMARY_3PLUS',d2_price_reset=False))
    out,why=entry(r,s,None)
    assert out and out['lane']=='NORMAL_B4P_LIQUID_M3' and out['entry_idx']==3
    bad=z[z.time!='09:32'].copy()
    badplan=a.provider_bars(bad,t2,10.,9.7)
    assert badplan is not None and len(badplan)<4
    assert entry(r,badplan,None)[0] is None
    print('MINUTE_ADAPTER_CAUSAL_OK no fabricated OHLC and missing M2 fails closed')

def day1_legacy_match():
    d=a.load_daily()
    prior='2026-09-29'; day='2026-09-30'
    l=d[d.date==prior].set_index('code')
    now=d[d.date==day]
    quotes=[]
    for _,r in now.iterrows():
        code=str(r.code)
        if code not in l.index or not bool(r.eligible):continue
        p=l.loc[code]
        if float(r.shares)<=0 or float(r.amount)<=0 or min(float(r.open),float(r.close),float(r.high),float(r.low))<=0:continue
        quotes.append(stub(day,r,p))
    class NoMinute:
        def minute_any(self,code):raise RuntimeError('fixture does not carry afternoon minute feed')
    today,e,coverage=a.classify(day,d[d.date<=prior],quotes,NoMinute())
    assert coverage>.94
    expected=pd.read_parquet(a.SEED/'event_history_to_20260930.parquet')
    expected=expected[(expected.date==day)&(expected.prior_streak>=3)]
    observed=set(x['code'] for x in e)
    target=set(expected.code.astype(str))
    assert observed==target,('DAY1_IDENTIFICATION_MISMATCH',sorted(observed-target),sorted(target-observed))
    print('DAY1_RECONSTRUCTION_OK',len(e),'matches frozen September30 events; daily rows',len(today))

def day1_priority_data_test():
    normal={'event_type':'PRIMARY_3PLUS','branch':'NORMAL_RED',
            'bucket':'B4P','d1_pm_above_ratio':None}
    assert not a.day1_priority_inputs_complete(normal)
    normal['d1_pm_above_ratio']=0.25
    assert a.day1_priority_inputs_complete(normal)
    normal['d1_pm_above_ratio']=None
    normal['event_type']='RECYCLE_2PLUS'
    assert a.day1_priority_inputs_complete(normal)
    print('PM_PRIORITY_DATA_FAIL_CLOSED_TEST_OK')

def notification_disabled_test():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    evt={'event_id':'600100_20261008_g0','code':'600100','name':'合成样本',
         'date':'2026-10-08','branch':'NORMAL_RED','prior_streak':4}
    plan={'lane':'NORMAL_B4P_LIQUID_M3','signal_time':'09:33'}
    q=SimpleNamespace(price=9.70,prev_close=10.,server_time='20261009093312')
    src=SimpleNamespace(sina_quote=lambda code: SimpleNamespace(price=9.70,prev_close=10.,server_time='20261009093312'))
    with tempfile.TemporaryDirectory(prefix='longtou_test_') as tmp:
        old=a.STATE;a.STATE=Path(tmp)
        try:
            with patch.object(a.subprocess,'run',side_effect=AssertionError('NO_REAL_FEISHU_SEND_ALLOWED')):
                got=a.notify_signal({'real_signals_armed':False},evt,plan,q,
                                    dt.datetime(2026,10,9,9,33,12,tzinfo=TZ),src)
                assert got=='RESEARCH_SIGNAL_ONLY'
                again=a.notify_signal({'real_signals_armed':False},evt,plan,q,
                                      dt.datetime(2026,10,9,9,33,18,tzinfo=TZ),src)
                assert again=='ALREADY_RECORDED'
                assert not (a.STATE/'delivery').exists()
        finally:a.STATE=old
    print('NOTIFICATION_DISABLED_AND_DEDUP_TEST_OK no external CLI called')

if __name__=='__main__':
    minute_test()
    notification_disabled_test()
    day1_priority_data_test()
    day1_legacy_match()
    a.selftest()
    print('LONGTOU_YIMAI_FULL_STAGING_TESTS_OK')