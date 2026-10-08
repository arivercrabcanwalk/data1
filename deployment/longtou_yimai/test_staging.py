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

def next_session_readiness_test():
    ts=lambda date:dt.datetime.fromisoformat(date).replace(tzinfo=TZ)
    assert a.next_observation_session(ts('2026-10-08T16:40:00'))=='2026-10-09'
    assert a.next_observation_session(ts('2026-10-09T09:20:00'))=='2026-10-09'
    assert a.next_observation_session(ts('2026-10-09T15:55:00'))=='2026-10-12'
    print('NEXT_TRADING_SESSION_READINESS_GATES_OK')

def notification_mocked_armed_test():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    evt={'event_id':'000678_20261008_g0','code':'000678','name':'合成提醒',
         'date':'2026-10-08','branch':'NORMAL_RED','prior_streak':4}
    plan={'lane':'NORMAL_B4P_LIQUID_M3','signal_time':'09:33'}
    q=SimpleNamespace(price=9.70,prev_close=10.,server_time='20261009093312')
    src=SimpleNamespace(sina_quote=lambda code:SimpleNamespace(
       price=9.70,prev_close=10.,server_time='20261009093312'))
    with tempfile.TemporaryDirectory(prefix='longtou_alert_test_') as tmp:
        old=a.STATE;a.STATE=Path(tmp)
        try:
            class Success:
                returncode=0
            calls=[]
            def mockrun(cmd,**kwargs):
                calls.append(cmd)
                return Success()
            with patch.object(a.subprocess,'run',side_effect=mockrun):
                with patch.dict(a.os.environ,{
                    'OPENCLAW_MESSAGE_CHANNEL':'feishu',
                    'OPENCLAW_MESSAGE_TARGET':'dummy_test_user'
                }):
                    outcome=a.notify_signal({'real_signals_armed':True},evt,plan,q,
                             dt.datetime(2026,10,9,9,33,12,tzinfo=TZ),src)
                    repeated=a.notify_signal({'real_signals_armed':True},evt,plan,q,
                             dt.datetime(2026,10,9,9,33,20,tzinfo=TZ),src)
            assert outcome=='FEISHU_SENT' and repeated=='ALREADY_RECORDED'
            assert len(calls)==1
            assert calls[0][:4]==['/usr/local/bin/openclaw','message','send','--channel']
            import json
            outcome_detail=json.loads((a.STATE/'delivery'/(evt['event_id']+'.json')).read_text())
            assert outcome_detail['delivery_status']=='FEISHU_SENT'
        finally:a.STATE=old
    print('ARMED_FEISHU_MOCKED_DISPATCH_AND_EXACTLY_ONCE_OK no real network send')

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

def day1_daily_notification_mock_test():
    import tempfile,json
    from pathlib import Path
    from unittest.mock import patch
    ev1={'code':'000678','name':'襄阳轴承','date':'2026-10-08',
         'prior_streak':4,'event_type':'PRIMARY_3PLUS',
         'branch':'NORMAL_RED','bucket':'B4P','ret':-.05042668735453848,
         'close':12.24,'d1_pm_above_ratio':0.0}
    ev2={'code':'000011','name':'深物业A','date':'2026-10-08',
         'prior_streak':3,'event_type':'PRIMARY_3PLUS',
         'branch':'TOUCH_BREAK','bucket':'B3','ret':-.009803921568627527,
         'close':12.12,'d1_pm_above_ratio':None}
    snap={'day':'2026-10-08','captured_at':'2026-10-08T15:12:30+08:00',
          'quote_coverage':.9602,'daily_coverage':.9933,'events':[ev2,ev1]}
    msg=a.format_day1_daily_message(snap)
    assert '襄阳轴承' in msg and '深物业A' in msg
    assert '明日正式V22买点监测：1只' in msg
    assert '仅记录观察' in msg
    assert 'Day1≠买点' in msg
    with tempfile.TemporaryDirectory(prefix='longtou_day1_notify_') as temp:
        old=a.STATE;a.STATE=Path(temp)
        root=a.STATE
        (root/'events').mkdir();(root/'daily').mkdir()
        (root/'events/2026-10-08.json').write_text(json.dumps(snap,ensure_ascii=False))
        (root/'daily/2026-10-08.parquet').write_bytes(b'dummy-fixture')
        try:
            results=[]
            def fake_cli(cmd,**kwargs):
                results.append(cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(a,'fs_preflight',return_value=(True,'TEST')):
                with patch.object(a,'market_calendar',return_value=['2026-10-08','2026-10-09']):
                    with patch.dict(a.os.environ,{'OPENCLAW_MESSAGE_CHANNEL':'feishu','OPENCLAW_MESSAGE_TARGET':'MOCK_ONLY_DO_NOT_SEND'}):
                        with patch.object(a.subprocess,'run',side_effect=fake_cli):
                            first=a.send_day1_daily({'real_signals_armed':True},'2026-10-08')
                            repeat=a.send_day1_daily({'real_signals_armed':True},'2026-10-08')
                        assert first=='DAY1_NOTICE_FEISHU_SENT' and repeat=='DAY1_NOTICE_ALREADY_FEISHU_SENT'
                        assert len(results)==1
                        assert '【龙头一买' in results[0][results[0].index('--message')+1]
                        log=json.loads((root/'day1_delivery/2026-10-08.json').read_text())
                        assert log['event_count']==2 and log['watch_count']==1
                        assert log['status']=='FEISHU_SENT'
                        assert (root/'day1_delivery/2026-10-08.json').stat().st_mode&0o777==0o600
                        with patch.object(a,'load_daily',side_effect=AssertionError('SHOULD_NOT_REBUILD_EXISTING_EOD')):
                            with patch.object(a,'send_day1_daily',return_value='DAY1_NOTICE_ALREADY_FEISHU_SENT') as done:
                                # Keep this test from depending on the wall-clock or provider.
                                with patch.object(a,'local_now',return_value=dt.datetime(2026,10,8,16,30,tzinfo=TZ)):
                                    with patch.object(a,'is_open',return_value=True):
                                        a.postclose({'real_signals_armed':True})
                                assert done.call_count==1
        finally:a.STATE=old
    empty=dict(snap,events=[])
    assert '没有符合冻结生命周期' in a.format_day1_daily_message(empty)
    print('DAY1_EOD_FEISHU_DIGEST_TEST_OK 2 candidates, dedupe, mock send, zero-candidate message, no actual notification')

if __name__=='__main__':
    minute_test()
    next_session_readiness_test()
    day1_daily_notification_mock_test()
    notification_mocked_armed_test()
    notification_disabled_test()
    day1_priority_data_test()
    day1_legacy_match()
    a.selftest()
    print('LONGTOU_YIMAI_FULL_STAGING_TESTS_OK')