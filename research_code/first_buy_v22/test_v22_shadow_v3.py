# -*- coding: utf-8 -*-
"""Unmodified frozen V22 shadow observational engine acceptance tests."""
import tempfile, datetime as dt,json
from pathlib import Path
import zoneinfo
import pandas as pd,numpy as np
import v21_forward_engine as core
import v22_shadow_watch_v3 as w
from test_v21_forward import bars

def sample(branch='TOUCH_BREAK',bucket='B3',ret=-.02,amt=3e9,vol=2):
 return pd.Series({'event_id':'002418_20261008_g0','code':'002418','name':'合成测试股票',
                   'date':pd.Timestamp('2026-10-08'),'d2_date':pd.Timestamp('2026-10-09'),
                   'event_type':'PRIMARY_3PLUS','prior_streak':3,
                   'branch':branch,'bucket':bucket,'ret':ret,'vol_prev':vol,'amount':amt,
                   'close':10.,'d2_price_reset':False})

def synthetic_weak_open():
 s=bars(price=9.50)
 for i,p in enumerate([9.52,9.68,9.92]):
  s.loc[i,['open','high','low','close','amount']]=[9.5 if i==0 else p-.03,p+.01,p-.06,p,p*100000]
 s.loc[3,['open','high','low','close','amount']]=[9.90,9.95,9.85,9.92,992000]
 return s

def run():
 s=synthetic_weak_open();p=sample()
 a=w.candidates(p,s)
 assert [x['lane'] for x in a]==['TOUCH_B3_REBOUND_M3']
 assert a[0]['entry_clock']=='09:33' and a[0]['entry_price']==9.90
 # No post-decision price may influence a rule chosen after only the first three completed minutes.
 v=s.copy();v.loc[10:239,'close']=11.;v.loc[10:239,'amount']=1.1e6
 assert w.candidates(p,v)==a
 # Explicit no-future check: the fourth bar's HIGH, LOW, CLOSE, VOLUME and AMOUNT
 # are NOT available at that bar's open and must not alter the entry decision.
 prefix=s.copy()
 prefix.loc[3,['high','low','close','volume','amount']]=[np.nan]*5
 prefix.loc[4:239,['open','high','low','close','volume','amount']]=[np.nan]*6
 assert w.candidates(p,prefix)==a
 p['bucket']='B4P'
 assert [x['lane'] for x in w.candidates(p,s)]==['TOUCH_B4P_REBOUND_M3']
 p['branch']='POSITIVE_BREAK';p['ret']=.07;p['vol_prev']=2;p['prior_streak']=4
 assert [x['lane'] for x in w.candidates(p,s)]==['POS_B4P_LEAD_M3']
 p['bucket']='B3';p['ret']=.01;p['amount']=3e9
 assert [x['lane'] for x in w.candidates(p,s)]==['POS_B3_LIQ_M3']
 p['branch']='FAILED_OPEN'
 assert [x['lane'] for x in w.candidates(p,s)]==['FAILED_B3_RECOVERY_M3']
 p['branch']='NORMAL_RED'
 assert [x['lane'] for x in w.candidates(p,s)]==['NORMAL_B3_GAP_IMPULSE_M3']
 p['branch']='PANIC';p['bucket']='B4P';p['prior_streak']=4
 assert [x['lane'] for x in w.candidates(p,s)]==['PANIC_B4P_FAST_RECOVERY_M3']
 # Frozen exact source and protocol still agree.
 w.verify()
 print('V22_SHADOW_V3_UNIT_TEST_OK 7 causal branch patterns + no lookahead + unchanged hashes')

if __name__=='__main__':run()