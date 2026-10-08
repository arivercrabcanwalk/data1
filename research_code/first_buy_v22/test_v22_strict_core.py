# -*- coding: utf-8 -*-
"""Strict source-time boundary tests for every V21 entry/exit lane."""
import pandas as pd,numpy as np
import v22_strict_causal_core as c
from test_v21_forward import bars,row

def stock(branch,bucket,amt=2e9,ret=-.04,close_vwap=1.00):
 r=row(branch=branch,bucket=bucket,amt=amt)
 r['event_type']='PRIMARY_3PLUS'
 r['ret']=ret;r['close_vwap']=close_vwap;r['vol_prev']=2.0;r['gap']=0.0
 return r

def unknown_after(s,idx):
 z=s.copy()
 # idx entry OPEN is known, its OHLCV content isn't known until a minute later.
 z.loc[idx,['high','low','close','volume','amount']]=[np.nan]*5
 if idx+1<len(z):
  z.loc[idx+1:,['open','high','low','close','volume','amount']]=[np.nan]*6
 return z

def assert_lane(r,s,lane,idx,pm=None):
 full,reason=c.entry(r,s,pm)
 assert full is not None and full['lane']==lane and full['entry_idx']==idx,(lane,full,reason)
 short=unknown_after(s,idx)
 x,why=c.entry(r,short,pm)
 assert x is not None and x==full,(lane,x,why,full)
 return full

def run():
 # D2 OPEN entry from pre-open Day1 weakness; no minute close was seen.
 s=bars(9.5)
 s.loc[0,'open']=9.5
 r=stock('PANIC','B4P',close_vwap=.96)
 full,why=c.entry(r,s)
 assert full is not None and full['entry_idx']==0
 short=unknown_after(s,0)
 assert c.entry(r,short)[0]==full

 # NORMAL_RED B4P M3 below open triggers at M4 open.
 s=bars(10.0)
 s.loc[2,['close','low','amount']]=[9.75,9.70,975000.]
 s.loc[3,'open']=9.76
 r=stock('NORMAL_RED','B4P')
 assert_lane(r,s,'NORMAL_B4P_LIQUID_M3',3,pm=.9)
 locked=s.copy();locked.loc[3,'open']=11.
 assert c.entry(r,locked,d1_pm=.9)[0] is None or c.entry(r,locked,d1_pm=.9)[0]['lane']!='NORMAL_B4P_LIQUID_M3'

 # W5 belongs to PRIMARY only; first 5 completed bars visible.
 s=bars(10.0)
 for i in range(5):
  px=10-.06*(i+1)
  s.loc[i,['open','high','low','close','amount']]=[px+.02,px+.03,px-.05,px,px*100000]
 s.loc[5,'open']=9.70
 r=stock('NORMAL_RED','B4P',amt=1e9)
 assert_lane(r,s,'NORMAL_B4P_DOUBLE_WASHOUT_M5',5,pm=.2)
 r['event_type']='RECYCLE_2PLUS'
 assert c.entry(r,s,.2)[0] is None or c.entry(r,s,.2)[0]['lane']!='NORMAL_B4P_DOUBLE_WASHOUT_M5'

 # Core 4P five fully repaired minutes enters M6; early M4 chart irrelevant.
 s=bars(10.0)
 s.loc[5,'open']=10.0
 r=stock('NORMAL_RED','B4P',amt=1e9)
 assert_lane(r,s,'CORE_RED_4P_REPAIR45',5,pm=.9)

 # PANIC3 and positive B4P causal first price+VWAP reclaim.
 s=bars(10.0)
 s.loc[0,['close','high','amount']]=[10.10,10.15,1005000.]
 s.loc[1,'open']=10.09
 r=stock('PANIC','B3',amt=2e9)
 assert_lane(r,s,'PANIC_3_LIQ_RECLAIM45',1)
 r=stock('POSITIVE_BREAK','B4P',amt=2e9,ret=.06)
 assert_lane(r,s,'POS_4P_VOLUME_RECLAIM35',1)

 # P5 requires 5-minute rebound after a >=3% Day2 discount.
 s=bars(9.6)
 for i in range(5):
  px=9.6+.08*(i+1)
  s.loc[i,['open','high','low','close','amount']]=[px-.03,px+.03,px-.06,px,px*100000]
 s.loc[0,'open']=9.5
 s.loc[5,'open']=10.
 r=stock('PANIC','B4P',close_vwap=.98)
 assert_lane(r,s,'B4P_PANIC_DEEP_IMPULSE5',5)

 # T1 causal early exit uses signal close and next open, never future T1 closes.
 s=bars(10.0)
 s.loc[0,['close','high','amount']]=[11.,11.0,1100000.]
 s.loc[1,'open']=10.9
 r={'entry_price':10.,'lane':'B4P_PANIC_CAPITULATION_OPEN'}
 full,status=c.exit_t1(r,s)
 assert full is not None and full['exit_price']==10.9 and status=='CLOSED'
 short=unknown_after(s,1)
 assert c.exit_t1(r,short)[0]==full

 # Same input fed twice yields identical deterministic plans.
 assert c.entry(stock('NORMAL_RED','B4P',amt=1e9),bars(10.0),d1_pm=.9)[0]['entry_price']==10.0
 print('V22_STRICT_CORE_TEST_OK 7 entry lanes, T1 causal exit, locked-price gate, no future-bar dependence')

if __name__=='__main__':run()