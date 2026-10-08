# -*- coding: utf-8 -*-
"""V22 non-anticipating evaluator of legacy V21 7 lanes.

This is a read-only research comparison, NOT a replacement for the pre-registered V21.
At every candidate entry only signal-completed minute bars plus entry-bar OPEN
are ever examined. In particular, the target minute's LOW/HIGH/CLOSE/volume
and all later bars are unknown until after the entry decision.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

COST=.0052
VERSION='V22_STRICT_PREFIX_SHADOW'
START='2026-10-08'

def is_good_prefix(s,n):
    """n completed minute bars; no future-day data completeness tests."""
    if s is None or len(s)<n:return False
    z=s.iloc[:n]
    if z[['open','high','low','close','volume','amount']].isna().any().any():return False
    if (z.volume<=0).any() or (z.amount<=0).any():return False
    if n>=1 and str(z.iloc[0].time)!='09:31':return False
    if n>=2 and str(z.iloc[-1].time) != str(
      f'{9+(30+n)//60:02d}:{(30+n)%60:02d}'):return False
    return True

def cvwap_prefix(s,n):
    if not is_good_prefix(s,n):return None
    z=s.iloc[:n]
    return (z.amount.cumsum()/z.volume.cumsum()).to_numpy()

def scalar(r,col,default=np.nan):
    v=r.get(col,default)
    try:return float(v) if pd.notna(v) else float(default)
    except (ValueError,TypeError):return float(default)

def is_candidate(r):
    return str(r.get('event_type')) in ['PRIMARY_3PLUS','RECYCLE_2PLUS'] and scalar(r,'prior_streak')>=3

def entry(r,s,d1_pm=None):
    """Return a causally executable plan, or a reason string."""
    if not is_candidate(r):return None,'NOT_VALID_FIRST_BUY_LIFECYCLE'
    if s is None or len(s)<1:return None,'AWAIT_OPENING_PRICE'
    d1_close=scalar(r,'close');op=float(s.iloc[0].open)
    if not np.isfinite(op) or op<=0 or not d1_close>0:return None,'INVALID_AUCTION_PRICE'
    gap=op/d1_close-1
    if abs(gap)>.12:return None,'PRICE_BASIS_OR_GAP_RESET'
    # A D2 price-reset value is itself visible in D2 opening price.
    if bool(r.get('d2_price_reset',False)):return None,'D2_RESET'
    br=str(r.branch);bu=str(r.bucket);pri=str(r.event_type)=='PRIMARY_3PLUS'
    upper=math.floor(d1_close*110+.5+1e-8)/100

    def make(lane,idx,signal):
        # We must know the M(idx+1) opening price, but nothing after that.
        if len(s)<=idx:return None
        bar=s.iloc[idx]
        if str(bar.time)!=f'{9+(31+idx)//60:02d}:{(31+idx)%60:02d}':
            return None
        px=float(bar.open)
        if not np.isfinite(px) or px<=0:return None
        # At the limit-up print, being first in queue is unknown; skip conservatively.
        if px>=upper-.005:return None
        clock=('09:30' if idx==0 else str(s.iloc[idx-1].time))
        return {'event_id':str(r.event_id),'code':str(r.code),
                'name':str(r['name']),'day1':str(r.date.date()),
                'lane':lane,'entry_idx':idx,'entry_clock':clock,
                'entry_price':px,'signal_time':signal,'d2_gap':gap}
    if br=='PANIC' and bu=='B4P':
        if scalar(r,'close_vwap')<=.965:
            x=make('B4P_PANIC_CAPITULATION_OPEN',0,'09:25')
            return (x,'TRIGGERED') if x is not None else (None,'UNFILLABLE_OR_MISSING_OPEN')
        if gap<=-.03 and is_good_prefix(s,5):
            cv=cvwap_prefix(s,5)
            if cv is not None and float(s.iloc[4].close)/op-1>=.03 and float(s.iloc[4].close)>=cv[4]:
                x=make('B4P_PANIC_DEEP_IMPULSE5',5,'09:35')
                return (x,'TRIGGERED') if x is not None else (None,'UNFILLABLE_OR_AWAIT_M6')
        return None,'NO_P5_SIGNAL'
    if br=='PANIC' and bu=='B3':
        if scalar(r,'amount')<1e9:return None,'D1_VOLUME_GATE'
        for i in range(15):
            n=i+1
            if not is_good_prefix(s,n):break
            cv=cvwap_prefix(s,n)
            if cv is not None and float(s.iloc[i].close)>=max(d1_close,op,float(cv[-1])):
                x=make('PANIC_3_LIQ_RECLAIM45',i+1,str(s.iloc[i].time))
                return (x,'TRIGGERED') if x is not None else (None,'UNFILLABLE_OR_AWAIT_NEXT_OPEN')
        return None,'NO_P3_RECLAIM'
    if br=='POSITIVE_BREAK' and bu=='B4P':
        if scalar(r,'gap')>.05 or scalar(r,'vol_prev')<1.5:return None,'POS_DAY1_GATE'
        for i in range(5):
            n=i+1
            if not is_good_prefix(s,n):break
            cv=cvwap_prefix(s,n)
            if cv is not None and float(s.iloc[i].close)>=max(d1_close,op,float(cv[-1])):
                x=make('POS_4P_VOLUME_RECLAIM35',i+1,str(s.iloc[i].time))
                return (x,'TRIGGERED') if x is not None else (None,'UNFILLABLE_OR_AWAIT_NEXT_OPEN')
        return None,'NO_POS_EARLY_RECLAIM'
    if br=='NORMAL_RED' and bu=='B4P':
        # M3 opening displacement is observed at 09:33, before M4 opens.
        if (pri and scalar(r,'amount')>=1.6e9 and scalar(r,'close_loc')<=.5
             and is_good_prefix(s,3)):
            impulse=float(s.iloc[2].close)/op-1
            if abs(impulse)>=.01 and impulse<=.03:
                x=make('NORMAL_B4P_LIQUID_M3',3,'09:33')
                if x is not None:return x,'TRIGGERED'
        if pri and d1_pm is not None and d1_pm<=.40 and is_good_prefix(s,5):
            if float(s.iloc[4].close)/op-1<=-.01:
                x=make('NORMAL_B4P_DOUBLE_WASHOUT_M5',5,'09:35')
                if x is not None:return x,'TRIGGERED'
        # Last accepted completed signal bar by 09:45.
        for i in range(4,15):
            n=i+1
            if not is_good_prefix(s,n):break
            cv=cvwap_prefix(s,n)
            if cv is None:break
            z=s.iloc[:n]
            valid=(z.close.to_numpy()>=np.maximum(d1_close,op))&(z.close.to_numpy()>=cv)
            if valid[i-4:i+1].all():
                x=make('CORE_RED_4P_REPAIR45',i+1,str(s.iloc[i].time))
                return (x,'TRIGGERED') if x is not None else (None,'UNFILLABLE_OR_AWAIT_NEXT_OPEN')
        return None,'NO_B4P_NORMAL_SIGNAL'
    return None,'OUTSIDE_FROZEN_V21_LANES'

def exit_t1(trade,s):
    """Same V21 lane-specific exits, checking only signal-prefix + next OPEN, or EOD close."""
    ep=float(trade['entry_price']);lane=trade['lane']
    if ep<=0:return None,'BAD_ENTRY'
    if s is None or len(s)<1:return None,'WAIT_T1_DATA'
    op=float(s.iloc[0].open)
    early=None;rule=None
    if lane=='B4P_PANIC_CAPITULATION_OPEN':
        if is_good_prefix(s,1) and float(s.iloc[0].close)/ep-1>.08:
            early=1;rule='PO_T1_M2_TP'
    elif lane=='PANIC_3_LIQ_RECLAIM45':
        if is_good_prefix(s,5) and float(s.iloc[4].close)/op-1>.005:
            early=5;rule='P3_T1_M6_TP'
    elif lane=='NORMAL_B4P_LIQUID_M3':
        if is_good_prefix(s,3) and float(s.iloc[2].close)/op-1>.05:
            early=3;rule='LM3_T1_M4_TP'
    elif lane=='POS_4P_VOLUME_RECLAIM35':
        if is_good_prefix(s,3) and float(s.iloc[2].close)/ep-1<-.02:
            early=3;rule='POS_T1_M4_STOP'
    elif lane=='CORE_RED_4P_REPAIR45':
        if is_good_prefix(s,3) and float(s.iloc[2].close)/op-1<-.015:
            early=3;rule='R4_T1_M4_STOP'
    if early is not None and len(s)>early:
        px=float(s.iloc[early].open)
        clock=str(s.iloc[early-1].time)
    else:
        if len(s)<240:return None,'WAIT_T1_CLOSE'
        if str(s.iloc[-1].time)!='15:00':return None,'WAIT_T1_CLOSE'
        px=float(s.iloc[-1].close)
        clock='15:00'
        rule='T1_CLOSE'
    if not px>0:return None,'INVALID_T1_PRICE'
    return dict(trade,exit_clock=clock,exit_rule=rule,exit_price=px,
                net_ret=px/ep-1-COST,positive=px/ep-1-COST>0),'CLOSED'