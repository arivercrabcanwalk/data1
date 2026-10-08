# -*- coding: utf-8 -*-
"""V22 shadow lanes: pre-specified research-only first-buy opportunities.
Never alters the V21 frozen engine, V21 forward protocol, or real-money orders.
No T1/future returns are inspected when registering an entry.
"""
from __future__ import annotations
from pathlib import Path
import argparse,datetime as dt,json,hashlib,math,os
import numpy as np,pandas as pd
import duckdb
import v21_forward_engine as core
from v17_structural_lab import lifecycle_events

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_shadow_forward_v3';OUT.mkdir(exist_ok=True)
CALENDAR_END='2026-09-30'
SHADOW={
 'version':'V22_SHADOW_STRICT_NO_FUTURE_V3',
 'freeze_session':'2026-10-08',
 'entry_priority':'Do not choose a winner across shadow lanes; log all independent rule triggers.',
 'data_scope':'At M4 open use only first 3 completed minute bars plus M4 opening price; do not use M4 low or later bars.',
 'shadow_execution':'Signal confirmed at the end of M3, next M4 open; default T1 close. Model costs 0.52% per full trip.',
 'target':'All first-break high-visibility stocks after >=3 consecutive limit-ups. R2 only observe.',
 'arms':{
  'TOUCH_B3_REBOUND_M3':{'branch':'TOUCH_BREAK','bucket':'B3','d2_gap_max':-.04,'m3_vs_d2_open_min':.02,'m3_above_cvwap':True},
  'TOUCH_B4P_REBOUND_M3':{'branch':'TOUCH_BREAK','bucket':'B4P','m3_vs_d2_open_min':.025,'m3_above_cvwap':True},
  'POS_B4P_LEAD_M3':{'branch':'POSITIVE_BREAK','bucket':'B4P','d1_ret_min':.05,'d1_vol_prev_min':1.5,'m3_vs_d2_open_min':0.,'m3_above_cvwap':True},
  'POS_B3_LIQ_M3':{'branch':'POSITIVE_BREAK','bucket':'B3','d1_amount_min':2e9,'m3_vs_d2_open_min':.015,'m3_above_cvwap':True},
  'FAILED_B3_RECOVERY_M3':{'branch':'FAILED_OPEN','bucket':'B3','m3_vs_d2_open_min':.025,'m3_above_cvwap':True},
  'NORMAL_B3_GAP_IMPULSE_M3':{'branch':'NORMAL_RED','bucket':'B3','d2_gap_max':-.03,'m3_vs_d2_open_min':.02,'m3_above_cvwap':True},
  'PANIC_B4P_FAST_RECOVERY_M3':{'branch':'PANIC','bucket':'B4P','d2_gap_max':-.05,'m3_vs_d2_open_min':.03,'m3_above_cvwap':True}
 },
 'status':'RESEARCH_SHADOW_NOT_PROMOTED',
 'stopping_rule':'No threshold changes without a new independently registered version; first 10, 30, 50 pre-registered OOS trades reviewed.',
}
def json_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
def dump(path,content):
 p=Path(path);p.parent.mkdir(exist_ok=True,parents=True)
 t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(content,ensure_ascii=False,indent=2,default=str)+'\n')
 os.replace(t,p)
def freeze():
 assert core.require_frozen()
 path=OUT/'shadow_protocol.json'
 if path.exists():
  obj=json.loads(path.read_text())
  if obj['sha256']!=json_hash(SHADOW) or obj['engine_sha256']!=core.sha_file(__file__):
   raise RuntimeError('V22 SHADOW RULE/ENGINE MUTATED')
  print('ALREADY_SHADOW_FROZEN',obj['at']);return
 obj={'at':core.now().isoformat(),'sha256':json_hash(SHADOW),
      'engine_sha256':core.sha_file(__file__),
      'v21_baseline_sha256':core.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet'),
      'rules':SHADOW}
 dump(path,obj);print('SHADOW_FROZEN',obj['at'],obj['sha256'])
def verify():
 obj=json.loads((OUT/'shadow_protocol.json').read_text())
 if obj['sha256']!=json_hash(SHADOW) or obj['engine_sha256']!=core.sha_file(__file__):
  raise RuntimeError('V22 SHADOW PROTOCOL/ENGINE CHANGED')
 if obj['v21_baseline_sha256']!=core.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet'):
  raise RuntimeError('V21 historical baseline modified')
 assert core.require_frozen()
 return obj

def record_immutable(stage,day,items):
 dest=OUT/'snapshots'/stage/(day+'.json')
 checksum=json_hash(items)
 if dest.exists():
  old=json.loads(dest.read_text())
  if old['checksum']!=checksum:raise RuntimeError(f'V22 immutable snapshot changed: {dest}')
  return old
 obj={'date':day,'captured_at':core.now().isoformat(),'checksum':checksum,'items':items}
 dump(dest,obj);return obj

def early_minutes_valid(s):
    # Only three COMPLETED OHLCV bars are allowed; M4 OPEN is the only known M4 field.
    if s is None or len(s)<4:return False
    b=s.iloc[:3]
    if b.time.astype(str).tolist()!=['09:31','09:32','09:33']:return False
    if str(s.iloc[3].time)!='09:34':return False
    if b[['open','high','low','close','volume','amount']].isna().any().any():return False
    if (b.volume<=0).any() or (b.amount<=0).any():return False
    return bool(pd.notna(s.iloc[3].open) and float(s.iloc[3].open)>0)

def candidates(r,minute):
    """Decision uses only D1 facts, M1-M3 closes and minute-4 OPEN."""
    if not early_minutes_valid(minute):return []
    if bool(r.get('d2_price_reset',False)):return []
    price1=float(r.close);opening=float(minute.iloc[0].open)
    if price1<=0 or opening<=0 or abs(opening/price1-1)>.12:return []
    cv=core.calc_cvwap(minute.iloc[:3].copy())
    if cv is None:return []
    m3close=float(minute.iloc[2].close);m3ret=m3close/opening-1
    if m3close<cv[2]:return []
    px=float(minute.iloc[3].open)
    upper=math.floor(price1*110+.5+1e-8)/100
    # Do not use minute-4 LOW to decide if the open is buyable.
    if not np.isfinite(px) or px<=0 or px>=upper-.005:return []
    br=str(r.branch);bu=str(r.bucket);gap=opening/price1-1
    ret=float(r.ret);volprev=core.val(r,'vol_prev');amt=core.val(r,'amount')
    arms=[]
    def add(name,condition):
        if condition:
            arms.append({'event_id':str(r.event_id),'day1':str(r.date.date()),
                'code':str(r.code).zfill(6),'name':str(r['name']),
                'event_type':str(r.event_type),
                'branch':br,'bucket':bu,'lane':name,
                'd2_date':str(pd.Timestamp(r.d2_date).date()),
                'signal_time':'09:33','entry_clock':'09:33','entry_price':px,
                'd2_gap':gap,'m3_ret':m3ret,
                'd1_amount':amt,'d1_vol_prev':volprev,
                'paper_only':True,'future_label_used':False})
    add('TOUCH_B3_REBOUND_M3',br=='TOUCH_BREAK' and bu=='B3' and gap<=-.04 and m3ret>=.02)
    add('TOUCH_B4P_REBOUND_M3',br=='TOUCH_BREAK' and bu=='B4P' and m3ret>=.025)
    add('POS_B4P_LEAD_M3',br=='POSITIVE_BREAK' and bu=='B4P' and ret>=.05 and volprev>=1.5 and m3ret>=0)
    add('POS_B3_LIQ_M3',br=='POSITIVE_BREAK' and bu=='B3' and amt>=2e9 and m3ret>=.015)
    add('FAILED_B3_RECOVERY_M3',br=='FAILED_OPEN' and bu=='B3' and m3ret>=.025)
    add('NORMAL_B3_GAP_IMPULSE_M3',br=='NORMAL_RED' and bu=='B3' and gap<=-.03 and m3ret>=.02)
    add('PANIC_B4P_FAST_RECOVERY_M3',br=='PANIC' and bu=='B4P' and gap<=-.05 and m3ret>=.03)
    return arms

def shadow_replay():
 """Historical diagnostics can identify fragility but are NOT a new held-out performance claim."""
 P=pd.read_parquet(ROOT/'v20_results/yao_scored.parquet').copy()
 P['date']=pd.to_datetime(P.date);P['d2_date']=pd.to_datetime(P.d2_date)
 P=P[(P.prior_streak>=3)&P.exit_t1_close.notna()&~P.d2_price_reset.fillna(False)&~P.exit_t1_reset.fillna(False)]
 lookup={}
 with duckdb.connect() as con:
  for day,g in P.dropna(subset=['d2_date']).groupby(P.d2_date.dt.strftime('%Y-%m-%d')):
   if not isinstance(day,str):continue
   lookup[day]=core.read_minutes(day,g.code.astype(str).tolist(),con)
 found=[]
 for _,r in P.iterrows():
  day=str(r.d2_date.date());minutes=lookup.get(day,{}).get(str(r.code))
  if not early_minutes_valid(minutes):continue
  a=candidates(r,minutes)
  for z in a:
   z['ret_t1_close']=float(r.exit_t1_close)/float(z['entry_price'])-1-.0052
   z['subset']='JAN_JUN' if r.date<=pd.Timestamp('2026-06-30') else 'JUL_SEP_PSEUDO'
   found.append(z)
 df=pd.DataFrame(found)
 if len(df):
  df.to_csv(OUT/'shadow_historical_diagnostic.csv',index=False,encoding='utf-8-sig')
 stats=[]
 if len(df):
  for name,q in df.groupby('lane'):
   for seg,z in q.groupby('subset'):
    r=z.ret_t1_close
    stats.append({'lane':name,'subset':seg,'n':len(r),'positive':int((r>0).sum()),
                  'win_rate':float((r>0).mean()),'mean':float(r.mean()),'worst':float(r.min())})
 payload={'not_true_unseen':True,'historical_shadow_events':len(df),
          'freeze_before_diagnostics_claim':True,'stats':stats,
          'caution':'The arm ideas were inspired by hindsight mistakes. These results are exploratory, not confirmatory.'}
 dump(OUT/'shadow_historical_diagnostics.json',payload)
 print(json.dumps(payload,ensure_ascii=False,indent=2))

def scan():
 obj=verify()
 daily,newdays=core.hydrate_daily_history()
 report={'asof':core.now().isoformat(),'protocol_sha256':obj['sha256'],
  'latest_raw_daily':core.trading_day_partitions()[-1],
  'new_days':newdays,'day1_registered':0,'shadow_signals':0,
  'complete_shadow_trades':0,'preregistered_complete':0,
  'prospective_win_rate':None,'prospective_mean':None,
  'status':'WAITING_FOR_NEW_DATA'}
 if not newdays:
  dump(OUT/'shadow_status.json',report)
  print(json.dumps(report,ensure_ascii=False,indent=2));return
 ev=lifecycle_events(daily)
 ev['date']=pd.to_datetime(ev.date);ev['d2_date']=pd.to_datetime(ev.d2_date)
 ev['f2_date']=pd.to_datetime(ev.f2_date)
 ev=ev[(ev.date>=pd.Timestamp('2026-10-08'))&(ev.prior_streak>=3)].copy()
 report['day1_registered']=len(ev)
 if len(ev)==0:
  for day in newdays:record_immutable('day1',day,[])
  report['status']='NO_NEW_FIRST_BREAK'
  dump(OUT/'shadow_status.json',report)
  print(json.dumps(report,ensure_ascii=False,indent=2));return
 byday={}
 for _,r in ev.iterrows():
  d=str(r.date.date())
  byday.setdefault(d,[]).append({'event_id':r.event_id,'code':r.code,'name':r['name'],
    'event_type':r.event_type,'height':int(r.prior_streak),'branch':r.branch,'bucket':r.bucket})
 for day in newdays:record_immutable('day1',day,byday.get(day,[]))
 d2mins=core.read_for_events(ev.dropna(subset=['d2_date']),'d2_date')
 t1mins=core.read_for_events(ev.dropna(subset=['f2_date']),'f2_date')
 entries=[];completed=[]
 for _,r in ev.iterrows():
  if pd.isna(r.d2_date):continue
  day=str(r.d2_date.date());code=str(r.code).zfill(6)
  bars=d2mins.get(day,{}).get(code)
  if not early_minutes_valid(bars):continue
  if abs(float(bars.iloc[0].open)-float(r.d2_open))>.011:continue
  arms=candidates(r,bars)
  for z in arms:
   entries.append(z)
   if pd.isna(r.f2_date):continue
   tday=str(r.f2_date.date());t1=t1mins.get(tday,{}).get(code)
   if not core.minute_complete(t1):continue
   if abs(float(t1.iloc[-1].close)-float(r.exit_t1_close))>.011:continue
   ex=float(t1.iloc[-1].close)
   y=dict(z,t1_date=tday,exit_clock='15:00',exit_price=ex,
          ret_t1=ex/z['entry_price']-1-.0052)
   d1snapshot=json.loads((OUT/'snapshots'/'day1'/(z['day1']+'.json')).read_text())
   # Shadow is registered as a RULE before Day1; timely Day1 classification required for strict forward subset.
   stamp=dt.datetime.fromisoformat(d1snapshot['captured_at'])
   d2t=dt.datetime.fromisoformat(day+'T09:25:00+08:00')
   y['pre_registered_day1']=bool(stamp<d2t)
   completed.append(y)
 groups={}
 for z in entries:groups.setdefault(z['d2_date'],[]).append(z)
 for day,z in groups.items():record_immutable('day2_shadow',day,z)
 groups={}
 for z in completed:groups.setdefault(z['t1_date'],[]).append(z)
 for day,z in groups.items():record_immutable('t1_shadow',day,z)
 dump(OUT/'shadow_entry_register.json',entries)
 dump(OUT/'shadow_completion_register.json',completed)
 report['shadow_signals']=len(entries);report['complete_shadow_trades']=len(completed)
 valid=[x for x in completed if x['pre_registered_day1']]
 report['preregistered_complete']=len(valid)
 if valid:
  rs=np.array([x['ret_t1'] for x in valid])
  report['prospective_win_rate']=float(np.mean(rs>0))
  report['prospective_mean']=float(np.mean(rs))
  report['prospective_worst']=float(np.min(rs))
 report['status']='UPDATED'
 dump(OUT/'shadow_status.json',report)
 print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
 parser=argparse.ArgumentParser()
 group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--init',action='store_true')
 group.add_argument('--scan',action='store_true')
 group.add_argument('--historical-diagnostic',action='store_true')
 a=parser.parse_args()
 if a.init:freeze()
 elif a.scan:scan()
 else:verify();shadow_replay()

if __name__=='__main__':main()