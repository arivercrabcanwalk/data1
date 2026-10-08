# -*- coding: utf-8 -*-
"""Live-data paper observation of strict prefix-only V21 lanes (V22 research).
Does not modify, replace, or place orders for frozen V21.
"""
from pathlib import Path
import json,hashlib,argparse,datetime as dt,os
import pandas as pd,numpy as np
import v21_forward_engine as old
import v22_strict_causal_core as policy
from v17_structural_lab import lifecycle_events

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_strict_forward';OUT.mkdir(exist_ok=True)
CONFIG={
 'version':'V22_STRICT_PREFIX_SHADOW',
 'historical_last_day':'2026-09-30',
 'first_prospective_day1':'2026-10-08',
 'rule':'Entry checks only D1-close facts plus completed Day2 bars and the next minute OPEN; no later minute OHLCV, no future outcomes.',
 'entry_policy_source':'v22_strict_causal_core.py',
 'exit_policy':'Frozen V21 T1 lane-specific conditions using completed bars and next OPEN or T1 close',
 'cost_round_trip':.0052,
 'unfilled_upper_open':'Exclude buy entry if entering bar opens at known upper limit; never peek at the same bar low',
 'no_order_routing':True,
 'separate_from_v21':True
}
def sha_config():return hashlib.sha256(json.dumps(CONFIG,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def dump(path,d):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')
 os.replace(temp,path)
def init():
 old.require_frozen()
 p=OUT/'strict_protocol.json'
 if p.exists():
  obj=json.loads(p.read_text())
  assert obj['config_hash']==sha_config() and obj['monitor_sha']==old.sha_file(__file__) and obj['core_sha']==old.sha_file(ROOT/'v22_strict_causal_core.py')
  print('STRICT_PROTOCOL_ALREADY_FROZEN');return
 obj={'registered_at':old.now().isoformat(),'config_hash':sha_config(),
      'monitor_sha':old.sha_file(__file__),'core_sha':old.sha_file(ROOT/'v22_strict_causal_core.py'),
      'v21_baseline_sha':old.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet'),
      'config':CONFIG}
 dump(p,obj)
 print('V22_STRICT_PROTOCOL_FROZEN',obj['registered_at'],obj['config_hash'])
def verify():
 obj=json.loads((OUT/'strict_protocol.json').read_text())
 assert obj['config_hash']==sha_config()
 assert obj['monitor_sha']==old.sha_file(__file__)
 assert obj['core_sha']==old.sha_file(ROOT/'v22_strict_causal_core.py')
 assert obj['v21_baseline_sha']==old.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet')
 old.require_frozen()
 return obj

def snapshot(kind,day,arr):
 p=OUT/'snapshots'/kind/(day+'.json')
 fingerprint=hashlib.sha256(json.dumps(arr,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
 if p.exists():
  d=json.loads(p.read_text())
  if d['digest']!=fingerprint:raise RuntimeError(f'IMMUTABLE STRICT SNAPSHOT MUTATION: {p}')
  return d
 d={'date':day,'created_at':old.now().isoformat(),'digest':fingerprint,'records':arr}
 dump(p,d);return d

def scan():
 frozen=verify()
 d,newdays=old.hydrate_daily_history()
 status={
  'asof':old.now().isoformat(),'protocol_hash':frozen['config_hash'],
  'new_day_partitions':newdays,'last_raw_daily':old.trading_day_partitions()[-1],
  'eligible_day1_count':0,'strict_day2_entries':0,'strict_t1_closed':0,
  'strict_pre_registered_trades':0,'win_rate':None,'average_return':None,
  'delayed_backfill_trades':0,'state':'AWAITING_NEW_DATA'}
 if not newdays:
  dump(OUT/'strict_forward_status.json',status)
  print(json.dumps(status,ensure_ascii=False,indent=2));return
 E=lifecycle_events(d)
 E['date']=pd.to_datetime(E.date); E['d2_date']=pd.to_datetime(E.d2_date);E['f2_date']=pd.to_datetime(E.f2_date)
 E=E[(E.date>='2026-10-08')&(E.prior_streak>=3)].copy()
 status['eligible_day1_count']=len(E)
 if E.empty:
  for day in newdays:snapshot('day1',day,[])
  status['state']='NO_DAY1_EVENTS'
  dump(OUT/'strict_forward_status.json',status)
  print(json.dumps(status,ensure_ascii=False,indent=2));return
 ymins=old.read_for_events(E,'date')
 m2=old.read_for_events(E.dropna(subset=['d2_date']),'d2_date')
 m3=old.read_for_events(E.dropna(subset=['f2_date']),'f2_date')
 day1={};entries=[];closed=[];registered=[]
 for _,r in E.sort_values(['date','event_id']).iterrows():
  day=str(r.date.date());code=str(r.code).zfill(6)
  dm=ymins.get(day,{}).get(code)
  pm=old.pm_above_ratio(dm) if dm is not None else None
  day1row={
    'event_id':str(r.event_id),'code':code,'name':str(r['name']),
    'day1':day,'event_type':str(r.event_type),
    'prior_streak':int(r.prior_streak),'branch':str(r.branch),
    'bucket':str(r.bucket),'day1_price':float(r.close),
    'pm_above_ratio':pm,
    'd1_amount':float(r.amount) if pd.notna(r.amount) else None}
  day1.setdefault(day,[]).append(day1row)
  if pd.isna(r.d2_date):
   registered.append(dict(day1row,status='AWAIT_D2'));continue
  next_day=str(r.d2_date.date())
  s=m2.get(next_day,{}).get(code)
  if s is None or len(s)<1:
   registered.append(dict(day1row,status='AWAIT_D2_MINUTES',d2_date=next_day));continue
  if r.branch=='NORMAL_RED' and r.bucket=='B4P' and r.event_type=='PRIMARY_3PLUS' and pm is None:
   registered.append(dict(day1row,status='DAY1_PM_MISSING_PRIORITIZATION_UNKNOWN',d2_date=next_day));continue
  if abs(float(s.iloc[0].open)-float(r.d2_open))>.011:
   registered.append(dict(day1row,status='OPEN_SOURCE_MISMATCH',d2_date=next_day));continue
  selected,why=policy.entry(r,s,pm)
  if selected is None:
   registered.append(dict(day1row,status='D2_PASS',why=why,d2_date=next_day));continue
  selected['d2_date']=next_day
  selected['cost_model']=.0052
  entries.append(selected)
  if pd.isna(r.f2_date):
   registered.append(dict(day1row,status='D2_SIGNAL_T1_PENDING',lane=selected['lane'],d2_date=next_day));continue
  t1day=str(r.f2_date.date())
  mt=m3.get(t1day,{}).get(code)
  ret,reason=policy.exit_t1(selected,mt)
  if ret is None:
   registered.append(dict(day1row,status='WAIT_T1_PRICE',reason=reason,
                          lane=selected['lane'],d2_date=next_day,t1_date=t1day));continue
  # The T1 result is a LABEL, never a signal feature.
  ret['t1_date']=t1day
  ret['exit_close_source']='MINUTE_PREFIX_NEXT_OPEN_OR_FINAL_EOD'
  if mt is not None and len(mt)==240 and str(mt.iloc[-1].time)=='15:00':
   ret['t1_day_final_close_check']=float(mt.iloc[-1].close)
  ret['entry_was_model_proxy']=True
  closed.append(ret)
  registered.append(dict(day1row,status='T1_SIMULATED_CLOSED',lane=selected['lane'],
                         d2_date=next_day,t1_date=t1day))
 for day in newdays:snapshot('day1',day,day1.get(day,[]))
 entby={}
 for r in entries:entby.setdefault(r['d2_date'],[]).append(r)
 for day,arr in entby.items():snapshot('day2_entry',day,arr)
 t1by={}
 for rec in closed:
  created=json.loads((OUT/'snapshots'/'day1'/(rec['day1']+'.json')).read_text())['created_at']
  registered_at=dt.datetime.fromisoformat(created)
  first_auction=dt.datetime.fromisoformat(rec['d2_date']+'T09:25:00+08:00')
  rec['registered_before_day2']=bool(registered_at<first_auction)
  rec['evidence_status']='PRE_REGISTERED' if rec['registered_before_day2'] else 'DELAYED_BACKFILL'
  t1by.setdefault(rec['t1_date'],[]).append(rec)
 for day,arr in t1by.items():snapshot('t1_result',day,arr)
 dump(OUT/'all_day1_register.json',registered)
 dump(OUT/'all_entries.json',entries)
 dump(OUT/'all_closed.json',closed)
 status['strict_day2_entries']=len(entries)
 status['strict_t1_closed']=len(closed)
 valid=[z for z in closed if z['registered_before_day2']]
 status['strict_pre_registered_trades']=len(valid)
 status['delayed_backfill_trades']=len(closed)-len(valid)
 if valid:
  x=np.array([z['net_ret'] for z in valid])
  status['win_rate']=float(np.mean(x>0))
  status['average_return']=float(np.mean(x))
  status['worst_return']=float(np.min(x))
 status['state']='REPLAYED_NEW_DATA'
 dump(OUT/'strict_forward_status.json',status)
 print(json.dumps(status,ensure_ascii=False,indent=2))

def main():
 p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
 m.add_argument('--init',action='store_true');m.add_argument('--scan',action='store_true')
 args=p.parse_args()
 if args.init:init()
 else:scan()
if __name__=='__main__':main()