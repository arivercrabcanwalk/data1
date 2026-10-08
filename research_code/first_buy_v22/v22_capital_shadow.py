# -*- coding: utf-8 -*-
"""Pre-registered V22 prospective cash allocation comparisons on V21 frozen signals.
NO order routing, no cherry-picking based on realized returns.
Historical reports are research-only and not used to select the winning arm.
"""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os
import pandas as pd,numpy as np
import v21_forward_engine as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_capital_forward';OUT.mkdir(exist_ok=True)
CFG={
 'status':'SHADOW_ONLY_NO_LIVE_ORDERS',
 'freeze':'2026-10-08',
 'signal_source':'v21_forward_results/T1_closed.json ONLY; only Day1 preregistered before Day2 09:25',
 'max_slots':[1,2,3],
 'position_notional':'min(available_cash, NAV_estimate/max_slots), skip if less than 50% target',
 'selection_order':'first available executable Day2 clock; same-minute ticker sorted',
 'buy_fee':0.0026,'sell_fee':0.0026,
 'slippage_stress_per_side':[0,0.001,0.0026],
 'no_short_borrowing':True,
 'cash_interest':0,
 'daily_nav':'current cash plus mark of open shares at local raw daily close less liquidation fee',
 'limitation':'Daily OHLC marks, no queue/bidask/orderbook; does not trade. Only closed cohorts are compared until pending fills resolve.'
}
def hash_json(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def dump(path,dict_):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 t=path.with_suffix('.tmp')
 t.write_text(json.dumps(dict_,ensure_ascii=False,indent=2,default=str)+'\n')
 os.replace(t,path)
def init():
 core.require_frozen()
 p=OUT/'portfolio_protocol.json'
 if p.exists():
  d=json.loads(p.read_text())
  assert d['config_hash']==hash_json(CFG) and d['engine_hash']==core.sha_file(__file__)
  print('CAPITAL_PROTOCOL_ALREADY_FROZEN');return
 d={'config_hash':hash_json(CFG),'engine_hash':core.sha_file(__file__),
    'v21_historical_sha':core.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet'),
    'registered_at':core.now().isoformat(),'config':CFG}
 dump(p,d)
 print('CAPITAL_PROTOCOL_FROZEN',d['registered_at'],d['config_hash'])
def verify():
 d=json.loads((OUT/'portfolio_protocol.json').read_text())
 if d['config_hash']!=hash_json(CFG) or d['engine_hash']!=core.sha_file(__file__):raise RuntimeError('CAPITAL EXPERIMENT MUTATED')
 if d['v21_historical_sha']!=core.sha_file(ROOT/'v21_results/v21_52_trade_stack.parquet'):raise RuntimeError('V21 BASELINE MUTATED')
 core.require_frozen()
 return d

def simulate(trades, daily,slots,fee):
 """Decisions use trade timestamps/entry prices and previous close marks, never future trade returns."""
 if not trades:return dict(n=0,return_pct=None),pd.DataFrame(),pd.DataFrame()
 events=[]
 for t in trades:
  req=['event_id','code','name','d2_date','entry_clock','entry_price','t1_date','exit_clock','exit_price']
  if not all(k in t and t[k] is not None for k in req):raise RuntimeError('INCOMPLETE V21 OOS RECORD')
  code=str(t['code']).zfill(6);ev=t['event_id']
  events.append((str(t['d2_date']),str(t['entry_clock']),1,code,'BUY',ev,t['name'],float(t['entry_price'])))
  events.append((str(t['t1_date']),str(t['exit_clock']),0,code,'SELL',ev,t['name'],float(t['exit_price'])))
 events=sorted(events)
 d=daily.copy()
 d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
 d['code']=d.code.astype(str).str.zfill(6)
 d=d[d.date>='2026-10-08']
 marks=dict(zip(zip(d.code,d.date),d.close.astype(float)))
 dates=sorted(d.date.unique())
 cash=1.;held={};last={};log=[];equity=[]
 for day in dates:
  for event in (ev for ev in events if ev[0]==day):
   _,hour,order,code,side,eid,name,px=event
   if side=='BUY':
    if len(held)>=slots:log.append({'date':day,'event_id':eid,'status':'NO_SLOT'});continue
    nav=cash+sum(x['shares']*last.get(x['code'],x['entry_price']) for x in held.values())
    target=nav/slots;budget=min(cash,target)
    if budget<target*.5:log.append({'date':day,'event_id':eid,'status':'INSUFFICIENT_CASH'});continue
    shares=budget/(px*(1+fee));cash-=budget
    held[eid]={'code':code,'name':name,'shares':shares,'buy_cash':budget,'entry_price':px}
    log.append({'date':day,'time':hour,'event_id':eid,'name':name,'status':'BOUGHT','price':px,'budget':budget})
   else:
    if eid not in held:continue
    x=held.pop(eid)
    proceeds=x['shares']*px*(1-fee);cash+=proceeds
    log.append({'date':day,'time':hour,'event_id':eid,'name':name,'status':'SOLD','price':px,
                'realized_return':proceeds/x['buy_cash']-1})
  for h in held.values():
   key=(h['code'],day)
   if key not in marks:raise RuntimeError(f'MISSING DAILY CLOSE: {key}')
   last[h['code']]=marks[key]
  nav=cash+sum(h['shares']*last[h['code']]*(1-fee) for h in held.values())
  equity.append({'date':day,'nav':nav,'cash':cash,'held_count':len(held)})
 q=pd.DataFrame(equity);m=pd.DataFrame(log)
 if len(q):
  worst=float((q.nav/q.nav.cummax()-1).min())
  final=float(q.nav.iloc[-1])
 else:worst=None;final=None
 result={'max_slots':slots,'fee_per_side':fee,
         'candidate_trades':len(trades),'closed_executed':int(sum(z.get('status')=='SOLD' for z in log)),
         'skipped':int(sum(z.get('status') in ['NO_SLOT','INSUFFICIENT_CASH'] for z in log)),
         'terminal_nav':final,'account_return':final-1 if final is not None else None,
         'worst_close_to_close_drawdown':worst,
         'warning':'Paper account; hindsight outcomes are accepted only from pre-registered Day1 trades. Interim results may be incomplete if other signals still await T1.'}
 return result,q,m

def scan():
 proto=verify()
 datafile=ROOT/'v21_forward_results/T1_closed.json'
 obj={'updated_at':core.now().isoformat(),'capital_protocol_hash':proto['config_hash'],
      'base_v21_protocol_hash':json.loads((ROOT/'v21_forward_results/protocol_frozen.json').read_text())['spec_hash'],
      'completed_v21_preregistered_trades':0,'pending_entries':None,
      'scenario_results':[],'status':'WAITING_OOS_TRADES'}
 if not datafile.exists():
  dump(OUT/'capital_status.json',obj)
  print(json.dumps(obj,ensure_ascii=False,indent=2));return
 T=json.loads(datafile.read_text())
 good=[x for x in T if x.get('registered_before_day2')==True]
 if len(good)==0:
  dump(OUT/'capital_status.json',obj)
  print(json.dumps(obj,ensure_ascii=False,indent=2));return
 if len(set(x['event_id'] for x in good))!=len(good):raise RuntimeError('DUPLICATE CAPITAL EVENTS')
 daily,newdays=core.hydrate_daily_history()
 if daily is None:raise RuntimeError('NO NEW DAILY DATA WHILE FORWARD TRADES EXISTS')
 obj['completed_v21_preregistered_trades']=len(good)
 statusfile=ROOT/'v21_forward_results/forward_status.json'
 if statusfile.exists():obj['pending_entries']=json.loads(statusfile.read_text()).get('pending_entries')
 for slots in CFG['max_slots']:
  for slip in CFG['slippage_stress_per_side']:
   m,q,l=simulate(good,daily,slots,CFG['buy_fee']+slip)
   m['additional_slippage_per_side']=slip
   obj['scenario_results'].append(m)
   label=f'slots{slots}_slip{int(slip*10000)}bps'
   if len(q):q.to_csv(OUT/('equity_'+label+'.csv'),index=False)
   if len(l):l.to_csv(OUT/('execution_'+label+'.csv'),index=False)
 obj['status']='SHADOW_CAPITAL_ACCOUNT_ONLY'
 dump(OUT/'capital_status.json',obj)
 print(json.dumps(obj,ensure_ascii=False,indent=2))
def main():
 a=argparse.ArgumentParser()
 g=a.add_mutually_exclusive_group(required=True)
 g.add_argument('--init',action='store_true')
 g.add_argument('--scan',action='store_true')
 args=a.parse_args()
 if args.init:init()
 if args.scan:scan()
if __name__=='__main__':main()