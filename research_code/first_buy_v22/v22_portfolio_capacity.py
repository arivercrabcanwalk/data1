# -*- coding: utf-8 -*-
"""Historical fixed-slot NAV simulation (diagnostic; optimized historical picks aren't truly OOS).

Keeps acquisition/sale clock ordering, cash limits, fixed 1/N target stakes and realistic overlaps.
Does NOT assume 100% reinvestment on each of 52 simultaneous or overlapping trades.
"""
from pathlib import Path
import pandas as pd, numpy as np,json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_results'; OUT.mkdir(exist_ok=True)
V=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy()
E=pd.read_parquet(ROOT/'v17_results/events.parquet')[['event_id','d2_date','d2_close']].copy()
V=V.merge(E,on='event_id',validate='one_to_one')
V['d2_date']=pd.to_datetime(V.d2_date);V['f2_date']=pd.to_datetime(V.f2_date)
H=pd.read_parquet(ROOT/'v17_data/daily.parquet',columns=['code','date','close'])
H=H[(H.date>='2026-01-01')&(H.date<='2026-09-30')].copy()
dates=sorted(H.date.unique())
hmap={(c,d):float(close) for c,d,close in H[['code','date','close']].itertuples(index=False,name=None)}
V['code']=V.code.astype(str).str.zfill(6)
def clock_entry(x):
 if str(x)=='OPEN':return '09:30'
 if str(x)=='M4_OPEN':return '09:33'
 if str(x)=='M6_OPEN':return '09:35'
 h,m=map(int,str(x)[:5].split(':'))
 return f'{h-(m==0):02d}:{(m-1)%60:02d}'
def clock_exit(x):
 if x=='T1_M2_OPEN_IF_M1_FROM_ENTRY_GT_8PCT':return '09:31'
 if x=='T1_M6_OPEN_IF_M5_VS_T1_OPEN_GT_0.5PCT':return '09:35'
 if str(x).startswith('T1_M4_OPEN'):return '09:33'
 return '15:00'
V['buy_clock']=V.entry_time.map(clock_entry)
V['sell_clock']=V.final_exit_rule.map(clock_exit)
V['sell_price']=(V.entry_price*(1+V.final_ret+.0052)).round(2)
assert V.buy_clock.notna().all() and V.sell_clock.notna().all()
ev=[]
for r in V.itertuples(index=False):
 ev.extend([
  {'date':str(r.d2_date.date()),'time':r.buy_clock,'kind':'BUY','event_id':r.event_id,
   'name':r.name,'code':r.code,'price':float(r.entry_price),'lane':r.lane},
  {'date':str(r.f2_date.date()),'time':r.sell_clock,'kind':'SELL','event_id':r.event_id,
   'name':r.name,'code':r.code,'price':float(r.sell_price),'lane':r.lane}
 ])
ev=pd.DataFrame(ev)
def simulate(slots=3,side_cost=.0026,extra_slip=.0):
 fee=side_cost+extra_slip
 cash=1.;pos={};last_mark={};execs=[];ledger=[]
 nskipped=0
 for day in dates:
  today=ev[ev.date==day].sort_values(['time','kind','event_id'],ascending=[True,False,True])
  for t in today.itertuples(index=False):
   if t.kind=='BUY':
    if len(pos)>=slots:
     execs.append({'event_id':t.event_id,'kind':'SKIPPED','day':day,'reason':'NO_FREE_SLOT','stock':t.name});nskipped+=1;continue
    nav=cash+sum(x['shares']*last_mark.get(x['code'],x['entry']) for x in pos.values())
    stake=min(cash,nav/slots)
    if stake<(nav/slots)*.5:
     execs.append({'event_id':t.event_id,'kind':'SKIPPED','day':day,'reason':'CASH_BELOW_HALF_TARGET','stock':t.name});nskipped+=1;continue
    shares=stake/((1+fee)*t.price)
    cash-=stake
    pos[t.event_id]={'shares':shares,'entry':t.price,'code':t.code,'name':t.name,
                     'cost':stake,'buy_day':day}
    execs.append({'event_id':t.event_id,'kind':'BUY','day':day,'time':t.time,'stock':t.name,'cash_notional':stake,'price':t.price})
   else:
    if t.event_id not in pos:continue
    x=pos.pop(t.event_id)
    gross=x['shares']*t.price
    proceeds=gross*(1-fee)
    cash+=proceeds
    execs.append({'event_id':t.event_id,'kind':'SELL','day':day,'time':t.time,'stock':t.name,
                  'price':t.price,'pnl_nav_units':proceeds-x['cost'],'realized_ret':proceeds/x['cost']-1})
  for x in pos.values():
   px=hmap.get((x['code'],day))
   if px is None:raise ValueError(('missing daily mark',x['code'],day))
   last_mark[x['code']]=px
  mtm=sum(x['shares']*last_mark[x['code']]*(1-fee) for x in pos.values())
  nav=cash+mtm
  ledger.append(dict(date=day,cash=cash,nav=nav,open_positions=len(pos),cap_utilization=1-cash/max(nav,1e-10)))
 eq=pd.DataFrame(ledger)
 dd=(eq.nav/eq.nav.cummax()-1)
 ret=eq.nav.pct_change().fillna(0)
 by=eq.assign(month=eq.date.str[:7]).groupby('month').agg(first=('nav','first'),last=('nav','last'))
 stats={'slots':slots,'fee_per_side':fee,'executed_trades':int(sum(x['kind']=='SELL' for x in execs)),
        'skipped_signals':nskipped,'terminal_nav':float(eq.nav.iloc[-1]),'net_roi':float(eq.nav.iloc[-1]-1),
        'worst_daily_drawdown':float(dd.min()),'max_concurrent':int(eq.open_positions.max()),
        'mean_capital_utilization':float(eq.cap_utilization.mean()),
        'min_daily_ret':float(ret.min()),'max_daily_ret':float(ret.max())}
 return stats,eq,pd.DataFrame(execs)
summary={}
for slots in [1,2,3,5]:
 for slip in [0,.001,.0026]:
  stats,eq,ex=simulate(slots,extra_slip=slip)
  summary[f'slots{slots}_extra_per_side{slip}']=stats
  if slots==3 and slip==0:
   eq.to_csv(OUT/'v21_3slot_historical_equity.csv',index=False)
   ex.to_csv(OUT/'v21_3slot_executions.csv',index=False)
  print(stats)
(OUT/'historical_portfolio_capacity.json').write_text(json.dumps({
 'status':'RETROSPECTIVE_PORTFOLIO_DIAGNOSTIC_NOT_TRUE_OOS',
 'starting_nav':1,
 'fee_model':'base 0.26% per-side x2, plus optional per-side stress',
 'marking':'Actual daily close for overnight positions; signal-time cash from previous day mark',
 'limitations':'Intraminute open may be unfillable; no queue priority, auction constraints, impact, sizing market impact, tax, or corporate actions. Exact strategy performance remains selected retrospectively.',
 'stats':summary
},indent=2))