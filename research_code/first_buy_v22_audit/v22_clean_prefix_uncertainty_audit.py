# -*- coding: utf-8 -*-
"""Historical 52 event-by-event causality, cost, confidence and executable bar gates."""
import json, math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta
from v21_forward_engine import read_for_events,pm_above_ratio
import v22_strict_causal_core as policy
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit';OUT.mkdir(exist_ok=True)
X=pd.read_csv(ROOT/'v22_results/strict_causal_completed_historical.csv',dtype={'code':str})
E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
for col in ['date','d2_date','f2_date']:E[col]=pd.to_datetime(E[col])
S=E[E.event_id.astype(str).isin(X.event_id.astype(str))].copy()
sby=read_for_events(S,'d2_date');tby=read_for_events(S,'f2_date');pby=read_for_events(S,'date')
assert len(S)==52 and len(X)==52
tests=[];priced=[]
for _,r in S.iterrows():
 x=X.loc[X.event_id==r.event_id].iloc[0]
 code=str(r.code).zfill(6); d2=str(r.d2_date.date());t1=str(r.f2_date.date());d1=str(r.date.date())
 s=sby[d2][code].copy();t=tby[t1][code].copy()
 p=pby.get(d1,{}).get(code)
 pm=pm_above_ratio(p) if p is not None else None
 full,why=policy.entry(r,s,pm)
 assert full is not None and full['lane']==x.lane and abs(full['entry_price']-x.entry_price)<.005
 i=full['entry_idx']
 # After entry decision: only the OPEN price at index i is observable; all other data unknown.
 cut=s.copy()
 cut.loc[i,['high','low','close','volume','amount']]=np.nan
 if i+1<len(cut):
  cut.loc[i+1:,['open','high','low','close','volume','amount']]=np.nan
 short,re=policy.entry(r,cut,pm)
 pit_ok=bool(short is not None and short==full)
 # Mutation invariance in T1 exit after minute+1 open for early exits
 fex,why=policy.exit_t1(full,t)
 exit_label=str(fex['exit_rule']) if fex else 'MISSING'
 t1_cut_ok=None
 if fex and exit_label!='T1_CLOSE':
  j={'PO_T1_M2_TP':1,'P3_T1_M6_TP':5,'LM3_T1_M4_TP':3,
     'POS_T1_M4_STOP':3,'R4_T1_M4_STOP':3}[exit_label]
  tt=t.copy()
  tt.loc[j,['high','low','close','volume','amount']]=np.nan
  if j+1<len(tt):tt.loc[j+1:,['open','high','low','close','volume','amount']]=np.nan
  t1_now,reason=policy.exit_t1(full,tt)
  t1_cut_ok=bool(t1_now is not None and t1_now==fex)
 buy_px=float(fex['entry_price']);sell_px=float(fex['exit_price'])
 d2upper=math.floor(float(r.close)*110+.5+1e-8)/100
 d2lower=math.floor(float(r.close)*90+.5+1e-8)/100
 t1lower=math.floor(float(r.d2_close)*90+.5+1e-8)/100
 sell_i=(-1 if exit_label=='T1_CLOSE' else
    {'PO_T1_M2_TP':1,'P3_T1_M6_TP':5,'LM3_T1_M4_TP':3,
     'POS_T1_M4_STOP':3,'R4_T1_M4_STOP':3}[exit_label])
 bar=s.iloc[i];sb=t.iloc[sell_i]
 priced.append({'event_id':r.event_id,'code':code,'name':r['name'],'d1':d1,'entry_clock':full['entry_clock'],
         'entry':buy_px,'d2_lower':d2lower,'d2_upper':d2upper,
         'buy_oneprice':bool(bar.high==bar.low),'buy_upper_locked':bool(buy_px>=d2upper-.005 and bar.high==bar.low),
         'buy_lower_locked':bool(abs(buy_px-d2lower)<.005 and bar.high==bar.low),
         'buy_bar_volume':float(bar.volume),
         'sell':sell_px,'sell_clock':fex['exit_clock'],'t1_lower':t1lower,
         'sell_oneprice':bool(sb.high==sb.low),'sell_lower_locked':bool(abs(sell_px-t1lower)<.005 and sb.high==sb.low),
         'sell_bar_volume':float(sb.volume),
         'd2_missing_any_bar':bool(s[['open','high','low','close','volume','amount']].isna().any().any()),
         't1_missing_any_bar':bool(t[['open','high','low','close','volume','amount']].isna().any().any()),
         'ret':float(fex['net_ret']),'lane':fex['lane']})
 tests.append({'event_id':r.event_id,'pit_entry_match':pit_ok,'exit':exit_label,'early_exit_pit_match':t1_cut_ok})
tests=pd.DataFrame(tests);fills=pd.DataFrame(priced)
assert tests.pit_entry_match.all() and tests[tests.early_exit_pit_match.notna()].early_exit_pit_match.all()
fills.to_csv(OUT/'strict_52_bar_feasibility.csv',index=False)
tests.to_csv(OUT/'strict_52_prefix_checks.csv',index=False)
n=len(X);wins=int((X.net_ret>0).sum());z=1.96;p=wins/n
den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z/den*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
# Day1 blocks bootstrap; pre-specified deterministic rng makes summary reproducible, not confirmatory interval.
X['day1']=pd.to_datetime(X.day1).dt.strftime('%Y-%m-%d')
days=sorted(X.day1.unique());rs=np.random.default_rng(20261008)
blocks=[X.loc[X.day1==day,'net_ret'].to_numpy(dtype=float) for day in days]
arr=[]
for _ in range(4000):
 indices=rs.integers(0,len(blocks),size=len(blocks))
 v=np.concatenate([blocks[i] for i in indices])
 arr.append([float((v>0).mean()),float(v.mean())])
a=np.array(arr)
stats={'n':n,'wins':wins,'win_rate':p,'wilson95':[center-half,center+half],
       'same_sample_dayblock_bootstrap_win95':np.quantile(a[:,0],[.025,.975]).tolist(),
       'same_sample_dayblock_bootstrap_mean95':np.quantile(a[:,1],[.025,.975]).tolist(),
       'bootstrap_warning':'Conditional on hand-picked 52 historical trades, ignores all model searching; NOT valid independent predictive confidence.',
       'pit_entry_pass':int(tests.pit_entry_match.sum()),
       'pit_early_exit_pass':int(tests.early_exit_pit_match.fillna(False).sum()),
       'early_exit_n':int(tests.early_exit_pit_match.notna().sum()),
       'n_buy_one_price':int(fills.buy_oneprice.sum()),
       'n_sell_one_price':int(fills.sell_oneprice.sum()),
       'n_buy_upper_locked':int(fills.buy_upper_locked.sum()),
       'n_buy_lower_locked':int(fills.buy_lower_locked.sum()),
       'n_sell_lower_locked':int(fills.sell_lower_locked.sum()),
       'buy_lower_lock_cases':fills.loc[fills.buy_lower_locked,['name','d1','entry']].to_dict('records'),
       'sell_lower_lock_cases':fills.loc[fills.sell_lower_locked,['name','d1','sell']].to_dict('records'),
       'sell_price_computed_from_actual_minute':int(len(fills)),
       'd2_missing_any_bar':int(fills.d2_missing_any_bar.sum()),
       't1_missing_any_bar':int(fills.t1_missing_any_bar.sum())}
(OUT/'strict_prefix_and_uncertainty.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stats,ensure_ascii=False,indent=2))