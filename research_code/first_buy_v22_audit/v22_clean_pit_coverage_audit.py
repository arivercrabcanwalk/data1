# -*- coding: utf-8 -*-
"""Read-only audit: event point-in-time equivalence, coverage attrition, historical selection multiplicity."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from v17_structural_lab import lifecycle_events
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit'
OUT.mkdir(exist_ok=True)
B=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy()
E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
D=pd.read_parquet(ROOT/'v17_data/daily.parquet').copy()
P=pd.read_parquet(ROOT/'v20_results/yao_scored.parquet').copy()
for col in ['date','d2_date','f2_date']:
 for z in [E,P,B]:
  if col in z.columns:z[col]=pd.to_datetime(z[col])
D['date']=pd.to_datetime(D.date)
test=E[E.event_id.astype(str).isin(set(B.event_id.astype(str)))].copy()
# Additional previously excluded events include all historical candidate categories.
univ=E[(E.date>='2026-01-01')&(E.date<='2026-09-30')&(E.prior_streak>=3)].copy()
allcheck=univ.sample(n=min(160,len(univ)),random_state=20261008)
allcheck=pd.concat([test,allcheck]).drop_duplicates('event_id').sort_values('date')
checks=[]
by_code={c:z.copy() for c,z in D.groupby('code')}
for _,r in allcheck.iterrows():
 code=str(r.code);day=pd.Timestamp(r.date)
 v=by_code.get(code)
 if v is None:continue
 z=v[v.date<=day].copy().sort_values('date')
 if z.empty:continue
 g=z.groupby('code',sort=False)
 for k in range(1,6):
  z[f'f{k}_date']=g.date.shift(-k);z[f'f{k}_open']=g.open.shift(-k)
  z[f'f{k}_close']=g.close.shift(-k)
  z[f'f{k}_reset']=g.price_reset.shift(-k).fillna(False)
 ep=lifecycle_events(z)
 this=ep[(ep.date==day)&(ep.event_id==r.event_id)] if not ep.empty else ep
 valid=(len(this)==1)
 changed={}
 if valid:
  q=this.iloc[0]
  for col in ['event_type','generation','prior_streak','branch','bucket','close','close_vwap','vol_prev','amount','ret','gap','close_loc','market_height']:
   a=r[col];b=q[col]
   same=(pd.isna(a) and pd.isna(b)) or a==b or (isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)) and np.isfinite(a) and np.isfinite(b) and abs(a-b)<1e-9)
   if not same:changed[col]={'full':str(a),'asof':str(b)}
 checks.append({'event_id':str(r.event_id),'date':str(day.date()),'name':str(r['name']),
                'in_v21':str(r.event_id) in set(B.event_id.astype(str)),
                'asof_event_exists':valid,'changed_fields':changed,
                'full_rearm_within_30d':bool(r.rearmed_within_30d),
                'asof_rearm_within_30d':bool(this.iloc[0].rearmed_within_30d) if valid else None,
                'full_has_future_t1':pd.notna(r.exit_t1_close),
                'asof_has_future_t1':pd.notna(this.iloc[0].exit_t1_close) if valid else None})
report={'tests':len(checks),'not_found':sum(not x['asof_event_exists'] for x in checks),
        'exante_feature_mismatches':sum(bool(x['changed_fields']) for x in checks),
        'matches_v21':sum(x['in_v21'] for x in checks),
        'full_data_later_rearm_changed':sum(x['full_rearm_within_30d']!=x['asof_rearm_within_30d'] for x in checks if x['asof_event_exists']),
        'full_events_have_future_t1':sum(x['full_has_future_t1'] for x in checks),
        'asof_events_have_future_t1':sum(x['asof_has_future_t1'] for x in checks),
        'mismatch_cases':[x for x in checks if not x['asof_event_exists'] or x['changed_fields']][:20]}
# Panel coverage denominator and explicit losses
p=P[(P.prior_streak>=3)&P.exit_t1_close.notna()&
    ~P.d2_price_reset.fillna(False)&~P.exit_t1_reset.fillna(False)].copy()
p['minute_available']=p.d2_open_proxy.notna()
p['d2_open_from_daily']=p.d2_open.astype(float)
p['unrestricted_d2_daily_ret']=p.exit_t1_close/p.d2_open_from_daily-1-.0052
p['v21']=p.event_id.isin(set(B.event_id))
seg=[]
for name,q in [('minute_available',p[p.minute_available]),('minute_missing',p[~p.minute_available])]:
 x=q.unrestricted_d2_daily_ret
 seg.append({'group':name,'n':len(q),'daily_open_proxy_mean':float(x.mean()),'daily_open_proxy_win':float((x>0).mean()),
             'daily_open_proxy_worst':float(x.min()),'v21_selected':int(q.v21.sum()),'median_d1_amount':float(q.amount.median()),
             'branch_counts':q.branch.value_counts().to_dict()})
report['minute_coverage']={'with_t1_and_no_price_reset':len(p),'available':int(p.minute_available.sum()),
                            'missing':int((~p.minute_available).sum()),'segments':seg,
                            'caution':'Daily price proxy comparison is not evidence of executable fills, as halted or 1-price limit bars may be missing from minute panels.'}
# Label selection: pipeline uses three major time ranges repeatedly
report['pseudo_oos_is_seen']=True
report['metadata_point_in_time_verified']=json.loads((ROOT/'v17_results/canonical_audit.json').read_text()).get('label_point_in_time_verified')
report['baseline_52']={'wins':int((B.final_ret>0).sum()),'n':len(B),'unique_d2_dates':int(test.d2_date.nunique()),
                        'unique_day1_dates':int(test.date.nunique()),
                        'distinct_codes':int(B.code.nunique())}
(OUT/'pit_coverage_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
pd.DataFrame(checks).to_csv(OUT/'day1_asof_checks.csv',index=False)
print(json.dumps(report,ensure_ascii=False,indent=2))