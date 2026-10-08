# -*- coding: utf-8 -*-
"""Read-only feature firewall: strip every future-origin metadata field from 52 events."""
import json
from pathlib import Path
import pandas as pd,numpy as np
from v21_forward_engine import read_for_events,pm_above_ratio
import v22_strict_causal_core as core
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit';OUT.mkdir(exist_ok=True)
X=pd.read_csv(ROOT/'v22_results/strict_causal_entries_historical.csv',dtype={'code':str})
E=pd.read_parquet(ROOT/'v17_results/events.parquet')
for col in ['date','d2_date','f2_date']:E[col]=pd.to_datetime(E[col])
E=E[E.event_id.isin(X.event_id)].copy()
mins=read_for_events(E,'d2_date')
pmdata=read_for_events(E,'date')
allowed=['event_id','code','name','date','event_type','prior_streak','close','d2_price_reset',
         'branch','bucket','close_vwap','amount','gap','vol_prev','close_loc','ret']
issues=[];same=0
for _,r in E.iterrows():
 code=str(r.code).zfill(6);d=str(r.d2_date.date());day=str(r.date.date())
 s=mins.get(d,{}).get(code)
 pm=pm_above_ratio(pmdata.get(day,{}).get(code))
 full,status=core.entry(r,s,pm)
 # Future timeline fields are NOT handed to model, including reset status on T1.
 smaller=r[allowed].copy()
 # The D2 reset flag is known when D2 open prints; recompute from D1 CLOSE and D2 OPEN.
 smaller['d2_price_reset']=bool(abs(float(s.iloc[0].open)/float(r.close)-1)>.12)
 clean,status2=core.entry(smaller,s,pm)
 if full is not None and clean==full:same+=1
 else:issues.append({'event_id':r.event_id,'full':full,'firewalled':clean})
report={'events':len(E),'entry_same_with_no_future_columns':same,'not_same':len(issues),
        'never_passed_to_core':['f1_date','f1_open','f2_date','f2_open','exit_t1_close',
                    'exit_t1_reset','t1_ret','watch_until','full_30d_observable','rearmed_within_30d','oracle_ret'],
        'future_fields_in_legacy_event_schema':True,
        'meaning':'Although the legacy E row contains future fields, current V22 strict entry produces the same result when only causal features are supplied. Treat removing future fields at the data boundary as a hardening requirement.',
        'mismatch':issues}
assert len(E)==52 and len(issues)==0
(OUT/'feature_firewall_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))