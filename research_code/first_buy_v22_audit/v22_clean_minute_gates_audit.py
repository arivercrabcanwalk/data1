# -*- coding: utf-8 -*-
"""Read-only policy/market-data gate attrition audit."""
import json
from pathlib import Path
import pandas as pd, numpy as np
from v21_forward_engine import read_for_events,minute_complete
import v22_strict_causal_core as strict
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit';OUT.mkdir(exist_ok=True)
E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
for col in ['date','d2_date','f2_date']:E[col]=pd.to_datetime(E[col])
E=E[(E.date>='2026-01-01')&(E.date<='2026-09-30')&(E.prior_streak>=3)].copy()
E=E[(
 ((E.branch=='PANIC')&E.bucket.isin(['B3','B4P'])) |
 ((E.branch=='NORMAL_RED')&(E.bucket=='B4P')) |
 ((E.branch=='POSITIVE_BREAK')&(E.bucket=='B4P'))
 )].copy()
D1=read_for_events(E,'date')
D2=read_for_events(E.dropna(subset=['d2_date']),'d2_date')
stats=[];records=[]
for _,r in E.iterrows():
 c=str(r.code).zfill(6);day=str(r.date.date())
 d1=D1.get(day,{}).get(c)
 d2=D2.get(str(r.d2_date.date()),{}).get(c) if pd.notna(r.d2_date) else None
 def gate(s):
  if s is None:return 'FILE_MISSING'
  if len(s)<3:return 'FIRST3_MISSING'
  if str(s.iloc[0].time)!='09:31' or str(s.iloc[2].time)!='09:33':return 'PREFIX_LABEL_BAD'
  if s.iloc[:3][['open','high','low','close','volume','amount']].isna().any().any():return 'FIRST3_NULL'
  if (s.iloc[:3].volume<=0).any() or (s.iloc[:3].amount<=0).any():return 'FIRST3_ZERO'
  if minute_complete(s):return 'FULL_240_VALID'
  if len(s)==240 and str(s.iloc[119].time)=='13:00':return 'LUNCH_LABEL_13_00'
  if len(s)!=240:return f'ROWCOUNT_{len(s)}'
  return 'OTHER_FULL_DAY_INVALID'
 a=gate(d1);b=gate(d2)
 d2valid=b in ['FULL_240_VALID','LUNCH_LABEL_13_00','OTHER_FULL_DAY_INVALID','ROWCOUNT_240']  # early prefix complete
 if d2valid and d2 is not None and len(d2)>=5:
  v=(r.branch=='NORMAL_RED' and r.bucket=='B4P' and r.event_type=='PRIMARY_3PLUS' and a!='FULL_240_VALID')
 else:v=False
 records.append({'event_id':r.event_id,'name':r['name'],'code':c,'date':day,
                 'branch':r.branch,'bucket':r.bucket,'event_type':r.event_type,
                 'd1_gate':a,'d2_gate':b,'normal_b4p_global_pm_block':bool(v)})
R=pd.DataFrame(records)
payload={
 'n_events':len(R),'d1_gate_counts':R.d1_gate.value_counts().to_dict(),
 'd2_gate_counts':R.d2_gate.value_counts().to_dict(),
 'blocked_normal_primary_if_d1_pm_missing':int(R.normal_b4p_global_pm_block.sum()),
 'blocked_examples':R[R.normal_b4p_global_pm_block][['date','name','d1_gate','d2_gate']].head(22).to_dict('records'),
 'note':'Strict monitor currently refuses all NORMAL_RED B4P PRIMARY lanes when Day1 PM is missing, even though LM3 may be testable. Full-day validity is assessed retrospectively for this data audit, not used as a signal.'
}
(OUT/'minute_data_quality.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n')
R.to_csv(OUT/'minute_gate_by_event.csv',index=False)
print(json.dumps(payload,ensure_ascii=False,indent=2))