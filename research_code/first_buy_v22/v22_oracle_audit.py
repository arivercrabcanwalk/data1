# -*- coding: utf-8 -*-
"""HINDSIGHT ONLY: oracle upper-bound opportunity audit, never a trading signal."""
from pathlib import Path
import numpy as np,pandas as pd,json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_results';OUT.mkdir(exist_ok=True)
P=pd.read_parquet(ROOT/'v20_results/yao_scored.parquet').copy()
P['date']=pd.to_datetime(P.date);P['d2_date']=pd.to_datetime(P.d2_date);P['f2_date']=pd.to_datetime(P.f2_date)
B=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy()
base_ids=set(B.event_id.astype(str))
P=P[(P.date>='2026-01-01')&(P.date<='2026-09-30')&
    (P.prior_streak>=3)&P.exit_t1_close.notna()&
    ~P.d2_price_reset.fillna(False)&~P.exit_t1_reset.fillna(False)&
    P.d2_open_proxy.notna()].copy()
# At the time of entry, a bar open at a locked upper limit cannot be assumed fillable.
P['d2_upper']=np.floor(P.close*110+.5+1e-8)/100
names={
 'OPEN':('d2_open_proxy','ret_t1_d2_open_proxy'),
 'M2_OPEN':('entry_after_m1','ret_t1_entry_after_m1'),
 'M4_OPEN':('entry_after_m3','ret_t1_entry_after_m3'),
 'M6_OPEN':('entry_after_m5','ret_t1_entry_after_m5')}
for name,(col,ret) in names.items():
    P['usable_'+name]=(P[col].notna()&(P[col]>0)&(P[col]<P.d2_upper-.005)&P[ret].notna())
    P['oracle_'+name]=P[ret].where(P['usable_'+name])
# Oracle gets to choose the best of four known entries in hindsight.
cols=['oracle_'+x for x in names]
P['oracle_ret']=P[cols].max(axis=1)
P['oracle_entry']=P[cols].fillna(-np.inf).idxmax(axis=1).str.replace('oracle_','',regex=False)
P.loc[P.oracle_ret.isna(),'oracle_entry']=None
P['in_v21']=P.event_id.astype(str).isin(base_ids)
P['selection_gap']=np.where(P.in_v21,np.nan,P.oracle_ret)
P.to_parquet(OUT/'oracle_event_options.parquet',index=False)
U=P[~P.in_v21&P.oracle_ret.notna()].copy()
U=U.sort_values(['d2_date','oracle_ret'],ascending=[True,False])
for label,z in [('ALL',P),('V21_ELIGIBLE',P[P.in_v21]),('MISSED',U)]:
 print('---',label,len(z),'primary',sum(z.event_type=='PRIMARY_3PLUS'),'recycle',sum(z.event_type=='RECYCLE_2PLUS'))
 for name in names:
  q=z.loc[z['usable_'+name],names[name][1]]
  print(name,len(q),'win',round((q>0).mean(),3),'avg',round(q.mean(),4),'p10',round(q.quantile(.1),4),'gt10%',int((q>=.1).sum()),'lt-5%',int((q<=-.05).sum()))
 r=z.oracle_ret.dropna()
 print('HINDSIGHT_BEST',len(r),'avg',round(r.mean(),4),'positive',round((r>0).mean(),3),'gt5',int((r>.05).sum()),'gt10',int((r>.1).sum()),'gt15',int((r>.15).sum()))
print('NEW HIGH WINNERS ORACLE top 60')
show=['date','d2_date','code','name','event_type','prior_streak','branch','bucket','oracle_entry','oracle_ret','ret_t1_d2_open_proxy','ret_t1_entry_after_m3','ret_t1_entry_after_m5','ret','vol_prev','amount','close_loc']
f={'oracle_ret':lambda x:f'{x:+.1%}','ret_t1_d2_open_proxy':lambda x:f'{x:+.1%}' if pd.notna(x) else '-','ret_t1_entry_after_m3':lambda x:f'{x:+.1%}' if pd.notna(x) else '-','ret_t1_entry_after_m5':lambda x:f'{x:+.1%}' if pd.notna(x) else '-','ret':lambda x:f'{x:+.1%}','amount':lambda x:f'{x/1e8:.1f}亿' if pd.notna(x) else '-'}
print(U.nlargest(60,'oracle_ret')[show].to_string(index=False,formatters=f))
print('MISSED BY MONTH AND TYPE')
U['ym']=U.date.dt.strftime('%Y-%m')
print(U.groupby(['ym','branch','bucket']).oracle_ret.agg(n='size',gt10=lambda x:(x>.1).sum(),gt5=lambda x:(x>.05).sum(),avg='mean').round(3).to_string())
print('SAME-D2-CONFLICTS')
W=P[P.in_v21].groupby('d2_date').size()
print('V21 simultaneous days',int((W>1).sum()),'max',int(W.max()),'groups',W[W>1].to_string())
payload={
 'status':'POSTHOC_DIAGNOSTIC_NOT_EXECUTABLE',
 'date_end':'2026-09-30',
 'selection_options':list(names),
 'events':len(P),'missed':len(U),'missed_gt10_oracle':int((U.oracle_ret>.1).sum()),
 'missed_gt5_oracle':int((U.oracle_ret>.05).sum()),
 'warning':'Maximum price-timed return for each missed event is obtained using future outcome. This is a deliberately unattainable oracle upper bound, not a causal policy.'
}
(OUT/'oracle_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
U[show].to_csv(OUT/'missed_oracle_candidates.csv',index=False,encoding='utf-8-sig')