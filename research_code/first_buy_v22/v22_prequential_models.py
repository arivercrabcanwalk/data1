# -*- coding: utf-8 -*-
"""Small fixed model-family prequential OOS research. No future input features and no oracle entry times.
Candidate threshold is selected solely on Mar-Jun walk-forward OOF, then Jul-Sep is scored once.
Jul-Sep remains pseudo-OOS as repeatedly studied before this file.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_results';OUT.mkdir(exist_ok=True)
P=pd.read_parquet(OUT/'oracle_event_options.parquet').copy()
P['date']=pd.to_datetime(P.date);P['d2_date']=pd.to_datetime(P.d2_date);P['f2_date']=pd.to_datetime(P.f2_date)
B=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy()
P=P[P.usable_M6_OPEN & ~P.exit_t1_reset.fillna(False)].copy()
P['r']=P.ret_t1_entry_after_m5.astype(float)
P['win']=(P.r>0).astype(int)
P['month']=P.d2_date.dt.to_period('M').astype(str)
P['d2_gap_exante']=P.d2_open_proxy/P.close-1
P['amount_log']=np.log1p(P.amount.clip(lower=0))
P['vol_prev_log']=np.log1p(P.vol_prev.clip(lower=0))
P['ret20_clip']=P.ret20.clip(-.9,3)
P['m5_ret_clip']=P.m5_ret.clip(-.2,.2)
P['m1_ret_clip']=P.m1_ret.clip(-.2,.2)
P['m3_ret_clip']=P.m3_ret.clip(-.2,.2)
P['streak_clip']=P.prior_streak.clip(3,10)
P['is_recycle']=P.event_type.eq('RECYCLE_2PLUS').astype(int)
num_cols=['streak_clip','is_recycle','ret','gap','range','vol_prev_log','amount_log',
          'close_loc','close_vwap','ret20_clip','broad_up','market_height','market_gap',
          'd2_gap_exante','m1_ret_clip','m3_ret_clip','m5_ret_clip',
          'm1_to_m5_low','m5_above_cvwap','pm_above_ratio','pm_minus_am']
cat_cols=['branch','bucket']
ALL=num_cols+cat_cols
P=P.replace([np.inf,-np.inf],np.nan)

def make_model(kind):
 pre=ColumnTransformer([
  ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',RobustScaler())]),num_cols),
  ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_cols)],
  remainder='drop',sparse_threshold=0)
 model=(Ridge(alpha=100) if kind=='Ridge' else LogisticRegression(C=.25,max_iter=1000))
 return Pipeline([('pre',pre),('model',model)])

def fit_predict(kind,tr,te):
 mdl=make_model(kind)
 if kind=='Ridge':
  mdl.fit(tr[ALL],tr.r)
  return mdl.predict(te[ALL])
 mdl.fit(tr[ALL],tr.win)
 return mdl.predict_proba(te[ALL])[:,1]

def get_oof(kind):
 pieces=[]
 for vmonth in ['2026-03','2026-04','2026-05','2026-06']:
  tr=P[(P.month<vmonth)&(P.date<='2026-06-30')]
  te=P[P.month==vmonth].copy()
  if len(tr)<25 or len(te)==0:continue
  te['score']=fit_predict(kind,tr,te)
  pieces.append(te)
 return pd.concat(pieces) if pieces else P.iloc[:0].assign(score=[])

def score_selection(z,threshold,top_per_day=1):
 x=z[(~z.in_v21)&z.score.ge(threshold)].copy()
 x=x.sort_values(['d2_date','score','event_id'],ascending=[True,False,True])
 if top_per_day is not None:x=x.groupby('d2_date',as_index=False,group_keys=False).head(top_per_day)
 return x

def met(x):
 r=x.r.dropna()
 if not len(r):return {'n':0,'win':None,'avg':None,'worst':None,'sum':0,'month_count':0}
 return dict(n=len(r),win=float((r>0).mean()),avg=float(r.mean()),worst=float(r.min()),
             sum=float(r.sum()),month_count=int(x.month.nunique()),names=x.name.astype(str).tolist())

def train_choose(oof):
 missed=oof[~oof.in_v21]
 # Fixed candidate menu; fitted thresholds use only pseudo-prequential Mar-Jun model scores.
 qlevels=[.50,.65,.75,.85,.90,.95]
 picks=[]
 for q in qlevels:
  th=float(missed.score.quantile(q))
  for max_daily in [1,2]:
   z=score_selection(oof,th,max_daily)
   m=met(z)
   # Strong acceptance standard; n>=6 across 3 months; no big left-tail.
   if (m['n']>=6 and m['month_count']>=3 and m['win']>=.75
       and m['avg']>=.06 and m['worst']>=-.05):
    picks.append((m['sum'],m['avg'],m['win'],q,-max_daily,th,max_daily,m))
 return sorted(picks,reverse=True),len(qlevels)*2

results=[]
for kind in ['Ridge','Logit']:
 oof=get_oof(kind)
 choices,ntried=train_choose(oof)
 print('MODEL',kind,'OOF_all',met(oof),'OOF_missing',met(oof[~oof.in_v21]))
 print('PREREG_ACCEPTED_TRAIN_CANDIDATES',len(choices),'from',ntried)
 for row in choices[:5]:print('   ',row)
 # Take only the top train-only policy; no peeking at Jul-Sep before choose.
 trained=choices[0] if choices else None
 if trained is None:
  result={'model':kind,'train_accepted':False,'train_tries':ntried,'oof_missing':met(oof[~oof.in_v21])}
  results.append(result)
  continue
 _,_,_,q,neg_daily,th,max_daily,trainstat=trained
 tr=P[P.date<='2026-06-30'];te=P[(P.date>='2026-06-30')&(P.date<='2026-09-30')].copy()
 te['score']=fit_predict(kind,tr,te)
 selected=score_selection(te,th,max_daily)
 trainsel=score_selection(oof,th,max_daily)
 result={'model':kind,'train_accepted':True,'train_tries':ntried,'threshold_from_train':th,
        'q_level':q,'max_per_day':max_daily,
        'training_mar_jun_OOF':met(trainsel),'pseudo_jul_sep_once':met(selected)}
 outname=f'v22_{kind.lower()}_prequential.csv'
 outcols=['event_id','date','d2_date','code','name','branch','bucket','prior_streak','score','entry_after_m5','r']
 selected[outcols].to_csv(OUT/outname,index=False,encoding='utf-8-sig')
 print('SELECTED pseudo',result)
 results.append(result)

(OUT/'prequential_models.json').write_text(json.dumps({
  'rule_date':'2026-10-08',
  'is_pseudo_oos_only':True,
  'candidate_entry':'Day2 M6 open after the first 5 complete minutes',
  'exit':'T1 close with 0.52% cost',
  'feature_columns':ALL,
  'candidate_menu':'2 model types, 6 training-score quantiles, max 1 or 2 per D2 day; any survivor must pass strict training-only filters',
  'results':results
},ensure_ascii=False,indent=2,default=str))