# -*- coding: utf-8 -*-
"""Count inspected candidate families without selecting a single profitable rule."""
from pathlib import Path
import ast,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit';OUT.mkdir(exist_ok=True)
X=pd.read_parquet(ROOT/'v21_results/exit_audit_signals.parquet')
X['date']=pd.to_datetime(X.date)
n_exit=0;by_lane={}
for lane,G in X.groupby('lane'):
 tr=G[G.date<='2026-06-30']
 if len(tr)<3:continue
 count=0
 for n in [1,3,5]:
  for feat in [f't1_signal_m{n}',f't1_m{n}_vs_open']:
   vals=pd.to_numeric(tr[feat],errors='coerce').dropna()
   if len(vals)<3:continue
   ths=set(round(float(vals.quantile(q)),3) for q in [.2,.33,.5,.67,.8])
   count+=len(ths)*2*2
 by_lane[lane]=count;n_exit+=count
lows=[None,-.08,-.06,-.05,-.04,-.03,-.025,-.02,-.015,-.01,0]
highs=[None,.01,.02,.03,.04,.05,.06,.07,.08,.10,.12,.15]
n_universal=sum(1 for n in [1,3,5] for l in lows for h in highs if not(l is None and h is None) and not(l is not None and h is not None and l>=h))
# An approximate exploratory combinatorial family size before reward-based quality cuts, not actual passing policies.
P=pd.read_parquet(ROOT/'v20_results/yao_scored.parquet')
B=pd.read_parquet(ROOT/'v20_results/v20_49_trade_stack.parquet')
P['date']=pd.to_datetime(P.date)
P=P[~P.event_id.isin(set(B.event_id.astype(str)))&(P.event_type=='PRIMARY_3PLUS')&P.bucket.isin(['B3','B4P'])]
feats={
 'OPEN':['ret','gap','range','vol_prev','amount','close_loc','close_vwap_dist','am_above_ratio','pm_above_ratio','pm_minus_am','pm_mean_vwap_dist','last60_above_ratio','last30_ret','late_giveback','d1_auction_gap','d2_auction_gap_final','auction_gap_upgrade','auction_vol_log_ratio','market_height','rank','followers','height_edge'],
 'M1':['m1_ret','m1_above_cvwap'],
 'M3':['m1_ret','m1_above_cvwap','m3_ret','m3_above_cvwap','m3_above_count'],
 'M5':['m1_ret','m1_above_cvwap','m3_ret','m3_above_cvwap','m3_above_count','m5_ret','m5_above_cvwap','m5_above_count','m1_to_m5_low']}
cnt=[]
for (branch,bucket),G in P.groupby(['branch','bucket']):
 tr=G[G.date<='2026-06-30']
 if len(tr)<6:continue
 for e in ['OPEN','M1','M3','M5']:
  cols=feats['OPEN']+(feats[e] if e!='OPEN' else [])
  d=0
  for f in cols:
   if f not in G.columns:continue
   x=pd.to_numeric(tr[f],errors='coerce').dropna()
   if len(x)<6 or x.nunique()<3:continue
   nums=set()
   for q in [.2,.33,.4,.5,.6,.67,.8]:
    v=float(x.quantile(q));a=abs(v)
    th=round(v/1e8)*1e8 if a>=1e9 else round(v/1e7)*1e7 if a>=1e8 else round(v,0) if a>=10 else round(v,1) if a>=1 else round(v,3)
    nums.add(th)
   d+=2*len(nums)
  cnt.append({'branch':branch,'bucket':bucket,'entry':e,'univariate_conditions':d,
              'single_condition_capacity':d,
              'pairs_upperbound_before_reward_filter':min(d,55)*(min(d,55)-1)//2})
report={'prior_2026_optimize_exits_lane_specific_tried_or_evaluated':n_exit,
        'lane_specific_detail':by_lane,
        'universal_exit_parameter_combinations_before_na_filter':n_universal,
        'two_more_dense_entry_searches_exist':True,
        'single_and_pair_condition_search_capacity':cnt,
        'simple_condition_count_sum':sum(x['univariate_conditions'] for x in cnt),
        'upperbound_two_condition_candidates':sum(x['pairs_upperbound_before_reward_filter'] for x in cnt),
        'v22_Ridge_Logit_explicit_policy_trials':24,
        'v22_model_policy_passing':0,
        'critical_design_issue':'July-September results were included in pass/fail gates repeatedly in 2026. No untouched holdout remains.',
        'note':'Lane count and universal candidate loop counts come from code-defined parameter grids; combinatorial capacities are upper bounds before outcome filters, not actual fully scored trial counts. Do not add them as an exact false discovery denominator.'}
(OUT/'search_multiplicity_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))