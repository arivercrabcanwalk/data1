from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v19_results'; OUT.mkdir(exist_ok=True)
COST=.0052
OPEN_CVWAP_MAX=.965
M5_GAP_MAX=-.03
M5_GAIN_MIN=.03

def met(df,col):
    r=pd.to_numeric(df[col],errors='coerce').dropna()
    if len(r)==0:
        return {'n':0,'win_rate':None,'mean':None,'median':None,'bad3':None,'worst':None,'sum':0.0}
    return {
        'n':int(len(r)),'win_rate':float((r>0).mean()),'mean':float(r.mean()),
        'median':float(r.median()),'bad3':float((r<=-.03).mean()),
        'worst':float(r.min()),'sum':float(r.sum())
    }

def split(df,col):
    return {
        'train_jan_jun':met(df[df.date<='2026-06-30'],col),
        'pseudo_oos_jul_sep':met(df[df.date>='2026-07-01'],col),
        'all_2026':met(df,col)
    }

def main():
    T=pd.read_parquet(OUT/'b4p_panic_timing_table.parquet').copy()
    T['date']=pd.to_datetime(T.date)

    open_lane=T[T.d1_close_vwap<=OPEN_CVWAP_MAX].copy()
    open_lane['lane']='B4P_PANIC_CAPITULATION_OPEN'
    open_lane['tier']='TEMPORALLY_CONFIRMED_EXPERIMENTAL'
    open_lane['entry_time']='OPEN'
    open_lane['entry_price']=open_lane.d2_open
    open_lane['t1_ret']=open_lane.open_ret

    # Standard fallback keeps the V18 small-n 5-minute structure, but only for events
    # not already assigned to the open lane.
    m5=T[
        (T.d1_close_vwap>OPEN_CVWAP_MAX)&
        (T.d2_gap<=M5_GAP_MAX)&
        (T.m5_ret_from_open>=M5_GAIN_MIN)&T.m5_above_vwap
    ].copy()
    m5['lane']='B4P_PANIC_DEEP_IMPULSE5'
    m5['tier']='SMALL_N_RESEARCH'
    m5['entry_time']=m5.entry_time_after_m5
    m5['entry_price']=m5.entry_after_m5
    m5['t1_ret']=m5.ret_after_m5

    tree=pd.concat([open_lane,m5],ignore_index=True,sort=False)
    tree=tree.sort_values(['date','code'])
    tree.to_parquet(OUT/'v19_panic_decision_tree_trades.parquet',index=False)

    # Compare whether waiting improves the exact open-lane subset.
    wait_compare={}
    for col in ['open_ret','ret_after_m1','ret_after_m3','ret_after_m5']:
        wait_compare[col]=split(open_lane,col)

    # Threshold neighborhood: important anti-overfit audit.
    cvwap_neighborhood=[]
    for th in [.955,.96,.9625,.965,.9675,.97]:
        z=T[T.d1_close_vwap<=th]
        cvwap_neighborhood.append({'threshold':th,**split(z,'open_ret')})

    m5_neighborhood=[]
    for th in [.02,.025,.0275,.03,.0325,.035,.04]:
        z=T[(T.d2_gap<=M5_GAP_MAX)&(T.m5_ret_from_open>=th)&T.m5_above_vwap]
        m5_neighborhood.append({'threshold':th,**split(z,'ret_after_m5')})

    # Stress open lane by doubling the incremental round-trip cost and dropping top winners.
    stress=open_lane.copy()
    stress['double_cost_ret']=stress.open_ret-COST
    ranked=open_lane.sort_values('open_ret',ascending=False)
    stress_metrics={
        'base':split(open_lane,'open_ret'),
        'double_cost_all':met(stress,'double_cost_ret'),
        'drop_top1_all':met(ranked.iloc[1:],'open_ret'),
        'drop_top2_all':met(ranked.iloc[2:],'open_ret')
    }

    # Combine with frozen V18 stack for a research-only aggregate view.
    S=pd.read_parquet(ROOT/'v18_results/frozen_stack_trades.parquet').copy()
    S['date']=pd.to_datetime(S.date); S['ret']=S.t1_ret; S['source']='V18_FROZEN'
    V=tree[['event_id','date','code','name','t1_ret','lane']].copy()
    V['ret']=V.t1_ret; V['source']='V19_'+V.lane
    combo=pd.concat([
        S[['event_id','date','code','name','ret','source']],
        V[['event_id','date','code','name','ret','source']]
    ],ignore_index=True).sort_values(['date','event_id'])
    if combo.event_id.duplicated().any():
        raise RuntimeError('unexpected duplicate event across V18/V19 branches')
    combo.to_parquet(OUT/'v18_v19_research_stack.parquet',index=False)

    summary={
        'status':'V19_FROZEN_RESEARCH_NOT_PRODUCTION',
        'data_through':'2026-09-30',
        'round_trip_cost':COST,
        'core_finding':'B4P PANIC should not use one universal timing rule. A severe Day1 close-below-VWAP capitulation subset supports direct open entry; the remaining deep-gap cases still need 5-minute confirmation.',
        'open_lane':{
            'name':'B4P_PANIC_CAPITULATION_OPEN',
            'rule':'B4P PANIC and Day1 close/VWAP <= 0.965. Submit for Day2 opening match; no Day2 minute confirmation is required.',
            'status':'TEMPORALLY_CONFIRMED_EXPERIMENTAL',
            'metrics':split(open_lane,'open_ret'),
            'wait_comparison':wait_compare,
            'stress':stress_metrics,
            'threshold_neighborhood':cvwap_neighborhood
        },
        'fallback_lane':{
            'name':'B4P_PANIC_DEEP_IMPULSE5',
            'rule':'If not in open lane: Day2 gap <= -3%, first 5 complete minutes gain >=3% from Day2 open, minute-5 close >= cumulative VWAP; enter minute-6 open.',
            'status':'SMALL_N_RESEARCH',
            'metrics':split(m5,'t1_ret'),
            'threshold_neighborhood':m5_neighborhood
        },
        'decision_tree':split(tree,'t1_ret'),
        'combined_v18_v19_research_stack':split(combo,'ret'),
        'anti_overfit_warning':'The open lane was discovered with access to 2026 history. Jul-Sep is only temporal confirmation, not true unseen data. The rule is frozen now and must not be altered after the next trading session is observed.'
    }
    (OUT/'v19_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    main()