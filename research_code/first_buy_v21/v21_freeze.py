from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v21_results'; OUT.mkdir(exist_ok=True)
COST=.0052

def met(df,col):
    r=pd.to_numeric(df[col],errors='coerce').dropna()
    if len(r)==0:
        return {'n':0,'win_rate':None,'mean':None,'median':None,'worst':None,'bad3':None,'sum':0.0}
    return {'n':int(len(r)),'win_rate':float((r>0).mean()),'mean':float(r.mean()),
            'median':float(r.median()),'worst':float(r.min()),'bad3':float((r<=-.03).mean()),
            'sum':float(r.sum())}

def split(df,col):
    return {
        'train_jan_jun':met(df[df.date<='2026-06-30'],col),
        'pseudo_oos_jul_sep':met(df[df.date>='2026-07-01'],col),
        'all_2026':met(df,col)
    }

def main():
    # The 52-entry universe is V20's 49 entries + 3 non-overlapping M5 double-washout entries.
    B=pd.read_parquet(ROOT/'v20_results/v20_49_trade_stack.parquet').copy()
    B['date']=pd.to_datetime(B.date)
    X=pd.read_parquet(OUT/'exit_audit_52.parquet').copy()
    X['date']=pd.to_datetime(X.date)

    assert len(B)==49
    assert len(X)==52
    assert not X.event_id.duplicated().any()

    X['final_ret']=X.t1_close_ret
    X['final_exit_rule']='T1_CLOSE'

    # 1) V19 capitulation OPEN: T1 first-minute extreme strength is harvested early.
    m=X.lane.eq('B4P_PANIC_CAPITULATION_OPEN') & (X.t1_signal_m1>.08)
    X.loc[m,'final_ret']=X.loc[m,'t1_after_m1_ret']
    X.loc[m,'final_exit_rule']='T1_M2_OPEN_IF_M1_FROM_ENTRY_GT_8PCT'

    # 2) B3 PANIC core: if T1 first 5 minutes accelerate from the T1 open, harvest at M6.
    m=X.lane.eq('PANIC_3_LIQ_RECLAIM45') & (X.t1_m5_vs_open>.005)
    X.loc[m,'final_ret']=X.loc[m,'t1_after_m5_ret']
    X.loc[m,'final_exit_rule']='T1_M6_OPEN_IF_M5_VS_T1_OPEN_GT_0.5PCT'

    # 3) V20 liquid M3 lane: large T1 first-3-minute acceleration is harvested at M4.
    m=X.lane.eq('NORMAL_B4P_LIQUID_M3') & (X.t1_m3_vs_open>.05)
    X.loc[m,'final_ret']=X.loc[m,'t1_after_m3_ret']
    X.loc[m,'final_exit_rule']='T1_M4_OPEN_IF_M3_VS_T1_OPEN_GT_5PCT'

    # 4) Positive-break lane: cut if the position is already down >2% by T1 minute 3.
    m=X.lane.eq('POS_4P_VOLUME_RECLAIM35') & (X.t1_signal_m3<-.02)
    X.loc[m,'final_ret']=X.loc[m,'t1_after_m3_ret']
    X.loc[m,'final_exit_rule']='T1_M4_OPEN_STOP_IF_M3_FROM_ENTRY_LT_-2PCT'

    # 5) 4P NORMAL core: cut if T1 first 3 minutes fall >1.5% from the T1 open.
    m=X.lane.eq('CORE_RED_4P_REPAIR45') & (X.t1_m3_vs_open<-.015)
    X.loc[m,'final_ret']=X.loc[m,'t1_after_m3_ret']
    X.loc[m,'final_exit_rule']='T1_M4_OPEN_STOP_IF_M3_VS_T1_OPEN_LT_-1.5PCT'

    # Freeze.
    X=X.sort_values(['date','event_id']).reset_index(drop=True)
    X.to_parquet(OUT/'v21_52_trade_stack.parquet',index=False)

    base49=B.copy()
    base49['base_ret']=base49.trade_ret

    # Extra entry branch stats.
    extra=X[X.source.eq('EXTRA3')].copy()
    full_doublewashout=pd.read_parquet(OUT/'v21_double_washout_extra.parquet').copy()
    full_doublewashout['date']=pd.to_datetime(full_doublewashout.date)

    monthly=[]
    for month,g in X.groupby(X.date.dt.to_period('M').astype(str)):
        m=met(g,'final_ret');m['month']=month;monthly.append(m)

    stress=X.copy();stress['double_cost_ret']=stress.final_ret-COST
    ranked=X.sort_values('final_ret',ascending=False)
    stress_summary={
        'double_cost_all':met(stress,'double_cost_ret'),
        'drop_top1_all':met(ranked.iloc[1:],'final_ret'),
        'drop_top2_all':met(ranked.iloc[2:],'final_ret'),
        'drop_top3_all':met(ranked.iloc[3:],'final_ret'),
    }

    final_metrics=split(X,'final_ret')
    base_metrics=split(base49.rename(columns={'base_ret':'final_ret'}),'final_ret')
    theoretical_compound=float((1+X.sort_values('date').final_ret).prod()-1)

    payload={
        'status':'V21_RESEARCH_FREEZE_NOT_PRODUCTION',
        'freeze_date':'2026-10-08',
        'data_through':'2026-09-30',
        'round_trip_cost':COST,
        'base_v20_49':base_metrics,
        'entry_expansion':{
            'lane':'NORMAL_B4P_DOUBLE_WASHOUT_M5',
            'rule':'PRIMARY + NORMAL_RED + B4P; Day1 pm_above_ratio <= 0.40; Day2 M5 return from Day2 open <= -1%; enter M6 open; default exit T1 close.',
            'new_nonoverlap_trades':int(len(extra)),
            'new_trade_metrics_before_exit_overlay':met(extra,'t1_close_ret'),
            'new_trade_names':extra[['date','code','name']].astype({'code':str}).to_dict('records')
        },
        'exit_overlays':[
            {'lane':'B4P_PANIC_CAPITULATION_OPEN','rule':'If T1 minute-1 close / original entry - 1 > +8%, sell T1 minute-2 open; otherwise T1 close.'},
            {'lane':'PANIC_3_LIQ_RECLAIM45','rule':'If T1 minute-5 close / T1 open - 1 > +0.5%, sell T1 minute-6 open; otherwise T1 close.'},
            {'lane':'NORMAL_B4P_LIQUID_M3','rule':'If T1 minute-3 close / T1 open - 1 > +5%, sell T1 minute-4 open; otherwise T1 close.'},
            {'lane':'POS_4P_VOLUME_RECLAIM35','rule':'If T1 minute-3 close / original entry - 1 < -2%, sell T1 minute-4 open; otherwise T1 close.'},
            {'lane':'CORE_RED_4P_REPAIR45','rule':'If T1 minute-3 close / T1 open - 1 < -1.5%, sell T1 minute-4 open; otherwise T1 close.'}
        ],
        'final_52':final_metrics,
        'monthly':monthly,
        'stress':stress_summary,
        'naive_sequential_full_capital_compound':theoretical_compound,
        'post52_strict_search':'No further branch candidate passed the frozen no-degradation gates with at least 3 Jan-Jun events across 2 months and at least 2 Jul-Sep observations.',
        'warning':'Jul-Sep is pseudo-OOS, not true unseen. Simple return sum and naive full-capital compounding are not account returns because signals and holding periods overlap.'
    }
    (OUT/'v21_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({
        'base49':payload['base_v20_49'],
        'final52':payload['final_52'],
        'stress':payload['stress'],
        'naive_compound':payload['naive_sequential_full_capital_compound']
    },ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()