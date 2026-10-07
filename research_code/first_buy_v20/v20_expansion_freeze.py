from pathlib import Path
import json, math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v20_results'
OUT.mkdir(exist_ok=True)
COST=.0052

LANE='NORMAL_B4P_LIQUID_M3'
AMOUNT_MIN=1.6e9
CLOSE_LOC_MAX=.50
M3_ABS_MIN=.01
M3_UPPER_MAX=.03

def met(df,col='trade_ret'):
    r=pd.to_numeric(df[col],errors='coerce').dropna()
    if len(r)==0:
        return {'n':0,'win_rate':None,'mean':None,'median':None,'worst':None,'bad3':None,'sum':0.0}
    return {'n':int(len(r)),'win_rate':float((r>0).mean()),'mean':float(r.mean()),
            'median':float(r.median()),'worst':float(r.min()),'bad3':float((r<=-.03).mean()),'sum':float(r.sum())}

def split(df,col='trade_ret'):
    return {'train_jan_jun':met(df[df.date<='2026-06-30'],col),
            'pseudo_oos_jul_sep':met(df[df.date>='2026-07-01'],col),
            'all_2026':met(df,col)}

def main():
    P=pd.read_parquet(OUT/'yao_scored.parquet').copy()
    P['date']=pd.to_datetime(P.date); P['d2_date']=pd.to_datetime(P.d2_date)
    BASE=pd.read_parquet(ROOT/'v19_results/v18_v19_research_stack.parquet').copy()
    BASE['date']=pd.to_datetime(BASE.date); BASE['trade_ret']=BASE['ret']
    base_ids=set(BASE.event_id.astype(str))

    pool=P[
        (~P.event_id.astype(str).isin(base_ids))&
        (P.event_type=='PRIMARY_3PLUS')&
        (P.branch=='NORMAL_RED')&(P.bucket=='B4P')&
        P.ret_t1_entry_after_m3.notna()&
        ~P.d2_price_reset.fillna(False)&~P.exit_t1_reset.fillna(False)
    ].copy()

    lane=pool[
        (pool.amount>=AMOUNT_MIN)&
        (pool.close_loc<=CLOSE_LOC_MAX)&
        (pool.m3_ret.abs()>=M3_ABS_MIN)&
        (pool.m3_ret<=M3_UPPER_MAX)
    ].copy()
    lane['lane']=LANE
    lane['tier']='V20_TEMPORALLY_CONFIRMED_EXPERIMENTAL'
    lane['entry_time']='M4_OPEN'
    lane['entry_price']=lane.entry_after_m3
    lane['trade_ret']=lane.ret_t1_entry_after_m3

    # Fillability guard: M4 entry cannot be assumed at a locked limit-up.
    lane['d2_upper']=np.floor(lane['close']*110+.5+1e-8)/100
    lane['entry_below_upper']=lane.entry_price < lane.d2_upper-.005
    lane=lane[lane.entry_below_upper].copy()
    lane.to_parquet(OUT/'v20_expansion_lane_trades.parquet',index=False)

    combined=pd.concat([
        BASE[['event_id','date','code','name','trade_ret']].assign(lane='V19_BASE'),
        lane[['event_id','date','code','name','trade_ret','lane']]
    ],ignore_index=True).sort_values(['date','event_id'])
    if combined.event_id.duplicated().any():
        raise RuntimeError('duplicate events in combined stack')
    combined.to_parquet(OUT/'v20_49_trade_stack.parquet',index=False)

    # Neighborhood audit.
    neighborhood=[]
    for amt in [1.5e9,1.6e9,1.7e9,1.8e9,2.0e9]:
        for loc in [.4,.5,.6]:
            for low in [.005,.01,.015]:
                for cap in [.025,.03,.035]:
                    z=pool[(pool.amount>=amt)&(pool.close_loc<=loc)&
                           (pool.m3_ret.abs()>=low)&(pool.m3_ret<=cap)].copy()
                    z['trade_ret']=z.ret_t1_entry_after_m3
                    neighborhood.append({'amount_min':amt,'close_loc_max':loc,'m3_abs_min':low,'m3_upper_max':cap,
                                         **split(z)})
    # Stress candidate.
    stress=lane.copy(); stress['double_cost_ret']=stress.trade_ret-COST
    ranked=lane.sort_values('trade_ret',ascending=False)
    stress_summary={'base':split(lane),
                    'double_cost_all':met(stress,'double_cost_ret'),
                    'drop_top1_all':met(ranked.iloc[1:]),
                    'drop_top2_all':met(ranked.iloc[2:])}

    payload={
        'status':'V20_EXPANSION_FROZEN_RESEARCH_NOT_PRODUCTION',
        'data_through':'2026-09-30',
        'objective':'Increase trade count without lowering historical average return, positive-return rate, or worst trade versus the frozen V19 41-trade stack.',
        'base_v19':split(BASE),
        'lane':{
            'name':LANE,
            'rule':{
                'day1':'PRIMARY, NORMAL_RED, B4P',
                'day1_amount_min':AMOUNT_MIN,
                'day1_close_location_max':CLOSE_LOC_MAX,
                'day2_after_three_complete_minutes_abs_move_min':M3_ABS_MIN,
                'day2_after_three_complete_minutes_move_max':M3_UPPER_MAX,
                'entry':'minute-4 open',
                'exit':'T1 close',
                'round_trip_cost':COST
            },
            'metrics':split(lane),
            'stress':stress_summary
        },
        'combined_49':split(combined),
        'neighborhood':neighborhood,
        'discipline':'Jul-Sep is pseudo-OOS only. V20 is frozen before any data after 2026-09-30 is used; no threshold may be changed after the next unseen session is observed.'
    }
    (OUT/'v20_expansion_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'lane':payload['lane']['metrics'],'combined_49':payload['combined_49'],
                      'stress':stress_summary},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()