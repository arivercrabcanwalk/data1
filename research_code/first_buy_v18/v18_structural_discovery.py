from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parent
RAW=Path.home()/'Documents/stock_data_windows_mirror'
OUT=ROOT/'v18_results'
OUT.mkdir(exist_ok=True)
COST=.0052
TRAIN_END=pd.Timestamp('2026-06-30')
PSEUDO_START=pd.Timestamp('2026-07-01')
DATA_END=pd.Timestamp('2026-09-30')

def metrics(df, col='t1_ret'):
    r=pd.to_numeric(df[col],errors='coerce').dropna()
    if len(r)==0:
        return {'n':0,'win_rate':None,'mean':None,'median':None,'bad3':None,'worst':None,'sum':0.0}
    return {
        'n':int(len(r)),
        'win_rate':float((r>0).mean()),
        'mean':float(r.mean()),
        'median':float(r.median()),
        'bad3':float((r<=-.03).mean()),
        'worst':float(r.min()),
        'sum':float(r.sum()),
    }

def split_metrics(df, col='t1_ret'):
    train=df[df.date<=TRAIN_END]
    pseudo=df[(df.date>=PSEUDO_START)&(df.date<=DATA_END)]
    return {'train_jan_jun':metrics(train,col),'pseudo_oos_jul_sep':metrics(pseudo,col),'all_2026':metrics(df,col)}

def add_minute_row(rows,r,s,lane,signal_i,why):
    if signal_i+1>=len(s):
        return
    ent=s.iloc[signal_i+1]
    px=float(ent.open)
    if not np.isfinite(px) or px<=0:
        return
    # A locked limit-up next-minute open is not assumed fillable.
    d2_upper=np.floor(float(r.close)*110+.5+1e-8)/100
    if px>=d2_upper-.005 and float(ent.low)>=d2_upper-.005:
        return
    ret=(float(r.exit_t1_close)/px-1-COST) if pd.notna(r.exit_t1_close) and not bool(r.exit_t1_reset) else np.nan
    rows.append({
        'event_id':r.event_id,'date':pd.Timestamp(r.date),'d2_date':pd.Timestamp(r.d2_date),
        'code':str(r.code),'name':r.name,'event_type':r.event_type,'generation':int(r.generation),
        'prior_streak':int(r.prior_streak),'branch':r.branch,'d1_quality':r.d1_quality,'bucket':r.bucket,
        'ret':float(r.ret),'gap':float(r.gap),'range':float(r.range),'vol_prev':float(r.vol_prev) if pd.notna(r.vol_prev) else np.nan,
        'amount':float(r.amount) if pd.notna(r.amount) else np.nan,'d2_gap':float(r.d2_gap),
        'lane':lane,'signal_time':str(s.iloc[signal_i].time),'entry_time':str(ent.time),
        'entry_price':px,'t1_ret':ret,'why':why
    })

def build_extra_minute_primitives(events):
    e=events[
        (events.date>=pd.Timestamp('2026-01-01'))&(events.date<=DATA_END)&
        events.d2_date.notna()&events.exit_t1_close.notna()&
        ~events.d2_price_reset.fillna(False)&events.d2_gap.abs().le(.12)
    ].copy()
    rows=[]
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        for day,g in e.groupby(e.d2_date.dt.strftime('%Y-%m-%d')):
            files=[str(p) for p in (RAW/'1min'/('date='+day)).glob('*.parquet')]
            if not files:
                continue
            codes=g.code.astype(str).unique().tolist()
            m=con.execute(
                'SELECT code,time,open,high,low,close,volume,amount FROM read_parquet(?,union_by_name=true) '
                'WHERE code IN (SELECT unnest(?)) ORDER BY code,time',
                [files,codes]
            ).fetchdf()
            for r in g.itertuples(index=False):
                s=m[m.code.astype(str)==str(r.code)].copy().reset_index(drop=True)
                if len(s)<7:
                    continue
                s['cvwap']=s.amount.cumsum()/s.volume.cumsum().replace(0,np.nan)
                o=float(s.iloc[0].open)
                c1=float(s.iloc[0].close)
                # Research-only: red/green auction followed by a >=3% first-minute flush, then immediate second-minute rebound.
                if r.branch=='NORMAL_RED' and r.bucket=='B3' and c1/o-1<=-.03:
                    c2=float(s.iloc[1].close)
                    if c2>c1:
                        add_minute_row(rows,r,s,'B3_NORMAL_FLUSH_REBOUND2',1,'m1_flush_le_-3pct_then_m2_close_rebounds')
                # Small-n temporal hypothesis: 4+ board panic, deep gap, 5-minute >=3% impulse above cumulative VWAP.
                if r.branch=='PANIC' and r.bucket=='B4P' and r.d2_gap<=-.03:
                    c5=float(s.iloc[4].close)
                    cv=float(s.iloc[4].cvwap) if pd.notna(s.iloc[4].cvwap) else np.nan
                    if c5/o-1>=.03 and np.isfinite(cv) and c5>=cv:
                        add_minute_row(rows,r,s,'B4P_PANIC_DEEP_IMPULSE5',4,'deep_gap_5m_gain_ge_3pct_and_close_ge_cvwap')
    return pd.DataFrame(rows)

def existing_lane(actions, mask, lane, tier):
    z=actions[mask].copy()
    z['lane']=lane
    z['tier']=tier
    return z

def deep_absorbed_open(events,daily):
    z=events[
        (events.date>=pd.Timestamp('2026-01-01'))&(events.date<=DATA_END)&
        (events.branch=='POSITIVE_BREAK')&(events.bucket=='B4P')&
        (events.d1_quality=='ABSORBED')&
        (events.vol_prev>=1.3)&events.d2_gap.between(-.08,-.03)&
        events.d2_open.notna()&events.exit_t1_close.notna()&
        ~events.d2_price_reset.fillna(False)&~events.exit_t1_reset.fillna(False)
    ].copy()
    z['entry_price']=z.d2_open.astype(float)
    z['t1_ret']=z.exit_t1_close/z.entry_price-1-COST
    z['signal_time']='09:25'
    z['entry_time']='OPEN'
    z['lane']='POS_4P_DEEP_ABSORBED_OPEN'
    z['tier']='RESEARCH_ONLY'
    z['why']='absorbed_B4P_positive_break_volprev_ge_1.3_d2_gap_-8_to_-3'
    # Exit study is explicitly research-only. It uses T1 close information only to decide whether to hold to T2.
    t1=daily[['code','date','ret','close_vwap']].rename(
        columns={'date':'f2_date','ret':'t1_day_ret','close_vwap':'t1_close_vwap'})
    z['f2_date']=pd.to_datetime(z.f2_date)
    z=z.merge(t1,on=['code','f2_date'],how='left')
    hold=(z.t1_day_ret>=.04)&(z.t1_close_vwap>=.995)&~z.exit_t2_reset.fillna(False)
    z['adaptive_hold_t2']=hold
    z['adaptive_ret']=np.where(hold,z.exit_t2_close/z.entry_price-1-COST,z.t1_ret)
    return z

def neighborhood(actions):
    rows=[]
    for vol in [1.3,1.5,1.7,2.0]:
        for gap_cap in [.03,.05,.07]:
            for cut in ['09:35','09:40']:
                z=actions[
                    (actions.branch=='POSITIVE_BREAK')&(actions.bucket=='B4P')&
                    (actions.action=='RECLAIM1')&(actions.gap<=gap_cap)&
                    (actions.vol_prev>=vol)&(actions.signal_time<=cut)
                ].copy()
                sm=split_metrics(z)
                rows.append({'vol_prev_min':vol,'day1_gap_cap':gap_cap,'signal_cut':cut,**sm})
    return rows

def main():
    E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
    E['date']=pd.to_datetime(E.date); E['d2_date']=pd.to_datetime(E.d2_date)
    A=pd.read_parquet(ROOT/'v17_results/action_candidates.parquet').copy()
    A['date']=pd.to_datetime(A.date); A['d2_date']=pd.to_datetime(A.d2_date)
    D=pd.read_parquet(ROOT/'v17_data/daily.parquet').copy(); D['date']=pd.to_datetime(D.date)
    C17=pd.read_parquet(ROOT/'v17_results/candidate_v17_trades.parquet').copy(); C17['date']=pd.to_datetime(C17.date)

    extra=build_extra_minute_primitives(E)
    extra.to_parquet(OUT/'extra_minute_primitives.parquet',index=False)

    lanes=[]
    # Frozen V17 CORE is carried forward unchanged.
    core=C17[C17.tier=='CORE'].copy()
    core['tier']='CORE_FROZEN'
    lanes.append(core)

    # V18 temporally-confirmed experimental lane: broad thresholds, early causal reclaim.
    pos=existing_lane(
        A,
        (A.branch=='POSITIVE_BREAK')&(A.bucket=='B4P')&(A.action=='RECLAIM1')&
        (A.gap<=.05)&(A.vol_prev>=1.5)&(A.signal_time<='09:35'),
        'POS_4P_VOLUME_RECLAIM35','TEMPORALLY_CONFIRMED_EXPERIMENTAL'
    )
    lanes.append(pos)

    # Small-n structural hypotheses: retain for future unseen validation, not promotion.
    b3=existing_lane(
        A,
        (A.branch=='NORMAL_RED')&(A.bucket=='B3')&(A.action=='IMPULSE3')&(A.ret<=-.03),
        'B3_NORMAL_DEEP_IMPULSE3','SMALL_N_RESEARCH'
    )
    lanes.append(b3)
    if len(extra):
        p5=extra[extra.lane=='B4P_PANIC_DEEP_IMPULSE5'].copy(); p5['tier']='SMALL_N_RESEARCH'; lanes.append(p5)
        fr=extra[extra.lane=='B3_NORMAL_FLUSH_REBOUND2'].copy(); fr['tier']='SMALL_N_RESEARCH'; lanes.append(fr)

    research_open=deep_absorbed_open(E,D)
    if len(research_open):
        lanes.append(research_open)

    L=pd.concat(lanes,ignore_index=True,sort=False)
    L=L[(L.date>=pd.Timestamp('2026-01-01'))&(L.date<=DATA_END)].copy()
    L.to_parquet(OUT/'lane_trades.parquet',index=False)

    lane_summ={}
    for (tier,lane),g in L.groupby(['tier','lane']):
        lane_summ[f'{tier}|{lane}']=split_metrics(g)
        if lane=='POS_4P_DEEP_ABSORBED_OPEN':
            lane_summ[f'{tier}|{lane}']['adaptive_exit']=split_metrics(g,'adaptive_ret')

    # Unique frozen stack includes only V17 CORE + the one V18 temporally-confirmed experimental lane.
    stack=L[L.tier.isin(['CORE_FROZEN','TEMPORALLY_CONFIRMED_EXPERIMENTAL'])].copy()
    tier_rank={'CORE_FROZEN':0,'TEMPORALLY_CONFIRMED_EXPERIMENTAL':1}
    stack['_rank']=stack.tier.map(tier_rank)
    stack=stack.sort_values(['event_id','_rank','signal_time']).drop_duplicates('event_id',keep='first').drop(columns='_rank')
    stack.to_parquet(OUT/'frozen_stack_trades.parquet',index=False)

    # User casebook remains a biased diagnostic set; it never participates in lane selection.
    case=pd.read_csv(ROOT/'v17_results/user_casebook_v17.csv',dtype={'code':str})
    case['date']=pd.to_datetime(case.date)
    labels=L[['event_id','tier','lane','t1_ret']].copy()
    labels=labels.sort_values(['event_id','tier','lane']).groupby('event_id',as_index=False).agg(
        v18_tiers=('tier',lambda x:';'.join(sorted(set(x)))),
        v18_lanes=('lane',lambda x:';'.join(sorted(set(x)))),
        v18_lane_returns=('t1_ret',lambda x:';'.join('' if pd.isna(v) else f'{float(v):+.4f}' for v in x))
    )
    case=case.merge(labels,on='event_id',how='left')
    case['v18_any_lane']=case.v18_lanes.notna()
    case['v18_promotable_research']=case.v18_tiers.fillna('').str.contains('CORE_FROZEN|TEMPORALLY_CONFIRMED_EXPERIMENTAL')
    case.to_csv(OUT/'casebook_coverage_v18.csv',index=False)

    v17_flag=case.candidate_trade.fillna(False).astype(bool)
    union_flag=v17_flag|case.v18_any_lane
    new_v18_flag=case.v18_any_lane&~v17_flag
    unexplained=case[~union_flag].copy()
    unexplained_groups=(unexplained.groupby(['branch','bucket']).size().sort_values(ascending=False))
    case_summary={
        'casebook_events':int(len(case)),
        'v17_candidate_events':int(v17_flag.sum()),
        'v18_any_research_lane_events':int(case.v18_any_lane.sum()),
        'v17_or_v18_any_research_union_events':int(union_flag.sum()),
        'new_events_explained_by_v18_structures':int(new_v18_flag.sum()),
        'v18_core_or_temporally_confirmed_events':int(case.v18_promotable_research.sum()),
        'unexplained_events':int((~union_flag).sum()),
        'unexplained_by_branch_bucket':{f'{a}|{b}':int(v) for (a,b),v in unexplained_groups.items()},
        'warning':'The user-nominated casebook is biased and is used only for coverage diagnostics, never rule selection.'
    }
    (OUT/'casebook_summary_v18.json').write_text(json.dumps(case_summary,ensure_ascii=False,indent=2))

    summary={
        'status':'V18_RESEARCH_FREEZE_NOT_PRODUCTION',
        'data_through':'2026-09-30',
        'split_note':'Jul-Sep is pseudo-OOS temporal confirmation only because those months were visible during research. True unseen validation begins on the next newly ingested trading session.',
        'train_window':'2026-01-01..2026-06-30',
        'pseudo_oos_window':'2026-07-01..2026-09-30',
        'round_trip_cost':COST,
        'price_reset_guard':{'threshold_abs_gap':.12,'canonical_rows_flagged':int(D.price_reset.sum())},
        'promotion_policy':{
            'CORE_FROZEN':'V17 core carried unchanged; no threshold retuning in V18.',
            'TEMPORALLY_CONFIRMED_EXPERIMENTAL':'Requires at least 5 Jan-Jun trades and 3 Jul-Sep trades with positive mean in both windows; still not production.',
            'SMALL_N_RESEARCH':'Interesting structure but sample count is below promotion threshold.',
            'RESEARCH_ONLY':'Hypothesis or exit study; not eligible for buy recommendation.'
        },
        'lane_metrics':lane_summ,
        'frozen_stack':split_metrics(stack),
        'pos_reclaim_neighborhood':neighborhood(A),
        'casebook':case_summary,
        'frozen_for_next_unseen':{
            'buy_lanes':['CORE_RED_4P_REPAIR45','PANIC_3_LIQ_RECLAIM45','POS_4P_VOLUME_RECLAIM35'],
            'observation_only':['B3_NORMAL_DEEP_IMPULSE3','B4P_PANIC_DEEP_IMPULSE5','B3_NORMAL_FLUSH_REBOUND2','POS_4P_DEEP_ABSORBED_OPEN'],
            'no_retroactive_changes_after_next_session_seen':True
        }
    }
    (OUT/'v18_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({
        'status':summary['status'],
        'price_reset_guard':summary['price_reset_guard'],
        'frozen_stack':summary['frozen_stack'],
        'casebook':case_summary,
        'lane_metrics':lane_summ
    },ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    main()