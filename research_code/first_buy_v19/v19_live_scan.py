from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v19_results'; OUT.mkdir(exist_ok=True)
E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
E['date']=pd.to_datetime(E.date)

OPEN_CVWAP_MAX=.965

def plan(r):
    # Lifecycle-only branch still stays out of buy execution.
    if r.bucket=='R2':
        return ('OBSERVE_ONLY','LIFECYCLE','R2 remains observation-only.')

    # V19 optimization of B4P PANIC timing.
    if r.branch=='PANIC' and r.bucket=='B4P':
        if pd.notna(r.close_vwap) and float(r.close_vwap)<=OPEN_CVWAP_MAX:
            return (
                'WATCH_V19_OPEN_CAPITULATION',
                'TEMPORALLY_CONFIRMED_EXPERIMENTAL',
                'Day1 close/VWAP <= 0.965. V19 experimental rule enters at Day2 opening match; no 1m/3m/5m confirmation.'
            )
        return (
            'WATCH_V19_M5_PANIC_REPAIR',
            'SMALL_N_RESEARCH',
            'If Day2 gap <= -3%, require first 5 completed minutes gain >=3% from open and minute-5 close >= cumulative VWAP; enter minute-6 open.'
        )

    # Frozen V18 branches.
    if r.branch=='NORMAL_RED' and r.bucket=='B4P':
        return ('WATCH_CORE_RED_4P_REPAIR45','CORE_FROZEN',
                'Day2 require five consecutive repaired completed minutes by 09:45; enter next-minute open.')
    if r.branch=='PANIC' and r.bucket=='B3' and pd.notna(r.amount) and r.amount>=1e9:
        return ('WATCH_CORE_PANIC_3_LIQ_RECLAIM45','CORE_FROZEN',
                'Day2 reclaim Day1 close + Day2 open + cumulative VWAP by 09:45; enter next-minute open.')
    if r.branch=='POSITIVE_BREAK' and r.bucket=='B4P' and r.gap<=.05 and pd.notna(r.vol_prev) and r.vol_prev>=1.5:
        return ('WATCH_EXP_POS_4P_VOLUME_RECLAIM35','TEMPORALLY_CONFIRMED_EXPERIMENTAL',
                'Day2 causal reclaim by 09:35; enter next-minute open.')
    if r.branch=='NORMAL_RED' and r.bucket=='B3' and r.ret<=-.03:
        return ('WATCH_SMALL_N_B3_NORMAL_DEEP_IMPULSE3','SMALL_N_RESEARCH',
                'D2 gap <=-3%; first 3 completed minutes gain >=2% from open and satisfy VWAP strength; enter minute-4 open.')
    if r.branch=='NORMAL_RED' and r.bucket=='B3':
        return ('WATCH_SMALL_N_FLUSH_REBOUND2','SMALL_N_RESEARCH',
                'Research only: first minute flush >=3% from D2 open, minute 2 closes above minute 1; enter minute-3 open.')
    if r.branch=='POSITIVE_BREAK' and r.bucket=='B4P' and r.d1_quality=='ABSORBED' and pd.notna(r.vol_prev) and r.vol_prev>=1.3:
        return ('WATCH_RESEARCH_POS_4P_DEEP_ABSORBED_OPEN','RESEARCH_ONLY',
                'Observe only if Day2 gap is -8% to -3%; not production.')
    return ('PASS_V19','NO_LANE','No frozen V19 lane.')

latest=E.date.max()
rows=[]
for r in E[E.date==latest].itertuples(index=False):
    p,tier,why=plan(r)
    rows.append({
        'date':str(r.date.date()),'code':str(r.code),'name':r.name,'event_id':r.event_id,
        'prior_streak':int(r.prior_streak),'branch':r.branch,'bucket':r.bucket,'d1_quality':r.d1_quality,
        'day1_ret':float(r.ret),'day1_gap':float(r.gap),
        'day1_close_vwap':None if pd.isna(r.close_vwap) else float(r.close_vwap),
        'day1_vol_prev':None if pd.isna(r.vol_prev) else float(r.vol_prev),
        'day1_amount':None if pd.isna(r.amount) else float(r.amount),
        'plan':p,'tier':tier,'why':why
    })

(OUT/'latest_day1_plan_v19.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
freeze={
    'status':'V19_FROZEN_BEFORE_NEXT_UNSEEN_SESSION',
    'data_through':'2026-09-30',
    'no_retroactive_changes_after_next_session_seen':True,
    'b4p_panic_timing_tree':[
        {
            'priority':1,
            'lane':'B4P_PANIC_CAPITULATION_OPEN',
            'tier':'TEMPORALLY_CONFIRMED_EXPERIMENTAL',
            'condition':'Day1 branch=PANIC, bucket=B4P, Day1 close/VWAP <= 0.965',
            'entry':'Day2 opening match',
        },
        {
            'priority':2,
            'lane':'B4P_PANIC_DEEP_IMPULSE5',
            'tier':'SMALL_N_RESEARCH',
            'condition':'not open lane; Day2 gap <= -3%; minute-5 close/open >= +3%; minute-5 close >= cumulative VWAP',
            'entry':'minute-6 open',
        }
    ],
    'latest_watchlist':rows
}
(OUT/'next_unseen_freeze_v19.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2))