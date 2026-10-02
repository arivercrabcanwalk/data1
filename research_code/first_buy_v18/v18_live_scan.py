from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v18_results'
E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
E['date']=pd.to_datetime(E.date)

def plan(r):
    if r.bucket=='R2':
        return ('OBSERVE_ONLY','LIFECYCLE','R2 remains in the 30-day lifecycle but is not a V18 buy lane.')
    if r.branch=='NORMAL_RED' and r.bucket=='B4P':
        return ('WATCH_CORE_RED_4P_REPAIR45','CORE_FROZEN',
                'On D2 require five consecutive repaired completed minutes by 09:45; fill only next-minute open.')
    if r.branch=='PANIC' and r.bucket=='B3' and pd.notna(r.amount) and r.amount>=1e9:
        return ('WATCH_CORE_PANIC_3_LIQ_RECLAIM45','CORE_FROZEN',
                'On D2 require reclaim of Day1 close, D2 open and cumulative VWAP by 09:45; fill next-minute open.')
    if r.branch=='POSITIVE_BREAK' and r.bucket=='B4P' and r.gap<=.05 and pd.notna(r.vol_prev) and r.vol_prev>=1.5:
        return ('WATCH_EXP_POS_4P_VOLUME_RECLAIM35','TEMPORALLY_CONFIRMED_EXPERIMENTAL',
                'On D2 require causal reclaim by 09:35; fill next-minute open. Research status, not production.')
    if r.branch=='NORMAL_RED' and r.bucket=='B3' and r.ret<=-.03:
        return ('WATCH_SMALL_N_B3_NORMAL_DEEP_IMPULSE3','SMALL_N_RESEARCH',
                'Only if D2 gaps <=-3% and the first three completed minutes gain >=2% from open while above cumulative VWAP; fill minute-4 open.')
    if r.branch=='PANIC' and r.bucket=='B4P':
        return ('WATCH_SMALL_N_B4P_PANIC_DEEP_IMPULSE5','SMALL_N_RESEARCH',
                'Only if D2 gaps <=-3% and five completed minutes gain >=3% from open while above cumulative VWAP; fill minute-6 open.')
    if r.branch=='NORMAL_RED' and r.bucket=='B3':
        return ('WATCH_SMALL_N_FLUSH_REBOUND2','SMALL_N_RESEARCH',
                'Secondary research only: if D2 first completed minute flushes >=3% from D2 open and minute 2 closes above minute 1, fill minute-3 open.')
    if r.branch=='POSITIVE_BREAK' and r.bucket=='B4P' and r.d1_quality=='ABSORBED' and pd.notna(r.vol_prev) and r.vol_prev>=1.3:
        return ('WATCH_RESEARCH_POS_4P_DEEP_ABSORBED_OPEN','RESEARCH_ONLY',
                'Observe only if D2 auction gap is between -8% and -3%; this open-entry/exit study is not eligible for promotion yet.')
    return ('PASS_V18','NO_LANE','Current Day1 structure does not map to a frozen V18 lane.')

latest=E.date.max()
rows=[]
for r in E[E.date==latest].itertuples(index=False):
    p,tier,why=plan(r)
    rows.append({
        'date':str(r.date.date()),'code':str(r.code),'name':r.name,'event_id':r.event_id,
        'event_type':r.event_type,'generation':int(r.generation),'prior_streak':int(r.prior_streak),
        'branch':r.branch,'d1_quality':r.d1_quality,'bucket':r.bucket,
        'day1_ret':float(r.ret),'day1_gap':float(r.gap),'vol_prev':None if pd.isna(r.vol_prev) else float(r.vol_prev),
        'amount':None if pd.isna(r.amount) else float(r.amount),'plan':p,'tier':tier,'why':why
    })
(OUT/'latest_day1_plan_v18.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
freeze={
    'status':'FROZEN_BEFORE_NEXT_UNSEEN_SESSION',
    'data_through':'2026-09-30',
    'latest_day1_date':str(latest.date()),
    'no_retroactive_rule_change_after_next_session_seen':True,
    'core_and_temporally_confirmed_buy_lanes':[
        'CORE_RED_4P_REPAIR45','PANIC_3_LIQ_RECLAIM45','POS_4P_VOLUME_RECLAIM35'
    ],
    'observation_only_lanes':[
        'B3_NORMAL_DEEP_IMPULSE3','B4P_PANIC_DEEP_IMPULSE5',
        'B3_NORMAL_FLUSH_REBOUND2','POS_4P_DEEP_ABSORBED_OPEN'
    ],
    'latest_watchlist':rows
}
(OUT/'next_unseen_freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
print(json.dumps(rows,ensure_ascii=False,indent=2))