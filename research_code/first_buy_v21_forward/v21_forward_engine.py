# -*- coding: utf-8 -*-
"""V21 strict chronological out-of-sample paper-replay monitor.

The decision function never reads T1 outcomes or future minutes to decide Day2 entry.
Historical replay is a QA diagnostic, NOT forward performance.
Only data received after the V21 freeze may enter the forward ledger.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os
from pathlib import Path
import zoneinfo
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RAW=Path.home()/'Documents/stock_data_windows_mirror'
OUT=ROOT/'v21_forward_results'
OUT.mkdir(exist_ok=True)
HISTORY_END='2026-09-30'
START_DAY1='2026-10-08'
COST=.0052
TZ=zoneinfo.ZoneInfo('Asia/Shanghai')
LANES=[
 'B4P_PANIC_CAPITULATION_OPEN','B4P_PANIC_DEEP_IMPULSE5',
 'NORMAL_B4P_LIQUID_M3','NORMAL_B4P_DOUBLE_WASHOUT_M5',
 'CORE_RED_4P_REPAIR45','PANIC_3_LIQ_RECLAIM45','POS_4P_VOLUME_RECLAIM35'
]
CONFIG={
 'model':'V21_FIRST_BUY_FROZEN_PROSPECTIVE',
 'baseline_git_commit':'1f32304',
 'historical_data_end':HISTORY_END,
 'first_new_day1':START_DAY1,
 'timezone':'Asia/Shanghai',
 'roundtrip_cost':COST,
 'universe':'A-share main board, PRIMARY_3PLUS and RECYCLE_2PLUS rearmed to >=3 boards; exactly-2-board R2 observations never bought',
 'rule_priority':'Earliest causally executable Day2 signal wins; same stock/event one buy',
 'expansion_scope':'LM3 and W5 only PRIMARY_3PLUS; frozen R4/P3/PV may include RECYCLE_2PLUS with >=3 boards',
 'normal_b4p_lm3':{'d1_amount_min':1.6e9,'d1_close_loc_max':.50,'m3_abs_min':.01,'m3_up_max':.03},
 'normal_b4p_w5':{'d1_pm_above_ratio_max':.4,'d2_m5_ret_max':-.01},
 'normal_b4p_r4':{'repaired_full_minutes':5,'last_signal_time':'09:45'},
 'panic_b3':{'d1_amount_min':1e9,'last_signal_time':'09:45'},
 'positive_b4p':{'d1_gap_max':.05,'d1_vol_prev_min':1.5,'last_signal_time':'09:35'},
 'panic_b4p_open':{'d1_close_vwap_max':.965},
 'panic_b4p_m5':{'d2_gap_max':-.03,'m5_return_min':.03},
 'T1_exits':{
   'B4P_PANIC_CAPITULATION_OPEN':'if first minute profit >8%, M2 open, else close',
   'PANIC_3_LIQ_RECLAIM45':'if 5-minute rise vs T1 open >0.5%, M6 open, else close',
   'NORMAL_B4P_LIQUID_M3':'if 3-minute rise vs T1 open >5%, M4 open, else close',
   'POS_4P_VOLUME_RECLAIM35':'if 3-minute price vs entry <-2%, M4 open, else close',
   'CORE_RED_4P_REPAIR45':'if 3-minute fall vs T1 open <-1.5%, M4 open, else close',
   'otherwise':'T1 closing auction proxy'
 },
 'execution_note':'Only a paper model. A bar open/close is a quote/price proxy, not guaranteed fill.',
 'retrospective_asof_note':'Signals first generated after the price event are classified as backfilled OOS, never contemporaneous alerts.'
}

def now(): return dt.datetime.now(TZ)
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(path): return sha_bytes(Path(path).read_bytes())
def dump_atomic(path, payload):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def spec_hash():
    return sha_bytes(json.dumps(CONFIG,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))

def register():
    path=OUT/'protocol_frozen.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['spec_hash']==spec_hash(),'PROTOCOL HASH MISMATCH'
        print('ALREADY_FROZEN',old['registered_at'],old['spec_hash'])
        return
    files=['V21_METHOD.md','v21_freeze.py','v21_results/v21_summary.json','v21_results/v21_52_trade_stack.parquet']
    payload={
        'registered_at':now().isoformat(),
        'spec_hash':spec_hash(),
        'engine_sha256':sha_file(__file__),
        'historical_sources_sha256':{f:sha_file(ROOT/f) for f in files},
        'config':CONFIG,
        'scope':'Prospective monitoring from Day1 Oct 8 close onward. No retrospective re-tuning.'
    }
    dump_atomic(path,payload)
    print('FROZEN',payload['registered_at'],payload['spec_hash'])

def require_frozen():
    path=OUT/'protocol_frozen.json'
    if not path.exists(): raise RuntimeError('Unregistered model. Use --init first.')
    obj=json.loads(path.read_text())
    if obj['spec_hash']!=spec_hash(): raise RuntimeError('FROZEN CONFIG CHANGED - version bump required')
    if obj['engine_sha256']!=sha_file(__file__): raise RuntimeError('FROZEN ENGINE CHANGED - version bump required')
    for f,h in obj['historical_sources_sha256'].items():
        if sha_file(ROOT/f)!=h: raise RuntimeError(f'FROZEN HISTORICAL FILE CHANGED: {f}')
    return obj

def trading_day_partitions():
    p=RAW/'daily/tdx_parquet'
    return sorted(str(x.name[5:]) for x in p.glob('date=*') if (x/'daily_ohlcv.parquet').exists())

def read_minutes(day,codes,con=None):
    # date partition files are not assumed to be complete if fewer than 240 bars for an individual stock.
    d=RAW/'1min'/('date='+str(day))
    files=[str(p) for p in d.glob('*.parquet')]
    if not files or not codes: return {}
    own=con is None
    if own: con=duckdb.connect()
    try:
        codes=[str(c).zfill(6) for c in codes]
        sql='SELECT code,time,open,high,low,close,volume,amount FROM read_parquet(?,union_by_name=true) WHERE code IN (SELECT unnest(?)) ORDER BY code,time'
        m=con.execute(sql,[files,codes]).fetchdf()
    finally:
        if own:con.close()
    if m.empty:return {}
    m.code=m.code.astype(str).str.zfill(6)
    return {c:z.reset_index(drop=True) for c,z in m.groupby('code')}

def minute_complete(s):
    if s is None or len(s)!=240:return False
    labels=[f'{9+(30+i)//60:02d}:{(30+i)%60:02d}' for i in range(1,121)]
    labels += [f'{13+i//60:02d}:{i%60:02d}' for i in range(1,121)]
    return bool(s.time.astype(str).tolist()==labels and
      not s[['open','high','low','close','volume']].isna().any().any() and
      (pd.to_numeric(s.volume,errors='coerce')>=0).all())

def calc_cvwap(s):
    amt=pd.to_numeric(s.amount,errors='coerce')
    vol=pd.to_numeric(s.volume,errors='coerce')
    if amt.isna().any() or (amt<=0).any() or vol.isna().any() or (vol<0).any(): return None
    if (vol.cumsum()<=0).any(): return None
    return (amt.cumsum()/vol.cumsum()).to_numpy()

def pm_above_ratio(s):
    if not minute_complete(s):return None
    cv=calc_cvwap(s)
    if cv is None:return None
    # Replicate V20 09:31~11:30, 13:01~15:00; 13:00 label is not present in the source.
    v=pd.to_numeric(s.close,errors='coerce').to_numpy()/cv-1
    mask=s.time.astype(str).ge('13:00').to_numpy()
    return float(np.mean(v[mask]>=0)) if mask.sum() else None

def val(r,key,default=np.nan):
    v=r.get(key,default)
    try:return float(v) if pd.notna(v) else float(default)
    except (TypeError,ValueError):return float(default)

def is_primary(r):
    return str(r.get('event_type')) in ('PRIMARY_3PLUS','RECYCLE_2PLUS') and val(r,'prior_streak')>=3

def choose_entry(r,s,d1_pm=None):
    """Choose a single entry solely from information available by the entry bar's open."""
    if not is_primary(r):return None,'NON_PRIMARY'
    if not minute_complete(s):return None,'D2_MINUTES_INCOMPLETE'
    if bool(r.get('d2_price_reset',False)):return None,'D2_PRICE_BASIS_RESET'
    d1close=val(r,'close'); op=float(s.iloc[0].open)
    if d1close<=0 or op<=0:return None,'INVALID_OPEN'
    gap=op/d1close-1
    if abs(gap)>.12:return None,'D2_GAP_BASIS_RESET'
    br=str(r.get('branch')); bucket=str(r.get('bucket'))
    cv=calc_cvwap(s)
    candidates=[]
    def add(lane,i,signal_bar,reason):
        if i>=len(s):return
        px=float(s.iloc[i].open)
        upper=math.floor(d1close*110+.5+1e-8)/100
        # A stock locked limit-up at the execution bar is not assumed buyable.
        if not np.isfinite(px) or px<=0:return
        if px>=upper-.005 and float(s.iloc[i].low)>=upper-.005:return
        candidates.append({
         'event_id':str(r.event_id),'code':str(r.code),'name':str(r['name']),
         'day1':str(pd.Timestamp(r.date).date()),
         'lane':lane,'signal_time':signal_bar,
         'entry_clock':('09:30' if i==0 else str(s.iloc[i-1].time)),
         'entry_bar_label':str(s.iloc[i].time),
         'entry_price':px,'entry_idx':i,'reason':reason,
         'd2_gap':gap,'d1_amount':val(r,'amount'),
         'd1_height':int(val(r,'prior_streak')),
         'd1_branch':br
        })
    if br=='PANIC' and bucket=='B4P':
        if val(r,'close_vwap')<=.965:
            add('B4P_PANIC_CAPITULATION_OPEN',0,'09:25','D1 close/VWAP<=.965')
        elif (cv is not None and gap<=-.03 and
              float(s.iloc[4].close)/op-1>=.03 and
              float(s.iloc[4].close)>=float(cv[4])):
            add('B4P_PANIC_DEEP_IMPULSE5',5,str(s.iloc[4].time),'D2 gap<=-3%; m5 +3% and >= VWAP')
    elif br=='PANIC' and bucket=='B3' and val(r,'amount')>=1e9 and cv is not None:
        for i in range(min(240,len(s)-1)):
            if str(s.iloc[i].time)>'09:45':break
            if float(s.iloc[i].close)>=max(d1close,op,float(cv[i])):
                add('PANIC_3_LIQ_RECLAIM45',i+1,str(s.iloc[i].time),'first price+VWAP reclaim')
                break
    elif br=='POSITIVE_BREAK' and bucket=='B4P' and val(r,'gap')<=.05 and val(r,'vol_prev')>=1.5 and cv is not None:
        for i in range(min(240,len(s)-1)):
            if str(s.iloc[i].time)>'09:35':break
            if float(s.iloc[i].close)>=max(d1close,op,float(cv[i])):
                add('POS_4P_VOLUME_RECLAIM35',i+1,str(s.iloc[i].time),'early price+VWAP reclaim')
                break
    elif br=='NORMAL_RED' and bucket=='B4P':
        m3=float(s.iloc[2].close)/op-1
        if (r.event_type=='PRIMARY_3PLUS' and val(r,'amount')>=1.6e9 and val(r,'close_loc')<=.50 and
            abs(m3)>=.01 and m3<=.03):
            add('NORMAL_B4P_LIQUID_M3',3,str(s.iloc[2].time),'liquid M3 displacement')
        if r.event_type=='PRIMARY_3PLUS' and d1_pm is not None and d1_pm<=.40 and float(s.iloc[4].close)/op-1<=-.01:
            add('NORMAL_B4P_DOUBLE_WASHOUT_M5',5,str(s.iloc[4].time),'weak D1 PM and D2 M5 double washout')
        if cv is not None:
            good=(s.close.to_numpy()>=np.maximum(d1close,op))&(s.close.to_numpy()>=cv)
            for i in range(4,min(240,len(s)-1)):
                if str(s.iloc[i].time)>'09:45':break
                if bool(np.all(good[i-4:i+1])):
                    add('CORE_RED_4P_REPAIR45',i+1,str(s.iloc[i].time),'five consecutive price+VWAP repairs')
                    break
    if not candidates:return None,'NO_FROZEN_SIGNAL_OR_UNFILLABLE'
    chosen=sorted(candidates,key=lambda z:(z['entry_idx'],LANES.index(z['lane'])))[0]
    chosen['other_lanes_later']=[x['lane'] for x in candidates if x is not chosen]
    return chosen,'TRIGGERED'

def choose_exit(trade,s):
    if not minute_complete(s):return None,'T1_MINUTES_INCOMPLETE'
    entry=float(trade['entry_price']); op=float(s.iloc[0].open)
    if entry<=0:return None,'BAD_ENTRY'
    lane=trade['lane']; sell_idx=None; rule='T1_CLOSE'
    if lane=='B4P_PANIC_CAPITULATION_OPEN' and float(s.iloc[0].close)/entry-1>.08:
        sell_idx=1;rule='CAP_M1_PROFIT_GT_8PCT'
    elif lane=='PANIC_3_LIQ_RECLAIM45' and float(s.iloc[4].close)/op-1>.005:
        sell_idx=5;rule='P3_T1_M5_UP_GT_0.5PCT'
    elif lane=='NORMAL_B4P_LIQUID_M3' and float(s.iloc[2].close)/op-1>.05:
        sell_idx=3;rule='LM3_T1_M3_UP_GT_5PCT'
    elif lane=='POS_4P_VOLUME_RECLAIM35' and float(s.iloc[2].close)/entry-1<-.02:
        sell_idx=3;rule='POS_T1_M3_ENTRY_LOSS_LT_-2PCT'
    elif lane=='CORE_RED_4P_REPAIR45' and float(s.iloc[2].close)/op-1<-.015:
        sell_idx=3;rule='R4_T1_M3_OPEN_FALL_LT_-1.5PCT'
    bar=s.iloc[sell_idx] if sell_idx is not None else s.iloc[-1]
    exitpx=float(bar.open if sell_idx is not None else bar.close)
    if exitpx<=0:return None,'BAD_EXIT'
    # Minute bar labels time at end of minute, the bar open is one minute earlier.
    clock=(str(s.iloc[sell_idx-1].time) if sell_idx is not None else '15:00')
    v=exitpx/entry-1-COST
    result=dict(trade,exit_clock=clock,exit_price=exitpx,exit_rule=rule,
                final_ret=v,positive=(v>0),
                bar_single_price=bool(bar.high==bar.low),exit_bar_volume=float(bar.volume))
    return result,'COMPLETE'

def hydrate_daily_history():
    """Build causal features using at most the latest fully available raw daily date.

    Only price-volume fields are needed for V21. Future theme labels are never loaded.
    Historical canon is read-only and is not re-saved.
    """
    h=pd.read_parquet(ROOT/'v17_data/daily.parquet',
        columns=['code','name','board','date','open','high','low','close','volume','amount'])
    h['date']=pd.to_datetime(h.date).dt.strftime('%Y-%m-%d')
    if h.date.max()!=HISTORY_END:raise RuntimeError('HISTORICAL BASELINE DATE CHANGED')
    t=now()
    incoming=[]
    for day in trading_day_partitions():
        if day<=HISTORY_END: continue
        # The current session must not be scored before its normal full-day close.
        if day>t.date().isoformat():continue
        if day==t.date().isoformat() and (t.hour,t.minute)<(15,15):continue
        file=RAW/'daily/tdx_parquet'/('date='+day)/'daily_ohlcv.parquet'
        q=pd.read_parquet(file)
        q=q[q.code.astype(str).str.fullmatch(r'(600|601|603|605|000|001|002|003)[0-9]{3}')].copy()
        if len(q)<2500: raise RuntimeError(f'Daily source incomplete {day}: {len(q)} main-board rows')
        if q[['open','high','low','close','volume']].isna().any().any():
            raise RuntimeError(f'Daily source contains missing critical prices {day}')
        q=q[['code','name','board','date','open','high','low','close','volume','amount']]
        q['date']=pd.to_datetime(q.date).dt.strftime('%Y-%m-%d')
        if not q.date.eq(day).all():raise RuntimeError(f'DAILY PARTITION DATE MISMATCH {day}')
        incoming.append(q)
    if not incoming:return None,[]
    new=pd.concat(incoming,ignore_index=True)
    d=pd.concat([h,new],ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    if d.duplicated(['code','date']).any():raise RuntimeError('DUPLICATE DAILY OBSERVATIONS')
    for c in ['open','high','low','close']:
        d[c]=pd.to_numeric(d[c],errors='coerce').round(2)
    for c in ['volume','amount']:d[c]=pd.to_numeric(d[c],errors='coerce')
    # Recalculate only backward-looking indicators. No full-period fitted labels/weights.
    ratio=d.amount/d.volume.replace(0,np.nan)/((d.high+d.low+d.close)/3)
    rawfactor=pd.Series(np.select([ratio.between(.7,1.3),ratio.between(70,130)],[1.,100.],default=np.nan),index=d.index)
    d['volume_factor']=rawfactor.groupby(d.code).ffill()
    d['shares']=d.volume*d.volume_factor
    d['vwap']=d.amount/d.shares.replace(0,np.nan)
    d['price_valid']=(d.open.gt(0)&d.high.gt(0)&d.low.gt(0)&d.close.gt(0)&d.volume.gt(0)&
       d.high.ge(d[['open','close','low']].max(axis=1))&
       d.low.le(d[['open','close','high']].min(axis=1)))
    d['eligible']=d.price_valid&~d.name.fillna('').str.contains('ST|退',case=False)
    cal=sorted(d.date.unique()); idx={day:i for i,day in enumerate(cal)}
    d['di']=d.date.map(idx)
    g=d.groupby('code',sort=False)
    d['prev_close']=g.close.shift();d['prev_date']=g.date.shift()
    d['adjacent']=(d.di-d.prev_date.map(idx)==1)
    d['upper']=np.floor(d.prev_close*110+.5+1e-8)/100
    d['lower']=np.floor(d.prev_close*90+.5+1e-8)/100
    d['ret']=d.close/d.prev_close-1;d['gap']=d.open/d.prev_close-1
    d['range']=(d.high-d.low)/d.prev_close
    d['price_reset']=d.adjacent & d.gap.abs().gt(.12)
    d['lu']=(d.close-d.upper).abs().lt(.005)&d.eligible&d.adjacent&~d.price_reset
    d['streak']=d.groupby('code').lu.transform(lambda x:x.groupby((~x).cumsum()).cumsum()).astype(int)
    d['prior_streak']=d.groupby('code').streak.shift().fillna(0).astype(int)
    d['vol_prev']=d.shares/d.groupby('code').shares.shift()
    d['close_loc']=(d.close-d.low)/(d.high-d.low).replace(0,np.nan)
    d['close_vwap']=d.close/d.vwap.replace(0,np.nan)
    d['touch_lu']=d.high.ge(d.upper-.005)&d.eligible&d.adjacent&~d.price_reset
    d['open_lu']=d.open.ge(d.upper-.005)&d.eligible&d.adjacent&~d.price_reset
    d['d1_quality']=np.select([
      d.close_loc.ge(.65)&d.close_vwap.ge(.99),
      d.close_loc.ge(.35)&d.close_vwap.ge(.97)],['ABSORBED','MIXED'],default='WEAK')
    g=d.groupby('code',sort=False)
    for k in range(1,6):
        d[f'f{k}_date']=g.date.shift(-k)
        d[f'f{k}_open']=g.open.shift(-k)
        d[f'f{k}_close']=g.close.shift(-k)
        d[f'f{k}_reset']=g.price_reset.shift(-k).fillna(False)
    d['date']=pd.to_datetime(d.date)
    for k in range(1,6):d[f'f{k}_date']=pd.to_datetime(d[f'f{k}_date'])
    # These unrelated columns are not used by lifecycle_events / execution policy.
    return d,sorted(new.date.unique())

def read_for_events(events,date_col):
    result={}
    with duckdb.connect() as con:
        for day,g in events.groupby(events[date_col].dt.strftime('%Y-%m-%d')):
            if pd.isna(day):continue
            result[day]=read_minutes(day,g.code.astype(str).tolist(),con)
    return result

def historical_audit():
    """Rebuild Day2 signal decisions without consulting the historical 52-trade index."""
    E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
    E['date']=pd.to_datetime(E.date);E['d2_date']=pd.to_datetime(E.d2_date)
    E['f2_date']=pd.to_datetime(E.f2_date)
    E=E[(E.date>='2026-01-01')&(E.date<=HISTORY_END)&(E.prior_streak>=3)].copy()
    relevant=E[((E.branch=='PANIC') & (E.bucket.isin(['B3','B4P'])))|
               ((E.branch.isin(['NORMAL_RED','POSITIVE_BREAK']))&(E.bucket=='B4P'))].copy()
    d1_mins=read_for_events(relevant,'date')
    d2_mins=read_for_events(relevant.dropna(subset=['d2_date']),'d2_date')
    t1_mins=read_for_events(relevant.dropna(subset=['f2_date']),'f2_date')
    decisions=[];completed=[]
    for _,r in relevant.iterrows():
        day=str(r.date.date()); nxt=str(r.d2_date.date()) if pd.notna(r.d2_date) else None
        prev=d1_mins.get(day,{}).get(str(r.code))
        pm=pm_above_ratio(prev) if prev is not None else None
        s=d2_mins.get(nxt,{}).get(str(r.code)) if nxt else None
        entry,reason=choose_entry(r,s,pm)
        if entry is None: continue
        entry.update(d2_date=nxt,pm_above_ratio=pm)
        decisions.append(entry)
        dt1=str(r.f2_date.date()) if pd.notna(r.f2_date) else None
        s1=t1_mins.get(dt1,{}).get(str(r.code)) if dt1 else None
        if s1 is None:continue
        ex,status=choose_exit(entry,s1)
        if ex is not None:completed.append(ex)
    df=pd.DataFrame(decisions)
    vf=pd.DataFrame(completed)
    baseline=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet')
    pinned=set(baseline.event_id.astype(str))
    found=set(df.event_id.astype(str)) if len(df) else set()
    misses=sorted(pinned-found); extra=sorted(found-pinned)
    mismatches=[]
    if len(df):
        z=df.merge(baseline[['event_id','lane','entry_price','final_ret']].rename(columns={
            'lane':'pinned_lane','entry_price':'pinned_entry','final_ret':'pinned_ret'
        }),on='event_id',how='inner')
        for _,r in z.iterrows():
            if r.lane!=r.pinned_lane or abs(r.entry_price-r.pinned_entry)>.011:
                mismatches.append(dict(event_id=r.event_id,code=r.code,name=r['name'],
                                       chosen_lane=r.lane,pinned_lane=r.pinned_lane,
                                       chosen_entry=r.entry_price,pinned_entry=r.pinned_entry))
    report={'kind':'HISTORICAL_CAUSAL_REPLAY_QA_NOT_FORWARD','observed_at':now().isoformat(),
            'event_candidates':len(relevant),'signal_candidates':len(df),'pinned_trades':len(baseline),
            'missing_original':misses,'newly_detected':extra,'differing_entries':mismatches,
            'completed_replay':len(vf)}
    if len(vf):
        report['replay_win_rate']=float((vf.final_ret>0).mean())
        report['replay_mean']=float(vf.final_ret.mean())
    dump_atomic(OUT/'historical_causal_replay_QA.json',report)
    if len(df):df.to_csv(OUT/'historical_causal_signals_QA.csv',index=False,encoding='utf-8-sig')
    if len(vf):vf.to_csv(OUT/'historical_causal_completed_QA.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(report,ensure_ascii=False,indent=2))

def frozen_snapshot(kind,day,records):
    """Append-only per-session fact; never rewrite a snapshot with later information."""
    d=OUT/'snapshots'/kind;d.mkdir(parents=True,exist_ok=True)
    file=d/(day+'.json')
    signature=sha_bytes(json.dumps(records,ensure_ascii=False,sort_keys=True,default=str).encode())
    if file.exists():
        old=json.loads(file.read_text())
        if old['signature']!=signature:
            raise RuntimeError(f'IMMUTABLE SNAPSHOT REVISION DETECTED: {file}')
        return old['created_at']
    obj={'snapshot_stage':kind,'session':day,'created_at':now().isoformat(),
         'signature':signature,'records':records}
    dump_atomic(file,obj)
    return obj['created_at']

def scan_new():
    freeze=require_frozen()
    newest=trading_day_partitions()[-1]
    d,newdays=hydrate_daily_history()
    report={
      'protocol_hash':freeze['spec_hash'],
      'evaluated_at':now().isoformat(),
      'historical_last_session':HISTORY_END,
      'raw_daily_latest_any':newest,
      'new_daily_sessions':newdays,
      'day1_primary_events':0,'open_day2_events':0,'triggered_entries':0,
      'closed_T1_trades':0,'win_rate':None,'mean_ret':None,
      'status':'WAITING_FOR_POST_FREEZE_DATA',
      'note':'No future trade outcome may be imputed. Only complete T1 trading sessions count.'
    }
    if not newdays:
        dump_atomic(OUT/'forward_status.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return
    # Build only known-to-date event observations without viewing later days than present.
    from v17_structural_lab import lifecycle_events
    E=lifecycle_events(d)
    if E.empty:raise RuntimeError('NO LIFECYCLE EVENTS')
    E['date']=pd.to_datetime(E.date); E['d2_date']=pd.to_datetime(E.d2_date)
    E['f2_date']=pd.to_datetime(E.f2_date)
    new=E[(E.date>=START_DAY1)&E.prior_streak.ge(3)].copy()
    report['day1_primary_events']=len(new)
    if new.empty:
        report['status']='SCANNED_NO_PRIMARY_DAY1'
        for day in newdays:frozen_snapshot('day1',day,[])
        dump_atomic(OUT/'forward_status.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2));return

    # The entire Day1 universe is registered before knowing whether any trade works.
    relevant=new.copy()
    d1_mins=read_for_events(relevant,'date')
    d2_mins=read_for_events(relevant.dropna(subset=['d2_date']),'d2_date')
    t1_mins=read_for_events(relevant.dropna(subset=['f2_date']),'f2_date')
    by_day={}
    decisions=[];closed=[];watch=[]
    for _,r in relevant.sort_values(['date','code']).iterrows():
        day=str(r.date.date()); event=str(r.event_id);code=str(r.code).zfill(6)
        y=d1_mins.get(day,{}).get(code)
        pm=pm_above_ratio(y) if y is not None else None
        ystatus='COMPLETE' if minute_complete(y) else 'MISSING_FULL_DAY1_MINUTES'
        d1_record={
            'event_id':event,'day1':day,'code':code,'name':str(r['name']),
            'prior_streak':int(r.prior_streak),'branch':str(r.branch),
            'bucket':str(r.bucket),'d1_close':float(r.close),
            'd1_amount':float(r.amount) if pd.notna(r.amount) else None,
            'd1_close_vwap':float(r.close_vwap) if pd.notna(r.close_vwap) else None,
            'd1_vol_prev':float(r.vol_prev) if pd.notna(r.vol_prev) else None,
            'pm_above_ratio':pm,'day1_minute_status':ystatus,
        }
        by_day.setdefault(day,[]).append(d1_record)
        nxt=str(r.d2_date.date()) if pd.notna(r.d2_date) else None
        case={**d1_record,'d2_date':nxt,'status':'DAY1_REGISTERED_AWAIT_D2','lane':None,
              'entry_price':None,'exit_price':None,'net_ret':None}
        if nxt is None:watch.append(case);continue
        minute=d2_mins.get(nxt,{}).get(code)
        if not minute_complete(minute):
            case['status']='WAITING_COMPLETE_D2_MINUTES'
            watch.append(case);continue
        report['open_day2_events']+=1
        if abs(float(minute.iloc[0].open)-float(r.d2_open))>.011:
            case['status']='RAW_DAILY_D2_OPEN_MISMATCH'
            watch.append(case);continue
        # PM is essential for the W5 branch and its relative order against R4.
        if r.branch=='NORMAL_RED' and r.bucket=='B4P' and r.event_type=='PRIMARY_3PLUS' and pm is None:
            case['status']='DAY1_PM_MISSING_CANNOT_ORDER_LANES'
            watch.append(case);continue
        chosen,reason=choose_entry(r,minute,pm)
        if chosen is None:
            case['status']='D2_PASS';case['reason']=reason
            watch.append(case);continue
        case.update(chosen)
        case['status']='SIMULATED_D2_ENTRY_PENDING_T1'
        decisions.append(case.copy())
        t1=str(r.f2_date.date()) if pd.notna(r.f2_date) else None
        if t1 is None:watch.append(case);continue
        sm=t1_mins.get(t1,{}).get(code)
        if not minute_complete(sm):
            case['status']='SIMULATED_D2_ENTRY_WAITING_T1_MINUTES'
            case['t1_date']=t1
            watch.append(case);continue
        # Verify D2 original open and T1 close match raw daily source (price-basis guard).
        if abs(float(minute.iloc[0].open)-float(r.d2_open))>.011 or abs(float(sm.iloc[-1].close)-float(r.exit_t1_close))>.011:
            case['status']='RAW_DAILY_MINUTE_PRICE_MISMATCH'
            watch.append(case);continue
        outcome,status=choose_exit(chosen,sm)
        if outcome is None:
            case['status']=status;watch.append(case);continue
        case.update(outcome)
        t1_lower=math.floor(float(r.d2_close)*90+.5+1e-8)/100
        case['exit_locked_limit_down']=bool(outcome['bar_single_price'] and
            abs(float(outcome['exit_price'])-t1_lower)<.005)
        case['t1_date']=t1
        case['status']='CLOSED_SIMULATED_OOS'
        closed.append(case.copy())
        watch.append(case)

    # Register all Day1 decisions in their own records, before any completed outcomes.
    for day in newdays:
        frozen_snapshot('day1',day,by_day.get(day,[]))
    # Append-only D2 casebook; contains only entrance-time facts, not future exits.
    by_d2={}
    for rec in watch:
        nxt=rec.get('d2_date')
        if nxt is None or rec.get('entry_price') is None:continue
        clean={k:rec.get(k) for k in [
          'event_id','day1','code','name','d2_date','lane','signal_time',
          'entry_clock','entry_price','d2_gap','reason'
        ]}
        by_d2.setdefault(nxt,[]).append(clean)
    for day,recs in sorted(by_d2.items()):
        frozen_snapshot('day2_entrance',day,recs)
    by_t1={}
    for rec in closed:
        if rec.get('t1_date'):
            by_t1.setdefault(rec['t1_date'],[]).append({k:rec.get(k) for k in [
                'event_id','code','name','lane','day1','d2_date','t1_date','entry_price',
                'exit_clock','exit_price','exit_rule','final_ret','bar_single_price'
            ]})
    for day,recs in sorted(by_t1.items()):
        frozen_snapshot('T1_result',day,recs)
    for name,items in [('day1_watchlist',watch),('day2_triggered',decisions),('T1_closed',closed)]:
        dump_atomic(OUT/(name+'.json'),items)

    for rec in closed:
        # If Day1 was registered only AFTER the Day2 entry, it is OOS data but not timely precommitted paper trading.
        path=OUT/'snapshots'/'day1'/(rec['day1']+'.json')
        first=json.loads(path.read_text())['created_at']
        declared=dt.datetime.fromisoformat(first)
        d2_auction=dt.datetime.fromisoformat(rec['d2_date']+'T09:25:00+08:00')
        rec['registered_before_day2']=bool(declared<d2_auction)
        rec['forward_evidence_type']='DAY1_PRE_REGISTERED' if rec['registered_before_day2'] else 'DELAYED_BACKFILL'
    if len(closed):
        c=pd.DataFrame(closed)
        def m(q):
            r=pd.to_numeric(q.final_ret,errors='coerce').dropna()
            if not len(r):
                return {'n':0,'win_rate':None,'mean_ret':None,'worst_ret':None}
            return {'n':int(len(r)),'win_rate':float((r>0).mean()),
                    'mean_ret':float(r.mean()),'median_ret':float(r.median()),
                    'worst_ret':float(r.min()),'sum_return_units':float(r.sum())}
        registered=c[c.registered_before_day2]
        delayed=c[~c.registered_before_day2]
        report['preregistered_metrics']=m(registered)
        report['delayed_backfill_metrics']=m(delayed)
        report['all_postfreeze_descriptive_metrics']=m(c)
        # Headline performance always excludes trades registered after their Day2 open.
        if len(registered):
            report['win_rate']=float((registered.final_ret>0).mean())
            report['mean_ret']=float(registered.final_ret.mean())
        report['preregistered_day1_trade_count']=len(registered)
        report['delayed_backfill_count']=len(delayed)
    report['triggered_entries']=len(decisions)
    report['closed_T1_trades']=len(closed)
    report['sell_limit_down_lock_risk_count']=sum(bool(x.get('exit_locked_limit_down',False)) for x in closed)
    report['pending_entries']=int(len(decisions)-len(closed))
    report['status']='UPDATED'
    dump_atomic(OUT/'forward_status.json',report)
    if len(closed):dump_atomic(OUT/'T1_closed.json',closed)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--init',action='store_true')
    a.add_argument('--scan',action='store_true')
    a.add_argument('--historical-audit',action='store_true')
    args=a.parse_args()
    if sum([args.init,args.scan,args.historical_audit])!=1:
        a.error('Choose exactly one of --init/--scan/--historical-audit')
    if args.init:register()
    elif args.scan:scan_new()
    else:historical_audit()

if __name__=='__main__':
    main()