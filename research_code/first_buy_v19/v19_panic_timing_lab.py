from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import duckdb

ROOT=Path(__file__).resolve().parent
RAW=Path.home()/'Documents/stock_data_windows_mirror'
OUT=ROOT/'v19_results'; OUT.mkdir(exist_ok=True)
COST=.0052
TRAIN_END=pd.Timestamp('2026-06-30')
TEST_START=pd.Timestamp('2026-07-01')
DATA_END=pd.Timestamp('2026-09-30')

def met(z, col='ret'):
    r=pd.to_numeric(z[col],errors='coerce').dropna()
    if len(r)==0:
        return {'n':0,'win':None,'mean':None,'median':None,'bad3':None,'bad5':None,'worst':None,'sum':0.0}
    return {
        'n':int(len(r)),
        'win':float((r>0).mean()),
        'mean':float(r.mean()),
        'median':float(r.median()),
        'bad3':float((r<=-.03).mean()),
        'bad5':float((r<=-.05).mean()),
        'worst':float(r.min()),
        'sum':float(r.sum()),
    }

def split(z,col='ret'):
    return {
        'train_jan_jun':met(z[z.date<=TRAIN_END],col),
        'pseudo_oos_jul_sep':met(z[(z.date>=TEST_START)&(z.date<=DATA_END)],col),
        'all_2026':met(z,col)
    }

def build_table():
    E=pd.read_parquet(ROOT/'v17_results/events.parquet').copy()
    E['date']=pd.to_datetime(E.date); E['d2_date']=pd.to_datetime(E.d2_date)
    E=E[
        (E.date>=pd.Timestamp('2026-01-01'))&(E.date<=DATA_END)&
        (E.branch=='PANIC')&(E.bucket=='B4P')&
        E.d2_date.notna()&E.exit_t1_close.notna()&
        ~E.d2_price_reset.fillna(False)&~E.exit_t1_reset.fillna(False)&
        E.d2_gap.abs().le(.12)
    ].copy()
    rows=[]
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        for day,g in E.groupby(E.d2_date.dt.strftime('%Y-%m-%d')):
            files=[str(p) for p in (RAW/'1min'/('date='+day)).glob('*.parquet')]
            if not files:
                continue
            codes=g.code.astype(str).unique().tolist()
            m=con.execute(
                'SELECT code,time,open,high,low,close,volume,amount FROM read_parquet(?,union_by_name=true) '
                'WHERE code IN (SELECT unnest(?)) ORDER BY code,time',[files,codes]
            ).fetchdf()
            for r in g.itertuples(index=False):
                s=m[m.code.astype(str)==str(r.code)].copy().reset_index(drop=True)
                if len(s)<7:
                    continue
                for c in ['open','high','low','close','volume','amount']:
                    s[c]=pd.to_numeric(s[c],errors='coerce')
                s['cum_amount']=s.amount.cumsum()
                s['cum_volume']=s.volume.cumsum()
                s['cvwap']=s.cum_amount/s.cum_volume.replace(0,np.nan)
                o=float(s.iloc[0].open)
                if not np.isfinite(o) or o<=0:
                    continue
                row={
                    'event_id':r.event_id,'date':pd.Timestamp(r.date),'d2_date':pd.Timestamp(r.d2_date),
                    'code':str(r.code),'name':r.name,'prior_streak':int(r.prior_streak),
                    'd1_ret':float(r.ret),'d1_gap':float(r.gap),'d1_range':float(r.range),
                    'd1_vol_prev':float(r.vol_prev) if pd.notna(r.vol_prev) else np.nan,
                    'd1_amount':float(r.amount) if pd.notna(r.amount) else np.nan,
                    'd1_close_loc':float(r.close_loc) if pd.notna(r.close_loc) else np.nan,
                    'd1_close_vwap':float(r.close_vwap) if pd.notna(r.close_vwap) else np.nan,
                    'd2_gap':float(r.d2_gap),'d2_open':o,'t1_close':float(r.exit_t1_close),
                    'm1_volume':float(s.iloc[0].volume) if pd.notna(s.iloc[0].volume) else np.nan,
                    'm1_amount':float(s.iloc[0].amount) if pd.notna(s.iloc[0].amount) else np.nan,
                }
                row['open_ret']=row['t1_close']/o-1-COST
                for k in [1,2,3,5]:
                    i=k-1
                    x=s.iloc[i]
                    row[f'm{k}_time']=str(x.time)
                    row[f'm{k}_ret_from_open']=float(x.close)/o-1
                    row[f'm{k}_low_from_open']=float(s.iloc[:k].low.min())/o-1
                    row[f'm{k}_high_from_open']=float(s.iloc[:k].high.max())/o-1
                    row[f'm{k}_close_vs_d1']=float(x.close)/float(r.close)-1
                    cv=float(x.cvwap) if pd.notna(x.cvwap) else np.nan
                    row[f'm{k}_cvwap']=cv
                    row[f'm{k}_above_vwap']=bool(np.isfinite(cv) and float(x.close)>=cv)
                    row[f'm{k}_close_vs_vwap']=float(x.close)/cv-1 if np.isfinite(cv) and cv>0 else np.nan
                    if k < len(s):
                        px=float(s.iloc[k].open)
                        row[f'entry_after_m{k}']=px
                        row[f'ret_after_m{k}']=row['t1_close']/px-1-COST if np.isfinite(px) and px>0 else np.nan
                        row[f'entry_time_after_m{k}']=str(s.iloc[k].time)
                # simple rebound diagnostics
                row['m2_rebound_from_m1_close']=float(s.iloc[1].close)/float(s.iloc[0].close)-1
                row['m3_rebound_from_m1_low']=float(s.iloc[2].close)/float(s.iloc[0].low)-1
                row['minute_amount_ok']=bool(pd.notna(s.iloc[:5].amount).all())
                rows.append(row)
    T=pd.DataFrame(rows).sort_values(['date','code']).reset_index(drop=True)
    T.to_parquet(OUT/'b4p_panic_timing_table.parquet',index=False)
    return T

def eval_fixed(T):
    rules={}
    def add(name,mask,col):
        z=T[mask & T[col].notna()].copy(); z['ret']=z[col]
        rules[name]={'entry_col':col,'metrics':split(z),'events':z[['date','code','name','d2_gap','d1_amount','d1_vol_prev','ret']].to_dict('records')}
    # Auction/open-only families.
    add('OPEN_GAP_GE_0',T.d2_gap>=0,'open_ret')
    add('OPEN_GAP_M3_P3',T.d2_gap.between(-.03,.03),'open_ret')
    add('OPEN_GAP_GE_M3_AMT_GE1B',(T.d2_gap>=-.03)&(T.d1_amount>=1e9),'open_ret')
    add('OPEN_GAP_GE_M3_VOL_GE1',(T.d2_gap>=-.03)&(T.d1_vol_prev>=1),'open_ret')
    # One-minute confirmation.
    add('M1_GREEN_VWAP',(T.m1_ret_from_open>=0)&T.m1_above_vwap,'ret_after_m1')
    add('M1_PLUS1_VWAP',(T.m1_ret_from_open>=.01)&T.m1_above_vwap,'ret_after_m1')
    add('M1_PLUS2_VWAP',(T.m1_ret_from_open>=.02)&T.m1_above_vwap,'ret_after_m1')
    add('M1_DEEPGAP_PLUS1_VWAP',(T.d2_gap<=-.03)&(T.m1_ret_from_open>=.01)&T.m1_above_vwap,'ret_after_m1')
    # Three-minute confirmation.
    add('M3_PLUS1_VWAP',(T.m3_ret_from_open>=.01)&T.m3_above_vwap,'ret_after_m3')
    add('M3_PLUS2_VWAP',(T.m3_ret_from_open>=.02)&T.m3_above_vwap,'ret_after_m3')
    add('M3_DEEPGAP_PLUS2_VWAP',(T.d2_gap<=-.03)&(T.m3_ret_from_open>=.02)&T.m3_above_vwap,'ret_after_m3')
    add('M3_DEEPGAP_PLUS3_VWAP',(T.d2_gap<=-.03)&(T.m3_ret_from_open>=.03)&T.m3_above_vwap,'ret_after_m3')
    # Five-minute confirmation, including frozen V18 research rule.
    add('M5_DEEPGAP_PLUS2_VWAP',(T.d2_gap<=-.03)&(T.m5_ret_from_open>=.02)&T.m5_above_vwap,'ret_after_m5')
    add('M5_DEEPGAP_PLUS3_VWAP',(T.d2_gap<=-.03)&(T.m5_ret_from_open>=.03)&T.m5_above_vwap,'ret_after_m5')
    add('M5_DEEPGAP_PLUS4_VWAP',(T.d2_gap<=-.03)&(T.m5_ret_from_open>=.04)&T.m5_above_vwap,'ret_after_m5')
    return rules

def train_only_grid(T):
    # Low-complexity candidate family. Selection/ranking uses Jan-Jun only.
    # Jul-Sep is used only as a stability veto, not to rank candidates.
    tr=T[T.date<=TRAIN_END].copy()
    candidates=[]
    gap_bands=[
        ('ALL',pd.Series(True,index=tr.index),lambda z:pd.Series(True,index=z.index)),
        ('GAP_GE_M3',tr.d2_gap>=-.03,lambda z:z.d2_gap>=-.03),
        ('DEEP_GAP',tr.d2_gap<=-.03,lambda z:z.d2_gap<=-.03),
        ('GAP_M8_M3',tr.d2_gap.between(-.08,-.03),lambda z:z.d2_gap.between(-.08,-.03)),
    ]
    d1_filters=[
        ('NONE',pd.Series(True,index=tr.index),lambda z:pd.Series(True,index=z.index)),
        ('AMT_GE1B',tr.d1_amount>=1e9,lambda z:z.d1_amount>=1e9),
        ('VOL_GE1',tr.d1_vol_prev>=1,lambda z:z.d1_vol_prev>=1),
        ('VOL_GE15',tr.d1_vol_prev>=1.5,lambda z:z.d1_vol_prev>=1.5),
        ('CVWAP_GE097',tr.d1_close_vwap>=.97,lambda z:z.d1_close_vwap>=.97),
    ]
    action_specs=[
        ('OPEN','open_ret',[(None,None)]),
        ('M1','ret_after_m1',[(x,'vwap') for x in [0,.01,.02,.03]]),
        ('M3','ret_after_m3',[(x,'vwap') for x in [.01,.02,.03,.04]]),
        ('M5','ret_after_m5',[(x,'vwap') for x in [.02,.03,.04,.05]]),
    ]
    for gname,gmask,gfn in gap_bands:
        for dname,dmask,dfn in d1_filters:
            for act,col,params in action_specs:
                for threshold,kind in params:
                    m=gmask&dmask
                    if act!='OPEN':
                        m=m&(tr[f'{act.lower()}_ret_from_open']>=threshold)&tr[f'{act.lower()}_above_vwap']
                    q=tr[m&tr[col].notna()].copy()
                    r=q[col]
                    if len(r)<4 or q.date.dt.to_period('M').nunique()<3:
                        continue
                    win=float((r>0).mean()); mean=float(r.mean()); bad=float((r<=-.03).mean()); worst=float(r.min())
                    sd=float(r.std(ddof=1)) if len(r)>1 else 0
                    score=mean-.5*sd/math.sqrt(len(r))+.015*(win-.5)-.01*bad
                    candidates.append({
                        'gap':gname,'d1_filter':dname,'action':act,'threshold':threshold,'entry_col':col,
                        'train_n':len(r),'train_months':int(q.date.dt.to_period('M').nunique()),
                        'train_win':win,'train_mean':mean,'train_bad3':bad,'train_worst':worst,'score':score,
                        '_gfn':gfn,'_dfn':dfn
                    })
    candidates=sorted(candidates,key=lambda x:x['score'],reverse=True)
    audited=[]
    te=T[(T.date>=TEST_START)&(T.date<=DATA_END)].copy()
    for c in candidates[:80]:
        m=c['_gfn'](te)&c['_dfn'](te)
        if c['action']!='OPEN':
            a=c['action'].lower()
            m=m&(te[f'{a}_ret_from_open']>=c['threshold'])&te[f'{a}_above_vwap']
        q=te[m&te[c['entry_col']].notna()].copy(); q['ret']=q[c['entry_col']]
        test=met(q)
        # Stability veto: no promotion if test mean <=0, any <=-8% tail, or no later observation.
        stable=bool(test['n']>=1 and test['mean'] is not None and test['mean']>0 and test['worst']>-0.08)
        out={k:v for k,v in c.items() if not k.startswith('_')}
        out['pseudo_oos']=test; out['stability_veto_pass']=stable
        audited.append(out)
    return audited

def main():
    T=build_table()
    fixed=eval_fixed(T)
    grid=train_only_grid(T)
    payload={
        'status':'V19_PANIC_TIMING_RESEARCH',
        'data_through':'2026-09-30',
        'universe':'B4P PANIC Day1 only',
        'events':int(len(T)),
        'train_events':int((T.date<=TRAIN_END).sum()),
        'pseudo_oos_events':int((T.date>=TEST_START).sum()),
        'round_trip_cost':COST,
        'principle':'Test whether strong auction or 1m/3m confirmation can safely replace the frozen 5m confirmation. Train ranking uses Jan-Jun only; Jul-Sep is a veto, not an optimizer.',
        'fixed_rules':fixed,
        'train_only_grid_top80':grid,
    }
    (OUT/'panic_timing_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({
        'events':payload['events'],'train_events':payload['train_events'],'pseudo_oos_events':payload['pseudo_oos_events'],
        'fixed':{k:v['metrics'] for k,v in fixed.items()},
        'grid_top20':grid[:20]
    },ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    main()