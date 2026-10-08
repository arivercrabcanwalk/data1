from pathlib import Path
import pandas as pd, numpy as np, json
ROOT=Path('/Users/xxx/Documents/duanlong_primary_upgrade_20260924')
P=pd.read_parquet(ROOT/'v20_results/yao_scored.parquet').copy();P['date']=pd.to_datetime(P.date)
B=pd.read_parquet(ROOT/'v21_results/v21_52_trade_stack.parquet').copy();B['date']=pd.to_datetime(B.date)
ids=set(B.event_id.astype(str))
P=P[(~P.event_id.astype(str).isin(ids))&(P.event_type=='PRIMARY_3PLUS')&(P.bucket.isin(['B3','B4P']))].copy()
entries={'OPEN':'ret_t1_d2_open_proxy','M1':'ret_t1_entry_after_m1','M3':'ret_t1_entry_after_m3','M5':'ret_t1_entry_after_m5'}
pre=['ret','gap','range','vol_prev','amount','close_loc','close_vwap_dist','am_above_ratio','pm_above_ratio','pm_minus_am','pm_mean_vwap_dist','last60_above_ratio','last30_ret','late_giveback','d1_auction_gap','d2_auction_gap_final','auction_gap_upgrade','auction_vol_log_ratio','market_height','rank','followers','height_edge']
av={'OPEN':pre,'M1':pre+['m1_ret','m1_above_cvwap'],'M3':pre+['m1_ret','m1_above_cvwap','m3_ret','m3_above_cvwap','m3_above_count'],'M5':pre+['m1_ret','m1_above_cvwap','m3_ret','m3_above_cvwap','m3_above_count','m5_ret','m5_above_cvwap','m5_above_count','m1_to_m5_low']}
def met(q,col):
    r=pd.to_numeric(q[col],errors='coerce').dropna()
    if not len(r):return None
    z=q.loc[r.index]
    return {'n':len(r),'win':float((r>0).mean()),'mean':float(r.mean()),'worst':float(r.min()),'sum':float(r.sum()),'months':z.date.dt.to_period('M').nunique()}
def bm(q):
    r=q.final_ret.dropna();return {'n':len(r),'win':float((r>0).mean()),'mean':float(r.mean()),'worst':float(r.min()),'sum':float(r.sum())}
base={'tr':bm(B[B.date<='2026-06-30']),'te':bm(B[B.date>='2026-07-01']),'all':bm(B)}
def rounder(v):
    if not np.isfinite(v):return None
    a=abs(v)
    if a>=1e9:return round(v/1e8)*1e8
    if a>=1e8:return round(v/1e7)*1e7
    if a>=10:return round(v,0)
    if a>=1:return round(v,1)
    return round(v,3)
def conds(tr,fs):
    out=[]
    for f in fs:
        if f not in tr:continue
        x=pd.to_numeric(tr[f],errors='coerce').dropna()
        if len(x)<6 or x.nunique()<3:continue
        for q in [.2,.33,.4,.5,.6,.67,.8]:
            t=rounder(float(x.quantile(q)))
            if t is None:continue
            out += [(f'{f}>={t}',f,'ge',t),(f'{f}<={t}',f,'le',t)]
    return list({x[0]:x for x in out}.values())
def mask(z,c):
    _,f,op,t=c;x=pd.to_numeric(z[f],errors='coerce');return x>=t if op=='ge' else x<=t
def combo(base_df,cand,col):
    x=pd.concat([base_df[['date','final_ret']].rename(columns={'final_ret':'r'}),cand[['date',col]].rename(columns={col:'r'})])
    r=x.r.dropna();return {'n':len(r),'win':float((r>0).mean()),'mean':float(r.mean()),'worst':float(r.min()),'sum':float(r.sum())}
out=[]
for (br,bucket),G in P.groupby(['branch','bucket']):
    tr=G[G.date<='2026-06-30'];te=G[G.date>='2026-07-01']
    if len(tr)<6:continue
    for e,col in entries.items():
        cs=conds(tr,[f for f in av[e] if f in G])
        singles=[]
        for c in cs:
            a=met(tr[mask(tr,c)],col)
            if a and a['n']>=3 and a['months']>=2 and a['mean']>0 and a['worst']>=base['all']['worst']:
                singles.append((a['mean'],c))
        singles=sorted(singles,reverse=True)[:60]
        specs=[(c,) for _,c in singles]
        for i,(_,c1) in enumerate(singles):
            for _,c2 in singles[i+1:]:
                if c1[1]==c2[1]:continue
                specs.append((c1,c2))
        seen=set()
        for sp in specs:
            k=tuple(sorted(c[0] for c in sp))
            if k in seen:continue
            seen.add(k)
            mt=pd.Series(True,index=tr.index);me=pd.Series(True,index=te.index)
            for c in sp:mt &= mask(tr,c);me &= mask(te,c)
            a=met(tr[mt],col);b=met(te[me],col)
            if not a or not b or a['n']<3 or a['months']<2 or b['n']<2:continue
            if a['mean']<base['tr']['mean'] or a['win']<base['tr']['win'] or a['worst']<base['all']['worst']:continue
            if b['mean']<base['te']['mean'] or b['win']<base['te']['win'] or b['worst']<base['all']['worst']:continue
            ct=combo(B[B.date<='2026-06-30'],tr[mt],col);ce=combo(B[B.date>='2026-07-01'],te[me],col);ca=combo(B,pd.concat([tr[mt],te[me]]),col)
            ok=True
            for lab,m in [('tr',ct),('te',ce),('all',ca)]:
                bb=base[lab]
                if m['mean']+1e-12<bb['mean'] or m['win']+1e-12<bb['win'] or m['worst']+1e-12<bb['worst']:ok=False
            if ok:
                z=pd.concat([tr[mt],te[me]])
                out.append({'branch':br,'bucket':bucket,'entry':e,'entry_col':col,'rules':[x[0] for x in sp],'train':a,'pseudo':b,'combined_all':ca,'event_ids':list(z.event_id.astype(str))})
u={}
for x in out:
    k=(x['entry'],tuple(sorted(x['event_ids'])))
    if k not in u or len(x['rules'])<len(u[k]['rules']):u[k]=x
out=sorted(u.values(),key=lambda x:(x['train']['mean'],x['train']['n']),reverse=True)
(ROOT/'v21_results/post52_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str))
print('base',base)
print('count',len(out))
for i,x in enumerate(out[:50],1):
    print(i,x['branch'],x['bucket'],x['entry'],x['rules'],x['train'],x['pseudo'],x['combined_all'])