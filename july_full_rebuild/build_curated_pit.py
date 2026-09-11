from __future__ import annotations
import json,re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CUR=HERE/'curated'; PIT=HERE/'pit_data'; PIT.mkdir(parents=True,exist_ok=True)
DATA=HERE.parent/'2026-07'

def norm_code(s):
    x=s.astype(str).str.replace(r'\.0$','',regex=True); y=x.str.extract(r'(\d{6})',expand=False); return y.fillna(x)

def family(x):
    x=str(x)
    if '芯片' in x or '光刻' in x:return '芯片'
    if '机器人' in x:return '机器人'
    if '电网' in x or x=='电力':return '智能电网'
    if '算力' in x or '数据中心' in x:return '算力'
    if '中报' in x or '业绩' in x:return '业绩增长'
    if '医药' in x:return '医药'
    if '航天' in x:return '商业航天'
    if '食品' in x or '消费' in x:return '大消费'
    if '锂' in x:return '锂电池'
    if 'AI' in x or '人工智能' in x:return 'AI'
    if '通信' in x:return '通信'
    return x

def mainline_match(theme,raw):
    f=family(theme); r=str(raw)
    if f=='芯片': return '芯片' in r
    if f=='机器人': return '机器人' in r
    if f=='智能电网': return ('电网' in r or '电力' in r)
    if f=='算力': return ('算力' in r or '数据中心' in r or '云计算' in r)
    if f=='业绩增长': return ('业绩' in r or '中报' in r)
    if f=='医药': return '医药' in r
    if f=='商业航天': return '航天' in r
    if f=='大消费': return ('消费' in r or '食品' in r)
    if f=='锂电池': return '锂' in r
    if f=='AI': return ('AI' in r or '人工智能' in r or '大模型' in r)
    if f=='通信': return ('通信' in r or 'PCB' in r)
    return theme in r

def load_seed():
    ds=[]
    for p in sorted(CUR.glob('part*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip(): ds.append(json.loads(line))
    ds=sorted(ds,key=lambda x:x['date'])
    if len(ds)!=23 or len({x['date'] for x in ds})!=23: raise RuntimeError(f'seed dates invalid: {len(ds)}')
    return ds

def day_path(date):
    xs=list(DATA.glob(f'part-*/date={date}/minute1.parquet'))
    if len(xs)!=1: raise RuntimeError(f'{date} minute path count={len(xs)}')
    return xs[0]

def load_day_agg(date,prev_close):
    p=day_path(date)
    df=pd.read_parquet(p,columns=['code','datetime','open','high','low','close','amount'])
    df['code']=norm_code(df.code); df['datetime']=pd.to_datetime(df.datetime)
    for c in ['open','high','low','close','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['open','high','low','close']).sort_values(['code','datetime'])
    d=df.groupby('code').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),amount=('amount','sum')).reset_index()
    d['prev_close']=d.code.map(prev_close)
    d['day_ret']=np.where(d.prev_close.notna()&(d.prev_close>0),d.close/d.prev_close-1,d.close/d.open-1)
    d['intraday_ret']=d.close/d.open-1
    d['range_close_pos']=((d.close-d.low)/(d.high-d.low).replace(0,np.nan)).fillna(.5).clip(0,1)
    return d.set_index('code',drop=False),dict(zip(d.code,d.close))

def lifecycle(rows):
    hist=defaultdict(list); out=[]; day_index={d:i for i,d in enumerate(sorted({r['date'] for r in rows}))}
    for r in rows:
        f=r['theme_family']; di=day_index[r['date']]; cur=r['source_limit_up_count']; prev=hist[f]; consecutive=bool(prev and prev[-1][0]==di-1); pc=prev[-1][1] if prev else 0
        if not prev: ph='launch'
        elif consecutive and cur>=max(8,pc*1.35): ph='ferment'
        elif consecutive and cur>=10 and cur>=pc*.90: ph='climax'
        elif consecutive and cur<pc*.65: ph='first_divergence'
        elif consecutive: ph='confirm'
        elif cur>=6: ph='repair'
        else: ph='day0'
        vals=[x[1] for x in prev]+[cur]
        r['continuity_3d']=sum(v>0 for v in vals[-3:])/min(3,len(vals)); r['continuity_5d']=sum(v>0 for v in vals[-5:])/min(5,len(vals)); r['theme_phase']=ph
        r['theme_strength_score']=min(100,30*min(cur,20)/20 + 25*int(r['is_mainline']) + 15*r['continuity_3d'] + 15*r['continuity_5d'] + 15*r['member_up_ratio'])
        prev.append((di,cur)); out.append(r)
    return out

def expectations(days,theme_df):
    rows=[]
    for i,d in enumerate(days):
        date=d['date']; target=days[i+1]['date'] if i+1<len(days) else date; z=theme_df[theme_df.date==date].sort_values(['is_mainline','theme_strength_score'],ascending=False).head(3); th='|'.join(z.theme.astype(str))
        ph=d['phase']
        if ph=='ice': sc='repair_or_continue_ice'; trigger='09:35-10:00市场广度显著改善且前日核心无批量负反馈'; invalid='广度继续恶化或高位负反馈扩散'; action='仅冰点修复先锋/否则空仓'
        elif ph=='climax': sc='divergence_or_continuation'; trigger='核心换手承接+主线成员协同不掉队'; invalid='一致高开后炸裂或题材协同快速转弱'; action='禁止追高潮，只等核心第一次分歧转一致'
        elif ph in ('repair','warming'): sc='continuation_or_failure'; trigger='主线核心超预期+题材成员协同+市场广度确认'; invalid='核心和题材同时低于预期'; action='龙头主升/核心弱转强，失败则降权'
        else: sc='defense_or_switch'; trigger='旧周期负反馈收敛+新方向成员同步'; invalid='无新主线且亏钱效应扩大'; action='默认防守，仅低位切换试错'
        rows.append({'date':date,'target_date':target,'theme':th,'scenario':sc,'trigger_conditions':trigger,'invalid_conditions':invalid,'planned_actions':action,'created_at':f'{date}T16:35:00+08:00'})
    return pd.DataFrame(rows)

def main():
    seed=load_seed(); mp=pd.read_csv(PIT/'member_name_code.csv',dtype=str).fillna(''); name2code=dict(zip(mp.name,mp.code)); all_names={n for d in seed for t in d['themes'] for n in t['members']}|{d['dragon'] for d in seed}|{t['leader'] for d in seed for t in d['themes']}
    resolved=sum(n in name2code for n in all_names); ratio=resolved/max(len(all_names),1)
    if ratio<.90: raise RuntimeError(f'name resolution only {resolved}/{len(all_names)}={ratio:.1%}')
    markets=[]; themes=[]; ladders=[]; roles=[]; sources=[]; prev_close={}; streak=defaultdict(int)
    for d in seed:
        date=d['date']; agg,close_map=load_day_agg(date,prev_close)
        market_ret=pd.to_numeric(agg.day_ret,errors='coerce').dropna(); adv=int((market_ret>0).sum()); dec=int((market_ret<0).sum()); turnover=float(agg.amount.sum())
        dragon_code=name2code.get(d['dragon'],'')
        markets.append({'date':date,'limit_up_count':d['lu'],'limit_down_count':d['ld'],'broken_board_count':'','seal_rate':'','max_board':d['max_board'],'continuous_board_count':d['cont'],'advance_count':adv,'decline_count':dec,'yesterday_limitup_premium':'','turnover_amount':turnover,'emotion_phase':d['phase'],'emotion_temperature':d['temp'],'mainline_raw':d['mainline'],'market_dragon_name':d['dragon'],'market_dragon_code':dragon_code})
        day_limit_codes=set()
        for rank,t in enumerate(d['themes'],1):
            mapped=[]
            for order,name in enumerate(t['members'],1):
                code=name2code.get(name,'')
                if not code or code not in agg.index: continue
                mapped.append(code); day_limit_codes.add(code)
                ladders.append({'date':date,'code':code,'name':name,'theme':t['name'],'theme_family':family(t['name']),'board_level_july_min':'','source_member_order':order,'is_theme_leader':name==t['leader']})
            leader_code=name2code.get(t['leader'],'')
            z=agg.loc[[c for c in mapped if c in agg.index]] if mapped else pd.DataFrame()
            up=float((z.intraday_ret>0).mean()) if len(z) else 0; amount=float(z.amount.sum()) if len(z) else 0
            themes.append({'date':date,'theme':t['name'],'theme_family':family(t['name']),'source_limit_up_count':t['count'],'mapped_member_count':len(mapped),'member_coverage':len(mapped)/max(t['count'],1),'leader_name':t['leader'],'leader_code':leader_code,'is_mainline':mainline_match(t['name'],d['mainline']),'theme_rank':rank,'member_up_ratio':up,'member_amount':amount})
            if leader_code and leader_code in agg.index:
                roles.append({'date':date,'code':leader_code,'name':t['leader'],'theme':t['name'],'role':'theme_dragon','recognition_score':92 if mainline_match(t['name'],d['mainline']) else 80,'source':'dated_web_leader'})
            if len(z):
                cap_code=str(z.sort_values('amount',ascending=False).iloc[0].code); cap_name=mp.loc[mp.code==cap_code,'name'].iloc[0] if (mp.code==cap_code).any() else cap_code
                if cap_code!=leader_code: roles.append({'date':date,'code':cap_code,'name':cap_name,'theme':t['name'],'role':'capacity_core','recognition_score':82,'source':'same_day_github_amount'})
                ela=z.sort_values(['intraday_ret','range_close_pos'],ascending=False).iloc[0]; ela_code=str(ela.code); ela_name=mp.loc[mp.code==ela_code,'name'].iloc[0] if (mp.code==ela_code).any() else ela_code
                if ela_code not in {leader_code,cap_code}: roles.append({'date':date,'code':ela_code,'name':ela_name,'theme':t['name'],'role':'elastic_core','recognition_score':78,'source':'same_day_github_strength'})
        # Consecutive limit-up ladder inside July: exact from audited daily member presence; pre-July history intentionally unknown.
        for code in list(streak):
            if code not in day_limit_codes: streak[code]=0
        for code in day_limit_codes: streak[code]+=1
        for row in [x for x in ladders if x['date']==date]: row['board_level_july_min']=streak[row['code']]
        if dragon_code and dragon_code in agg.index:
            roles.append({'date':date,'code':dragon_code,'name':d['dragon'],'theme':'market','role':'market_dragon','recognition_score':100,'source':'dated_web_market_dragon'})
        sources.append({'date':date,'source_url':d['source_url'],'source_name':'连板网历史交易日浏览器渲染页','available_at':f'{date}T16:30:00+08:00','evidence_type':'manual_browser_audited_same_day_close','captured_fact':f"LU={d['lu']};LD={d['ld']};max_board={d['max_board']};phase={d['phase']};mainline={d['mainline']};member themes={','.join(t['name'] for t in d['themes'])}",'pit_ok':True,'future_sections_excluded':True})
        prev_close=close_map
    themes=lifecycle(themes); tdf=pd.DataFrame(themes); ldf=pd.DataFrame(ladders); rdf=pd.DataFrame(roles).drop_duplicates(['date','code','role','theme']); mdf=pd.DataFrame(markets); sdf=pd.DataFrame(sources); edf=expectations(seed,tdf)
    mdf.to_csv(PIT/'market_snapshot.csv',index=False,encoding='utf-8-sig'); tdf.to_csv(PIT/'theme_snapshot.csv',index=False,encoding='utf-8-sig'); ldf.to_csv(PIT/'ladder_snapshot.csv',index=False,encoding='utf-8-sig'); rdf.to_csv(PIT/'role_snapshot.csv',index=False,encoding='utf-8-sig'); edf.to_csv(PIT/'expectation_snapshot.csv',index=False,encoding='utf-8-sig'); sdf.to_csv(PIT/'source_evidence.csv',index=False,encoding='utf-8-sig')
    meta={'dates':len(seed),'unique_names':len(all_names),'resolved_names':resolved,'resolution_ratio':ratio,'ladder_rows':len(ldf),'distinct_ladder_codes':int(ldf.code.nunique()),'roles':len(rdf),'policy':'web T-close audited facts usable T+1; role enrichments use T GitHub minutes at close and are usable T+1'}
    (PIT/'build_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
