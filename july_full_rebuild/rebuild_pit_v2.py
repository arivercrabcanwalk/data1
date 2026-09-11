from __future__ import annotations
import re, time, json
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'pit_data'; OUT.mkdir(parents=True,exist_ok=True)
DAYS=['2026-07-01','2026-07-02','2026-07-03','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10','2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22','2026-07-23','2026-07-24','2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31']
PHASE={'冰点期':'ice','修复期':'repair','升温期':'warming','高潮期':'climax','降温期':'cooling','退潮期':'retreat'}
# Only code-resolution fallback. Theme/role facts still come from each dated archive page.
KNOWN={
'多氟多':'002407','海南海药':'000566','埃斯顿':'002747','康欣新材':'600076','星网锐捷':'002396','宜宾纸业':'600793','福莱新材':'605488',
'恒尚节能':'603137','富春染织':'605189','航发科技':'600391','甘化科工':'000576','中电港':'001287','晋拓股份':'603211','华孚时尚':'002042',
'晨光新材':'605399','万通发展':'600246','大恒科技':'600288','宝地矿业':'601121','大名城':'600094','浙江美大':'002677','兴业股份':'603928',
'崇达技术':'002815','视源股份':'002841','亚联机械':'001395','星网宇达':'002829','立方制药':'003020','柘中股份':'002346','日科化学':'300214','睿能科技':'603933',
'哈药股份':'600664','东山精密':'002384','宁波中百':'600857','九安医疗':'002432','道明光学':'002632','沃顿科技':'000920','永安药业':'002365','艾艾精工':'603580',
'立新能源':'001258','智度股份':'000676','大有能源':'600403','湖南发展':'000722','中嘉博创':'000889','长裕集团':'603407','大族激光':'002008','孚日股份':'002083',
'凯迪股份':'605288','金牛化工':'600722','爱丽家居':'603221','托伦斯':'301583','长城军工':'601606','长缆科技':'002879','新亚制程':'002388','哈三联':'002900','贤丰控股':'002141',
'华天酒店':'000428','创新医疗':'002173','中电鑫龙':'002298','明新旭腾':'605068','一鸣食品':'605179','苏州科达':'603660','江淮汽车':'600418','海兴电力':'603556','泛微网络':'603039','香江控股':'600162'
}
UA={'User-Agent':'Mozilla/5.0 (compatible; causal-market-review-research/1.0; +https://github.com/arivercrabcanwalk/data1)'}
S=requests.Session(); S.headers.update(UA)

def clean(x):
    x=re.sub(r'^[★🔥]+','',str(x).strip())
    x=re.sub(r'(?:反·?|反)?\s*\d+天\d+板$','',x)
    x=re.sub(r'\d+连板$','',x); x=re.sub(r'\d+板$','',x)
    return x.strip()

def fetch(day,i):
    url=f'https://lianban.net/days/{day}.html'
    if i: time.sleep(5.0)
    last=None
    for k in range(5):
        r=S.get(url,timeout=35); last=r
        if r.status_code==200: return url,r.text
        if r.status_code==429:
            wait=min(75,15*(k+1)); print(day,'429 wait',wait,flush=True); time.sleep(wait); continue
        r.raise_for_status()
    raise RuntimeError(f'{day} fetch failed status={last.status_code if last else None}')

def amap(soup):
    out=dict(KNOWN)
    for a in soup.find_all('a'):
        href=a.get('href') or ''; m=re.search(r'/gu/(\d{6})\.html',href)
        if not m: m=re.search(r'(?<!\d)(\d{6})(?!\d)',href)
        if m:
            n=clean(a.get_text(' ',strip=True))
            if n: out[n]=m.group(1)
    return out

def metric(text,label,pat,default=''):
    m=re.search(rf'{re.escape(label)}\s*{pat}',text)
    return m.group(1) if m else default

def parse(day,i):
    url,html=fetch(day,i); soup=BeautifulSoup(html,'lxml'); text=soup.get_text('\n',strip=True); codes=amap(soup)
    sm=re.search(r'今日结论\s*(冰点期|修复期|升温期|高潮期|降温期|退潮期)\s*([\d.]+)°\s*·\s*涨停\s*(\d+)\s*·\s*跌停\s*(\d+)\s*·\s*最高\s*(\d+)板\s*·\s*主线\s*([^\n]+)',text)
    if not sm: raise RuntimeError(day+' summary missing')
    pcn,temp,up,down,maxb,mainraw=sm.groups()
    cont=int(metric(text,'连板',r'(\d+)',0)); seal=float(metric(text,'封板率',r'([\d.]+)%',0))/100; broken=int(metric(text,'炸板',r'(\d+)',0)); prem=float(metric(text,'昨涨停今表现',r'([+-]?[\d.]+)%',0))/100
    ad=re.search(r'上涨/下跌\s*([\d,]+)/([\d,]+)',text); adv=int(ad.group(1).replace(',','')) if ad else 0; dec=int(ad.group(2).replace(',','')) if ad else 0
    tv=re.search(r'两市成交\s*([\d.]+)万亿',text); turnover=float(tv.group(1))*1e12 if tv else ''
    # Strictly same-day summary block only. Stop before 明日推演 so future-sample widgets cannot leak in.
    start=text.find('当日主线'); stop=text.find('明日推演',start+1); top=text[start:stop if stop>start else start+2500]
    md=re.search(r'总龙头\s*([^\n]+?)(?:(\d+)天(\d+)板|(\d+)板)?(?:\n|$)',top)
    mdname=clean(md.group(1)) if md else ''
    mdcode=codes.get(mdname,KNOWN.get(mdname,''))
    # Theme lines have stable form: 题材涨停N家龙头 NAME ... followed by catalyst line.
    pat=re.compile(r'([^\n]{1,24}?)涨停\s*(\d+)家龙头\s*([^\n]+)')
    found=[]
    for m in pat.finditer(top):
        th=m.group(1).strip(); cnt=int(m.group(2)); rawleader=m.group(3).strip(); leader=clean(rawleader); code=codes.get(leader,KNOWN.get(leader,''))
        # strip accidental trailing board text from leader but preserve real Chinese name
        found.append((th,cnt,leader,code))
    # Deduplicate and keep factual strength-board list visible at close.
    ded=[]; seen=set()
    for x in found:
        if x[0] in seen: continue
        seen.add(x[0]); ded.append(x)
    found=ded[:8]
    if len(found)<2: raise RuntimeError(f'{day} too few themes: {found}')
    market={'date':day,'limit_up_count':int(up),'limit_down_count':int(down),'broken_board_count':broken,'seal_rate':seal,'max_board':int(maxb),'continuous_board_count':cont,'advance_count':adv,'decline_count':dec,'yesterday_limitup_premium':prem,'turnover_amount':turnover,'emotion_phase':PHASE[pcn],'emotion_temperature':float(temp),'mainline_raw':mainraw,'market_dragon_name':mdname,'market_dragon_code':mdcode}
    themes=[]; ladder=[]
    main_tokens=[x.strip() for x in re.split('[、，,/]',mainraw) if x.strip()]
    for rank,(th,cnt,leader,code) in enumerate(found,1):
        ismain=any((q in th or th in q) for q in main_tokens)
        themes.append({'date':day,'theme':th,'catalyst':'see dated source','catalyst_first_visible_at':f'{day}T15:30:00+08:00','limit_up_count':cnt,'strong_stock_count':cnt,'max_board':'','first_board_count':'','second_board_count':'','higher_board_count':'','broken_board_count':'','big_loss_count':'','breadth':'','turnover_amount':'','continuity_3d':'','continuity_5d':'','theme_phase':'','theme_strength_score':'','leader_name':leader,'leader_code':code,'is_mainline':ismain,'theme_rank':rank})
        if code:
            ladder.append({'date':day,'code':code,'name':leader,'theme':th,'board_level':'','is_first_board':'','is_continuous_board':'','seal_time':'','broken_times':'','is_theme_leader':True})
    if mdcode:
        ladder.append({'date':day,'code':mdcode,'name':mdname,'theme':'market','board_level':int(maxb),'is_first_board':False,'is_continuous_board':True,'seal_time':'','broken_times':'','is_theme_leader':True})
    source={'date':day,'source_url':url,'source_name':'连板网历史收盘复盘','published_at':f'{day}T15:30:00+08:00','available_at':f'{day}T15:30:00+08:00','evidence_type':'same_day_close_only','captured_fact':f'{up}涨停/{down}跌停/最高{maxb}板/情绪{pcn}/主线{mainraw}; only 当日主线 before 明日推演','pit_ok':True}
    return market,themes,ladder,source

def enrich(themes):
    df=pd.DataFrame(themes); dayidx={d:i for i,d in enumerate(DAYS)}; hist={}; phases=[]; c3=[]; c5=[]; scores=[]
    for r in df.itertuples(index=False):
        th=str(r.theme); di=dayidx[r.date]; cur=float(r.limit_up_count); h=hist.setdefault(th,[]); prev=[x for x in h if x[0]<di]; consecutive=bool(prev and prev[-1][0]==di-1); pc=prev[-1][1] if prev else 0
        if not prev: ph='launch'
        elif consecutive and cur>=max(8,pc*1.35): ph='ferment'
        elif consecutive and cur>=10 and cur>=pc*.9: ph='climax'
        elif consecutive and cur<pc*.65: ph='first_divergence'
        elif consecutive: ph='confirm'
        elif cur>=6: ph='repair'
        else: ph='day0'
        vals=[x[1] for x in prev]+[cur]; c3.append(sum(v>0 for v in vals[-3:])/min(3,len(vals))); c5.append(sum(v>0 for v in vals[-5:])/min(5,len(vals)))
        score=min(100,25*min(cur,20)/20 + (30 if bool(r.is_mainline) else 0) + 15*min(float(r.theme_rank),4)**-1 + 15*c3[-1] + 15*c5[-1])
        phases.append(ph); scores.append(score); h.append((di,cur))
    df['theme_phase']=phases; df['continuity_3d']=c3; df['continuity_5d']=c5; df['theme_strength_score']=scores
    return df

def build_roles(market,theme):
    rows=[]; t=theme
    for d in DAYS:
        mr=market[market.date==d].iloc[0]; used=set(); mc=str(mr.market_dragon_code)
        if len(mc)==6:
            rows.append({'date':d,'code':mc,'name':mr.market_dragon_name,'theme':'market','role':'market_dragon','recognition_score':100,'lead_lag_score':'','capacity_score':'','ladder_score':100,'theme_score':100,'role_confidence':0.95}); used.add(mc)
        z=t[t.date==d].sort_values(['is_mainline','theme_strength_score','theme_rank'],ascending=[False,False,True])
        for r in z.itertuples(index=False):
            c=str(r.leader_code); 
            if len(c)!=6 or c in used: continue
            role='theme_dragon' if bool(r.is_mainline) else 'front_row'
            rows.append({'date':d,'code':c,'name':r.leader_name,'theme':r.theme,'role':role,'recognition_score':min(99,68+0.25*float(r.theme_strength_score)),'lead_lag_score':'','capacity_score':'','ladder_score':'','theme_score':r.theme_strength_score,'role_confidence':0.85 if role=='theme_dragon' else 0.72}); used.add(c)
    return pd.DataFrame(rows)

def build_expect(market,theme):
    rows=[]
    for i,d in enumerate(DAYS):
        mr=market[market.date==d].iloc[0]; target=DAYS[i+1] if i+1<len(DAYS) else d; tops=theme[theme.date==d].sort_values(['is_mainline','theme_strength_score'],ascending=False).head(3); names='|'.join(tops.theme.tolist())
        if mr.emotion_phase=='ice': sc='repair_or_continue_ice'; trig='早盘广度改善+核心无批量负反馈'; invalid='广度恶化或高位负反馈扩散'; act='只做冰点修复先锋/否则空仓'
        elif mr.emotion_phase=='climax': sc='divergence_or_continuation'; trig='核心换手承接+主线前排不掉队'; invalid='一致高开后炸板扩散'; act='不追高潮，只等核心第一次分歧转一致'
        elif mr.emotion_phase in ('repair','warming'): sc='continuation_or_failure'; trig='主线核心超预期+早盘市场广度确认'; invalid='核心及板块同步低于预期'; act='龙头主升/核心弱转强，失败降仓'
        else: sc='defense_or_switch'; trig='旧周期负反馈收敛+新方向多股同步'; invalid='无新主线且亏钱效应扩大'; act='默认防守，仅低位切换试错'
        rows.append({'date':d,'target_date':target,'theme':names,'scenario':sc,'trigger_conditions':trig,'invalid_conditions':invalid,'planned_actions':act,'created_at':f'{d}T15:35:00+08:00'})
    return pd.DataFrame(rows)

def main():
    ms=[]; ts=[]; ls=[]; ss=[]
    for i,d in enumerate(DAYS):
        m,t,l,s=parse(d,i); ms.append(m); ts+=t; ls+=l; ss.append(s); print(d,m['emotion_phase'],'themes',len(t),'ladder',len(l),'codes',sum(bool(x['code']) for x in l),flush=True)
    mdf=pd.DataFrame(ms); tdf=enrich(ts); ldf=pd.DataFrame(ls); rdf=build_roles(mdf,tdf); edf=build_expect(mdf,tdf); sdf=pd.DataFrame(ss)
    if len(mdf)!=23 or mdf.date.nunique()!=23: raise SystemExit('market incomplete')
    if ldf.date.nunique()!=23 or rdf.date.nunique()!=23: raise SystemExit(f'role/ladder incomplete ladder_days={ldf.date.nunique()} role_days={rdf.date.nunique()}')
    mdf.to_csv(OUT/'market_snapshot.csv',index=False,encoding='utf-8-sig'); tdf.to_csv(OUT/'theme_snapshot.csv',index=False,encoding='utf-8-sig'); ldf.to_csv(OUT/'ladder_snapshot.csv',index=False,encoding='utf-8-sig'); rdf.to_csv(OUT/'role_snapshot.csv',index=False,encoding='utf-8-sig'); edf.to_csv(OUT/'expectation_snapshot.csv',index=False,encoding='utf-8-sig'); sdf.to_csv(OUT/'source_evidence.csv',index=False,encoding='utf-8-sig')
    print('COMPLETE',{'market':len(mdf),'themes':len(tdf),'ladder':len(ldf),'roles':len(rdf),'expectations':len(edf),'sources':len(sdf)},flush=True)
if __name__=='__main__': main()
