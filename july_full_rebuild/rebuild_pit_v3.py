from __future__ import annotations
import re
import pandas as pd
from bs4 import BeautifulSoup
import rebuild_pit_v2 as b


def parse(day,i):
    url,html=b.fetch(day,i)
    soup=BeautifulSoup(html,'lxml')
    text=soup.get_text('\n',strip=True)
    flat=soup.get_text(' ',strip=True)
    codes=b.amap(soup)
    sm=re.search(r'今日结论\s*(冰点期|修复期|升温期|高潮期|降温期|退潮期)\s*([\d.]+)°\s*·\s*涨停\s*(\d+)\s*·\s*跌停\s*(\d+)\s*·\s*最高\s*(\d+)板\s*·\s*主线\s*([^\n]+)',text)
    if not sm: raise RuntimeError(day+' summary missing')
    pcn,temp,up,down,maxb,mainraw=sm.groups()
    cont=int(b.metric(text,'连板',r'(\d+)',0)); seal=float(b.metric(text,'封板率',r'([\d.]+)%',0))/100; broken=int(b.metric(text,'炸板',r'(\d+)',0)); prem=float(b.metric(text,'昨涨停今表现',r'([+-]?[\d.]+)%',0))/100
    ad=re.search(r'上涨/下跌\s*([\d,]+)/([\d,]+)',text); adv=int(ad.group(1).replace(',','')) if ad else 0; dec=int(ad.group(2).replace(',','')) if ad else 0
    tv=re.search(r'两市成交\s*([\d.]+)万亿',text); turnover=float(tv.group(1))*1e12 if tv else ''

    # Flat text survives DOM span splitting. Restrict to the first factual mainline section.
    start=flat.find('当日主线')
    if start<0: raise RuntimeError(day+' mainline section missing')
    top=flat[start:start+4500]
    # Cut at the forecast narrative when possible; never parse 同景/前瞻 fields.
    cut_candidates=[x for x in [top.find('明日推演今日'),top.find('竞价强股雷达'),top.find('今日AI复盘')] if x>300]
    if cut_candidates: top=top[:min(cut_candidates)]

    md=re.search(r'总龙头\s*([^\s]+?)(?:\s*(\d+)天(\d+)板|\s*(\d+)板)?(?:\s|$)',top)
    mdname=b.clean(md.group(1)) if md else ''
    # If regex captured only a bare name, code map / verified fallback resolves it.
    mdcode=codes.get(mdname,b.KNOWN.get(mdname,''))

    # Examples after flattening: 机器人 涨停 23家 龙头 埃斯顿 2天2板 ...
    pat=re.compile(r'([^\s·→]{1,22})\s*涨停\s*(\d+)家\s*龙头\s*([^\s·→]{2,18})')
    found=[]
    for m in pat.finditer(top):
        th=m.group(1).strip(); cnt=int(m.group(2)); leader=b.clean(m.group(3)); code=codes.get(leader,b.KNOWN.get(leader,''))
        if th in ('涨停','当日主线','强度榜','主线板块') or cnt<=0: continue
        found.append((th,cnt,leader,code))
    ded=[]; seen=set()
    for x in found:
        if x[0] in seen: continue
        seen.add(x[0]); ded.append(x)
    found=ded[:8]
    if len(found)<2:
        # Diagnostic deliberately fails the readiness chain rather than silently fabricating fields.
        raise RuntimeError(f'{day} too few themes after flat parse: {found}; sample={top[:1200]}')

    market={'date':day,'limit_up_count':int(up),'limit_down_count':int(down),'broken_board_count':broken,'seal_rate':seal,'max_board':int(maxb),'continuous_board_count':cont,'advance_count':adv,'decline_count':dec,'yesterday_limitup_premium':prem,'turnover_amount':turnover,'emotion_phase':b.PHASE[pcn],'emotion_temperature':float(temp),'mainline_raw':mainraw,'market_dragon_name':mdname,'market_dragon_code':mdcode}
    themes=[]; ladder=[]; main_tokens=[x.strip() for x in re.split('[、，,/]',mainraw) if x.strip()]
    for rank,(th,cnt,leader,code) in enumerate(found,1):
        ismain=any((q in th or th in q) for q in main_tokens)
        themes.append({'date':day,'theme':th,'catalyst':'see dated source','catalyst_first_visible_at':f'{day}T15:30:00+08:00','limit_up_count':cnt,'strong_stock_count':cnt,'max_board':'','first_board_count':'','second_board_count':'','higher_board_count':'','broken_board_count':'','big_loss_count':'','breadth':'','turnover_amount':'','continuity_3d':'','continuity_5d':'','theme_phase':'','theme_strength_score':'','leader_name':leader,'leader_code':code,'is_mainline':ismain,'theme_rank':rank})
        if code:
            ladder.append({'date':day,'code':code,'name':leader,'theme':th,'board_level':'','is_first_board':'','is_continuous_board':'','seal_time':'','broken_times':'','is_theme_leader':True})
    if mdcode:
        ladder.append({'date':day,'code':mdcode,'name':mdname,'theme':'market','board_level':int(maxb),'is_first_board':False,'is_continuous_board':True,'seal_time':'','broken_times':'','is_theme_leader':True})
    source={'date':day,'source_url':url,'source_name':'连板网历史收盘复盘','published_at':f'{day}T15:30:00+08:00','available_at':f'{day}T15:30:00+08:00','evidence_type':'same_day_close_only','captured_fact':f'{up}涨停/{down}跌停/最高{maxb}板/情绪{pcn}/主线{mainraw}; factual mainline block only; 明日推演/同景/前瞻 excluded','pit_ok':True}
    return market,themes,ladder,source

b.parse=parse
if __name__=='__main__': b.main()
