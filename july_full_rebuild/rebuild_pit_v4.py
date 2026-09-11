from __future__ import annotations
import re
import pandas as pd
from bs4 import BeautifulSoup
import rebuild_pit_v2 as b

HEADER = re.compile(r'(?:^|\s)(\d{2})\s*🔥?\s*([^\s]{1,24})\s+(\d+)家涨停')

def parse_board(rest: str):
    m=re.match(r'\s*(?:反[·\s]*)?(\d+)天(\d+)板',rest)
    if m:return int(m.group(2)),True
    m=re.match(r'\s*(\d+)连板',rest)
    if m:return int(m.group(1)),False
    m=re.match(r'\s*(\d+)板',rest)
    if m:return int(m.group(1)),False
    return 1,False

def parse(day,i):
    url,html=b.fetch(day,i); soup=BeautifulSoup(html,'lxml'); text=soup.get_text('\n',strip=True); flat=soup.get_text(' ',strip=True); codes=b.amap(soup)
    sm=re.search(r'今日结论\s*(冰点期|修复期|升温期|高潮期|降温期|退潮期)\s*([\d.]+)°\s*·\s*涨停\s*(\d+)\s*·\s*跌停\s*(\d+)\s*·\s*最高\s*(\d+)板\s*·\s*主线\s*([^\n]+)',text)
    if not sm: raise RuntimeError(day+' summary missing')
    pcn,temp,up,down,maxb,mainraw=sm.groups()
    cont=int(b.metric(text,'连板',r'(\d+)',0)); seal=float(b.metric(text,'封板率',r'([\d.]+)%',0))/100; broken=int(b.metric(text,'炸板',r'(\d+)',0)); prem=float(b.metric(text,'昨涨停今表现',r'([+-]?[\d.]+)%',0))/100
    ad=re.search(r'上涨/下跌\s*([\d,]+)/([\d,]+)',text); adv=int(ad.group(1).replace(',','')) if ad else 0; dec=int(ad.group(2).replace(',','')) if ad else 0
    tv=re.search(r'两市成交\s*([\d.]+)万亿',text); turnover=float(tv.group(1))*1e12 if tv else ''
    top_start=flat.find('当日主线'); top_end=flat.find('🔮 明日推演',top_start+1); top=flat[top_start:top_end if top_end>top_start else top_start+4000]
    md=re.search(r'总龙头\s+([^\s]+)',top); mdname=b.clean(md.group(1)) if md else ''; mdcode=codes.get(mdname,b.KNOWN.get(mdname,''))

    start=flat.find('热点题材全部涨停个股')
    if start<0: raise RuntimeError(day+' detailed theme section missing')
    end=flat.find('实时快讯',start)
    sec=flat[start:end if end>start else start+25000]
    hs=list(HEADER.finditer(sec))
    if len(hs)<2: raise RuntimeError(f'{day} detailed headers missing: {sec[:1200]}')
    main_tokens=[x.strip() for x in re.split('[、，,/]',mainraw) if x.strip()]
    themes=[]; ladder=[]
    # Stock anchors are resolved from page links; only names preceded by a seal timestamp inside this theme block count as members.
    stock_names=sorted(codes.keys(),key=len,reverse=True)
    for rank,h in enumerate(hs[:12],1):
        bend=hs[rank].start() if rank<len(hs) else len(sec)
        block=sec[h.end():bend]; theme=h.group(2).strip(); count=int(h.group(3)); members=[]; occupied=[]
        for name in stock_names:
            pos=0
            while True:
                j=block.find(name,pos)
                if j<0: break
                pre=block[max(0,j-18):j]
                tm=list(re.finditer(r'(\d{2}:\d{2})\s*(★?)\s*$',pre))
                if tm:
                    mm=tm[-1]; seal_time=mm.group(1); star=bool(mm.group(2)); code=codes[name]; after=block[j+len(name):j+len(name)+25]; board,is_repack=parse_board(after)
                    members.append((j,name,code,seal_time,star,board,is_repack)); break
                pos=j+len(name)
        members.sort(key=lambda x:x[0])
        # Name lookup can miss a few DOM variants. Require material coverage, but never invent absent members.
        unique=[]; seen=set()
        for x in members:
            if x[2] in seen: continue
            seen.add(x[2]); unique.append(x)
        members=unique
        if count>=5 and len(members)<max(2,min(count,5)//2):
            raise RuntimeError(f'{day} {theme} member coverage too low {len(members)}/{count}')
        leader=next((x for x in members if x[4]), members[0] if members else (0,'','', '',False,1,False))
        first_tm=min([x[0] for x in members],default=len(block)); catalyst=block[:first_tm].strip()[:300]
        ismain=any(q in theme or theme in q for q in main_tokens)
        themes.append({'date':day,'theme':theme,'catalyst':catalyst,'catalyst_first_visible_at':f'{day}T15:30:00+08:00','limit_up_count':count,'strong_stock_count':len(members),'max_board':max([x[5] for x in members] or [1]),'first_board_count':sum(x[5]==1 for x in members),'second_board_count':sum(x[5]==2 for x in members),'higher_board_count':sum(x[5]>=3 for x in members),'broken_board_count':'','big_loss_count':'','breadth':'','turnover_amount':'','continuity_3d':'','continuity_5d':'','theme_phase':'','theme_strength_score':'','leader_name':leader[1],'leader_code':leader[2],'is_mainline':ismain,'theme_rank':rank,'member_coverage':len(members)/max(count,1)})
        for _,name,code,tm,star,board,is_repack in members:
            ladder.append({'date':day,'code':code,'name':name,'theme':theme,'board_level':board,'is_first_board':board==1,'is_continuous_board':board>=2,'is_repack':is_repack,'seal_time':tm,'broken_times':'','is_theme_leader':star})
    if not any(x['is_mainline'] for x in themes):
        # Daily archive labels sometimes use aliases (国产芯片 vs 芯片). Mark top factual themes that overlap semantic suffixes.
        for x in themes:
            if any(('芯片' in q and '芯片' in x['theme']) or ('机器人' in q and '机器人' in x['theme']) or ('消费' in q and ('消费' in x['theme'] or '食品' in x['theme'])) or ('电网' in q and ('电力' in x['theme'] or '电网' in x['theme'])) for q in main_tokens): x['is_mainline']=True
    market={'date':day,'limit_up_count':int(up),'limit_down_count':int(down),'broken_board_count':broken,'seal_rate':seal,'max_board':int(maxb),'continuous_board_count':cont,'advance_count':adv,'decline_count':dec,'yesterday_limitup_premium':prem,'turnover_amount':turnover,'emotion_phase':b.PHASE[pcn],'emotion_temperature':float(temp),'mainline_raw':mainraw,'market_dragon_name':mdname,'market_dragon_code':mdcode}
    if mdcode and not any(x['code']==mdcode for x in ladder): ladder.append({'date':day,'code':mdcode,'name':mdname,'theme':'market','board_level':int(maxb),'is_first_board':False,'is_continuous_board':True,'is_repack':False,'seal_time':'','broken_times':'','is_theme_leader':True})
    source={'date':day,'source_url':url,'source_name':'连板网历史收盘复盘','published_at':f'{day}T16:30:00+08:00','available_at':f'{day}T16:30:00+08:00','evidence_type':'same_day_close_full_visible_theme_members','captured_fact':f'{up}涨停/{down}跌停/最高{maxb}板/情绪{pcn}/主线{mainraw}; detailed theme members+seal time; 明日推演/同景/前瞻 excluded','pit_ok':True}
    return market,themes,ladder,source

# Replace v2 parser; reuse causal lifecycle/role/expectation builders and strict 23-day completion checks.
b.parse=parse
if __name__=='__main__': b.main()
