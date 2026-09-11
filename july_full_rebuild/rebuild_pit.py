from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pit_data"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [
    '2026-07-01','2026-07-02','2026-07-03','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10',
    '2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22',
    '2026-07-23','2026-07-24','2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31'
]

PHASE_MAP = {
    '冰点期':'ice','修复期':'repair','升温期':'warming','高潮期':'climax','降温期':'cooling','退潮期':'retreat'
}

UA = {'User-Agent':'Mozilla/5.0 (compatible; JulyPITResearch/1.0)'}


def clean_name(x: str) -> str:
    x = x.strip().replace('★','')
    x = re.sub(r'反\s*\d+天\d+板$', '', x)
    x = re.sub(r'\d+连板$', '', x)
    x = re.sub(r'\d+天\d+板$', '', x)
    x = re.sub(r'\d+板$', '', x)
    return x.strip()


def anchor_code_map(soup: BeautifulSoup) -> Dict[str, str]:
    out = {}
    for a in soup.find_all('a'):
        name = clean_name(a.get_text(' ', strip=True))
        href = a.get('href') or ''
        m = re.search(r'(?<!\d)(\d{6})(?!\d)', href)
        if name and m:
            out[name] = m.group(1)
    return out


def extract_metric(text: str, label: str, suffix: str = '') -> Optional[float]:
    # page text is newline-heavy; accept arbitrary whitespace between label and value.
    p = rf'{re.escape(label)}\s*([+-]?[\d,.]+(?:\.\d+)?)\s*{re.escape(suffix)}'
    m = re.search(p, text)
    if not m:
        return None
    return float(m.group(1).replace(',',''))


def parse_day(day: str) -> dict:
    url = f'https://lianban.net/days/{day}.html'
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    text = soup.get_text('\n', strip=True)
    cmap = anchor_code_map(soup)

    # Only same-day observable facts. Explicitly ignore sections such as 明日推演/同景日期/前瞻回验.
    sm = re.search(r'今日结论\s*(冰点期|修复期|升温期|高潮期|降温期|退潮期)\s*([\d.]+)°\s*·\s*涨停\s*(\d+)\s*·\s*跌停\s*(\d+)\s*·\s*最高\s*(\d+)板\s*·\s*主线\s*([^\n]+)', text)
    if not sm:
        raise RuntimeError(f'cannot parse summary for {day}')
    phase_cn, temp, up, down, max_board, mainline = sm.groups()
    phase = PHASE_MAP[phase_cn]

    def after_label(label, pat):
        m = re.search(rf'{label}\s*{pat}', text)
        return m.group(1) if m else ''

    continuous = int(float(after_label('连板', r'(\d+)') or 0))
    seal_rate = float(after_label('封板率', r'([\d.]+)%') or 0) / 100
    broken = int(float(after_label('炸板', r'(\d+)') or 0))
    premium = float(after_label('昨涨停今表现', r'([+-]?[\d.]+)%') or 0) / 100
    advdec = re.search(r'上涨/下跌\s*([\d,]+)/([\d,]+)', text)
    adv = int(advdec.group(1).replace(',','')) if advdec else 0
    dec = int(advdec.group(2).replace(',','')) if advdec else 0
    turnover = None
    mt = re.search(r'两市成交\s*([\d.]+)万亿', text)
    if mt:
        turnover = float(mt.group(1)) * 1e12

    # Top-line thematic blocks, restricted to the factual "热点题材全部涨停个股" section.
    sec = text.split('热点题材全部涨停个股', 1)
    factual = sec[1] if len(sec) == 2 else text
    factual = factual.split('实时快讯', 1)[0]
    # Blocks begin 01/02/... and stop at next block. Use first 12 visible themes.
    starts = list(re.finditer(r'(?m)^\s*(\d{2})(?:🔥)?\s*([^\n]+)\s*$', factual))
    themes = []
    ladder = []
    for i, st in enumerate(starts[:12]):
        end = starts[i+1].start() if i+1 < len(starts) else len(factual)
        block = factual[st.end():end]
        theme = st.group(2).strip()
        mcnt = re.search(r'(\d+)家涨停', block)
        if not mcnt:
            continue
        count = int(mcnt.group(1))
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        catalyst = ''
        for x in lines[:8]:
            if ('涨停' not in x and not re.match(r'^\d{2}:\d{2}', x) and len(x) >= 8):
                catalyst = x[:240]
                break
        leader_name = ''
        leader_code = ''
        stock_rows = []
        for x in lines:
            mm = re.match(r'^(\d{2}:\d{2})(★?)([^\n]+)$', x)
            if not mm:
                continue
            tm, star, raw = mm.groups()
            nm = clean_name(raw)
            if not nm or len(nm) > 18:
                continue
            code = cmap.get(nm, '')
            board_m = re.search(r'(?:反\s*)?(\d+)天(\d+)板|(?:(\d+)连板)', raw)
            board = 1
            if board_m:
                vals = [v for v in board_m.groups() if v]
                board = int(vals[-1]) if vals else 1
            is_leader = bool(star)
            if is_leader and not leader_name:
                leader_name, leader_code = nm, code
            stock_rows.append({'date':day,'code':code,'name':nm,'theme':theme,'board_level':board,
                               'is_first_board':board==1,'is_continuous_board':board>=2,
                               'seal_time':tm,'broken_times':'','is_theme_leader':is_leader})
        if not leader_name and stock_rows:
            leader_name, leader_code = stock_rows[0]['name'], stock_rows[0]['code']
        themes.append({'date':day,'theme':theme,'catalyst':catalyst,'catalyst_first_visible_at':f'{day}T15:30:00+08:00',
                       'limit_up_count':count,'strong_stock_count':count,'max_board':max([x['board_level'] for x in stock_rows] or [1]),
                       'first_board_count':sum(x['board_level']==1 for x in stock_rows),'second_board_count':sum(x['board_level']==2 for x in stock_rows),
                       'higher_board_count':sum(x['board_level']>=3 for x in stock_rows),'broken_board_count':'','big_loss_count':'',
                       'breadth':'','turnover_amount':'','continuity_3d':'','continuity_5d':'','theme_phase':'','theme_strength_score':'',
                       'leader_name':leader_name,'leader_code':leader_code,'is_mainline':theme in mainline})
        ladder.extend(stock_rows)

    # If the page's compact mainline labels differ from detailed blocks, retain them as factual themes too.
    main_themes = [x.strip() for x in re.split('[、,，/]', mainline) if x.strip()]
    present = {x['theme'] for x in themes}
    for th in main_themes:
        if th not in present:
            themes.append({'date':day,'theme':th,'catalyst':'','catalyst_first_visible_at':f'{day}T15:30:00+08:00',
                           'limit_up_count':'','strong_stock_count':'','max_board':'','first_board_count':'','second_board_count':'',
                           'higher_board_count':'','broken_board_count':'','big_loss_count':'','breadth':'','turnover_amount':'',
                           'continuity_3d':'','continuity_5d':'','theme_phase':'','theme_strength_score':'',
                           'leader_name':'','leader_code':'','is_mainline':True})

    # Market dragon from factual summary text. Use only same-day board result.
    md = re.search(r'总龙头\s*([^\n]+?)(\d+)天(\d+)板', text)
    market_dragon_name = clean_name(md.group(1)) if md else ''
    market_dragon_code = cmap.get(market_dragon_name, '') if market_dragon_name else ''

    return {
        'market': {'date':day,'limit_up_count':int(up),'limit_down_count':int(down),'broken_board_count':broken,
                   'seal_rate':seal_rate,'max_board':int(max_board),'continuous_board_count':continuous,
                   'advance_count':adv,'decline_count':dec,'yesterday_limitup_premium':premium,
                   'turnover_amount':turnover if turnover is not None else '', 'emotion_phase':phase,
                   'emotion_temperature':float(temp),'mainline_raw':mainline,
                   'market_dragon_name':market_dragon_name,'market_dragon_code':market_dragon_code},
        'themes': themes,
        'ladder': ladder,
        'source': {'date':day,'source_url':url,'source_name':'连板网历史收盘复盘','published_at':f'{day}T15:30:00+08:00',
                   'available_at':f'{day}T15:30:00+08:00','evidence_type':'same_day_close_reconstruction',
                   'captured_fact':f'收盘事实: {up}涨停/{down}跌停/最高{max_board}板; 主线={mainline}; 仅使用当日事实字段,排除明日推演/同景统计/前瞻回验',
                   'pit_ok':True}
    }


def infer_theme_lifecycle(theme_df: pd.DataFrame) -> pd.DataFrame:
    theme_df = theme_df.copy()
    counts = {}
    last_day = {}
    prev_count = {}
    histories = {}
    phases=[]; c3=[]; c5=[]; strengths=[]
    day_index={d:i for i,d in enumerate(DAYS)}
    for r in theme_df.itertuples(index=False):
        th=r.theme; di=day_index[r.date]; cur=float(r.limit_up_count) if str(r.limit_up_count) not in ('','nan') else 0
        hist=histories.setdefault(th,[])
        # only past/current observations
        prior=[x for x in hist if x[0] < di]
        seen=len(prior)
        consecutive = bool(prior and prior[-1][0] == di-1)
        pc=prior[-1][1] if prior else 0
        if seen==0: ph='launch'
        elif consecutive and cur>=max(6,pc*1.35): ph='ferment'
        elif consecutive and cur>=pc and cur>=10: ph='climax'
        elif consecutive and cur<pc*0.65: ph='first_divergence'
        elif consecutive: ph='confirm'
        elif seen>0 and cur>=6: ph='repair'
        else: ph='day0'
        hvals=[x[1] for x in prior]+[cur]
        c3.append(sum(v>0 for v in hvals[-3:])/min(3,len(hvals)))
        c5.append(sum(v>0 for v in hvals[-5:])/min(5,len(hvals)))
        # Low-DOF score based on observable same-day structure only.
        mb=float(r.max_board) if str(r.max_board) not in ('','nan') else 1
        score=min(100.0, 6*min(cur,12)/12*10 + 20*min(mb,6)/6 + (15 if bool(r.is_mainline) else 0))
        strengths.append(score); phases.append(ph); hist.append((di,cur))
    theme_df['theme_phase']=phases; theme_df['continuity_3d']=c3; theme_df['continuity_5d']=c5; theme_df['theme_strength_score']=strengths
    return theme_df


def build_roles(market_df: pd.DataFrame, theme_df: pd.DataFrame, ladder_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    md = market_df.set_index('date')
    for d in DAYS:
        t=theme_df[theme_df.date==d].copy()
        if t.empty: continue
        t=t.sort_values(['is_mainline','theme_strength_score','limit_up_count'], ascending=False, na_position='last')
        market_name=str(md.loc[d,'market_dragon_name'] or ''); market_code=str(md.loc[d,'market_dragon_code'] or '')
        if market_code:
            rows.append({'date':d,'code':market_code,'name':market_name,'theme':'market','role':'market_dragon','recognition_score':100,
                         'lead_lag_score':'','capacity_score':'','ladder_score':100,'theme_score':100,'role_confidence':0.9})
        used=set([market_code]) if market_code else set()
        for tr in t.head(6).itertuples(index=False):
            code=str(getattr(tr,'leader_code','') or ''); name=str(getattr(tr,'leader_name','') or '')
            if not code or code in used: continue
            role='theme_dragon' if bool(tr.is_mainline) else 'front_row'
            ts=float(tr.theme_strength_score or 0)
            rows.append({'date':d,'code':code,'name':name,'theme':tr.theme,'role':role,'recognition_score':min(99,65+0.3*ts),
                         'lead_lag_score':'','capacity_score':'','ladder_score':min(100,15*float(tr.max_board or 1)),'theme_score':ts,'role_confidence':0.8})
            used.add(code)
    return pd.DataFrame(rows)


def build_expectations(market_df: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for i,d in enumerate(DAYS):
        target=DAYS[i+1] if i+1<len(DAYS) else d
        mr=market_df[market_df.date==d].iloc[0]
        tops=theme_df[theme_df.date==d].sort_values(['is_mainline','theme_strength_score'],ascending=False).head(3)
        names='|'.join(tops.theme.astype(str).tolist())
        if mr.emotion_phase=='ice':
            scenario='repair_or_continue_ice'; trig='10:00全市场广度明显改善且昨日核心不出现批量负反馈'; invalid='广度继续恶化/高位跌停扩散'; act='仅小仓冰点修复先锋;否则空仓'
        elif mr.emotion_phase=='climax':
            scenario='cooling_or_strong_continuation'; trig='核心换手承接且主线前排继续扩散'; invalid='高开一致后炸板扩散/高位负反馈'; act='禁止追高潮;只做核心第一次分歧转一致'
        elif mr.emotion_phase in ('repair','warming'):
            scenario='continuation_or_divergence'; trig='主线核心超预期且早盘市场广度>=50%'; invalid='核心低于预期且板块同步走弱'; act='龙头主升/核心弱转强;失败则降仓'
        else:
            scenario='defense_or_low_level_switch'; trig='旧周期负反馈收敛且新方向出现多股同步'; invalid='无新主线/亏钱效应扩大'; act='默认空仓;只允许低位切换试错'
        rows.append({'date':d,'target_date':target,'theme':names,'scenario':scenario,'trigger_conditions':trig,
                     'invalid_conditions':invalid,'planned_actions':act,'created_at':f'{d}T15:35:00+08:00'})
    return pd.DataFrame(rows)


def main():
    markets=[]; themes=[]; ladders=[]; sources=[]; errors=[]
    for d in DAYS:
        try:
            x=parse_day(d); markets.append(x['market']); themes.extend(x['themes']); ladders.extend(x['ladder']); sources.append(x['source'])
            print(d, 'OK', x['market']['emotion_phase'], len(x['themes']), 'themes', len(x['ladder']), 'ladder rows')
        except Exception as e:
            errors.append((d,repr(e))); print(d,'ERROR',repr(e))
    if errors:
        raise SystemExit('reconstruction errors: '+json.dumps(errors,ensure_ascii=False))
    market_df=pd.DataFrame(markets)
    theme_df=infer_theme_lifecycle(pd.DataFrame(themes))
    ladder_df=pd.DataFrame(ladders)
    role_df=build_roles(market_df,theme_df,ladder_df)
    exp_df=build_expectations(market_df,theme_df)
    src_df=pd.DataFrame(sources)
    market_df.to_csv(OUT/'market_snapshot.csv',index=False,encoding='utf-8-sig')
    theme_df.to_csv(OUT/'theme_snapshot.csv',index=False,encoding='utf-8-sig')
    ladder_df.to_csv(OUT/'ladder_snapshot.csv',index=False,encoding='utf-8-sig')
    role_df.to_csv(OUT/'role_snapshot.csv',index=False,encoding='utf-8-sig')
    exp_df.to_csv(OUT/'expectation_snapshot.csv',index=False,encoding='utf-8-sig')
    src_df.to_csv(OUT/'source_evidence.csv',index=False,encoding='utf-8-sig')
    print('WROTE',len(market_df),len(theme_df),len(ladder_df),len(role_df),len(exp_df),len(src_df))

if __name__=='__main__':
    main()
