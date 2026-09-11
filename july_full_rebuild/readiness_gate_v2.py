from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
PIT=ROOT/'pit_data'
DAYS=['2026-07-01','2026-07-02','2026-07-03','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10','2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22','2026-07-23','2026-07-24','2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31']
REQ=['market_snapshot.csv','theme_snapshot.csv','ladder_snapshot.csv','role_snapshot.csv','expectation_snapshot.csv','source_evidence.csv']
issues=[]
for f in REQ:
    p=PIT/f
    if not p.exists(): issues.append('missing '+f)
for f in REQ:
    p=PIT/f
    if not p.exists(): continue
    df=pd.read_csv(p,dtype={'code':str,'leader_code':str,'market_dragon_code':str}).fillna('')
    if 'date' not in df: issues.append(f+' no date'); continue
    got=set(df.date.astype(str)); miss=[d for d in DAYS if d not in got]
    if miss: issues.append(f+' missing dates '+','.join(miss))

if not issues:
    m=pd.read_csv(PIT/'market_snapshot.csv').fillna('')
    t=pd.read_csv(PIT/'theme_snapshot.csv',dtype={'leader_code':str}).fillna('')
    l=pd.read_csv(PIT/'ladder_snapshot.csv',dtype={'code':str}).fillna('')
    r=pd.read_csv(PIT/'role_snapshot.csv',dtype={'code':str}).fillna('')
    e=pd.read_csv(PIT/'expectation_snapshot.csv').fillna('')
    s=pd.read_csv(PIT/'source_evidence.csv').fillna('')
    for d in DAYS:
        td=t[t.date==d]; ld=l[l.date==d]; rd=r[r.date==d]; ed=e[e.date==d]; sd=s[s.date==d]
        if len(td)<2: issues.append(f'{d}: fewer than 2 factual themes')
        if len(ld)<3: issues.append(f'{d}: fewer than 3 resolved theme members')
        if len(rd)<1: issues.append(f'{d}: no resolved market/theme role')
        if len(ed)<1: issues.append(f'{d}: no next-session expectation')
        if len(sd)<1 or not all(str(x).lower() in ('true','1') for x in sd.pit_ok): issues.append(f'{d}: source PIT not verified')
        if 'member_coverage' in td:
            meaningful=td[pd.to_numeric(td.limit_up_count,errors='coerce').fillna(0)>=5]
            if len(meaningful) and (pd.to_numeric(meaningful.member_coverage,errors='coerce').fillna(0)<0.30).any(): issues.append(f'{d}: meaningful theme member coverage below 30%')
    bad=' '.join(s.captured_fact.astype(str).tolist())
    if '同景日期' in bad or '次日上涨概率' in bad or '前瞻回验' in bad: issues.append('future-stat text leaked into evidence')
    # Full-member floor prevents the earlier leader-only PIT from passing.
    if len(l)<180: issues.append(f'ladder too sparse for member-level confirmation: {len(l)} rows < 180')
    if len(set(l.code.astype(str)))<80: issues.append('too few distinct resolved stocks for theme confirmation')

if issues:
    print('NOT READY')
    for x in issues: print('-',x)
    raise SystemExit(2)
print('READY: 23 trading days, six PIT layers, member-level theme coverage and no future-stat evidence passed.')
