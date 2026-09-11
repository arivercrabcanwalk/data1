from pathlib import Path
import json, math
import pandas as pd

HERE=Path(__file__).resolve().parent; PIT=HERE/'pit_data'
DAYS=['2026-07-01','2026-07-02','2026-07-03','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10','2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22','2026-07-23','2026-07-24','2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31']
REQ=['market_snapshot.csv','theme_snapshot.csv','ladder_snapshot.csv','role_snapshot.csv','expectation_snapshot.csv','source_evidence.csv','member_name_code.csv','build_meta.json']
issues=[]
for f in REQ:
    if not (PIT/f).exists(): issues.append('missing '+f)
if issues:
    print('NOT READY'); [print('-',x) for x in issues]; raise SystemExit(2)

m=pd.read_csv(PIT/'market_snapshot.csv',dtype={'market_dragon_code':str}).fillna('')
t=pd.read_csv(PIT/'theme_snapshot.csv',dtype={'leader_code':str}).fillna('')
l=pd.read_csv(PIT/'ladder_snapshot.csv',dtype={'code':str}).fillna('')
r=pd.read_csv(PIT/'role_snapshot.csv',dtype={'code':str}).fillna('')
e=pd.read_csv(PIT/'expectation_snapshot.csv').fillna('')
s=pd.read_csv(PIT/'source_evidence.csv').fillna('')
meta=json.loads((PIT/'build_meta.json').read_text(encoding='utf-8'))

for name,df in [('market',m),('theme',t),('ladder',l),('role',r),('expectation',e),('source',s)]:
    got=set(df.date.astype(str)); miss=[d for d in DAYS if d not in got]
    if miss: issues.append(f'{name} missing dates: {miss}')
if len(m)!=23: issues.append(f'market rows {len(m)} != 23')
if float(meta.get('resolution_ratio',0))<.90: issues.append(f"name resolution {meta.get('resolution_ratio')} < 90%")
if len(l)<520: issues.append(f'ladder/member rows {len(l)} < 520')
if l.code.astype(str).nunique()<420: issues.append(f'distinct mapped member codes {l.code.astype(str).nunique()} < 420')
if len(r)<70: issues.append(f'role rows {len(r)} < 70')

for d in DAYS:
    td=t[t.date==d]; ld=l[l.date==d]; rd=r[r.date==d]; ed=e[e.date==d]; sd=s[s.date==d]
    if len(td)<2: issues.append(f'{d}: fewer than 2 audited themes')
    if len(rd)<2: issues.append(f'{d}: fewer than 2 resolved roles')
    if len(ed)!=1: issues.append(f'{d}: expectation rows={len(ed)}')
    if len(sd)!=1: issues.append(f'{d}: source rows={len(sd)}')
    for x in td.itertuples(index=False):
        count=int(float(x.source_limit_up_count)); mapped=int(float(x.mapped_member_count)); need=max(2,math.ceil(count*.30))
        if mapped<need: issues.append(f'{d}/{x.theme}: mapped {mapped}/{count}, need >= {need}')
        if mapped>count: issues.append(f'{d}/{x.theme}: mapped exceeds source count')
    if len(sd):
        if str(sd.iloc[0].pit_ok).lower() not in ('true','1'): issues.append(f'{d}: pit_ok false')
        if str(sd.iloc[0].future_sections_excluded).lower() not in ('true','1'): issues.append(f'{d}: future sections not excluded')
        if d not in str(sd.iloc[0].source_url): issues.append(f'{d}: source URL date mismatch')

bad=' '.join(s.captured_fact.astype(str).tolist())
for forbidden in ['同景日期','次日上涨概率','前瞻回验','未来收益']:
    if forbidden in bad: issues.append('future-stat evidence leaked: '+forbidden)

if issues:
    print('NOT READY')
    for x in issues: print('-',x)
    raise SystemExit(2)
print('READY FORMAL JULY PIT')
print(json.dumps({'days':23,'mapped_member_rows':len(l),'distinct_codes':int(l.code.nunique()),'roles':len(r),'resolution_ratio':meta['resolution_ratio']},ensure_ascii=False))
