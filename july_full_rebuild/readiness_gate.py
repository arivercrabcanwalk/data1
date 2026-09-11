from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRADING_DAYS = [
    '2026-07-01','2026-07-02','2026-07-03','2026-07-06','2026-07-07','2026-07-08','2026-07-09','2026-07-10',
    '2026-07-13','2026-07-14','2026-07-15','2026-07-16','2026-07-17','2026-07-20','2026-07-21','2026-07-22',
    '2026-07-23','2026-07-24','2026-07-27','2026-07-28','2026-07-29','2026-07-30','2026-07-31'
]
REQUIRED = ['market_snapshot.csv','theme_snapshot.csv','ladder_snapshot.csv','role_snapshot.csv','expectation_snapshot.csv','source_evidence.csv']
KEYS = {
    'market_snapshot.csv':['date','emotion_phase'],
    'theme_snapshot.csv':['date','theme','theme_phase'],
    'ladder_snapshot.csv':['date','code','theme'],
    'role_snapshot.csv':['date','code','theme','role'],
    'expectation_snapshot.csv':['date','target_date','scenario','created_at'],
    'source_evidence.csv':['date','source_url','available_at','captured_fact','pit_ok'],
}

def fail(msg):
    print('NOT_READY:', msg)
    raise SystemExit(2)

def main():
    data = ROOT/'pit_data'
    if not data.exists(): fail('pit_data directory missing')
    tables={}
    for name in REQUIRED:
        p=data/name
        if not p.exists(): fail(f'{name} missing')
        df=pd.read_csv(p,dtype=str).fillna('')
        missing=[c for c in KEYS[name] if c not in df.columns]
        if missing: fail(f'{name} missing columns {missing}')
        tables[name]=df
    for day in TRADING_DAYS:
        for name,df in tables.items():
            if day not in set(df['date']): fail(f'{day} missing {name}')
    ev=tables['source_evidence.csv']
    if not ev['pit_ok'].str.lower().isin(['true','1','yes']).all(): fail('source_evidence contains non-PIT evidence')
    exp=tables['expectation_snapshot.csv']
    for r in exp.itertuples(index=False):
        if r.date!='2026-07-01' and r.created_at[:10] > r.date: fail(f'late expectation record for {r.date}')
    print('READY: all 23 trading days have all six PIT layers; formal backtest may start.')

if __name__=='__main__': main()
