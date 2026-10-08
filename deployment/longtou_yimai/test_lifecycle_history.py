# -*- coding: utf-8 -*-
"""Regression against every historical event label; no trading or notification."""
from pathlib import Path
from collections import Counter
import pandas as pd
import longtou_runtime as app

def run():
    d=app.load_daily()
    e=pd.read_parquet(app.SEED/'event_history_to_20260930.parquet')
    e=e.sort_values(['code','date'])
    by_code={code:z.copy() for code,z in d.groupby('code')}
    mismatch=[]
    totals=Counter()
    for r in e.itertuples(index=False):
        code=str(r.code)
        pred,gen=app.lifecycle_identity_at_day1(by_code[code],code,r.date)
        totals['events']+=1
        totals[str(r.event_type)]+=1
        if pred!=r.event_type or int(gen or 0)!=int(r.generation):
            mismatch.append({'code':code,'date':str(r.date),
                             'expected':(r.event_type,r.generation),'actual':(pred,gen)})
    print('LIFECYCLE_ALL_HISTORY',dict(totals),'mismatch_count',len(mismatch),
          'first_mismatches',mismatch[:18])
    assert not mismatch, ('FROZEN_LIFECYCLE_REGRESSION_FAILED', mismatch[:20])
    print('LIFECYCLE_REPLAY_CAUSAL_ALL_EVENTS_PASS')
if __name__=='__main__':run()