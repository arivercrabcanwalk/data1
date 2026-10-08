# -*- coding: utf-8 -*-
"""Independent postclose diagnostic capture during /data incident.

NEVER creates a production Day1 registration, never executes V22 buy signals,
never reads Feishu credentials and never sends messages.
"""
from __future__ import annotations
import datetime as dt,hashlib,json,os
from pathlib import Path
import pandas as pd
import longtou_runtime as core

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run():
    now=core.local_now()
    day=now.date().isoformat()
    if (now.hour,now.minute)<(15,12):raise RuntimeError('WAIT_AFTER_15_12')
    if not core.is_open(day):raise RuntimeError('NOT_TRADING_DAY')
    c=core.freeze_integrity()
    if c.get('real_signals_armed'):raise RuntimeError('QUARANTINE_MUST_REMAIN_DISARMED')
    d=core.load_daily()
    out=core.ROOT/'quarantine'/day
    if out.exists():raise RuntimeError('QUARANTINE_CAPTURE_ALREADY_EXISTS_NO_OVERWRITE')
    market=core.Market(timeout=5)
    try:
        quote,coverage,stale=core.today_live_quotes(market,day)
        daily,events,dcov=core.classify(day,d,quote,market)
        src={q.code:q for q in quote}
        dual=[]
        for ev in events:
            q=src[ev['code']]
            item={'event_id':ev['event_id'],'code':ev['code'],
                  'tencent_quote_time':str(q.server_time),
                  'verified':False,'reason':None}
            try:
                other=market.sina_quote(ev['code'])
                pp=float(other.price);pprev=float(other.prev_close)
                item['verified']=(pp>0 and pprev>0 and
                    abs(pp/float(q.price)-1)<.008 and
                    abs(pprev/float(q.prev_close)-1)<.005)
                item['reason']='MATCH' if item['verified'] else 'PRICE_DISAGREEMENT'
            except Exception as err:
                item['reason']='SECONDARY_SOURCE_'+type(err).__name__
            dual.append(item)
    finally:market.close()
    # All records are observational. This path is NOT loaded by normal monitor.
    os.umask(0o077)
    out.mkdir(parents=True,mode=0o700,exist_ok=False)
    ds=out/'observed_daily.parquet'
    pd.DataFrame(daily,columns=core.FIELDS).to_parquet(ds,index=False,compression='zstd')
    es=out/'observed_candidates.json'
    core.atomic(es,{'date':day,'source':'TENCENT_EOD_DIAGNOSTIC_ONLY',
                     'observed_at':core.local_now().isoformat(),
                     'not_valid_for_trading':True,'never_registered_as_day1':True,
                     'events':events})
    manifest={'observed_at':core.local_now().isoformat(),'day':day,
              'purpose':'INCIDENT_QUARANTINE_NONTRADING_RESEARCH_ONLY',
              'production_registration':False,'data_disk_health':core.fs_preflight()[1],
              'quotes':len(quote),'quote_coverage':coverage,'stale_quote_count':stale,
              'daily_rows':len(daily),'daily_coverage':dcov,
              'candidate_count':len(events),'candidate_cross_checks':dual,
              'files':{p.name:sha(p) for p in (ds,es)},
              'no_feishu_sends':True,'no_live_signals':True}
    core.atomic(out/'manifest.json',manifest)
    print(json.dumps({'quarantine_saved':str(out),'candidate_count':len(events),
                      'cross_verified':sum(x['verified'] for x in dual),
                      'quote_coverage':coverage,'daily_coverage':dcov,
                      'not_live_registered':True},ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:run()
    except Exception as err:
        print('QUARANTINE_CAPTURE_FAIL',type(err).__name__,str(err)[:200])
        raise