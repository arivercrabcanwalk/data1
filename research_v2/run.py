"""Run the frozen 28-arm study on the pinned, real GitHub Parquet package."""
from __future__ import annotations
import argparse
import hashlib
import os
import platform
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from core import SOURCE_COMMIT, Market, Portfolio, configurations, make_candidates, save_json
from data_io import audit_and_daily, day_arrays, read_minute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('research_v2_results'))
    parser.add_argument('--start',default='2026-06-01')
    parser.add_argument('--end',default='2026-08-31')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start_clock=time.monotonic()
    cfgs=configurations()
    save_json(args.out/'frozen_experiments.json',[asdict(c) for c in cfgs])
    save_json(args.out/'execution_status.json',{'status':'started','source_commit':SOURCE_COMMIT})
    try:
        daily,fmap,audit=audit_and_daily(args.root,args.out)
        m=Market(daily)
        candidates=make_candidates(m)
        records=[dict(asdict(c),date=m.dates[t],anchor=m.dates[c.anchor_day])
                 for t,cs in candidates.items() for c in cs]
        pd.DataFrame(records).to_csv(args.out/'all_candidates.csv.gz',index=False,compression='gzip')
        audit['candidate_counts_all_history']=dict(Counter(c['method'] for c in records))
        audit['bad_gap_stock_days_over_20pct']=int(m.bad_gap.sum())
        audit['big_candles']=int(m.big.sum())
        audit['st_unknown_stock_days']=int(daily.st_known.isna().sum())
        audit['current_name_st_approximations_used']=0
        save_json(args.out/'audit.json',audit)
        ports=[Portfolio(c) for c in cfgs]
        eval_days=[t for t,d in enumerate(m.dates) if args.start<=d<=args.end]
        if not eval_days or eval_days[0]<60: raise ValueError('At least 60 prior observations required')
        print('CANDIDATES',len(records),'EVAL_DAYS',len(eval_days),flush=True)
        references=[]
        for step,t in enumerate(eval_days):
            cs=candidates.get(t,[])
            codes={c.code for c in cs}
            codes.update(code for p in ports for code in p.positions)
            arrays=day_arrays(read_minute(fmap[m.dates[t]],codes)) if codes else {}
            for p in ports: p.day(t,m,cs,arrays)
            r=m.ret[t]
            references.append(dict(date=m.dates[t],breadth=float(m.breadth[t]),
                raw_mean_return=float(np.nanmean(r)),raw_median_return=float(np.nanmedian(r)),
                observations=int(np.isfinite(r).sum())))
            if (step+1)%10==0 or step+1==len(eval_days):
                print('REPLAY',step+1,len(eval_days),m.dates[t],flush=True)
        summaries=[p.finish(args.out) for p in ports]
        save_json(args.out/'summary.json',summaries)
        flat=[]
        for s in summaries:
            r={k:v for k,v in s.items() if not isinstance(v,(dict,list))}
            r.update({'return_'+k:v for k,v in s['months'].items()})
            flat.append(r)
            print('RESULT',s['name'],'return',round(s['total_return'],8),
                  'mdd',round(s['daily_max_drawdown'],8),'intraday_mdd',round(s['minute_max_drawdown'],8),
                  'closed',s['closed_trades'],'open',s['open_positions'],'months',s['months'],flush=True)
        pd.DataFrame(flat).to_csv(args.out/'comparison.csv',index=False)
        pd.DataFrame(references).to_csv(args.out/'market_descriptive_reference.csv',index=False)
        events=[]
        for p in ports:
            for r in p.opportunities: events.append(dict(variant=p.cfg.name,**r))
        e=pd.DataFrame(events)
        if len(e):
            event_summary=e.groupby(['variant','method','code','anchor'],sort=False).agg(
                monitored_days=('date','size'),first_monitored=('date','min'),
                filled=('status',lambda z:int((z=='filled').any()))).reset_index()
            event_summary.to_csv(args.out/'event_opportunity_denominators.csv.gz',index=False,compression='gzip')
        program=args.out/'program';program.mkdir(exist_ok=True)
        hashes={}
        for name in ['core.py','data_io.py','run.py','test_core.py','test_data_io.py']:
            src=Path(__file__).with_name(name)
            shutil.copy2(src,program/name)
            hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
        (program/'requirements.txt').write_text('numpy==2.2.6\npandas==2.2.3\npyarrow==19.0.1\npytest==8.3.5\n')
        import pyarrow
        manifest=dict(status='completed',source_commit=SOURCE_COMMIT,
            run_commit=os.getenv('GITHUB_SHA','local'),run_id=os.getenv('GITHUB_RUN_ID'),
            python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyarrow=pyarrow.__version__,
            start=args.start,end=args.end,eval_days=len(eval_days),variants=len(cfgs),source_sha256=hashes,
            runtime_seconds=round(time.monotonic()-start_clock,2),primary_model='MAIN_CLOSE',
            optimization='none; no performance-based parameter refit',
            scope='technical proxy; unadjusted price return; not complete author industrial research system',
            validation='historical replay, not genuinely unseen out-of-sample',
            limits=['ST updates after 2026-07-30 unavailable; last historical status carried forward',
                    '13:00 source records quarantined from intraday arrays; missing minutes never invented',
                    'no corporate-action accounting or adjusted total returns',
                    'no verified daily limit prices or order queue',
                    'point-in-time completeness of supplied universe unverified',
                    'retrospective supplied methods; no performance guarantee'])
        save_json(args.out/'execution_status.json',manifest)
        print('COMPLETED',manifest,flush=True)
    except Exception as error:
        save_json(args.out/'execution_status.json',{'status':'failed','source_commit':SOURCE_COMMIT,
            'run_commit':os.getenv('GITHUB_SHA','local'),'error':repr(error)})
        raise


if __name__=='__main__':
    main()
