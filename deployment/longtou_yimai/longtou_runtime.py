# -*- coding: utf-8 -*-
"""龙头一买: separate production integration, fail-closed, alert-only.

V22 strict causal historical research is NOT a guaranteed profitable model.
This process does not place securities orders or alter existing services/holdings.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, re, subprocess, sys, time
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'state'
SEED=ROOT/'seed'
R50=Path('/opt/leader-relaunch-r50')
sys.path.insert(0,str(R50))
from r50_market import Market
from longtou_core import entry as causal_entry
TZ=ZoneInfo('Asia/Shanghai')
MAIN=re.compile(r'^(600|601|603|605|000|001|002|003)[0-9]{3}$')
FIELDS=['code','name','date','open','high','low','close','shares','amount','eligible','lu','streak','prior_streak','price_reset','close_loc','vol_prev']

def local_now():return dt.datetime.now(TZ)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=str)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def immutable(path,obj):
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        if old!=obj:raise RuntimeError('IMMUTABLE_SOURCE_REVISION:'+str(path))
        return
    atomic(path,obj)

def config():
    c=json.loads((ROOT/'config'/'settings.json').read_text(encoding='utf-8'))
    if c.get('strategy')!='龙头一买' or c.get('version')!='V22_STRICT':
        raise RuntimeError('STRATEGY_VERSION_MISMATCH')
    return c

def freeze_integrity():
    manifest=json.loads((SEED/'manifest.json').read_text())
    for p,v in manifest['files'].items():
        if sha(ROOT/p)!=v['sha256']:raise RuntimeError('SEED_SHA256_CHANGED:'+p)
    c=config()
    if c.get('frozen_entry_sha256')!=sha(ROOT/'longtou_core.py'):
        raise RuntimeError('V22_CORE_HASH_CHANGED')
    return c

def fs_preflight():
    # An ext4 mount displaying stale 'rw' is not evidence of block-device health.
    # The device has to exist, and its root must be readable.
    if not Path('/sys/block/vdb').exists() or not Path('/dev/vdb').is_block_device():
        return False,'DATA_DEVICE_VDB_MISSING'
    try:
        with os.scandir('/data') as entries:
            for i,_ in enumerate(entries):
                if i>=1:break
        os.statvfs('/data')
    except OSError as e:return False,'DATA_MOUNT_IO_FAILURE:'+type(e).__name__
    return True,'DATA_DEVICE_PRESENT_AND_DIRECTORY_READABLE'

def market_calendar():
    p=R50/'trade_calendar.json'
    j=json.loads(p.read_text())
    return sorted(str(z) for z in j['trade_dates'])
def is_open(day):return day in set(market_calendar())
def prev_open(day):
    dates=market_calendar()
    earlier=[x for x in dates if x<day]
    if not earlier:raise RuntimeError('CALENDAR_MISSING_PREV')
    return earlier[-1]

def load_daily():
    seed=pd.read_parquet(SEED/'daily_history_to_20260930.parquet',
                         columns=[x for x in FIELDS if x!='price_reset']+['price_reset'])
    items=[seed]
    for p in sorted((STATE/'daily').glob('*.parquet')):
        z=pd.read_parquet(p)
        missing=set(FIELDS)-set(z.columns)
        if missing:raise RuntimeError('DAILY_COLUMNS_MISSING:'+','.join(sorted(missing)))
        items.append(z[FIELDS].copy())
    d=pd.concat(items,ignore_index=True)
    d['code']=d.code.astype(str).str.zfill(6)
    d['date']=d.date.astype(str)
    if d.duplicated(['code','date']).any():raise RuntimeError('DUPLICATE_DAILY')
    return d

def prior_latest(d,date):
    x=d[d.date<date].sort_values(['code','date'])
    return x.groupby('code',sort=False).tail(1).set_index('code',drop=False)

def today_live_quotes(m,day):
    univ,quoted=m.all_quotes()
    fresh=[]
    stale=0
    for code,q in quoted.items():
        stamp=str(q.server_time)
        if not stamp.startswith(day.replace('-','')) or len(stamp)<14:
            stale+=1;continue
        if stamp[8:12]<'1455':stale+=1;continue
        if not MAIN.fullmatch(code):continue
        if re.search(r'ST|退',str(q.name),re.I):continue
        if min(q.open,q.high,q.low,q.price)<=0:continue
        shares=float(q.volume)*100.
        if shares<=0 or float(q.amount)<=0:continue
        ratio=float(q.amount)/shares/float(q.price)
        if not .70<=ratio<=1.30:continue
        fresh.append(q)
    eligible={code:name for code,name in univ.items()
              if MAIN.fullmatch(str(code)) and not re.search('ST|退',str(name),re.I)}
    coverage=len(fresh)/max(1,len(eligible))
    if len(fresh)<2750 or coverage<.94:
        raise RuntimeError(f'EOD_FEED_STALE_OR_INCOMPLETE:{len(fresh)}/{len(eligible)}')
    return fresh,coverage,stale

def realtime_pm(m,code,day):
    src,z=m.minute_any(code)
    if str(src).replace('-','')!=day.replace('-',''):return None
    if z.empty or not {'time','cum_volume','cum_amount','price'}.issubset(z.columns):
        return None
    if z.time.duplicated().any():return None
    if '09:30' not in set(z.time) or '15:00' not in set(z.time):return None
    z=z.sort_values('time')
    pm=z[z.time>='13:00'].copy()
    if len(pm)<100:return None
    auction=z[z.time=='09:30'].iloc[-1]
    vol=pm.cum_volume.astype(float)-float(auction.cum_volume)
    amt=pm.cum_amount.astype(float)-float(auction.cum_amount)
    if (vol<=0).any() or (amt<=0).any():return None
    vwap=amt/(vol*100.)
    return float((pm.price.astype(float)>=vwap).mean())

def lifecycle_identity_at_day1(d, code, current_day):
    """Reconstruct V17 original FIRST/RECYCLE armed state from PAST bars only.

    A rearmed RECYCLE remains armed even if the 30-day cooldown expires
    *before* its subsequent Day1, unlike a simplistic 'last event <= 31d'.
    The next observed stock day after an earlier Day1 is now historical,
    so this reproduces V17's parent_d2 window without lookahead.
    """
    z=d[(d.code==code)&d.eligible.fillna(False)&(d.date<current_day)].sort_values('date')
    state='DORMANT'
    generation=0
    watch_until=None
    for i,r in enumerate(z.itertuples(index=False)):
        day=pd.Timestamp(r.date)
        lu=bool(r.lu)
        streak=int(r.streak) if pd.notna(r.streak) else 0
        if bool(r.price_reset):
            if state=='PRIMARY_ARMED':
                state='DORMANT';generation=0;watch_until=None
            elif state=='RECYCLE_ARMED':
                state='COOLDOWN'
            continue
        if state=='COOLDOWN' and watch_until is not None and day>watch_until:
            state='DORMANT';generation=0;watch_until=None
        if state=='DORMANT' and lu and streak>=3:
            state='PRIMARY_ARMED';generation=0
        elif state=='COOLDOWN' and watch_until is not None and day<=watch_until and lu and streak>=2:
            state='RECYCLE_ARMED';generation+=1
        if state in ('PRIMARY_ARMED','RECYCLE_ARMED') and not lu:
            need=3 if state=='PRIMARY_ARMED' else 2
            prior=int(r.prior_streak) if pd.notna(r.prior_streak) else 0
            if prior>=need:
                if i+1>=len(z):
                    # We cannot know this earlier Day1's next stock session yet.
                    # A valid new Day1 cannot occur immediately after that broken day.
                    return None,None
                parent_d2=pd.Timestamp(z.iloc[i+1].date)
                watch_until=parent_d2+pd.Timedelta(days=30)
                state='COOLDOWN'
    if state=='PRIMARY_ARMED':
        return 'PRIMARY_3PLUS',generation
    if state=='RECYCLE_ARMED':
        return 'RECYCLE_2PLUS',generation
    return None,None

def classify(day,d,quotes,m):
    previous=prev_open(day)
    if d.date.max()!=previous:raise RuntimeError('HISTORY_GAP_EXPECTED:'+previous+' ACTUAL:'+d.date.max())
    last=prior_latest(d,day)
    events=[];daily=[]
    lookup={q.code:q for q in quotes}
    prev_hot=last[last.streak.astype(int)>=3]
    absent=set(prev_hot.index)-set(lookup)
    if absent:raise RuntimeError('CRITICAL_PREVIOUS_BOARD_QUOTES_MISSING:'+str(len(absent)))
    for code,q in lookup.items():
        if code not in last.index:continue
        prev=last.loc[code]
        if str(prev['date'])!=previous:continue
        pclose=float(prev.close);op=float(q.open);hi=float(q.high);lo=float(q.low)
        end=float(q.price);amt=float(q.amount);shares=float(q.volume)*100.
        if min(pclose,op,hi,lo,end,amt,shares)<=0:continue
        if not lo<=min(op,end)<=max(op,end)<=hi:continue
        if abs(float(q.prev_close)-pclose)>.021:continue
        if abs(op/pclose-1)>.12:continue
        lim=math.floor(pclose*110+.5+1e-8)/100
        if abs(end-lim)<.005 and hi>=lim-.005:
            lu=True
        else:lu=False
        st=(int(prev.streak)+1) if lu else 0
        sr={
          'code':code,'name':str(q.name),'date':day,'open':op,'high':hi,
          'low':lo,'close':end,'shares':shares,'amount':amt,'eligible':True,
          'lu':bool(lu),'streak':st,'prior_streak':int(prev.streak),
          'price_reset':False,'close_loc':((end-lo)/(hi-lo) if hi>lo else None),
          'vol_prev':(shares/float(prev.shares) if float(prev.shares)>0 else None)}
        daily.append(sr)
        if int(prev.streak)<3 or lu:continue
        rel=end/pclose-1;gap=op/pclose-1
        opened_upper=op>=lim-.005
        touched_upper=hi>=lim-.005
        if opened_upper and rel<0:branch='FAILED_OPEN'
        elif rel<=-.08:branch='PANIC'
        elif touched_upper and rel<0:branch='TOUCH_BREAK'
        elif rel<0:branch='NORMAL_RED'
        else:branch='POSITIVE_BREAK'
        kind,gen=lifecycle_identity_at_day1(d,code,day)
        if kind is None:
            raise RuntimeError('LIFECYCLE_STATE_INDETERMINATE:'+code+':'+day)
        pm=None
        if branch=='NORMAL_RED' and st==0 and int(prev.streak)>=4 and kind=='PRIMARY_3PLUS':
            try:pm=realtime_pm(m,code,day)
            except Exception:pm=None
        evt={
          **sr,'event_id':f'{code}_{day.replace("-","")}_g{gen}',
          'event_type':kind,'generation':gen,'prior_streak':int(prev.streak),
          'branch':branch,'bucket':'B3' if int(prev.streak)==3 else 'B4P',
          'gap':gap,'ret':rel,'close_vwap':end/(amt/shares),
          'd1_pm_above_ratio':pm,'source':'tencent_quote+minute',
          'quote_timestamp':str(q.server_time)}
        events.append(evt)
    min_coverage=len(daily)/max(1,len([x for x in last.index if MAIN.fullmatch(x)]))
    if min_coverage<.94:raise RuntimeError('EOD_DAILY_FILTERED_COVERAGE_LOW:'+str(round(min_coverage,4)))
    return daily,events,min_coverage

def quote_stamp(q,day,now):
    stamp=str(q.server_time)
    if not stamp.startswith(day.replace('-','')) or len(stamp)<14:return None
    try:when=dt.datetime.strptime(stamp,'%Y%m%d%H%M%S').replace(tzinfo=TZ)
    except ValueError:return None
    if abs((now-when).total_seconds())>25:return None
    return when

def provider_bars(z,now,opening,lastprice):
    """Only use fully observed historical prefix and LIVE quote reference price.

    Tencent minutes are cumulative, not historical OHLC bars. The representative
    prices are minute CLOSE proxies. The last synthetic OPEN is the live quote
    received at detection; never call it a guaranteed next-minute fill.
    """
    if z is None or z.empty:return None
    req={'time','price','cum_volume','cum_amount'}
    if not req.issubset(z.columns) or z.time.duplicated().any():return None
    z=z.sort_values('time').reset_index(drop=True)
    if (pd.to_numeric(z.cum_volume,errors='coerce').diff().dropna() < 0).any():return None
    if (pd.to_numeric(z.cum_amount,errors='coerce').diff().dropna() < -0.01).any():return None
    if not (z.time=='09:30').any():return None
    auction=z[z.time=='09:30'].iloc[-1]
    if not float(auction.price)>0:return None
    if abs(float(auction.price)/float(opening)-1)>.015:return None
    a_v=float(auction.cum_volume);a_m=float(auction.cum_amount)
    minutes=[]
    for i in range(1,16):
        label=f'09:{30+i:02d}'
        completed=dt.datetime.combine(now.date(),dt.time(9,30+i),TZ)
        if (now-completed).total_seconds()<8:break
        sub=z[z.time==label]
        if len(sub)!=1:break
        curr=sub.iloc[0]
        prev=z[z.time==('09:30' if i==1 else f'09:{29+i:02d}')]
        if len(prev)!=1:break
        before=prev.iloc[0]
        dv=float(curr.cum_volume)-float(before.cum_volume)
        da=float(curr.cum_amount)-float(before.cum_amount)
        if dv<=0 or da<=0:return None
        c=float(curr.price)
        if c<=0:return None
        minute_vwap=da/(dv*100.)
        if not np.isfinite(minute_vwap) or not .75 <= minute_vwap/c <= 1.25:return None
        op=float(opening if i==1 else before.price)
        minutes.append({'time':label,'open':op,'high':max(op,c),'low':min(op,c),
                        'close':c,'volume':dv*100.,'amount':da})
    if not minutes:
        # The 09:30 opening signal must work before the first minute closes.
        # Only the observed opening price is known: never create a fake M1 close.
        if (now.hour,now.minute)!=(9,30):return None
        return pd.DataFrame([{'time':'09:31','open':float(opening),'high':np.nan,
                              'low':np.nan,'close':np.nan,'volume':np.nan,'amount':np.nan}])
    current_idx=len(minutes)
    if current_idx>15:return None
    # Actual quote observed after the most recent fully completed minute.
    # No current-bar high/low/close/volume is accessed.
    next_label=f'09:{31+current_idx:02d}'
    minutes.append({'time':next_label,'open':float(lastprice),'high':np.nan,
                    'low':np.nan,'close':np.nan,'volume':np.nan,'amount':np.nan})
    return pd.DataFrame(minutes)

def selftest():
    from longtou_core import entry
    base=pd.Series(dict(event_id='600100_20261008_g0',name='合成样本',code='600100',
                        date=pd.Timestamp('2026-10-08'),close=10.,amount=2e9,
                        close_loc=.30,close_vwap=.98,ret=-.03,gap=-.02,
                        vol_prev=2.,branch='NORMAL_RED',bucket='B4P',
                        prior_streak=4,event_type='PRIMARY_3PLUS',d2_price_reset=False))
    s=pd.DataFrame([
        dict(time='09:31',open=9.9,high=9.9,low=9.8,close=9.8,volume=1e6,amount=9.8e6),
        dict(time='09:32',open=9.8,high=9.8,low=9.7,close=9.7,volume=1e6,amount=9.7e6),
        dict(time='09:33',open=9.7,high=9.7,low=9.6,close=9.6,volume=1e6,amount=9.6e6),
        dict(time='09:34',open=9.62,high=np.nan,low=np.nan,close=np.nan,volume=np.nan,amount=np.nan),
    ])
    out,why=entry(base,s,None)
    assert out and out['lane']=='NORMAL_B4P_LIQUID_M3' and out['entry_idx']==3
    no_future=s.copy()
    no_future.loc[3,['high','low','close','volume','amount']]=[float('nan')]*5
    assert entry(base,no_future,None)==(out,why)
    assert (SEED/'manifest.json').exists()
    print('LONGTOU_YIMAI_SELFTEST_OK causal M3 and frozen-seed checks')

def postclose(c):
    now=local_now();day=now.date().isoformat()
    if not is_open(day):print('SKIP_NON_TRADING_DAY');return
    if (now.hour,now.minute)<(15,8):raise RuntimeError('NOT_AFTER_CLOSE')
    d=load_daily()
    market=Market(timeout=5)
    try:
        qs,coverage,stale=today_live_quotes(market,day)
        daily,events,valid_coverage=classify(day,d,qs,market)
    finally:market.close()
    snapshot={'day':day,'source':'R50_TENCENT_CROSS_INFRA',
              'quote_coverage':coverage,'daily_coverage':valid_coverage,
              'quote_stale_count':stale,'captured_at':now.isoformat(),
              'events':events}
    path=STATE/'events'/(day+'.json')
    if path.exists():raise RuntimeError('EVENT_DAY_ALREADY_REGISTERED')
    p=STATE/'daily'/(day+'.parquet')
    if p.exists():raise RuntimeError('DAILY_DAY_ALREADY_REGISTERED')
    # Use temp file and atomic rename for self-owned state only.
    p.parent.mkdir(exist_ok=True,parents=True)
    temp=p.with_name(p.name+'.tmp')
    pd.DataFrame(daily,columns=FIELDS).to_parquet(temp,index=False)
    os.replace(temp,p)
    immutable(path,snapshot)
    print('DAY1_RECORDED',day,'quotes',len(qs),'daily',len(daily),'events',len(events),
          'coverage',round(valid_coverage,3))

def notify_signal(c,evt,plan,quote,when,market):
    """Single real notification is possible only after explicit arming and data-health gates."""
    signal_day=when.date().isoformat()
    eid=str(evt['event_id'])
    folder=STATE/'signals';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(eid+'.json')
    if dest.exists():return 'ALREADY_RECORDED'
    # External independent quote protects against one-source data glitch.
    try:
        ref=market.sina_quote(evt['code'])
        price=float(ref.price);prev=float(ref.prev_close)
        if price<=0 or prev<=0 or abs(price/quote.price-1)>.008 or abs(prev/quote.prev_close-1)>.005:
            return 'REJECT_CROSS_SOURCE_MISMATCH'
    except Exception:
        return 'REJECT_CROSS_SOURCE_UNAVAILABLE'
    armed=bool(c.get('real_signals_armed',False))
    channel=os.getenv('OPENCLAW_MESSAGE_CHANNEL','').strip()
    target=os.getenv('OPENCLAW_MESSAGE_TARGET','').strip()
    # Fail before consuming one-time dedupe record if the route is unconfigured.
    if armed and (channel!='feishu' or not target):
        return 'BLOCKED_FEISHU_ROUTE_MISSING'
    payload={
      'strategy':'龙头一买','mode':'REAL_NOTIFICATION' if armed else 'STAGING_NO_MESSAGE',
      'event_id':eid,'day1':evt['date'],'day2':signal_day,
      'code':evt['code'],'name':evt['name'],'branch':evt['branch'],
      'lane':plan['lane'],'reference_price':float(quote.price),
      'source_quote_time':str(quote.server_time),'signal_time':plan['signal_time'],
      'detected_at':when.isoformat(),
      'not_an_executable_fill':True,
      'rule_version':'V22_STRICT_CAUSAL','notification_attempted':armed
    }
    # Record *before* external send so retries cannot generate duplicate orders/messages.
    immutable(dest,payload)
    if not armed:
        return 'RESEARCH_SIGNAL_ONLY'
    msg=(
        '【龙头一买｜Day2买点观察】\n'
        f"{evt['name']} {evt['code']}\n"
        f"Day1: {evt['date']} {evt['branch']} {evt['prior_streak']}连板后分歧\n"
        f"买点分支: {plan['lane']}｜完成信号: {plan['signal_time']}\n"
        f"发现时间: {when:%H:%M:%S}｜实时参考价: {float(quote.price):.2f}元\n"
        f"腾讯/新浪双源核验完成｜不保证此价成交｜仅为研究信号，非自动下单"
    )
    cmd=['openclaw','message','send','--channel',channel,'--target',target,
         '--message',msg,'--json']
    try:
        run=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           timeout=45,check=False)
        outcome='FEISHU_SENT' if run.returncode==0 else 'FEISHU_SEND_FAILED'
    except (OSError,subprocess.TimeoutExpired):
        outcome='FEISHU_SEND_ERROR'
    # A one-time result for operational audits. Do NOT automatically retry
    # unknown delivery states: Telegram/Feishu may have accepted the message.
    audit={'event_id':eid,'attempted_at':local_now().isoformat(),
           'delivery_status':outcome,'at_most_once_policy':True,
           'no_automatic_retry_if_unknown':True}
    atomic(STATE/'delivery'/(eid+'.json'),audit)
    return outcome

def monitor(c):
    now=local_now();day=now.date().isoformat()
    if not is_open(day):print('NO_TRADING_SESSION');return
    ok,why=fs_preflight()
    if not ok:raise RuntimeError('FAIL_CLOSED_STORAGE:'+why)
    seed=load_daily()
    previous=prev_open(day)
    if seed.date.max()!=previous:
        raise RuntimeError('CRITICAL_DAILY_GAP:'+str(seed.date.max())+'!='+previous)
    file=STATE/'events'/(previous+'.json')
    if not file.exists():raise RuntimeError('PRE_REGISTERED_DAY1_MISSING:'+previous)
    obj=json.loads(file.read_text())
    created=dt.datetime.fromisoformat(obj['captured_at'])
    if created>=dt.datetime.combine(now.date(),dt.time(9,25),TZ):
        raise RuntimeError('DAY1_NOT_REGISTERED_PREOPEN')
    events=[r for r in obj.get('events',[])
            if r['branch'] in ('PANIC','NORMAL_RED','POSITIVE_BREAK')
            and r['bucket'] in ('B3','B4P')]
    if len(events)>32:raise RuntimeError('UNBOUNDED_EVENT_COUNT:'+str(len(events)))
    codes=[e['code'] for e in events]
    print('LONGTOU_YIMAI_MONITOR_STARTED',day,'candidates',len(events),
          'armed',c.get('real_signals_armed',False),flush=True)
    if not events:return
    market=Market(timeout=5)
    last_status={}
    try:
        while True:
            at=local_now()
            if at.date().isoformat()!=day or (at.hour,at.minute)>=(9,47):break
            if (at.hour,at.minute)<(9,30):
                time.sleep(4);continue
            ok,why=fs_preflight()
            if not ok:
                print('DATA_HEALTH_LOST_FAIL_CLOSED',why,flush=True)
                break
            try: quotes=market.tencent_quotes(codes)
            except Exception as exc:
                print('LIVE_QUOTE_SOURCE_FAILURE',type(exc).__name__,flush=True)
                time.sleep(12);continue
            for e in events:
                eid=e['event_id'];code=e['code']
                if (STATE/'signals'/(eid+'.json')).exists():continue
                q=quotes.get(code)
                if q is None or quote_stamp(q,day,at) is None:
                    last_status[eid]='STALE_OR_MISSING_QUOTE';continue
                if abs(float(q.prev_close)-float(e['close']))>.021:
                    last_status[eid]='D2_PREVCLOSE_MISMATCH';continue
                if min(float(q.price),float(q.open),float(q.high),float(q.low))<=0:
                    last_status[eid]='INVALID_QUOTE_OHLC';continue
                try:
                    sourceday,z=market.minute_any(code)
                    if str(sourceday).replace('-','')!=day.replace('-',''):
                        last_status[eid]='MINUTES_WRONG_SESSION';continue
                    bars=provider_bars(z,at,float(q.open),float(q.price))
                    if bars is None:last_status[eid]='MINUTES_INCOMPLETE';continue
                    row=pd.Series(dict(e, date=pd.Timestamp(e['date']),d2_price_reset=False))
                    plan,reason=causal_entry(row,bars,e.get('d1_pm_above_ratio'))
                    if plan is None:
                        last_status[eid]=reason;continue
                    signal_time=str(plan['signal_time'])
                    if signal_time!='09:25':
                        s_dt=dt.datetime.combine(at.date(),dt.time.fromisoformat(signal_time),TZ)
                        if (at-s_dt).total_seconds()>55:
                            last_status[eid]='STALE_CATCHUP_SIGNAL_SKIPPED';continue
                    else:
                        if (at.hour,at.minute)>(9,31):
                            last_status[eid]='LATE_OPEN_SIGNAL_SKIPPED';continue
                    result=notify_signal(c,e,plan,q,at,market)
                    last_status[eid]=result
                    print('BUY_POINT_RESULT',code,plan['lane'],result,flush=True)
                except Exception as exc:
                    last_status[eid]='SOURCE_ERROR:'+type(exc).__name__
            atomic(STATE/'last_monitor_status.json',{
                'day':day,'at':at.isoformat(),'armed':c.get('real_signals_armed',False),
                'last_status':last_status})
            time.sleep(8)
    finally:market.close()
    print('LONGTOU_YIMAI_MONITOR_ENDED',day,'signals_recorded',
          len(list((STATE/'signals').glob('*.json'))) if (STATE/'signals').exists() else 0,
          flush=True)

def health(c):
    ok,why=fs_preflight()
    info={'strategy':'龙头一买','checked_at':local_now().isoformat(),
          'storage_ok':ok,'storage_reason':why,'signal_armed':bool(c.get('real_signals_armed')),
          'calendar_source':str(R50/'trade_calendar.json'),
          'historical_seed':'2026-09-30','current_data_day':None,
          'notification_route_present':bool(os.getenv('OPENCLAW_MESSAGE_TARGET')) and
                                     os.getenv('OPENCLAW_MESSAGE_CHANNEL')=='feishu'}
    try:
        h=load_daily()
        info['current_data_day']=h.date.max()
        info['code_count']=int(h.code.nunique())
        info['history_rows']=int(len(h))
    except Exception as e:info['history_error']=type(e).__name__
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['selftest','preflight','postclose','watch'])
    a=p.parse_args()
    if a.command=='selftest':selftest();return
    c=freeze_integrity()
    if a.command=='preflight':
        h=health(c)
        why_not=[]
        if not h['storage_ok']:why_not.append(h['storage_reason'])
        if not h['notification_route_present']:why_not.append('NOTIFICATION_ROUTE_MISSING')
        if not h['signal_armed']:why_not.append('SIGNAL_ARMING_DISABLED')
        previous=prev_open(local_now().date().isoformat())
        if h.get('current_data_day')!=previous:
            why_not.append('DAY1_HISTORY_NOT_TO_PREV_SESSION:'+str(previous))
        day1=STATE/'events'/(previous+'.json')
        if not day1.exists():why_not.append('DAY1_IMMUTABLE_EOD_SNAPSHOT_MISSING')
        h['ready_for_live_signal']=not why_not
        h['readiness_blockers']=why_not
        print(json.dumps(h,ensure_ascii=False,indent=2))
        if why_not:sys.exit(4)
        return
    okay,why=fs_preflight()
    if not okay:raise RuntimeError('FAIL_CLOSED_STORAGE:'+why)
    if a.command=='postclose':postclose(c)
    if a.command=='watch':monitor(c)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('LONGTOU_YIMAI_FAIL_CLOSED',type(exc).__name__,str(exc)[:270],file=sys.stderr)
        sys.exit(4)