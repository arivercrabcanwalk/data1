from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / '2026-07'
PIT = HERE / 'pit_data'
OUT = HERE / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START = 1_000_000.0
LOT = 100
COMMISSION = 0.0003
STAMP = 0.0005
SLIP = 0.0005
MAX_POS = 2
CHECKS = ['09:35','09:45','10:00']
ENTRY_TIMES = ['09:45','10:00']


def norm_code(s):
    x=s.astype(str).str.replace(r'\.0$','',regex=True)
    y=x.str.extract(r'(\d{6})',expand=False)
    return y.fillna(x)


def next_minute(t):
    h,m=map(int,t.split(':')); m+=1
    if m>=60: h+=1; m-=60
    return f'{h:02d}:{m:02d}'


def pct(v):
    try:
        v=float(v)
        if np.isnan(v): return np.nan
        return abs(v/100 if abs(v)>1 else v)
    except Exception:
        return np.nan


class Engine:
    def __init__(self):
        self.cash=START
        self.pos={}
        self.prev_close={}
        self.trades=[]
        self.fills=[]
        self.daily=[]
        self.review_log=[]
        self.mode_weights=defaultdict(lambda:1.0)
        self.mode_pause_until={}
        self.peak=START
        self.risk_state='normal'
        self.recovery_positive_days=0
        self.day_index={}

        self.market=pd.read_csv(PIT/'market_snapshot.csv',dtype={'market_dragon_code':str}).fillna('')
        self.theme=pd.read_csv(PIT/'theme_snapshot.csv',dtype={'leader_code':str}).fillna('')
        self.ladder=pd.read_csv(PIT/'ladder_snapshot.csv',dtype={'code':str}).fillna('')
        self.roles=pd.read_csv(PIT/'role_snapshot.csv',dtype={'code':str}).fillna('')
        self.expect=pd.read_csv(PIT/'expectation_snapshot.csv').fillna('')
        for df in [self.ladder,self.roles]:
            if 'code' in df: df['code']=norm_code(df['code'])
        self.days=sorted(self.market.date.unique().tolist())
        self.day_index={d:i for i,d in enumerate(self.days)}

    def paths(self):
        ps=list(DATA.glob('part-*/date=*/minute1.parquet'))
        return sorted(ps,key=lambda p:p.parent.name.split('=',1)[1])

    def load_day(self,path):
        date=path.parent.name.split('=',1)[1]
        df=pd.read_parquet(path,columns=['code','datetime','open','high','low','close','volume','amount'])
        df['code']=norm_code(df.code); df['datetime']=pd.to_datetime(df.datetime)
        for c in ['open','high','low','close','volume','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=['open','high','low','close']).sort_values(['code','datetime'])
        df['minute']=df.datetime.dt.strftime('%H:%M')
        df['cum_amount']=df.groupby('code').amount.cumsum()
        df['run_high']=df.groupby('code').high.cummax(); df['run_low']=df.groupby('code').low.cummin()
        d=df.groupby('code').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),amount=('amount','sum')).reset_index()
        d['prev_close']=d.code.map(self.prev_close)
        d['ret']=np.where(d.prev_close.notna()&(d.prev_close>0),d.close/d.prev_close-1,np.nan)
        stp=path.parent/'daily_stock_status.parquet'
        st=pd.read_parquet(stp) if stp.exists() else pd.DataFrame({'code':d.code})
        st['code']=norm_code(st.code)
        keep=[c for c in ['code','is_st','is_star_st','is_new_listing_initial','is_suspended','price_limit_up_pct','price_limit_down_pct'] if c in st.columns]
        d=d.merge(st[keep],on='code',how='left')
        for c in ['is_st','is_star_st','is_new_listing_initial']:
            if c not in d: d[c]=False
        d['eligible']=~(d.is_st.fillna(False).astype(bool)|d.is_star_st.fillna(False).astype(bool)|d.is_new_listing_initial.fillna(False).astype(bool))
        if 'is_suspended' in d:
            # Null suspension status is uncertain: exclude rather than assume tradable.
            d['eligible'] &= d.is_suspended.fillna(True).eq(False)
        snaps={}
        for t in CHECKS+['09:30','10:01','14:30','15:00']:
            z=df[df.minute==t].copy()
            z['prev_close']=z.code.map(self.prev_close)
            z['ret_now']=np.where(z.prev_close.notna()&(z.prev_close>0),z.close/z.prev_close-1,np.nan)
            z['range_pos']=((z.close-z.run_low)/(z.run_high-z.run_low).replace(0,np.nan)).fillna(.5).clip(0,1)
            snaps[t]=z
        return date,df,d.set_index('code',drop=False),snaps

    def row_at(self,df,code,t):
        x=df[(df.code==code)&(df.minute==t)]
        return None if x.empty else x.iloc[0]

    def market_breadth(self,snap):
        r=snap.ret_now.replace([np.inf,-np.inf],np.nan).dropna()
        return float((r>0).mean()) if len(r) else np.nan

    def theme_breadth(self,date,theme,snap):
        mem=self.ladder[(self.ladder.date==date)&(self.ladder.theme==theme)].code
        mem=set(x for x in mem.astype(str) if len(x)==6)
        if len(mem)<2: return np.nan
        z=snap[snap.code.isin(mem)].ret_now.dropna()
        return float((z>0).mean()) if len(z)>=2 else np.nan

    def prior_day(self,date):
        i=self.day_index.get(date,0)
        return self.days[i-1] if i>0 else None

    def equity(self,close_map):
        return self.cash+sum(p['shares']*close_map.get(c,p['entry_price']) for c,p in self.pos.items())

    def can_buy(self,rr,raw):
        if rr is None or not bool(rr.eligible) or not np.isfinite(raw) or raw<=0: return False
        p=pct(rr.get('price_limit_up_pct',np.nan)); pc=rr.prev_close
        if np.isfinite(p) and np.isfinite(pc) and pc>0 and raw>=pc*(1+p)*0.998: return False
        return True

    def can_sell(self,rr,raw):
        if rr is None or not np.isfinite(raw) or raw<=0: return False
        p=pct(rr.get('price_limit_down_pct',np.nan)); pc=rr.prev_close
        if np.isfinite(p) and np.isfinite(pc) and pc>0 and raw<=pc*(1-p)*1.002: return False
        return True

    def mode_weight(self,mode,date):
        until=self.mode_pause_until.get(mode,-1)
        if self.day_index[date] <= until: return 0.0
        return float(self.mode_weights[mode])

    def total_cap(self,prior_phase):
        base={'ice':.25,'repair':.55,'warming':.65,'climax':.35,'cooling':.25,'retreat':.15}.get(prior_phase,0)
        if self.risk_state=='defensive': base=min(base,.25)
        return base

    def buy(self,date,code,mode,raw,tm,rr,equity,cap,role,theme,phase,tbreadth,mbreadth,prior_phase):
        w=self.mode_weight(mode,date)
        if w<=0 or code in self.pos or len(self.pos)>=MAX_POS or not self.can_buy(rr,raw): return False
        current=sum(p['shares']*p['entry_price'] for p in self.pos.values())/max(equity,1)
        fraction=min(.22*w,cap-current)
        if fraction<.07: return False
        px=raw*(1+SLIP); budget=min(self.cash*.97,equity*fraction)
        sh=int(budget/px/LOT)*LOT
        if sh<LOT:return False
        gross=sh*px; fee=gross*COMMISSION
        if gross+fee>self.cash:return False
        self.cash-=gross+fee
        self.pos[code]={'code':code,'entry_date':date,'entry_time':tm,'entry_price':px,'shares':sh,'basis':gross+fee,
                        'mode':mode,'role':role,'theme':theme,'prior_phase':prior_phase,'theme_breadth':tbreadth,
                        'market_breadth':mbreadth,'hold_days':0,'max_price':px,'min_price':px}
        self.fills.append({'date':date,'time':tm,'code':code,'side':'BUY','price':px,'shares':sh,'mode':mode,'theme':theme,'role':role})
        return True

    def sell(self,date,code,raw,tm,reason,rr,max_seen=None,min_seen=None):
        if code not in self.pos or not self.can_sell(rr,raw): return False
        p=self.pos[code]
        if max_seen is not None:p['max_price']=max(p['max_price'],max_seen)
        if min_seen is not None:p['min_price']=min(p['min_price'],min_seen)
        px=raw*(1-SLIP); gross=p['shares']*px; fee=gross*(COMMISSION+STAMP)
        self.cash+=gross-fee; pnl=gross-fee-p['basis']; ret=pnl/p['basis']
        tags=[]
        if ret<0 and p['prior_phase']=='climax': tags.append('LATE_CYCLE_ENTRY')
        if ret<0 and (not np.isfinite(p['theme_breadth']) or p['theme_breadth']<.5): tags.append('NO_THEME_SUPPORT')
        if ret<0 and p['role'] not in ('market_dragon','theme_dragon','capacity_core','elastic_core'): tags.append('WRONG_ROLE')
        self.trades.append({**p,'exit_date':date,'exit_time':tm,'exit_price':px,'pnl':pnl,'net_return':ret,
                            'exit_reason':reason,'mfe':p['max_price']/p['entry_price']-1,'mae':p['min_price']/p['entry_price']-1,
                            'error_tags':'|'.join(tags)})
        self.fills.append({'date':date,'time':tm,'code':code,'side':'SELL','price':px,'shares':p['shares'],'mode':p['mode'],'theme':p['theme'],'role':p['role'],'reason':reason})
        del self.pos[code]; return True

    def update_review(self,date,day_return):
        notes=[]
        modes=sorted(set(t['mode'] for t in self.trades))
        for mode in modes:
            ts=[t for t in self.trades if t['mode']==mode]
            recent=ts[-5:]
            consec=0
            for t in reversed(ts):
                if t['net_return']<0: consec+=1
                else: break
            old=self.mode_weights[mode]
            if consec>=3:
                self.mode_weights[mode]=0.0
                self.mode_pause_until[mode]=min(len(self.days)-1,self.day_index[date]+2)
                notes.append(f'{mode}:3连亏->暂停2交易日')
            elif consec>=2:
                self.mode_weights[mode]=0.5; notes.append(f'{mode}:2连亏->半权重')
            elif len(recent)>=4:
                avg=float(np.mean([x['net_return'] for x in recent])); win=float(np.mean([x['net_return']>0 for x in recent]))
                if avg>0.02 and win>=0.5:self.mode_weights[mode]=min(1.2,max(1.0,old)); notes.append(f'{mode}:跨样本正期望->1.2x上限')
                elif avg<-.01:self.mode_weights[mode]=.65; notes.append(f'{mode}:近期负期望->0.65x')
                else:self.mode_weights[mode]=1.0
            elif self.day_index[date] > self.mode_pause_until.get(mode,-1) and old==0:
                self.mode_weights[mode]=.5
        if day_return>0:self.recovery_positive_days+=1
        else:self.recovery_positive_days=0
        current_eq=self.daily[-1]['equity'] if self.daily else START
        dd=current_eq/self.peak-1
        if dd<=-.05 and self.risk_state!='defensive': self.risk_state='defensive'; notes.append('组合回撤<=-5%->防守态')
        if self.risk_state=='defensive' and self.recovery_positive_days>=2:
            self.risk_state='normal'; notes.append('连续2个正收益交易日->恢复正常风险态')
        return ';'.join(notes) if notes else '新增样本不足，不修改规则'

    def choose_mode(self,phase,role,theme_phase,mb35,mb_now,tb,ret_now,gap,range_pos):
        if not np.isfinite(mb_now) or not np.isfinite(ret_now) or not np.isfinite(gap): return None
        core=role in ('market_dragon','theme_dragon','capacity_core','elastic_core')
        if not core:return None
        if gap>.06 or ret_now>.085:return None
        if phase=='ice':
            if mb_now>=.55 and np.isfinite(mb35) and mb_now-mb35>=.08 and (not np.isfinite(tb) or tb>=.50) and .008<=ret_now<=.07 and range_pos>=.62:
                return 'ice_repair_pioneer'
            return None
        if phase=='climax':
            if -.04<=gap<=.025 and .006<=ret_now<=.055 and range_pos>=.68 and mb_now>=.45 and (not np.isfinite(tb) or tb>=.50):
                return 'core_first_divergence_to_consensus'
            return None
        if phase in ('repair','warming'):
            if -.025<=gap<=.04 and .012<=ret_now<=.07 and range_pos>=.65 and mb_now>=.50 and (not np.isfinite(tb) or tb>=.52):
                return 'dragon_mainrise' if role in ('market_dragon','theme_dragon') and theme_phase in ('confirm','ferment','climax','repair') else 'core_weak_to_strong'
            return None
        if phase in ('cooling','retreat'):
            if theme_phase in ('launch','day0') and mb_now>=.62 and .015<=ret_now<=.065 and range_pos>=.72 and (not np.isfinite(tb) or tb>=.60):
                return 'low_level_switch'
            return None
        return None

    def run(self):
        paths=self.paths(); assert len(paths)==23, len(paths)
        prev_eq=START
        for path in paths:
            date,df,dr,sn=self.load_day(path); close_map=dr.close.to_dict(); prior=self.prior_day(date)
            if prior is None:
                e=self.equity(close_map); self.peak=max(self.peak,e); self.daily.append({'date':date,'equity':e,'day_return':0.0,'drawdown':0.0,'positions':0,'risk_state':self.risk_state,'prior_phase':'cold_start'}); self.prev_close=close_map; prev_eq=e; continue
            prior_market=self.market[self.market.date==prior].iloc[0]; phase=prior_market.emotion_phase; cap=self.total_cap(phase)
            # Manage existing positions first; T+1 enforced by only evaluating positions from earlier dates.
            for code in list(self.pos):
                if code not in dr.index: continue
                p=self.pos[code]; p['hold_days']+=1; rr=dr.loc[code]
                z=df[df.code==code]
                if len(z):
                    p['max_price']=max(p['max_price'],float(z.high.max())); p['min_price']=min(p['min_price'],float(z.low.min()))
                s35=self.row_at(df,code,'09:35'); n36=self.row_at(df,code,'09:36')
                if s35 is not None and n36 is not None:
                    rtot=float(s35.close)/p['entry_price']-1; dayret=float(s35.close)/float(rr.prev_close)-1 if np.isfinite(rr.prev_close) else 0
                    if rtot<=-.045 or (dayret<=-.03 and float(s35.close)<float(rr.open)):
                        if self.sell(date,code,float(n36.open),'09:36','hard_or_expectation_loss',rr,float(s35.run_high),float(s35.run_low)): continue
                s10=self.row_at(df,code,'10:00'); n01=self.row_at(df,code,'10:01')
                if s10 is not None and n01 is not None and code in self.pos:
                    rtot=float(s10.close)/p['entry_price']-1; peak=max(p['max_price'],float(s10.run_high))/p['entry_price']-1; pull=float(s10.close)/max(p['max_price'],float(s10.run_high))-1
                    dayret=float(s10.close)/float(rr.prev_close)-1 if np.isfinite(rr.prev_close) else 0
                    rp=(float(s10.close)-float(s10.run_low))/(float(s10.run_high)-float(s10.run_low)) if float(s10.run_high)>float(s10.run_low) else .5
                    if rtot<=-.05:
                        if self.sell(date,code,float(n01.open),'10:01','hard_loss_cut',rr,float(s10.run_high),float(s10.run_low)): continue
                    if dayret<=-.025 and rp<.40:
                        if self.sell(date,code,float(n01.open),'10:01','expectation_failed',rr,float(s10.run_high),float(s10.run_low)): continue
                    if peak>=.08 and pull<=-.04:
                        if self.sell(date,code,float(n01.open),'10:01','protect_right_tail',rr,float(s10.run_high),float(s10.run_low)): continue
                    if rtot>=.12 and rp<.60:
                        if self.sell(date,code,float(n01.open),'10:01','take_profit_after_12pct',rr,float(s10.run_high),float(s10.run_low)): continue
                    if p['hold_days']>=3 and rtot<.03:
                        if self.sell(date,code,float(n01.open),'10:01','time_stop_no_followthrough',rr,float(s10.run_high),float(s10.run_low)): continue
                    if p['hold_days']>=5:
                        if self.sell(date,code,float(n01.open),'10:01','max_hold_5d',rr,float(s10.run_high),float(s10.run_low)): continue

            # Prior-day roles are the only thematic/role information available before today's close.
            roles=self.roles[self.roles.date==prior].copy()
            roles=roles[roles.role.isin(['market_dragon','theme_dragon','capacity_core','elastic_core'])]
            roles=roles.sort_values(['recognition_score','theme_score'],ascending=False).drop_duplicates('code').head(6)
            mb35=self.market_breadth(sn['09:35'])
            for tm in ENTRY_TIMES:
                if len(self.pos)>=MAX_POS: break
                snap=sn[tm]; mb=self.market_breadth(snap)
                if not np.isfinite(mb): continue
                snap_idx=snap.set_index('code',drop=False)
                for r in roles.itertuples(index=False):
                    if len(self.pos)>=MAX_POS: break
                    code=str(r.code)
                    if len(code)!=6 or code not in dr.index or code not in snap_idx.index or code in self.pos: continue
                    rr=dr.loc[code]; sr=snap_idx.loc[code]
                    if isinstance(sr,pd.DataFrame): sr=sr.iloc[0]
                    pc=float(rr.prev_close) if np.isfinite(rr.prev_close) else np.nan
                    if not np.isfinite(pc) or pc<=0: continue
                    gap=float(rr.open)/pc-1; rn=float(sr.close)/pc-1
                    rp=float(sr.range_pos)
                    theme_name=str(r.theme)
                    # Market dragon may have theme='market'; use its matching theme if present in prior theme leaders.
                    if theme_name=='market':
                        mm=self.theme[(self.theme.date==prior)&(self.theme.leader_code.astype(str)==code)]
                        if len(mm): theme_name=str(mm.iloc[0].theme)
                    tb=self.theme_breadth(prior,theme_name,snap) if theme_name!='market' else np.nan
                    th=self.theme[(self.theme.date==prior)&(self.theme.theme==theme_name)]
                    tph=str(th.iloc[0].theme_phase) if len(th) else 'unknown'
                    mode=self.choose_mode(phase,str(r.role),tph,mb35,mb,tb,rn,gap,rp)
                    if not mode or self.mode_weight(mode,date)<=0: continue
                    nt=next_minute(tm); nb=self.row_at(df,code,nt)
                    if nb is None: continue
                    eq=self.equity(close_map)
                    if self.buy(date,code,mode,float(nb.open),nt,rr,eq,cap,str(r.role),theme_name,tph,tb,mb,phase):
                        break

            # End-of-day mark and causal review update.
            for code,p in self.pos.items():
                z=df[df.code==code]
                if p['entry_date']==date: z=z[z.minute>=p['entry_time']]
                if len(z): p['max_price']=max(p['max_price'],float(z.high.max())); p['min_price']=min(p['min_price'],float(z.low.min()))
            e=self.equity(close_map); self.peak=max(self.peak,e); dayret=e/prev_eq-1; dd=e/self.peak-1
            self.daily.append({'date':date,'equity':e,'day_return':dayret,'drawdown':dd,'positions':len(self.pos),'risk_state':self.risk_state,'prior_phase':phase,'market_breadth_0935':mb35})
            note=self.update_review(date,dayret)
            self.review_log.append({'date':date,'used_prior_day':prior,'prior_emotion':phase,'equity':e,'day_return':dayret,'drawdown':dd,'risk_state_after_review':self.risk_state,
                                    'rule_update':note,'mode_weights_after':json.dumps(dict(self.mode_weights),ensure_ascii=False)})
            prev_eq=e; self.prev_close=close_map

        # Month-end liquidation at last available close for evaluation only.
        last_date=self.days[-1]
        last_path=paths[-1]; _,df,dr,sn=self.load_day(last_path)
        for code in list(self.pos):
            if code in dr.index:
                self.sell(last_date,code,float(dr.loc[code].close),'15:00','month_end_liquidation',dr.loc[code])
        final=self.cash
        if self.daily:
            self.daily[-1]['equity']=final
            if len(self.daily)>1:self.daily[-1]['day_return']=final/self.daily[-2]['equity']-1
        eq=pd.DataFrame(self.daily)
        eq['drawdown']=eq.equity/eq.equity.cummax()-1
        tr=pd.DataFrame(self.trades); fi=pd.DataFrame(self.fills); rv=pd.DataFrame(self.review_log)
        tr.to_csv(OUT/'trades.csv',index=False,encoding='utf-8-sig'); fi.to_csv(OUT/'fills.csv',index=False,encoding='utf-8-sig'); rv.to_csv(OUT/'daily_review.csv',index=False,encoding='utf-8-sig'); eq.to_csv(OUT/'equity_curve.csv',index=False,encoding='utf-8-sig')
        gp=float(tr.loc[tr.pnl>0,'pnl'].sum()) if len(tr) else 0; gl=float(-tr.loc[tr.pnl<0,'pnl'].sum()) if len(tr) else 0
        summary={'scope':'2026-07 only; web PIT close facts become usable T+1; GitHub 1-minute prices for execution','start_cash':START,'final_equity':final,'total_return':final/START-1,
                 'max_drawdown':float(eq.drawdown.min()) if len(eq) else 0,'closed_trades':int(len(tr)),'win_rate':float((tr.net_return>0).mean()) if len(tr) else 0,
                 'profit_factor':gp/gl if gl>0 else (999 if gp>0 else 0),'total_pnl':float(tr.pnl.sum()) if len(tr) else 0,'ending_positions':0,
                 'future_leakage_policy':'No same-day close theme data used intraday. T close snapshot/role/review/expectation only used from T+1. No lianban 明日推演/同景日期/前瞻回验 fields.'}
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        lines=['# July Full Rebuild Causal Backtest','',json.dumps(summary,ensure_ascii=False,indent=2),'','## Trades']
        if len(tr):
            for i,t in tr.iterrows(): lines.append(f"{i+1}. {t.code} {t.theme} {t.mode} | {t.entry_date} {t.entry_time} {t.entry_price:.4f} -> {t.exit_date} {t.exit_time} {t.exit_price:.4f} | {t.net_return:.2%} | PnL {t.pnl:.2f} | {t.exit_reason} | {t.error_tags}")
        else: lines.append('No trades')
        (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    Engine().run()
