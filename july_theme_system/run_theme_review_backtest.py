from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('2026-07')
SYS=Path('july_theme_system')
OUT=Path('july_theme_results'); OUT.mkdir(exist_ok=True)
START=1_000_000.0
OBS=['09:45','10:00','10:30']
FILL={'09:45':'09:46','10:00':'10:01','10:30':'10:31'}
ROLE_RANK={'theme_leader':0,'capacity_core':1,'elastic_core':2,'trend_core':3,'market_high':4,'supplement':5,'follower':9,'negative_anchor':99}
ELIGIBLE_CORE={'theme_leader','capacity_core','elastic_core','trend_core','market_high','supplement'}


def norm_code(s):
    x=s.astype(str).str.replace(r'\.0$','',regex=True)
    y=x.str.extract(r'(\d{6})',expand=False)
    return y.fillna(x)

def abs_pct(v):
    try:
        if pd.isna(v): return np.nan
        x=abs(float(v)); return x/100 if x>1 else x
    except Exception: return np.nan

def next_open_bar(df,code,minute):
    z=df[(df.code==code)&(df.minute==minute)]
    return None if z.empty else z.iloc[0]

def safe_bool(x):
    if pd.isna(x): return False
    if isinstance(x,str): return x.strip().lower() in {'1','true','yes'}
    return bool(x)

class ThemeReviewBT:
    def __init__(self):
        self.rules=json.loads((SYS/'playbook_rules_v1.json').read_text(encoding='utf-8'))
        self.ctx=pd.read_csv(SYS/'july_2026_theme_timeline.csv',dtype=str)
        self.roles=pd.read_csv(SYS/'daily_core_roles.csv',dtype=str)
        self.themes=pd.read_csv(SYS/'daily_theme_priority.csv',dtype=str)
        self.aliases=pd.read_csv(SYS/'theme_aliases.csv',dtype=str)
        self.exceptions=pd.read_csv(SYS/'tradeability_exceptions.csv',dtype=str)
        self.overrides=pd.read_csv(SYS/'identity_overrides.csv',dtype=str)
        self.name_override=dict(zip(self.overrides.stock_code,self.overrides.canonical_name))
        self.alias_map=defaultdict(set)
        for r in self.aliases.itertuples(index=False): self.alias_map[r.theme].add(r.role_theme)
        self.cash=START; self.pos={}; self.last_close={}; self.peak=START
        self.trades=[]; self.fills=[]; self.daily=[]; self.eq=[]; self.signals=[]; self.predictions=[]; self.missed=[]; self.errors=[]; self.learning_hist=[]
        self.learn={m:{'consec_losses':0,'half_slots':0,'pause_days':0} for m in self.rules['playbooks'] if m!='cash'}
        self.entered_on_day=defaultdict(set)

    def paths(self):
        p=sorted(ROOT.glob('part-*/date=*/minute1.parquet'),key=lambda x:x.parent.name)
        assert len(p)==23
        return p

    def context_day(self,date):
        x=self.ctx[self.ctx.valid_from==date]
        return None if x.empty else x.iloc[0]

    def theme_rows(self,date): return self.themes[self.themes.valid_from==date].copy()
    def role_rows(self,date):
        x=self.roles[self.roles.valid_from==date].copy()
        x['stock_code']=norm_code(x.stock_code)
        x['stock_name']=x.apply(lambda r:self.name_override.get(r.stock_code,r.stock_name),axis=1)
        return x

    def role_theme_names(self,theme): return {theme}|self.alias_map.get(theme,set())

    def phase_spec(self,date):
        c=self.context_day(date)
        if c is None: return 'cold_start',{'max_equity_exposure':0.0,'allowed':['cash']}
        phase=c.market_phase
        assert phase in self.rules['market_risk'], f'unmapped phase {phase}'
        return phase,self.rules['market_risk'][phase]

    def load_day(self,path):
        cols=['code','datetime','open','high','low','close','volume','amount']
        df=pd.read_parquet(path,columns=cols); df['code']=norm_code(df.code); df['datetime']=pd.to_datetime(df.datetime); df=df.sort_values(['code','datetime'])
        for c in ['open','high','low','close','volume','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
        df=df.dropna(subset=['open','high','low','close']); df['minute']=df.datetime.dt.strftime('%H:%M')
        daily=df.groupby('code').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),amount=('amount','sum')).reset_index()
        daily['prev_close']=daily.code.map(self.last_close)
        daily['ret']=np.where(daily.prev_close.notna()&(daily.prev_close>0),daily.close/daily.prev_close-1,np.nan)
        day_open=dict(zip(daily.code,daily.open)); prev=dict(zip(daily.code,daily.prev_close))
        df['run_high']=df.groupby('code').high.cummax(); df['run_low']=df.groupby('code').low.cummin()
        snaps={}
        for t in ['09:35']+OBS:
            z=df[df.minute==t].copy(); z['day_open']=z.code.map(day_open); z['prev_close']=z.code.map(prev); z['ret_now']=z.close/z.prev_close-1; z['open_gap']=z.day_open/z.prev_close-1
            den=(z.run_high-z.run_low).replace(0,np.nan); z['range_pos']=((z.close-z.run_low)/den).fillna(.5).clip(0,1); snaps[t]=z.set_index('code',drop=False)
        sp=path.parent/'daily_stock_status.parquet'; st=pd.read_parquet(sp); st['code']=norm_code(st.code); st=st.set_index('code',drop=False)
        return df,daily.set_index('code',drop=False),st,snaps

    def market_stats(self,snap):
        r=snap.ret_now.replace([np.inf,-np.inf],np.nan).dropna()
        return {'adv_ratio':float((r>0).mean()) if len(r) else 0.0,'median_ret':float(r.median()) if len(r) else 0.0,'n':int(len(r))}

    def status_row(self,st,code):
        if code not in st.index: return None
        x=st.loc[code]
        return x.iloc[0] if isinstance(x,pd.DataFrame) else x

    def can_trade_stock(self,st,code,side,raw,prev_close,bar=None):
        if not np.isfinite(raw) or raw<=0: return False,'invalid_price'
        sr=self.status_row(st,code)
        if sr is not None:
            for c in ['is_st','is_star_st','is_new_listing_initial']:
                if c in sr and safe_bool(sr[c]): return False,c
            if 'is_suspended' in sr and safe_bool(sr['is_suspended']): return False,'suspended'
            if side=='BUY':
                p=abs_pct(sr.get('price_limit_up_pct',np.nan))
                if np.isfinite(p) and np.isfinite(prev_close) and raw>=prev_close*(1+p)*.998: return False,'locked_limit_up'
            else:
                p=abs_pct(sr.get('price_limit_down_pct',np.nan))
                if np.isfinite(p) and np.isfinite(prev_close) and raw<=prev_close*(1-p)*1.002: return False,'locked_limit_down'
        if bar is not None and 'volume' in bar and pd.notna(bar.volume) and float(bar.volume)<=0: return False,'zero_volume_fill_bar'
        return True,''

    def equity(self,close_map): return self.cash+sum(p['shares']*close_map.get(c,p['last_mark']) for c,p in self.pos.items())
    def exposure(self,mark_map,eq): return sum(p['shares']*mark_map.get(c,p['last_mark']) for c,p in self.pos.items())/max(eq,1e-9)

    def candidate_table(self,date):
        roles=self.role_rows(date); themes=self.theme_rows(date)
        out=[]
        for t in themes[themes.tradable_next_day.astype(int)==1].itertuples(index=False):
            names=self.role_theme_names(t.theme)
            rr=roles[(roles.theme.isin(names))&(roles.tradable_next_day.astype(int)==1)&(roles.role.isin(ELIGIBLE_CORE))]
            for r in rr.itertuples(index=False):
                out.append({'code':r.stock_code,'name':self.name_override.get(r.stock_code,r.stock_name),'role_theme':r.theme,'theme':t.theme,'role':r.role,'theme_state':t.theme_state,'priority':int(t.priority),'allowed_modes':t.allowed_modes.split('|'),'theme_reason':t.reason})
        if not out: return pd.DataFrame(columns=['code','name','role_theme','theme','role','theme_state','priority','allowed_modes','theme_reason'])
        x=pd.DataFrame(out).drop_duplicates(['code','theme','role'])
        return x

    def theme_snapshot(self,date,theme,snap,roles_day,st,candidate_code=None):
        names=self.role_theme_names(theme); rr=roles_day[roles_day.theme.isin(names)].copy(); vals=[]; near=False; capacity_positive=False
        for r in rr.itertuples(index=False):
            if r.stock_code not in snap.index: continue
            s=snap.loc[r.stock_code]; s=s.iloc[0] if isinstance(s,pd.DataFrame) else s
            ret=float(s.ret_now) if pd.notna(s.ret_now) else np.nan
            if not np.isfinite(ret): continue
            vals.append((r.stock_code,r.role,ret))
            if r.role=='capacity_core' and ret>0: capacity_positive=True
            sr=self.status_row(st,r.stock_code); lim=abs_pct(sr.get('price_limit_up_pct',np.nan)) if sr is not None else np.nan
            if r.stock_code!=candidate_code and np.isfinite(lim) and ret>=.9*lim: near=True
        others=[v for v in vals if v[0]!=candidate_code]; nonneg=sum(v[2]>=0 for v in others); pos=sum(v[2]>0 for v in vals)
        same_confirm=(nonneg>=2) or (near and capacity_positive)
        med=float(np.median([v[2] for v in vals])) if vals else np.nan
        return {'n':len(vals),'positive':pos,'other_nonnegative':nonneg,'median':med,'near_limit_other':near,'capacity_positive':capacity_positive,'same_confirm':same_confirm,'values':vals}

    def negative_anchor_median(self,date,snap):
        r=self.role_rows(date); r=r[r.role=='negative_anchor']; vals=[]
        for x in r.itertuples(index=False):
            if x.stock_code in snap.index:
                s=snap.loc[x.stock_code]; s=s.iloc[0] if isinstance(s,pd.DataFrame) else s
                if pd.notna(s.ret_now) and np.isfinite(float(s.ret_now)): vals.append(float(s.ret_now))
        return float(np.median(vals)) if vals else 0.0

    def explicit_loss_status(self,date,code,held_theme):
        r=self.role_rows(date); neg=((r.stock_code==code)&(r.role=='negative_anchor')).any()
        t=self.theme_rows(date); names={held_theme}|self.alias_map.get(held_theme,set()); retreat=((t.theme.isin(names))&(t.theme_state=='retreat')).any()
        return bool(neg or retreat)

    def retained_core(self,date,code,held_theme=None):
        r=self.role_rows(date); x=r[(r.stock_code==code)&(r.tradable_next_day.astype(int)==1)&(r.role.isin(ELIGIBLE_CORE))]
        if held_theme is not None:
            names={held_theme}|self.alias_map.get(held_theme,set()); x=x[x.theme.isin(names)]
        return not x.empty

    def pass_mode(self,mode,row,s,s935,snap,roles_day,st,date,phase,phase_allowed):
        reasons=[]; ret=float(s.ret_now); gap=float(s.open_gap); rp=float(s.range_pos); m=self.market_stats(snap); m935=self.market_stats(s935)
        th=self.theme_snapshot(date,row.theme,snap,roles_day,st,row.code); state=row.theme_state
        if mode not in phase_allowed: reasons.append('phase_disallows_mode')
        if mode not in row.allowed_modes: reasons.append('theme_disallows_mode')
        if m['median_ret']<-.015 and mode!='ice_point_repair': reasons.append('market_median_below_-1.5%')
        if mode=='leader_divergence_to_consensus':
            if state not in {'confirm','accelerate','repair','divergence'}: reasons.append('wrong_theme_state')
            if row.role not in {'theme_leader','capacity_core','trend_core','elastic_core'}: reasons.append('role_not_core_enough')
            if not(-.04<=gap<=.05): reasons.append('gap_outside')
            if not(float(s.close)>=float(s.prev_close) or float(s.close)>float(s.day_open)): reasons.append('not_recovered')
            if rp<.60: reasons.append('range_position_low')
            if not th['same_confirm']: reasons.append('theme_not_confirmed')
            if ret>=.085: reasons.append('already_too_extended')
        elif mode=='leader_weak_to_strong':
            if state not in {'new_trial','confirm','repair','divergence'}: reasons.append('wrong_theme_state')
            if not(-.035<=gap<=.02): reasons.append('gap_outside')
            if ret<.01: reasons.append('return_below_1%')
            if not(float(s.close)>float(s.day_open)): reasons.append('below_or_equal_open')
            if rp<.70: reasons.append('range_position_low')
            if not th['same_confirm']: reasons.append('theme_not_confirmed')
            if m['adv_ratio']<m935['adv_ratio']: reasons.append('breadth_worsening')
        elif mode=='low_level_switch':
            if state!='new_trial': reasons.append('not_new_trial')
            tr=self.theme_rows(date); old_bad=((tr.theme_state=='retreat')|(tr.theme_state=='watch')).any() or any(k in phase for k in ['主跌','退潮','分歧','冰点'])
            if not old_bad: reasons.append('no_old_cycle_stress')
            if th['n']<2: reasons.append('fewer_than_2_named_reps')
            if not(ret>0 and float(s.close)>float(s.day_open)): reasons.append('candidate_not_positive_above_open')
            if th['positive']<2: reasons.append('fewer_than_2_positive_reps')
            if self.negative_anchor_median(date,snap)>0: reasons.append('old_negative_anchors_reversing')
            if ret>=.085: reasons.append('already_too_extended')
        elif mode=='supplement_after_leader_confirmed':
            if row.role!='supplement': reasons.append('not_supplement')
            if state not in {'confirm','accelerate'}: reasons.append('wrong_theme_state')
            names=self.role_theme_names(row.theme); leaders=roles_day[(roles_day.theme.isin(names))&(roles_day.role=='theme_leader')]
            leader_ok=False
            for l in leaders.itertuples(index=False):
                if l.stock_code in snap.index:
                    q=snap.loc[l.stock_code]; q=q.iloc[0] if isinstance(q,pd.DataFrame) else q
                    if pd.notna(q.ret_now) and float(q.ret_now)>0 and float(q.close)>float(q.day_open): leader_ok=True
            if not leader_ok: reasons.append('leader_not_confirmed_strong')
            if not(ret>0 and rp>=.70): reasons.append('supplement_not_strong')
            if th['other_nonnegative']<2 and th['positive']<3: reasons.append('theme_breadth_contracting')
        elif mode=='ice_point_repair':
            if not any(k in phase for k in ['冰点','主跌','恐慌']): reasons.append('prior_phase_not_ice')
            if row.role not in {'theme_leader','capacity_core','trend_core','elastic_core'}: reasons.append('role_not_core')
            if m['adv_ratio']-m935['adv_ratio']<.10: reasons.append('breadth_improvement_below_10pp')
            if m['adv_ratio']<.30: reasons.append('advancer_ratio_below_30%')
            if ret<.01: reasons.append('candidate_return_below_1%')
            if th['positive']<2: reasons.append('fewer_than_2_positive_reps')
        else: reasons.append('unknown_mode')
        return len(reasons)==0,reasons,th,m,m935

    def mode_scale(self,mode):
        s=self.learn[mode]
        if s['pause_days']>0: return 0.0
        if s['half_slots']>0: return .5
        return 1.0

    def buy(self,date,obs_t,fill_t,row,raw_bar,st,daily,mark_map,cap,mode):
        code=row.code
        if code in self.pos: return False,'already_held'
        if len(self.pos)>=int(self.rules['execution']['max_positions']): return False,'max_positions'
        same=sum(p['theme']==row.theme for p in self.pos.values())
        if same>=int(self.rules['execution']['same_theme_max_positions']): return False,'same_theme_cap'
        scale=self.mode_scale(mode)
        if scale<=0: return False,'mode_paused'
        if code not in daily.index: return False,'missing_daily_row'
        prev=float(daily.loc[code].prev_close) if pd.notna(daily.loc[code].prev_close) else np.nan
        ok,reason=self.can_trade_stock(st,code,'BUY',float(raw_bar.open),prev,raw_bar)
        if not ok: return False,reason
        eq=self.equity(mark_map); exp=self.exposure(mark_map,eq); remain=max(0.0,float(cap)-exp)
        frac=min(float(self.rules['playbooks'][mode]['max_single_position'])*scale,remain)
        if frac<.05: return False,'insufficient_exposure_budget'
        slip=float(self.rules['execution']['slippage_each_side']); comm=float(self.rules['execution']['commission_each_side']); lot=int(self.rules['execution']['lot_size'])
        px=float(raw_bar.open)*(1+slip); budget=min(self.cash/(1+comm),eq*frac); sh=int(budget/px/lot)*lot
        if sh<lot: return False,'cash_or_lot_too_small'
        gross=px*sh; fee=gross*comm
        if gross+fee>self.cash: return False,'cash_shortfall'
        self.cash-=gross+fee
        self.pos[code]={'code':code,'name':row['name'],'theme':row.theme,'role':row.role,'mode':mode,'entry_date':date,'obs_time':obs_t,'entry_time':fill_t,'entry_price':px,'shares':sh,'basis':gross+fee,'hold_days':0,'max_price':px,'min_price':px,'last_mark':px,'entry_phase':self.context_day(date).market_phase,'theme_state_entry':row.theme_state,'priority_entry':int(row.priority),'scale_entry':scale}
        self.entered_on_day[date].add(code)
        self.fills.append({'date':date,'observation_time':obs_t,'time':fill_t,'code':code,'name':row['name'],'theme':row.theme,'role':row.role,'side':'BUY','raw_price':float(raw_bar.open),'fill_price':px,'shares':sh,'mode':mode,'reason':'all_playbook_gates_passed'})
        if self.learn[mode]['half_slots']>0: self.learn[mode]['half_slots']-=1
        return True,''

    def sell(self,date,time,code,raw_bar,st,daily,reason):
        if code not in self.pos: return False,'not_held'
        p=self.pos[code]
        if date==p['entry_date']: return False,'T+1'
        if code not in daily.index: return False,'missing_daily_row'
        prev=float(daily.loc[code].prev_close) if pd.notna(daily.loc[code].prev_close) else np.nan
        ok,why=self.can_trade_stock(st,code,'SELL',float(raw_bar.open if hasattr(raw_bar,'open') else raw_bar),prev,raw_bar if hasattr(raw_bar,'open') else None)
        if not ok:return False,why
        raw=float(raw_bar.open if hasattr(raw_bar,'open') else raw_bar); slip=float(self.rules['execution']['slippage_each_side']); comm=float(self.rules['execution']['commission_each_side']); stamp=float(self.rules['execution']['stamp_tax_sell'])
        px=raw*(1-slip); gross=px*p['shares']; fee=gross*(comm+stamp); self.cash+=gross-fee; pnl=gross-fee-p['basis']; nr=pnl/p['basis']
        tr={**{k:p[k] for k in ['code','name','theme','role','mode','entry_date','obs_time','entry_time','entry_price','shares','entry_phase','theme_state_entry','priority_entry','scale_entry']},'exit_date':date,'exit_time':time,'exit_price':px,'holding_days':p['hold_days'],'pnl':pnl,'net_return':nr,'MFE':p['max_price']/p['entry_price']-1,'MAE':p['min_price']/p['entry_price']-1,'exit_reason':reason}
        self.trades.append(tr); self.fills.append({'date':date,'observation_time':'','time':time,'code':code,'name':p['name'],'theme':p['theme'],'role':p['role'],'side':'SELL','raw_price':raw,'fill_price':px,'shares':p['shares'],'mode':p['mode'],'reason':reason}); del self.pos[code]
        return True,''

    def update_intraday_extremes(self,df,date,until=None):
        for code,p in list(self.pos.items()):
            z=df[df.code==code]
            if p['entry_date']==date: z=z[z.minute>=p['entry_time']]
            if until is not None: z=z[z.minute<=until]
            if not z.empty:
                p['max_price']=max(p['max_price'],float(z.high.max())); p['min_price']=min(p['min_price'],float(z.low.min())); p['last_mark']=float(z.close.iloc[-1])

    def process_exits(self,date,df,daily,st,snaps):
        # explicit prior-close demotion is known before today's trading; execute at first available minute open.
        first_min=df.minute.min()
        for code in list(self.pos):
            if date==self.pos[code]['entry_date']: continue
            self.pos[code]['hold_days']+=1
            if self.explicit_loss_status(date,code,self.pos[code]['theme']):
                z=df[df.code==code]
                if not z.empty:
                    b=z.iloc[0]
                    if self.sell(date,str(b.minute),code,b,st,daily,'leader_loss_of_status')[0]: continue
        # Remaining exits use information through completed 10:00 minute, then execute 10:01 open.
        self.update_intraday_extremes(df,date,until='10:00')
        if '10:00' not in snaps: return
        snap=snaps['10:00']; roles_day=self.role_rows(date)
        for code in list(self.pos):
            p=self.pos[code]
            if date==p['entry_date'] or code not in snap.index: continue
            s=snap.loc[code]; s=s.iloc[0] if isinstance(s,pd.DataFrame) else s
            b=next_open_bar(df,code,'10:01')
            if b is None: continue
            rentry=float(s.close)/p['entry_price']-1; dret=float(s.ret_now) if pd.notna(s.ret_now) else 0.0
            tstat=self.theme_snapshot(date,p['theme'],snap,roles_day,st,code)
            peak_pull=float(s.close)/p['max_price']-1
            if rentry<=-.05:
                if self.sell(date,'10:01',code,b,st,daily,'hard_failure')[0]: continue
            if dret<=-.025 and float(s.close)<float(s.day_open) and not tstat['same_confirm']:
                if self.sell(date,'10:01',code,b,st,daily,'expectation_failure')[0]: continue
            if p['max_price']/p['entry_price']-1>=.10 and peak_pull<=-.05:
                if self.sell(date,'10:01',code,b,st,daily,'right_tail_protection')[0]: continue
            if p['hold_days']>=3 and rentry<.03 and not self.retained_core(date,code,p['theme']):
                if self.sell(date,'10:01',code,b,st,daily,'time_stop_no_core_followthrough')[0]: continue
            if p['hold_days']>=7 and not self.retained_core(date,code,p['theme']):
                if self.sell(date,'10:01',code,b,st,daily,'max_holding_7d')[0]: continue

    def process_entries(self,date,df,daily,st,snaps):
        ctx=self.context_day(date)
        if ctx is None: return
        phase,ps=self.phase_spec(date); cap=float(ps['max_equity_exposure']); phase_allowed=[x for x in ps['allowed'] if x!='cash']; cand=self.candidate_table(date); roles_day=self.role_rows(date)
        if cand.empty or not phase_allowed:return
        bought_codes=set()
        for obs in OBS:
            fill=FILL[obs]
            if obs not in snaps: continue
            snap=snaps[obs]; s935=snaps['09:35']; marks={c:(float(snap.loc[c].close) if c in snap.index else p['last_mark']) for c,p in self.pos.items()}
            passed=[]
            for r in cand.itertuples(index=False):
                if r.code in self.pos or r.code in bought_codes or r.code not in snap.index or r.code not in s935.index: continue
                s=snap.loc[r.code]; s=s.iloc[0] if isinstance(s,pd.DataFrame) else s
                if not(pd.notna(s.ret_now) and pd.notna(s.open_gap)): continue
                for mode in phase_allowed:
                    if mode not in r.allowed_modes: continue
                    if self.mode_scale(mode)<=0:
                        self.signals.append({'date':date,'observation_time':obs,'code':r.code,'name':r.name,'theme':r.theme,'role':r.role,'theme_state':r.theme_state,'priority':r.priority,'mode':mode,'passed':False,'block_reasons':'mode_paused_by_prior_review'})
                        continue
                    ok,reasons,th,m,m935=self.pass_mode(mode,r,s,s935,snap,roles_day,st,date,phase,phase_allowed)
                    self.signals.append({'date':date,'observation_time':obs,'code':r.code,'name':r.name,'theme':r.theme,'role':r.role,'theme_state':r.theme_state,'priority':r.priority,'mode':mode,'passed':ok,'block_reasons':'|'.join(reasons),'ret_now':float(s.ret_now),'open_gap':float(s.open_gap),'range_pos':float(s.range_pos),'theme_rep_n':th['n'],'theme_positive':th['positive'],'market_adv':m['adv_ratio'],'market_median':m['median_ret']})
                    if ok:
                        passed.append((int(r.priority),ROLE_RANK.get(r.role,50),phase_allowed.index(mode),r.code,r,mode))
            passed=sorted(passed,key=lambda x:(x[0],x[1],x[2],x[3]))
            seen=set()
            for _,_,_,code,r,mode in passed:
                if code in seen or code in self.pos or code in bought_codes: continue
                b=next_open_bar(df,code,fill)
                if b is None: continue
                # mark current held positions at observation; candidate uses current observation close for exposure only.
                mark_map={c:(float(snap.loc[c].close) if c in snap.index else p['last_mark']) for c,p in self.pos.items()}
                ok,reason=self.buy(date,obs,fill,r,b,st,daily,mark_map,cap,mode)
                if ok: bought_codes.add(code)
                seen.add(code)
                if len(self.pos)>=int(self.rules['execution']['max_positions']): break

    def score_predictions(self,date,daily):
        tr=self.theme_rows(date); rr=self.role_rows(date)
        for t in tr.itertuples(index=False):
            names=self.role_theme_names(t.theme); r=rr[rr.theme.isin(names)]; vals=[]
            for x in r.itertuples(index=False):
                if x.stock_code in daily.index and pd.notna(daily.loc[x.stock_code].ret): vals.append(float(daily.loc[x.stock_code].ret))
            med=float(np.median(vals)) if vals else np.nan; pos=float(np.mean(np.array(vals)>0)) if vals else np.nan
            if t.theme_state in {'retreat','watch'}: hit=bool(np.isfinite(med) and (med<=0 or pos<.5))
            elif t.theme_state=='new_trial': hit=bool(len(vals)>=2 and pos>=.5)
            else: hit=bool(np.isfinite(med) and pos>=.5)
            self.predictions.append({'date':date,'theme':t.theme,'prior_state':t.theme_state,'priority':int(t.priority),'tradable':int(t.tradable_next_day),'named_samples':len(vals),'close_median_return':med,'positive_share':pos,'prediction_hit':hit})

    def collect_missed(self,date,daily):
        c=self.candidate_table(date); traded=self.entered_on_day[date]
        for r in c.itertuples(index=False):
            if r.code in traded or r.code not in daily.index or pd.isna(daily.loc[r.code].ret): continue
            ret=float(daily.loc[r.code].ret)
            if ret>=.05:self.missed.append({'date':date,'code':r.code,'name':r.name,'theme':r.theme,'role':r.role,'daily_return':ret,'note':'plan_universe_winner_not_entered; review only, never backfilled'})

    def apply_learning_after_close(self,date,new_trade_start):
        closed=self.trades[new_trade_start:]
        events=[]
        bymode=defaultdict(list)
        for t in closed:
            if t['exit_reason']=='evaluation_month_end': continue
            bymode[t['mode']].append(t)
        for mode,ts in bymode.items():
            for t in ts:
                if t['net_return']<0: self.learn[mode]['consec_losses']+=1
                else: self.learn[mode]['consec_losses']=0
                if t['net_return']<0:
                    self.errors.append({'date':date,'code':t['code'],'name':t['name'],'theme':t['theme'],'mode':mode,'net_return':t['net_return'],'exit_reason':t['exit_reason'],'error_class':{'hard_failure':'预期/择时错误','expectation_failure':'预期错误','leader_loss_of_status':'核心地位/周期错误','time_stop_no_core_followthrough':'持续性/机会成本错误','max_holding_7d':'持仓管理错误','right_tail_protection':'利润保护退出'}.get(t['exit_reason'],'正常模式亏损'),'action':'先记录机制；达到最小样本前禁止调阈值'})
            c=self.learn[mode]['consec_losses']
            if c>=3:
                self.learn[mode]['pause_days']=max(self.learn[mode]['pause_days'],2); events.append(f'{mode}:连续{c}笔过程内亏损->后续2交易日暂停')
            elif c>=2:
                self.learn[mode]['half_slots']=max(self.learn[mode]['half_slots'],2); events.append(f'{mode}:连续{c}笔过程内亏损->后续2个合格入场半仓')
        self.learning_hist.append({'date':date,'events':'；'.join(events) if events else '无规则修改；只追加样本','state_json':json.dumps(self.learn,ensure_ascii=False)})
        # pause applies to the next N trading days, so decrement only states that were already active before today is handled at next day end externally.

    def decrement_pause_after_day(self,pre_pause):
        for m,v in pre_pause.items():
            if v>0 and self.learn[m]['pause_days']>0:self.learn[m]['pause_days']-=1

    def run(self):
        bench=1.0; prev_eq=START
        for path in self.paths():
            date=path.parent.name.split('=',1)[1]; pre_pause={m:s['pause_days'] for m,s in self.learn.items()}; new_trade_start=len(self.trades)
            df,daily,st,snaps=self.load_day(path); close_map=dict(zip(daily.code,daily.close))
            self.process_exits(date,df,daily,st,snaps)
            self.process_entries(date,df,daily,st,snaps)
            self.update_intraday_extremes(df,date,until=None)
            eq=self.equity(close_map); self.peak=max(self.peak,eq); dd=eq/self.peak-1; dayret=eq/prev_eq-1
            r=daily.ret.dropna(); bret=float(r.mean()) if len(r) else 0.0; bench*=1+bret
            self.score_predictions(date,daily); self.collect_missed(date,daily); self.apply_learning_after_close(date,new_trade_start)
            phase,ps=self.phase_spec(date); pred_today=[x for x in self.predictions if x['date']==date]; hitrate=float(np.mean([x['prediction_hit'] for x in pred_today])) if pred_today else np.nan
            self.daily.append({'date':date,'prior_close_market_phase':phase,'max_exposure_allowed':float(ps['max_equity_exposure']),'allowed_modes':'|'.join(ps['allowed']),'eod_equity':eq,'strategy_day_return':dayret,'drawdown':dd,'benchmark_day_equal_weight':bret,'benchmark_cumulative':bench-1,'positions_eod':len(self.pos),'prediction_hit_rate':hitrate,'new_closed_trades':len(self.trades)-new_trade_start,'new_entries':len(self.entered_on_day[date]),'learning_state_after_close':json.dumps(self.learn,ensure_ascii=False)})
            self.eq.append({'date':date,'equity':eq,'day_return':dayret,'drawdown':dd,'benchmark_cumulative':bench-1}); prev_eq=eq
            self.decrement_pause_after_day(pre_pause)
            self.last_close=close_map

        # Month-end measurement: attempt executable close liquidation; otherwise retain mark-to-market open position.
        last_path=self.paths()[-1]; date=last_path.parent.name.split('=',1)[1]; df,daily,st,snaps=self.load_day(last_path)
        # last_close was already advanced, so daily prev_close here is not used for signal generation; use status/last known prior in can_sell conservatively via current last_close.
        for code in list(self.pos):
            z=df[df.code==code]
            if z.empty: continue
            b=z.iloc[-1].copy(); b['open']=b['close']; b['volume']=max(float(b.get('volume',1)),1)
            # measurement sale is after all July trading decisions; no learning is applied.
            p=self.pos[code]; prev=self.last_close.get(code,np.nan)
            sr=self.status_row(st,code); down=abs_pct(sr.get('price_limit_down_pct',np.nan)) if sr is not None else np.nan
            locked=np.isfinite(down) and np.isfinite(prev) and float(b['close'])<=prev*(1-down)*1.002
            if not locked:self.sell(date,'15:00',code,b,st,daily,'evaluation_month_end')
        final_mark={c:self.last_close.get(c,p['last_mark']) for c,p in self.pos.items()}; final=self.equity(final_mark)
        if self.eq:
            self.eq[-1]['equity']=final; self.eq[-1]['day_return']=final/self.eq[-2]['equity']-1 if len(self.eq)>1 else final/START-1
            peak=START
            for e in self.eq:
                peak=max(peak,e['equity']); e['drawdown']=e['equity']/peak-1
        t=pd.DataFrame(self.trades); f=pd.DataFrame(self.fills); d=pd.DataFrame(self.daily); e=pd.DataFrame(self.eq); s=pd.DataFrame(self.signals); pr=pd.DataFrame(self.predictions); mi=pd.DataFrame(self.missed); er=pd.DataFrame(self.errors); lh=pd.DataFrame(self.learning_hist)
        for name,x in [('trades',t),('fills',f),('daily_review',d),('equity_curve',e),('candidate_signal_audit',s),('theme_prediction_scores',pr),('missed_opportunities',mi),('error_log',er),('learning_history',lh)]: x.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
        gp=float(t.loc[t.pnl>0,'pnl'].sum()) if len(t) else 0.; gl=float(-t.loc[t.pnl<0,'pnl'].sum()) if len(t) else 0.; pf=gp/gl if gl>0 else (999. if gp>0 else 0.)
        summary={'method_version':self.rules['version'],'data_scope':'GitHub 2026-07 minute1 + same-day status; web context only if prior-close valid_from <= decision date','start_equity':START,'final_equity_mark_to_market':final,'total_return':final/START-1,'max_drawdown':float(min(x['drawdown'] for x in self.eq)) if self.eq else 0.,'benchmark_equal_weight_return_from_available_july_prev_closes':bench-1,'closed_trades':int(len(t)),'open_positions_month_end':int(len(self.pos)),'win_rate':float((t.net_return>0).mean()) if len(t) else 0.,'profit_factor':pf,'total_closed_pnl':float(t.pnl.sum()) if len(t) else 0.,'prediction_hit_rate':float(pr.prediction_hit.mean()) if len(pr) else np.nan,'missed_plan_universe_winners':int(len(mi)),'rule_threshold_changes_during_july':0,'learning_policy':'only causal half-size/pause responses; no threshold retuning unless >=8 closed samples with repeated mechanism'}
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        open_rows=[]
        for p in self.pos.values(): open_rows.append({**p,'month_end_mark':self.last_close.get(p['code'],p['last_mark']),'unrealized_return':self.last_close.get(p['code'],p['last_mark'])/p['entry_price']-1})
        pd.DataFrame(open_rows).to_csv(OUT/'open_positions.csv',index=False,encoding='utf-8-sig')
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ThemeReviewBT().run()
