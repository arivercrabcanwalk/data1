from __future__ import annotations
import json, math
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('2026-07'); OUT=Path('july_review_results'); OUT.mkdir(exist_ok=True)
START=1_000_000.0; COMM=0.0003; STAMP=0.0005; SLIP=0.0005; LOT=100; MAXPOS=3
TIMES=['09:45','10:00','10:30']; MODES=['core_weak_to_strong','core_divergence_repair','ice_repair_core']

def norm_code(s):
    x=s.astype(str).str.replace(r'\.0$','',regex=True); y=x.str.extract(r'(\d{6})',expand=False); return y.fillna(x)
def market(d):
    r=d['ret'].replace([np.inf,-np.inf],np.nan).dropna()
    return {'adv_ratio':float((r>0).mean()),'median_ret':float(r.median()),'mean_ret':float(r.mean()),'up5':int((r>=.05).sum()),'down5':int((r<=-.05).sum()),'up9':int((r>=.09).sum()),'down9':int((r<=-.09).sum()),'n':int(len(r))} if len(r) else {'adv_ratio':.5,'median_ret':0,'mean_ret':0,'up5':0,'down5':0,'up9':0,'down9':0,'n':0}
def snap_market(s):
    r=s['ret_now'].replace([np.inf,-np.inf],np.nan).dropna(); return {'adv_ratio':float((r>0).mean()),'median_ret':float(r.median())} if len(r) else {'adv_ratio':.5,'median_ret':0}
def regime(m):
    a,med,u,d=m['adv_ratio'],m['median_ret'],m['up5'],m['down5']
    if a<.32 or med<=-.012 or d>max(20,2*u): return 'ice',0.0
    if a<.45 or med<-.004 or d>1.35*max(u,1): return 'weak',.25
    if a>.68 and med>.009: return 'hot',.55
    if a>.58 and med>.004 and u>d: return 'strong',.70
    return 'neutral',.45
def pct(v):
    if pd.isna(v): return np.nan
    v=float(v); return v/100 if v>1 else v

def row_at(df,code,t):
    x=df[(df.code==code)&(df.minute==t)]; return None if x.empty else x.iloc[0]
def next_minute(t):
    h,m=map(int,t.split(':')); m+=1
    if m==60: h+=1; m=0
    return f'{h:02d}:{m:02d}'

class BT:
  def __init__(self):
    self.cash=START; self.pos={}; self.trades=[]; self.fills=[]; self.reviews=[]; self.eq=[]
    self.prev_close={}; self.prev_feat=None; self.prev_mkt=None; self.ret_hist=defaultdict(lambda:deque(maxlen=5)); self.streak=defaultdict(int)
    self.peak=START; self.weights={m:1.0 for m in MODES}; self.pause={m:0 for m in MODES}; self.closed_today=[]
  def equity(self,close): return self.cash+sum(p['shares']*close.get(c,p['entry']) for c,p in self.pos.items())
  def exposure_cap(self,base,prev_eq):
    self.peak=max(self.peak,prev_eq); dd=prev_eq/self.peak-1; cap=base
    if dd<=-.08: cap=0
    elif dd<=-.05: cap=min(cap,.25)
    elif dd<=-.03: cap=min(cap,.45)
    return cap
  def review_weights(self):
    notes=[]
    for m in MODES:
      if self.pause[m]>0: self.pause[m]-=1
      ts=[t for t in self.trades if t['mode']==m]; recent=ts[-5:]
      if self.pause[m]>0: self.weights[m]=0.0; continue
      if len(ts)<3: self.weights[m]=1.0; continue
      consec=0
      for t in reversed(ts):
        if t['net_return']<0: consec+=1
        else: break
      if m in self.closed_today and consec>=3:
        self.pause[m]=2; self.weights[m]=0.0; notes.append(f'{m}:3连亏,暂停2日'); continue
      avg=float(np.mean([t['net_return'] for t in recent])); win=float(np.mean([t['net_return']>0 for t in recent]))
      if consec>=2: self.weights[m]=.5; notes.append(f'{m}:2连亏,半仓')
      elif len(recent)>=4 and avg<-.005: self.weights[m]=.65; notes.append(f'{m}:近期负期望,降权')
      elif len(recent)>=4 and avg>.012 and win>=.5: self.weights[m]=1.2; notes.append(f'{m}:近期正期望,小幅升权')
      else: self.weights[m]=1.0
    return ';'.join(notes) if notes else '不改规则，仅记录新样本'
  def load(self,path):
    df=pd.read_parquet(path,columns=['code','datetime','open','high','low','close','volume','amount']); df['code']=norm_code(df.code); df['datetime']=pd.to_datetime(df.datetime); df=df.sort_values(['code','datetime'])
    for c in ['open','high','low','close','volume','amount']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['open','high','low','close']); df['minute']=df.datetime.dt.strftime('%H:%M'); df['cum_amount']=df.groupby('code').amount.cumsum(); df['run_high']=df.groupby('code').high.cummax(); df['run_low']=df.groupby('code').low.cummin()
    d=df.groupby('code').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),amount=('amount','sum')).reset_index(); d['prev_close']=d.code.map(self.prev_close); d['ret']=np.where(d.prev_close.notna()&(d.prev_close>0),d.close/d.prev_close-1,d.close/d.open-1); d['close_pos']=((d.close-d.low)/(d.high-d.low).replace(0,np.nan)).fillna(.5).clip(0,1)
    sp=path.parent/'daily_stock_status.parquet'; s=pd.read_parquet(sp) if sp.exists() else pd.DataFrame({'code':d.code}); s['code']=norm_code(s.code)
    ks=[c for c in ['code','is_st','is_star_st','is_new_listing_initial','price_limit_up_pct','price_limit_down_pct','market_board'] if c in s.columns]; d=d.merge(s[ks],on='code',how='left')
    for c in ['is_st','is_star_st','is_new_listing_initial']:
      if c not in d: d[c]=False
    d['eligible']=~(d.is_st.fillna(False).astype(bool)|d.is_star_st.fillna(False).astype(bool)|d.is_new_listing_initial.fillna(False).astype(bool))
    o=d.set_index('code').open; snaps={}
    for t in ['09:35']+TIMES+['14:30']:
      z=df[df.minute==t].copy(); z['prev_close']=z.code.map(self.prev_close); z['day_open']=z.code.map(o); z['ret_now']=z.close/z.prev_close-1; z['open_gap']=z.day_open/z.prev_close-1; z['range_pos']=((z.close-z.run_low)/(z.run_high-z.run_low).replace(0,np.nan)).fillna(.5).clip(0,1); snaps[t]=z
    return df,d,s,snaps
  def features(self,d):
    f=d.copy(); m3=[];m5=[];ls=[]
    for r in f.itertuples(index=False):
      rr=float(r.ret) if np.isfinite(r.ret) else 0.; self.ret_hist[r.code].append(rr); a=list(self.ret_hist[r.code]); m3.append(float(np.prod([1+x for x in a[-3:]])-1)); m5.append(float(np.prod([1+x for x in a[-5:]])-1))
      pu=pct(getattr(r,'price_limit_up_pct',np.nan)); pc=getattr(r,'prev_close',np.nan); lup=np.isfinite(pu) and np.isfinite(pc) and pc>0 and r.close>=pc*(1+pu)*.995; self.streak[r.code]=self.streak[r.code]+1 if lup else 0; ls.append(self.streak[r.code])
    f['mom3']=m3;f['mom5']=m5;f['limit_streak']=ls; ok=f.eligible&np.isfinite(f.ret)&(f.amount>0)
    for c in ['ret','mom3','amount']: f[c+'_rank']=f.loc[ok,c].rank(pct=True).reindex(f.index).fillna(0)
    f['leader_score']=100*(.25*f.ret_rank+.30*f.mom3_rank+.25*f.amount_rank+.10*f.close_pos+.10*np.minimum(f.limit_streak,3)/3); f.loc[~ok,'leader_score']=-1; return f
  def can_buy(self,rr,px):
    if not rr.eligible or not np.isfinite(px) or px<=0:return False
    p=pct(rr.get('price_limit_up_pct',np.nan)); pc=rr.prev_close
    return not(np.isfinite(p) and np.isfinite(pc) and pc>0 and px>=pc*(1+p)*.998)
  def can_sell(self,rr,px):
    p=pct(rr.get('price_limit_down_pct',np.nan)); pc=rr.prev_close
    return np.isfinite(px) and px>0 and not(np.isfinite(p) and np.isfinite(pc) and pc>0 and px<=pc*(1-p)*1.002)
  def buy(self,code,mode,date,t,raw,eq,cap,leader,prreg,w,rr):
    if code in self.pos or len(self.pos)>=MAXPOS or w<=0 or not self.can_buy(rr,raw): return False
    exp=sum(p['shares']*p['entry'] for p in self.pos.values())/max(eq,1); frac=min(.25*w,max(0,cap-exp))
    if frac<.08:return False
    px=raw*(1+SLIP); budget=min(self.cash*.98,eq*frac); sh=int(budget/px/LOT)*LOT
    if sh<LOT:return False
    gross=sh*px; fee=gross*COMM
    if gross+fee>self.cash:return False
    self.cash-=gross+fee; self.pos[code]={'code':code,'mode':mode,'entry_date':date,'entry_time':t,'entry':px,'shares':sh,'basis':gross+fee,'leader_score':leader,'prior_regime':prreg,'mode_weight':w,'hold_days':0,'max_price':px,'min_price':px}; self.fills.append({'date':date,'time':t,'code':code,'side':'BUY','price':px,'shares':sh,'reason':mode}); return True
  def sell(self,code,date,t,raw,reason,rr,max_seen=None,min_seen=None):
    if code not in self.pos or not self.can_sell(rr,raw):return False
    p=self.pos[code]; p['max_price']=max(p['max_price'],max_seen if max_seen is not None else raw); p['min_price']=min(p['min_price'],min_seen if min_seen is not None else raw)
    px=raw*(1-SLIP); gross=p['shares']*px; fee=gross*(COMM+STAMP); self.cash+=gross-fee; pnl=gross-fee-p['basis']; nr=pnl/p['basis']
    self.trades.append({'code':code,'mode':p['mode'],'entry_date':p['entry_date'],'entry_time':p['entry_time'],'entry_price':p['entry'],'shares':p['shares'],'exit_date':date,'exit_time':t,'exit_price':px,'holding_days':p['hold_days'],'pnl':pnl,'net_return':nr,'exit_reason':reason,'leader_score':p['leader_score'],'prior_regime':p['prior_regime'],'mode_weight_at_entry':p['mode_weight'],'mfe_from_entry':p['max_price']/p['entry']-1,'mae_from_entry':p['min_price']/p['entry']-1}); self.closed_today.append(p['mode']); self.fills.append({'date':date,'time':t,'code':code,'side':'SELL','price':px,'shares':p['shares'],'reason':reason}); del self.pos[code]; return True
  def run(self):
    paths=sorted(ROOT.glob('part-*/date=*/minute1.parquet'),key=lambda p:p.parent.name); assert paths; prev_eq=START
    for dayi,path in enumerate(paths):
      date=path.parent.name.split('=',1)[1]; self.closed_today=[]; df,d,st,sn=self.load(path); dr=d.set_index('code',drop=False); close=dict(zip(d.code,d.close))
      prreg,base=('cold_start',0.0) if self.prev_mkt is None else regime(self.prev_mkt); cap=self.exposure_cap(base,prev_eq); plan=dict(self.weights)
      existing=list(self.pos)
      for code in existing:
        if code not in dr.index: continue
        p=self.pos[code]; p['hold_days']+=1
        z=sn['10:00']; zz=z[z.code==code]
        if zz.empty: continue
        s=zz.iloc[0]; max_seen=max(p['max_price'],float(s.run_high)); min_seen=min(p['min_price'],float(s.run_low)); rentry=float(s.close)/p['entry']-1; dret=float(s.ret_now) if np.isfinite(s.ret_now) else 0; nt=next_minute('10:00'); nb=row_at(df,code,nt)
        if nb is None: continue
        if rentry<=-.045 and self.sell(code,date,nt,float(nb.open),'hard_loss_cut_after_T1',dr.loc[code],max_seen,min_seen): continue
        if p['hold_days']>=1 and dret<=-.025 and float(s.close)<float(s.day_open) and self.sell(code,date,nt,float(nb.open),'expectation_failed',dr.loc[code],max_seen,min_seen): continue
        if p['hold_days']>=3 and rentry<.03 and self.sell(code,date,nt,float(nb.open),'time_stop_no_followthrough',dr.loc[code],max_seen,min_seen): continue
        peak=max_seen/p['entry']-1; pull=float(s.close)/max_seen-1
        if peak>=.08 and pull<=-.045 and self.sell(code,date,nt,float(nb.open),'protect_right_tail',dr.loc[code],max_seen,min_seen): continue
        if p['hold_days']>=5 and self.sell(code,date,nt,float(nb.open),'max_holding_5d',dr.loc[code],max_seen,min_seen): continue
      if self.prev_feat is not None and cap>=.08 and len(self.pos)<MAXPOS:
        core=self.prev_feat[(self.prev_feat.eligible)&(self.prev_feat.leader_score>=0)].nlargest(120,'leader_score').set_index('code'); m935=snap_market(sn['09:35'])
        used=set(); mc=defaultdict(int)
        for t in TIMES:
          if len(self.pos)>=MAXPOS: break
          s=sn[t]; mn=snap_market(s); x=s[s.code.isin(core.index)].copy()
          if x.empty: continue
          x['cur_rank']=x.ret_now.rank(pct=True); cand=[]
          for r in x.itertuples(index=False):
            if r.code in self.pos or r.code in used or not np.isfinite(r.prev_close): continue
            pr=core.loc[r.code]; L=float(pr.leader_score); rg=float(r.open_gap); rn=float(r.ret_now); rp=float(r.range_pos); cr=float(r.cur_rank)
            if plan['core_weak_to_strong']>0 and L>=70 and float(pr.ret)>=.02 and -.03<=rg<=.04 and rn>=.012 and rp>=.68 and mn['adv_ratio']>=.50 and cr>=.60: cand.append((L+120*rn+8*rp,r.code,'core_weak_to_strong'))
            if plan['core_divergence_repair']>0 and L>=72 and (float(pr.ret)>=.03 or float(pr.mom3)>=.08) and -.055<=rg<=.015 and rn>=.002 and rp>=.72 and mn['adv_ratio']>=.44: cand.append((L+100*rn+10*rp,r.code,'core_divergence_repair'))
            if plan['ice_repair_core']>0 and prreg in ('ice','weak') and L>=68 and mn['adv_ratio']>=.52 and mn['adv_ratio']-m935['adv_ratio']>=.06 and rn>=.01 and rp>=.70: cand.append((L+120*rn+15*(mn['adv_ratio']-m935['adv_ratio']),r.code,'ice_repair_core'))
          best={}
          for q in cand:
            if q[1] not in best or q[0]>best[q[1]][0]: best[q[1]]=q
          for score,code,mode in sorted(best.values(),reverse=True):
            if len(self.pos)>=MAXPOS: break
            if code in used or mc[mode]>=2: continue
            sr=x[x.code==code].iloc[0]
            if float(sr.ret_now)>.085: continue
            nt=next_minute(t); nb=row_at(df,code,nt)
            if nb is None or code not in dr.index: continue
            if self.buy(code,mode,date,nt,float(nb.open),self.equity(close),cap,float(core.loc[code].leader_score),prreg,plan[mode],dr.loc[code]): used.add(code); mc[mode]+=1
      for code,p in self.pos.items():
        z=df[df.code==code]
        if p['entry_date']==date: z=z[z.minute>=p['entry_time']]
        if len(z): p['max_price']=max(p['max_price'],float(z.high.max())); p['min_price']=min(p['min_price'],float(z.low.min()))
      feat=self.features(d); mkt=market(feat); e=self.equity(close); self.peak=max(self.peak,e); dd=e/self.peak-1; dayret=e/prev_eq-1; note=self.review_weights(); rg2,base2=regime(mkt)
      self.reviews.append({'date':date,'prior_regime_used':prreg,'exposure_cap_used':cap,'eod_equity':e,'strategy_day_return':dayret,'drawdown':dd,**mkt,'regime_after_close':rg2,'tomorrow_base_exposure':base2,'positions_eod':len(self.pos),'mode_weights_used':json.dumps(plan,ensure_ascii=False),'mode_weights_after_review':json.dumps(self.weights,ensure_ascii=False),'review_update':note})
      self.eq.append({'date':date,'equity':e,'day_return':dayret,'drawdown':dd}); prev_eq=e; self.prev_feat=feat; self.prev_mkt=mkt; self.prev_close=close
    date=paths[-1].parent.name.split('=',1)[1]
    for code in list(self.pos):
      if code in dr.index:self.sell(code,date,'15:00',float(dr.loc[code].close),'evaluation_month_end_liquidation',dr.loc[code])
    final=self.cash; total=final/START-1; eq=pd.DataFrame(self.eq)
    if len(eq): eq.loc[eq.index[-1],'equity']=final; eq.loc[eq.index[-1],'day_return']=final/eq.loc[eq.index[-2],'equity']-1 if len(eq)>1 else total; self.peak=max(START,float(eq.equity.cummax().max())); eq['drawdown']=eq.equity/eq.equity.cummax()-1
    tr=pd.DataFrame(self.trades); rv=pd.DataFrame(self.reviews); fi=pd.DataFrame(self.fills); tr.to_csv(OUT/'trades.csv',index=False,encoding='utf-8-sig'); rv.to_csv(OUT/'daily_review.csv',index=False,encoding='utf-8-sig'); fi.to_csv(OUT/'fills.csv',index=False,encoding='utf-8-sig'); eq.to_csv(OUT/'equity_curve.csv',index=False,encoding='utf-8-sig')
    gp=float(tr.loc[tr.pnl>0,'pnl'].sum()) if len(tr) else 0; gl=float(-tr.loc[tr.pnl<0,'pnl'].sum()) if len(tr) else 0; pf=gp/gl if gl>0 else (999 if gp>0 else 0)
    sm={'data_scope':'ONLY GitHub 2026-07 minute1.parquet + same-day daily_stock_status.parquet','trading_days':len(paths),'start_cash':START,'final_equity':final,'total_return':total,'max_drawdown':float(eq.drawdown.min()) if len(eq) else 0,'closed_trades':int(len(tr)),'win_rate':float((tr.net_return>0).mean()) if len(tr) else 0,'profit_factor':pf,'avg_trade_return':float(tr.net_return.mean()) if len(tr) else 0,'median_trade_return':float(tr.net_return.median()) if len(tr) else 0,'total_pnl':float(tr.pnl.sum()) if len(tr) else 0,'ending_positions':0,'method':'market review -> core leadership -> 3 playbooks -> T+1 exits -> post-close causal mode weighting/pause + drawdown exposure control'}; (OUT/'summary.json').write_text(json.dumps(sm,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 2026-07 复盘驱动前向回测','',f"月收益 {total:.2%} | 最终权益 {final:,.2f} | 最大回撤 {sm['max_drawdown']:.2%} | 交易 {len(tr)} | 胜率 {sm['win_rate']:.2%} | PF {pf:.3f}",'','## 每日复盘']
    for r in self.reviews: lines += [f"### {r['date']}",f"市场 adv={r['adv_ratio']:.1%}, median={r['median_ret']:.2%}, +5%={r['up5']}, -5%={r['down5']}, 收盘={r['regime_after_close']}",f"策略 日收益={r['strategy_day_return']:.2%}, 回撤={r['drawdown']:.2%}, 当日仓位上限={r['exposure_cap_used']:.0%}",f"复盘更新 {r['review_update']}；次日基础仓位={r['tomorrow_base_exposure']:.0%}",'']
    lines+=['## 每笔交易']
    for i,t in tr.iterrows(): lines.append(f"{i+1}. {t.code} | {t['mode']} | {t.entry_date} {t.entry_time} {t.entry_price:.3f} -> {t.exit_date} {t.exit_time} {t.exit_price:.3f} | {t.net_return:.2%} | PnL {t.pnl:.2f} | {t.exit_reason}")
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(sm,ensure_ascii=False,indent=2))

if __name__=='__main__': BT().run()
