from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_theme_review_backtest import ThemeReviewBT, START, OUT, abs_pct, norm_code


class ProductionThemeReviewBT(ThemeReviewBT):
    """Production runner with audit fixes only; strategy rules remain frozen."""

    def role_rows(self,date):
        """Return point-in-time role universe with explicit data-coverage overrides.

        Historical facts are never deleted: roles missing from the local minute
        layer remain observation-only. Additional executable peers must have a
        source published no later than the prior close and are stored separately.
        """
        x=super().role_rows(date).copy()
        add_path=Path('july_theme_system/additional_core_roles.csv')
        if add_path.exists():
            add=pd.read_csv(add_path,dtype=str)
            add=add[add.valid_from==date].copy()
            if not add.empty:
                add['stock_code']=norm_code(add.stock_code)
                add['stock_name']=add.apply(lambda r:self.name_override.get(r.stock_code,r.stock_name),axis=1)
                x=pd.concat([x,add[x.columns]],ignore_index=True)
        ov_path=Path('july_theme_system/role_tradeability_overrides.csv')
        if ov_path.exists() and not x.empty:
            ov=pd.read_csv(ov_path,dtype=str)
            ov=ov[ov.valid_from==date]
            for r in ov.itertuples(index=False):
                mask=x.stock_code.eq(str(r.stock_code).zfill(6))
                x.loc[mask,'tradable_next_day']=str(r.tradable_next_day)
        return x.drop_duplicates(['valid_from','theme','role','stock_code'],keep='last')

    def buy(self,date,obs_t,fill_t,row,raw_bar,st,daily,mark_map,cap,mode):
        # Candidate rows come from DataFrame.itertuples(), so use attribute access.
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
        self.pos[code]={'code':code,'name':row.name,'theme':row.theme,'role':row.role,'mode':mode,'entry_date':date,'obs_time':obs_t,'entry_time':fill_t,'entry_price':px,'shares':sh,'basis':gross+fee,'hold_days':0,'max_price':px,'min_price':px,'last_mark':px,'entry_phase':self.context_day(date).market_phase,'theme_state_entry':row.theme_state,'priority_entry':int(row.priority),'scale_entry':scale}
        self.entered_on_day[date].add(code)
        self.fills.append({'date':date,'observation_time':obs_t,'time':fill_t,'code':code,'name':row.name,'theme':row.theme,'role':row.role,'side':'BUY','raw_price':float(raw_bar.open),'fill_price':px,'shares':sh,'mode':mode,'reason':'all_playbook_gates_passed'})
        if self.learn[mode]['half_slots']>0: self.learn[mode]['half_slots']-=1
        return True,''

    def export_all(self, final, bench):
        t=pd.DataFrame(self.trades); f=pd.DataFrame(self.fills); d=pd.DataFrame(self.daily); e=pd.DataFrame(self.eq)
        s=pd.DataFrame(self.signals); pr=pd.DataFrame(self.predictions); mi=pd.DataFrame(self.missed); er=pd.DataFrame(self.errors); lh=pd.DataFrame(self.learning_hist)
        for name,x in [('trades',t),('fills',f),('daily_review',d),('equity_curve',e),('candidate_signal_audit',s),('theme_prediction_scores',pr),('missed_opportunities',mi),('error_log',er),('learning_history',lh)]:
            x.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
        gp=float(t.loc[t.pnl>0,'pnl'].sum()) if len(t) else 0.0
        gl=float(-t.loc[t.pnl<0,'pnl'].sum()) if len(t) else 0.0
        pf=gp/gl if gl>0 else (999.0 if gp>0 else 0.0)
        summary={
            'method_version':self.rules['version'],
            'data_scope':'GitHub 2026-07 minute1 + same-day status; web theme/core context only after prior close valid_from',
            'cold_start':'2026-07-01 is observation-only because GitHub July-only scope has no 2026-06-30 prior close',
            'start_equity':START,
            'final_equity_mark_to_market':final,
            'total_return':final/START-1,
            'max_drawdown':float(min(x['drawdown'] for x in self.eq)) if self.eq else 0.0,
            'benchmark_equal_weight_return_from_available_july_prev_closes':bench-1,
            'closed_trades':int(len(t)),
            'open_positions_month_end':int(len(self.pos)),
            'win_rate':float((t.net_return>0).mean()) if len(t) else 0.0,
            'profit_factor':pf,
            'total_closed_pnl':float(t.pnl.sum()) if len(t) else 0.0,
            'prediction_hit_rate':float(pr.prediction_hit.mean()) if len(pr) else np.nan,
            'missed_plan_universe_winners':int(len(mi)),
            'rule_threshold_changes_during_july':0,
            'learning_policy':'causal half-size/pause only; no threshold retuning unless >=8 closed samples and repeated mechanism',
            'month_end_measurement':'sell at 2026-07-31 final minute only if status/price permits; otherwise mark to close'
        }
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        open_rows=[]
        for p in self.pos.values():
            mark=self.last_close.get(p['code'],p['last_mark'])
            open_rows.append({**p,'month_end_mark':mark,'unrealized_return':mark/p['entry_price']-1})
        pd.DataFrame(open_rows).to_csv(OUT/'open_positions.csv',index=False,encoding='utf-8-sig')
        print(json.dumps(summary,ensure_ascii=False,indent=2))

    def run(self):
        bench=1.0; prev_eq=START; last_bundle=None
        paths=self.paths()
        for path in paths:
            date=path.parent.name.split('=',1)[1]
            pre_pause={m:s['pause_days'] for m,s in self.learn.items()}
            new_trade_start=len(self.trades)
            df,daily,st,snaps=self.load_day(path)
            last_bundle=(date,df,daily,st,snaps)
            close_map=dict(zip(daily.code,daily.close))

            self.process_exits(date,df,daily,st,snaps)
            self.process_entries(date,df,daily,st,snaps)
            self.update_intraday_extremes(df,date,until=None)

            eq=self.equity(close_map)
            self.peak=max(self.peak,eq)
            dd=eq/self.peak-1
            dayret=eq/prev_eq-1
            r=daily.ret.dropna()
            bret=float(r.mean()) if len(r) else 0.0
            bench*=1+bret

            self.score_predictions(date,daily)
            self.collect_missed(date,daily)
            self.apply_learning_after_close(date,new_trade_start)
            phase,ps=self.phase_spec(date)
            pred_today=[x for x in self.predictions if x['date']==date]
            hitrate=float(np.mean([x['prediction_hit'] for x in pred_today])) if pred_today else np.nan
            self.daily.append({
                'date':date,'prior_close_market_phase':phase,'max_exposure_allowed':float(ps['max_equity_exposure']),
                'allowed_modes':'|'.join(ps['allowed']),'eod_equity':eq,'strategy_day_return':dayret,'drawdown':dd,
                'benchmark_day_equal_weight':bret,'benchmark_cumulative':bench-1,'positions_eod':len(self.pos),
                'prediction_hit_rate':hitrate,'new_closed_trades':len(self.trades)-new_trade_start,
                'new_entries':len(self.entered_on_day[date]),'learning_state_after_close':json.dumps(self.learn,ensure_ascii=False)
            })
            self.eq.append({'date':date,'equity':eq,'day_return':dayret,'drawdown':dd,'benchmark_cumulative':bench-1})
            prev_eq=eq
            self.decrement_pause_after_day(pre_pause)
            self.last_close=close_map

        # Measurement only. Reuse the cached ORIGINAL July-31 daily frame, whose
        # prev_close was computed before last_close advanced to July-31.
        date,df,daily,st,snaps=last_bundle
        for code in list(self.pos):
            z=df[df.code==code]
            if z.empty: continue
            b=z.iloc[-1].copy()
            b['open']=b['close']
            b['volume']=max(float(b.get('volume',1)),1)
            prev=float(daily.loc[code].prev_close) if code in daily.index and pd.notna(daily.loc[code].prev_close) else np.nan
            sr=self.status_row(st,code)
            down=abs_pct(sr.get('price_limit_down_pct',np.nan)) if sr is not None else np.nan
            locked=np.isfinite(down) and np.isfinite(prev) and float(b['close'])<=prev*(1-down)*1.002
            if not locked:
                self.sell(date,'15:00',code,b,st,daily,'evaluation_month_end')

        final_mark={c:self.last_close.get(c,p['last_mark']) for c,p in self.pos.items()}
        final=self.equity(final_mark)
        if self.eq:
            prior=self.eq[-2]['equity'] if len(self.eq)>1 else START
            self.eq[-1]['equity']=final
            self.eq[-1]['day_return']=final/prior-1
            peak=START
            for row in self.eq:
                peak=max(peak,row['equity'])
                row['drawdown']=row['equity']/peak-1
            self.daily[-1]['eod_equity']=final
            self.daily[-1]['strategy_day_return']=final/prior-1
            self.daily[-1]['drawdown']=self.eq[-1]['drawdown']
            self.daily[-1]['positions_eod']=len(self.pos)
        self.export_all(final,bench)


if __name__=='__main__':
    ProductionThemeReviewBT().run()
