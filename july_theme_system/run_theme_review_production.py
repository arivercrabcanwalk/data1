from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from run_theme_review_backtest import ThemeReviewBT, START, OUT, abs_pct


class ProductionThemeReviewBT(ThemeReviewBT):
    """Production runner.

    Keeps the causal engine logic frozen, but fixes evaluation bookkeeping by
    reusing the original July-31 daily frame (and therefore its true prior
    close) instead of reloading July-31 after last_close has advanced.
    """

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
