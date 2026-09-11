from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('2026-07')
SYS=Path('july_theme_system')
OUT=Path('july_theme_results')


def norm_code(s):
    x=s.astype(str).str.replace(r'\.0$','',regex=True)
    return x.str.extract(r'(\d{6})',expand=False).fillna(x)

def abs_pct(v):
    try:
        x=abs(float(v)); return x/100 if x>1 else x
    except Exception:return np.nan

def build_market_structure():
    prev={}; rows=[]
    paths=sorted(ROOT.glob('part-*/date=*/minute1.parquet'),key=lambda p:p.parent.name)
    for path in paths:
        date=path.parent.name.split('=',1)[1]
        m=pd.read_parquet(path,columns=['code','open','high','low','close','amount']); m['code']=norm_code(m.code)
        d=m.groupby('code').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),amount=('amount','sum')).reset_index(); d['prev_close']=d.code.map(prev); d['ret']=d.close/d.prev_close-1
        st=pd.read_parquet(path.parent/'daily_stock_status.parquet'); st['code']=norm_code(st.code); keep=[c for c in ['code','price_limit_up_pct','price_limit_down_pct','is_st','is_star_st','is_new_listing_initial'] if c in st.columns]; d=d.merge(st[keep],on='code',how='left')
        lup=[]; ldn=[]; touched=[]
        for r in d.itertuples(index=False):
            if not np.isfinite(r.prev_close) or r.prev_close<=0:
                lup.append(False);ldn.append(False);touched.append(False);continue
            up=abs_pct(getattr(r,'price_limit_up_pct',np.nan)); dn=abs_pct(getattr(r,'price_limit_down_pct',np.nan))
            lup.append(bool(np.isfinite(up) and r.close>=r.prev_close*(1+up)*.995))
            touched.append(bool(np.isfinite(up) and r.high>=r.prev_close*(1+up)*.995))
            ldn.append(bool(np.isfinite(dn) and r.close<=r.prev_close*(1-dn)*1.005))
        d['limit_up_close']=lup; d['limit_down_close']=ldn; d['touched_limit_up']=touched; d['broken_board']=d.touched_limit_up&~d.limit_up_close
        r=d.ret.replace([np.inf,-np.inf],np.nan).dropna(); touched_n=int(d.touched_limit_up.sum()); closed_n=int(d.limit_up_close.sum()); broken=int(d.broken_board.sum())
        rows.append({'date':date,'universe_with_prev_close':int(r.notna().sum()),'advancers':int((r>0).sum()),'decliners':int((r<0).sum()),'advancer_ratio':float((r>0).mean()) if len(r) else np.nan,'median_return':float(r.median()) if len(r) else np.nan,'mean_return':float(r.mean()) if len(r) else np.nan,'up_5pct_or_more':int((r>=.05).sum()),'down_5pct_or_more':int((r<=-.05).sum()),'limit_up_close_est':closed_n,'limit_down_close_est':int(d.limit_down_close.sum()),'touched_limit_up_est':touched_n,'broken_board_est':broken,'seal_rate_est':closed_n/touched_n if touched_n else np.nan,'total_amount':float(d.amount.sum())})
        prev=dict(zip(d.code,d.close))
    x=pd.DataFrame(rows); x.to_csv(OUT/'market_structure.csv',index=False,encoding='utf-8-sig'); return x


def load_or_empty(name):
    p=OUT/name
    return pd.read_csv(p) if p.exists() and p.stat().st_size>0 else pd.DataFrame()

def fmtpct(x):
    try:return f'{float(x):.2%}'
    except:return ''

def build_reports():
    market=build_market_structure(); ctx=pd.read_csv(SYS/'july_2026_theme_timeline.csv',dtype=str); themes=pd.read_csv(SYS/'daily_theme_priority.csv',dtype=str); roles=pd.read_csv(SYS/'daily_core_roles.csv',dtype=str)
    trades=load_or_empty('trades.csv'); daily=load_or_empty('daily_review.csv'); pred=load_or_empty('theme_prediction_scores.csv'); missed=load_or_empty('missed_opportunities.csv'); errors=load_or_empty('error_log.csv'); learning=load_or_empty('learning_history.csv'); fills=load_or_empty('fills.csv')
    if not trades.empty:
        mode_stats=trades.groupby('mode').agg(trades=('net_return','size'),win_rate=('net_return',lambda s:float((s>0).mean())),avg_return=('net_return','mean'),median_return=('net_return','median'),total_pnl=('pnl','sum'),avg_MFE=('MFE','mean'),avg_MAE=('MAE','mean')).reset_index()
        gp=trades.assign(g=np.where(trades.pnl>0,trades.pnl,0),l=np.where(trades.pnl<0,-trades.pnl,0)).groupby('mode')[['g','l']].sum().reset_index(); gp['profit_factor']=np.where(gp.l>0,gp.g/gp.l,np.where(gp.g>0,np.inf,0)); mode_stats=mode_stats.merge(gp[['mode','profit_factor']],on='mode',how='left')
    else: mode_stats=pd.DataFrame(columns=['mode','trades','win_rate','avg_return','median_return','total_pnl','avg_MFE','avg_MAE','profit_factor'])
    mode_stats.to_csv(OUT/'playbook_statistics.csv',index=False,encoding='utf-8-sig')

    if not daily.empty:
        daily['date']=pd.to_datetime(daily.date); daily['week']=daily.date.dt.to_period('W-FRI').astype(str)
        w=daily.groupby('week').agg(start_date=('date','min'),end_date=('date','max'),end_equity=('eod_equity','last'),avg_exposure_cap=('max_exposure_allowed','mean'),avg_prediction_hit=('prediction_hit_rate','mean'),entries=('new_entries','sum'),closed_trades=('new_closed_trades','sum')).reset_index()
        w['start_equity']=w.end_equity.shift(1).fillna(1_000_000.0); w['week_return']=w.end_equity/w.start_equity-1
        w.to_csv(OUT/'weekly_review.csv',index=False,encoding='utf-8-sig')
    else: w=pd.DataFrame()

    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    summary['market_structure_days']=int(len(market)); summary['playbook_count']=int(len(mode_stats)); summary['weekly_review_count']=int(len(w)); summary['error_log_count']=int(len(errors)); summary['missed_opportunity_count']=int(len(missed)); summary['fill_count']=int(len(fills))
    (OUT/'monthly_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    lines=['# 2026年7月 题材—核心—复盘驱动前向回测报告','',f"方法版本：{summary.get('method_version','')}",f"月收益率：{fmtpct(summary.get('total_return'))}",f"最大回撤：{fmtpct(summary.get('max_drawdown'))}",f"闭合交易：{summary.get('closed_trades',0)}；胜率：{fmtpct(summary.get('win_rate'))}；Profit Factor：{summary.get('profit_factor',0):.3f}",f"题材预期命中率（仅复盘评分，不用于当月阈值回改）：{fmtpct(summary.get('prediction_hit_rate'))}",'','## 方法边界','- 价格与成交只读GitHub 2026-07分钟行情和同日股票状态。','- 互联网题材/核心信息采用收盘后点时资料，统一从下一交易日生效。','- 7月1日为冷启动观察日；不引入6月30日行情补昨收。','- 交易触发后最早下一分钟open成交；A股T+1；含手续费、印花税、滑点及涨跌停/停牌/新股检查。','- 月内不依据月底结果重调阈值；只允许预先规定的连续亏损降仓/暂停。','']
    for c in ctx.itertuples(index=False):
        date=c.review_date; lines += [f'## {date} 盘后复盘',f'市场：{c.market_phase}；强势：{c.strong_themes}；弱/退潮：{c.weak_or_retreat_themes}',f'核心梯队：{c.core_ladder}',f'赚钱效应：{c.earning_effect}',f'亏钱效应：{c.loss_effect}',f'次日计划：{c.next_day_plan}']
        mm=market[market.date==date]
        if not mm.empty:
            r=mm.iloc[0]; lines.append(f"GitHub行情审计：上涨比例={fmtpct(r.advancer_ratio)}，中位涨跌={fmtpct(r.median_return)}，涨停估算={int(r.limit_up_close_est)}，跌停估算={int(r.limit_down_close_est)}，炸板估算={int(r.broken_board_est)}，成交额={r.total_amount:,.0f}")
        tt=trades[trades.entry_date==date] if not trades.empty and 'entry_date' in trades else pd.DataFrame()
        if not tt.empty:
            for r in tt.itertuples(index=False): lines.append(f"入场：{r.code} {r.name}｜{r.theme}｜{r.mode}｜{r.entry_time} {r.entry_price:.3f}｜最终 {fmtpct(r.net_return)}｜退出 {r.exit_date} {r.exit_reason}")
        else: lines.append('入场：无。空仓/未触发也是计划执行结果。')
        ee=errors[errors.date==date] if not errors.empty and 'date' in errors else pd.DataFrame()
        if not ee.empty:
            for r in ee.itertuples(index=False): lines.append(f"错误记录：{r.code} {r.error_class}；{r.action}")
        lr=learning[learning.date==date] if not learning.empty and 'date' in learning else pd.DataFrame()
        if not lr.empty: lines.append(f"学习更新：{lr.iloc[-1].events}")
        lines.append('')
    lines += ['# 每一笔交易']
    if trades.empty: lines.append('本月无闭合交易。')
    else:
        for i,r in enumerate(trades.itertuples(index=False),1): lines.append(f"{i}. {r.code} {r.name}｜{r.theme}/{r.role}｜{r.mode}｜{r.entry_date} {r.entry_time} {r.entry_price:.3f} -> {r.exit_date} {r.exit_time} {r.exit_price:.3f}｜收益 {fmtpct(r.net_return)}｜PnL {r.pnl:,.2f}｜MFE {fmtpct(r.MFE)}｜MAE {fmtpct(r.MAE)}｜退出：{r.exit_reason}")
    lines += ['','# 周复盘']
    if not w.empty:
        for r in w.itertuples(index=False): lines.append(f"{r.week}｜收益 {fmtpct(r.week_return)}｜入场 {int(r.entries)}｜闭合 {int(r.closed_trades)}｜平均允许仓位 {fmtpct(r.avg_exposure_cap)}｜题材预期命中 {fmtpct(r.avg_prediction_hit)}")
    lines += ['','# 模式统计']
    if not mode_stats.empty:
        for r in mode_stats.itertuples(index=False): lines.append(f"{r.mode}｜{int(r.trades)}笔｜胜率 {fmtpct(r.win_rate)}｜平均 {fmtpct(r.avg_return)}｜PF {r.profit_factor:.3f}｜平均MFE {fmtpct(r.avg_MFE)}｜平均MAE {fmtpct(r.avg_MAE)}｜PnL {r.total_pnl:,.2f}")
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'PASS','report':str(OUT/'REPORT.md'),'market_structure_days':len(market),'weekly_rows':len(w),'playbook_rows':len(mode_stats)},ensure_ascii=False))

if __name__=='__main__': build_reports()
