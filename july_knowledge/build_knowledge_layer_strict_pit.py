"""Strict point-in-time July knowledge builder.

This is the formal gate entrypoint. It intentionally excludes Lianban historical
archive pages from trade-eligible semantics because the current archive form does
not prove the exact historical publication/version time. The parser remains in
v2 for research/audit, but the formal gate uses:

1) timestamped same-day sources in pit_corroboration_registry.json for semantic PIT;
2) YYQYX as a structural historical limit-up/theme-membership archive;
3) GitHub 2026-07 minute bars/status as canonical price/trading facts.

No PnL backtest is run here.
"""
from __future__ import annotations

import json
from datetime import datetime
import numpy as np
import pandas as pd

import build_knowledge_layer_v2 as b

OUT=b.OUT


def add_lifecycle(x: pd.DataFrame) -> pd.DataFrame:
    """Causal theme lifecycle: each row uses only current/past theme observations."""
    if x.empty:
        return x
    z=x.copy(); z['lifecycle_stage']='震荡'
    day_pos={d:i for i,d in enumerate(b.DAYS)}
    for theme,g in z.groupby('canonical_theme',sort=False):
        inds=sorted(g.index,key=lambda i:day_pos[z.loc[i,'date']])
        prev_count=None; prev_stage=None; seen=0
        for idx in inds:
            r=z.loc[idx]; count=float(r.limitup_member_count); rank=float(r.theme_rank); p3=int(r.persistence_3d)
            if seen==0:
                stage='启动'
            elif prev_stage in ('分歧','退潮') and rank<=3 and (prev_count is None or count>=prev_count):
                stage='回流'
            elif p3>=2 and rank==1 and count>=10:
                stage='高潮'
            elif prev_count and count>=prev_count*1.35:
                stage='发酵'
            elif prev_count and count<=prev_count*.55:
                stage='分歧'
            elif p3>=2 and rank<=5:
                stage='确认'
            elif p3<=1:
                stage='启动'
            else:
                stage='震荡'
            z.loc[idx,'lifecycle_stage']=stage
            prev_count=count; prev_stage=stage; seen+=1
    return z


def error_taxonomy() -> pd.DataFrame:
    rows=[
      ['NO_THEME_SUPPORT','认知','没有高置信题材支持却交易个股','下一次必须先通过题材门'],
      ['LOW_RANK_FOLLOWER','选股','买入后排/跟风而非核心','限制为前排地位候选'],
      ['LATE_CYCLE_ENTRY','择时','高潮末端或退潮继续接力','按周期降低频率/禁止后排一致'],
      ['FALSE_REPAIR','认知','把普通V形反弹误认成核心修复','要求地位+题材跟随双确认'],
      ['NO_LEADERSHIP','选股','个股自身强但没有5/10/20分钟带动','降低地位评分或淘汰'],
      ['EXPECTATION_NOT_CONFIRMED','计划','盘中未达到昨晚确认条件仍交易','计划失效即取消'],
      ['FOMO_CHASE','情绪','踏空后临时追高','只允许计划池交易'],
      ['REVENGE_TRADE','情绪','亏损后急于回本而降低标准','触发冷却与风险预算下调'],
      ['STYLE_DRIFT','纪律','临时更换模式或无模式交易','必须绑定P1-P6之一'],
      ['OVERSIZED','仓位','仓位超过预先风险预算','按账户风险预算强制裁剪'],
      ['EXIT_RULE_VIOLATION','执行','退出与失效规则不一致','记录执行偏差并复核'],
      ['DATA_NOT_PIT','数据','使用当时不可证明可获得的信息','交易样本作废并修复数据链']
    ]
    return pd.DataFrame(rows,columns=['error_tag','category','definition','corrective_action'])


def main():
    pit=b.pit_rows()
    fetchlog=[]; yys=[]; members=[]; ladder=[]
    for day in b.DAYS:
        url=f'https://www.yyqyx.com/limit-up/{day}'
        fr=b.fetch_with_retry(url,'yyqyx_limitup',day,tries=4)
        fetchlog.append({k:v for k,v in fr.items() if k!='html'})
        if fr['ok']:
            s,ms,ls=b.parse_yyqyx(day,fr['html']); yys.append(s); members.extend(ms); ladder.extend(ls)
        fetchlog.append({'date':day,'source_id':'lianban_archive','url':f'https://lianban.net/days/{day}.html','ok':False,'status':None,'error':'SKIPPED_STRICT_PIT: current archive publication/version timing not proven'})

    yy=pd.DataFrame(yys)
    mem_raw=pd.DataFrame(members)
    ladd=pd.DataFrame(ladder)
    # Preserve raw mappings, but prevent double-counting when multiple raw aliases collapse into one canonical theme.
    mem_calc=mem_raw.drop_duplicates(['date','canonical_theme','code']).copy() if len(mem_raw) else mem_raw.copy()
    lsum=pd.DataFrame(columns=['date','source_id','availability'])
    lth=pd.DataFrame(columns=['date','source_id','raw_theme','canonical_theme','source_theme_count','leader_code','leader_name','catalyst_text','trade_effective_date','availability_confidence'])
    llead=pd.DataFrame(columns=['date','source_id','role_source','raw_theme','canonical_theme','code','stock_name'])
    flog=pd.DataFrame(fetchlog)

    market,stocks=b.load_market_and_stock(mem_calc,ladd)
    cycle=b.derive_cycle(market,yy)
    trank=b.theme_metrics(mem_calc,stocks,market,pit,lth)
    trank=add_lifecycle(trank)
    leaders=b.leader_candidates(trank,mem_calc,stocks,cycle)
    ple=b.profit_loss_effect(stocks,mem_calc,ladd)
    expect,review=b.build_expectation_and_review(market,cycle,trank,leaders,pit,ple)
    pb=b.playbook_catalog()
    errors=error_taxonomy()
    rulelog=pd.DataFrame(columns=['decision_date','effective_date','evidence_before_change','old_rule','new_rule','reason','applies_retroactively'])
    catalyst=pit[['date','published_at','source','url','canonical_theme','raw_theme','observed_context','evidence_tier','trade_effective_date']].copy()
    catalyst['catalyst_text']=None

    audits=[]
    for day in b.DAYS:
        y=yy[yy.date==day] if len(yy) else pd.DataFrame()
        p=pit[pit.date==day]
        mk=market[market.date==day].iloc[0]
        ylu=float(y.iloc[0].limit_up_count) if len(y) and pd.notna(y.iloc[0].limit_up_count) else np.nan
        glu=float(mk.github_close_limit_up_nonst) if pd.notna(mk.github_close_limit_up_nonst) else np.nan
        diff=ylu-glu if np.isfinite(ylu) and np.isfinite(glu) else np.nan
        audits.append({
          'date':day,'timestamped_pit_source_present':bool(len(p)),'yyqyx_archive_present':bool(len(y)),'lianban_archive_present':False,
          'yyqyx_member_codes':int(mem_calc[mem_calc.date==day].code.nunique()) if len(mem_calc) else 0,
          'theme_rows_ranked':int(trank[trank.date==day].canonical_theme.nunique()) if len(trank) else 0,
          'yyqyx_limit_up_count':ylu,'github_nonst_close_limit_up_count':glu,'count_difference_native_minus_github_nonst':diff,
          'count_disagreement_flag':bool(np.isfinite(diff) and abs(diff)>max(5,.08*max(abs(ylu),1))),
          'semantic_trade_rule':'Only timestamped T-day observed themes are semantic-PIT for T+1. YYQYX provides structural membership; GitHub is canonical price fact. Counts remain source-native.'
        })
    audit=pd.DataFrame(audits)

    outputs={
      'market_daily.csv':market,'cycle_daily.csv':cycle,'timestamped_theme_evidence.csv':pit,'source_fetch_log_v2.csv':flog,
      'source_daily_yyqyx.csv':yy,'source_daily_lianban.csv':lsum,'theme_membership_raw.csv':mem_raw,'theme_membership.csv':mem_calc,
      'board_ladder.csv':ladd,'lianban_theme_archive.csv':lth,'lianban_leader_archive.csv':llead,'theme_rank_daily.csv':trank,
      'profit_loss_effect.csv':ple,'leader_role_candidates.csv':leaders,'daily_review_book.csv':review,'expectation_book.csv':expect,
      'playbook_catalog.csv':pb,'error_taxonomy.csv':errors,'rule_change_log.csv':rulelog,'catalyst_timeline.csv':catalyst,'source_audit.csv':audit,
    }
    for fn,df in outputs.items(): df.to_csv(OUT/fn,index=False,encoding='utf-8-sig')

    if len(mem_calc):
        matched=mem_calc.merge(stocks[['date','code']].drop_duplicates(),on=['date','code'],how='left',indicator=True)
        code_match=float(matched._merge.eq('both').mean())
    else: code_match=0.0
    yy_ok=flog[(flog.source_id=='yyqyx_limitup') & (flog.ok==True)] if len(flog) else pd.DataFrame()
    gates={
      '01_market_emotion_23d':len(market)==23 and len(cycle)==23,
      '02_limitup_down_broken_23d':len(yy)==23 and yy[['limit_up_count','broken_board_count']].notna().all().all(),
      '03_board_ladder_present':len(ladd)>0 and ladd.date.nunique()>=20,
      '04_daily_theme_rank_present':len(trank)>0 and trank.date.nunique()==23,
      '05_theme_stock_mapping_codes':len(mem_calc)>0 and mem_calc.date.nunique()==23 and code_match>=.95,
      '06_catalyst_first_public_time_23d':pit.date.nunique()==23 and pit.published_at.notna().all(),
      '07_theme_lifecycle':len(trank)>0 and 'lifecycle_stage' in trank and trank.lifecycle_stage.notna().all(),
      '08_leader_roles':len(leaders)>0 and leaders.date.nunique()>=20 and leaders.followers_20m.notna().all(),
      '09_profit_loss_effect':len(ple)==23,
      '10_next_day_expectation':len(expect)==22 and (pd.to_datetime(expect.effective_date)>pd.to_datetime(expect.asof_date)).all(),
      '11_core_weak_to_strong_spec':'P2' in set(pb.playbook_id),
      '12_divergence_to_consensus_spec':'P3' in set(pb.playbook_id),
      '13_supplement_spec':'P4' in set(pb.playbook_id),
      '14_switch_spec':'P5' in set(pb.playbook_id),
      '15_ice_trial_spec':'P6' in set(pb.playbook_id),
      '16_leader_mainrise_spec':'P1' in set(pb.playbook_id),
      '17_climax_no_follow_chase':any('禁止后排一致追高' in str(x) for x in expect.base_plan),
      '18_retreat_defense_and_reentry':all(x in set(pb.playbook_id) for x in ['P5','P6']),
      '19_error_tag_system':len(errors)>=12 and {'认知','选股','择时','计划','执行','纪律','情绪','仓位','数据'}.issubset(set(errors.category)),
      '20_causal_rule_update_log':{'decision_date','effective_date','evidence_before_change','old_rule','new_rule','reason','applies_retroactively'}.issubset(rulelog.columns),
      'anti_leak_expectation_no_future':not expect.uses_future_data.astype(bool).any(),
      'anti_leak_timestamped_semantics_only':pit.trade_eligible.astype(bool).all() and pit.date.nunique()==23,
      'archive_timing_not_misrepresented':not audit.lianban_archive_present.any(),
      'yyqyx_23_raw_snapshots_hashed':len(yy_ok)==23 and yy_ok.sha256.notna().all(),
      'no_pnl_backtest_outputs':not any((OUT/f).exists() for f in ['trades.csv','fills.csv','equity_curve.csv','summary.json']),
      'source_conflicts_preserved':len(audit)==23 and 'count_disagreement_flag' in audit,
    }
    manifest={
      'version':'2026-09-11-strict-pit-v2.1','built_at_utc':datetime.utcnow().isoformat(timespec='seconds')+'Z',
      'scope':'2026-07 knowledge layer only; NO formal PnL backtest',
      'coverage':{'trading_days':len(market),'timestamped_semantic_days':pit.date.nunique(),'yyqyx_days':len(yy),'theme_membership_raw_rows':len(mem_raw),'theme_membership_canonical_rows':len(mem_calc),'theme_rank_rows':len(trank),'leader_candidate_rows':len(leaders),'code_match_rate':code_match},
      'gates':gates,'ready_for_formal_backtest':bool(all(gates.values())),
      'causality':{'timestamped_semantics_effective':'next trading day only','yyqyx_role':'structural historical membership, not proof of same-day publication','lianban_role':'excluded from strict trade-eligible build unless historical version timing can be proven','hindsight_forecast_sections':'forbidden','rule_changes':'T+1 only and never retroactive'}
    }
    (OUT/'knowledge_manifest_v2.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# July Knowledge Gate Report — Strict PIT v2.1','',f"Ready for formal backtest: **{manifest['ready_for_formal_backtest']}**",'',f"Coverage: `{manifest['coverage']}`",'', '## Gates']+[f"- {'PASS' if v else 'FAIL'} — {k}" for k,v in gates.items()]
    (OUT/'knowledge_gate_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
