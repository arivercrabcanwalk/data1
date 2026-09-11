import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'generated_v2'


def manifest():
    return json.loads((OUT/'knowledge_manifest_v2.json').read_text(encoding='utf-8'))


def test_all_hard_gates_pass():
    m=manifest()
    failed=[k for k,v in m['gates'].items() if not v]
    assert not failed, failed
    assert m['ready_for_formal_backtest'] is True


def test_complete_calendar_and_timestamped_evidence():
    market=pd.read_csv(OUT/'market_daily.csv')
    pit=pd.read_csv(OUT/'timestamped_theme_evidence.csv')
    audit=pd.read_csv(OUT/'source_audit.csv')
    assert market.date.nunique()==23
    assert pit.date.nunique()==23
    assert audit.date.nunique()==23
    assert pit.published_at.notna().all()


def test_theme_membership_maps_to_codes():
    m=pd.read_csv(OUT/'theme_membership.csv',dtype={'code':str})
    assert m.date.nunique()==23
    assert m.code.str.fullmatch(r'\d{6}').mean()>.99
    assert m.canonical_theme.notna().all()


def test_theme_rank_and_leader_candidates_exist_every_day():
    t=pd.read_csv(OUT/'theme_rank_daily.csv')
    l=pd.read_csv(OUT/'leader_role_candidates.csv',dtype={'code':str})
    assert t.date.nunique()==23
    assert l.date.nunique()>=20
    assert {'theme_score','theme_rank','pit_timestamped_corroborated'}.issubset(t.columns)
    assert {'leadership_score','followers_5m','followers_10m','followers_20m','role_candidate'}.issubset(l.columns)


def test_expectation_is_strictly_forward():
    x=pd.read_csv(OUT/'expectation_book.csv')
    assert len(x)==22
    assert (pd.to_datetime(x.effective_date)>pd.to_datetime(x.asof_date)).all()
    assert not x.uses_future_data.astype(bool).any()


def test_no_pnl_backtest_artifacts():
    names={p.name for p in OUT.iterdir()}
    assert {'trades.csv','fills.csv','equity_curve.csv','summary.json'}.isdisjoint(names)


def test_rule_log_is_nonretroactive_schema():
    x=pd.read_csv(OUT/'rule_change_log.csv')
    required={'decision_date','effective_date','evidence_before_change','old_rule','new_rule','reason','applies_retroactively'}
    assert required.issubset(x.columns)


def test_playbook_catalog_has_all_six_modes():
    p=pd.read_csv(OUT/'playbook_catalog.csv')
    assert set(p.playbook_id)=={'P1','P2','P3','P4','P5','P6'}
