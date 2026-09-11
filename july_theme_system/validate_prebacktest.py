from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path('2026-07')
SYS = Path('july_theme_system')

trade_files = sorted(ROOT.glob('part-*/date=*/minute1.parquet'), key=lambda p: p.parent.name)
assert trade_files, 'No July minute files found'
trade_dates = [p.parent.name.split('=', 1)[1] for p in trade_files]
assert len(trade_dates) == 23, f'Expected 23 July trading days, got {len(trade_dates)}'

ctx = pd.read_csv(SYS / 'july_2026_theme_timeline.csv', dtype=str)
base_roles = pd.read_csv(SYS / 'daily_core_roles.csv', dtype=str)
additional_roles = pd.read_csv(SYS / 'additional_core_roles.csv', dtype=str)
role_overrides = pd.read_csv(SYS / 'role_tradeability_overrides.csv', dtype=str)
themes = pd.read_csv(SYS / 'daily_theme_priority.csv', dtype=str)
aliases = pd.read_csv(SYS / 'theme_aliases.csv', dtype=str)
exceptions = pd.read_csv(SYS / 'tradeability_exceptions.csv', dtype=str)
identity_overrides = pd.read_csv(SYS / 'identity_overrides.csv', dtype=str)
rules = json.loads((SYS / 'playbook_rules_v1.json').read_text(encoding='utf-8'))

required_ctx = ['review_date','valid_from','source_publish_at','market_phase','strong_themes','weak_or_retreat_themes','core_ladder','earning_effect','loss_effect','next_day_plan','source_url','evidence']
for c in required_ctx:
    assert c in ctx.columns, f'missing context column {c}'
    assert ctx[c].fillna('').str.strip().ne('').all(), f'blank context field {c}'
assert list(ctx.review_date) == trade_dates, 'Timeline must contain each July trading day exactly once in order'
assert ctx.review_date.nunique() == 23

# Point-in-time source check: close reviews become usable only on the next trading day.
pub = pd.to_datetime(ctx.source_publish_at)
review = pd.to_datetime(ctx.review_date)
assert (pub.dt.date == review.dt.date).all(), 'Review source must be published on review_date'
assert (pub.dt.hour >= 15).all(), 'Close-review source must be post-close/late-session; otherwise use a separate intraday feed'
assert ctx.source_url.str.startswith('https://').all(), 'Every context row needs a canonical https source'
expected_next = trade_dates[1:] + ['2026-08-03']
next_by_review = dict(zip(trade_dates, expected_next))
assert list(ctx.valid_from) == expected_next, 'valid_from must be next trading day; prevents same-day hindsight use'

# Frozen rules and leakage barriers.
assert rules['effective_policy'] == 'frozen_before_backtest'
assert rules['data_policy']['old_backtest_forbidden'] is True
assert rules['data_policy']['august_forbidden'] is True
assert rules['data_policy']['same_day_new_theme'].startswith('not tradable')
assert rules['execution']['t_plus_one'] is True
assert rules['learning']['single_trade_threshold_tuning_forbidden'] is True
assert rules['learning']['july_end_retuning_forbidden'] is True
assert rules['learning']['rule_change_min_closed_samples'] >= 8

# Core-role schema. Keep historical observation facts, but create a separate EFFECTIVE executable universe.
required_roles = ['review_date','valid_from','theme','role','stock_name','stock_code','tradable_next_day','evidence_url']
for df, label in [(base_roles,'base roles'),(additional_roles,'additional roles')]:
    for c in required_roles:
        assert c in df.columns, f'{label} missing {c}'
        assert df[c].fillna('').str.strip().ne('').all(), f'{label} blank {c}'
    assert df.stock_code.str.fullmatch(r'\d{6}').all(), f'{label} stock_code must be six digits'
    assert df.evidence_url.str.startswith('https://').all(), f'{label} evidence URL missing'
    for r in df.itertuples(index=False):
        assert r.review_date in next_by_review, f'{label}: review date outside July review chain {r.review_date}'
        assert r.valid_from == next_by_review[r.review_date], f'{label}: non-causal review/valid pair {r.review_date}->{r.valid_from}'

allowed_roles = {'market_high','theme_leader','capacity_core','trend_core','elastic_core','supplement','follower','negative_anchor'}
assert set(base_roles.role).issubset(allowed_roles)
assert set(additional_roles.role).issubset(allowed_roles)

# Additional cores may complement missing local leaders but cannot overwrite a same-key historical row silently.
base_keys=set(zip(base_roles.valid_from,base_roles.theme,base_roles.role,base_roles.stock_code))
add_keys=set(zip(additional_roles.valid_from,additional_roles.theme,additional_roles.role,additional_roles.stock_code))
assert not (base_keys & add_keys), f'duplicate additional role keys: {list(base_keys & add_keys)[:10]}'
roles=pd.concat([base_roles,additional_roles],ignore_index=True)

# Tradeability overrides preserve public historical facts while explicitly disabling local execution where minute data is absent.
required_override=['valid_from','stock_code','tradable_next_day','reason','evidence_url']
for c in required_override:
    assert c in role_overrides.columns
    assert role_overrides[c].fillna('').str.strip().ne('').all(), f'blank role override {c}'
assert role_overrides.stock_code.str.fullmatch(r'\d{6}').all()
assert role_overrides.evidence_url.str.startswith('https://').all()
assert set(role_overrides.tradable_next_day.astype(int)).issubset({0}), 'role overrides may only restrict, never manufacture tradability'
for o in role_overrides.itertuples(index=False):
    mask=(roles.valid_from==o.valid_from)&(roles.stock_code==o.stock_code)
    assert mask.any(), f'role override has no historical role row: {o.valid_from} {o.stock_code}'
    roles.loc[mask,'tradable_next_day']=o.tradable_next_day

assert (roles.loc[roles.role == 'negative_anchor','tradable_next_day'].astype(int) == 0).all(), 'negative anchors must be untradeable'

# Institutional or local-data restrictions are distinct from negative feedback.
exc_keys = set(zip(exceptions.valid_from, exceptions.stock_code))
ovr_keys = set(zip(role_overrides.valid_from, role_overrides.stock_code))
for r in roles[roles.tradable_next_day.astype(int) == 0].itertuples(index=False):
    if r.role == 'negative_anchor':
        continue
    assert (r.valid_from, r.stock_code) in exc_keys or (r.valid_from, r.stock_code) in ovr_keys, f'Unexplained non-tradable core: {r.valid_from} {r.stock_code} {r.role}'
assert exceptions.reason.fillna('').str.strip().ne('').all()
assert exceptions.evidence_url.str.startswith('https://').all()

# Canonical identity corrections remain explicit and auditable.
assert identity_overrides.stock_code.str.fullmatch(r'\d{6}').all()
assert identity_overrides.canonical_name.fillna('').str.strip().ne('').all()
assert identity_overrides.identity_source_url.str.startswith('https://').all()
assert ((identity_overrides.stock_code == '300364') & (identity_overrides.canonical_name == '中文在线')).any(), '300364 canonical identity correction missing'

# Theme lifecycle/priority is machine-readable rather than hidden in prose.
required_themes = ['review_date','valid_from','theme','theme_state','priority','tradable_next_day','allowed_modes','reason','evidence_url']
for c in required_themes:
    assert c in themes.columns
    assert themes[c].fillna('').str.strip().ne('').all(), f'blank theme field {c}'
allowed_states = {'new_trial','confirm','accelerate','divergence','repair','retreat','watch'}
assert set(themes.theme_state).issubset(allowed_states), f'unknown theme states: {set(themes.theme_state)-allowed_states}'
assert themes.evidence_url.str.startswith('https://').all()
assert set(themes.valid_from).issubset(set(expected_next)), 'theme valid_from outside point-in-time next-day schedule'
assert (themes.loc[themes.theme_state.isin(['retreat','watch']),'tradable_next_day'].astype(int) == 0).all(), 'retreat/watch themes cannot be tradable'

# A tradable theme used for a JULY decision must connect to >=1 named EFFECTIVELY tradable role.
alias_map = {}
for a in aliases.itertuples(index=False):
    alias_map.setdefault(a.theme, set()).add(a.role_theme)
july_decision_themes = themes[(themes.tradable_next_day.astype(int) == 1) & (themes.valid_from.isin(trade_dates))]
for t in july_decision_themes.itertuples(index=False):
    names = {t.theme} | alias_map.get(t.theme, set())
    x = roles[(roles.valid_from == t.valid_from) & (roles.theme.isin(names)) & (roles.tradable_next_day.astype(int) == 1)]
    assert len(x) >= 1, f'Tradable July theme has no effective named tradable core: {t.valid_from} {t.theme}'

# Every next-day July context has at least one effective tradable named core.
for d in trade_dates[1:]:
    x = roles[(roles.valid_from == d) & (roles.tradable_next_day.astype(int) == 1)]
    assert len(x) >= 1, f'No effective tradable named core for {d}'

# Local data is mandatory ONLY for executable roles. Observation-only facts may be absent from the local minute layer.
missing_tradable=[]; missing_observation=[]
for d in trade_dates[1:]:
    p = next(p for p in trade_files if p.parent.name == f'date={d}')
    codes = pd.read_parquet(p, columns=['code'])['code'].astype(str).str.extract(r'(\d{6})', expand=False).dropna().unique()
    code_set = set(codes)
    day_roles=roles[roles.valid_from==d]
    for r in day_roles.itertuples(index=False):
        if r.stock_code not in code_set:
            rec=(d,r.stock_code,r.stock_name,r.role)
            if int(r.tradable_next_day)==1: missing_tradable.append(rec)
            else: missing_observation.append(rec)
assert not missing_tradable, f'EXECUTABLE core codes missing from same-day GitHub minute data: {missing_tradable[:20]}'

# Ensure each playbook is operational, not prose-only, and theme rows only call declared playbooks.
playbook_names = set(rules['playbooks'])
for name, spec in rules['playbooks'].items():
    if name == 'cash':
        continue
    for key in ['prerequisites','entry_windows','trigger','forbidden','max_single_position']:
        assert key in spec and spec[key] not in (None, '', []), f'{name} missing {key}'
for t in themes.itertuples(index=False):
    for mode in t.allowed_modes.split('|'):
        assert mode in playbook_names, f'Unknown allowed mode {mode} in {t.valid_from} {t.theme}'

print(json.dumps({
    'status': 'PASS',
    'july_trading_days': len(trade_dates),
    'context_rows': len(ctx),
    'theme_state_rows': len(themes),
    'july_tradable_theme_rows': len(july_decision_themes),
    'base_core_role_rows': len(base_roles),
    'additional_core_role_rows': len(additional_roles),
    'effective_core_role_rows': len(roles),
    'unique_named_stocks': int(roles.stock_code.nunique()),
    'tradeability_exceptions': len(exceptions),
    'local_data_tradeability_overrides': len(role_overrides),
    'observation_only_rows_missing_local_minutes': len(missing_observation),
    'all_july_tradable_themes_have_effective_named_cores': True,
    'all_executable_named_codes_exist_in_next_day_github_minute_data': True,
    'point_in_time_valid_from_verified': True,
    'rules_frozen': True,
    'backtest_started': False
}, ensure_ascii=False, indent=2))
