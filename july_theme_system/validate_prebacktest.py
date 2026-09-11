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
roles = pd.read_csv(SYS / 'daily_core_roles.csv', dtype=str)
themes = pd.read_csv(SYS / 'daily_theme_priority.csv', dtype=str)
aliases = pd.read_csv(SYS / 'theme_aliases.csv', dtype=str)
exceptions = pd.read_csv(SYS / 'tradeability_exceptions.csv', dtype=str)
overrides = pd.read_csv(SYS / 'identity_overrides.csv', dtype=str)
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

# Core-role schema and causality.
required_roles = ['review_date','valid_from','theme','role','stock_name','stock_code','tradable_next_day','evidence_url']
for c in required_roles:
    assert c in roles.columns
    assert roles[c].fillna('').str.strip().ne('').all(), f'blank core field {c}'
allowed_roles = {'market_high','theme_leader','capacity_core','trend_core','elastic_core','supplement','follower','negative_anchor'}
assert set(roles.role).issubset(allowed_roles), f'unknown roles: {set(roles.role)-allowed_roles}'
assert roles.stock_code.str.fullmatch(r'\d{6}').all(), 'stock_code must be six digits'
assert roles.evidence_url.str.startswith('https://').all()
assert (roles.loc[roles.role == 'negative_anchor','tradable_next_day'].astype(int) == 0).all(), 'negative anchors must be untradeable'

# Institutional restrictions are separate from negative feedback.
exc_keys = set(zip(exceptions.valid_from, exceptions.stock_code))
for r in roles[roles.tradable_next_day.astype(int) == 0].itertuples(index=False):
    if r.role == 'negative_anchor':
        continue
    assert (r.valid_from, r.stock_code) in exc_keys, f'Unexplained non-tradable core: {r.valid_from} {r.stock_code} {r.role}'
assert exceptions.reason.fillna('').str.strip().ne('').all()
assert exceptions.evidence_url.str.startswith('https://').all()

# Canonical identity corrections remain explicit and auditable.
assert overrides.stock_code.str.fullmatch(r'\d{6}').all()
assert overrides.canonical_name.fillna('').str.strip().ne('').all()
assert overrides.identity_source_url.str.startswith('https://').all()
assert ((overrides.stock_code == '300364') & (overrides.canonical_name == '中文在线')).any(), '300364 canonical identity correction missing'

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

# A tradable theme must connect to at least one named tradable role, exact or via an explicit alias.
alias_map = {}
for a in aliases.itertuples(index=False):
    alias_map.setdefault(a.theme, set()).add(a.role_theme)
for t in themes[themes.tradable_next_day.astype(int) == 1].itertuples(index=False):
    names = {t.theme} | alias_map.get(t.theme, set())
    x = roles[(roles.valid_from == t.valid_from) & (roles.theme.isin(names)) & (roles.tradable_next_day.astype(int) == 1)]
    assert len(x) >= 1, f'Tradable theme has no named tradable core: {t.valid_from} {t.theme}'

# Every next-day July context has at least one tradable named core.
for d in trade_dates[1:]:
    x = roles[(roles.valid_from == d) & (roles.tradable_next_day.astype(int) == 1)]
    assert len(x) >= 1, f'No tradable named core for {d}'

# Verify every named code actually exists in the GitHub minute parquet on its valid day.
missing = []
for d in trade_dates[1:]:
    p = next(p for p in trade_files if p.parent.name == f'date={d}')
    codes = pd.read_parquet(p, columns=['code'])['code'].astype(str).str.extract(r'(\d{6})', expand=False).dropna().unique()
    code_set = set(codes)
    for code in roles.loc[roles.valid_from == d, 'stock_code'].unique():
        if code not in code_set:
            missing.append((d, code))
assert not missing, f'Named core codes missing from same-day GitHub minute data: {missing[:20]}'

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
    'core_role_rows': len(roles),
    'unique_named_stocks': int(roles.stock_code.nunique()),
    'tradeability_exceptions': len(exceptions),
    'identity_overrides': len(overrides),
    'all_tradable_themes_have_named_cores': True,
    'all_named_codes_exist_in_next_day_github_minute_data': True,
    'point_in_time_valid_from_verified': True,
    'rules_frozen': True,
    'backtest_started': False
}, ensure_ascii=False, indent=2))
