from __future__ import annotations
import inspect, json
from pathlib import Path
import pandas as pd
from run_theme_review_backtest import ThemeReviewBT, FILL
from run_theme_review_production import ProductionThemeReviewBT

SYS=Path('july_theme_system')
rules=json.loads((SYS/'playbook_rules_v1.json').read_text(encoding='utf-8'))

# 1) Static source boundaries: execution engine must not read old strategy folders or August price data.
src=(SYS/'run_theme_review_backtest.py').read_text(encoding='utf-8')+(SYS/'run_theme_review_production.py').read_text(encoding='utf-8')
assert "Path('2026-07')" in src, 'July ROOT not explicit'
assert "Path('2026-08')" not in src and '2026-08/' not in src, 'August price path forbidden'
assert "Path('backtest')" not in src and "Path(\"backtest\")" not in src, 'Old backtest path forbidden'

# 2) Confirmation minute and fill minute are distinct and forward-moving.
assert FILL == {'09:45':'09:46','10:00':'10:01','10:30':'10:31'}
assert rules['execution']['confirmation_then_fill']=='next-minute open'
assert rules['execution']['t_plus_one'] is True

pbt=ProductionThemeReviewBT()

# 3) Production limit handling uses exact 0.01 tick-rounded limit prices.
status=pd.DataFrame([{
    'code':'000001','is_st':False,'is_star_st':False,'is_new_listing_initial':False,'is_suspended':False,
    'price_limit_up_pct':10.0,'price_limit_down_pct':-10.0
}]).set_index('code',drop=False)
ok,reason=pbt.can_trade_stock(status,'000001','SELL',9.01,10.0,pd.Series({'volume':100.0}))
assert ok, 'One tick above a 9.00 down-limit must be sellable when minute volume is positive'
ok,reason=pbt.can_trade_stock(status,'000001','SELL',9.00,10.0,pd.Series({'volume':100.0}))
assert not ok and reason=='locked_limit_down', 'Exact down-limit must be blocked conservatively'
ok,reason=pbt.can_trade_stock(status,'000001','BUY',10.99,10.0,pd.Series({'volume':100.0}))
assert ok, 'One tick below an 11.00 up-limit must be buyable when minute volume is positive'
ok,reason=pbt.can_trade_stock(status,'000001','BUY',11.00,10.0,pd.Series({'volume':100.0}))
assert not ok and reason=='locked_limit_up'

# 4) New listing is observation-only, never accidentally traded.
status_new=status.copy(); status_new.loc['000001','is_new_listing_initial']=True
ok,reason=pbt.can_trade_stock(status_new,'000001','BUY',10.5,10.0,pd.Series({'volume':100.0}))
assert not ok and reason=='is_new_listing_initial'

# 5) Same-day sell after entry is prohibited by inherited sell() itself.
pbt.pos['000001']={'code':'000001','name':'TEST','theme':'T','role':'theme_leader','mode':'leader_weak_to_strong','entry_date':'2026-07-02','obs_time':'09:45','entry_time':'09:46','entry_price':10.0,'shares':100,'basis':1000.3,'hold_days':0,'max_price':10.0,'min_price':10.0,'last_mark':10.0,'entry_phase':'x','theme_state_entry':'confirm','priority_entry':1,'scale_entry':1.0}
daily=pd.DataFrame([{'code':'000001','prev_close':10.0}]).set_index('code',drop=False)
bar=pd.Series({'open':10.2,'volume':100.0})
ok,reason=pbt.sell('2026-07-02','10:01','000001',bar,status,daily,'unit_test')
assert not ok and reason=='T+1', 'Same-day sell must be rejected'
pbt.pos.clear(); pbt.trades.clear(); pbt.fills.clear()

# 6) Production buy path works with itertuples candidates.
pbt.cash=100000.0
cand=pd.DataFrame([{'code':'000001','name':'TEST','theme':'T','role':'theme_leader','theme_state':'confirm','priority':1,'allowed_modes':['leader_weak_to_strong'],'theme_reason':'test'}]).itertuples(index=False)
row=next(cand)
daily2=pd.DataFrame([{'code':'000001','prev_close':10.0}]).set_index('code',drop=False)
bar2=pd.Series({'open':10.1,'volume':1000.0})
pbt.context_day=lambda date: pd.Series({'market_phase':'unit_test_phase'})
ok,reason=pbt.buy('2026-07-02','09:45','09:46',row,bar2,status,daily2,{},0.30,'leader_weak_to_strong')
assert ok, f'Production buy path failed: {reason}'
assert pbt.pos['000001']['name']=='TEST' and pbt.fills[-1]['name']=='TEST'

# 7) Data-coverage overrides preserve facts but exclude non-executable names; point-in-time replacements are present.
pbt2=ProductionThemeReviewBT()
r20=pbt2.role_rows('2026-07-20'); assert int(r20.loc[r20.stock_code=='603580','tradable_next_day'].iloc[0])==0
r30=pbt2.role_rows('2026-07-30'); assert int(r30.loc[r30.stock_code=='003032','tradable_next_day'].iloc[0])==0
assert {'002558','002517'}.issubset(set(r30.loc[r30.tradable_next_day.astype(int)==1,'stock_code']))
r31=pbt2.role_rows('2026-07-31'); assert int(r31.loc[r31.stock_code=='003032','tradable_next_day'].iloc[0])==0
assert {'300996','300605','002558','002517','300071'}.issubset(set(r31.loc[r31.tradable_next_day.astype(int)==1,'stock_code']))

# 8) Causal learning policy: no threshold changes; only future half-size/pause state.
assert rules['learning']['single_trade_threshold_tuning_forbidden'] is True
assert rules['learning']['rule_change_min_closed_samples']>=8
assert rules['learning']['july_end_retuning_forbidden'] is True

# 9) Every non-cash playbook has explicit observation windows and next-minute entry windows.
for name,spec in rules['playbooks'].items():
    if name=='cash': continue
    assert len(spec['entry_windows'])==len(spec['observation_windows'])
    for obs,entry in zip(spec['observation_windows'],spec['entry_windows']):
        assert FILL[obs]==entry, f'{name}: {obs} must fill at next minute open {FILL[obs]}, got {entry}'

# 10) Core universe cannot be generated by price ranking: it must come from prior web context tables.
cand_src=inspect.getsource(ThemeReviewBT.candidate_table)
assert "self.role_rows(date)" in cand_src and "self.theme_rows(date)" in cand_src
assert 'nlargest' not in cand_src and '.rank(' not in cand_src, 'Core candidates must not be reconstructed from ex-post price ranking'

# 11) Mode selection is deterministic by predeclared theme priority/role/mode order, never future PnL.
entry_src=inspect.getsource(ThemeReviewBT.process_entries)
assert 'sorted(passed' in entry_src
assert 'pnl' not in entry_src.lower() and 'future' not in entry_src.lower()

print(json.dumps({
    'status':'PASS',
    'july_only_price_boundary':True,
    'old_backtest_forbidden':True,
    'next_minute_execution_verified':True,
    't_plus_one_verified':True,
    'exact_tick_price_limit_verified':True,
    'new_listing_block_verified':True,
    'production_buy_path_verified':True,
    'effective_role_overrides_verified':True,
    'prior_context_core_universe_verified':True,
    'causal_learning_contract_verified':True
},ensure_ascii=False,indent=2))
