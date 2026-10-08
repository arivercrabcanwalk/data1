# -*- coding: utf-8 -*-
"""Deterministic read-only V22 cleanliness gate.

A PASS on the causal-code subtests is not a PASS on unbiased predictive performance.
No strategy edits, no retraining, no data mining, no restart of disabled scheduled tasks.
"""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v22_clean_audit'
def get(name):return json.loads((OUT/name).read_text())
pit=get('pit_coverage_report.json')
prefix=get('strict_prefix_and_uncertainty.json')
firewall=get('feature_firewall_report.json')
search=get('search_multiplicity_inventory.json')
quality=get('minute_data_quality.json')
replay=json.loads((ROOT/'v22_results/strict_causal_replay_summary.json').read_text())
shadow=json.loads((ROOT/'v22_shadow_forward_v3/shadow_historical_diagnostics.json').read_text())
models=json.loads((ROOT/'v22_results/prequential_models.json').read_text())
forward=json.loads((ROOT/'v22_strict_forward/strict_forward_status.json').read_text())
v21=json.loads((ROOT/'v21_results/v21_summary.json').read_text())
v22_h=json.loads((ROOT/'v22_results/v22_consolidated_summary.json').read_text())
assert len(replay['entry_mismatches'])==1 and not replay['missing'] and not replay['new']
assert pit['not_found']==0 and pit['exante_feature_mismatches']==0
assert prefix['pit_entry_pass']==52 and prefix['early_exit_n']==prefix['pit_early_exit_pass']
assert firewall['entry_same_with_no_future_columns']==52
assert search['prior_2026_optimize_exits_lane_specific_tried_or_evaluated']>=500
assert search['universal_exit_parameter_combinations_before_na_filter']>=350
assert pit['minute_coverage']['missing']>0
assert all(not item['train_accepted'] for item in models['results'])
assert forward['strict_t1_closed']==0
assert abs(v21['final_52']['all_2026']['mean']-.07088381843990328)<1e-10
report={
 'asof':'2026-10-08',
 'scope':'V22 strict 7 first-buy lanes and shadow 7 discovery lanes; no R2 first-buy',
 'automated_monitoring':'PAUSED_MANUALLY_CONFIRMED_THROUGH_CHAT_AUTOMATION',
 'verdict':'NOT_CERTIFIED_PREDICTIVE_QUALITY; HIGH_RESEARCH_SELECTION_BIAS',
 'execution_causality_subcheck':'PASS_52_OF_52',
 'point_in_time_day1_subcheck':f"PASS_{pit['tests']}_OF_{pit['tests']}",
 'future_field_firewall_subcheck':'PASS_52_OF_52',
 'prior_exit_candidate_families_at_least':search['prior_2026_optimize_exits_lane_specific_tried_or_evaluated']+
          search['universal_exit_parameter_combinations_before_na_filter'],
 'genuine_out_of_sample_finished_trades':0,
 'minimum_remaining_known_gaps':[
    'No independently untouched forward sample after the final October 8 freeze.',
    'July-September repeatedly entered rule-selection pass/fail gates; it is pseudo-OOS only.',
    'Causal entry signal code shares historical event tables containing future-label fields; current entry does not access them, but schema is not isolated.',
    'Archived research panels drop events without observable T1 or minute entry data; missingness is systematic, and future T1 resets are excluded post-hoc.',
    'Minute OPEN and closing proxies plus fixed 0.52% costs are not real executable orderbook fills.',
    'Day2 entry snapshots may be produced after Day2 close; preregistered Day1 alone does not prove intraday signal delivery or fill.',
    'Shadow seven lanes came from hindsight opportunities and lack independent validation.',
    'Canonical topic label snapshots are not proven point-in-time (unused by the strictly audited entry fields).'
 ],
 'hold_main_version':'V21 historical archive / V22 strict version research-only, no auto forward process enabled',
 'historical_baseline':{
    'n':52,'wins':44,'win_rate':v21['final_52']['all_2026']['win_rate'],
    'mean':v21['final_52']['all_2026']['mean'],
    'strict_causal_replay_mean':replay['historical_replay_metrics']['avg'],
    'win_wilson95_descriptive_only':prefix['wilson95']
 },
 'data_gaps':{'missing_D2_open_minute_events':pit['minute_coverage']['missing'],
   'eligible_D2_trade_event_pool':pit['minute_coverage']['with_t1_and_no_price_reset'],
   'candidate_day1_raw_minute_not_valid':quality['n_events']-quality['d1_gate_counts'].get('FULL_240_VALID',0)},
 'evidence_files':[
  'pit_coverage_report.json','day1_asof_checks.csv','strict_prefix_and_uncertainty.json',
  'strict_52_prefix_checks.csv','strict_52_bar_feasibility.csv','feature_firewall_report.json',
  'search_multiplicity_inventory.json','minute_data_quality.json']
}
(OUT/'cleanliness_gate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('AUDIT_TESTS_PASSED_CAUSAL_ENGINE; CERTIFICATION_NOT_GRANTED')
print(json.dumps(report,ensure_ascii=False,indent=2))