import json
from pathlib import Path
import pandas as pd

SYS=Path('july_theme_system')
ctx=pd.read_csv(SYS/'july_2026_theme_timeline.csv',dtype=str)
rules=json.loads((SYS/'playbook_rules_v1.json').read_text(encoding='utf-8'))
observed=set(ctx.market_phase)
mapped=set(rules['market_risk'])
missing=sorted(observed-mapped)
assert not missing, f'Observed market phases without explicit risk mapping: {missing}'
for phase in observed:
    spec=rules['market_risk'][phase]
    assert 0 <= float(spec['max_equity_exposure']) <= 1
    assert 'cash' in spec['allowed']
print(json.dumps({'status':'PASS','observed_market_phases':len(observed),'explicitly_mapped':len(observed),'unmapped':[]},ensure_ascii=False))
