import json, re
from pathlib import Path
import pandas as pd

SYS=Path('july_theme_system')
themes=pd.read_csv(SYS/'daily_theme_priority.csv',dtype=str)
groups=pd.read_csv(SYS/'theme_group_map.csv',dtype=str)
chains=pd.read_csv(SYS/'industry_chain_reference.csv',dtype=str)

for df,cols,name in [
    (groups,['theme','canonical_group'],'theme_group_map'),
    (chains,['canonical_group','chain_structure','what_to_watch','catalyst_types','negative_signals','structural_source_url','usage_boundary'],'industry_chain_reference')]:
    for c in cols:
        assert c in df.columns, f'{name} missing {c}'
        assert df[c].fillna('').str.strip().ne('').all(), f'{name} blank {c}'

missing_theme=sorted(set(themes.theme)-set(groups.theme))
assert not missing_theme, f'July themes missing canonical chain mapping: {missing_theme}'
missing_group=sorted(set(groups.canonical_group)-set(chains.canonical_group))
assert not missing_group, f'Canonical groups missing chain reference: {missing_group}'
assert chains.structural_source_url.str.startswith('https://').all()

# Structural reference must never carry six-digit stock codes or historical stock membership.
# Historical members live only in point-in-time daily_core_roles/additional_core_roles.
text='\n'.join(chains.fillna('').astype(str).agg(' | '.join,axis=1).tolist())
assert not re.search(r'(?<!\d)\d{6}(?!\d)', text), 'Static six-digit stock code found in industry-chain reference'
for forbidden_col in ['stock_code','stock_name','member_code','constituent_code']:
    assert forbidden_col not in chains.columns, f'Static stock-membership column forbidden: {forbidden_col}'

print(json.dumps({
  'status':'PASS',
  'unique_july_theme_labels':int(themes.theme.nunique()),
  'mapped_theme_labels':int(groups.theme.nunique()),
  'canonical_industry_chains':int(chains.canonical_group.nunique()),
  'all_july_themes_mapped':True,
  'no_static_stock_codes_or_membership_columns_in_chain_reference':True
},ensure_ascii=False,indent=2))
