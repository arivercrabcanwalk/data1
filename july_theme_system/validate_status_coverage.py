from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_theme_review_production import ProductionThemeReviewBT
from run_theme_review_backtest import norm_code

ROOT=Path('2026-07')
bt=ProductionThemeReviewBT()
paths=bt.paths()
issues=[]; unknown_suspension_with_volume=0; checked=0
for path in paths[1:]:
    date=path.parent.name.split('=',1)[1]
    roles=bt.role_rows(date)
    roles=roles[roles.tradable_next_day.astype(int)==1]
    if roles.empty: continue
    st=pd.read_parquet(path.parent/'daily_stock_status.parquet')
    st['code']=norm_code(st.code)
    st=st.drop_duplicates('code').set_index('code')
    vol=pd.read_parquet(path,columns=['code','volume'])
    vol['code']=norm_code(vol.code)
    vol['volume']=pd.to_numeric(vol.volume,errors='coerce').fillna(0)
    posvol=vol.groupby('code').volume.sum()
    for r in roles.drop_duplicates('stock_code').itertuples(index=False):
        checked+=1; code=r.stock_code
        if code not in st.index:
            issues.append((date,code,'missing_status_row')); continue
        s=st.loc[code]
        for flag in ['is_st','is_star_st','is_new_listing_initial']:
            if flag not in s.index or pd.isna(s[flag]):
                issues.append((date,code,f'unknown_{flag}'))
            elif bool(s[flag]):
                issues.append((date,code,f'restricted_{flag}'))
        if 'is_suspended' in s.index:
            if pd.isna(s['is_suspended']):
                if float(posvol.get(code,0))>0:
                    unknown_suspension_with_volume+=1
                else:
                    issues.append((date,code,'unknown_suspension_without_positive_volume_evidence'))
            elif bool(s['is_suspended']):
                issues.append((date,code,'explicitly_suspended'))
        elif float(posvol.get(code,0))<=0:
            issues.append((date,code,'no_suspension_field_and_no_positive_volume_evidence'))
        for fld in ['price_limit_up_pct','price_limit_down_pct']:
            if fld not in s.index or pd.isna(s[fld]):
                issues.append((date,code,f'missing_{fld}'))
            else:
                try:
                    if not np.isfinite(float(s[fld])): issues.append((date,code,f'nonfinite_{fld}'))
                except Exception:
                    issues.append((date,code,f'invalid_{fld}'))
        if float(posvol.get(code,0))<=0:
            issues.append((date,code,'no_positive_month_day_volume_for_executable_core'))
assert not issues, f'Executable core local-status issues: {issues[:30]}'
print(json.dumps({
  'status':'PASS',
  'executable_core_day_rows_checked':checked,
  'unknown_is_suspended_but_positive_volume_evidence':unknown_suspension_with_volume,
  'all_executable_cores_have_status_and_positive_volume':True,
  'all_executable_cores_non_st_non_new_listing_non_suspended_by_available_evidence':True,
  'price_limit_rules_present':True
},ensure_ascii=False,indent=2))
