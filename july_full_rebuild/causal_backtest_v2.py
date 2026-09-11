from __future__ import annotations
import numpy as np
import pandas as pd
from causal_backtest import Engine, norm_code

class FullEngine(Engine):
    def load_day(self,path):
        date,df,dr,sn=super().load_day(path)
        # Correct conservative suspension handling: explicit suspension excludes; a null flag does not,
        # because an actual minute bar at the decision/execution minute is direct evidence that the stock traded.
        eligible=~(dr.is_st.fillna(False).astype(bool)|dr.is_star_st.fillna(False).astype(bool)|dr.is_new_listing_initial.fillna(False).astype(bool))
        if 'is_suspended' in dr.columns:
            eligible &= ~dr.is_suspended.fillna(False).astype(bool)
        dr['eligible']=eligible
        self._current_df=df
        self._current_dr=dr
        return date,df,dr,sn

    def theme_breadth(self,date,theme,snap):
        # Full prior-day factual theme members only. At today's decision minute, measure both level and
        # improvement since 09:35. This is a causal co-movement/leadership confirmation, not a future return.
        mem=self.ladder[(self.ladder.date==date)&(self.ladder.theme==theme)].code.astype(str)
        mem=set(x for x in mem if len(x)==6)
        if len(mem)<3 or snap.empty:return np.nan
        cur=snap[snap.code.isin(mem)].copy()
        if len(cur)<3:return np.nan
        cur=cur[['code','ret_now']].dropna().drop_duplicates('code')
        if len(cur)<3:return np.nan
        base=self._current_df[(self._current_df.code.isin(mem))&(self._current_df.minute=='09:35')][['code','close']].copy()
        if base.empty:return np.nan
        base['prev_close']=base.code.map(self.prev_close)
        base['ret35']=np.where(base.prev_close.notna()&(base.prev_close>0),base.close/base.prev_close-1,np.nan)
        z=cur.merge(base[['code','ret35']],on='code',how='inner').dropna()
        if len(z)<3:return np.nan
        positive=float((z.ret_now>0).mean())
        follow=float(((z.ret_now-z.ret35)>0).mean())
        # 60% current breadth + 40% breadth improvement. A score above 0.5 means the prior-day theme
        # is not merely represented by one isolated leader at the decision point.
        return 0.60*positive+0.40*follow

    def choose_mode(self,phase,role,theme_phase,mb35,mb_now,tb,ret_now,gap,range_pos):
        # No theme-member confirmation -> no trade. This blocks the previous leader-only false positives.
        if not np.isfinite(tb): return None
        return super().choose_mode(phase,role,theme_phase,mb35,mb_now,tb,ret_now,gap,range_pos)

if __name__=='__main__':
    FullEngine().run()
