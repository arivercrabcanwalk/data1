from __future__ import annotations
import pandas as pd
from causal_backtest_v2 import FullEngine

class FormalEngine(FullEngine):
    def __init__(self):
        super().__init__()
        if 'theme_score' not in self.roles.columns:
            key=self.theme[['date','theme','theme_strength_score']].drop_duplicates(['date','theme']).rename(columns={'theme_strength_score':'theme_score'})
            self.roles=self.roles.merge(key,on=['date','theme'],how='left')
            self.roles['theme_score']=pd.to_numeric(self.roles['theme_score'],errors='coerce').fillna(0.0)
            self.roles.loc[self.roles.role.eq('market_dragon'),'theme_score']=100.0
        self.roles['recognition_score']=pd.to_numeric(self.roles['recognition_score'],errors='coerce').fillna(0.0)

if __name__=='__main__':
    FormalEngine().run()
