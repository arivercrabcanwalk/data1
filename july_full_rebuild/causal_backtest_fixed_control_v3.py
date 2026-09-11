from pathlib import Path
import causal_backtest as cb
from causal_backtest_v3 import FormalEngine

cb.OUT=Path(__file__).resolve().parent/'results_fixed_control'
cb.OUT.mkdir(parents=True,exist_ok=True)

class FixedFormalControl(FormalEngine):
    def update_review(self,date,day_return):
        return 'FROZEN_CONTROL: same PIT and execution rules; no post-close mode/risk adaptation'

if __name__=='__main__':
    FixedFormalControl().run()
