from pathlib import Path
import causal_backtest as cb
from causal_backtest_v2 import FullEngine

cb.OUT = Path(__file__).resolve().parent / 'results_fixed_control'
cb.OUT.mkdir(parents=True, exist_ok=True)

class FixedControl(FullEngine):
    def update_review(self,date,day_return):
        # Control experiment: same PIT, same roles, same entries/exits, but no post-close
        # mode weighting, pausing, portfolio defensive-state transition or recovery learning.
        # Market/theme phase still advances because that is new public information, not parameter learning.
        return 'FROZEN_CONTROL: no post-close strategy adaptation'

if __name__=='__main__':
    FixedControl().run()
