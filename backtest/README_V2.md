# Three/Five-Wave Backtest V2

This revision fixes the structural problems exposed by the first 2026-03 to 2026-08 backtest, where all 60 closed trades came from `right_pullback` and the true wave-3 / wave-5 modules produced zero trades.

## What changed

1. **Wave divergence is now a multi-day causal state machine.**
   - Day T can arm a wave-3 or wave-5 divergence after a new low and MACD momentum improvement.
   - The strategy does not buy on Day T.
   - It waits up to three later trading days for a reversal bar.
   - Only after that close confirmation can the next trading day become an entry day.

2. **Wave-5 exhaustion is independent from wave-3 execution.**
   - It looks for a three-low price staircase and weaker five-day negative MACD histogram area at the final low.

3. **The >=7% big-bar definition now uses close vs previous close, not close vs open.**
   - This fixes the prior misclassification of gap-up and gap-down candles.

4. **Gap anchors are corrected.**
   - For a gap big bar, the previous close is used as the right-side support anchor.
   - Otherwise the big-bar open is used.

5. **Early anchors are no longer blocked by MACD warm-up.**
   - Right-side anchors can be created from the beginning of the sample.

6. **Right-pullback entries require a healthier volume structure.**
   - Big-bar anchor first.
   - Pullback volume must shrink versus the anchor bar and not expand materially versus the 10-day average.
   - A pullback is only a setup, not an immediate order.

7. **`pingbu_qingyun` is now a separate signal.**
   - At least three trading days after the big bar.
   - Closing prices must keep the big-bar midpoint.
   - The platform low becomes the stop reference.

8. **Entries use real one-minute confirmation instead of buying at the next open.**
   - No entry before 09:45.
   - No open gap above 3%.
   - No chase above 3% vs previous close.
   - The support anchor must not be broken intraday.
   - Price must rebound at least 1% from the running low.
   - At least 10 bars must pass after the running low.
   - Price must recover near/above running VWAP.
   - Recent up-bar amount must be strong enough versus down-bar amount.

9. **Absolute gating replaces forced top-3 buying.**
   - A signal must pass an absolute score threshold.
   - Liquidity must pass a cross-sectional floor.
   - Right-side signals must pass a 20-day relative-strength floor.
   - If no candidate passes, the strategy stays in cash.

10. **Market regime controls maximum portfolio exposure.**
    - The regime uses breadth, percentage above 5-day average, median stock return and total market amount.
    - Exposure is limited to 0%, 30%, 50% or 80% depending on the prior close regime.

11. **Broken setups cannot be recycled immediately.**
    - Big-bar anchors are invalidated after an effective close break.
    - An `anchor_break` trade exit creates a five-trading-day code cooldown.
    - Setup generation itself has a duplicate-signal cooldown.

12. **Deterministic ranking is used.**
    - Candidate score descending, signal priority, then code ascending.
    - This removes the prior boundary-selection instability.

13. **Diagnostics are much richer.**
    Results now include:
    - `signals.csv`
    - `signal_diagnostics.csv`
    - `entry_rejections.csv`
    - `market_regime.csv`
    - `equity_curve.csv`
    - `monthly_equity.csv`
    - `trades.csv`
    - `summary.json`

    Trades include candidate score, entry timestamp, MFE and MAE. Summary also separates <=3-day and >=4-day outcomes so the prior false-entry problem is visible immediately.

## Industry / mainline filter

The repository currently contains no industry, concept or supply-chain mapping. V2 therefore does **not** invent one.

If `backtest/sector_map.csv` is added with:

```csv
code,sector
000001,Banking
300308,Optical Modules
```

V2 automatically computes 5-day sector momentum ranks, requires a minimum sector rank, and adds sector strength to candidate scoring. Without this file, the backtest explicitly reports `sector_filter_enabled: false` and uses only market regime, stock relative strength, liquidity and the technical setup.

This is deliberate: the source methodology says stock selection should come from industry-chain logic, so the program should never silently replace missing sector data with fabricated labels.

## Anti-lookahead rules

- Divergence only uses the current close and earlier bars.
- A divergence is armed at the close; later reversal confirmation is required.
- A close-confirmed setup can only enter on the next trading day.
- Intraday entry confirmation only uses minute bars up to the candidate entry timestamp.
- Close-based stops are executed no earlier than the next trading day.

## Tests

Run:

```bash
python -m unittest backtest/test_three_five_wave_backtest.py -v
```

The workflow runs these tests before the full backtest.
