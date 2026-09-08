# Three/Five-Wave V3: causal execution, risk sizing, and disciplined research

V3 is a research rewrite of the V2 strategy. It is designed to correct the implementation and evaluation issues found in the V2 audit before any attempt to optimize headline returns.

## What V3 fixes

1. **Minute causality**: a completed 1-minute bar may confirm a signal, but the earliest execution price is the **next minute's open**. The confirmation bar's high/low is excluded from post-entry MFE/MAE.
2. **True market exposure control**: the previous close's market regime is applied at the next open to both new entries and existing positions. If the cap falls from 80% to 0%, sellable existing positions are reduced/closed; unsellable limit-down/suspended positions are reported as exposure breaches.
3. **Risk-normalized position sizing**: each new position is sized from a fixed 1% account risk budget divided by the distance from entry to the technical stop. Wider stops mechanically receive smaller size. A per-position notional cap remains in force.
4. **Realized vs unrealized P&L**: `open_positions.csv` separately reports terminal marks and unrealized P&L. `summary.json` reports realized and unrealized components independently.
5. **Intraday portfolio drawdown**: the final production fill stream is replayed through 1-minute prices to produce `intraday_equity_curve.csv` and an intraday maximum drawdown, in addition to end-of-day drawdown.
6. **Benchmarking**: the report includes a causal point-in-time equal-weight A-share universe benchmark derived from the repository, plus excess return, beta, alpha, Sharpe, Sortino and Calmar.
7. **Warm-up honesty**: January-February data are not present in the repository. V3 does not fabricate prehistory. Wave signals are enabled only after genuine in-repository history exists, and the warm-up limitation is written into the final report.
8. **Point-in-time sector discipline**: historical sector filtering is enabled only by `backtest/sector_map_history.csv` with `date,code,sector`. A static present-day map is ignored by default because applying it backward can introduce look-ahead/classification bias.
9. **Independent method tests**: wave-3, wave-5, `平步青云`, and `故地重游` are each evaluated separately before production combination.
10. **Anti-overfit research protocol**: only three predeclared confirmation profiles are compared. No broad parameter grid is searched. Profile selection uses March-June development results; July is a validation gate. August is not used by programmatic selection. Because August has been inspected in earlier human iterations, V3 explicitly labels it as an *algorithmic holdout, not pristine untouched out-of-sample data*.
11. **Monthly walk-forward audit**: May-August each select a profile using prior data only, then report the next month's outcome.

## Methodology retained from the source material

- `>=7%` landmark bullish bars remain the right-side anchor standard.
- Three-wave divergence and five-wave exhaustion remain multi-day structures; divergence arms first, reversal confirmation follows later.
- `故地重游` uses the landmark bar start point (previous close when there is a qualifying gap).
- `平步青云` uses the landmark bar body midpoint/platform support logic.
- No signal is forced: absolute gates can leave the portfolio in cash.

## Files

- `backtest/v3_backtest.py` - V3 research + production orchestrator
- `backtest/v3_shared.py` - signal, profile, minute-confirmation and point-in-time sector logic
- `backtest/v3_engine.py` - risk-normalized portfolio simulation and intraday equity replay
- `backtest/v3_research.py` - development/validation selection and walk-forward audit
- `backtest/v3_config.json` - fixed risk and evaluation protocol
- `backtest/v3_profiles.json` - three predeclared confirmation-strength profiles
- `backtest/test_v3_backtest.py` - causality/risk regression tests
- `backtest/sector_map_history.example.csv` - point-in-time sector mapping format
- `backtest/results_v3/` - generated result set (GitHub Actions artifact)

## Research split

- Development: 2026-03-02 through 2026-06-30
- Validation: 2026-07-01 through 2026-07-31
- Algorithmic holdout report: 2026-08-01 through 2026-08-31

The holdout is not called pristine OOS because earlier human research has already viewed August results. A truly untouched test requires later data that was never used in the research process.

## Point-in-time sector map

Optional file:

```csv
date,code,sector
2026-03-02,000001,银行
2026-03-02,300308,光模块
```

The mapping should reflect what was known on that historical date. Do not populate it using a current constituent list and silently backfill history.

## Reproduce

```bash
python -m unittest backtest/test_three_five_wave_backtest.py -v
python -m unittest backtest/test_v3_backtest.py -v
python backtest/v3_backtest.py
```
