import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"


def test_manifest_ready_contract():
    m = json.loads((OUT / "knowledge_manifest.json").read_text(encoding="utf-8"))
    assert m["coverage"]["trading_days"] == 23
    assert m["anti_leakage"]["web_data_effective"] == "next trading day only"
    assert m["anti_leakage"]["forward_sections_forbidden"] is True


def test_expectations_never_same_day():
    x = pd.read_csv(OUT / "expectation_book.csv")
    assert len(x) == 22
    assert (pd.to_datetime(x.effective_date) > pd.to_datetime(x.asof_date)).all()
    assert not x.uses_future_data.astype(bool).any()


def test_no_backtest_outputs_here():
    forbidden = {"trades.csv", "fills.csv", "equity_curve.csv", "summary.json"}
    assert forbidden.isdisjoint({p.name for p in OUT.iterdir()})


def test_source_conflicts_are_preserved():
    x = pd.read_csv(OUT / "source_audit.csv")
    assert len(x) == 23
    assert "web_count_disagreement" in x.columns
    assert "action" in x.columns


def test_market_source_is_github():
    x = pd.read_csv(OUT / "market_daily.csv")
    assert len(x) == 23
    assert x.date.nunique() == 23
