from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "backtest", "results")
os.makedirs(OUT, exist_ok=True)

# -------------------------------
# Backtest assumptions
# -------------------------------
START_CASH = 1_000_000.0
MAX_POSITIONS = 3
COMMISSION = 0.0003
SELL_STAMP = 0.0005
SLIPPAGE = 0.0010
MAX_HOLD_DAYS = 12

# All signal calculations use information available at the CLOSE of signal day.
# All entries/exits are executed no earlier than the NEXT trading day's first
# tradable minute, so there is no same-day hindsight execution.


def load_daily_from_minute() -> tuple[pd.DataFrame, pd.DataFrame]:
    minute_files = sorted(glob.glob(os.path.join(ROOT, "2026-*", "part-*", "date=*", "minute1.parquet")))
    status_files = sorted(glob.glob(os.path.join(ROOT, "2026-*", "part-*", "date=*", "daily_stock_status.parquet")))
    if not minute_files:
        raise RuntimeError("No minute parquet files found")

    bars = []
    exec_rows = []
    for i, fp in enumerate(minute_files, 1):
        x = pd.read_parquet(fp, columns=["code", "datetime", "open", "high", "low", "close", "volume", "amount"])
        x["datetime"] = pd.to_datetime(x["datetime"])
        x = x.sort_values(["code", "datetime"])
        g = x.groupby("code", sort=False)
        d = g.agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"), amount=("amount", "sum"), first_open=("open", "first"),
            first_close=("close", "first"), last_close=("close", "last")
        ).reset_index()
        date = x["datetime"].dt.date.iloc[0]
        d["date"] = pd.Timestamp(date)
        bars.append(d[["code", "date", "open", "high", "low", "close", "volume", "amount"]])
        exec_rows.append(d[["code", "date", "first_open", "first_close", "last_close"]])
        if i % 20 == 0:
            print(f"aggregated {i}/{len(minute_files)} minute files")

    daily = pd.concat(bars, ignore_index=True)
    exec_px = pd.concat(exec_rows, ignore_index=True)

    statuses = []
    for fp in status_files:
        s = pd.read_parquet(fp)
        if "date" not in s.columns:
            date_str = fp.split("date=")[-1].split(os.sep)[0]
            s["date"] = pd.Timestamp(date_str)
        else:
            s["date"] = pd.to_datetime(s["date"])
        keep = [c for c in ["code", "date", "is_st", "is_star_st", "is_new_listing_initial", "is_suspended", "price_limit_up_pct", "price_limit_down_pct", "market_board", "security_status"] if c in s.columns]
        statuses.append(s[keep])
    status = pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame(columns=["code", "date"])

    daily["code"] = daily["code"].astype(str)
    exec_px["code"] = exec_px["code"].astype(str)
    if not status.empty:
        status["code"] = status["code"].astype(str)
    return daily.sort_values(["code", "date"]), exec_px.merge(status, on=["code", "date"], how="left")


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def add_indicators_one(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["ret"] = g["close"].pct_change()
    g["dif"] = ema(g["close"], 12) - ema(g["close"], 26)
    g["dea"] = ema(g["dif"], 9)
    g["hist"] = (g["dif"] - g["dea"]) * 2
    g["amt_ma10"] = g["amount"].rolling(10, min_periods=5).mean()
    g["vol_ma10"] = g["volume"].rolling(10, min_periods=5).mean()
    g["range"] = (g["high"] - g["low"]).replace(0, np.nan)
    g["close_pos"] = (g["close"] - g["low"]) / g["range"]
    g["body_ret"] = g["close"] / g["open"] - 1
    g["drawdown20"] = g["close"] / g["close"].rolling(20, min_periods=10).max() - 1
    return g


def prior_low_candidate(g: pd.DataFrame, i: int, min_gap=5, max_gap=45):
    # Historical-only search: at day i close, search earlier low j and require
    # an observable rebound after j and before i. No future bars relative to i.
    lo = max(0, i - max_gap)
    hi = i - min_gap
    if hi <= lo:
        return None
    hist = g.iloc[lo:hi]
    if len(hist) < 5:
        return None
    # choose lowest CLOSE in the historical window
    jloc = int(hist["close"].values.argmin())
    j = lo + jloc
    if j >= i - 2:
        return None
    mid = g.iloc[j + 1:i]
    if mid.empty:
        return None
    rebound = mid["high"].max() / g.iloc[j]["close"] - 1
    if rebound < 0.055:
        return None
    return j, rebound


def detect_signals(g: pd.DataFrame) -> pd.DataFrame:
    g = add_indicators_one(g)
    n = len(g)
    sig = np.zeros(n, dtype=bool)
    sig_type = np.array([""] * n, dtype=object)
    score = np.full(n, np.nan)
    anchor = np.full(n, np.nan)
    prior_low_idx = np.full(n, -1, dtype=int)

    # Cache strong reversal bars as future "故地重游" anchors.
    anchors: list[dict] = []

    for i in range(n):
        row = g.iloc[i]
        if i < 30 or not np.isfinite(row["dif"]):
            continue

        # First: right-side 故地重游, based only on a previously confirmed >=7% bar.
        for a in reversed(anchors):
            age = i - a["i"]
            if age < 3:
                continue
            if age > 60:
                break
            dist = row["close"] / a["start"] - 1
            small_stable = -0.01 <= row["body_ret"] <= 0.03 and row["close_pos"] >= 0.45
            if -0.005 <= dist <= 0.03 and small_stable:
                sig[i] = True
                sig_type[i] = "right_pullback"
                anchor[i] = a["start"]
                score[i] = 1.0 + min(2.0, row["amount"] / max(row["amt_ma10"], 1.0))
                break

        # Three-wave / five-wave structural divergence.
        cand = prior_low_candidate(g, i)
        if cand is None:
            # Still record today's big bar as an anchor after signal tests.
            pass
        else:
            j, rebound = cand
            p = g.iloc[j]
            price_new_low = row["close"] < p["close"] * 0.995
            dif_div = np.isfinite(p["dif"]) and row["dif"] > p["dif"] + 0.05 * max(abs(p["dif"]), 0.01)
            hist_div = np.isfinite(p["hist"]) and row["hist"] > p["hist"]
            bullish_reversal = (
                row["body_ret"] >= 0.03
                and row["close"] > g.iloc[i - 1]["close"]
                and row["close_pos"] >= 0.70
                and row["amount"] >= 0.8 * max(row["amt_ma10"], 1.0)
            )
            if price_new_low and dif_div and hist_div and bullish_reversal:
                sig[i] = True
                sig_type[i] = "wave3_div_reversal"
                anchor[i] = row["open"]
                prior_low_idx[i] = j
                divergence = (row["dif"] - p["dif"]) / max(abs(p["dif"]), 0.05)
                dd = max(0.0, -row["drawdown20"])
                amtq = min(3.0, row["amount"] / max(row["amt_ma10"], 1.0))
                score[i] = 2.0 * divergence + 2.0 * dd + rebound + 0.2 * amtq

                # Approximate five-wave exhaustion: there is an even earlier low k,
                # then a rebound into j, then another rebound before i, and current
                # price makes another low while DIF/hist are materially stronger.
                c2 = prior_low_candidate(g.iloc[:j + 1].copy(), j, min_gap=4, max_gap=45) if j >= 12 else None
                if c2 is not None:
                    k, rebound2 = c2
                    pk = g.iloc[k]
                    if p["close"] < pk["close"] and np.isfinite(pk["dif"]) and row["dif"] > p["dif"] > pk["dif"]:
                        sig_type[i] = "wave5_exhaust_reversal"
                        score[i] += 0.75 + rebound2

        # A >=7% confirmed big bullish bar becomes a future right-side anchor.
        if row["body_ret"] >= 0.07 and row["close_pos"] >= 0.70:
            anchors.append({"i": i, "start": row["open"]})
            # keep bounded
            anchors = [a for a in anchors if i - a["i"] <= 65]

    g["signal"] = sig
    g["signal_type"] = sig_type
    g["signal_score"] = score
    g["anchor"] = anchor
    g["prior_low_idx"] = prior_low_idx
    return g


def is_bad_status(r: pd.Series) -> bool:
    for c in ["is_st", "is_star_st", "is_new_listing_initial"]:
        if c in r and pd.notna(r[c]) and bool(r[c]):
            return True
    if "is_suspended" in r and pd.notna(r["is_suspended"]) and bool(r["is_suspended"]):
        return True
    return False


def pct_decimal(value: float) -> float:
    """Normalize status percentages stored as either 10.0 or 0.10."""
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def tradable_buy(r: pd.Series, prev_close: float) -> bool:
    if is_bad_status(r):
        return False
    px = float(r["first_open"])
    if not np.isfinite(px) or px <= 0:
        return False
    lim = r.get("price_limit_up_pct", np.nan)
    if pd.notna(lim) and prev_close > 0 and px >= prev_close * (1 + pct_decimal(lim) - 0.001):
        return False
    # Avoid chasing the article explicitly warns against: >3% intraday/open chase.
    if prev_close > 0 and px / prev_close - 1 > 0.03:
        return False
    return True


def tradable_sell(r: pd.Series, prev_close: float) -> bool:
    if "is_suspended" in r and pd.notna(r["is_suspended"]) and bool(r["is_suspended"]):
        return False
    px = float(r["first_open"])
    if not np.isfinite(px) or px <= 0:
        return False
    lim = r.get("price_limit_down_pct", np.nan)
    if pd.notna(lim) and prev_close > 0 and px <= prev_close * (1 - abs(pct_decimal(lim)) + 0.001):
        return False
    return True


@dataclass
class Position:
    code: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    anchor: float
    signal_type: str
    signal_date: pd.Timestamp
    hold_days: int = 0
    highest_close: float = 0.0


def run_backtest(daily: pd.DataFrame, exec_px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Keep the grouping key explicitly.  Recent pandas versions exclude the
    # grouping column from GroupBy.apply output, so relying on apply to carry
    # `code` through silently drops the identifier and breaks the portfolio
    # simulation.
    signal_parts = []
    for code, g in daily.groupby("code", sort=False):
        out = detect_signals(g)
        out["code"] = str(code)
        signal_parts.append(out)
    daily = pd.concat(signal_parts, ignore_index=True)

    daily = daily.sort_values(["date", "code"])
    exec_px = exec_px.sort_values(["date", "code"])
    dates = sorted(daily["date"].unique())
    by_day = {d: x.set_index("code") for d, x in daily.groupby("date")}
    ex_day = {d: x.set_index("code") for d, x in exec_px.groupby("date")}

    cash = START_CASH
    positions: dict[str, Position] = {}
    pending_entries: list[dict] = []
    trades = []
    curve = []

    prev_date = None
    for date in dates:
        day = by_day[date]
        ex = ex_day.get(date)
        if ex is None:
            continue

        # 1) Execute exits decided using previous close information.
        if prev_date is not None:
            prev = by_day[prev_date]
            for code in list(positions.keys()):
                if code not in day.index or code not in ex.index or code not in prev.index:
                    continue
                pos = positions[code]
                pr = prev.loc[code]
                er = ex.loc[code]
                exit_reason = None
                # Article-derived anchor breach: use close signal, execute next day.
                if float(pr["close"]) < pos.anchor * 0.995:
                    exit_reason = "anchor_break"
                # Short-term implementation extension: time stop and trailing loss.
                elif pos.hold_days >= MAX_HOLD_DAYS:
                    exit_reason = "time_stop"
                elif pos.hold_days >= 3 and float(pr["close"]) < pos.highest_close * 0.92:
                    exit_reason = "trail_8pct"
                if exit_reason and tradable_sell(er, float(pr["close"])):
                    raw = float(er["first_open"])
                    px = raw * (1 - SLIPPAGE)
                    proceeds = pos.shares * px * (1 - COMMISSION - SELL_STAMP)
                    cash += proceeds
                    pnl = proceeds - pos.shares * pos.entry_price * (1 + COMMISSION)
                    trades.append({
                        "code": code, "signal_type": pos.signal_type, "signal_date": pos.signal_date,
                        "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                        "exit_date": pd.Timestamp(date), "exit_price": px, "exit_reason": exit_reason,
                        "hold_days": pos.hold_days, "pnl": pnl,
                        "return_pct": px / pos.entry_price - 1,
                    })
                    del positions[code]

        # 2) Execute prior-day signals at today's first tradable minute.
        if prev_date is not None and pending_entries:
            pending_entries.sort(key=lambda z: z["score"], reverse=True)
            for pe in pending_entries:
                if len(positions) >= MAX_POSITIONS:
                    break
                code = pe["code"]
                if code in positions or code not in ex.index or code not in by_day[prev_date].index:
                    continue
                er = ex.loc[code]
                pr = by_day[prev_date].loc[code]
                if not tradable_buy(er, float(pr["close"])):
                    continue
                slots = MAX_POSITIONS - len(positions)
                alloc = min(cash / slots, START_CASH * 0.50)
                if alloc < 10_000:
                    continue
                raw = float(er["first_open"])
                px = raw * (1 + SLIPPAGE)
                shares = alloc / (px * (1 + COMMISSION))
                cost = shares * px * (1 + COMMISSION)
                cash -= cost
                positions[code] = Position(
                    code=code, shares=shares, entry_date=pd.Timestamp(date), entry_price=px,
                    anchor=float(pe["anchor"]), signal_type=pe["signal_type"],
                    signal_date=pd.Timestamp(prev_date), highest_close=float(pr["close"]),
                )

        # 3) Mark positions and increment hold days.
        equity = cash
        for code, pos in positions.items():
            if code in day.index:
                c = float(day.loc[code]["close"])
                equity += pos.shares * c
                pos.highest_close = max(pos.highest_close, c)
                pos.hold_days += 1
            else:
                equity += pos.shares * pos.entry_price
        curve.append({"date": pd.Timestamp(date), "equity": equity, "cash": cash, "positions": len(positions)})

        # 4) Generate signals at today's close for NEXT trading day only.
        cands = day[day["signal"] == True].copy()
        if len(cands):
            cands = cands[np.isfinite(cands["anchor"]) & np.isfinite(cands["signal_score"])]
            cands = cands.sort_values("signal_score", ascending=False)
            pending_entries = [
                {"code": str(code), "score": float(r["signal_score"]), "anchor": float(r["anchor"]), "signal_type": str(r["signal_type"])}
                for code, r in cands.iterrows()
            ]
        else:
            pending_entries = []
        prev_date = date

    curve = pd.DataFrame(curve)
    trades = pd.DataFrame(trades)
    if len(curve):
        curve["peak"] = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / curve["peak"] - 1
        total_return = curve["equity"].iloc[-1] / START_CASH - 1
        max_dd = curve["drawdown"].min()
    else:
        total_return = max_dd = np.nan

    stats = {
        "start_cash": START_CASH,
        "final_equity": float(curve["equity"].iloc[-1]) if len(curve) else None,
        "total_return": float(total_return),
        "multiple": float(1 + total_return),
        "max_drawdown": float(max_dd),
        "trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else None,
        "avg_trade_return": float(trades["return_pct"].mean()) if len(trades) else None,
        "median_trade_return": float(trades["return_pct"].median()) if len(trades) else None,
    }
    if len(trades):
        stats["by_signal"] = trades.groupby("signal_type").agg(
            trades=("pnl", "size"), win_rate=("pnl", lambda x: float((x > 0).mean())),
            avg_return=("return_pct", "mean"), total_pnl=("pnl", "sum")
        ).reset_index().to_dict("records")
        stats["by_exit"] = trades.groupby("exit_reason").agg(
            trades=("pnl", "size"), avg_return=("return_pct", "mean"), total_pnl=("pnl", "sum")
        ).reset_index().to_dict("records")
    return curve, trades, stats


def main():
    daily, exec_px = load_daily_from_minute()
    print("daily rows", len(daily), "stocks", daily["code"].nunique(), "dates", daily["date"].nunique())
    curve, trades, stats = run_backtest(daily, exec_px)
    curve.to_csv(os.path.join(OUT, "equity_curve.csv"), index=False)
    trades.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
