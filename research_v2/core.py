"""Causal, long-only A-share technical-proxy research engine (not live trading).

Signals consume completed bars only. Execution is an adverse-price, interval
participation model, NOT order-book replay. Unadjusted prices are never silently
converted into total returns. No news, sector or current-name ST backfilling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_COMMIT = "4de6d9311a961e05661c6914d6972544a812caf2"
COLS = ["open", "high", "low", "close", "volume", "amount"]
ST_CUTOFF = "2026-07-30"
METHODS = ("EARLY", "GDZY", "PBQY", "FXXG", "SMALL", "DIV_PROXY")


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (Path, pd.Timestamp)): return str(x)
    raise TypeError(type(x).__name__)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=json_default, allow_nan=False), encoding="utf-8")


def norm_code(s: pd.Series) -> pd.Series:
    z = s.astype(str).str.replace(r"\.0$", "", regex=True)
    z = z.str.extract(r"(\d{1,6})", expand=False).str.zfill(6)
    if z.isna().any() or not z.str.fullmatch(r"\d{6}").all():
        raise ValueError("Invalid security code")
    return z


def minute_slot(s: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(s, errors="raise")
    m = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
    am = (m >= 571) & (m <= 690)
    pm = (m >= 781) & (m <= 900)
    if not np.all(am | pm) or np.any(dt.dt.second.to_numpy() != 0):
        raise ValueError("Expected end-labelled 09:31-11:30/13:01-15:00 bars")
    return np.where(am, m - 571, m - 781 + 120).astype(np.int16)


def clock(k: int) -> str:
    m = 571 + k if k < 120 else 781 + k - 120
    return f"{m // 60:02d}:{m % 60:02d}"


def read_minute(path: Path, codes: set[str] | None = None) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    # A file-level reader avoids a Hive date=... partition schema collision.
    tab = pq.ParquetFile(path).read(columns=["code", "datetime"] + COLS)
    if codes is not None:
        tab = tab.filter(pc.is_in(tab["code"], value_set=__import__("pyarrow").array(sorted(codes))))
    df = tab.to_pandas()
    df["code"] = norm_code(df["code"])
    df["slot"] = minute_slot(df["datetime"])
    return df.sort_values(["code", "slot"], kind="stable")


def bool_number(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"true": 1., "false": 0., "1": 1.,
                                           "0": 0., "1.0": 1., "0.0": 0.})


def audit_and_daily(root: Path, out: Path) -> tuple[pd.DataFrame, dict[str, Path], dict]:
    import pyarrow.parquet as pq
    fs = sorted(root.glob("2026-*/part-*/date=*/minute1.parquet"))
    if len(fs) != 126:
        raise ValueError(f"Pinned study expects 126 daily minute files; found {len(fs)}")
    daily, receipts, audits = [], [], []
    fmap: dict[str, Path] = {}
    status_counts = Counter()
    for n, p in enumerate(fs):
        day = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(p)).group(1)
        if day in fmap: raise ValueError("Duplicate daily input")
        fmap[day] = p
        df = read_minute(p)
        x = df[COLS].to_numpy(float)
        bad = (~np.isfinite(x)).any(axis=1) | (x[:, :4] <= 0).any(axis=1)
        bad |= (x[:, 1] < x[:, 2]) | (x[:, 0] < x[:, 2]) | (x[:, 0] > x[:, 1])
        bad |= (x[:, 3] < x[:, 2]) | (x[:, 3] > x[:, 1]) | (x[:, 4:] < 0).any(axis=1)
        dup = int(df.duplicated(["code", "slot"]).sum())
        wrong_date = int((df["datetime"].astype(str).str[:10] != day).sum())
        a = dict(date=day, rows=len(df), bad_rows=int(bad.sum()), duplicate_rows=dup,
                 wrong_date=wrong_date, no_flow=int(((x[:, 4] <= 1e-6) | (x[:, 5] <= 1e-6)).sum()))
        if bad.any() or dup or wrong_date:
            save_json(out / "failed_audit.json", a)
            raise ValueError(f"Raw audit failed: {a}")
        good = (x[:, 4] > 1e-6) & (x[:, 5] > 1e-6)
        ratio = x[good, 5] / (x[good, 4] * x[good, 3])
        a["amount_volume_price_median"] = float(np.median(ratio)) if len(ratio) else None
        d = df.groupby("code", sort=False).agg(open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            amount=("amount", "sum"), bars=("slot", "size"))
        d["date"] = day
        sp = p.with_name("daily_stock_status.parquet")
        if not sp.exists(): raise ValueError(f"Missing daily status: {sp}")
        st = pq.ParquetFile(sp).read().to_pandas()
        st["code"] = norm_code(st["code"])
        if st["code"].duplicated().any(): raise ValueError("Duplicate status code")
        for c in ["status_confidence", "status_source"]:
            status_counts.update({f"{c}:{k}": int(v) for k, v in st[c].value_counts(dropna=False).items()})
        st = st.set_index("code")
        trusted = st["status_source"].astype(str).str.contains("historical_stock_status:bak_daily_historical", regex=False)
        d["st_raw"] = bool_number(st["is_st"]).where(trusted).reindex(d.index)
        d["st_historical_evidence"] = trusted.reindex(d.index).fillna(False)
        d["new_initial"] = bool_number(st["is_new_listing_initial"]).reindex(d.index)
        d["status_present"] = d.index.isin(st.index)
        a["status_rows"] = len(st)
        a["historical_st_rows"] = int(trusted.sum())
        a["historical_st_minute_codes"] = int(d["st_historical_evidence"].sum())
        a["stock_days"] = len(d)
        a["incomplete_stock_days"] = int((d["bars"] != 240).sum())
        a["status_missing_for_minute_codes"] = int((~d["status_present"]).sum())
        audits.append(a)
        daily.append(d.reset_index())
        for f in (p, sp):
            receipts.append(dict(path=str(f.relative_to(root)), bytes=f.stat().st_size,
                                 sha256=hashlib.file_digest(f.open("rb"), "sha256").hexdigest()))
        if (n + 1) % 21 == 0 or n + 1 == len(fs): print("AUDIT", n + 1, len(fs), flush=True)
    daily_df = pd.concat(daily, ignore_index=True).sort_values(["code", "date"])
    # Trust historical evidence per ROW, not the confidence label or date alone.
    # Persist last known status; this does not verify subsequent status changes.
    daily_df["st_known"] = daily_df["st_raw"].where(daily_df["date"] <= ST_CUTOFF)
    daily_df["st_known"] = daily_df.groupby("code", sort=False)["st_known"].ffill()
    daily_df.to_csv(out / "daily_bars.csv.gz", index=False, compression="gzip")
    pd.DataFrame(receipts).to_csv(out / "data_sha256.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "daily_audit.csv", index=False)
    audit = dict(source_commit=SOURCE_COMMIT, run_commit=os.getenv("GITHUB_SHA", "local"),
        minute_files=len(fs), status_files=len(fs), minute_bytes=sum(p.stat().st_size for p in fs),
        raw_rows=sum(a["rows"] for a in audits), stock_days=len(daily_df),
        codes=int(daily_df.code.nunique()), start=min(fmap), end=max(fmap),
        bad_rows=sum(a["bad_rows"] for a in audits), duplicates=sum(a["duplicate_rows"] for a in audits),
        no_flow_rows=sum(a["no_flow"] for a in audits),
        incomplete_stock_days=sum(a["incomplete_stock_days"] for a in audits),
        status_counts=dict(status_counts),
        amount_volume_price_median_range=[min(a["amount_volume_price_median"] for a in audits),
                                         max(a["amount_volume_price_median"] for a in audits)],
        st_policy="use only per-row bak_daily_historical evidence through 2026-07-30; carry last known status forward; current-name approximations discarded",
        adjustment_policy="prices as provided; no corporate-action ledger; not total return",
        missing=["verified adjustment factors", "corporate actions", "point-in-time sectors/news",
                 "actual daily price-limit prices", "order book", "official benchmark"],
        timestamp_assumption="end-labelled local exchange minute, inferred from observed session grid")
    save_json(out / "audit.json", audit)
    return daily_df, fmap, audit


class Market:
    def __init__(self, daily: pd.DataFrame):
        self.dates = sorted(daily.date.unique())
        self.codes = sorted(daily.code.unique())
        self.di = {d: i for i, d in enumerate(self.dates)}
        self.ci = {c: i for i, c in enumerate(self.codes)}
        self.x = {}
        for col in COLS + ["bars", "st_known", "new_initial"]:
            self.x[col] = daily.pivot(index="date", columns="code", values=col).reindex(
                index=self.dates, columns=self.codes).to_numpy(float)
        close = self.x["close"]
        self.prev = np.vstack([np.full((1, len(self.codes)), np.nan), close[:-1]])
        self.ret = close / self.prev - 1
        self.gap = self.x["open"] / self.prev - 1
        self.bad_gap = np.abs(self.gap) > .20
        self.big = (self.ret >= .07) & (close > self.x["open"]) & ~self.bad_gap
        self.ma20 = pd.DataFrame(close).rolling(20, min_periods=20).mean().to_numpy()
        self.am20 = pd.DataFrame(self.x["amount"]).rolling(20, min_periods=20).mean().to_numpy()
        self.vm20prev = pd.DataFrame(self.x["volume"]).shift().rolling(20, min_periods=20).mean().to_numpy()
        self.h20prev = pd.DataFrame(self.x["high"]).shift().rolling(20, min_periods=20).max().to_numpy()
        self.r20 = pd.DataFrame(close).pct_change(20, fill_method=None).to_numpy()
        self.dif = (pd.DataFrame(close).ewm(span=12, adjust=False).mean() -
                    pd.DataFrame(close).ewm(span=26, adjust=False).mean()).to_numpy()
        finite_ma = np.isfinite(self.ma20) & np.isfinite(close)
        self.breadth = np.sum((close > self.ma20) & finite_ma, axis=1) / np.maximum(1, finite_ma.sum(axis=1))


@dataclass(frozen=True)
class Candidate:
    method: str
    code: str
    day: int
    anchor_day: int
    support: float
    prev_close: float
    score: float
    liquidity: float

    @property
    def key(self) -> tuple:
        return self.method, self.code, self.anchor_day


def make_candidates(m: Market) -> dict[int, list[Candidate]]:
    result = defaultdict(list)
    o, h, l, c, v = [m.x[k] for k in COLS[:5]]
    n = len(m.dates)
    for j, code in enumerate(m.codes):
        if not code.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")):
            continue
        b, z = -1, -1
        origin = mid = platform = zsupport = np.nan
        active = pb_valid = zactive = False
        pivots: list[int] = []
        for t in range(n - 1):
            if not np.isfinite(c[t, j]):
                active = pb_valid = zactive = False
                continue
            if m.bad_gap[t, j] or not np.isfinite(m.prev[t, j]):
                active = pb_valid = zactive = False
            p = t - 2
            if p >= 2 and np.all(np.isfinite(l[p-2:p+3, j])):
                if l[p, j] < np.min(l[p-2:p, j]) and l[p, j] <= np.min(l[p+1:p+3, j]):
                    pivots.append(p)
            if m.big[t, j]:
                b = t
                origin = min(o[t, j], m.prev[t, j])
                mid = (o[t, j] + c[t, j]) / 2
                active = pb_valid = True
                platform = np.inf
            elif b >= 0:
                active = active and c[t, j] >= origin * .997
                pb_valid = pb_valid and c[t, j] >= mid
                platform = min(platform, l[t, j])
            if z >= 0: zactive = zactive and c[t, j] >= zsupport * .997
            small = (0 < m.ret[t, j] < .07 and c[t, j] > o[t, j]
                     and c[t, j] > m.h20prev[t, j]
                     and v[t, j] > 1.2 * m.vm20prev[t, j] and not m.bad_gap[t, j])
            if small:
                z = t
                zsupport = min(o[t, j], m.prev[t, j]) if m.ret[t, j] <= .05 else (o[t, j] + c[t, j]) / 2
                zactive = True
            eligible = (m.x["bars"][t, j] == 240 and m.am20[t, j] >= 1e8
                        and m.x["st_known"][t, j] == 0 and m.x["new_initial"][t, j] != 1
                        and not m.bad_gap[t, j])
            if not eligible: continue
            def add(method: str, a: int, support: float) -> None:
                if np.isfinite(support) and support > 0:
                    score = float(m.r20[a, j]) if np.isfinite(m.r20[a, j]) else -1.
                    liq = float(m.am20[a, j]) if np.isfinite(m.am20[a, j]) else 0.
                    result[t + 1].append(Candidate(method, code, t + 1, a, float(support),
                                                   float(c[t, j]), score, liq))
            if m.big[t, j]:
                add("EARLY", t, origin)
                if t >= 59 and len(pivots) >= 2:
                    p1, p2 = pivots[-2:]
                    if (5 <= p2 - p1 <= 40 and 2 <= t - p2 <= 10
                        and l[p2, j] < l[p1, j]
                        and not m.bad_gap[p1:t+1, j].any()
                        and m.dif[p1, j] < m.dif[p2, j] < 0
                        and l[p2, j] <= .85 * np.nanmax(h[max(0, p2-20):p2, j])):
                        add("DIV_PROXY", t, l[p2, j])
            if t >= 1 and m.big[t-1, j] and v[t, j] > v[t-1, j] and abs(m.ret[t, j]) <= .05:
                add("FXXG", t-1, l[t, j])
            recent = b >= 0 and pd.Timestamp(m.dates[b]) >= pd.Timestamp(m.dates[t]) - pd.DateOffset(months=3)
            if recent and active and t > b and origin <= c[t, j] <= origin * 1.03:
                add("GDZY", b, origin)
            if recent and active and pb_valid and t - b >= 3 and np.isfinite(platform):
                add("PBQY", b, platform)
            if zactive and 0 <= t-z < 10:
                add("SMALL", z, zsupport)
    return result


def day_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for code, g in df.groupby("code", sort=False):
        x = np.full((240, 6), np.nan)
        x[g["slot"].to_numpy(int)] = g[COLS].to_numpy(float)
        out[code] = x
    return out


def five_bars(x: np.ndarray) -> np.ndarray:
    if x.shape != (240, 6): raise ValueError("Expected 240x6 daily grid")
    y = x.reshape(48, 5, 6)
    complete = np.isfinite(y).all(axis=(1, 2))
    out = np.column_stack((y[:, 0, 0], np.max(y[:, :, 1], axis=1),
        np.min(y[:, :, 2], axis=1), y[:, -1, 3], y[:, :, 4].sum(axis=1), y[:, :, 5].sum(axis=1)))
    out[~complete] = np.nan
    return out


def signal_slot(c: Candidate, x: np.ndarray) -> int | None:
    """Return execution minute; every decision accesses strictly earlier bars."""
    f = five_bars(x)
    stop = c.support * .997
    for end in range(1, 44):
        k = (end + 1) * 5
        if np.any(f[:end+1, 3] < stop): return None
        if c.method in ("EARLY", "DIV_PROXY"):
            if np.isfinite(f[end]).all() and f[end, 3] > f[end, 0]: return k
        elif end >= 2:
            a, b, d = f[end-2:end+1]
            if (np.isfinite([a, b, d]).all() and a[2] <= c.support * 1.005
                and b[3] >= c.support and d[3] >= c.support
                and d[2] >= b[2] and d[3] >= b[3]):
                return k
    return None


def fee(notional: float, side: str) -> float:
    commission = max(5., round(notional * .0003, 2))
    transfer = round(notional * .00001, 2)
    stamp = round(notional * .0005, 2) if side == "sell" else 0.
    return commission + transfer + stamp


def execution(row: np.ndarray, side: str, requested: int, slip: float, participation: float,
              min_buy: int = 100) -> tuple[float, int, str]:
    if not np.isfinite(row).all(): return 0., 0, "missing_bar"
    o, h, l, c, v, a = row
    if v <= 1e-6 or a <= 1e-6: return 0., 0, "no_flow"
    if h - l < .005: return 0., 0, "single_price"
    price = (math.ceil(o * (1 + slip) * 100 - 1e-9) if side == "buy"
             else math.floor(o * (1 - slip) * 100 + 1e-9)) / 100
    if price < l - 1e-8 or price > h + 1e-8: return price, 0, "slip_outside_bar"
    capacity = int(min(v * participation, a * participation / price) // 100) * 100
    q = min(int(requested // 100) * 100, capacity)
    if q < (min_buy if side == "buy" else 100): return price, 0, "capacity_or_lot"
    return price, q, "filled"


@dataclass(frozen=True)
class Config:
    name: str
    methods: tuple[str, ...]
    exit: str = "CLOSE"
    slip: float = .0015
    participation: float = .01
    regime: bool = False
    initial_cash: float = 1_000_000.
    max_positions: int = 3
    max_stock: float = .20
    max_exposure: float = .60
    risk_per_trade: float = .005
    max_open_risk: float = .015
    monitor: int = 100


def configurations() -> list[Config]:
    groups = [("MAIN", ("GDZY", "PBQY"))] + [(a, (a,)) for a in METHODS]
    out = [Config(f"{name}_{exit}", methods, exit) for name, methods in groups
           for exit in ("CLOSE", "INTRADAY", "PROTECT")]
    out += [Config("MAIN_CLOSE_SLIP5", ("GDZY", "PBQY"), slip=.0005),
            Config("MAIN_CLOSE_SLIP30", ("GDZY", "PBQY"), slip=.003)]
    out += [Config(f"REGIME_{e}", ("GDZY", "PBQY"), e, regime=True)
            for e in ("CLOSE", "INTRADAY", "PROTECT")]
    out += [Config("MAIN_CLOSE_CAP05", ("GDZY", "PBQY"), participation=.005),
            Config("MAIN_CLOSE_CAP2", ("GDZY", "PBQY"), participation=.02)]
    return out


class Portfolio:
    def __init__(self, config: Config):
        self.cfg = config
        self.cash = config.initial_cash
        self.positions: dict[str, dict] = {}
        self.closed: list[dict] = []
        self.fills: list[dict] = []
        self.opportunities: list[dict] = []
        self.daily: list[dict] = []
        self.used: set[tuple] = set()
        self.cooldown: dict[str, int] = {}
        self.rejects = Counter()
        self.peak = config.initial_cash
        self.minute_mdd = 0.
        self.total_fees = 0.
        self.gap_exposure = []
        self.stale_marks = 0

    def equity(self) -> float:
        return self.cash + sum(p["qty"] * p["mark"] for p in self.positions.values())

    def open_risk(self) -> float:
        return sum(p["qty"] * max(p["mark"] - p["stop"] + p["mark"] * (.00081 + self.cfg.slip), 0.) + 5.
                   for p in self.positions.values())

    def day(self, t: int, m: Market, candidates: list[Candidate], arrays: dict[str, np.ndarray]) -> None:
        cfg = self.cfg
        date = m.dates[t]
        pool = [c for c in candidates if c.method in cfg.methods and c.key not in self.used
                and c.code not in self.positions and t - self.cooldown.get(c.code, -999) > 3]
        pool.sort(key=lambda c: (-c.score, -c.liquidity, c.code, c.method))
        selected: list[Candidate] = []
        seen = set()
        for cand in pool:
            if cand.code not in seen:
                seen.add(cand.code)
                selected.append(cand)
        self.rejects["monitor_overflow"] += max(0, len(selected) - cfg.monitor)
        selected = selected[:cfg.monitor]
        scheduled = defaultdict(list)
        gate = not cfg.regime or m.breadth[t-1] >= .5
        for cand in selected:
            log = dict(date=date, method=cand.method, code=cand.code,
                       anchor=m.dates[cand.anchor_day], support=cand.support,
                       waiting_days=t-cand.anchor_day, status="no_confirmation")
            self.opportunities.append(log)
            if not gate:
                log["status"] = "regime_blocked"
                continue
            if cand.code not in arrays:
                log["status"] = "missing_day"
                continue
            k = signal_slot(cand, arrays[cand.code])
            if k is not None:
                log["signal_time"] = clock(k-1)
                scheduled[k].append((cand, log))
        fives = {code: five_bars(a) for code, a in arrays.items()}
        for code, p in self.positions.items():
            a = arrays.get(code)
            if a is not None and np.isfinite(a[0, 0]) and abs(a[0, 0] / p["mark"] - 1) > .20:
                self.gap_exposure.append(dict(date=date, code=code, prior=p["mark"], open=float(a[0, 0])))
        for k in range(240):
            available_cash = self.cash
            snapshot_equity = self.equity()
            snapshot_exposure = snapshot_equity - self.cash
            snapshot_risk = self.open_risk()
            snapshot_count = len(self.positions)
            held_start = set(self.positions)
            for code in held_start:
                p = self.positions[code]
                a = arrays.get(code)
                if a is not None and np.isfinite(a[k]).all():
                    p["mae"] = min(p["mae"], float(a[k, 2] / p["entry_price"] - 1))
                    p["mfe"] = max(p["mfe"], float(a[k, 1] / p["entry_price"] - 1))
                if p["pending"] is None or p["entry_day"] >= t or k > 235:
                    continue
                want = p["qty"] if p["pending"] == "stop" else p["tp_remaining"]
                row = a[k] if a is not None else np.full(6, np.nan)
                price, q, reason = execution(row, "sell", want, cfg.slip, cfg.participation)
                if not q:
                    self.rejects["sell:" + reason] += 1
                    continue
                cost = fee(price*q, "sell")
                proceeds = price*q-cost
                self.cash += proceeds
                self.total_fees += cost
                p["sold_proceeds"] += proceeds
                p["qty"] -= q
                self.fills.append(dict(date=date, time=clock(k), code=code, side="sell", price=price,
                                       qty=q, fee=cost, reason=p["pending"], method=p["method"]))
                if not p["qty"]:
                    p.update(exit_date=date, exit_time=clock(k), pnl=p["sold_proceeds"]-p["initial_cost"], status="closed")
                    self.closed.append(p.copy())
                    del self.positions[code]
                    self.cooldown[code] = t
                elif p["pending"] == "profit":
                    p["tp_remaining"] -= q
                    if p["tp_remaining"] <= 0:
                        p["protected"] = True
                        p["stop"] = max(p["stop"], p["entry_price"])
                        p["pending"] = None
            for cand, log in scheduled.get(k, []):
                def reject(reason: str) -> None:
                    log["status"] = reason
                    self.rejects["buy:" + reason] += 1
                if cand.code in held_start or cand.code in self.positions or snapshot_count >= cfg.max_positions:
                    reject("position_slots"); continue
                row = arrays[cand.code][k]
                if not np.isfinite(row).all(): reject("missing_bar"); continue
                if abs(row[0] / cand.prev_close - 1) > .20:
                    reject("opening_discontinuity"); continue
                px = math.ceil(row[0] * (1 + cfg.slip) * 100 - 1e-9) / 100
                if px > cand.prev_close * 1.03:
                    reject("chase_over_3pct"); continue
                stop = cand.support * .997
                risk_per_share = px-stop + px*(.00112 + cfg.slip)
                distance = risk_per_share / px
                if px <= stop or not .005 <= distance <= .12:
                    reject("risk_distance"); continue
                budget = min(snapshot_equity*cfg.risk_per_trade,
                             snapshot_equity*cfg.max_open_risk - snapshot_risk)
                value_cap = min(snapshot_equity*cfg.max_stock,
                                snapshot_equity*cfg.max_exposure - snapshot_exposure,
                                max(0., (available_cash-5.)/1.00031))
                request = min(500_000, math.floor(max(0., min(max(0., budget-10.)/risk_per_share, value_cap/px))/100)*100)
                price, q, reason = execution(row, "buy", request, cfg.slip, cfg.participation,
                                             200 if cand.code.startswith("688") else 100)
                if not q: reject(reason); continue
                cost = fee(price*q, "buy")
                spent = price*q + cost
                if spent > available_cash + 1e-6: reject("cash"); continue
                self.cash -= spent
                available_cash -= spent
                snapshot_exposure += price*q
                snapshot_risk += q*risk_per_share + 10.
                snapshot_count += 1
                self.total_fees += cost
                self.used.add(cand.key)
                self.positions[cand.code] = dict(code=cand.code, method=cand.method,
                    anchor=m.dates[cand.anchor_day], entry_day=t, entry_date=date, entry_time=clock(k),
                    entry_price=price, entry_qty=q, qty=q, initial_cost=spent, sold_proceeds=0.,
                    stop=stop, initial_stop=stop, initial_risk=price-stop, mark=float(row[3]),
                    pending=None, protected=False, tp_remaining=0,
                    mae=min(0., float(row[3]/price-1)), mfe=max(0., float(row[3]/price-1)))
                self.fills.append(dict(date=date, time=clock(k), code=cand.code, side="buy", price=price,
                                       qty=q, fee=cost, reason="signal", method=cand.method))
                log.update(status="filled", entry_time=clock(k), price=price, qty=q)
            for code, p in self.positions.items():
                a = arrays.get(code)
                if a is not None and np.isfinite(a[k, 3]): p["mark"] = float(a[k, 3])
                if k % 5 != 4 or code not in fives: continue
                bar = fives[code][k//5]
                if not np.isfinite(bar).all(): continue
                if cfg.exit != "CLOSE" and bar[3] < p["stop"]:
                    p["pending"] = "stop"
                elif (cfg.exit == "PROTECT" and p["pending"] is None and not p["protected"]
                      and bar[3] >= p["entry_price"] + 2*p["initial_risk"]):
                    half = (p["qty"]//200)*100
                    if half:
                        p["pending"] = "profit"
                        p["tp_remaining"] = half
            eq = self.equity()
            self.peak = max(self.peak, eq)
            self.minute_mdd = min(self.minute_mdd, eq/self.peak-1)
            if self.cash < -1e-6: raise AssertionError("Negative cash")
        for code, p in self.positions.items():
            j = m.ci[code]
            close = m.x["close"][t, j]
            if not np.isfinite(close):
                self.stale_marks += 1
                continue
            p["mark"] = float(close)
            if cfg.exit == "CLOSE" and close < p["stop"]: p["pending"] = "stop"
            if m.big[t, j] and p["pending"] != "stop":
                origin = min(m.x["open"][t, j], m.prev[t, j])
                mid = (m.x["open"][t, j] + close)/2
                new = origin if close-p["entry_price"] >= 2*p["initial_risk"] else mid
                p["stop"] = max(p["stop"], float(new*.997))
        eq = self.equity()
        contribution = sum(p["pnl"] for p in self.closed) + sum(
            p["sold_proceeds"]+p["qty"]*p["mark"]-p["initial_cost"] for p in self.positions.values())
        if abs(eq-cfg.initial_cash-contribution) > .01: raise AssertionError("Ledger does not reconcile")
        self.daily.append(dict(date=date, equity=eq, cash=self.cash, holdings=len(self.positions),
                               exposure=(eq-self.cash)/eq, open_risk=self.open_risk()/eq,
                               breadth_previous=float(m.breadth[t-1])))

    def finish(self, out: Path) -> dict:
        path = out / self.cfg.name
        path.mkdir(exist_ok=True)
        d = pd.DataFrame(self.daily)
        d.to_csv(path/"equity.csv", index=False)
        pd.DataFrame(self.fills).to_csv(path/"fills.csv", index=False)
        pd.DataFrame(self.opportunities).to_csv(path/"opportunities.csv.gz", index=False, compression="gzip")
        closed = [dict(p, net_return=p["pnl"]/p["initial_cost"]) for p in self.closed]
        opened = [dict(p, status="open_mark", pnl=p["sold_proceeds"]+p["qty"]*p["mark"]-p["initial_cost"])
                  for p in self.positions.values()]
        pd.DataFrame(closed + opened).to_csv(path/"trades.csv", index=False)
        save_json(path/"rejections.json", dict(self.rejects))
        save_json(path/"held_discontinuities.json", self.gap_exposure)
        eq = np.r_[self.cfg.initial_cash, d.equity.to_numpy()]
        mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
        prior = self.cfg.initial_cash
        months = {}
        for month, g in d.groupby(d.date.str[:7], sort=True):
            last = float(g.iloc[-1].equity)
            months[month] = last/prior - 1
            prior = last
        pnls = np.array([p["pnl"] for p in self.closed])
        profit, loss = float(pnls[pnls>0].sum()), float(-pnls[pnls<0].sum())
        contributions = Counter()
        for p in closed + opened: contributions[p["code"]] += p["pnl"]
        top = contributions.most_common(1)
        opp = pd.DataFrame(self.opportunities)
        strict = d[d.date <= ST_CUTOFF]
        summary = dict(name=self.cfg.name, start=d.iloc[0].date, end=d.iloc[-1].date,
            days=len(d), total_return=float(eq[-1]/eq[0]-1), terminal_equity=float(eq[-1]),
            terminal_cash=self.cash, terminal_market_value=float(eq[-1]-self.cash),
            daily_max_drawdown=mdd, minute_max_drawdown=self.minute_mdd,
            closed_trades=len(closed), open_positions=len(opened),
            closed_win_rate=float((pnls>0).mean()) if len(pnls) else None,
            closed_profit_factor=profit/loss if loss else None, fees=self.total_fees,
            average_daily_exposure=float(d.exposure.mean()), months=months,
            realized_closed_pnl=float(pnls.sum()), marked_open_pnl=float(sum(p["pnl"] for p in opened)),
            opportunities=len(opp), filled_entries=int(sum(f["side"]=="buy" for f in self.fills)),
            opportunity_status=dict(Counter(x["status"] for x in self.opportunities)),
            st_reliable_window_return=float(strict.iloc[-1].equity/eq[0]-1) if len(strict) else None,
            held_gap_events=len(self.gap_exposure), stale_mark_stock_days=self.stale_marks,
            largest_code_pnl=dict(code=top[0][0], pnl=top[0][1]) if top else None,
            terminal_return_minus_top_code_pnl=float((eq[-1]-eq[0]-(top[0][1] if top else 0))/eq[0]))
        save_json(path/"summary.json", summary)
        return summary
