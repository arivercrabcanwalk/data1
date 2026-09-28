#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from first_buy_v12_research import load_daily, build_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtest" / "results_first_buy_v16"
OUT.mkdir(parents=True, exist_ok=True)

OPEN_FEATURES = [
    "prior_streak", "market_height_prev", "height_gap", "height_ratio",
    "is_two_board_recycle", "is_three_board", "is_four_plus",
    "is_recycle", "generation_clip", "days_since_parent_event", "rearm_delay_days",
    "event_body_ret", "event_red",
    "ret5", "ret10", "ret20", "lu10", "lu20", "max_streak20",
    "event_ret", "event_gap", "range", "vol_prev_ratio", "vol_med10",
    "amount_ratio10", "log_amount", "close_vwap", "close_loc",
    "touched_upper", "d1_gap",
]

CONFIRM_EXTRA = [
    "m30_ret_open", "m30_ret_event", "m30_range", "m30_close_loc",
    "m30_close_vwap", "m30_reclaim_event", "m30_low_vs_event",
    "m30_high_vs_event", "m30_amount_frac",
]
CONFIRM_FEATURES = OPEN_FEATURES + CONFIRM_EXTRA


def add_history_features(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.sort_values(["code", "date"]).copy()
    g = x.groupby("code", group_keys=False)
    for n in [5, 10, 20]:
        x[f"ret{n}"] = x["close"] / g["close"].shift(n) - 1.0
        x[f"lu{n}"] = (
            g["lu_calc"].rolling(n, min_periods=max(3, n // 2)).sum()
            .reset_index(level=0, drop=True)
            .groupby(x["code"]).shift(1)
        )
    x["max_streak20"] = (
        g["streak_calc"].rolling(20, min_periods=5).max()
        .reset_index(level=0, drop=True)
        .groupby(x["code"]).shift(1)
    )
    amount_med10 = (
        g["amount"].rolling(10, min_periods=5).median()
        .reset_index(level=0, drop=True)
        .groupby(x["code"]).shift(1)
    )
    x["amount_ratio10"] = x["amount"] / amount_med10.replace(0, np.nan)
    x["log_amount"] = np.log1p(x["amount"].clip(lower=0))
    x["event_ret"] = x["ret"]
    x["event_gap"] = x["gap"]
    x["event_body_ret"] = x["close"] / x["open"].replace(0, np.nan) - 1.0
    x["event_red"] = (x["ret"] < 0).astype(float)
    x["touched_upper"] = (x["high"] >= x["upper_calc"] - 0.0051).astype(float)

    # Tradability on D+1: exclude one-price limit-up opens.
    x["d1_tradable"] = (
        (x["d1_open"] < x["d1_upper_calc"] - 0.0051)
        | (x["d1_low"] < x["d1_upper_calc"] - 0.0051)
    )
    return x


def build_lifecycle_state(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Causal observation state machine.

    Primary lifecycle:
      live >=3 limit-up streak -> first non-limit-up day is Day1.

    After every Day1:
      keep the stock on the watchlist for 30 calendar days from Day2.
      If it reaches a NEW >=2 consecutive limit-up streak within that window,
      it becomes re-armed. The next non-limit-up day is a recycled Day1.
      A recycled Day1 starts a fresh 30-day watch window.

    Observation state is deliberately independent of whether a particular model
    bought or passed the prior Day2. This prevents sample membership from changing
    when the research model changes, while fully preserving the requested PASS ->
    keep watching behavior.
    """
    event_rows: list[dict] = []
    watch_rows: list[dict] = []
    event_index: dict[str, int] = {}
    dataset_last = pd.Timestamp(daily["date"].max())

    for code, z in daily.groupby("code", sort=False):
        z = z.sort_values("date").reset_index(drop=True)
        state = "DORMANT"
        generation = 0
        parent_event_id = None
        parent_event_date = None
        parent_decision_date = None
        watch_until = None
        rearm_date = None
        lifecycle_id = 0

        for row in z.itertuples(index=False):
            date = pd.Timestamp(row.date)
            lu = bool(row.lu_calc)
            streak = int(row.streak_calc)

            # Expired cooldown returns to dormant. A later fresh 3+ streak can start
            # a brand-new lifecycle even after the old stock was removed from watch.
            if state == "COOLDOWN" and watch_until is not None and date > watch_until:
                state = "DORMANT"
                generation = 0
                parent_event_id = None
                parent_event_date = None
                parent_decision_date = None
                watch_until = None
                rearm_date = None

            if state == "DORMANT":
                if lu and streak >= 3:
                    lifecycle_id += 1
                    state = "PRIMARY_ARMED"
                    generation = 0
                    parent_event_id = None
                    parent_event_date = None
                    parent_decision_date = None
                    rearm_date = date

            elif state == "COOLDOWN":
                # Re-arm only if the SECOND consecutive limit-up itself occurs
                # inside the 30-calendar-day window.
                if (
                    watch_until is not None
                    and date <= watch_until
                    and lu
                    and streak >= 2
                ):
                    state = "RECYCLE_ARMED"
                    generation += 1
                    rearm_date = date
                    if parent_event_id in event_index:
                        j = event_index[parent_event_id]
                        event_rows[j]["rearmed_within_30d"] = True
                        event_rows[j]["rearm_date"] = date
                        event_rows[j]["rearm_delay_days"] = (
                            (date - parent_decision_date).days
                            if parent_decision_date is not None else np.nan
                        )

            # Once armed, wait as long as necessary for the first break. If the
            # 2-board re-arm happened inside the 30-day window, the eventual break
            # remains valid even if it lands just after the original expiry date.
            if state in {"PRIMARY_ARMED", "RECYCLE_ARMED"} and not lu:
                min_streak = 3 if state == "PRIMARY_ARMED" else 2
                prior_streak = int(getattr(row, "prior_streak", 0))
                d1_date = getattr(row, "d1_date", pd.NaT)
                if prior_streak >= min_streak and pd.notna(d1_date):
                    event_id = f"{code}_{date.strftime('%Y%m%d')}_g{generation}"
                    d = row._asdict()
                    d.update({
                        "event_id": event_id,
                        "event_type": "PRIMARY_3PLUS" if state == "PRIMARY_ARMED" else "RECYCLE_2PLUS",
                        "lifecycle_id": int(lifecycle_id),
                        "generation": int(generation),
                        "generation_clip": float(min(generation, 3)),
                        "parent_event_id": parent_event_id,
                        "parent_event_date": parent_event_date,
                        "days_since_parent_event": (
                            float((date - parent_event_date).days)
                            if parent_event_date is not None else 0.0
                        ),
                        "rearm_date": rearm_date,
                        "rearm_delay_days": (
                            float((rearm_date - parent_decision_date).days)
                            if rearm_date is not None and parent_decision_date is not None else 0.0
                        ),
                        "rearmed_within_30d": False,
                        "watch_until": pd.Timestamp(d1_date) + pd.Timedelta(days=30),
                        "full_30d_observable": bool(pd.Timestamp(d1_date) + pd.Timedelta(days=30) <= dataset_last),
                    })
                    event_index[event_id] = len(event_rows)
                    event_rows.append(d)

                    parent_event_id = event_id
                    parent_event_date = date
                    parent_decision_date = pd.Timestamp(d1_date)
                    watch_until = parent_decision_date + pd.Timedelta(days=30)
                    rearm_date = None
                    state = "COOLDOWN"

            if state != "DORMANT":
                watch_rows.append({
                    "code": code,
                    "date": date,
                    "watch_state": state,
                    "generation": int(generation),
                    "parent_event_id": parent_event_id,
                    "watch_until": watch_until,
                    "days_remaining": (
                        max(0, int((watch_until - date).days))
                        if watch_until is not None and state == "COOLDOWN" else None
                    ),
                    "streak_calc": streak,
                    "market_height": int(row.market_height) if pd.notna(row.market_height) else None,
                })

    events = pd.DataFrame(event_rows)
    watch = pd.DataFrame(watch_rows)
    if events.empty:
        return events, watch

    events["date"] = pd.to_datetime(events["date"])
    events["d1_date"] = pd.to_datetime(events["d1_date"])
    events["month"] = events["date"].dt.strftime("%Y-%m")
    events["week"] = events["date"].dt.to_period("W-FRI").astype(str)
    events["is_two_board_recycle"] = (
        (events["event_type"] == "RECYCLE_2PLUS") & (events["prior_streak"] == 2)
    ).astype(float)
    events["is_three_board"] = (events["prior_streak"] == 3).astype(float)
    events["is_four_plus"] = (events["prior_streak"] >= 4).astype(float)
    events["is_recycle"] = (events["event_type"] == "RECYCLE_2PLUS").astype(float)
    events["height_ratio"] = (
        events["prior_streak"] / events["market_height_prev"].replace(0, np.nan)
    ).clip(lower=0.0, upper=1.5)

    # Keep only normal executable Day2 gap envelope.
    events = events[(events["d1_gap"] >= -0.102) & (events["d1_gap"] <= 0.102)].copy()

    events["pool_mid"] = (
        (events["height_gap"] <= 3)
        & (events["amount"] >= 2e8)
        & (events["range"] >= 0.05)
        & ((events["ret20"].isna()) | (events["ret20"] >= 0.15) | (events["prior_streak"] >= 5))
        & ((events["vol_prev_ratio"] >= 0.60) | (events["vol_med10"] >= 1.50))
    )
    events["pool_wide"] = (
        (events["height_gap"] <= 4)
        & (events["amount"] >= 1.5e8)
        & (events["range"] >= 0.04)
        & ((events["ret20"].isna()) | (events["ret20"] >= 0.08) | (events["prior_streak"] >= 5))
        & ((events["vol_prev_ratio"] >= 0.50) | (events["vol_med10"] >= 1.20))
    )
    events["pool_ultra"] = (
        (events["height_gap"] <= 5)
        & (events["amount"] >= 1e8)
        & (events["range"] >= 0.03)
        & ((events["ret20"].isna()) | (events["ret20"] >= 0.0) | (events["prior_streak"] >= 5))
        & ((events["vol_prev_ratio"] >= 0.40) | (events["vol_med10"] >= 1.00))
    )
    events["pool_max"] = (
        (events["height_gap"] <= 6)
        & (events["amount"] >= 8e7)
        & (events["range"] >= 0.025)
        & ((events["ret20"].isna()) | (events["ret20"] >= -0.05) | (events["prior_streak"] >= 5))
        & ((events["vol_prev_ratio"] >= 0.35) | (events["vol_med10"] >= 0.90))
    )
    # Baseline V15-compatible pool and the new full lifecycle pool.
    events["pool_all3plus"] = events["event_type"].eq("PRIMARY_3PLUS")
    events["pool_lifecycle"] = True
    return events, watch


def build_relaxed_events(daily: pd.DataFrame) -> pd.DataFrame:
    events, _ = build_lifecycle_state(daily)
    return events


def build_observation_pool(daily: pd.DataFrame) -> pd.DataFrame:
    _, watch = build_lifecycle_state(daily)
    if not watch.empty:
        watch["month"] = watch["date"].dt.strftime("%Y-%m")
        watch["week"] = watch["date"].dt.to_period("W-FRI").astype(str)
    return watch


def observation_diagnostics(daily: pd.DataFrame, events: pd.DataFrame) -> dict:
    watch = build_observation_pool(daily)
    day_counts = watch.groupby("date")["code"].nunique() if len(watch) else pd.Series(dtype=float)
    weekly_watch = watch.groupby("week")["code"].nunique() if len(watch) else pd.Series(dtype=float)
    weekly_day1 = events.groupby("week")["code"].count() if len(events) else pd.Series(dtype=float)

    full = events[events["full_30d_observable"]].copy() if len(events) else pd.DataFrame()
    rearm_rate = (
        float(full["rearmed_within_30d"].mean()) if len(full) else None
    )
    return {
        "watch_rule": "Day2 decision date + 30 calendar days; >=2 consecutive limit-ups inside window re-arm the stock",
        "watch_rows": int(len(watch)),
        "watch_unique_symbols": int(watch["code"].nunique()) if len(watch) else 0,
        "active_watch_per_day_mean": float(day_counts.mean()) if len(day_counts) else 0.0,
        "active_watch_per_day_median": float(day_counts.median()) if len(day_counts) else 0.0,
        "active_watch_per_day_max": int(day_counts.max()) if len(day_counts) else 0,
        "watch_state_counts": (
            watch["watch_state"].value_counts().astype(int).to_dict() if len(watch) else {}
        ),
        "events_total": int(len(events)),
        "primary_events": int(events["event_type"].eq("PRIMARY_3PLUS").sum()) if len(events) else 0,
        "recycle_events": int(events["event_type"].eq("RECYCLE_2PLUS").sum()) if len(events) else 0,
        "recycle_share": float(events["event_type"].eq("RECYCLE_2PLUS").mean()) if len(events) else 0.0,
        "full_30d_windows": int(len(full)),
        "rearmed_within_30d": int(full["rearmed_within_30d"].sum()) if len(full) else 0,
        "rearm_rate_30d": rearm_rate,
        "events_by_month": (
            events.groupby("month")["code"].count().astype(int).to_dict() if len(events) else {}
        ),
        "events_by_type": (
            events["event_type"].value_counts().astype(int).to_dict() if len(events) else {}
        ),
        "day1_by_prior_streak": (
            events["prior_streak"].value_counts().sort_index().astype(int).to_dict() if len(events) else {}
        ),
        "weekly_watch_symbols": {str(k): int(v) for k, v in weekly_watch.items()},
        "weekly_day1_events": {str(k): int(v) for k, v in weekly_day1.items()},
    }

def add_m30_features(e: pd.DataFrame, path_map: dict[str, str]) -> pd.DataFrame:
    e = e.copy()
    cols = {
        "m30_entry": np.nan,
        "m30_ret_open": np.nan,
        "m30_ret_event": np.nan,
        "m30_range": np.nan,
        "m30_close_loc": np.nan,
        "m30_close_vwap": np.nan,
        "m30_reclaim_event": np.nan,
        "m30_low_vs_event": np.nan,
        "m30_high_vs_event": np.nan,
        "m30_amount_frac": np.nan,
    }
    for c, v in cols.items():
        e[c] = v

    by_date = {}
    for idx, row in e.iterrows():
        d = pd.Timestamp(row["d1_date"]).strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append((idx, row["code"]))

    for d, pairs in by_date.items():
        path = path_map.get(d)
        if not path:
            continue
        codes = {c for _, c in pairs}
        cols = ["code","datetime","open","high","low","close","volume","amount"]
        try:
            m = pd.read_parquet(path, columns=cols, filters=[("code", "in", sorted(codes))])
        except Exception:
            m = pd.read_parquet(path, columns=cols)
        m["code"] = m["code"].astype(str).str.zfill(6)
        m = m[m["code"].isin(codes)].sort_values(["code","datetime"])
        if m.empty:
            continue
        for code, z in m.groupby("code"):
            z = z.reset_index(drop=True)
            # First 30 completed minutes are features. Execution is the NEXT minute open.
            # This prevents filling on the same minute close used to make the decision.
            if len(z) < 31:
                continue
            confirm = z.iloc[:30]
            next_bar = z.iloc[30]
            idxs = [idx for idx, c in pairs if c == code]
            if not idxs:
                continue
            first_open = float(confirm.iloc[0]["open"])
            last_close = float(confirm.iloc[-1]["close"])
            next_open = float(next_bar["open"])
            hh = float(confirm["high"].max())
            ll = float(confirm["low"].min())
            den = hh - ll
            vol = float(confirm["volume"].sum())
            amt = float(confirm["amount"].sum())
            vwap = amt / vol if vol > 0 else np.nan
            if np.isfinite(vwap) and last_close > 0 and vwap / last_close > 20:
                vwap /= 100.0
            for idx in idxs:
                event_close = float(e.at[idx, "close"])
                d1_amount = float(e.at[idx, "d1_amount"]) if "d1_amount" in e.columns and pd.notna(e.at[idx, "d1_amount"]) else np.nan
                e.at[idx, "m30_entry"] = next_open
                e.at[idx, "m30_ret_open"] = last_close / first_open - 1.0 if first_open > 0 else np.nan
                e.at[idx, "m30_ret_event"] = last_close / event_close - 1.0 if event_close > 0 else np.nan
                e.at[idx, "m30_range"] = (hh - ll) / first_open if first_open > 0 else np.nan
                e.at[idx, "m30_close_loc"] = (last_close - ll) / den if den > 0 else 0.5
                e.at[idx, "m30_close_vwap"] = last_close / vwap - 1.0 if np.isfinite(vwap) and vwap > 0 else np.nan
                e.at[idx, "m30_reclaim_event"] = float(last_close >= event_close * 0.998)
                e.at[idx, "m30_low_vs_event"] = ll / event_close - 1.0 if event_close > 0 else np.nan
                e.at[idx, "m30_high_vs_event"] = hh / event_close - 1.0 if event_close > 0 else np.nan
                e.at[idx, "m30_amount_frac"] = amt / d1_amount if np.isfinite(d1_amount) and d1_amount > 0 else np.nan
    return e


def attach_d1_amount(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["d1_amount"] = x.groupby("code")["amount"].shift(-1)
    return x


def add_fixed_targets(e: pd.DataFrame) -> pd.DataFrame:
    x = e.copy()
    x["h3_ret_open"] = x["h3_close"] / x["d1_open"] - 1.0
    x["h5_ret_open"] = x["h5_close"] / x["d1_open"] - 1.0
    x["h3_ret_confirm"] = x["h3_close"] / x["m30_entry"] - 1.0
    x["h5_ret_confirm"] = x["h5_close"] / x["m30_entry"] - 1.0
    return x


def make_models():
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
    ])
    reg = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=5.0)),
    ])
    return clf, reg


def fit_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str):
    tr = train[train[target].notna()].copy()
    te = test.copy()
    if len(tr) < 8:
        return pd.Series(np.nan, index=te.index), {}
    Xtr = tr[features]
    Xte = te[features]
    yret = tr[target].astype(float)
    y = (yret > 0).astype(int)

    reg = make_models()[1]
    reg.fit(Xtr, yret)
    pret = reg.predict(Xte)

    coef = {}
    try:
        reg_coef = reg.named_steps["model"].coef_
        coef = {f: float(c) for f, c in zip(features, reg_coef)}
    except Exception:
        pass

    if y.nunique() < 2:
        pwin = np.full(len(te), float(y.iloc[0]))
    else:
        clf = make_models()[0]
        clf.fit(Xtr, y)
        pwin = clf.predict_proba(Xte)[:, 1]

    ret_component = 1.0 / (1.0 + np.exp(-pret / 0.08))
    score = 0.65 * pwin + 0.35 * ret_component
    return pd.Series(score, index=te.index), coef


@dataclass(frozen=True)
class ExitSpec:
    name: str
    max_hold: int
    stop_close: float | None
    take_close: float | None


EXIT_SPECS = [
    ExitSpec("H2", 2, None, None),
    ExitSpec("H3", 3, None, None),
    ExitSpec("H5", 5, None, None),
    ExitSpec("SL8_TP15_H3", 3, -0.08, 0.15),
    ExitSpec("SL8_TP20_H5", 5, -0.08, 0.20),
    ExitSpec("SL10_TP25_H5", 5, -0.10, 0.25),
]


def build_daily_groups(daily: pd.DataFrame):
    return {c: z.sort_values("date").reset_index(drop=True) for c, z in daily.groupby("code")}


def simulate_close_exit(row, entry_price: float, spec: ExitSpec, groups) -> float:
    if not np.isfinite(entry_price) or entry_price <= 0:
        return np.nan
    z = groups.get(row["code"])
    if z is None:
        return np.nan
    arr = z.index[z["date"] == row["d1_date"]].to_numpy()
    if len(arr) == 0:
        return np.nan
    start = int(arr[0])
    window = z.iloc[start:start + spec.max_hold]
    if len(window) < spec.max_hold:
        return np.nan
    for j, day in enumerate(window.itertuples(), 1):
        r = float(day.close) / entry_price - 1.0
        if spec.stop_close is not None and r <= spec.stop_close:
            return r
        if spec.take_close is not None and r >= spec.take_close:
            return r
        if j == spec.max_hold:
            return r
    return np.nan


def precompute_exit_returns(e: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    x = e.copy()
    groups = build_daily_groups(daily)
    for spec in EXIT_SPECS:
        o, c = [], []
        for _, row in x.iterrows():
            o.append(simulate_close_exit(row, float(row["d1_open"]), spec, groups))
            c.append(simulate_close_exit(row, float(row["m30_entry"]) if pd.notna(row["m30_entry"]) else np.nan, spec, groups))
        x[f"ret_open_{spec.name}"] = o
        x[f"ret_confirm_{spec.name}"] = c
    return x


def apply_hybrid(
    data: pd.DataFrame,
    open_score: pd.Series,
    confirm_score: pd.Series,
    train_open_scores: pd.Series,
    train_confirm_scores: pd.Series,
    q_open: float,
    q_watch: float,
    q_confirm: float,
    spec: ExitSpec,
):
    if train_open_scores.dropna().empty:
        return pd.DataFrame()
    open_thr = float(train_open_scores.dropna().quantile(q_open))
    watch_thr = float(train_open_scores.dropna().quantile(q_watch))
    if train_confirm_scores.dropna().empty:
        confirm_thr = np.inf
    else:
        confirm_thr = float(train_confirm_scores.dropna().quantile(q_confirm))

    rows = []
    for idx, row in data.iterrows():
        oscore = open_score.get(idx, np.nan)
        cscore = confirm_score.get(idx, np.nan)
        mode = None
        ret = np.nan
        entry = np.nan
        # Extreme low gaps must prove themselves intraday; do not buy them at the open.
        open_allowed = bool(row["d1_tradable"]) and float(row["d1_gap"]) >= -0.03
        if np.isfinite(oscore) and oscore >= open_thr and open_allowed:
            mode = "OPEN"
            ret = row[f"ret_open_{spec.name}"]
            entry = row["d1_open"]
        elif (
            np.isfinite(oscore) and oscore >= watch_thr
            and np.isfinite(cscore) and cscore >= confirm_thr
            and pd.notna(row["m30_entry"])
        ):
            mode = "M30"
            ret = row[f"ret_confirm_{spec.name}"]
            entry = row["m30_entry"]
        if mode is None or pd.isna(ret):
            continue
        rows.append({
            "idx": int(idx),
            "code": row["code"],
            "event_date": row["date"],
            "entry_date": row["d1_date"],
            "entry_mode": mode,
            "entry_price": float(entry),
            "open_score": float(oscore),
            "confirm_score": float(cscore) if np.isfinite(cscore) else None,
            "ret": float(ret),
            "prior_streak": int(row["prior_streak"]),
            "streak_bucket": (
                "2_recycle" if int(row["prior_streak"]) == 2
                else ("3" if int(row["prior_streak"]) == 3 else "4plus")
            ),
            "event_type": row.get("event_type", None),
            "generation": int(row.get("generation", 0)),
            "parent_event_id": row.get("parent_event_id", None),
            "height_gap": float(row["height_gap"]) if pd.notna(row["height_gap"]) else None,
            "ret20": float(row["ret20"]) if pd.notna(row["ret20"]) else None,
            "event_ret": float(row["event_ret"]),
            "d1_gap": float(row["d1_gap"]),
        })
    return pd.DataFrame(rows)


def metrics(trades: pd.DataFrame):
    if trades.empty:
        return {"trades":0,"win_rate":None,"mean":None,"median":None,"sum":0.0,"worst":None}
    r = trades["ret"].astype(float)
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "sum": float(r.sum()),
        "worst": float(r.min()),
    }


PROFILES = {
    "QUALITY": {"min_share":0.12, "min_win":0.58, "min_mean":0.005, "w_mean":110.0, "w_win":13.0, "w_n":0.22, "w_worst":50.0},
    "BALANCED": {"min_share":0.22, "min_win":0.54, "min_mean":0.000, "w_mean":100.0, "w_win":10.5, "w_n":0.65, "w_worst":38.0},
    "QUANTITY": {"min_share":0.32, "min_win":0.50, "min_mean":0.000, "w_mean":78.0, "w_win":8.0, "w_n":1.10, "w_worst":28.0},
}


def utility(m, profile):
    if m["trades"] == 0 or m["mean"] is None:
        return -1e9
    worst = min(0.0, m["worst"])
    return (
        profile["w_mean"] * m["mean"]
        + profile["w_win"] * (m["win_rate"] - 0.5)
        + profile["w_n"] * m["trades"]
        + profile["w_worst"] * worst
    )


def subset_pool(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    return df[df[f"pool_{pool}"]].copy()


def score_bundle(train, target, pool):
    tr = subset_pool(train, pool)
    te = subset_pool(target, pool)
    os, ocoef = fit_score(tr, te, OPEN_FEATURES, "h3_ret_open")
    cs, ccoef = fit_score(tr[tr["m30_entry"].notna()], te, CONFIRM_FEATURES, "h3_ret_confirm")
    # score train itself only to derive causal thresholds from the training distribution
    ots, _ = fit_score(tr, tr, OPEN_FEATURES, "h3_ret_open")
    confirm_train = tr[tr["m30_entry"].notna()].copy()
    cts, _ = fit_score(confirm_train, confirm_train, CONFIRM_FEATURES, "h3_ret_confirm")
    return tr, te, os, cs, ots, cts, ocoef, ccoef


def select_inner_config(inner_train, inner_val, profile_name):
    profile = PROFILES[profile_name]
    best = None
    q_opens = [0.55, 0.65, 0.75, 0.85]
    q_watchs = [0.25, 0.40, 0.55]
    q_confirms = [0.50, 0.65, 0.80]
    for pool in ["mid","wide","ultra","max","all3plus","lifecycle"]:
        tr, va, os, cs, ots, cts, _, _ = score_bundle(inner_train, inner_val, pool)
        if len(tr) < 8 or len(va) == 0:
            continue
        min_trades = max(1, int(math.ceil(profile["min_share"] * len(va))))
        candidates = []
        for qo in q_opens:
            for qw in q_watchs:
                if qw >= qo:
                    continue
                for qc in q_confirms:
                    for spec in EXIT_SPECS:
                        t = apply_hybrid(va, os, cs, ots, cts, qo, qw, qc, spec)
                        m = metrics(t)
                        if m["trades"] < min_trades:
                            continue
                        # Every profile must clear a positive-quality floor. Quantity is allowed
                        # to be broader, but never by accepting an explicitly losing validation slice.
                        if m["win_rate"] < profile["min_win"] or m["mean"] < profile["min_mean"]:
                            continue
                        candidates.append((utility(m, profile), pool, qo, qw, qc, spec, m))
        if candidates:
            z = max(candidates, key=lambda a: a[0])
            if best is None or z[0] > best[0]:
                best = z
    return best


def run_walkforward(e: pd.DataFrame):
    months = ["2026-03","2026-04","2026-05","2026-06","2026-07","2026-08"]
    outer_tests = ["2026-05","2026-06","2026-07","2026-08"]
    results = {}
    for profile_name in PROFILES:
        fold_rows = []
        all_trades = []
        for test_month in outer_tests:
            ti = months.index(test_month)
            inner_val_month = months[ti - 1]
            inner_train_months = months[:ti - 1]
            outer_train_months = months[:ti]
            if not inner_train_months:
                continue
            inner_train = e[e["month"].isin(inner_train_months)].copy()
            inner_val = e[e["month"] == inner_val_month].copy()
            chosen = select_inner_config(inner_train, inner_val, profile_name)
            if chosen is None:
                continue
            val_u, pool, qo, qw, qc, spec, val_m = chosen

            outer_train = e[e["month"].isin(outer_train_months)].copy()
            outer_test = e[e["month"] == test_month].copy()
            tr, te, os, cs, ots, cts, ocoef, ccoef = score_bundle(outer_train, outer_test, pool)
            trades = apply_hybrid(te, os, cs, ots, cts, qo, qw, qc, spec)
            m = metrics(trades)
            if not trades.empty:
                trades["test_month"] = test_month
                trades["profile"] = profile_name
                trades["pool"] = pool
                trades["exit"] = spec.name
                all_trades.append(trades)

            fold_rows.append({
                "test_month": test_month,
                "inner_val_month": inner_val_month,
                "pool": pool,
                "q_open": qo,
                "q_watch": qw,
                "q_confirm": qc,
                "exit": spec.name,
                "inner_val_utility": float(val_u),
                "inner_val_metrics": val_m,
                "outer_test_metrics": m,
                "outer_test_pool_candidates": int(len(te)),
                "top_open_coef": sorted(ocoef.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6],
                "top_confirm_coef": sorted(ccoef.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6],
            })
        combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        results[profile_name] = {
            "folds": fold_rows,
            "combined_metrics": metrics(combined),
            "trades": combined.to_dict("records") if not combined.empty else [],
        }
    return results


def pool_diagnostics(e: pd.DataFrame):
    out = {}
    for pool in ["mid","wide","ultra","max","all3plus","lifecycle"]:
        z = e[e[f"pool_{pool}"]]
        out[pool] = {
            "events": int(len(z)),
            "by_month": z["month"].value_counts().sort_index().astype(int).to_dict(),
            "prior_streak": z["prior_streak"].value_counts().sort_index().astype(int).to_dict(),
            "height_gap_median": float(z["height_gap"].median()) if len(z) else None,
            "ret20_median": float(z["ret20"].median()) if len(z) else None,
            "amount_median_m": float(z["amount"].median()/1e6) if len(z) else None,
        }
    return out


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main():
    daily, path_map = load_daily()
    daily = build_features(daily)
    daily = attach_d1_amount(daily)
    daily = add_history_features(daily)
    events = build_relaxed_events(daily)
    print("V16|lifecycle_day1_raw", len(events))
    print("V16|pool_diag", json.dumps(pool_diagnostics(events), ensure_ascii=False))

    events = add_m30_features(events, path_map)
    events = add_fixed_targets(events)
    events = precompute_exit_returns(events, daily)

    obs = build_observation_pool(daily)
    obs.to_csv(OUT/"observation_pool.csv", index=False)

    results = run_walkforward(events)
    payload = {
        "method": {
            "candidate_layer": "primary 3+ streaks create Day1; every Day1 starts a 30-calendar-day watch window; a new >=2 streak inside that window re-arms the stock and its next break becomes a recycled Day1",
            "buy_layer": "regularized logistic win model + ridge return model; high score open entry, medium score requires 30 completed minutes then executes at minute 31 open",
            "sell_layer": "nested walk-forward selects among close-based holding/stop/take rules; every profile must clear non-losing validation quality floors",
            "outer_tests": ["2026-05","2026-06","2026-07","2026-08"],
            "anti_leakage": "inner previous-month validation chooses pool/thresholds/exit; outer month is then predicted using only earlier months",
            "three_board_policy": "3-board starts a fresh lifecycle; recycled waves need only >=2 boards inside the active 30-day window; neither condition is an automatic buy",
            "expiry_policy": "if no new >=2 streak reaches its second board within 30 calendar days after Day2, the lifecycle expires; a later fresh >=3 streak may start a new lifecycle",
        },
        "observation_diagnostics": observation_diagnostics(daily, events),
        "pool_diagnostics": pool_diagnostics(events),
        "profiles": results,
    }
    payload = json_safe(payload)
    (OUT/"summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save the relaxed event table for auditability.
    save_cols = [
        "code","date","d1_date","month","week","event_id","event_type","lifecycle_id","generation","parent_event_id",
        "parent_event_date","watch_until","rearmed_within_30d","rearm_date","rearm_delay_days","days_since_parent_event",
        "prior_streak","market_height_prev","height_gap","height_ratio","is_two_board_recycle","is_three_board","is_four_plus","is_recycle",
        "ret5","ret10","ret20","event_ret","event_gap","event_body_ret","event_red","range","vol_prev_ratio","vol_med10",
        "amount","close_vwap","close_loc","touched_upper","d1_gap","d1_tradable",
        "pool_mid","pool_wide","pool_ultra","pool_max","pool_all3plus","pool_lifecycle","m30_entry","m30_ret_open","m30_ret_event",
        "m30_close_loc","m30_close_vwap","m30_reclaim_event","h3_ret_open","h3_ret_confirm",
    ]
    events[save_cols].to_csv(OUT/"relaxed_events.csv", index=False)

    rows = []
    for profile, r in results.items():
        for f in r["folds"]:
            rows.append({
                "profile":profile,
                "test_month":f["test_month"],
                "pool":f["pool"],
                "q_open":f["q_open"],
                "q_watch":f["q_watch"],
                "q_confirm":f["q_confirm"],
                "exit":f["exit"],
                **{f"test_{k}":v for k,v in f["outer_test_metrics"].items()},
            })
    pd.DataFrame(rows).to_csv(OUT/"walkforward_folds.csv", index=False)

    print("V16_SUMMARY_JSON_BEGIN")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("V16_SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
