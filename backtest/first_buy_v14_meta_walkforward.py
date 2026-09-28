#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import first_buy_v14_robust as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtest" / "results_first_buy_v14_meta"
OUT.mkdir(parents=True, exist_ok=True)

META_FEATURES = [
    "sector_rank", "prior_sector_rank", "sector_lu3", "prior_sector_lu3",
    "sector_n", "leader_flag", "prior_leader_flag", "shock_flag",
    "rank1_flag", "prior_rank1_flag", "meta_missing",
]
OPEN_FEATURES = base.OPEN_FEATURES + META_FEATURES
CONFIRM_FEATURES = OPEN_FEATURES + base.CONFIRM_EXTRA

MONTHS = ["2026-03","2026-04","2026-05","2026-06","2026-07","2026-08"]
OUTER_TESTS = ["2026-06","2026-07","2026-08"]

PROFILE = {
    "QUALITY": {"min_share":0.12, "min_win":0.60, "min_mean":0.00, "w_mean":120.0, "w_win":14.0, "w_n":0.18, "w_worst":55.0},
    "BALANCED": {"min_share":0.20, "min_win":0.55, "min_mean":0.00, "w_mean":105.0, "w_win":11.0, "w_n":0.55, "w_worst":42.0},
    "ACTIVE": {"min_share":0.30, "min_win":0.52, "min_mean":0.00, "w_mean":90.0, "w_win":9.0, "w_n":0.90, "w_worst":32.0},
}


def load_meta():
    rows = []
    for p in sorted((ROOT/"backtest"/"v13_meta").glob("batch_*.json")):
        rows.extend(json.loads(p.read_text(encoding="utf-8")))
    m = pd.DataFrame(rows)
    m = m[~m.get("pending", False).fillna(False)] if "pending" in m.columns else m
    m = m[m["code"].notna() & m["date"].notna()].copy()
    m["code"] = m["code"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("issue").drop_duplicates(["code","date"], keep="last")
    return m


def add_meta(events):
    m = load_meta()
    keep = ["code","date","sector_rank","sector_n","sector_lu3","leader","prior_leader",
            "prior_sector_rank","prior_sector_lu3","shock_base","theme_primary"]
    x = events.merge(m[keep], on=["code","date"], how="left")
    for c in ["sector_rank","sector_n","sector_lu3","prior_sector_rank","prior_sector_lu3"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["leader_flag"] = x["leader"].fillna(False).astype(float)
    x["prior_leader_flag"] = x["prior_leader"].fillna(False).astype(float)
    x["shock_flag"] = x["shock_base"].fillna(False).astype(float)
    x["rank1_flag"] = (x["sector_rank"] <= 1).astype(float)
    x["prior_rank1_flag"] = (x["prior_sector_rank"] <= 1).astype(float)
    x["meta_missing"] = x["sector_rank"].isna().astype(float)
    return x


def score_bundle(train, target, pool):
    tr = base.subset_pool(train, pool)
    te = base.subset_pool(target, pool)
    os, ocoef = base.fit_score(tr, te, OPEN_FEATURES, "h3_ret_open")
    cs, ccoef = base.fit_score(tr[tr["m30_entry"].notna()], te, CONFIRM_FEATURES, "h3_ret_confirm")
    ots, _ = base.fit_score(tr, tr, OPEN_FEATURES, "h3_ret_open")
    ctr = tr[tr["m30_entry"].notna()].copy()
    cts, _ = base.fit_score(ctr, ctr, CONFIRM_FEATURES, "h3_ret_confirm")
    return tr, te, os, cs, ots, cts, ocoef, ccoef


def candidate_configs():
    for pool in ["mid","wide","ultra","max"]:
        for qo in [0.60,0.70,0.80,0.88]:
            for qw in [0.35,0.50,0.60]:
                if qw >= qo:
                    continue
                for qc in [0.55,0.70,0.82]:
                    for spec in base.EXIT_SPECS:
                        yield pool,qo,qw,qc,spec


def fold_for_config(events, train_months, test_month, cfg):
    pool,qo,qw,qc,spec = cfg
    trn = events[events["month"].isin(train_months)].copy()
    tst = events[events["month"] == test_month].copy()
    tr,te,os,cs,ots,cts,ocoef,ccoef = score_bundle(trn,tst,pool)
    if len(tr) < 8 or len(te) == 0:
        return None
    trades = base.apply_hybrid(te,os,cs,ots,cts,qo,qw,qc,spec)
    return {
        "metrics": base.metrics(trades),
        "trades": trades,
        "candidates": len(te),
        "ocoef": ocoef,
        "ccoef": ccoef,
    }


def historical_cv(events, cutoff_month, cfg):
    # For deciding cutoff_month, only pseudo-OOS months strictly before it are allowed.
    ci = MONTHS.index(cutoff_month)
    val_months = MONTHS[2:ci]  # starts May; each needs at least Mar-Apr training
    fold_results = []
    for vm in val_months:
        vi = MONTHS.index(vm)
        train_months = MONTHS[:vi]
        r = fold_for_config(events, train_months, vm, cfg)
        if r is not None:
            fold_results.append((vm,r))
    return fold_results


def aggregate_cv(folds):
    parts = []
    month_stats = []
    for m,r in folds:
        t = r["trades"]
        if not t.empty:
            z = t.copy(); z["cv_month"] = m; parts.append(z)
        month_stats.append((m,r["metrics"]))
    alltr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    agg = base.metrics(alltr)
    positives = sum(1 for _,m in month_stats if m["mean"] is not None and m["mean"] > 0)
    nonneg = sum(1 for _,m in month_stats if m["mean"] is not None and m["mean"] >= 0)
    agg["positive_months"] = positives
    agg["nonnegative_months"] = nonneg
    agg["folds"] = len(month_stats)
    return agg, month_stats


def utility(m, p):
    if m["trades"] == 0 or m["mean"] is None:
        return -1e9
    worst = min(0.0, m["worst"])
    stability = (m["positive_months"] / m["folds"]) if m["folds"] else 0.0
    return (
        p["w_mean"]*m["mean"] +
        p["w_win"]*(m["win_rate"]-0.5) +
        p["w_n"]*math.log1p(m["trades"]) +
        p["w_worst"]*worst +
        2.5*stability
    )


def select_config(events, test_month, profile_name):
    p = PROFILE[profile_name]
    best = None
    for cfg in candidate_configs():
        folds = historical_cv(events,test_month,cfg)
        if not folds:
            continue
        agg,month_stats = aggregate_cv(folds)
        total_candidates = sum(r["candidates"] for _,r in folds)
        min_trades = max(2, math.ceil(p["min_share"]*total_candidates))
        if agg["trades"] < min_trades:
            continue
        if agg["win_rate"] < p["min_win"] or agg["mean"] < p["min_mean"]:
            continue
        # Require at least half of prior validation months to be non-negative.
        if agg["folds"] >= 2 and agg["nonnegative_months"] < math.ceil(agg["folds"]/2):
            continue
        u = utility(agg,p)
        if best is None or u > best[0]:
            best = (u,cfg,agg,month_stats)
    return best


def run(events):
    results={}
    for profile_name in PROFILE:
        folds=[]; alltr=[]
        for tm in OUTER_TESTS:
            selected=select_config(events,tm,profile_name)
            if selected is None:
                folds.append({"test_month":tm,"selected":None})
                continue
            u,cfg,cvagg,cvmonths=selected
            ti=MONTHS.index(tm)
            test=fold_for_config(events,MONTHS[:ti],tm,cfg)
            pool,qo,qw,qc,spec=cfg
            if test is None:
                continue
            trades=test["trades"]
            if not trades.empty:
                z=trades.copy(); z["test_month"]=tm; z["profile"]=profile_name; alltr.append(z)
            folds.append({
                "test_month":tm,"pool":pool,"q_open":qo,"q_watch":qw,"q_confirm":qc,"exit":spec.name,
                "selection_utility":u,"prior_cv":cvagg,
                "prior_cv_months":[{"month":m,**met} for m,met in cvmonths],
                "test_metrics":test["metrics"],"test_candidates":test["candidates"],
                "top_open_coef":sorted(test["ocoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:8],
                "top_confirm_coef":sorted(test["ccoef"].items(),key=lambda kv:abs(kv[1]),reverse=True)[:8],
            })
        combined=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
        results[profile_name]={
            "combined_metrics":base.metrics(combined),
            "folds":folds,
            "trades":combined.to_dict("records") if not combined.empty else [],
        }
    return results


def safe(v):
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,list): return [safe(x) for x in v]
    if isinstance(v,tuple): return [safe(x) for x in v]
    if isinstance(v,pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v,float) and math.isnan(v): return None
    return v


def main():
    daily,path_map=base.load_daily()
    daily=base.build_features(daily)
    daily=base.attach_d1_amount(daily)
    daily=base.add_history_features(daily)
    events=base.build_relaxed_events(daily)
    events=base.add_m30_features(events,path_map)
    events=base.add_fixed_targets(events)
    events=base.precompute_exit_returns(events,daily)
    events=add_meta(events)

    cover={
        "events_total":int(len(events)),
        "ultra_events":int(events["pool_ultra"].sum()),
        "ultra_with_meta":int(events.loc[events["pool_ultra"],"meta_missing"].eq(0).sum()),
        "meta_missing_ultra":int(events.loc[events["pool_ultra"],"meta_missing"].sum()),
    }
    result=run(events)
    payload=safe({
        "method":"V14 four-tier candidate recall + theme/leader metadata as soft features + causal minute-31 execution + multi-month nested walk-forward selection",
        "metadata_coverage":cover,
        "profiles":result,
    })
    (OUT/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    events.to_csv(OUT/"events_with_meta.csv",index=False)

    print("V13_META_SUMMARY_BEGIN")
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    print("V13_META_SUMMARY_END")


if __name__=="__main__":
    main()
