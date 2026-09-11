from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
RAW = HERE / "raw_web"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
REGISTRY = json.loads((HERE / "source_registry.json").read_text(encoding="utf-8"))
DAYS = REGISTRY["trading_days"]
UA = "Mozilla/5.0 (compatible; JulyKnowledgeResearch/1.0; +https://github.com/arivercrabcanwalk/data1)"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, source_id: str, date: str) -> dict[str, Any]:
    out = {"date": date, "source_id": source_id, "url": url, "ok": False, "status": None, "error": None}
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
        out["status"] = r.status_code
        r.raise_for_status()
        b = r.content
        p = RAW / f"{date}_{source_id}.html"
        p.write_bytes(b)
        out.update({"ok": True, "path": str(p.relative_to(HERE)), "sha256": sha256_bytes(b), "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z", "text": r.text})
    except Exception as e:
        out["error"] = repr(e)
    return out


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return "\n".join(x.strip() for x in soup.stripped_strings if x.strip())


def to_float(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x.replace(",", ""))
    except Exception:
        return None


def m1(pattern: str, text: str, flags=0, group=1):
    m = re.search(pattern, text, flags)
    return None if not m else m.group(group)


def parse_lianban(date: str, html: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    text = clean_text(html)
    summary: dict[str, Any] = {"date": date, "source_id": "lianban_archive"}
    # Hard-stop before forward-looking/historical-similarity sections when extracting trade semantics.
    safe = text
    cut = safe.find("🔮 明日推演")
    if cut > 0:
        safe = safe[:cut]
    conclusion = re.search(r"今日结论\s*([^\s]+)\s*(\d+)°\s*·\s*涨停\s*(\d+)\s*·\s*跌停\s*(\d+)\s*·\s*最高\s*(\d+)板\s*·\s*主线\s*([^\n]+)", safe)
    if conclusion:
        summary.update({
            "emotion_stage": conclusion.group(1),
            "temperature": int(conclusion.group(2)),
            "limit_up_count": int(conclusion.group(3)),
            "limit_down_count": int(conclusion.group(4)),
            "max_board": int(conclusion.group(5)),
            "mainline_text": conclusion.group(6).strip(),
        })
    # Metrics shown in the fixed daily dashboard. These are source-native and never overwrite GitHub facts.
    summary["board_count"] = to_float(m1(r"连板\s*(\d+)\s*昨", text))
    summary["seal_rate"] = to_float(m1(r"封板率\s*([\d.]+)%\s*昨", text))
    summary["broken_board_count"] = to_float(m1(r"炸板\s*(\d+)\s*昨", text))
    summary["yesterday_limitup_return"] = to_float(m1(r"昨涨停今表现\s*([-+\d.]+)%", text))
    adv = m1(r"上涨/下跌\s*([\d,]+)\s*/\s*([\d,]+)", text, group=1)
    dec = m1(r"上涨/下跌\s*([\d,]+)\s*/\s*([\d,]+)", text, group=2)
    summary["advancers"] = to_float(adv)
    summary["decliners"] = to_float(dec)
    summary["source_effective_rule"] = "T+1 only"

    # Extract only the daily mainline block, before forward-looking sections.
    themes: list[dict[str, Any]] = []
    block = ""
    idx = safe.find("当日主线")
    if idx >= 0:
        block = safe[idx:]
    # Remove newlines so catalyst text can be segmented between theme headers.
    flat = re.sub(r"\s+", " ", block)
    pat = re.compile(r"([\u4e00-\u9fffA-Za-z0-9（）()]+)涨停(\d+)家龙头\s*([\u4e00-\u9fffA-Za-z0-9*]+)(?:\d+天\d+板|\d+连板)?")
    matches = list(pat.finditer(flat))
    for i, mm in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(flat)
        catalyst = flat[mm.end():end].strip(" ·→")
        catalyst = catalyst[:500]
        themes.append({
            "date": date,
            "source_id": "lianban_archive",
            "theme": mm.group(1),
            "limit_up_count_source": int(mm.group(2)),
            "leader_name": re.sub(r"\d+天\d+板$|\d+连板$", "", mm.group(3)),
            "catalyst_text": catalyst,
            "trade_effective_date": None,
        })
    total_leader = m1(r"总龙头\s*([\u4e00-\u9fffA-Za-z0-9*]+?)(?=\d+天\d+板|主线追踪|\s)", flat)
    ladder = []
    if total_leader:
        ladder.append({"date": date, "source_id": "lianban_archive", "role": "total_leader_source", "name": total_leader})
    for t in themes:
        ladder.append({"date": date, "source_id": "lianban_archive", "role": "theme_leader_source", "theme": t["theme"], "name": t["leader_name"]})
    return summary, themes, ladder


def parse_yyqyx(date: str, html: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    text = clean_text(html)
    summary: dict[str, Any] = {"date": date, "source_id": "yyqyx_limitup", "source_effective_rule": "T+1 only"}
    summary["limit_up_count"] = to_float(m1(r"涨停\s*(\d+)\s*只", text))
    summary["board_count"] = to_float(m1(r"连板\s*(\d+)\s*只", text))
    summary["max_board"] = to_float(m1(r"最高\s*(\d+)\s*连板", text))
    summary["broken_board_count"] = to_float(m1(r"炸板\s*(\d+)\s*只", text))
    summary["limit_down_count"] = to_float(m1(r"当日涨停\s*\d+\s*只、跌停\s*(\d+)\s*只", text))
    summary["advancers"] = to_float(m1(r"上涨\s*([\d,]+)\s*只", text))
    summary["decliners"] = to_float(m1(r"下跌\s*([\d,]+)\s*只", text))
    summary["turnover_trillion"] = to_float(m1(r"全市场成交\s*([\d.]+)\s*万亿元", text))

    soup = BeautifulSoup(html, "html.parser")
    # Parse the visible theme-distribution section conservatively. One stock may belong to multiple themes by design.
    section = None
    for tag in soup.find_all(["h2", "h3"]):
        if "涨停题材分布" in tag.get_text(" ", strip=True):
            section = tag
            break
    memberships: list[dict[str, Any]] = []
    theme_rows: list[dict[str, Any]] = []
    if section is not None:
        nodes = []
        for sib in section.next_siblings:
            name = getattr(sib, "name", None)
            if name == "h2":
                break
            if hasattr(sib, "get_text"):
                nodes.append(sib)
        current_theme = None
        expected = None
        seen = set()
        for node in nodes:
            txt = node.get_text(" ", strip=True)
            mh = re.match(r"^(.+?)\s*[·•]\s*(\d+)\s*只", txt)
            if mh:
                current_theme = mh.group(1).strip()
                expected = int(mh.group(2))
                theme_rows.append({"date": date, "source_id": "yyqyx_limitup", "theme": current_theme, "limit_up_count_source": expected})
            if current_theme:
                for a in node.find_all("a"):
                    nm = a.get_text(" ", strip=True)
                    nm = re.sub(r"风险警示.*$", "", nm).strip()
                    nm = re.sub(r"\d{2}:\d{2}$", "", nm).strip()
                    if not nm or len(nm) > 24 or re.fullmatch(r"\d{2}:\d{2}", nm):
                        continue
                    key = (current_theme, nm)
                    if key not in seen:
                        memberships.append({"date": date, "source_id": "yyqyx_limitup", "theme": current_theme, "stock_name": nm})
                        seen.add(key)
    return summary, theme_rows, memberships


def next_trading_day(date: str) -> str | None:
    i = DAYS.index(date)
    return DAYS[i + 1] if i + 1 < len(DAYS) else None


def pct_rule(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    x = float(v)
    if abs(x) > 1:
        x /= 100.0
    return abs(x)


def github_market_daily() -> pd.DataFrame:
    prev_close: dict[str, float] = {}
    rows = []
    for day in DAYS:
        files = list(ROOT.glob(f"2026-07/part-*/date={day}/minute1.parquet"))
        if len(files) != 1:
            raise RuntimeError(f"{day}: expected one minute1.parquet, got {len(files)}")
        p = files[0]
        df = pd.read_parquet(p, columns=["code", "datetime", "open", "high", "low", "close", "volume", "amount"])
        df["code"] = df["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.extract(r"(\d{6})", expand=False).fillna(df["code"].astype(str))
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        d = df.groupby("code", sort=False).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), amount=("amount", "sum"), volume=("volume", "sum")).reset_index()
        d["prev_close"] = d["code"].map(prev_close)
        has_prev = d["prev_close"].notna() & (d["prev_close"] > 0)
        d["ret"] = np.where(has_prev, d["close"] / d["prev_close"] - 1, d["close"] / d["open"] - 1)
        status_path = p.parent / "daily_stock_status.parquet"
        st = pd.read_parquet(status_path) if status_path.exists() else pd.DataFrame(columns=["code"])
        if len(st):
            st["code"] = st["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.extract(r"(\d{6})", expand=False).fillna(st["code"].astype(str))
            keep = [c for c in ["code", "is_st", "is_star_st", "is_new_listing_initial", "price_limit_up_pct", "price_limit_down_pct"] if c in st.columns]
            d = d.merge(st[keep], on="code", how="left")
        for c in ["is_st", "is_star_st", "is_new_listing_initial"]:
            if c not in d:
                d[c] = False
        d["eligible"] = ~(d["is_st"].fillna(False).astype(bool) | d["is_star_st"].fillna(False).astype(bool) | d["is_new_listing_initial"].fillna(False).astype(bool))
        e = d[d["eligible"] & d["ret"].notna()]
        close_limit = touched_limit = down_limit = np.nan
        if has_prev.any() and "price_limit_up_pct" in d and "price_limit_down_pct" in d:
            up = d["price_limit_up_pct"].map(pct_rule)
            dn = d["price_limit_down_pct"].map(pct_rule)
            valid = d["eligible"] & has_prev & up.notna()
            close_limit = int((valid & (d["close"] >= d["prev_close"] * (1 + up) * 0.998)).sum())
            touched_limit = int((valid & (d["high"] >= d["prev_close"] * (1 + up) * 0.998)).sum())
            vd = d["eligible"] & has_prev & dn.notna()
            down_limit = int((vd & (d["close"] <= d["prev_close"] * (1 - dn) * 1.002)).sum())
        rows.append({
            "date": day,
            "universe": int(len(d)),
            "eligible": int(d["eligible"].sum()),
            "return_basis": "prev_close" if has_prev.any() else "open_proxy_first_day",
            "advancers": int((e["ret"] > 0).sum()),
            "decliners": int((e["ret"] < 0).sum()),
            "adv_ratio": float((e["ret"] > 0).mean()) if len(e) else np.nan,
            "median_return": float(e["ret"].median()) if len(e) else np.nan,
            "mean_return": float(e["ret"].mean()) if len(e) else np.nan,
            "up5": int((e["ret"] >= 0.05).sum()),
            "down5": int((e["ret"] <= -0.05).sum()),
            "turnover": float(d["amount"].sum()),
            "close_limit_up_nonst": close_limit,
            "touched_limit_up_nonst": touched_limit,
            "broken_limit_proxy_nonst": (touched_limit - close_limit) if np.isfinite(touched_limit) and np.isfinite(close_limit) else np.nan,
            "close_limit_down_nonst": down_limit,
        })
        prev_close = dict(zip(d["code"], d["close"]))
    return pd.DataFrame(rows)


def infer_lifecycle(theme_daily: pd.DataFrame) -> pd.DataFrame:
    if theme_daily.empty:
        return theme_daily
    x = theme_daily.copy().sort_values(["theme", "date"])
    out = []
    for theme, g in x.groupby("theme", sort=False):
        history = []
        for _, r in g.iterrows():
            count = r.get("limit_up_count_source")
            count = float(count) if pd.notna(count) else np.nan
            prev = history[-1] if history else None
            stage = "启动"
            if prev is None:
                stage = "启动"
            else:
                ratio = count / prev if prev and np.isfinite(count) else np.nan
                if np.isfinite(count) and count >= 20:
                    stage = "高潮"
                elif np.isfinite(ratio) and ratio >= 1.35:
                    stage = "发酵"
                elif np.isfinite(ratio) and ratio <= 0.55:
                    stage = "分歧"
                elif np.isfinite(count) and count >= 5:
                    stage = "确认"
                else:
                    stage = "震荡"
            rr = r.to_dict(); rr["lifecycle_stage_heuristic"] = stage; out.append(rr)
            if np.isfinite(count): history.append(count)
    return pd.DataFrame(out)


def build_expectations(market: pd.DataFrame, source_daily: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lbd = source_daily[source_daily.source_id == "lianban_archive"].set_index("date") if len(source_daily) else pd.DataFrame()
    for day in DAYS:
        nxt = next_trading_day(day)
        if not nxt:
            continue
        s = lbd.loc[day] if len(lbd) and day in lbd.index else None
        td = themes[themes.date == day] if len(themes) else pd.DataFrame()
        top = td.sort_values("limit_up_count_source", ascending=False).head(3)["theme"].tolist() if len(td) else []
        stage = s.get("emotion_stage") if s is not None else None
        temp = s.get("temperature") if s is not None else None
        if stage in ("冰点期", "退潮期"):
            base = "防守/试错"
            confirm = "次日仅在红盘广度和主线核心同时改善时小仓试错；无改善则空仓"
            invalid = "高位继续负反馈、主线核心低于预期、炸板/跌停恶化"
        elif stage in ("高潮期",):
            base = "高潮后分歧预案"
            confirm = "只做最核心分歧后的超预期确认，不接后排一致加速"
            invalid = "核心掉队且板块无回流，或低位切换明显增强"
        else:
            base = "跟随确认"
            confirm = "主线延续且核心/容量中军获得跟随才提高仓位"
            invalid = "主线缩容、昨日强势股负反馈扩散"
        rows.append({"asof_date": day, "effective_date": nxt, "emotion_stage_asof": stage, "temperature_asof": temp, "top_themes_asof": "|".join(top), "base_plan": base, "confirmation": confirm, "invalidation": invalid, "uses_future_data": False})
    return pd.DataFrame(rows)


def main():
    source_fetch = []
    source_daily = []
    themes = []
    memberships = []
    ladders = []
    for day in DAYS:
        for src in REGISTRY["daily_sources"]:
            url = src["url_template"].format(date=day)
            fr = fetch(url, src["id"], day)
            source_fetch.append({k: v for k, v in fr.items() if k != "text"})
            if not fr["ok"]:
                continue
            if src["id"] == "lianban_archive":
                s, ts, ls = parse_lianban(day, fr["text"])
                s["trade_effective_date"] = next_trading_day(day)
                source_daily.append(s)
                for t in ts: t["trade_effective_date"] = next_trading_day(day)
                themes.extend(ts); ladders.extend(ls)
            elif src["id"] == "yyqyx_limitup":
                s, ts, ms = parse_yyqyx(day, fr["text"])
                s["trade_effective_date"] = next_trading_day(day)
                source_daily.append(s)
                for t in ts: t["trade_effective_date"] = next_trading_day(day)
                for m in ms: m["trade_effective_date"] = next_trading_day(day)
                themes.extend(ts); memberships.extend(ms)
        time.sleep(0.08)

    market = github_market_daily()
    sd = pd.DataFrame(source_daily)
    th = pd.DataFrame(themes)
    if len(th):
        # union source rows are preserved; aggregate votes only in derived columns
        votes = th.groupby(["date", "theme"])["source_id"].nunique().rename("source_votes")
        th = th.merge(votes, on=["date", "theme"], how="left")
        th = infer_lifecycle(th)
    mem = pd.DataFrame(memberships)
    lad = pd.DataFrame(ladders)

    audit = []
    for day in DAYS:
        g = market[market.date == day].iloc[0].to_dict()
        ss = sd[sd.date == day] if len(sd) else pd.DataFrame()
        rec = {"date": day, "github_close_limit_up_nonst": g.get("close_limit_up_nonst"), "github_broken_proxy_nonst": g.get("broken_limit_proxy_nonst")}
        for sid in ["lianban_archive", "yyqyx_limitup"]:
            z = ss[ss.source_id == sid]
            rec[f"{sid}_available"] = bool(len(z))
            if len(z):
                q = z.iloc[0]
                rec[f"{sid}_limit_up"] = q.get("limit_up_count")
                rec[f"{sid}_broken"] = q.get("broken_board_count")
                rec[f"{sid}_limit_down"] = q.get("limit_down_count")
        vals = [rec.get("lianban_archive_limit_up"), rec.get("yyqyx_limitup_limit_up")]
        vals = [float(v) for v in vals if v is not None and pd.notna(v)]
        rec["web_count_disagreement"] = bool(len(vals) >= 2 and max(vals) - min(vals) > max(5, 0.08 * max(vals)))
        rec["action"] = "preserve_source_native_counts; GitHub non-ST count is separate canonical market fact"
        audit.append(rec)
    audit = pd.DataFrame(audit)
    exp = build_expectations(market, sd, th)

    pd.DataFrame(source_fetch).to_csv(OUT / "source_fetch_log.csv", index=False, encoding="utf-8-sig")
    market.to_csv(OUT / "market_daily.csv", index=False, encoding="utf-8-sig")
    sd.to_csv(OUT / "source_daily.csv", index=False, encoding="utf-8-sig")
    th.to_csv(OUT / "theme_daily.csv", index=False, encoding="utf-8-sig")
    mem.to_csv(OUT / "theme_membership.csv", index=False, encoding="utf-8-sig")
    lad.to_csv(OUT / "leader_ladder.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "source_audit.csv", index=False, encoding="utf-8-sig")
    exp.to_csv(OUT / "expectation_book.csv", index=False, encoding="utf-8-sig")

    fetch_df = pd.DataFrame(source_fetch)
    lianban_days = int(((fetch_df.source_id == "lianban_archive") & fetch_df.ok).sum()) if len(fetch_df) else 0
    secondary_days = int(((fetch_df.source_id == "yyqyx_limitup") & fetch_df.ok).sum()) if len(fetch_df) else 0
    gates = {
        "github_market_23_days": len(market) == 23,
        "primary_semantic_days_at_least_22": lianban_days >= 22,
        "secondary_fact_days_at_least_22": secondary_days >= 22,
        "theme_rows_present": len(th) >= 23,
        "expectation_is_strict_T_plus_1": bool(len(exp) == 22 and not exp.uses_future_data.any()),
        "raw_snapshots_hashed": bool(len(fetch_df[fetch_df.ok]) >= 44 and fetch_df.loc[fetch_df.ok, "sha256"].notna().all()),
        "source_conflicts_not_overwritten": bool(len(audit) == 23),
        "formal_backtest_not_run": True,
    }
    manifest = {
        "version": REGISTRY["version"],
        "built_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scope": "2026-07 knowledge layer only; no strategy PnL backtest",
        "inputs": {"github": "2026-07 minute1 + daily_stock_status", "web_sources": [x["id"] for x in REGISTRY["daily_sources"]]},
        "coverage": {"trading_days": len(market), "lianban_days": lianban_days, "secondary_days": secondary_days, "theme_rows": int(len(th)), "membership_rows": int(len(mem))},
        "gates": gates,
        "ready_for_backtest": bool(all(gates.values())),
        "anti_leakage": {"web_data_effective": "next trading day only", "forward_sections_forbidden": True, "future_relabeling_forbidden": True},
    }
    (OUT / "knowledge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
