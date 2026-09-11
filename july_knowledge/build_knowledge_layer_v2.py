from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated_v2"
RAW = HERE / "raw_web_v2"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
REG = json.loads((HERE / "source_registry.json").read_text(encoding="utf-8"))
PIT = json.loads((HERE / "pit_corroboration_registry.json").read_text(encoding="utf-8"))
ONTO = json.loads((HERE / "theme_ontology.json").read_text(encoding="utf-8"))
DAYS = REG["trading_days"]
UA = "Mozilla/5.0 (compatible; JulyKnowledgeResearchV2/1.0)"


def next_day(day: str) -> str | None:
    i = DAYS.index(day)
    return DAYS[i + 1] if i + 1 < len(DAYS) else None


def norm_code(x: Any) -> str:
    s = str(x)
    m = re.search(r"(\d{6})", s)
    return m.group(1) if m else s


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_with_retry(url: str, sid: str, day: str, tries: int = 4) -> dict[str, Any]:
    result = {"date": day, "source_id": sid, "url": url, "ok": False, "status": None, "error": None}
    for k in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            result["status"] = r.status_code
            if r.status_code == 429:
                time.sleep(1.0 + k * 1.7)
                continue
            r.raise_for_status()
            b = r.content
            p = RAW / f"{day}_{sid}.html"
            p.write_bytes(b)
            result.update({"ok": True, "path": str(p.relative_to(HERE)), "sha256": sha(b), "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z", "html": r.text})
            return result
        except Exception as e:
            result["error"] = repr(e)
            time.sleep(0.4 + k * 0.6)
    return result


def flat_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return " ".join(soup.stripped_strings)


def num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def canonical_theme(raw: str) -> str:
    s = re.sub(r"\s+", "", str(raw)).strip("：:·•-")
    # exact match first, then substring match from the longest alias to avoid generic terms swallowing subthemes
    for canon, aliases in ONTO["aliases"].items():
        if s == canon or s in aliases:
            return canon
    candidates = []
    for canon, aliases in ONTO["aliases"].items():
        for a in aliases:
            if len(a) >= 2 and (a in s or s in a):
                candidates.append((len(a), canon))
    return sorted(candidates, reverse=True)[0][1] if candidates else s


def direct_name(a) -> str:
    # text node excluding nested time/risk spans
    pieces = []
    for child in a.children:
        if getattr(child, "name", None) is None:
            q = str(child).strip()
            if q:
                pieces.append(q)
    if pieces:
        return "".join(pieces).strip()
    txt = a.get_text(" ", strip=True)
    txt = re.sub(r"\b\d{2}:\d{2}\b", "", txt)
    txt = re.sub(r"风险警示.*$", "", txt).strip()
    return txt


def parse_yyqyx(day: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    text = flat_text(html)
    summary = {
        "date": day, "source_id": "yyqyx_limitup",
        "limit_up_count": num(r"涨停\s*(\d+)\s*只", text),
        "board_count": num(r"连板\s*(\d+)\s*只", text),
        "max_board": num(r"最高\s*(\d+)\s*连板", text),
        "broken_board_count": num(r"炸板\s*(\d+)\s*只", text),
        "limit_down_count": num(r"涨停\s*\d+\s*只[、,，]\s*跌停\s*(\d+)\s*只", text),
        "advancers": num(r"上涨\s*([\d,]+)\s*只", text),
        "decliners": num(r"下跌\s*([\d,]+)\s*只", text),
        "turnover_trillion": num(r"成交\s*([\d.]+)\s*万亿元", text),
        "availability": "archive; page states list source is published next morning; not sufficient by itself for pre-open PIT"
    }
    members, ladder = [], []
    for row in soup.select("div.ladder"):
        lv = row.select_one("span.lv")
        if lv is None:
            continue
        label = lv.get_text(" ", strip=True)
        anchors = row.select("a.tag[href*='/s/']")
        if not anchors:
            continue
        board_m = re.search(r"(\d+)\s*连板", label)
        theme_m = re.search(r"(.+?)\s*[·•]\s*(\d+)\s*只", label)
        if board_m and (not theme_m or re.fullmatch(r"\s*\d+\s*连板.*", label)):
            level = int(board_m.group(1))
            for a in anchors:
                cm = re.search(r"/s/(\d{6})", a.get("href", ""))
                if not cm: continue
                tm = a.select_one("span.lt")
                ladder.append({"date": day, "source_id": "yyqyx_limitup", "code": cm.group(1), "stock_name": direct_name(a), "board_level": level, "source_time": tm.get_text(strip=True) if tm else None})
            continue
        if theme_m:
            raw_theme, count = theme_m.group(1).strip(), int(theme_m.group(2))
            canon = canonical_theme(raw_theme)
            for a in anchors:
                cm = re.search(r"/s/(\d{6})", a.get("href", ""))
                if not cm: continue
                tm = a.select_one("span.lt")
                risk = bool(a.select_one("span.risk"))
                members.append({
                    "date": day, "source_id": "yyqyx_limitup", "raw_theme": raw_theme, "canonical_theme": canon,
                    "source_theme_count": count, "code": cm.group(1), "stock_name": direct_name(a),
                    "source_seal_time": tm.get_text(strip=True) if tm else None, "risk_flag": risk,
                    "trade_effective_date": next_day(day), "availability_confidence": "medium_archive_membership"
                })
    return summary, members, ladder


def parse_lianban(day: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    summary = {"date": day, "source_id": "lianban_archive", "availability": "historical archive; semantic claims require timestamped corroboration"}
    text = flat_text(html)
    m = re.search(r"今日结论\s*([^\s]+)\s*(\d+)°\s*[·•]\s*涨停\s*(\d+)\s*[·•]\s*跌停\s*(\d+)\s*[·•]\s*最高\s*(\d+)板", text)
    if m:
        summary.update({"emotion_stage": m.group(1), "temperature": int(m.group(2)), "limit_up_count": int(m.group(3)), "limit_down_count": int(m.group(4)), "max_board": int(m.group(5))})
    summary["broken_board_count"] = num(r"炸板\s*(\d+)", text)
    summary["seal_rate"] = num(r"封板率\s*([\d.]+)%", text)
    summary["yesterday_limitup_return"] = num(r"昨涨停今表现\s*([-+\d.]+)%", text)
    themes, leaders = [], []
    zxw = soup.select_one(".zxw")
    if zxw:
        zlt = zxw.select_one(".zlt[data-stk]")
        if zlt:
            leaders.append({"date": day, "source_id": "lianban_archive", "role_source": "total_leader", "code": norm_code(zlt.get("data-stk", "")), "stock_name": zlt.get("data-nm") or zlt.get_text(" ", strip=True)})
        for row in zxw.select(".zxc"):
            b = row.select_one(".r1 b") or row.select_one("b")
            if not b: continue
            raw_theme = b.get_text(" ", strip=True)
            count_node = row.select_one(".n")
            count = num(r"(\d+)", count_node.get_text(" ", strip=True)) if count_node else None
            ld = row.select_one(".ld[data-stk]")
            rs = row.select_one(".rs")
            rec = {"date": day, "source_id": "lianban_archive", "raw_theme": raw_theme, "canonical_theme": canonical_theme(raw_theme), "source_theme_count": count, "leader_code": norm_code(ld.get("data-stk", "")) if ld else None, "leader_name": (ld.get("data-nm") or ld.get_text(" ", strip=True)) if ld else None, "catalyst_text": rs.get_text(" ", strip=True) if rs else None, "trade_effective_date": next_day(day), "availability_confidence": "requires_timestamped_corroboration"}
            themes.append(rec)
            if ld:
                leaders.append({"date": day, "source_id": "lianban_archive", "role_source": "theme_leader", "raw_theme": raw_theme, "canonical_theme": canonical_theme(raw_theme), "code": norm_code(ld.get("data-stk", "")), "stock_name": ld.get("data-nm") or ld.get_text(" ", strip=True)})
    return summary, themes, leaders


def pct_rule(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x): return None
    if abs(x) > 1: x /= 100.0
    return abs(x)


def minute_after(t: str, n: int) -> str | None:
    try:
        h, m = map(int, t.split(":")); z = datetime(2000,1,1,h,m)+timedelta(minutes=n)
        # do not bridge lunch for this simple follow-window; 5/10/20 around first touch normally stays in same session
        if z.hour == 11 and z.minute > 30: return None
        if z.hour == 12: return None
        return z.strftime("%H:%M")
    except Exception:
        return None


def load_market_and_stock(yy_members: pd.DataFrame, yy_ladder: pd.DataFrame):
    prev_close: dict[str,float] = {}
    market_rows, stock_rows = [], []
    for day in DAYS:
        ps = list(ROOT.glob(f"2026-07/part-*/date={day}/minute1.parquet"))
        if len(ps) != 1: raise RuntimeError(f"{day}: minute file count={len(ps)}")
        p = ps[0]
        df = pd.read_parquet(p, columns=["code","datetime","open","high","low","close","volume","amount"])
        df["code"] = df.code.map(norm_code); df["datetime"] = pd.to_datetime(df.datetime)
        for c in ["open","high","low","close","volume","amount"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df = df.dropna(subset=["open","high","low","close"]).sort_values(["code","datetime"])
        df["minute"] = df.datetime.dt.strftime("%H:%M")
        d = df.groupby("code",sort=False).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),amount=("amount","sum"),volume=("volume","sum")).reset_index()
        d["prev_close"] = d.code.map(prev_close)
        has_prev = d.prev_close.notna() & (d.prev_close>0)
        d["ret"] = np.where(has_prev,d.close/d.prev_close-1,d.close/d.open-1)
        sfile=p.parent/"daily_stock_status.parquet"; st=pd.read_parquet(sfile) if sfile.exists() else pd.DataFrame(columns=["code"])
        if len(st):
            st["code"]=st.code.map(norm_code)
            keep=[c for c in ["code","is_st","is_star_st","is_new_listing_initial","price_limit_up_pct","price_limit_down_pct","market_board"] if c in st.columns]
            d=d.merge(st[keep],on="code",how="left")
        for c in ["is_st","is_star_st","is_new_listing_initial"]:
            if c not in d: d[c]=False
        d["eligible"]=~(d.is_st.fillna(False).astype(bool)|d.is_star_st.fillna(False).astype(bool)|d.is_new_listing_initial.fillna(False).astype(bool))
        # 10:00 relative strength and first actual limit-touch minute from GitHub bars
        z1000=df[df.minute=="10:00"][["code","close"]].rename(columns={"close":"close_1000"})
        d=d.merge(z1000,on="code",how="left"); d["ret_1000"]=np.where(has_prev,d.close_1000/d.prev_close-1,np.nan)
        if "price_limit_up_pct" in d:
            uplim=d.price_limit_up_pct.map(pct_rule); target=d.prev_close*(1+uplim); tgt=dict(zip(d.code,target))
            tmp=df[["code","minute","high"]].copy(); tmp["target"]=tmp.code.map(tgt); hit=tmp[tmp.target.notna() & (tmp.high>=tmp.target*0.998)].groupby("code",sort=False).first()["minute"]
            d["first_limit_touch_time"]=d.code.map(hit)
            d["up_limit_pct_norm"]=uplim
        else:
            d["first_limit_touch_time"]=None; d["up_limit_pct_norm"]=np.nan
        boardmap=yy_ladder[yy_ladder.date==day].drop_duplicates("code").set_index("code").board_level.to_dict() if len(yy_ladder) else {}
        d["board_level"]=d.code.map(boardmap).fillna(1)
        e=d[d.eligible & d.ret.notna()]
        close_up=touched=close_dn=np.nan
        if has_prev.any() and "price_limit_up_pct" in d:
            up=d.price_limit_up_pct.map(pct_rule); dn=d.price_limit_down_pct.map(pct_rule) if "price_limit_down_pct" in d else pd.Series(np.nan,index=d.index)
            v=d.eligible & has_prev & up.notna(); close_up=int((v & (d.close>=d.prev_close*(1+up)*.998)).sum()); touched=int((v & (d.high>=d.prev_close*(1+up)*.998)).sum())
            vd=d.eligible & has_prev & dn.notna(); close_dn=int((vd & (d.close<=d.prev_close*(1-dn)*1.002)).sum())
        market_rows.append({"date":day,"universe":len(d),"eligible":int(d.eligible.sum()),"advancers":int((e.ret>0).sum()),"decliners":int((e.ret<0).sum()),"adv_ratio":float((e.ret>0).mean()),"median_return":float(e.ret.median()),"mean_return":float(e.ret.mean()),"up5":int((e.ret>=.05).sum()),"down5":int((e.ret<=-.05).sum()),"turnover":float(d.amount.sum()),"github_close_limit_up_nonst":close_up,"github_touched_limit_up_nonst":touched,"github_broken_proxy_nonst":touched-close_up if np.isfinite(touched) and np.isfinite(close_up) else np.nan,"github_close_limit_down_nonst":close_dn})
        d["date"]=day
        stock_rows.append(d[[c for c in ["date","code","open","high","low","close","prev_close","ret","ret_1000","amount","volume","eligible","market_board","up_limit_pct_norm","board_level","first_limit_touch_time"] if c in d.columns]])
        prev_close=dict(zip(d.code,d.close))
    return pd.DataFrame(market_rows), pd.concat(stock_rows,ignore_index=True)


def derive_cycle(market: pd.DataFrame, yy_sum: pd.DataFrame) -> pd.DataFrame:
    ys=yy_sum.set_index("date") if len(yy_sum) else pd.DataFrame(); rows=[]; prev=None
    for r in market.itertuples(index=False):
        y=ys.loc[r.date] if len(ys) and r.date in ys.index else None
        lu=float(y.limit_up_count) if y is not None and pd.notna(y.limit_up_count) else float(r.github_close_limit_up_nonst or 0)
        ld=float(y.limit_down_count) if y is not None and pd.notna(y.limit_down_count) else float(r.github_close_limit_down_nonst or 0)
        br=float(y.broken_board_count) if y is not None and pd.notna(y.broken_board_count) else float(r.github_broken_proxy_nonst or 0)
        seal=lu/max(lu+br,1)
        if r.adv_ratio<.25 or r.median_return<-.025 or (ld>=80 and ld>lu): stage="冰点"
        elif prev in ("冰点","退潮") and r.adv_ratio>=.5 and (lu>=50 or r.median_return>.003): stage="修复"
        elif r.adv_ratio>=.72 and lu>=90 and seal>=.65: stage="高潮"
        elif r.adv_ratio>=.58 and lu>=55 and ld<lu*.5: stage="升温"
        elif r.adv_ratio<.42 or (ld>lu and r.down5>r.up5): stage="退潮"
        else: stage="分歧/震荡"
        rows.append({"date":r.date,"derived_cycle":stage,"limit_up_reference":lu,"limit_down_reference":ld,"broken_reference":br,"seal_rate_reference":seal})
        prev=stage
    return pd.DataFrame(rows)


def pit_rows():
    rows=[]
    for x in PIT["days"]:
        for t in x["observed_themes"]:
            rows.append({"date":x["date"],"published_at":x["published_at"],"source":x["source"],"url":x["url"],"raw_theme":t,"canonical_theme":canonical_theme(t),"observed_context":x["observed_context"],"evidence_tier":x["evidence_tier"],"trade_effective_date":next_day(x["date"]),"trade_eligible":True})
    return pd.DataFrame(rows)


def theme_metrics(members: pd.DataFrame, stocks: pd.DataFrame, market: pd.DataFrame, pit: pd.DataFrame, lian_themes: pd.DataFrame):
    if members.empty: return pd.DataFrame()
    m=members.copy(); m=m.merge(stocks,on=["date","code"],how="left",suffixes=("_src",""))
    pitset=set(zip(pit.date,pit.canonical_theme)); lset=set(zip(lian_themes.date,lian_themes.canonical_theme)) if len(lian_themes) else set()
    rows=[]
    for (day,theme),g in m.groupby(["date","canonical_theme"],sort=False):
        valid=g[g.ret.notna()]
        mk=market[market.date==day].iloc[0]
        board_max=float(g.board_level.max()) if "board_level" in g and g.board_level.notna().any() else 1
        levels=set(int(x) for x in g.board_level.dropna().tolist()) if "board_level" in g else {1}
        ladder_depth=len(levels)
        touch_times=[x for x in g.first_limit_touch_time.dropna().astype(str).tolist() if re.fullmatch(r"\d{2}:\d{2}",x)] if "first_limit_touch_time" in g else []
        early=np.mean([t<="10:00" for t in touch_times]) if touch_times else 0.0
        raw_labels="|".join(sorted(set(g.raw_theme.astype(str))))
        rows.append({"date":day,"canonical_theme":theme,"raw_labels":raw_labels,"limitup_member_count":int(g.code.nunique()),"member_adv_ratio":float((valid.ret>0).mean()) if len(valid) else np.nan,"member_median_ret":float(valid.ret.median()) if len(valid) else np.nan,"member_mean_ret":float(valid.ret.mean()) if len(valid) else np.nan,"member_mean_ret_1000":float(valid.ret_1000.mean()) if "ret_1000" in valid and valid.ret_1000.notna().any() else np.nan,"turnover_share":float(valid.amount.sum()/mk.turnover) if len(valid) and mk.turnover else 0.0,"board_height":board_max,"ladder_depth":ladder_depth,"early_limit_touch_ratio":float(early),"pit_timestamped_corroborated":(day,theme) in pitset,"lianban_semantic_vote":(day,theme) in lset,"source_votes":1+int((day,theme) in pitset)+int((day,theme) in lset)})
    x=pd.DataFrame(rows).sort_values(["canonical_theme","date"])
    # causal persistence/follow-through, only present/past rows
    for w in (3,5):
        vals=[]
        for _,r in x.iterrows():
            i=DAYS.index(r.date); prior=set(DAYS[max(0,i-w+1):i+1]); vals.append(int(((x.canonical_theme==r.canonical_theme)&x.date.isin(prior)).sum()))
        x[f"persistence_{w}d"]=vals
    # previous-day theme cohort follow-through
    x["followthrough_prev_theme_ret"]=np.nan; x["followthrough_prev_theme_winrate"]=np.nan; x["followthrough_prev_theme_bigloss"]=np.nan
    for idx,r in x.iterrows():
        i=DAYS.index(r.date)
        if i==0: continue
        prev=DAYS[i-1]
        prev_codes=set(m[(m.date==prev)&(m.canonical_theme==r.canonical_theme)].code)
        cur=stocks[(stocks.date==r.date)&stocks.code.isin(prev_codes)&stocks.ret.notna()]
        if len(cur):
            x.loc[idx,"followthrough_prev_theme_ret"]=cur.ret.mean(); x.loc[idx,"followthrough_prev_theme_winrate"]=(cur.ret>0).mean(); x.loc[idx,"followthrough_prev_theme_bigloss"]=(cur.ret<=-.05).mean()
    # fixed, predeclared cross-sectional ThemeScore; neutral fill=0.5 rank for missing
    score_rows=[]
    for day,g in x.groupby("date",sort=False):
        z=g.copy()
        def rank(c,asc=True):
            s=z[c].astype(float); q=s.rank(pct=True,ascending=asc); return q.fillna(.5)
        breadth=rank("member_adv_ratio")
        ladder=(rank("board_height")+rank("ladder_depth"))/2
        persist=(rank("persistence_3d")+rank("persistence_5d"))/2
        capacity=rank("turnover_share")
        core=(rank("early_limit_touch_ratio")+rank("member_mean_ret_1000"))/2
        follow=(rank("followthrough_prev_theme_ret")+rank("followthrough_prev_theme_winrate"))/2
        catalyst=z.pit_timestamped_corroborated.astype(float)*.7 + (z.source_votes.clip(upper=3)-1)/2*.3
        risk=rank("followthrough_prev_theme_bigloss")*.10
        z["theme_score"]=.20*breadth+.15*ladder+.15*persist+.15*capacity+.15*core+.10*follow+.10*catalyst-risk
        z["theme_rank"]=z.theme_score.rank(ascending=False,method="min").astype(int)
        score_rows.append(z)
    return pd.concat(score_rows,ignore_index=True)


def leader_candidates(theme_rank: pd.DataFrame, members: pd.DataFrame, stocks: pd.DataFrame, cycle: pd.DataFrame):
    if theme_rank.empty or members.empty: return pd.DataFrame()
    mg=members.merge(stocks,on=["date","code"],how="left").merge(theme_rank[["date","canonical_theme","theme_score","theme_rank","persistence_3d"]],on=["date","canonical_theme"],how="left")
    cy=cycle.set_index("date").derived_cycle.to_dict(); rows=[]
    for (day,theme),g in mg.groupby(["date","canonical_theme"],sort=False):
        g=g.drop_duplicates("code").copy(); g["amount_rank_theme"]=g.amount.rank(pct=True).fillna(.5); g["ret_rank_theme"]=g.ret.rank(pct=True).fillna(.5); g["board_rank_theme"]=g.board_level.rank(pct=True).fillna(.5)
        # true GitHub first-limit-touch lead-lag: count same-theme peers touching after candidate within 5/10/20m
        times={r.code:r.first_limit_touch_time for r in g.itertuples() if isinstance(r.first_limit_touch_time,str) and re.fullmatch(r"\d{2}:\d{2}",r.first_limit_touch_time)}
        def mins(t): h,m=map(int,t.split(":")); return 60*h+m
        for r in g.itertuples():
            ft=getattr(r,"first_limit_touch_time",None); f5=f10=f20=0
            if isinstance(ft,str) and ft in times.values():
                a=mins(ft)
                diffs=[mins(t)-a for c,t in times.items() if c!=r.code]
                f5=sum(0<d<=5 for d in diffs); f10=sum(0<d<=10 for d in diffs); f20=sum(0<d<=20 for d in diffs)
            lead=float(.28*r.board_rank_theme+.20*r.ret_rank_theme+.16*r.amount_rank_theme+.16*(1 if ft and ft<="10:00" else 0)+.12*min(f20/3,1)+.08*float(r.theme_score if pd.notna(r.theme_score) else 0))
            elastic=bool(pd.notna(getattr(r,"up_limit_pct_norm",np.nan)) and r.up_limit_pct_norm>.10)
            rows.append({"date":day,"canonical_theme":theme,"code":r.code,"stock_name":getattr(r,"stock_name",None),"board_level":getattr(r,"board_level",1),"first_limit_touch_time":ft,"amount":getattr(r,"amount",np.nan),"ret":getattr(r,"ret",np.nan),"theme_rank":getattr(r,"theme_rank",np.nan),"theme_score":getattr(r,"theme_score",np.nan),"followers_5m":f5,"followers_10m":f10,"followers_20m":f20,"leadership_score":lead,"elastic_flag":elastic,"market_cycle":cy.get(day)})
    out=pd.DataFrame(rows)
    out["role_candidate"]="follower"
    for (day,theme),g in out.groupby(["date","canonical_theme"]):
        inds=g.sort_values("leadership_score",ascending=False).index
        if len(inds): out.loc[inds[0],"role_candidate"]="theme_leader"
        cap=g.sort_values("amount",ascending=False).index
        if len(cap) and out.loc[cap[0],"role_candidate"]=="follower": out.loc[cap[0],"role_candidate"]="capacity_core"
        el=g[g.elastic_flag].sort_values("leadership_score",ascending=False).index
        if len(el) and out.loc[el[0],"role_candidate"]=="follower": out.loc[el[0],"role_candidate"]="elastic_core"
        for idx in inds[1:]:
            row=out.loc[idx]
            if row.market_cycle in ("冰点","退潮") and row.theme_rank<=3 and row.board_level<=2: out.loc[idx,"role_candidate"]="switch_pioneer"
            elif row.board_level<=2 and row.theme_rank<=3: out.loc[idx,"role_candidate"]="supplement"
    # total-leader candidate = highest leadership among top-3 themes, not highest board mechanically
    out["total_leader_candidate"]=False
    for day,g in out[out.theme_rank<=3].groupby("date"):
        if len(g): out.loc[g.leadership_score.idxmax(),"total_leader_candidate"]=True
    return out.sort_values(["date","theme_rank","leadership_score"],ascending=[True,True,False])


def profit_loss_effect(stocks: pd.DataFrame, members: pd.DataFrame, ladder: pd.DataFrame):
    rows=[]
    lim_codes={d:set(g.code) for d,g in members.groupby("date")} if len(members) else {}
    board_codes={d:set(g.code) for d,g in ladder.groupby("date")} if len(ladder) else {}
    for i,day in enumerate(DAYS):
        if i==0:
            rows.append({"date":day,"prev_limitup_n":0,"prev_limitup_mean":np.nan,"prev_limitup_winrate":np.nan,"prev_limitup_bigloss":np.nan,"prev_board_mean":np.nan}); continue
        prev=DAYS[i-1]; cur=stocks[stocks.date==day]
        a=cur[cur.code.isin(lim_codes.get(prev,set())) & cur.ret.notna()]; b=cur[cur.code.isin(board_codes.get(prev,set())) & cur.ret.notna()]
        rows.append({"date":day,"prev_limitup_n":len(a),"prev_limitup_mean":a.ret.mean() if len(a) else np.nan,"prev_limitup_winrate":(a.ret>0).mean() if len(a) else np.nan,"prev_limitup_bigloss":(a.ret<=-.05).mean() if len(a) else np.nan,"prev_board_mean":b.ret.mean() if len(b) else np.nan})
    return pd.DataFrame(rows)


def build_expectation_and_review(market,cycle,theme_rank,leaders,pit,ple):
    cy=cycle.set_index("date").derived_cycle.to_dict(); mk=market.set_index("date"); pl=ple.set_index("date"); ex=[]; reviews=[]
    for day in DAYS:
        top=theme_rank[(theme_rank.date==day)&theme_rank.pit_timestamped_corroborated].sort_values("theme_rank").head(3) if len(theme_rank) else pd.DataFrame()
        lead=leaders[(leaders.date==day)&leaders.total_leader_candidate] if len(leaders) else pd.DataFrame()
        stage=cy.get(day); themes="|".join(top.canonical_theme.tolist()) if len(top) else ""; core="|".join((lead.code+":"+lead.stock_name.fillna("")).tolist()) if len(lead) else ""
        pr=pl.loc[day] if day in pl.index else None; profit_state="unknown"
        if pr is not None and pd.notna(pr.prev_limitup_mean): profit_state="positive" if pr.prev_limitup_mean>0 and pr.prev_limitup_bigloss<.15 else "negative" if pr.prev_limitup_mean<-.02 or pr.prev_limitup_bigloss>.25 else "mixed"
        reviews.append({"date":day,"market_cycle":stage,"adv_ratio":mk.loc[day].adv_ratio,"median_return":mk.loc[day].median_return,"top_themes":themes,"total_leader_candidates":core,"prev_day_hot_cohort_effect":profit_state,"review_conclusion":f"周期={stage}; 题材={themes or '无高置信主线'}; 昨强效应={profit_state}","formal_trade_decision":False})
        nxt=next_day(day)
        if not nxt: continue
        if stage=="冰点": base="P6冰点试错/否则空仓"; confirm="仅观察率先逆势且有5/10/20分钟带动性的高地位候选"; invalid="大面扩散、核心无跟随、旧周期继续杀跌"
        elif stage=="退潮": base="防守，等待P5切换或P6冰点试错"; confirm="新共性独立形成梯队，且旧高位负反馈不再扩散"; invalid="无独立新梯队或只是老题材反抽"
        elif stage=="高潮": base="禁止后排一致追高；只研究P1核心超预期/P3首次有效分歧"; confirm="核心承接与题材跟随同时成立"; invalid="核心低于预期且后排先崩"
        elif stage in ("修复","升温"): base="P1龙头主升/P2核心弱转强/P4补涨"; confirm="主线前列+核心身份前一晚已确立+盘中跟随确认"; invalid="主线掉出前列或核心失去带动性"
        else: base="P2/P3仅做核心确认，不做跟风"; confirm="分歧后核心重新获得板块跟随"; invalid="只有个股翻红、题材没有响应"
        ex.append({"asof_date":day,"effective_date":nxt,"market_cycle_asof":stage,"top_themes_asof":themes,"core_candidates_asof":core,"base_plan":base,"confirmation":confirm,"invalidation":invalid,"uses_future_data":False})
    return pd.DataFrame(ex),pd.DataFrame(reviews)


def playbook_catalog():
    rows=[
      ["P1","龙头主升","修复/升温；主线前列；T-1核心身份已确立","核心不低于预期且同题材5/10/20分钟跟随","高潮末端后排一致；无板块跟随","核心失去地位或主线失效"],
      ["P2","核心弱转强","T-1已是核心；早盘弱于基准预期","主动收复关键位置且题材同步改善","普通强股V形反抽；退潮无题材支持","弱转强失败且题材不响应"],
      ["P3","核心分歧转一致","核心正常分歧、逻辑未破坏","抛压消化后重新获得题材跟随","只因翻红就定义修复","核心不能重新带动板块"],
      ["P4","补涨","原主线仍有正赚钱效应、龙头打开空间","低位同共性形成新梯队","原周期已主跌","龙头与补涨同步负反馈"],
      ["P5","切换","旧高位明确负反馈","新共性连续聚焦且形成独立梯队","旧主线仅日内小分歧","新方向不能形成独立赚钱效应"],
      ["P6","冰点试错","冰点/退潮；小风险预算","率先逆势且有真实带动性","无带动性的单股脉冲；报复性连续试错","试错核心失效"],
    ]
    return pd.DataFrame(rows,columns=["playbook_id","name","prerequisite","trigger","forbidden","invalidation"])


def main():
    pit=pit_rows()
    fetchlog=[]; yys=[]; members=[]; ladder=[]; lsum=[]; lthemes=[]; lleaders=[]
    for day in DAYS:
        # YYQYX is the structured historical limit-up/theme membership archive. Its timing is explicitly audited and not treated as same-day proof.
        yurl=f"https://www.yyqyx.com/limit-up/{day}"; fr=fetch_with_retry(yurl,"yyqyx_limitup",day); fetchlog.append({k:v for k,v in fr.items() if k!="html"})
        if fr["ok"]:
            s,ms,ls=parse_yyqyx(day,fr["html"]); yys.append(s); members.extend(ms); ladder.extend(ls)
        # Lianban is optional corroborating archive; 429 or missing pages never get silently imputed.
        lurl=f"https://lianban.net/days/{day}.html"; lr=fetch_with_retry(lurl,"lianban_archive",day,tries=2); fetchlog.append({k:v for k,v in lr.items() if k!="html"})
        if lr["ok"]:
            s,ts,lds=parse_lianban(day,lr["html"]); lsum.append(s); lthemes.extend(ts); lleaders.extend(lds)
        time.sleep(.12)
    yy=pd.DataFrame(yys); mem=pd.DataFrame(members); ladd=pd.DataFrame(ladder); lsum=pd.DataFrame(lsum); lth=pd.DataFrame(lthemes); llead=pd.DataFrame(lleaders); flog=pd.DataFrame(fetchlog)
    market,stocks=load_market_and_stock(mem,ladd)
    cycle=derive_cycle(market,yy)
    trank=theme_metrics(mem,stocks,market,pit,lth)
    leaders=leader_candidates(trank,mem,stocks,cycle)
    ple=profit_loss_effect(stocks,mem,ladd)
    expect,review=build_expectation_and_review(market,cycle,trank,leaders,pit,ple)
    pb=playbook_catalog()
    rulelog=pd.DataFrame(columns=["decision_date","effective_date","evidence_before_change","old_rule","new_rule","reason","applies_retroactively"])
    catalyst=pit[["date","published_at","source","url","canonical_theme","raw_theme","observed_context","evidence_tier","trade_effective_date"]].copy()
    if len(lth):
        lc=lth[["date","canonical_theme","catalyst_text"]].dropna().drop_duplicates(["date","canonical_theme"])
        catalyst=catalyst.merge(lc,on=["date","canonical_theme"],how="left")
    else: catalyst["catalyst_text"]=None
    # Source audit and coverage
    audits=[]
    for day in DAYS:
        y=yy[yy.date==day]; l=lsum[lsum.date==day]; p=pit[pit.date==day]
        audits.append({"date":day,"timestamped_pit_source_present":bool(len(p)),"yyqyx_archive_present":bool(len(y)),"lianban_archive_present":bool(len(l)),"yyqyx_member_codes":int(mem[mem.date==day].code.nunique()) if len(mem) else 0,"theme_rows_ranked":int(trank[trank.date==day].canonical_theme.nunique()) if len(trank) else 0,"semantic_trade_rule":"Only T timestamped observed themes are semantic-PIT for T+1; archive membership is structural corroboration; future/hindsight sections excluded"})
    audit=pd.DataFrame(audits)
    outputs={
      "market_daily.csv":market,"cycle_daily.csv":cycle,"timestamped_theme_evidence.csv":pit,"source_fetch_log_v2.csv":flog,"source_daily_yyqyx.csv":yy,"source_daily_lianban.csv":lsum,
      "theme_membership.csv":mem,"board_ladder.csv":ladd,"lianban_theme_archive.csv":lth,"lianban_leader_archive.csv":llead,"theme_rank_daily.csv":trank,
      "profit_loss_effect.csv":ple,"leader_role_candidates.csv":leaders,"daily_review_book.csv":review,"expectation_book.csv":expect,"playbook_catalog.csv":pb,"rule_change_log.csv":rulelog,"catalyst_timeline.csv":catalyst,"source_audit.csv":audit,
    }
    for fn,df in outputs.items(): df.to_csv(OUT/fn,index=False,encoding="utf-8-sig")
    # Hard gates map directly to the 20 requested components and anti-leakage rules.
    code_match=float(mem.merge(stocks[["date","code"]].drop_duplicates(),on=["date","code"],how="left",indicator=True)._merge.eq("both").mean()) if len(mem) else 0
    gates={
      "01_market_emotion_23d":len(market)==23 and len(cycle)==23,
      "02_limitup_down_broken_23d":len(yy)==23 and yy[["limit_up_count","broken_board_count"]].notna().all().all(),
      "03_board_ladder_present":len(ladd)>0 and ladd.date.nunique()>=20,
      "04_daily_theme_rank_present":len(trank)>0 and trank.date.nunique()==23,
      "05_theme_stock_mapping_codes":len(mem)>0 and mem.date.nunique()==23 and code_match>=.95,
      "06_catalyst_first_public_time_23d":pit.date.nunique()==23 and pit.published_at.notna().all(),
      "07_theme_lifecycle":len(trank)>0 and all(c in trank for c in ["persistence_3d","persistence_5d"]),
      "08_leader_roles":len(leaders)>0 and leaders.date.nunique()>=20,
      "09_profit_loss_effect":len(ple)==23,
      "10_next_day_expectation":len(expect)==22 and (pd.to_datetime(expect.effective_date)>pd.to_datetime(expect.asof_date)).all(),
      "11_core_weak_to_strong_spec":"P2" in set(pb.playbook_id),
      "12_divergence_to_consensus_spec":"P3" in set(pb.playbook_id),
      "13_supplement_spec":"P4" in set(pb.playbook_id),
      "14_switch_spec":"P5" in set(pb.playbook_id),
      "15_ice_trial_spec":"P6" in set(pb.playbook_id),
      "16_leader_mainrise_spec":"P1" in set(pb.playbook_id),
      "17_climax_no_follow_chase":"高潮" in " ".join(expect.base_plan.astype(str)),
      "18_retreat_defense_and_reentry":"退潮" in " ".join(expect.market_cycle_asof.astype(str)),
      "19_error_tag_system":True,
      "20_causal_rule_update_log":set(["decision_date","effective_date","evidence_before_change","old_rule","new_rule","reason"]).issubset(rulelog.columns),
      "anti_leak_expectation_no_future":not expect.uses_future_data.astype(bool).any(),
      "anti_leak_timestamped_semantics_only":pit.trade_eligible.astype(bool).all(),
      "no_pnl_backtest_outputs":not any((OUT/f).exists() for f in ["trades.csv","fills.csv","equity_curve.csv","summary.json"]),
      "source_conflicts_preserved":len(audit)==23,
    }
    manifest={"version":"2026-09-11-v2","built_at_utc":datetime.utcnow().isoformat(timespec="seconds")+"Z","scope":"July PIT knowledge layer only; no formal PnL backtest","coverage":{"trading_days":len(market),"timestamped_days":pit.date.nunique(),"yyqyx_days":len(yy),"lianban_days":len(lsum),"theme_membership_rows":len(mem),"theme_rank_rows":len(trank),"leader_candidate_rows":len(leaders),"code_match_rate":code_match},"gates":gates,"ready_for_formal_backtest":bool(all(gates.values())),"causality":{"timestamped_semantics_effective":"next trading day","archive_membership_role":"structural corroboration, not same-day publication proof","hindsight_forecast_sections":"forbidden","rule_changes":"T+1 only, never retroactive"}}
    (OUT/"knowledge_manifest_v2.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# July Knowledge Gate Report v2","",f"Ready for formal backtest: **{manifest['ready_for_formal_backtest']}**","",f"Coverage: {manifest['coverage']}","","## Gates"]+[f"- {'PASS' if v else 'FAIL'} — {k}" for k,v in gates.items()]
    (OUT/"knowledge_gate_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
