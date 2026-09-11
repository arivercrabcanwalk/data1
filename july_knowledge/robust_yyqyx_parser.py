from __future__ import annotations

import re
from typing import Any
from bs4 import BeautifulSoup, NavigableString
import build_knowledge_layer_v2 as b

STOCK_HREF = re.compile(r"/s/(\d{6})(?:\.[A-Z]{2})?")
THEME_HEADER = re.compile(r"^\s*(.+?)\s*[·•]\s*(\d+)\s*只\s*$")
BOARD_HEADER = re.compile(r"^\s*(\d+)\s*连板\s*$")


def _stock_anchors(node):
    out=[]
    for a in node.find_all("a", href=True):
        m=STOCK_HREF.search(a.get("href", ""))
        if m:
            out.append((a,m.group(1)))
    return out


def _smallest_container(text_node, expected: int | None, stop_node=None):
    """Climb from a header text node to the smallest ancestor containing stock links."""
    cur=text_node.parent
    fallback=None
    for _ in range(8):
        if cur is None or cur == stop_node:
            break
        links=_stock_anchors(cur)
        if links:
            if fallback is None:
                fallback=cur
            if expected is None or len(links) <= max(expected+2, int(expected*1.15)+1):
                return cur
        cur=cur.parent
    return fallback


def _direct_stock_name(a) -> str:
    text=a.get_text(" ",strip=True)
    text=re.sub(r"\b\d{2}:\d{2}\b", "", text)
    text=text.replace("风险警示", "").strip()
    return text


def _seal_time(a) -> str | None:
    m=re.search(r"(\d{2}:\d{2})", a.get_text(" ",strip=True))
    return m.group(1) if m else None


def _section_bounds(soup, start_text: str, end_text: str | None):
    start=soup.find(lambda tag: getattr(tag,"name",None) in ("h2","h3") and start_text in tag.get_text(" ",strip=True))
    if start is None:
        return None, None
    end=None
    if end_text:
        for tag in start.find_all_next(["h2","h3"]):
            if tag is start:
                continue
            if end_text in tag.get_text(" ",strip=True):
                end=tag
                break
    return start,end


def _text_nodes_between(start,end,pattern):
    if start is None:
        return []
    out=[]
    for s in start.find_all_next(string=True):
        parent=getattr(s,"parent",None)
        if end is not None and (parent is end or end in getattr(parent,"parents",[])):
            break
        txt=str(s).strip()
        if pattern.match(txt):
            out.append(s)
    return out


def robust_parse_yyqyx(day: str, html: str):
    soup=BeautifulSoup(html,"html.parser")
    text=" ".join(soup.stripped_strings)

    def n(*patterns):
        for p in patterns:
            m=re.search(p,text)
            if m:
                try: return float(m.group(1).replace(",",""))
                except Exception: pass
        return None

    summary={
      "date":day,"source_id":"yyqyx_limitup",
      "limit_up_count":n(r"(?:当日)?涨停\s*(\d+)\s*只", r"共\s*(\d+)\s*只个股涨停"),
      "board_count":n(r"连板\s*(\d+)\s*只"),
      "max_board":n(r"最高\s*(\d+)\s*连板", r"最高\s*(\d+)\s*板"),
      "broken_board_count":n(r"炸板\s*(\d+)\s*只"),
      "limit_down_count":n(r"跌停\s*(\d+)\s*只"),
      "advancers":n(r"上涨\s*([\d,]+)\s*只"),
      "decliners":n(r"下跌\s*([\d,]+)\s*只"),
      "turnover_trillion":n(r"全市场成交\s*([\d.]+)\s*万亿元", r"成交\s*([\d.]+)\s*万亿元"),
      "availability":"archive; structural membership only; semantic PIT requires timestamped same-day corroboration"
    }

    members=[]; ladder=[]
    t_start,t_end=_section_bounds(soup,"涨停题材分布","全部涨停股")
    seen=set()
    for s in _text_nodes_between(t_start,t_end,THEME_HEADER):
        mm=THEME_HEADER.match(str(s).strip())
        if not mm: continue
        raw_theme=mm.group(1).strip(); expected=int(mm.group(2)); canon=b.canonical_theme(raw_theme)
        row=_smallest_container(s,expected,stop_node=t_start.parent if t_start else None)
        if row is None: continue
        links=_stock_anchors(row)
        if expected and len(links)>expected:
            links=links[:expected]
        for a,code in links:
            key=(raw_theme,code)
            if key in seen: continue
            seen.add(key)
            nm=_direct_stock_name(a)
            members.append({
              "date":day,"source_id":"yyqyx_limitup","raw_theme":raw_theme,"canonical_theme":canon,
              "source_theme_count":expected,"code":code,"stock_name":nm,"source_seal_time":_seal_time(a),
              "risk_flag":"ST" in nm.upper(),"trade_effective_date":b.next_day(day),
              "availability_confidence":"medium_archive_membership"
            })

    b_start,b_end=_section_bounds(soup,"连板天梯","涨停题材分布")
    seen_l=set()
    for s in _text_nodes_between(b_start,b_end,BOARD_HEADER):
        mm=BOARD_HEADER.match(str(s).strip())
        if not mm: continue
        level=int(mm.group(1)); row=_smallest_container(s,None,stop_node=b_start.parent if b_start else None)
        if row is None: continue
        for a,code in _stock_anchors(row):
            key=(level,code)
            if key in seen_l: continue
            seen_l.add(key); nm=_direct_stock_name(a)
            ladder.append({"date":day,"source_id":"yyqyx_limitup","code":code,"stock_name":nm,"board_level":level,"source_time":_seal_time(a)})

    if not members and t_start is not None:
        for elem in t_start.find_all_next(["div","li","section","p"]):
            if t_end is not None and elem is t_end: break
            txt=elem.get_text(" ",strip=True)
            mm=re.match(r"^(.+?)\s*[·•]\s*(\d+)\s*只",txt)
            links=_stock_anchors(elem)
            if not mm or not links: continue
            raw_theme=mm.group(1).strip(); expected=int(mm.group(2)); canon=b.canonical_theme(raw_theme)
            if len(links)>expected+2: continue
            for a,code in links[:expected]:
                key=(raw_theme,code)
                if key in seen: continue
                seen.add(key); nm=_direct_stock_name(a)
                members.append({"date":day,"source_id":"yyqyx_limitup","raw_theme":raw_theme,"canonical_theme":canon,"source_theme_count":expected,"code":code,"stock_name":nm,"source_seal_time":_seal_time(a),"risk_flag":"ST" in nm.upper(),"trade_effective_date":b.next_day(day),"availability_confidence":"medium_archive_membership_fallback"})

    return summary,members,ladder
