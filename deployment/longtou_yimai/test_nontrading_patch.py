# -*- coding: utf-8 -*-
"""Regression tests for independently confirmed no-trade EOD and recovery chains."""
import json, datetime as dt, tempfile
from pathlib import Path
from types import SimpleNamespace as Q
from unittest.mock import patch
import pandas as pd
import longtou_runtime as a

def quote(code,day="2026-10-13",price=13.78,prev=13.78,vol=0,amount=0,
          open_=0,high=0,low=0,src="tencent"):
    return Q(code=code,name="新华传媒",price=price,prev_close=prev,
      open=open_,high=high,low=low,volume=vol,amount=amount,
      server_time=day.replace("-","")+"161448",source=src)

class Fake:
    def __init__(self,t,s,fail_second=False):
        self.t=t;self.s=s;self.fail_second=fail_second
    def tencent_quotes(self,codes):return {codes[0]:self.t}
    def sina_quote(self,code):
        if self.fail_second: raise ConnectionError("synthetic outage")
        return self.s

def test_cross_source():
    prior=pd.Series(dict(close=13.78,name="新华传媒",date="2026-10-12",streak=10))
    z=a.confirm_two_source_no_trading(Fake(quote("600825"),quote("600825",src="sina")),
                                      "600825","2026-10-13",prior)
    assert z["excluded_from_day1"] and z["code"]=="600825"
    def rejects(m,needle):
        try:
            a.confirm_two_source_no_trading(m,"600825","2026-10-13",prior)
        except RuntimeError as e:assert needle in str(e),(needle,str(e))
        else:raise AssertionError("Expected fail-closed "+needle)
    rejects(Fake(quote("600825"),quote("600825",vol=2.0,src="sina")),
            "UNRESOLVED_PREV_BOARD_HAS_TRADING")
    rejects(Fake(quote("600825"),quote("600825",day="2026-10-12",src="sina")),
            "NONTRADING_QUOTE_STALE")
    rejects(Fake(quote("600825"),quote("600825",src="sina"),fail_second=True),
            "UNRESOLVED_PREV_BOARD_QUOTE")
    rejects(Fake(quote("600825"),quote("600825",price=13.70,src="sina")),
            "UNRESOLVED_PREV_BOARD_BASIS_DIFF")
    print("NO_TRADE_REQUIRES_TWO_FRESH_INDEPENDENT_ZERO_QUOTE_FEEDS_PASS")

def test_chain():
    old=a.STATE
    with tempfile.TemporaryDirectory(prefix="longtou_suspension_test_") as temp:
        a.STATE=Path(temp)
        try:
            p=a.STATE/"events"
            p.mkdir()
            (p/"2026-10-13.json").write_text(json.dumps({
                "day":"2026-10-13","nontrading_confirmed":[{"code":"600825"}]}))
            with patch.object(a,"market_calendar",return_value=[
                "2026-10-12","2026-10-13","2026-10-14","2026-10-15"]):
                assert a.prior_nontrading_chain("600825","2026-10-12","2026-10-14")
                assert not a.prior_nontrading_chain("600825","2026-10-12","2026-10-15")
                (p/"2026-10-14.json").write_text(json.dumps({
                    "day":"2026-10-14","nontrading_confirmed":[{"code":"600825"}]}))
                assert a.prior_nontrading_chain("600825","2026-10-12","2026-10-15")
        finally:a.STATE=old
    print("POST_SUSPENSION_CAUSAL_RESET_CHAIN_PASS")

def test_day1_gates():
    e={"event_type":"PRIMARY_3PLUS","branch":"POSITIVE_BREAK",
       "bucket":"B4P","prior_streak":6,"gap":-0.04,"vol_prev":1.14175}
    assert not a.day1_main_strategy_eligible(e)
    e["vol_prev"]=1.51
    assert a.day1_main_strategy_eligible(e)
    e={"event_type":"PRIMARY_3PLUS","branch":"NORMAL_RED","bucket":"B4P",
       "prior_streak":6,"d1_pm_above_ratio":0.26446}
    assert a.day1_main_strategy_eligible(e)
    e={"event_type":"PRIMARY_3PLUS","branch":"PANIC","bucket":"B3","prior_streak":3,
       "amount":1320339291}
    assert a.day1_main_strategy_eligible(e)
    e["prior_streak"]=2
    assert not a.day1_main_strategy_eligible(e)
    print("WATCH_ELIGIBILITY_MATCHES_V22_FROZEN_PRECONDITIONS_PASS")

if __name__=="__main__":
    test_cross_source()
    test_chain()
    test_day1_gates()
    print("NO_TRADING_PATCH_REGRESSION_PASS")