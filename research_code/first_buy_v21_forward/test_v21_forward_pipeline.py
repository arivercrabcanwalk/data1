# -*- coding: utf-8 -*-
"""Integration smoke test using synthetic Oct8 data (copied Sep30 daily); NEVER actual market results."""
import datetime as dt
import json
import tempfile
from pathlib import Path
import zoneinfo
import pandas as pd
import v21_forward_engine as f

def main():
    old=(f.RAW,f.OUT,f.now,f.require_frozen)
    with tempfile.TemporaryDirectory(prefix='v21_fake_oct8_') as td:
        root=Path(td); dst=root/'raw/daily/tdx_parquet/date=2026-10-08'
        dst.mkdir(parents=True)
        original=Path.home()/'Documents/stock_data_windows_mirror/daily/tdx_parquet/date=2026-09-30/daily_ohlcv.parquet'
        z=pd.read_parquet(original).copy()
        z['date']='2026-10-08'  # Artificial fixture, NOT an observed 2026-10-08 trade day.
        z.to_parquet(dst/'daily_ohlcv.parquet',index=False)
        try:
            f.RAW=root/'raw'
            f.OUT=root/'forward_test_only'
            f.OUT.mkdir()
            f.now=lambda:dt.datetime(2026,10,8,16,30,tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai'))
            f.require_frozen=lambda:{'spec_hash':'test_fixture_only'}
            f.scan_new()
            p=f.OUT/'forward_status.json'
            report=json.loads(p.read_text())
            assert report['day1_primary_events']==3
            assert report['triggered_entries']==0
            assert report['closed_T1_trades']==0
            assert report['win_rate'] is None
            snapshot=f.OUT/'snapshots/day1/2026-10-08.json'
            obj=json.loads(snapshot.read_text())
            assert len(obj['records'])==3
            checksum=obj['signature'];registered=obj['created_at']
            f.scan_new()
            obj2=json.loads(snapshot.read_text())
            assert obj2['signature']==checksum and obj2['created_at']==registered
            assert len(list((f.OUT/'snapshots/day1').glob('*.json')))==1
        finally:
            f.RAW,f.OUT,f.now,f.require_frozen=old
    print('FORWARD_PIPELINE_INTEGRATION_OK mocked 2026-10-08 fixture never used in real ledger')

if __name__=='__main__':
    main()