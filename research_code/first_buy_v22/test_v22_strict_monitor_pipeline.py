# -*- coding: utf-8 -*-
"""Synthetic date replay, never real Oct8 market evidence."""
from pathlib import Path
import tempfile,json,datetime as dt
import pandas as pd,zoneinfo
import v21_forward_engine as v21
import v22_strict_monitor as m

def main():
 old=(v21.RAW,v21.now,m.OUT,m.verify)
 with tempfile.TemporaryDirectory(prefix='v22_strict_fixture_') as tmp:
  root=Path(tmp);d=root/'raw/daily/tdx_parquet/date=2026-10-08'
  d.mkdir(parents=True)
  z=pd.read_parquet(Path.home()/'Documents/stock_data_windows_mirror/daily/tdx_parquet/date=2026-09-30/daily_ohlcv.parquet')
  z['date']='2026-10-08'
  z.to_parquet(d/'daily_ohlcv.parquet',index=False)
  try:
   v21.RAW=root/'raw'
   v21.now=lambda:dt.datetime(2026,10,8,16,30,tzinfo=zoneinfo.ZoneInfo('Asia/Shanghai'))
   m.OUT=root/'sandbox_only';m.OUT.mkdir()
   m.verify=lambda:{'config_hash':'synthetic_never_real'}
   m.scan()
   q=json.loads((m.OUT/'strict_forward_status.json').read_text())
   assert q['eligible_day1_count']==3
   assert q['strict_day2_entries']==0
   assert q['strict_t1_closed']==0
   assert q['win_rate'] is None
   path=m.OUT/'snapshots/day1/2026-10-08.json'
   oldsnap=json.loads(path.read_text())
   assert len(oldsnap['records'])==3
   m.scan()
   newsnap=json.loads(path.read_text())
   assert oldsnap==newsnap
  finally:
   v21.RAW,v21.now,m.OUT,m.verify=old
 print('V22_STRICT_MONITOR_PIPELINE_OK synthetic Oct8, no fabricated trades, immutable Day1')
if __name__=='__main__':main()