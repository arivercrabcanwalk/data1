import numpy as np
import pandas as pd
import pytest
from core import COLS, clock, five_bars
from data_io import attach_slots, day_arrays


def sample():
    times=['2026-07-20 '+clock(k)+':00' for k in range(240)]
    times[119]='2026-07-20 13:00:00'
    data=pd.DataFrame(np.tile([10,10.2,9.8,10.1,1000,10000],(240,1)),columns=COLS)
    data['code']='000001';data['datetime']=times
    data.loc[119,['open','high','low','close']]=99.
    return data


def test_exception_not_retimed_or_used_in_array():
    d=attach_slots(sample())
    assert len(d)==240 and (d.slot<0).sum()==1
    assert d.iloc[0]['open']==10 and d.iloc[-1]['close']==10.1
    a=day_arrays(d)['000001']
    assert np.isnan(a[119]).all() and np.isfinite(a[120]).all()
    assert np.nanmax(a[:,:4])<99
    assert np.isnan(five_bars(a)[23]).all()
    assert np.isfinite(five_bars(a)[24]).all()


def test_unknown_time_still_fails():
    d=sample();d.loc[119,'datetime']='2026-07-20 12:01:00'
    with pytest.raises(ValueError,match='Unrecognized'):attach_slots(d)


def test_valid_prior_bar_count_not_raw_count():
    d=attach_slots(sample())
    assert len(d)==240 and (d.slot>=0).sum()==239
    assert d.high.max()==99


def test_large_string_filtered_file(tmp_path):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    from data_io import read_minute
    d=sample();d['code']='000001'
    t=pa.Table.from_pandas(d,preserve_index=False)
    t=t.set_column(t.column_names.index('code'),'code',pa.array(d.code,type=pa.large_string()))
    p=tmp_path/'minute1.parquet';pq.write_table(t,p)
    actual=read_minute(p,{'000001'})
    assert len(actual)==240 and (actual.slot<0).sum()==1
