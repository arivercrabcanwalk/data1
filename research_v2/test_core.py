"""Regression tests: bar causality, A-share settlement, fills and ledger."""
import numpy as np
import pandas as pd
import pytest
from core import (Candidate, Config, Market, Portfolio, clock, configurations,
                  day_arrays, execution, fee, five_bars, minute_slot, norm_code,
                  signal_slot, make_candidates)


def bars(price=10., volume=1_000_000.):
    x = np.tile([price, price+.20, price-.20, price+.02, volume, price*volume], (240, 1))
    return x.astype(float)


def market(n=70, prices=None):
    dates = pd.bdate_range('2026-03-02', periods=n).strftime('%Y-%m-%d')
    rows = []
    for i, day in enumerate(dates):
        p = 10. if prices is None else prices[i]
        rows.append(dict(date=day, code='000001', open=p, high=p+.2, low=p-.2,
            close=p, volume=2e7, amount=2e8, bars=240, st_known=0., new_initial=0.))
    return Market(pd.DataFrame(rows))


@pytest.mark.parametrize('code,expected', [('1','000001'), ('000001','000001'),
                                          ('688001','688001'), ('300001.SZ','300001')])
def test_code(code, expected):
    assert norm_code(pd.Series([code])).iloc[0] == expected


def test_session_slots():
    d = pd.Series(['2026-06-01 09:31:00','2026-06-01 11:30:00',
                   '2026-06-01 13:01:00','2026-06-01 15:00:00'])
    assert list(minute_slot(d)) == [0,119,120,239]
    assert [clock(k) for k in [0,119,120,239]] == ['09:31','11:30','13:01','15:00']


@pytest.mark.parametrize('time', ['09:30:00','12:00:00','13:00:00','15:01:00','09:31:01'])
def test_bad_session(time):
    with pytest.raises(ValueError): minute_slot(pd.Series(['2026-06-01 '+time]))


def test_exact_five_not_rounded_groups():
    x = bars()
    x[:,0] = np.arange(240)
    x[:,3] = np.arange(240)
    f = five_bars(x)
    assert np.array_equal(f[:,0], np.arange(0,240,5))
    assert np.array_equal(f[:,3], np.arange(4,240,5))
    assert f[23,3] == 119 and f[24,0] == 120


def test_missing_bar_invalidates_only_its_five():
    x = bars(); x[7] = np.nan
    f = five_bars(x)
    assert np.isnan(f[1]).all() and np.isfinite(f[0]).all()


def test_signal_next_minute():
    x = bars()
    c = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    assert signal_slot(c,x) == 10
    assert clock(9) == '09:40' and clock(10) == '09:41'


def test_no_future_in_entry_signal():
    x = bars()
    c = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    y = x.copy(); y[10:] = bars(1.)[10:]
    assert signal_slot(c,x) == signal_slot(c,y) == 10


def test_support_test_plus_two_complete_bars():
    x = bars()
    c = Candidate('GDZY','000001',1,0,9.8,10.,0.,2e8)
    assert signal_slot(c,x) == 15


def test_pre_signal_support_break_cancels():
    x = bars(); x[4,3] = 9.
    c = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    assert signal_slot(c,x) is None


@pytest.mark.parametrize('side', ['buy','sell'])
def test_price_adverse_rounding(side):
    price,q,reason = execution(np.array([10.,10.2,9.8,10.,1e6,1e7]), side,10000,.0015,.01)
    assert price == (10.02 if side=='buy' else 9.98)
    assert q > 0 and reason == 'filled'


@pytest.mark.parametrize('row,reason', [([10,10,10,10,1e6,1e7],'single_price'),
    ([10,10.2,9.8,10,0,0],'no_flow'), ([10,10.01,9.99,10,1e6,1e7],'slip_outside_bar')])
def test_unexecutable(row,reason):
    assert execution(np.array(row,float),'buy',1000,.0015,.01)[2] == reason


def test_capacity_and_star_minimum():
    row = np.array([10,10.2,9.8,10,15000,150000],float)
    assert execution(row,'buy',1000,0,.01)[1] == 100
    assert execution(row,'buy',1000,0,.01,200)[1] == 0


def test_fee_floor_and_stamp():
    assert fee(1000,'buy') == 5.01
    assert fee(1000,'sell') == 5.51
    assert fee(100000,'sell') == 81.


def enter(exit='CLOSE'):
    m = market(3)
    p = Portfolio(Config('TEST',('EARLY',),exit))
    cand = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    x = bars()
    p.day(1,m,[cand],{'000001':x})
    assert len(p.positions) == 1
    return m,p,cand,x


def test_t1_and_entry_day_close_stop():
    m = market(3,[10.,9.5,9.4])
    p = Portfolio(Config('TEST',('EARLY',),'CLOSE'))
    cand = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    x = bars(); x[200:] = bars(9.5)[200:]
    p.day(1,m,[cand],{'000001':x})
    assert len(p.fills) == 1 and p.positions['000001']['pending'] == 'stop'
    p.day(2,m,[],{'000001':bars(9.4)})
    assert len(p.closed) == 1 and all(f['date'] != m.dates[1] for f in p.fills if f['side']=='sell')


def test_intraday_stop_waits_until_next_day_even_on_recovery():
    m = market(3)
    p = Portfolio(Config('TEST',('EARLY',),'INTRADAY'))
    cand = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    x = bars(); x[100:120] = bars(9.4)[100:120]
    p.day(1,m,[cand],{'000001':x})
    assert p.positions['000001']['pending'] == 'stop'
    assert len(p.fills) == 1
    p.day(2,m,[],{'000001':bars()})
    assert len(p.closed) == 1


def test_no_forced_end_liquidation(tmp_path):
    m,p,c,x = enter()
    s = p.finish(tmp_path)
    assert s['open_positions'] == 1 and s['closed_trades'] == 0
    assert s['terminal_cash'] < s['terminal_equity']
    assert abs(s['terminal_equity']-1e6-s['marked_open_pnl']) < .001


def test_stop_persists_through_untradable_day():
    m,p,c,x = enter('INTRADAY')
    p.positions['000001']['pending'] = 'stop'
    y = bars(); y[:,1:4] = 10.
    p.day(2,m,[],{'000001':y})
    assert '000001' in p.positions and p.positions['000001']['pending'] == 'stop'


def test_drawdown_includes_initial_capital(tmp_path):
    m,p,c,x = enter()
    s = p.finish(tmp_path)
    assert s['daily_max_drawdown'] <= 0
    assert s['minute_max_drawdown'] < 0


def test_freeze_twenty_eight_variants():
    cfgs = configurations()
    assert len(cfgs)==28 and len({c.name for c in cfgs})==28
    assert all(c.risk_per_trade == .005 for c in cfgs)


def test_daily_signal_prefix_invariant():
    prices = np.full(70,10.); prices[40] = 10.8; prices[41:] = 10.7
    a = market(70,prices)
    b = market(51,prices[:51])
    for m in (a,b):
        m.x['open'][40,0] = 10.
        m.big = (m.ret>=.07) & (m.x['close']>m.x['open']) & ~m.bad_gap
    x, y = make_candidates(a), make_candidates(b)
    assert x.get(41) and y.get(41)
    for k in range(1,51):
        assert x.get(k,[]) == y.get(k,[])


def test_latest_big_anchor_replaces_old():
    prices = np.full(70,10.); prices[40:45] = 10.8; prices[45:] = 11.6
    m = market(70,prices)
    m.x['open'][40,0] = 10.; m.x['open'][45,0] = 10.8
    m.big = (m.ret>=.07) & (m.x['close']>m.x['open']) & ~m.bad_gap
    records = make_candidates(m)
    after = [c for k,cs in records.items() if k>=47 for c in cs if c.method in ('GDZY','PBQY')]
    assert after and all(c.anchor_day == 45 for c in after)


def test_mae_not_updated_after_exit():
    m,p,c,x = enter('INTRADAY')
    p.positions['000001']['pending'] = 'stop'
    y = bars(); y[50:] = bars(1.)[50:]
    p.day(2,m,[],{'000001':y})
    assert len(p.closed) == 1 and p.closed[0]['mae'] > -.1


def test_protect_stop_overrides_profit_before_t1():
    m = market(3)
    p = Portfolio(Config('TEST',('EARLY',),'PROTECT'))
    c = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    x = bars(); x[30:60] = bars(11.)[30:60]; x[100:120] = bars(9.4)[100:120]
    p.day(1,m,[c],{'000001':x})
    assert p.positions['000001']['pending']=='stop' and len(p.fills)==1
    p.day(2,m,[],{'000001':bars()})
    assert len(p.closed)==1 and all(f['reason']=='stop' for f in p.fills if f['side']=='sell')


def test_same_minute_position_capacity_not_reused():
    m = market(3)
    p = Portfolio(Config('TEST',('EARLY',),'CLOSE',max_positions=1))
    c = Candidate('EARLY','000001',1,0,9.8,10.,0.,2e8)
    p.day(1,m,[c],{'000001':bars()})
    p.positions['000001']['pending']='stop'
    x = bars(); x[:10,1:4] = 10.
    other = Candidate('EARLY','000002',2,1,9.8,10.,0.,2e8)
    p.day(2,m,[other],{'000001':x,'000002':bars()})
    assert len(p.closed)==1 and not p.positions
    assert p.opportunities[-1]['status']=='position_slots'


def test_partial_exit_ledger(tmp_path):
    m,p,c,x = enter('INTRADAY')
    p.positions['000001']['pending']='stop'
    y=bars(volume=15000.)
    p.day(2,m,[],{'000001':y})
    result=p.finish(tmp_path)
    assert len([f for f in p.fills if f['side']=='sell'])>1
    assert abs(result['terminal_equity']-1e6-result['realized_closed_pnl']-result['marked_open_pnl'])<.01


def test_file_level_parquet_ignores_hive_partition_collision(tmp_path):
    pa=pytest.importorskip('pyarrow')
    pq=pytest.importorskip('pyarrow.parquet')
    directory=tmp_path/'date=2026-06-01'; directory.mkdir()
    path=directory/'status.parquet'
    pq.write_table(pa.table({'date':pa.array(['2026-06-01'],type=pa.large_string())}),path)
    assert pq.ParquetFile(path).read().num_rows==1
