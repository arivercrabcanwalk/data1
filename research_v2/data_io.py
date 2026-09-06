"""Audited input adapter: never invent a trading time for 13:00 source records.

Raw prices remain in end-of-day aggregates once observed. Out-of-grid records
cannot create an intraday signal or fill. Valid bar count, not raw count, gates
next-day entry eligibility. No missing minute is forward-filled into a signal.
"""
from __future__ import annotations
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from core import COLS, SOURCE_COMMIT, ST_CUTOFF, norm_code, minute_slot, bool_number, save_json


def attach_slots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    index, unique = pd.factorize(df['code'], sort=False)
    if (index < 0).any(): raise ValueError('Missing code')
    df['code'] = norm_code(pd.Series(unique)).to_numpy()[index]
    tag = df['datetime'].astype(str).str[11:]
    exceptional = tag.eq('13:00:00')
    df['slot'] = -1
    try:
        df.loc[~exceptional, 'slot'] = minute_slot(df.loc[~exceptional, 'datetime'])
    except ValueError as error:
        raise ValueError('Unrecognized timestamp; no silent repair allowed') from error
    return df.sort_values(['code', 'datetime'], kind='stable')


def read_minute(path: Path, codes: set[str] | None = None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    tab = pq.ParquetFile(path).read(columns=['code', 'datetime'] + COLS)
    if codes is not None:
        tab = tab.filter(pc.is_in(tab['code'], value_set=pa.array(sorted(codes), type=pa.large_string())))
    try:
        return attach_slots(tab.to_pandas())
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error


def day_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for code, g in df.groupby('code', sort=False):
        x = np.full((240, 6), np.nan)
        valid = g[g.slot >= 0]
        if valid.slot.duplicated().any(): raise ValueError('Duplicate valid minute')
        x[valid.slot.to_numpy(int)] = valid[COLS].to_numpy(float)
        out[code] = x
    return out


def audit_and_daily(root: Path, out: Path) -> tuple[pd.DataFrame, dict[str, Path], dict]:
    import pyarrow.parquet as pq
    fs = sorted(root.glob('2026-*/part-*/date=*/minute1.parquet'))
    if len(fs) != 126: raise ValueError(f'Expected 126 minute files, found {len(fs)}')
    daily, receipts, audits, quarantined = [], [], [], []
    fmap = {}
    status_counts = Counter()
    for n, p in enumerate(fs):
        day = re.search(r'date=(\d{4}-\d{2}-\d{2})', str(p)).group(1)
        if day in fmap: raise ValueError('Duplicate date input')
        fmap[day] = p
        df = read_minute(p)
        x = df[COLS].to_numpy(float)
        bad = (~np.isfinite(x)).any(axis=1) | (x[:, :4] <= 0).any(axis=1)
        bad |= (x[:, 1] < x[:, 2]) | (x[:, 0] < x[:, 2]) | (x[:, 0] > x[:, 1])
        bad |= (x[:, 3] < x[:, 2]) | (x[:, 3] > x[:, 1]) | (x[:, 4:] < 0).any(axis=1)
        dup = int(df.duplicated(['code', 'datetime']).sum())
        wrong_date = int((df.datetime.str[:10] != day).sum())
        a = dict(date=day, rows=len(df), bad_rows=int(bad.sum()), duplicate_rows=dup,
                 wrong_date=wrong_date, no_flow=int(((x[:, 4] <= 1e-6) | (x[:, 5] <= 1e-6)).sum()),
                 quarantined_intraday_rows=int((df.slot < 0).sum()))
        if bad.any() or dup or wrong_date:
            save_json(out/'failed_audit.json', a)
            raise ValueError(f'Raw audit failed: {a}')
        q = df[df.slot < 0].copy()
        if len(q):
            q['source_path'] = str(p.relative_to(root))
            quarantined.append(q)
            print('QUARANTINE', day, len(q), flush=True)
        good = (x[:, 4] > 1e-6) & (x[:, 5] > 1e-6)
        ratio = x[good, 5]/(x[good, 4]*x[good, 3])
        a['amount_volume_price_median'] = float(np.median(ratio)) if len(ratio) else None
        d = df.groupby('code', sort=False).agg(open=('open','first'), high=('high','max'),
            low=('low','min'), close=('close','last'), volume=('volume','sum'),
            amount=('amount','sum'), raw_bars=('slot','size'))
        valid_counts = df[df.slot >= 0].groupby('code',sort=False).size()
        d['bars'] = valid_counts.reindex(d.index).fillna(0).astype(int)
        d['date'] = day
        sp = p.with_name('daily_stock_status.parquet')
        if not sp.exists(): raise ValueError(f'Missing status: {sp}')
        st = pq.ParquetFile(sp).read().to_pandas()
        st['code'] = norm_code(st['code'])
        if st.code.duplicated().any(): raise ValueError('Duplicate status code')
        if not st.date.astype(str).eq(day).all(): raise ValueError('Wrong status date')
        for col in ['status_confidence', 'status_source']:
            status_counts.update({f'{col}:{k}':int(v) for k,v in st[col].value_counts(dropna=False).items()})
        st = st.set_index('code')
        trusted = st.status_source.astype(str).str.contains('historical_stock_status:bak_daily_historical', regex=False)
        d['st_raw'] = bool_number(st.is_st).where(trusted).reindex(d.index)
        d['st_historical_evidence'] = trusted.reindex(d.index).fillna(False)
        d['new_initial'] = bool_number(st.is_new_listing_initial).reindex(d.index)
        d['status_present'] = d.index.isin(st.index)
        a.update(status_rows=len(st), historical_st_rows=int(trusted.sum()),
                 historical_st_minute_codes=int(d.st_historical_evidence.sum()), stock_days=len(d),
                 incomplete_stock_days=int((d.bars != 240).sum()),
                 status_missing_for_minute_codes=int((~d.status_present).sum()))
        audits.append(a)
        daily.append(d.reset_index())
        for f in (p,sp):
            with f.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            receipts.append(dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=digest))
        if (n+1)%21 == 0 or n+1 == len(fs):
            pd.DataFrame(audits).to_csv(out/'daily_audit.csv', index=False)
            print('AUDIT', n+1, len(fs), flush=True)
    daily_df = pd.concat(daily, ignore_index=True).sort_values(['code','date'])
    daily_df['st_known'] = daily_df.st_raw.where(daily_df.date <= ST_CUTOFF)
    daily_df['st_known'] = daily_df.groupby('code', sort=False).st_known.ffill()
    daily_df.to_csv(out/'daily_bars.csv.gz', index=False, compression='gzip')
    pd.DataFrame(receipts).to_csv(out/'data_sha256.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'daily_audit.csv', index=False)
    if quarantined:
        pd.concat(quarantined, ignore_index=True).to_csv(out/'quarantined_intraday_rows.csv.gz', index=False, compression='gzip')
    audit = dict(source_commit=SOURCE_COMMIT,run_commit=os.getenv('GITHUB_SHA','local'),
        minute_files=len(fs),status_files=len(fs),minute_bytes=sum(p.stat().st_size for p in fs),
        raw_rows=sum(a['rows'] for a in audits),stock_days=len(daily_df),codes=int(daily_df.code.nunique()),
        start=min(fmap),end=max(fmap),bad_rows=sum(a['bad_rows'] for a in audits),
        duplicates=sum(a['duplicate_rows'] for a in audits),no_flow_rows=sum(a['no_flow'] for a in audits),
        incomplete_stock_days=sum(a['incomplete_stock_days'] for a in audits),
        quarantined_intraday_rows=sum(a['quarantined_intraday_rows'] for a in audits),
        quarantine_dates=[a['date'] for a in audits if a['quarantined_intraday_rows']],
        status_counts=dict(status_counts),
        amount_volume_price_median_range=[min(a['amount_volume_price_median'] for a in audits),max(a['amount_volume_price_median'] for a in audits)],
        st_policy='per-row historical evidence through 2026-07-30 only; carry last known status forward; ignore current-name approximations',
        adjustment_policy='prices as provided; no corporate-action ledger; not total return',
        timestamp_policy='13:00 records retained in raw EOD aggregates but quarantined from signals/fills; no shift to 11:30; valid 240-minute prior-day eligibility unchanged',
        timestamp_assumption='normal bars end-labelled local exchange time, inferred from session grid; supplier confirmation absent',
        missing=['verified adjustment factors','corporate actions','point-in-time sectors/news',
                 'actual daily limit prices','order book','official benchmark','proof of point-in-time universe completeness'])
    save_json(out/'audit.json',audit)
    return daily_df,fmap,audit
