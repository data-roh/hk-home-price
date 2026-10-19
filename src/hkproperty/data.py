import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .specs import SERIES

def read_monthly(source):
    df = pd.read_csv(source)
    if 'date' not in df:
        raise ValueError('CSV must include date and the series names in data_dictionary.csv')
    dates = pd.to_datetime(df.pop('date'), errors='raise')
    ix = pd.DatetimeIndex(dates).tz_localize(None).to_period('M').to_timestamp()
    if ix.duplicated().any():
        raise ValueError('Multiple rows in a calendar month. Aggregate explicitly before loading.')
    df.index = ix
    df.index.name = 'date'
    df = df.sort_index()
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not len(df):
        raise ValueError('Empty dataset')
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='MS')).rename_axis('date')

def audit_monthly(df):
    rows = []
    for name in SERIES:
        x = df[name] if name in df else pd.Series(dtype=float)
        observed = x.dropna()
        rows.append(dict(series=name, n_observed=len(observed), n_missing=int(x.isna().sum()), first=str(observed.index.min()) if len(observed) else None, last=str(observed.index.max()) if len(observed) else None, duplicate_value_fraction=float(observed.diff().eq(0).mean()) if len(observed)>1 else None, in_file=name in df))
    return pd.DataFrame(rows)

def assert_calendar_frame(df):
    expected = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    if not df.index.equals(expected.rename(df.index.name)):
        raise ValueError('The model requires an uninterrupted monthly calendar.')

def prepare_frame(df, spec):
    required = [spec.target] + list(spec.features)
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError(f'Missing series: {sorted(absent)}')
    work = df.loc[df.index >= spec.start, required].copy()
    if spec.feature_lag:
        work[list(spec.features)] = work[list(spec.features)].shift(spec.feature_lag)
    good = work.notna().all(axis=1)
    if not good.any():
        raise ValueError('No complete observations for this specification')
    first, last = good[good].index[[0, -1]]
    work = work.loc[first:last]
    bad = work.index[work.isna().any(axis=1)]
    if len(bad):
        raise ValueError('Internal missing months, not silently dropped: '+', '.join(bad.strftime('%Y-%m')[:12]))
    assert_calendar_frame(work)
    if len(work) < spec.min_obs:
        raise ValueError(f'{len(work)} complete months; require {spec.min_obs}')
    if not np.isfinite(work.to_numpy()).all():
        raise ValueError('Non-finite observations')
    return work

def to_monthly(long, cutoff):
    data = long.copy()
    data['date'] = pd.to_datetime(data['date'])
    if data['date'].dt.tz is not None:
        data['date'] = data['date'].dt.tz_localize(None)  # preserve vendor calendar-date labels
    cutoff = pd.Timestamp(cutoff)
    # Exclude in-progress calendar month; never forward-fill unobserved macro values.
    last_month_end = cutoff.to_period('M').to_timestamp(how='start') - pd.offsets.Day(1)
    data = data.loc[data['date'] <= last_month_end]
    frames = []
    for key, meta in SERIES.items():
        d = data.loc[data['series'].eq(key), ['date', 'value']].sort_values('date')
        if d.empty:
            continue
        if d['date'].duplicated().any():
            raise ValueError('Duplicate vendor dates for '+key)
        series = d.set_index('date')['value'].resample('MS').agg(meta['agg']).rename(key)
        frames.append(series)
    if not frames:
        raise ValueError('No data returned')
    return pd.concat(frames, axis=1).sort_index().rename_axis('date')

def data_fingerprint(df):
    return hashlib.sha256(df.to_csv().encode()).hexdigest()
