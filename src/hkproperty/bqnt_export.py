"""Execute in Bloomberg BQNT with the user's data entitlements. No credentials stored."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import pandas as pd
from .specs import SERIES
from .data import to_monthly

def extract_bqnt(output_dir='data/real', start='1997-02-01', as_of=None, service=None):
    if service is None:
        import bql
        service = bql.Service()
    import bql
    as_of = as_of or datetime.now(ZoneInfo('Asia/Hong_Kong')).date().isoformat()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, errors, requests = [], [], []
    for series, meta in SERIES.items():
        try:
            # Minimal documented BQL Request/execute pattern. No implicit PREV fill requested.
            item = service.data.px_last(dates=service.func.range(start, as_of))
            request = bql.Request(meta['ticker'], {series: item})
            response = service.execute(request)
            frame = response[0].df().reset_index()
            frame.to_csv(out / f'raw_{series}.csv', index=False)
            upper = {str(c).upper(): c for c in frame.columns}
            if 'DATE' not in upper or series not in frame:
                raise ValueError(f'Expected DATE and {series}; returned columns: {list(frame.columns)}')
            value = pd.to_numeric(frame[series], errors='raise')
            piece = pd.DataFrame({'date':pd.to_datetime(frame[upper['DATE']]), 'value':value})
            piece['series'], piece['ticker'] = series, meta['ticker']
            piece['unit'], piece['aggregation'] = meta['unit'], meta['agg']
            piece['retrieved_at'] = datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
            piece['source_published_at'] = None  # Unknown; vendor DATE is not a release timestamp.
            frames.append(piece)
            requests.append(dict(series=series, ticker=meta['ticker'], field='PX_LAST', status='success', n=len(piece)))
        except Exception as exc:
            errors.append(dict(series=series, ticker=meta['ticker'], error=str(exc)))
    pd.DataFrame(errors, columns=['series','ticker','error']).to_csv(out/'extraction_errors.csv', index=False)
    manifest = dict(as_of=as_of, start=start, requests=requests, errors=errors, actual_execution='BQNT only; check raw files', date_semantics='Unknown until verified in Terminal/source metadata', point_in_time=False, aggregation='fed mean; HSI/FX/market cap last; macro last; no missing-value imputation', ticker_validation='Transcribed from screenshots. Verify DES/FLDS, units, periodicity, and entitlement. No silent substitution.')
    (out/'extraction_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not frames:
        raise RuntimeError('No series extracted. See extraction_errors.csv')
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(out/'bql_observations_long.csv', index=False)
    monthly = to_monthly(long, as_of)
    monthly.to_csv(out/'monthly_data.csv')
    if errors:
        print(f'{len(errors)} series failed. Export is PARTIAL; do not fit missing models. See extraction_errors.csv')
    print('Wrote', out/'monthly_data.csv', 'and raw data, request errors, and manifest.')
    return monthly, manifest
