from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ModelSpec:
    key: str
    target: str
    features: Tuple[str, ...]
    start: str
    family: str
    transformation: str = 'level'
    seasonality: str = 'none'
    trend: bool = False
    ar_order: int = 1
    order: Tuple[int, int, int] = (2, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    feature_lag: int = 0
    min_obs: int = 72

HOME = ModelSpec('home', 'home_price', ('fed_rate', 'hsi', 'secondary_sales', 'home_rent', 'cnyhkd'), '1997-02-01', 'glsar')
GROWTH = ModelSpec('growth', 'home_price', HOME.features, '1997-02-01', 'growth')
OFFICE = ModelSpec('office', 'office_rent', ('market_cap', 'office_vacancy', 'inflation_yoy', 'unemployment', 'fed_rate'), '2006-01-01', 'sarimax')
SPECS = {'home': HOME, 'office': OFFICE}
LOG_FEATURES = {'hsi', 'secondary_sales', 'home_rent', 'cnyhkd', 'market_cap'}
SERIES = {
    'home_price': dict(ticker='HKRLPDAP Index', unit='index_1999_100', frequency='monthly', agg='last'),
    'fed_rate': dict(ticker='FEDL01 Index', unit='percent', frequency='intraday', agg='mean'),
    'hsi': dict(ticker='HSI Index', unit='index_points', frequency='intraday', agg='last'),
    'secondary_sales': dict(ticker='MIDJLTTL Index', unit='units', frequency='monthly', agg='last'),
    'home_rent': dict(ticker='HKRLPDAR Index', unit='index_1999_100', frequency='monthly', agg='last'),
    'cnyhkd': dict(ticker='CNYHKD BGN Curncy', unit='HKD_per_CNY', frequency='intraday', agg='last'),
    'office_rent': dict(ticker='COLNETE Index', unit='HKD_per_sq_ft', frequency='monthly', agg='last'),
    'market_cap': dict(ticker='HKMCTTL Index', unit='HKD_billion', frequency='daily', agg='last'),
    'office_vacancy': dict(ticker='COLLVACA Index', unit='percent', frequency='monthly', agg='last'),
    'inflation_yoy': dict(ticker='HKCPIY Index', unit='percent_yoy', frequency='monthly', agg='last'),
    'unemployment': dict(ticker='HKUERATE Index', unit='percent', frequency='monthly', agg='last'),
}
# Tickers transcribed from user screenshots, not independently Terminal-validated.
# Default month-end / monthly-mean choices are reconstruction assumptions, not disclosed BI settings.
