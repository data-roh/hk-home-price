"""An ECM challenger. Enforce the precondition; do not manufacture cointegration."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

@dataclass
class ECMResult:
    long_run: object
    short_run: object
    cointegration_pvalue: float
    adjustment_speed: float
    long_features: tuple

def fit_home_ecm(df,long_features=('home_rent','hsi','cnyhkd'),allow_exploratory=False):
    cols=['home_price','secondary_sales','fed_rate']+list(long_features)
    d=df[cols].copy()
    if d.isna().any().any():raise ValueError('ECM requires a contiguous complete monthly sample')
    if (d[['home_price','secondary_sales']+list(long_features)]<=0).any().any():raise ValueError('ECM logs require positive values')
    if len(d)<120:raise ValueError('Need at least 120 months')
    y=np.log(d.home_price);X=np.log(d[list(long_features)])
    # Engle-Granger residual test uses its own nonstandard critical values.
    p=float(coint(y,X,trend='c',autolag='aic')[1])
    if p>=.05 and not allow_exploratory:raise ValueError(f'Cointegration not supported (p={p:.3g}); ECM withheld. Assess integration order and breaks first.')
    lr=sm.OLS(y,sm.add_constant(X)).fit();ect=lr.resid.shift(1)
    Z=pd.DataFrame({'ect_lag1':ect,'dy_lag1':y.diff().shift(1),'d_sales_lag1':np.log(d.secondary_sales).diff().shift(1),'d_rate_lag1':d.fed_rate.diff().shift(1)})
    for col in long_features:Z['d_'+col+'_lag1']=X[col].diff().shift(1)
    use=pd.concat([y.diff().rename('dy'),Z],axis=1).dropna()
    sr=sm.OLS(use.dy,sm.add_constant(use.drop(columns='dy'))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    return ECMResult(lr,sr,p,float(sr.params['ect_lag1']),tuple(long_features))
