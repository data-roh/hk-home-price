"""Expanding-origin latest-vintage evaluation; never a claim of a PIT backtest."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .models import fit_model
from .scenarios import baseline_path
from .data import prepare_frame

# Shared pseudo-OOS protocol for the notebook, audit script, and app: every 3rd month from month 120, all origins.
PSEUDO_OOS=dict(horizons=(1,3,6,12),min_train=120,step=3,last_origins=None)

def rolling_evaluate(df,spec,horizons=(1,3,6,12),min_train=120,step=6,last_origins=12,oracle=False):
    # Preserve raw inputs for feature-lag handling in each fold.
    required=[spec.target]+list(spec.features)
    raw=df.loc[df.index>=spec.start,required]
    good=raw.notna().all(axis=1)
    if not good.any():raise ValueError('No complete data')
    raw=raw.loc[good[good].index[0]:good[good].index[-1]]
    if raw.isna().any().any():raise ValueError('Internal missing values; rolling validation refuses compressed time')
    maxh=max(horizons);origins=list(range(min_train,len(raw)-maxh+1,step))
    if last_origins:origins=origins[-last_origins:]
    if not origins:raise ValueError('Not enough history for chosen validation setup')
    rows=[];errors=[]
    for n in origins:
        train=raw.iloc[:n];test=raw.iloc[n:n+maxh]
        try:
            fitted=fit_model(train,spec)
            # Operational-X projection uses history only. Oracle mode is diagnostic, explicitly labelled.
            future=test[list(spec.features)] if oracle else baseline_path(train,spec.features,maxh,'last')
            fc=fitted.forecast(future)
            denom=float(train[spec.target].diff(12).abs().dropna().mean())
            for h in horizons:
                actual=float(test[spec.target].iloc[h-1]);pred=float(fc['prediction_median'].iloc[h-1]);rw=float(train[spec.target].iloc[-1])
                vals=list(train[spec.target])
                for j in range(h): vals.append(vals[-12])
                for method,point in [(spec.key+'_'+spec.seasonality,pred),('random_walk',rw),('seasonal_naive',float(vals[-1]))]:
                    rows.append(dict(origin=train.index[-1],date=test.index[h-1],horizon=h,method=method,actual=actual,prediction=point,error=actual-point,abs_error=abs(actual-point),squared_error=(actual-point)**2,mase=abs(actual-point)/denom if denom>0 else np.nan,coverage_95=(bool(fc.lower_95.iloc[h-1]<=actual<=fc.upper_95.iloc[h-1]) if method.startswith(spec.key+'_') else np.nan),interval_width=(float(fc.upper_95.iloc[h-1]-fc.lower_95.iloc[h-1]) if method.startswith(spec.key+'_') else np.nan),exog_mode='ORACLE_REALIZED_X' if oracle else 'TRAIN_ONLY_LAST_VALUE_PATH',data_vintage='LATEST_VINTAGE_NOT_PIT'))
        except Exception as exc:errors.append(dict(origin=str(train.index[-1]),error=str(exc)))
    return pd.DataFrame(rows),pd.DataFrame(errors,columns=['origin','error'])

def scorecard(predictions):
    if predictions.empty:return pd.DataFrame()
    g=predictions.groupby(['method','horizon'])
    out=g.agg(n=('error','size'),mae=('abs_error','mean'),mse=('squared_error','mean'),mase=('mase','mean'),coverage_95=('coverage_95','mean'),mean_interval_width=('interval_width','mean')).reset_index()
    out['rmse']=np.sqrt(out.pop('mse'))
    return out

def common_origin_comparison(results):
    """Compare on intersection of successful (origin,horizon), not survivorship-biased folds."""
    if not results:return pd.DataFrame(),pd.DataFrame()
    keysets=[]
    for pred in results:
        model_rows=pred.loc[~pred.method.isin(['random_walk','seasonal_naive'])]
        keysets.append(set(zip(model_rows.origin,model_rows.horizon)))
    common=set.intersection(*keysets)
    if not common:raise ValueError('No common successful origins across all specifications')
    joined=pd.concat(results,ignore_index=True)
    joined=joined.loc[[(o,h) in common for o,h in zip(joined.origin,joined.horizon)]]
    joined=joined.drop_duplicates(['method','origin','horizon'])
    return scorecard(joined),joined
