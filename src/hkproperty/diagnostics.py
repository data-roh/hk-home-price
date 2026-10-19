import warnings
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from statsmodels.tsa.stattools import adfuller,kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

def residual_diagnostics(fitted):
    r=fitted.result
    if fitted.spec.family=='sarimax':
        burn=max(int(r.loglikelihood_burn),12)
        e=np.asarray(r.filter_results.standardized_forecasts_error[0])[burn:]
        ardf=fitted.spec.order[0]+fitted.spec.order[2]+fitted.spec.seasonal_order[0]+fitted.spec.seasonal_order[2]
    else:
        e=np.asarray(r.wresid);ardf=fitted.spec.ar_order
    e=e[np.isfinite(e)]
    lb=acorr_ljungbox(e,lags=[6,12,24],model_df=ardf,return_df=True)
    arch=het_arch(e,nlags=min(12,len(e)//5))
    jb=jarque_bera(e)
    return dict(innovation=pd.Series(e,name='innovation'),ljung_box=lb,arch_lm_pvalue=float(arch[1]),normality_jb_pvalue=float(jb.pvalue),design_condition_number=float(np.linalg.cond(fitted.X.to_numpy())))

def stationarity_table(df):
    records=[]
    for col in df:
        s=df[col].dropna()
        for transform,z in [('level',s),('first_difference',s.diff().dropna())]:
            if len(z)<36 or z.std()==0:continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a=adfuller(z,regression='ct' if transform=='level' else 'c',autolag='AIC')
                k=kpss(z,regression='ct' if transform=='level' else 'c',nlags='auto')
            records.append(dict(series=col,transform=transform,adf_pvalue=a[1],kpss_pvalue=k[1],adf_null='unit root',kpss_null='trend stationary' if transform=='level' else 'level stationary',note='Diagnostics, not an automatic differencing rule. Breaks and low power matter.'))
    return pd.DataFrame(records)

def vif_table(X):
    return pd.DataFrame([dict(term=col,vif=variance_inflation_factor(X.to_numpy(),i)) for i,col in enumerate(X.columns) if col!='const'])
