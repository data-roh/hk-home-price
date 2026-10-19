from dataclasses import dataclass
from dataclasses import asdict
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from .specs import LOG_FEATURES
from .data import prepare_frame

def transform_features(raw, spec):
    z = raw[list(spec.features)].astype(float).copy()
    if spec.transformation == 'log':
        for col in z.columns.intersection(list(LOG_FEATURES)):
            if (z[col] <= 0).any(): raise ValueError('Cannot log non-positive '+col)
            z[col] = np.log(z[col])
    elif spec.transformation != 'level':
        raise ValueError('Use level or log transformation')
    t = z.index.year.to_numpy()*12 + z.index.month.to_numpy()
    if spec.seasonality.startswith('fourier'):
        k = int(spec.seasonality[-1])
        for j in range(1, k+1):
            z[f'sin{j}'] = np.sin(2*np.pi*j*t/12)
            z[f'cos{j}'] = np.cos(2*np.pi*j*t/12)
    elif spec.seasonality == 'dummies':
        for m in range(2,13): z[f'month_{m}'] = (z.index.month == m).astype(float)
    elif spec.seasonality != 'none': raise ValueError('Unknown seasonality')
    if spec.trend: z['linear_trend'] = (t-1990*12)/12
    return z

@dataclass
class FittedModel:
    spec: object
    raw: pd.DataFrame
    y: pd.Series
    X: pd.DataFrame
    mean: pd.Series
    scale: pd.Series
    result: object
    rho: np.ndarray
    messages: list
    converged: bool

    def design(self, future_raw):
        z = transform_features(future_raw, self.spec)
        z = ((z-self.mean)/self.scale).reindex(columns=self.mean.index)
        z.insert(0, 'const', 1.0)
        if not np.isfinite(z.to_numpy()).all(): raise ValueError('Invalid or missing future inputs')
        return z

    def coefficients(self):
        r = self.result
        params = pd.Series(r.params, index=getattr(r.params,'index',None))
        if not set(self.X.columns).issubset(params.index): params.index = self.X.columns
        rows=[]
        for col in self.X.columns:
            j=list(r.model.exog_names).index(col)
            divisor=1.0 if col=='const' else self.scale[col]
            rows.append(dict(term=col, coefficient_standardized=float(params[col]), coefficient_per_transformed_unit=float(params[col]/divisor), standard_error_per_transformed_unit=float(np.asarray(r.bse)[j]/divisor), p_value=float(np.asarray(r.pvalues)[j]), coefficient_scale='dependent log units' if self.spec.transformation=='log' else 'dependent original units', note='const is centered-design intercept; p-values are exploratory, not post-selection-valid'))
        return pd.DataFrame(rows)

    def forecast(self, future_raw, alpha=0.05):
        if not self.converged: raise ValueError('Non-converged model: forecasts withheld')
        expected=pd.date_range(self.raw.index[-1]+pd.offsets.MonthBegin(1), periods=len(future_raw), freq='MS')
        if not future_raw.index.equals(expected.rename(future_raw.index.name)):
            raise ValueError('Future inputs must begin next month and have consecutive monthly rows')
        lag=self.spec.feature_lag
        if lag:
            joined=pd.concat([self.raw[list(self.spec.features)],future_raw[list(self.spec.features)]])
            lagged=joined.shift(lag).loc[future_raw.index]
        else: lagged=future_raw
        Xf=self.design(lagged)
        if self.spec.family=='glsar':
            beta=np.asarray(self.result.params)
            residual=list(np.asarray(self.result.resid))
            armean=[]
            for h in range(len(Xf)):
                e=sum(self.rho[j]*residual[-j-1] for j in range(len(self.rho)))
                armean.append(e);residual.append(e)
            mu=Xf.to_numpy()@beta+np.asarray(armean)
            psi=np.zeros(len(Xf));psi[0]=1.0
            for h in range(1,len(Xf)):
                psi[h]=sum(self.rho[j-1]*psi[h-j] for j in range(1,min(len(self.rho),h)+1))
            variance=float(self.result.scale)*np.cumsum(psi**2)
        else:
            p=self.result.get_forecast(steps=len(Xf),exog=Xf)
            mu=np.asarray(p.predicted_mean);variance=np.asarray(p.var_pred_mean)
        z=stats.norm.ppf(1-alpha/2);se=np.sqrt(np.maximum(variance,0))
        if self.spec.transformation=='log':
            # Median is exp(mu); mean needs lognormal bias adjustment.
            center=np.exp(mu);expected_value=np.exp(mu+0.5*variance);lo=np.exp(mu-z*se);hi=np.exp(mu+z*se)
        else:
            center=mu;expected_value=mu;lo=mu-z*se;hi=mu+z*se
        return pd.DataFrame({'prediction_median':center,'prediction_mean':expected_value,'lower_95':lo,'upper_95':hi,'model_scale_mean':mu,'model_scale_variance':variance},index=future_raw.index)

    def metadata(self):
        return dict(spec=asdict(self.spec), n=len(self.y), start=self.raw.index[0].isoformat(),end=self.raw.index[-1].isoformat(),converged=self.converged,rho=self.rho.tolist(),messages=self.messages,interval='95% conditional prediction interval. Fixed parameters and future X. No parameter, exogenous-path, or model uncertainty.',fidelity='Model-family reconstruction; proprietary BI specification/coefficients unknown')

def fit_model(frame, spec):
    if spec.family=='growth':
        from .growth import fit_growth_model
        return fit_growth_model(frame, spec)
    if spec.family=='sarimax' and (spec.order[1] or spec.seasonal_order[1]):
        raise ValueError('Integrated errors require a separate constant/drift identification design; this prototype supports d=D=0. Use an explicit differenced/ECM specification instead of toggling d blindly.')
    original=prepare_frame(frame, spec)
    raw=frame.loc[original.index,[spec.target]+list(spec.features)].copy()
    y=original[spec.target].astype(float)
    if spec.transformation=='log':
        if (y<=0).any(): raise ValueError('Target must be positive for log model')
        y=np.log(y)
    z=transform_features(original,spec)
    mean=z.mean();scale=z.std(ddof=0)
    if (scale<1e-12).any(): raise ValueError('Constant features: '+','.join(scale[scale<1e-12].index))
    X=(z-mean)/scale;X.insert(0,'const',1.0)
    if np.linalg.matrix_rank(X.to_numpy())<len(X.columns): raise ValueError('Rank-deficient design matrix')
    messages=[];rho=np.array([])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if spec.family=='glsar':
            model=sm.GLSAR(y,X,rho=spec.ar_order,missing='raise')
            result=model.iterative_fit(maxiter=60,rtol=1e-8)
            rho=np.asarray(model.rho,dtype=float)
            converged=bool(getattr(result,'converged',True))
            if not ArmaProcess(np.r_[1,-rho],[1]).isstationary:
                raise ValueError('Estimated GLSAR errors are not stationary; re-specify rather than clipping AR coefficients')
        elif spec.family=='sarimax':
            model=SARIMAX(y,exog=X,order=spec.order,seasonal_order=spec.seasonal_order,trend='n',enforce_stationarity=True,enforce_invertibility=True,missing='raise')
            result=model.fit(disp=False,maxiter=350,method='lbfgs')
            converged=bool(result.mle_retvals.get('converged',False))
            if not converged: messages.append('Optimizer did not converge. Forecasts withheld.')
        else: raise ValueError('Unknown model family')
        messages += [str(w.message) for w in caught]
    return FittedModel(spec,raw,y,X,mean,scale,result,rho,messages,converged)
