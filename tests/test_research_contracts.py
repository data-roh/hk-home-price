from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from hkproperty.demo import synthetic_data
from hkproperty.data import read_monthly,prepare_frame,to_monthly
from hkproperty.specs import HOME,OFFICE,GROWTH
from hkproperty.models import fit_model,transform_features
from hkproperty.scenarios import baseline_path,apply_shocks
from hkproperty.validation import rolling_evaluate,common_origin_comparison
from hkproperty.extensions import fit_home_ecm

@pytest.fixture(scope='module')
def df():return synthetic_data()

@pytest.fixture(scope='module')
def home(df):return fit_model(df,HOME)

@pytest.fixture(scope='module')
def office(df):return fit_model(df,OFFICE)

def test_missing_month_is_not_compressed(df):
    broken=df.copy();broken.loc[broken.index[100],'hsi']=np.nan
    with pytest.raises(ValueError,match='Internal missing months'):prepare_frame(broken,HOME)

def test_duplicate_calendar_month_rejected():
    with pytest.raises(ValueError,match='Multiple rows'):read_monthly(StringIO('date,home_price\n2020-01-01,100\n2020-01-31,101\n'))

def test_partial_month_excluded_and_no_forward_fill():
    x=pd.DataFrame({'series':['hsi','hsi'],'date':['2026-08-31','2026-10-10'],'value':[100,120]})
    m=to_monthly(x,'2026-10-17');assert len(m)==1 and m.iloc[-1].hsi==100

@pytest.mark.parametrize('key',['home','office'])
def test_zero_shock_identity_and_intervals(key,request):
    f=request.getfixturevalue(key);p=baseline_path(f.raw,f.spec.features,12);a=f.forecast(p);b=f.forecast(apply_shocks(p,{}))
    np.testing.assert_allclose(a,b);assert np.isfinite(a.to_numpy()).all();assert (a.lower_95<a.prediction_median).all();assert (a.upper_95>a.prediction_median).all()

def test_ar_errors_are_not_an_ardl_multiplier(home):
    p=baseline_path(home.raw,HOME.features,12);a=home.forecast(p);b=home.forecast(apply_shocks(p,{'fed_rate':1.0}))
    beta=home.coefficients().set_index('term').loc['fed_rate','coefficient_per_transformed_unit']
    # A persistent 1pp shock has difference beta, NOT beta/(1-rho).
    np.testing.assert_allclose(b.prediction_median-a.prediction_median,beta,rtol=1e-8,atol=1e-8)

def test_rates_are_percentage_points_not_percent(home):
    p=baseline_path(home.raw,HOME.features,3);s=apply_shocks(p,{'fed_rate':1,'hsi':10})
    np.testing.assert_allclose(s.fed_rate-p.fed_rate,1);np.testing.assert_allclose(s.hsi/p.hsi,1.1)

def test_no_fake_seasonal_forecast_phase(df):
    spec=replace(HOME,seasonality='fourier2');a=transform_features(df.iloc[:100],spec);b=transform_features(df.iloc[100:112],spec)
    for col in ['sin1','cos1','sin2','cos2']:
        combined=transform_features(df.iloc[:112],spec);np.testing.assert_allclose(b[col],combined.loc[b.index,col])

def test_scaler_uses_training_only(df):
    train=df.iloc[:240];a=fit_model(train,HOME);altered=df.copy();altered.iloc[240:,altered.columns.get_loc('hsi')]*=99;b=fit_model(altered.iloc[:240],HOME)
    np.testing.assert_allclose(a.result.params,b.result.params);np.testing.assert_allclose(a.mean,b.mean)

def test_log_inverse_mean_vs_median(df):
    f=fit_model(df,replace(HOME,transformation='log'));p=f.forecast(baseline_path(f.raw,f.spec.features,6));assert (p.prediction_mean>=p.prediction_median).all()

def test_feature_lag_blocks_first_future_shock(df):
    f=fit_model(df,replace(HOME,feature_lag=1));base=baseline_path(f.raw,f.spec.features,3);a=f.forecast(base);b=f.forecast(apply_shocks(base,{'fed_rate':1}))
    assert abs(a.prediction_median.iloc[0]-b.prediction_median.iloc[0])<1e-7
    assert abs(a.prediction_median.iloc[1]-b.prediction_median.iloc[1])>1e-5

def test_future_target_cannot_change_origin_forecast(df):
    # One fixed origin; hold out X with a training-only path.
    short=df.iloc[:252].copy();a,e=rolling_evaluate(short,HOME,horizons=(1,3),min_train=240,step=20,last_origins=1)
    changed=short.copy();changed.loc[changed.index[240:],'home_price']*=5;b,e2=rolling_evaluate(changed,HOME,horizons=(1,3),min_train=240,step=20,last_origins=1)
    np.testing.assert_allclose(a.prediction,b.prediction);assert not np.allclose(a.actual,b.actual)

def test_no_interpolation_of_invalid_scenario(home):
    p=baseline_path(home.raw,HOME.features,3)
    with pytest.raises(ValueError,match='positive'):apply_shocks(p,{'hsi':-101})

def test_sarimax_contains_disclosed_arma_terms(office):
    assert office.result.model.order==(2,0,1);assert office.result.model.seasonal_order==(0,0,0,0);assert office.converged

def test_ecm_cointegration_gate():
    # Independent random walks, pre-generated fixed seed: no forced ECM fit.
    r=np.random.default_rng(123);n=260;d=pd.DataFrame({'home_price':np.exp(5+np.cumsum(r.normal(0,.01,n))),'secondary_sales':np.exp(8+np.cumsum(r.normal(0,.01,n))),'fed_rate':r.normal(3,.1,n),'home_rent':np.exp(4+np.cumsum(r.normal(0,.01,n))),'hsi':np.exp(9+np.cumsum(r.normal(0,.01,n))),'cnyhkd':np.exp(np.cumsum(r.normal(0,.01,n)))},index=pd.date_range('2000-01-01',periods=n,freq='MS'))
    result=None
    try:result=fit_home_ecm(d)
    except ValueError as e:assert 'Cointegration not supported' in str(e)
    if result is not None:assert result.cointegration_pvalue<.05


def test_integrated_errors_require_explicit_drift_design(df):
    with pytest.raises(ValueError,match='constant/drift identification'):
        fit_model(df,replace(OFFICE,order=(2,1,1)))


def test_growth_zero_shock_identity_and_intervals(df):
    g=fit_model(df,GROWTH);p=baseline_path(g.raw,GROWTH.features,12);a=g.forecast(p);b=g.forecast(apply_shocks(p,{}))
    np.testing.assert_allclose(a,b);assert (a.lower_95<a.prediction_median).all() and (a.upper_95>a.prediction_median).all()
    assert (np.diff(a.upper_95-a.lower_95)>0).all()


def test_growth_rate_shock_builds_through_momentum(df):
    # A permanent 1pp shock changes ΔX only at T+1; the log gap is γ·Σφ^j, not an immediate full effect.
    g=fit_model(df,GROWTH);p=baseline_path(g.raw,GROWTH.features,6);a=g.forecast(p);b=g.forecast(apply_shocks(p,{'fed_rate':1.0}))
    gamma=g.result.params['d_fed_rate'];phi=g.momentum
    np.testing.assert_allclose(np.log(b.prediction_median/a.prediction_median),gamma*np.cumsum(phi**np.arange(6)),rtol=1e-10)


def test_growth_simulation_reproducible_and_pairs_shocks(df):
    fitted=fit_model(df,GROWTH)
    future=baseline_path(fitted.raw,GROWTH.features,6)
    paths=fitted.simulate(future)
    assert paths.shape==(6,100) and paths.index.equals(future.index)
    pd.testing.assert_frame_equal(paths,fitted.simulate(future))
    assert not paths.equals(fitted.simulate(future,seed=43))
    assert np.isfinite(paths.to_numpy()).all() and (paths.to_numpy()>0).all()
    shocked=apply_shocks(future,{'fed_rate':-0.25})
    paired=fitted.simulate(shocked).to_numpy()/paths.to_numpy()
    expected=(fitted.forecast(shocked).prediction_median/fitted.forecast(future).prediction_median).to_numpy()
    np.testing.assert_allclose(paired,np.broadcast_to(expected[:,None],paired.shape))


def test_growth_simulation_propagates_monthly_errors(df):
    fitted=fit_model(df,GROWTH)
    future=baseline_path(fitted.raw,GROWTH.features,3)
    residuals=np.asarray(fitted.result.resid)
    residuals=residuals-residuals.mean()
    innovations=np.random.default_rng(42).choice(residuals,size=(3,100))
    paths=fitted.simulate(future)
    errors=np.log(paths.to_numpy())-fitted.forecast(future).model_scale_mean.to_numpy()[:,None]
    np.testing.assert_allclose(errors[0],innovations[0],atol=1e-12)
    np.testing.assert_allclose(errors[1],(1+fitted.momentum)*innovations[0]+innovations[1],atol=1e-12)


def test_growth_simulation_zero_noise_and_invalid_path_count(df):
    fitted=fit_model(df,GROWTH)
    result=SimpleNamespace(params=fitted.result.params,scale=0.,resid=np.zeros(len(fitted.y)))
    zero_noise=replace(fitted,result=result)
    future=baseline_path(fitted.raw,GROWTH.features,3)
    paths=zero_noise.simulate(future)
    expected=zero_noise.forecast(future).prediction_median.to_numpy()[:,None]
    np.testing.assert_allclose(paths,np.broadcast_to(expected,paths.shape))
    for invalid in [0,-1,1.5]:
        with pytest.raises(ValueError,match='positive integer'):
            fitted.simulate(future,n_paths=invalid)


def test_growth_runs_in_rolling_evaluation(df):
    pred,errors=rolling_evaluate(df,GROWTH,horizons=(1,3),min_train=240,step=12,last_origins=2)
    assert errors.empty and set(pred.method)=={'growth_none','random_walk','seasonal_naive'}


@pytest.mark.parametrize('spec',[HOME,GROWTH])
def test_scenario_path_attribution_closes_at_horizon(df,spec):
    from hkproperty.research import scenario_path,one_step_scenario
    f=fit_model(df,spec);shocks={'fed_rate':-1.0,'hsi':10.0}
    _,_,base,scen,attr=scenario_path(f,shocks,horizon=12,shape='ramp')
    assert len(base)==12
    np.testing.assert_allclose(attr.effect.sum(),scen.prediction_median.iloc[-1]-base.prediction_median.iloc[-1],atol=1e-8)
    one=one_step_scenario(f,shocks);path=scenario_path(f,shocks,horizon=1)
    np.testing.assert_allclose(one[3].prediction_median,path[3].prediction_median)
