"""Growth-rate (Δlog) challenger with price momentum.

    Δlog y_t = c + φ·Δlog y_{t-1} + ΔX_t·γ + ε_t

ΔX uses log differences for positive level predictors (LOG_FEATURES) and simple
differences for rate-type predictors (percentage points). Estimated by OLS with
Newey–West (HAC) standard errors. No long-run level relation is assumed, so the
model remains valid when the levels are not cointegrated.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .data import prepare_frame
from .specs import LOG_FEATURES


def growth_transform(levels: pd.DataFrame, features) -> pd.DataFrame:
    """Month-on-month change of each predictor on its modelling scale."""
    out = {}
    for col in features:
        x = levels[col].astype(float)
        if col in LOG_FEATURES:
            if (x <= 0).any():
                raise ValueError('Cannot log non-positive ' + col)
            out['d_' + col] = np.log(x).diff()
        else:
            out['d_' + col] = x.diff()
    return pd.DataFrame(out, index=levels.index)


@dataclass
class FittedGrowthModel:
    spec: object
    raw: pd.DataFrame
    y: pd.Series
    X: pd.DataFrame
    result: object
    hac_lags: int
    converged: bool = True
    messages: list = field(default_factory=list)
    # No AR error process: persistence enters the mean equation through the lagged growth term.
    rho: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def lag_term(self) -> str:
        return 'lag_d_' + self.spec.target

    @property
    def momentum(self) -> float:
        return float(self.result.params[self.lag_term])

    def coefficients(self) -> pd.DataFrame:
        r = self.result
        return pd.DataFrame({
            'term': r.params.index,
            'coefficient_per_transformed_unit': r.params.to_numpy(),
            'standard_error_per_transformed_unit': r.bse.to_numpy(),
            'p_value': r.pvalues.to_numpy(),
            'coefficient_scale': 'Δlog target per Δ(log) predictor; fed_rate per 1pp',
            'note': 'HAC standard errors; p-values are exploratory, not post-selection-valid',
        })

    def forecast(self, future_raw: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        expected = pd.date_range(self.raw.index[-1] + pd.offsets.MonthBegin(1), periods=len(future_raw), freq='MS')
        if not future_raw.index.equals(expected.rename(future_raw.index.name)):
            raise ValueError('Future inputs must begin next month and have consecutive monthly rows')
        features = list(self.spec.features)
        joined = pd.concat([self.raw[features].iloc[[-1]], future_raw[features]])
        dX = growth_transform(joined, features).iloc[1:]
        if not np.isfinite(dX.to_numpy()).all():
            raise ValueError('Invalid or missing future inputs')
        params = self.result.params
        gamma = params[['d_' + f for f in features]].to_numpy()
        c, phi = float(params['const']), self.momentum
        target = self.raw[self.spec.target]
        g_prev = float(np.log(target).diff().iloc[-1])
        growth_path = []
        for dx in dX.to_numpy():
            g_prev = c + phi * g_prev + float(dx @ gamma)
            growth_path.append(g_prev)
        mu = np.log(float(target.iloc[-1])) + np.cumsum(growth_path)
        # A shock at T+k moves the cumulative log change at T+h by sum_{i=0}^{h-k} φ^i.
        horizon_weights = np.cumsum(phi ** np.arange(len(growth_path)))
        variance = float(self.result.scale) * np.cumsum(horizon_weights ** 2)
        z = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(variance)
        return pd.DataFrame({
            'prediction_median': np.exp(mu),
            'prediction_mean': np.exp(mu + 0.5 * variance),
            'lower_95': np.exp(mu - z * se),
            'upper_95': np.exp(mu + z * se),
            'model_scale_mean': mu,
            'model_scale_variance': variance,
        }, index=future_raw.index)

    def simulate(self, future_raw: pd.DataFrame, n_paths: int = 100, seed: int = 42) -> pd.DataFrame:
        if not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError('n_paths must be a positive integer')
        forecast = self.forecast(future_raw)
        residuals = np.asarray(self.result.resid, dtype=float)
        if residuals.size == 0 or not np.isfinite(residuals).all():
            raise ValueError('Simulation requires finite fitted growth residuals')
        residuals = residuals - residuals.mean()
        innovations = np.random.default_rng(seed).choice(residuals, size=(len(forecast), n_paths))
        growth_error = np.zeros(n_paths)
        log_price_error = np.zeros(n_paths)
        paths = np.empty_like(innovations)
        for step, mean_log_price in enumerate(forecast['model_scale_mean']):
            growth_error = self.momentum * growth_error + innovations[step]
            log_price_error += growth_error
            paths[step] = np.exp(mean_log_price + log_price_error)
        return pd.DataFrame(paths, index=forecast.index,
                            columns=[f'path_{number + 1:03d}' for number in range(n_paths)])

    def metadata(self) -> dict:
        return dict(
            spec=asdict(self.spec), n=len(self.y), start=self.y.index[0].isoformat(), end=self.y.index[-1].isoformat(),
            converged=self.converged, momentum=self.momentum, hac_lags=self.hac_lags, messages=self.messages,
            interval='95% conditional prediction interval on the log scale. Fixed parameters and future X.',
            fidelity='Robustness challenger, not part of the disclosed BI specification',
        )


def fit_growth_model(frame: pd.DataFrame, spec, hac_lags: int = 6) -> FittedGrowthModel:
    if spec.feature_lag or spec.seasonality != 'none':
        raise ValueError('The growth challenger supports contemporaneous X without seasonal terms only')
    raw = prepare_frame(frame, spec)
    target = raw[spec.target].astype(float)
    if (target <= 0).any():
        raise ValueError('Target must be positive for a log-growth model')
    dy = np.log(target).diff().rename('d_' + spec.target)
    regressors = pd.concat([dy.shift(1).rename('lag_d_' + spec.target), growth_transform(raw, spec.features)], axis=1)
    data = pd.concat([dy, regressors], axis=1).dropna()
    y = data[dy.name]
    X = sm.add_constant(data.drop(columns=dy.name))
    result = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    return FittedGrowthModel(spec, raw, y, X, result, hac_lags)
