"""Research specification registry and scenario decomposition for the Home model.

The registry keeps baseline assumptions and empirical challengers in one place so
the notebook, audit script, and Streamlit app do not silently use different
specifications.
"""
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import pandas as pd

from .models import fit_model
from .scenarios import apply_shocks, baseline_path
from .specs import GROWTH, HOME, ModelSpec


@dataclass(frozen=True)
class ResearchSpecification:
    key: str
    label: str
    description: str
    spec: ModelSpec
    evidence_class: str
    comparison_group: str


HOME_RESEARCH_SPECS = {
    "baseline": ResearchSpecification(
        key="baseline",
        label="Level model · GLSAR(1)",
        description="Price-index levels on contemporaneous factor levels, with AR(1) errors.",
        spec=replace(HOME, key="baseline"),
        evidence_class="level specification",
        comparison_group="method",
    ),
    "growth": ResearchSpecification(
        key="growth",
        label="Growth model · Δlog + momentum",
        description="Monthly log price growth on its own lag and factor changes; no long-run equilibrium relation is imposed.",
        spec=GROWTH,
        evidence_class="specification challenger",
        comparison_group="method",
    ),
    "ar2": ResearchSpecification(
        key="ar2",
        label="AR-order challenger · GLSAR(2)",
        description="Adds a second autoregressive error lag because AR(1) leaves substantial innovation dependence.",
        spec=replace(HOME, key="ar2", ar_order=2),
        evidence_class="error-structure challenger",
        comparison_group="method",
    ),
    "fourier1": ResearchSpecification(
        key="fourier1",
        label="Seasonal challenger · Fourier(1)",
        description="Adds one annual sine/cosine pair to the mean equation; the source does not reveal where seasonality enters.",
        spec=replace(HOME, key="fourier1", seasonality="fourier1"),
        evidence_class="method challenger",
        comparison_group="method",
    ),
    "lag1": ResearchSpecification(
        key="lag1",
        label="Information-lag challenger · X(t−1)",
        description="Uses one-month-lagged predictors to reduce same-month timing ambiguity.",
        spec=replace(HOME, key="lag1", feature_lag=1),
        evidence_class="method challenger",
        comparison_group="method",
    ),
    "log": ResearchSpecification(
        key="log",
        label="Log-scale challenger",
        description="Logs the positive target and designated positive predictors; Fed rate remains in percentage points.",
        spec=replace(HOME, key="log", transformation="log"),
        evidence_class="method challenger",
        comparison_group="method",
    ),
    "baseline_2011": ResearchSpecification(
        key="baseline_2011",
        label="Level model · common 2011 sample",
        description="Baseline re-estimated from 2011-01 so transaction extensions are compared on the same sample.",
        spec=replace(HOME, key="baseline_2011", start="2011-01-01"),
        evidence_class="comparison control",
        comparison_group="transactions",
    ),
    "alt_secondary": ResearchSpecification(
        key="alt_secondary",
        label="Replace Midland sales with Land Registry secondary agreements",
        description="Substitutes a differently defined transaction series; it is not treated as the same measurement.",
        spec=ModelSpec(
            "alt_secondary",
            "home_price",
            ("fed_rate", "hsi", "alt__secondary_agreements", "home_rent", "cnyhkd"),
            "2011-01-01",
            "glsar",
        ),
        evidence_class="data-source challenger",
        comparison_group="transactions",
    ),
    "transactions_plus": ResearchSpecification(
        key="transactions_plus",
        label="Add total residential agreements",
        description="Adds one broad transaction-activity proxy while retaining the disclosed five predictors.",
        spec=ModelSpec(
            "transactions_plus",
            "home_price",
            HOME.features + ("alt__residential_agreements",),
            "2011-01-01",
            "glsar",
        ),
        evidence_class="variable challenger",
        comparison_group="transactions",
    ),
    "value_plus": ResearchSpecification(
        key="value_plus",
        label="Add residential consideration value",
        description="Adds aggregate transaction value as a separate activity/intensity proxy.",
        spec=ModelSpec(
            "value_plus",
            "home_price",
            HOME.features + ("alt__residential_value_hkd_mn",),
            "2011-01-01",
            "glsar",
        ),
        evidence_class="variable challenger",
        comparison_group="transactions",
    ),
}


FEATURE_LABELS = {
    "fed_rate": "Fed effective rate",
    "hsi": "Hang Seng Index",
    "secondary_sales": "Midland secondary sales",
    "home_rent": "Private domestic rent index",
    "cnyhkd": "CNY/HKD",
    "alt__secondary_agreements": "Land Registry secondary agreements",
    "alt__residential_agreements": "Land Registry residential agreements",
    "alt__residential_value_hkd_mn": "Residential consideration (HKD mn)",
}


def available_research_specs(frame: pd.DataFrame) -> dict[str, ResearchSpecification]:
    """Return only specifications whose target and predictors exist in ``frame``."""
    available = {}
    for key, item in HOME_RESEARCH_SPECS.items():
        required = {item.spec.target, *item.spec.features}
        if required.issubset(frame.columns):
            available[key] = item
    return available


def fit_research_spec(frame: pd.DataFrame, key: str):
    available = available_research_specs(frame)
    if key not in available:
        raise KeyError(f"Specification {key!r} is unavailable for the supplied columns")
    return fit_model(frame, available[key].spec)


def scenario_path(fitted, shocks: dict[str, float], horizon: int = 1, shape: str = "permanent"):
    """Return baseline/scenario forecasts over ``horizon`` months and shock attribution at T+horizon.

    Individual effects are one-at-a-time changes on the displayed prediction
    scale. For log and growth specifications, their sum can differ from the
    joint nonlinear change; that remainder is reported explicitly.
    """
    base_x = baseline_path(fitted.raw, fitted.spec.features, horizon=horizon, method="last")
    scenario_x = apply_shocks(base_x, shocks, shape=shape)
    base_fc = fitted.forecast(base_x)
    scenario_fc = fitted.forecast(scenario_x)
    base_value = float(base_fc["prediction_median"].iloc[-1])
    scenario_value = float(scenario_fc["prediction_median"].iloc[-1])
    effects = []
    for feature in fitted.spec.features:
        shock = float(shocks.get(feature, 0.0))
        isolated_x = apply_shocks(base_x, {feature: shock}, shape=shape)
        isolated_fc = fitted.forecast(isolated_x)
        effect = float(isolated_fc["prediction_median"].iloc[-1]) - base_value
        effects.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "shock": shock,
                "effect": effect,
            }
        )
    attributed = float(sum(row["effect"] for row in effects))
    remainder = scenario_value - base_value - attributed
    if abs(remainder) > 1e-10:
        effects.append(
            {
                "feature": "interaction_remainder",
                "label": "Nonlinear interaction remainder",
                "shock": np.nan,
                "effect": remainder,
            }
        )
    return base_x, scenario_x, base_fc, scenario_fc, pd.DataFrame(effects)


def one_step_scenario(fitted, shocks: dict[str, float]):
    """T+1 special case of :func:`scenario_path` with a permanent shock."""
    return scenario_path(fitted, shocks, horizon=1, shape="permanent")


def response_curve(fitted, feature: str, shock_grid, horizon: int = 1, shape: str = "permanent") -> pd.DataFrame:
    """Evaluate the T+horizon prediction over a feature-specific shock grid."""
    base_x = baseline_path(fitted.raw, fitted.spec.features, horizon=horizon, method="last")
    rows = []
    for shock in shock_grid:
        future_x = apply_shocks(base_x, {feature: float(shock)}, shape=shape)
        forecast = fitted.forecast(future_x)
        rows.append(
            {
                "feature": feature,
                "shock": float(shock),
                "prediction": float(forecast["prediction_median"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)
