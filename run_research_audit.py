"""Build empirical integration and model-comparison outputs from supplied data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hkproperty.data import read_monthly
from hkproperty.diagnostics import residual_diagnostics, stationarity_table, vif_table
from hkproperty.extensions import fit_home_ecm
from hkproperty.models import fit_model
from hkproperty.research import HOME_RESEARCH_SPECS, available_research_specs
from hkproperty.validation import PSEUDO_OOS, common_origin_comparison, rolling_evaluate


DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
BASE_COLUMNS = ["home_price", "fed_rate", "hsi", "secondary_sales", "home_rent", "cnyhkd"]


def is_monthly_contiguous(index: pd.DatetimeIndex) -> bool:
    if len(index) < 2:
        return True
    values = index.to_period("M").astype(int)
    return bool(np.diff(values).tolist() == [1] * (len(values) - 1))


def integration_decision(column: str, common_n: int, contiguous: bool, last: pd.Timestamp | None):
    if column in BASE_COLUMNS:
        return "baseline", "Use in the disclosed Home reconstruction."
    if column.startswith("alt__rvd_home_price__"):
        return "validation_y", "Same price construct as the target; use as an alternative Y or cross-source validation, never as X."
    if "lever_" in column:
        return "not_usable", "Only a 2026-10 snapshot is available, after the current target sample ends."
    if "is_complete_month" in column or column.endswith("snapshot_complete"):
        return "availability_flag", "Join to its source measure and filter incomplete snapshots; do not use as an economic predictor."
    if column in {"alt__secondary_agreements"}:
        return "replace_and_compare", "Compare as a replacement for Midland secondary sales on the common 2011+ sample."
    if column in {
        "alt__residential_agreements",
        "alt__residential_value_hkd_mn",
        "alt__primary_agreements",
        "alt__secondary_value_hkd_mn",
    }:
        return "candidate_x", "Test one variable or one pre-declared group at a time on the common 2011+ sample."
    if column.endswith("_mom_pct") or column.endswith("_yoy_pct"):
        return "alternate_transform", "Use in a changes specification instead of mixing mechanically with the source level."
    if column.startswith("alt__restaurant_") or column == "alt__gov_active_advertisements":
        return "defer", "Short 2020+ history contains internal missing months; current contiguous-sample estimator cannot use it without a separate missingness design."
    if "jobboard__csb" in column or "jobposting__csb" in column:
        return "short_sample_only", "At most 80 months from 2020; evaluate separately with completeness flags and do not compare directly with the long baseline."
    if common_n == 0 or last is None:
        return "not_usable", "No overlap with the current Home target sample."
    if not contiguous:
        return "defer", "Overlap is not a contiguous monthly block."
    return "review", "Definition and release timing require manual review before modeling."


def build_variable_audit() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "extension_model_input.csv", parse_dates=["date"]).set_index("date")
    dictionary = pd.read_csv(DATA_DIR / "extension_feature_dictionary.csv")
    rows = []
    for record in dictionary.to_dict("records"):
        column = record["column"]
        required = list(dict.fromkeys([*BASE_COLUMNS, column]))
        common = raw[required].dropna()
        first = common.index.min() if len(common) else None
        last = common.index.max() if len(common) else None
        contiguous = is_monthly_contiguous(common.index)
        decision, reason = integration_decision(column, len(common), contiguous, last)
        rows.append(
            {
                "column": column,
                "extension_role": record.get("extension_role"),
                "source": record.get("source"),
                "common_home_n": len(common),
                "common_first": first.strftime("%Y-%m") if first is not None else None,
                "common_last": last.strftime("%Y-%m") if last is not None else None,
                "contiguous_common_sample": contiguous,
                "observed_values": int(raw[column].notna().sum()),
                "unique_values": int(raw[column].nunique(dropna=True)),
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_model_comparison(frame: pd.DataFrame):
    available = available_research_specs(frame)
    fitted_rows = []
    prediction_groups: dict[str, list[pd.DataFrame]] = {}
    failed_rows = []
    for key, item in available.items():
        fitted = fit_model(frame, item.spec)
        diagnostics = residual_diagnostics(fitted)
        vifs = vif_table(fitted.X)
        ljung_box = diagnostics["ljung_box"]
        fitted_rows.append(
            {
                "specification": key,
                "label": item.label,
                "comparison_group": item.comparison_group,
                "evidence_class": item.evidence_class,
                "n": len(fitted.y),
                "first_month": fitted.raw.index.min().strftime("%Y-%m"),
                "last_month": fitted.raw.index.max().strftime("%Y-%m"),
                "rho_1": float(fitted.rho[0]) if len(fitted.rho) else np.nan,
                "rho_sum": float(np.sum(fitted.rho)) if len(fitted.rho) else np.nan,
                "converged": bool(fitted.converged),
                "aic": float(getattr(fitted.result, "aic", np.nan)),
                "bic": float(getattr(fitted.result, "bic", np.nan)),
                "ljung_box_pvalue_lag12": float(ljung_box.loc[12, "lb_pvalue"]),
                "arch_lm_pvalue": diagnostics["arch_lm_pvalue"],
                "design_condition_number": diagnostics["design_condition_number"],
                "max_vif": float(vifs["vif"].max()),
                "description": item.description,
            }
        )
        predictions, failures = rolling_evaluate(frame, item.spec, **PSEUDO_OOS)
        predictions_groups = prediction_groups.setdefault(item.comparison_group, [])
        predictions_groups.append(predictions)
        for failure in failures.to_dict("records"):
            failed_rows.append({"specification": key, **failure})

    score_frames = []
    detailed_frames = []
    for group, results in prediction_groups.items():
        scores, details = common_origin_comparison(results)
        scores.insert(0, "comparison_group", group)
        details.insert(0, "comparison_group", group)
        score_frames.append(scores)
        detailed_frames.append(details)
    return (
        pd.DataFrame(fitted_rows),
        pd.concat(score_frames, ignore_index=True),
        pd.concat(detailed_frames, ignore_index=True),
        pd.DataFrame(failed_rows, columns=["specification", "origin", "error"]),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit = build_variable_audit()
    audit.to_csv(OUTPUT_DIR / "variable_integration_audit.csv", index=False)

    frame = read_monthly(DATA_DIR / "extension_model_input.csv")
    stationarity_table(frame[BASE_COLUMNS]).to_csv(OUTPUT_DIR / "home_stationarity.csv", index=False)
    fits, scores, details, failures = build_model_comparison(frame)
    fits.to_csv(OUTPUT_DIR / "model_specification_summary.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "model_comparison_scores.csv", index=False)
    details.to_csv(OUTPUT_DIR / "model_comparison_predictions.csv", index=False)
    failures.to_csv(OUTPUT_DIR / "model_comparison_failures.csv", index=False)

    try:
        ecm = fit_home_ecm(frame[BASE_COLUMNS].dropna())
        ecm_status = f"estimated; cointegration p={ecm.cointegration_pvalue:.4g}"
    except ValueError as exc:
        ecm_status = f"withheld; {exc}"

    status = {
        "data_mode": "USER_SUPPLIED_LATEST_VINTAGE",
        "bqnt_runtime_required": False,
        "office_model_available": False,
        "office_blocker": "office_rent target is absent; office predictors alone cannot identify an office-rent model",
        "home_complete_months": int(frame[BASE_COLUMNS].dropna().shape[0]),
        "home_first_complete": frame[BASE_COLUMNS].dropna().index.min().strftime("%Y-%m"),
        "home_last_complete": frame[BASE_COLUMNS].dropna().index.max().strftime("%Y-%m"),
        "specifications_fitted": list(fits["specification"]),
        "failed_validation_origins": len(failures),
        "ecm_status": ecm_status,
        "interpretation": "Latest-vintage pseudo-OOS, not a release-vintage PIT backtest and not causal.",
    }
    (OUTPUT_DIR / "research_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
