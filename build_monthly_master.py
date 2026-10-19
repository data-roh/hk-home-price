from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FINAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FINAL_DIR
WORKSPACE_DIR = FINAL_DIR
BQNT_PATH = FINAL_DIR / "data" / "source" / "bqnt" / "monthly_data.csv"
BQNT_DICTIONARY_PATH = FINAL_DIR / "data" / "source" / "bqnt" / "data_dictionary.csv"
ALT_DIR = FINAL_DIR / "data" / "source" / "altdata"
ALT_PANEL_PATH = ALT_DIR / "regression_monthly_panel.csv"
ALT_PRICE_PATH = ALT_DIR / "home_price_monthly.csv"
ALT_LANDREG_PATH = ALT_DIR / "landreg_observations.csv"
ALT_RESTAURANT_PATH = ALT_DIR / "restaurant_district_snapshots.csv"
ALT_JOBS_PATH = ALT_DIR / "job_board_snapshots.csv"
ALT_JOB_POSTINGS_PATH = ALT_DIR / "job_posting_snapshots.csv"
ALT_VACANCIES_PATH = ALT_DIR / "official_vacancies_quarterly.csv"
ALT_LABOR_PATH = ALT_DIR / "official_labor_annual.csv"
DATA_DIR = FINAL_DIR / "data"
EXTENSION_ROLES = {
    "alt__rvd_home_price__all_classes": "validation_target_not_predictor",
    "alt__residential_agreements": "candidate_x",
    "alt__residential_agreements_mom_pct": "candidate_x_alternative_transform",
    "alt__residential_agreements_yoy_pct": "candidate_x_alternative_transform",
    "alt__residential_value_hkd_mn": "candidate_x",
    "alt__primary_agreements": "candidate_x",
    "alt__secondary_agreements": "candidate_x_alternative_to_midland_series",
    "alt__secondary_value_hkd_mn": "candidate_x",
    "alt__restaurant_license_stock": "exploratory_candidate_x",
    "alt__restaurant_snapshot_complete": "availability_flag",
    "alt__gov_active_advertisements": "exploratory_candidate_x",
    "alt__jobboard__csb__active_hk_postings": "exploratory_candidate_x",
    "alt__jobboard__csb__is_complete_month": "availability_flag",
    "alt__jobboard__lever_lalamove__active_hk_postings": "exploratory_candidate_x",
    "alt__jobboard__lever_lalamove__is_complete_month": "availability_flag",
    "alt__jobboard__lever_crypto__active_hk_postings": "exploratory_candidate_x",
    "alt__jobboard__lever_crypto__is_complete_month": "availability_flag",
    "alt__jobposting__csb__observed_postings": "exploratory_candidate_x",
    "alt__jobposting__csb__is_complete_month": "availability_flag",
    "alt__jobposting__lever_lalamove__observed_postings": "exploratory_candidate_x",
    "alt__jobposting__lever_lalamove__is_complete_month": "availability_flag",
    "alt__jobposting__lever_crypto__observed_postings": "exploratory_candidate_x",
    "alt__jobposting__lever_crypto__is_complete_month": "availability_flag",
}


def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.to_period("M").dt.to_timestamp()


def column_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_slug(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{column_slug(value)[:48]}_{digest}"


def load_monthly_file(path: Path, date_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"{path} must contain {date_column!r}")
    frame["date"] = month_start(frame.pop(date_column))
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate monthly rows in {path}")
    return frame


def load_landreg_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_LANDREG_PATH)
    frame["date"] = month_start(frame.pop("period"))
    if frame.duplicated(["date", "description"]).any():
        raise ValueError("Land Registry data must have one row per month and description")
    units = frame.pivot(index="date", columns="description", values="units")
    consideration = frame.pivot(index="date", columns="description", values="consideration_hkd_mn")
    units.columns = [f"alt__landreg__{unique_slug(column)}__units" for column in units.columns]
    consideration.columns = [f"alt__landreg__{unique_slug(column)}__consideration_hkd_mn" for column in consideration.columns]
    return pd.concat([units, consideration], axis=1).reset_index()


def load_restaurant_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_RESTAURANT_PATH)
    frame["date"] = month_start(frame.pop("snapshot_date"))
    monthly = frame.groupby("date").agg(
        district_count=("district18", "nunique"),
        complete_flags=("is_complete_month", lambda values: pd.to_numeric(values, errors="coerce").eq(1).all()),
        license_count=("license_count", "sum"),
        appeared=("appeared_since_previous", lambda values: values.sum(min_count=1)),
        disappeared=("disappeared_since_previous", lambda values: values.sum(min_count=1)),
    )
    monthly["is_complete"] = monthly["complete_flags"] & monthly["district_count"].eq(18)
    for column in ["license_count", "appeared", "disappeared"]:
        monthly.loc[~monthly["is_complete"], column] = pd.NA
    monthly = monthly.rename(
        columns={
            "district_count": "alt__restaurant_district_count",
            "is_complete": "alt__restaurant_snapshot_complete",
            "license_count": "alt__restaurant_license_stock_district_sum",
            "appeared": "alt__restaurant_license_appeared_district_sum",
            "disappeared": "alt__restaurant_license_disappeared_district_sum",
        }
    )
    return monthly.drop(columns="complete_flags").reset_index()


def load_job_board_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_JOBS_PATH)
    frame["date"] = month_start(frame.pop("snapshot_date"))
    if frame.duplicated(["date", "source_id"]).any():
        raise ValueError("Job board snapshots must have one row per month and source")
    measures = [
        "active_hk_postings",
        "total_board_postings",
        "posted_same_month_still_visible",
        "is_complete_month",
    ]
    pieces = []
    for source_id, source_frame in frame.groupby("source_id"):
        piece = source_frame.set_index("date")[measures].rename(
            columns=lambda column: f"alt__jobboard__{column_slug(source_id)}__{column}"
        )
        pieces.append(piece)
    return pd.concat(pieces, axis=1).reset_index()


def load_job_posting_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_JOB_POSTINGS_PATH)
    frame["date"] = month_start(frame.pop("snapshot_date"))
    if frame["posting_id"].isna().any():
        raise ValueError("Job posting snapshots contain missing posting_id values")
    grouped = frame.groupby(["date", "source_id"], dropna=False)
    summary = grouped.agg(
        observed_postings=("posting_id", "nunique"),
        employer_count=("employer", "nunique"),
        location_scope_count=("location_match", "nunique"),
        median_monthly_salary_hkd=("salary_month_hkd", "median"),
        is_complete_month=("is_complete_month", lambda values: pd.to_numeric(values, errors="coerce").eq(1).all()),
    ).reset_index()
    pieces = []
    measures = [
        "observed_postings",
        "employer_count",
        "location_scope_count",
        "median_monthly_salary_hkd",
        "is_complete_month",
    ]
    for source_id, source_frame in summary.groupby("source_id"):
        piece = source_frame.set_index("date")[measures].rename(
            columns=lambda column: f"alt__jobposting__{column_slug(source_id)}__{column}"
        )
        pieces.append(piece)
    return pd.concat(pieces, axis=1).reset_index()


def load_official_vacancies_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_VACANCIES_PATH)
    quarters = pd.PeriodIndex(frame.pop("period"), freq="Q")
    frame["date"] = quarters.to_timestamp(how="end").to_period("M").to_timestamp()
    frame["industry_key"] = frame["industry"].map(column_slug)
    if frame.duplicated(["date", "industry_key"]).any():
        raise ValueError("Official vacancy data must have one row per quarter and industry")
    vacancies = frame.pivot(index="date", columns="industry_key", values="vacancies")
    persons = frame.pivot(index="date", columns="industry_key", values="persons_engaged")
    vacancies.columns = [f"alt__official_vacancies__{column}__vacancies" for column in vacancies.columns]
    persons.columns = [f"alt__official_vacancies__{column}__persons_engaged" for column in persons.columns]
    return pd.concat([vacancies, persons], axis=1).reset_index()


def load_official_labor_monthly() -> pd.DataFrame:
    frame = pd.read_csv(ALT_LABOR_PATH)
    frame["date"] = pd.to_datetime(frame["year"].astype(str) + "-12-01")
    frame["metric_key"] = frame["source_id"].map(column_slug) + "__" + frame["metric"].map(column_slug)
    if frame.duplicated(["date", "metric_key"]).any():
        raise ValueError("Annual labor data must have one row per year and metric")
    wide = frame.pivot(index="date", columns="metric_key", values="value")
    wide.columns = [f"alt__official_labor__{column}" for column in wide.columns]
    return wide.reset_index()


def main() -> None:
    required_paths = [
        BQNT_PATH,
        BQNT_DICTIONARY_PATH,
        ALT_PANEL_PATH,
        ALT_PRICE_PATH,
        ALT_LANDREG_PATH,
        ALT_RESTAURANT_PATH,
        ALT_JOBS_PATH,
        ALT_JOB_POSTINGS_PATH,
        ALT_VACANCIES_PATH,
        ALT_LABOR_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required source files are missing:\n" + "\n".join(missing))

    bqnt = load_monthly_file(BQNT_PATH, "date")
    if "home_price" not in bqnt or "home_rent" not in bqnt:
        raise ValueError("BQNT data must retain home_price and home_rent as separate series")

    alt_panel = load_monthly_file(ALT_PANEL_PATH, "period")
    alt_panel = alt_panel.rename(columns={column: f"alt__{column}" for column in alt_panel.columns if column != "date"})

    alt_prices = pd.read_csv(ALT_PRICE_PATH)
    required_price_columns = {"period", "property_class", "price_index"}
    if not required_price_columns.issubset(alt_prices.columns):
        raise ValueError(f"Alternative price data must include {sorted(required_price_columns)}")
    alt_prices["date"] = month_start(alt_prices.pop("period"))
    if alt_prices.duplicated(["date", "property_class"]).any():
        raise ValueError("Duplicate date/property_class rows in alternative home price data")
    alt_prices["column"] = alt_prices["property_class"].map(column_slug).map(
        lambda name: f"alt__rvd_home_price__{name}"
    )
    alt_prices = alt_prices.pivot(index="date", columns="column", values="price_index").reset_index()
    alt_prices.columns.name = None

    monthly_sources = [
        alt_panel,
        alt_prices,
        load_landreg_monthly(),
        load_restaurant_monthly(),
        load_job_board_monthly(),
        load_job_posting_monthly(),
        load_official_vacancies_monthly(),
        load_official_labor_monthly(),
    ]
    master = bqnt
    for source in monthly_sources:
        master = master.merge(source, on="date", how="outer", validate="one_to_one")
    master = master.sort_values("date").reset_index(drop=True)
    if master["date"].duplicated().any():
        raise ValueError("Merged master contains duplicate months")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = DATA_DIR / "monthly_master.csv"
    master.to_csv(master_path, index=False, date_format="%Y-%m-%d")

    home_required = ["home_price", "fed_rate", "hsi", "secondary_sales", "home_rent", "cnyhkd"]
    absent = sorted(set(home_required) - set(master.columns))
    if absent:
        raise ValueError(f"Home model columns are missing from the integrated master: {absent}")
    master[["date", *home_required]].to_csv(
        DATA_DIR / "home_model_input.csv", index=False, date_format="%Y-%m-%d"
    )
    extension_columns = list(EXTENSION_ROLES)
    missing_extension = sorted(set(extension_columns) - set(master.columns))
    if missing_extension:
        raise ValueError(f"Curated extension columns are missing from the master: {missing_extension}")
    master[["date", *home_required, *extension_columns]].to_csv(
        DATA_DIR / "extension_model_input.csv", index=False, date_format="%Y-%m-%d"
    )

    bqnt_dictionary = pd.read_csv(BQNT_DICTIONARY_PATH).rename(columns={"series": "column"})
    dictionary_rows = []
    for row in bqnt_dictionary.to_dict("records"):
        dictionary_rows.append(
            {
                "column": row["column"],
                "source": "BQNT",
                "source_series": row["ticker"],
                "unit": row["unit"],
                "source_frequency": row["frequency"],
                "monthly_aggregation": row["agg"],
                "coverage_first": None,
                "coverage_last": None,
                "notes": row["validation_status"],
            }
        )

    alt_columns = [column for column in master.columns if column.startswith("alt__")]
    source_prefixes = {
        "alt__rvd_home_price__": "AltData home_price_monthly.csv",
        "alt__landreg__": "AltData landreg_observations.csv",
        "alt__restaurant_": "AltData restaurant_district_snapshots.csv",
        "alt__jobboard__": "AltData job_board_snapshots.csv",
        "alt__jobposting__": "AltData job_posting_snapshots.csv",
        "alt__official_vacancies__": "AltData official_vacancies_quarterly.csv",
        "alt__official_labor__": "AltData official_labor_annual.csv",
        "alt__": "AltData regression_monthly_panel.csv",
    }
    landreg_labels = {}
    for description in pd.read_csv(ALT_LANDREG_PATH)["description"].drop_duplicates():
        key = unique_slug(description)
        landreg_labels[f"alt__landreg__{key}__units"] = f"{description} (units)"
        landreg_labels[f"alt__landreg__{key}__consideration_hkd_mn"] = f"{description} (HKD million consideration)"
    for column in alt_columns:
        source = next(label for prefix, label in source_prefixes.items() if column.startswith(prefix))
        frequency = "annual; placed at reference-year December only" if column.startswith("alt__official_labor__") else "quarterly; placed at reference-quarter end only" if column.startswith("alt__official_vacancies__") else "monthly-aligned; original frequency/availability retained in source"
        dictionary_rows.append(
            {
                "column": column,
                "source": source,
                "source_series": landreg_labels.get(column, column.removeprefix("alt__")),
                "unit": "See AltData source schema",
                "source_frequency": frequency,
                "monthly_aggregation": "as exported; sparse lower-frequency values are not filled",
                "coverage_first": None,
                "coverage_last": None,
                "notes": "Latest-vintage retrospective data; inspect AltData README and source registry before modeling.",
            }
        )

    dictionary = pd.DataFrame(dictionary_rows)
    for column in dictionary["column"]:
        observed = master.loc[master[column].notna(), "date"] if column in master else pd.Series(dtype="datetime64[ns]")
        mask = dictionary["column"].eq(column)
        dictionary.loc[mask, "coverage_first"] = observed.min().strftime("%Y-%m") if len(observed) else None
        dictionary.loc[mask, "coverage_last"] = observed.max().strftime("%Y-%m") if len(observed) else None
    dictionary.to_csv(DATA_DIR / "monthly_data_dictionary.csv", index=False)
    extension_dictionary = dictionary.loc[
        dictionary["column"].isin([*home_required, *extension_columns])
    ].copy()
    extension_dictionary["extension_role"] = extension_dictionary["column"].map(
        lambda column: "baseline_y_or_x" if column in home_required else EXTENSION_ROLES[column]
    )
    extension_dictionary.to_csv(DATA_DIR / "extension_feature_dictionary.csv", index=False)

    coverage = pd.DataFrame(
        {
            "column": master.columns,
            "n_observed": [int(master[column].notna().sum()) for column in master.columns],
            "n_missing": [int(master[column].isna().sum()) for column in master.columns],
            "first_observed": [master.loc[master[column].notna(), "date"].min() for column in master.columns],
            "last_observed": [master.loc[master[column].notna(), "date"].max() for column in master.columns],
        }
    )
    coverage.to_csv(DATA_DIR / "monthly_coverage.csv", index=False, date_format="%Y-%m-%d")

    print(f"Wrote {master_path}")
    print(f"Rows: {len(master)} | months: {master['date'].min():%Y-%m} to {master['date'].max():%Y-%m}")
    print(f"Columns: {len(master.columns)} | Alternative-data columns: {len(alt_columns)}")
    complete_home = master.loc[master[home_required].notna().all(axis=1)]
    print(
        "Complete Home-model months:",
        len(complete_home),
        f"({complete_home['date'].min():%Y-%m} to {complete_home['date'].max():%Y-%m})",
    )

    source_paths = [
        BQNT_PATH,
        BQNT_DICTIONARY_PATH,
        ALT_PANEL_PATH,
        ALT_PRICE_PATH,
        ALT_LANDREG_PATH,
        ALT_RESTAURANT_PATH,
        ALT_JOBS_PATH,
        ALT_JOB_POSTINGS_PATH,
        ALT_VACANCIES_PATH,
        ALT_LABOR_PATH,
    ]
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_file": master_path.name,
        "monthly_rows": len(master),
        "columns": len(master.columns),
        "date_first": master["date"].min().strftime("%Y-%m"),
        "date_last": master["date"].max().strftime("%Y-%m"),
        "home_complete_months": len(complete_home),
        "home_first_complete": complete_home["date"].min().strftime("%Y-%m"),
        "home_last_complete": complete_home["date"].max().strftime("%Y-%m"),
        "sources": [
            {
                "path": str(path.relative_to(WORKSPACE_DIR)),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for path in source_paths
        ],
    }
    (DATA_DIR / "monthly_source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
