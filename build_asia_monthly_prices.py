"""Build a fixed monthly Hong Kong, China and Seoul housing-price comparison.

The China series is a constructed composite, not an official aggregate index.
Foreign series remain outside the fitted Hong Kong models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
from http.cookiejar import CookieJar
from urllib.parse import parse_qsl, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "asia_home_prices_monthly.csv"
META = ROOT / "data" / "asia_home_prices_monthly_sources.json"
NBS_CSV = "https://raw.githubusercontent.com/changao1/70-China-cities-housing-index-data-by-national-bureau-of-statistics/main/merged_housing_data_eng.csv"
NBS_PAGE = "https://www.stats.gov.cn/sj/zxfb/202609/t20260915_1965304.html"
REB_URL = "https://www.reb.or.kr/r-one/portal/stat/easyStatPage/A_2024_00178.do"
REB_DATA_URL = "https://www.reb.or.kr/r-one/portal/stat/sttsDataPreviewList.do"


def get_text(url: str) -> str:
    with urlopen(Request(url, headers={"User-Agent": "HK-home-price-research/1.0"}), timeout=40) as response:
        return response.read().decode("utf-8-sig")


class InputParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.value: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and attributes.get("id") == "firParam":
            self.value = attributes.get("value")


def hong_kong_monthly() -> pd.Series:
    frame = pd.read_csv(ROOT / "data" / "extension_model_input.csv", parse_dates=["date"])
    series = frame.set_index("date")["home_price"].dropna()
    series.index = series.index.strftime("%Y-%m")
    return series.rename("hong_kong")


def seoul_monthly() -> pd.Series:
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    with opener.open(Request(REB_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=40) as response:
        page = response.read().decode("utf-8")
    parser = InputParser()
    parser.feed(page)
    if parser.value is None:
        raise ValueError("REB query parameters were not found")
    parameters = dict(parse_qsl(parser.value))
    if parameters.get("statblId") != "A_2024_00178":
        raise ValueError("Unexpected REB table")
    parameters["wrttimeLastestVal"] = "300"
    request = Request(REB_DATA_URL, data=urlencode(parameters).encode("utf-8"),
                      headers={"User-Agent": "Mozilla/5.0", "Referer": REB_URL})
    with opener.open(request, timeout=40) as response:
        rows = json.loads(response.read().decode("utf-8"))["DATA"]
    matches = [row for row in rows if row.get("CATE1") == "서울" and row.get("CATE2") == "서울"]
    if len(matches) != 1:
        raise ValueError(f"Expected one Seoul aggregate, found {len(matches)}")
    monthly = {}
    for key, raw_value in matches[0].items():
        if key.startswith("COL_") and key[4:10].isdigit() and raw_value not in ("", "-"):
            monthly[pd.Period(key[4:10], freq="M")] = float(raw_value)
    if len(monthly) < 100:
        raise ValueError("REB Seoul series is unexpectedly short")
    monthly_series = pd.Series(monthly, dtype=float).sort_index()
    monthly_series.index = monthly_series.index.astype(str)
    return monthly_series.rename("seoul")


def china_70_city_monthly() -> pd.Series:
    # The NBS publishes previous-month = 100 for each city. Equal-weight the 70
    # log changes, then chain them into our descriptive composite.
    data = pd.read_csv(StringIO(get_text(NBS_CSV)))
    data["date"] = pd.to_datetime(dict(year=data["year"], month=data["month"], day=1))
    values = pd.to_numeric(data["existing_home_price_index"], errors="coerce")
    data = data.assign(value=values)
    counts = data.groupby("date").agg(observations=("value", "count"), cities=("city", "nunique"))
    if (counts[["observations", "cities"]] != 70).any().any() or (values <= 0).any():
        raise ValueError("NBS monthly 70-city series is incomplete or invalid")
    mean_log_change = np.log(data["value"] / 100).groupby(data["date"]).mean().sort_index()
    composite = 100 * np.exp(mean_log_change.cumsum() - mean_log_change.iloc[0])
    composite.index = composite.index.strftime("%Y-%m")
    return composite.rename("china_70_city")


def main() -> None:
    frame = pd.concat([hong_kong_monthly(), china_70_city_monthly(), seoul_monthly()], axis=1).sort_index()
    frame = frame.loc["2011-01":].dropna(how="any")
    expected = pd.period_range(frame.index[0], frame.index[-1], freq="M").astype(str)
    if len(frame) < 150 or not frame.index.equals(pd.Index(expected)) or (frame <= 0).any().any():
        raise ValueError("The monthly comparison is short, nonconsecutive or contains invalid values")
    frame.index.name = "month"
    frame.round(4).to_csv(OUT)
    metadata = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_month": frame.index[0], "last_month": frame.index[-1], "observations": len(frame),
        "series": {
            "hong_kong": {"provider": "Hong Kong RVD via supplied model input", "coverage": "private domestic, all classes", "method": "Official monthly price index, 1999 = 100", "url": "https://www.rvd.gov.hk/en/publications/property_market_statistics.html"},
            "china_70_city": {"provider": "National Bureau of Statistics of China; machine-readable transcription by Chang Gao", "coverage": "existing residential buildings in 70 cities", "method": "Our equal-city-weight chained geometric mean of official monthly city indices (previous month = 100); 2011-01 = 100. This composite is not an official national aggregate.", "url": NBS_PAGE, "transcription_url": "https://github.com/changao1/70-China-cities-housing-index-data-by-national-bureau-of-statistics"},
            "seoul": {"provider": "Korea Real Estate Board R-ONE", "coverage": "Seoul apartments, actual transactions", "method": "Official monthly sale-price index; 2026 June = 100 in retrieved vintage", "url": "https://www.reb.or.kr/r-one/portal/stat/easyStatPage/A_2024_00178.do"},
        },
    }
    META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(frame)} months, {frame.index[0]} to {frame.index[-1]} to {OUT}")


if __name__ == "__main__":
    main()
