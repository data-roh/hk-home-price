# Home Model Extension Review

## Baselines

Every extension is compared against two models on the same sample and the same forecast origins:

- **Level GLSAR(1)**: `home_price` on `fed_rate`, `hsi`, `secondary_sales`, `home_rent`, and `cnyhkd` with AR(1) errors. This reconstructs the BI model family, with transparent assumptions filling the undisclosed details.
- **Growth model**: Δlog `home_price` on its own lag and the changes in the same five predictors, with HAC errors. The level relation is not cointegrated, so **new variables enter here first, as changes**.

The **random walk** is the benchmark both must beat.

## AltData candidates and how to use them

| Candidate | Integrated columns / source | Decision and caution |
|---|---|---|
| RVD price indices, Classes A–E and groups | `alt__rvd_home_price__*`; `home_price_monthly.csv` | Same kind of index as the BQNT target. Use for cross-checks, or as **alternative targets** (e.g., large units), never as predictors of `home_price`. |
| Land Registry transaction counts and values | `alt__residential_*`, `alt__secondary_agreements`; `regression_monthly_panel.csv` | Available from 2011. Tested in notebook 01 §7.2; no out-of-sample gain (see below). Secondary agreements are a 0.99-correlated substitute for Midland sales. |
| Land Registry detail block | `alt__landreg__*`; `landreg_observations.csv` | 134 wide category columns. Screen housing-relevant definitions before any use. |
| FEHD restaurant licences, city and 18 districts | `restaurant_counts_city.csv`, `restaurant_district_snapshots.csv` | 2020+ (≈ 80 months). Weak or negative at the city level; the information is cross-sectional (see notebook 01 §9.3). Use only complete-month snapshots. |
| Government and firm-level hiring snapshots | `alt__gov_*`, `alt__jobboard__*`, `alt__jobposting__*` | CSB covers only government jobs; Lever covers one employer per series in a single month. Not usable as a finance/legal hiring signal. |
| Quarterly vacancy and annual labour statistics | `alt__official_vacancies__*`, `alt__official_labor__*` | Keep at native frequency. Do not forward-fill. Candidates only for MIDAS or state-space designs that respect release timing. |
| OpenRice review changes | `alt__openrice_*` | 15 restaurants without a durable cohort history. Not adopted. No Google review data is present. |

## Current empirical results

All numbers come from the shared `PSEUDO_OOS` protocol: expanding window from month 120, re-fitted every 3 months, horizons 1/3/6/12, future X at last value. Full tables are in `outputs/model_comparison_scores.csv` and `docs/analysis_review.md`.

**Method challengers** (23 common origins, 2020-01 to 2025-07), RMSE in index points:

| | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| Level GLSAR(1) | 5.46 | 11.54 | 19.66 | 31.22 |
| Growth | **4.53** | 10.74 | 21.93 | 41.17 |
| Fourier K=1 | 5.22 | 10.69 | 19.27 | 31.18 |
| AR(2) | 5.91 | 13.13 | 21.06 | 32.13 |
| Random walk | 5.38 | **10.43** | **17.35** | **28.10** |

The growth model wins only at one month but has the best interval coverage (87/91/83/70% vs. 74/70/74/43% for GLSAR). Fourier K=1 helps marginally; AR(2), lagged X, and logs are worse.

**Transaction extensions** (19 common origins, 2020-12 to 2025-06): T+1 RMSE is 4.24 for all four regression variants against 4.09 for the random walk, and 33.5–33.7 against 30.9 at T+12. The extensions are significant in sample but add nothing out of sample, so none is promoted.

## Next-variable research (notebook 01 §9)

The five current predictors cover rates, equities, activity, rent, and FX. Three Hong Kong-specific channels are missing.

### 1. Government policy stance

- **Construction:**
  - Event layer: stamp-duty and HKMA LTV changes coded −1 (tightening) / +1 (easing) by announcement month.
  - Monthly text-tone index from official releases: Policy Address, Budget, HKMA, Housing Bureau, and info.gov.hk.
  - Supply layer: land-sale programme and public-housing targets.
- **Evidence so far:** policy leans against the wind. Six-month price change before and after each event:

  | Event | Stance | 6m before | 6m after |
  |---|---|---|---|
  | 2010-11 SSD | Tightening | +11.6% | +12.7% |
  | 2012-10 BSD | Tightening | +12.0% | +6.8% |
  | 2013-02 DSD | Tightening | +13.0% | +2.6% |
  | 2016-11 flat 15% | Tightening | +10.6% | +8.4% |
  | 2023-10 halving | Easing | −9.5% | −3.9% |
  | 2024-02 removal | Easing | −10.8% | −3.5% |

- **Implication:** stance is endogenous, because it responds to prices. A naive regression biases the effect toward zero or flips its sign. Use the surprise component (e.g., the property-stock reaction on announcement day) or forward-looking text tone. Start with an event study. Verify dates against the Gazette or IRD before modelling.

### 2. Foreign finance and legal firms: office expansion and hiring

- **Sources:**
  - C&SD/InvestHK survey of regional HQs and offices (annual);
  - SFC licensed corporations and persons (quarterly);
  - Law Society foreign firms and lawyers (annual);
  - RVD and broker Grade A office vacancy, rent, and take-up;
  - a fixed panel of finance and legal employers' career pages or licensed job-board feeds.
- **Target pairing:** large units and rent. RVD Classes D–E (≥ 100 m²) rose about +58% from 2010-02 to 2026-08, against +132% for Classes A–C, and follow their own cycle. Test the hiring signal on the large-unit index and `home_rent` before the aggregate index.
- **Data today:** not usable. CSB covers government ads only, Lever covers one employer, and there is no authorised LinkedIn feed.
- **Caveats:** mixed-frequency handling is needed. A fixed employer panel is needed so that coverage drift is not read as hiring.
- **Side benefit:** Grade A office data would also unblock the Office model.

### 3. District commercial activity: restaurants

- **Signal:** 12-month net-opening rate = (openings − closures) / licence stock a year earlier. City licences grew from 15,934 to 17,292.
- **City level, 67 overlapping months:** correlation with 12-month price growth is 0.09 contemporaneously, −0.19 at a 3-month lead, and −0.40 at a 6-month lead. The 2021–22 reopening boom coincided with falling prices, so the city aggregate mostly reflects COVID reopening, not housing demand.
- **District level:** over the latest 12 months, North (Northern Metropolis), Kwun Tong, and Sai Kung led licence growth; Kwai Tsing, Tsuen Wan, and Yau Tsim Mong shrank. Test this cross-section against regional price indices with a short-sample protocol (min_train ≈ 48, h = 1–3) and a district panel, not against the HK-wide index.
- **Caveat:** a licence is not the same as business activity.

### Order of work

1. Policy stance plus a surprise measure.
2. Grade A office data and a finance/legal hiring panel, tested against Classes D–E and rent.
3. District restaurant panel against regional prices.
4. In parallel: HIBOR, mortgage applications/approvals/drawdowns, and release-date (vintage) information.

## Comparison contract

- Same dates, training window, origins, and horizons for the baseline and the extension.
- If a new variable shortens the sample, report the sample-change effect separately from the information gain.
- No future or unpublished values at an origin. Latest-vintage results are not described as point-in-time.
- Report RMSE/MAE with the random walk and seasonal naive alongside. Withhold strong conclusions when origins are few or fits fail.
- In-sample p-values and $R^2$ are not grounds for adoption. Promotion requires beating both the growth model and the random walk out of sample.

For the supporting theory see `docs/02_Methodology_Notes.md`; for the full roadmap see `docs/03_Extensions_and_New_Variables.md`.
