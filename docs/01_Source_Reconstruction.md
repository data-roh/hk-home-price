# Interpreting the Source Material and the Replication Boundary

This document analyzes the two Bloomberg Intelligence methodology screenshots provided by the user. The user later supplied monthly BQNT extracts, but the full report, estimated coefficients, original code, and release-vintage history were not provided. The sections below distinguish between **publicly disclosed facts**, **execution assumptions**, and **items that can be validated with the supplied data**. This project is not a package distributed or validated by Bloomberg.

## Home-price model

What can be confirmed from the screenshot:

- Multivariate generalized least squares regression.
- An auto-regressive covariance structure for the errors.
- Data since February 1997.
- Variable retention based on economic and statistical significance.
- Automatic incorporation of the latest actual values for explanatory variables, with paths through 2027 that account for trend and seasonal patterns.
- The dependent variable is the monthly RVD private domestic property price index, All Classes.

| Role | Ticker shown on screen | Variable | Unit and frequency shown on screen |
|---|---|---|---|
| Y | HKRLPDAP Index | RVD Private Domestic Price Index, All Classes | 1999=100, monthly |
| X | FEDL01 Index | US Federal Funds Effective Rate | %, intraday |
| X | HSI Index | Hang Seng Index | index points, intraday |
| X | MIDJLTTL Index | Midland Secondary Home Transaction Volume | transaction count, monthly |
| X | HKRLPDAR Index | RVD Private Domestic Rental Index, All Classes | 1999=100, monthly |
| X | CNYHKD BGN Curncy | Hong Kong dollar per Chinese yuan | HKD per 1 CNY, intraday |

The tickers, including `MIDJLTTL`, were transcribed from screenshots and have not been validated in Terminal. They must be rechecked against DES/FLDS and the raw data. A higher FX rate means more HKD per one yuan. Confusing this with the reciprocal CNY per HKD rate would reverse the sign interpretation.

The closest general form to the source model is:

\[
g_P(P_t)=a+\sum_{j=1}^5\beta_j g_j(X_{j,t-\ell_j})+u_t,
\qquad \phi(B)u_t=\epsilon_t.
\]

The transformations \(g\), variable lags \(\ell_j\), AR order, and exact estimator details are not disclosed on screen. Therefore the execution baseline R0 is defined as **level units, contemporaneous X, AR(1) errors, and an intercept**, with every one of these choices explicitly labeled as an assumption. The source does not explicitly state AR(1). `GLSAR` is one educational feasible-GLS approximation and does not imply that the original estimator used by Bloomberg Intelligence was literally `GLSAR`.

## Office-rent model

What can be confirmed from the screenshot:

- Multivariate SARIMAX.
- Monthly data since 2006.
- Two AR terms and one MA term.
- Automatic incorporation of the latest actual values, with paths through 2027 that account for trend and seasonal patterns.
- The dependent variable is Colliers Hong Kong Overall office net effective rent.

| Role | Ticker shown on screen | Variable | Unit and frequency shown on screen |
|---|---|---|---|
| Y | COLNETE Index | Colliers Office Net Effective Rent, Hong Kong Overall | HKD/sq ft, monthly |
| X | HKMCTTL Index | HK Exchange Total Market Capitalisation | HKD billion, daily |
| X | COLLVACA Index | Colliers Office Vacancy, Hong Kong Overall | %, monthly |
| X | HKCPIY Index | Composite CPI All Items, YoY change | %, monthly |
| X | HKUERATE Index | Hong Kong Unemployment Rate | %, monthly |
| X | FEDL01 Index | US Federal Funds Effective Rate | %, intraday |

The precise start month in 2006 is undisclosed, so the code uses the available common sample beginning in 2006-01. `HKCPIY` is a YoY rate of change, not a CPI level. Turning that into another YoY transform would create a different variable. The data definition also needs to confirm whether the unemployment rate is a seasonally adjusted three-month moving statistic and what exact geography or grade range the vacancy series covers.

General form:

\[
g_R(R_t)=a+X_t'\beta+u_t,
\]
\[
\phi(B)\Phi(B^{12})(1-B)^d(1-B^{12})^D u_t
 =\theta(B)\Theta(B^{12})\epsilon_t.
\]

What the source explicitly discloses is non-seasonal AR=2 and MA=1. The values of \(d,D,P,Q\), the seasonal period, transformations, and lags are not disclosed. The baseline choice `(2,0,1) × (0,0,0,0)` is therefore only the simplest transparent assumption and is not claimed to be the full original SARIMAX specification. The word `SARIMAX` alone is not enough to justify adding `(1,1,1,12)` or any other seasonal setting.

## Economic roles and identification limits

| Factor | Intended economic channel | Interpretation caution |
|---|---|---|
| US policy rate | Funding cost, discount rate, and the broader rate environment | It is not identical to the actual Hong Kong mortgage rate or pass-through speed |
| HSI / market cap | Wealth effects, risk appetite, financial-sector conditions | These can co-move with prices and rents because of common shocks; not causal identification |
| Housing transaction volume | Liquidity and demand activity | Simultaneously determined with prices; affected by supply, policy, and registration timing |
| Housing rent index | Housing demand and rental cash flow | May share a common trend or cointegrating relationship with sale prices |
| CNYHKD | Mainland purchasing power and FX environment | Not a direct measure of migration or capital flow |
| Vacancy rate | Slack in space demand and supply | Simultaneously determined with rent and new supply; Colliers definitions must be consistent |
| CPI and unemployment | Price-level change and labor-market conditions | Confirm nominal versus real interpretation, release lag, seasonal adjustment, and revisions |

## How seasonality should be framed

The user's concern is valid, but the screenshots themselves already mention seasonal-pattern adjustment. The right hypothesis is not “the original model ignored seasonality” but **where that seasonal adjustment enters the system, how it is implemented, and whether additional seasonal treatment improves out-of-sample performance**.

### What the screenshots disclose and what they do not

Both screenshots state, in substance, that the latest actual factor values are incorporated and that **trend and seasonal patterns are adjusted through 2027**. This means the producer at least claims to account for trend and seasonality when generating future paths. But that wording alone does not reveal which of the following implementations is used:

- Applying seasonal patterns to the future explanatory-variable paths \((X)\) through 2027.
- Including monthly dummies or Fourier terms in the mean equation for \((Y)\).
- Seasonally adjusting \((Y)\) or \((X)\) in advance.
- Including seasonal differencing \((D>0)\) or seasonal AR/MA terms.
- Including moving-calendar effects such as Lunar New Year, public holidays, or monthly business-day counts.

Given the wording and where it appears, it may be more likely that the adjustment is applied to future factor paths, but the screenshot does not allow that to be confirmed. Likewise, the label `SARIMAX` does not imply a nonzero seasonal order. The appropriate research framing is therefore:

> The Bloomberg Intelligence source explicitly discloses seasonal-pattern adjustment, but it does not disclose the exact seasonal specification. This study separates monthly dummies, Fourier terms, calendar effects, and seasonal error dynamics, and compares them under a common out-of-sample evaluation design.

Housing transaction counts in particular may be sensitive to month length, business days, and the timing of Lunar New Year. Price indices and rental series can have different seasonal structure from transaction volume. The fact that transaction activity is seasonal does not automatically imply strong 12-month seasonality in the price error process. Seasonally adjusting an explanatory variable, inserting seasonal patterns into future X paths, and adding seasonal AR terms to the Y error process are all different operations.

The first comparison should therefore hold the sample fixed and evaluate R0, Fourier K=1, Fourier K=2, monthly dummies, and then calendar variables. Seasonally adjusted unemployment data should not be mechanically adjusted again. Full-sample STL should not be run before training and validation because that would leak future information.

## What is still needed for full numerical replication

Exact transformations, lags, AR/SARIMA orders, the estimation end date, intercept or trend treatment, dummy variables, the variable-selection procedure, source-data vintages, historical release dates, original forecasts and explanatory-variable paths, and coefficient and residual diagnostics would all be needed. The attachments alone are not enough to recreate the original 2027 numbers or match the original fit exactly.

The correct description of this project is **“disclosed model-family reconstruction with transparent specification assumptions.”** It should not be presented in interviews as a literal copy of Bloomberg Intelligence source code.
