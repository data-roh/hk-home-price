# Extension Roadmap: Experimental Design That Separates Model Changes from New Variables

## 1. What should count as an improvement?

An improvement is not defined by a higher in-sample $R^2$ or by a larger number of statistically significant coefficients. The relevant standard is lower and more stable loss than the baseline under the same information timing, evaluation origins, and forecast horizons, together with reasonable interval coverage, interpretable economic pathways, and an update process that could be run operationally.

The core design is a 2×2 comparison:

| Experiment | Original variable set | New variable set |
|---|---|---|
| Keep the original model family | A: baseline GLSAR / SARIMAX | B: add new X to the same family |
| Apply a new method | C: seasonality / ECM / state-space / etc. | D: new method + new X |

The A→B contrast isolates the contribution of additional data, while A→C isolates the contribution of method changes. If only D looks better and the B/C comparisons are skipped, it becomes hard to tell what produced the improvement. Here, “original model” means the reconstructed disclosed family, not the undisclosed BI source code.

## 2. Seasonality: the first small experiment to run

This experiment should not be described as “adding seasonality that the original model missed.” The source already states that trend and seasonal patterns are adjusted through 2027. The research contribution here is instead **to decompose the undisclosed location and form of seasonality into transparent challenger specifications and measure their incremental value under the same rolling-origin design**. Future-X path seasonality, deterministic seasonality in the Y mean equation, and stochastic seasonality in the innovation process should not be combined all at once.

Hold the same y/X, transformations, lags, and AR order fixed, and compare:

1. S0: no explicit deterministic seasonal term.
2. S1: Fourier K=1, adding 2 coefficients.
3. S2: Fourier K=2, adding 4 coefficients.
4. S3: 11 monthly dummies.
5. S4: official-calendar Lunar New Year exposure and business-day counts.
6. S5: restricted seasonal ARMA challengers for any remaining lag-12 innovation dependence after mean-equation seasonality is included.

`models.py` supports Fourier K and monthly dummies; S1 (`fourier1`) is registered in the comparison. Out of sample it is the best GLSAR variant but only marginally better than the baseline, and it still loses to the random walk at 3–12 months, so seasonality is not the bottleneck. S4 requires a validated historical calendar CSV and an explicit definition. S5 can be explored through `ModelSpec.seasonal_order`, but it is not a validated default. Lunar New Year should not be approximated as “always February,” and annual holiday counts should not be copied mechanically into every month.

Housing transaction activity may be more sensitive to month length and holiday effects than the price index itself. That creates two separate experiments: seasonally adjusting transaction X before modeling, and adding seasonal terms to the Y mean equation. If the raw source is already seasonally adjusted, do not apply another seasonal adjustment on top. All filter choices and K selection should be done inside the training folds, with the final evaluation sample held fixed.

Official calendar starting point: https://data.gov.hk/en-data/dataset/hk-dpo-statistic-cal . Do not assume that the currently published files provide a complete historical record with all past revisions since 1997. Record the historical coverage and the future confirmed dates before using them.

## 3. Keep the model family and add new variables: priorities

| Priority | Target | New variable | Why add it | Unit / transform candidates | Frequency / information timing |
|---|---|---|---|---|---|
| 1 | Housing | Government policy stance: stamp-duty / LTV event layer, official-release text tone, land supply | HK government actively steers prices; currently the largest missing channel | event dummies (±1), monthly tone index, surprise component | announcement date ≠ effective date; stance is endogenous to prices |
| 1 | Housing | Foreign finance and legal firms: regional offices, SFC licensees, foreign law firms, Grade A office take-up, fixed-panel hiring | Expat / high-income demand for large units and rentals | counts, Δlog, take-up area; pair with RVD Classes D–E and rent | annual / quarterly / monthly; mixed-frequency, no forward fill |
| 1 | Housing | 1M/3M HIBOR, actual new mortgage rates | Direct pass-through path from US policy rates to Hong Kong borrowing costs | % level, pp changes, limited 1-3 month lags | daily / monthly, verify official publication timing |
| 1 | Housing | HKMA mortgage applications / approvals / drawdowns | Credit demand and funding inflow for transactions | log, Δlog, approval-to-application ratio | monthly, respect publication lag |
| 1 | Both markets | Official business days and Lunar New Year exposure | Calendar exposure and moving-holiday timing | counts or 0-1 exposure measures | future calendar known, verify historical coverage |
| 2 | Housing | Primary-market supply, completions, unsold inventory | Supply shocks not captured by secondary transaction volume | units / area, log or lagged stock | mixed monthly/quarterly/annual, do not use before release |
| 2 | Housing | Income, employment, households, net migration, population inflow | Quantity-side housing demand | real income, net inflow, growth rates | do not interpolate low-frequency data into fake monthly actuals |
| 1 | Office | Net absorption, new completions, leasable stock | Separate demand and supply behind vacancy changes | area, absorption/stock | harmonize provider, grade, geography, and frequency |
| 1 | Office | Financial / professional employment and occupation-specific hiring | Employment channel for office-space demand | new or active postings for fixed firm cohorts | monthly / quarterly and prospective daily snapshots |
| 2 | Office | Renewals, relocations, leased area, lease incentives | More direct drivers of net effective rent | area, rent-free period, cash-flow adjusted terms | requires provider data and usage rights |
| 2 | Both markets | Policy / financial-regime interactions | Regime-dependent rate transmission and capital-flow effects | HIBOR×policy regime, FX×Mainland demand | align with actual implementation and release dates |
| 2 | Housing (regional) | FEHD restaurant licences by district (18 districts) | Neighbourhood activity; regional demand | 12-month net-opening rate; district panel vs. regional prices | 2020+ only; short-sample protocol |
| 3 | Both markets | OpenRice / Google review changes | Exploratory local activity proxies | net change within a fixed restaurant cohort and coverage | newly started snapshots cannot be backfilled into history |

The first additional variables should be **policy stance, finance/legal firm activity, and HIBOR/mortgage flows**, not broad web-review proxies. Their economic channels are clear and specific to Hong Kong, and official histories exist. Notebook 01 §9 gives first evidence:

- **Policy leans against the wind.** Every tightening followed a six-month price rise of 11–13%, and both easings followed falls of about 10%. Stance is endogenous, so use the surprise component or forward-looking text tone.
- **Large units follow their own cycle.** Classes D–E rose +58% since 2010-02 vs. +132% for A–C, so a finance/legal hiring signal should be tested on large units and rent first.
- **Restaurants are weak at the city level.** Correlation with price growth is 0.09 at lead 0 and −0.40 at a 6-month lead. Their value is in the district cross-section.

This is a research priority before performance testing, not a claim about eventual forecasting gains. Because the level relation is not cointegrated, each channel enters the **growth model** as a change, one at a time, and must beat both the growth model and the random walk out of sample. Full detail is in `docs/extension_review.md`.

Verified official sources:

- HIBOR publication: https://www.hkab.org.hk/en/rates/hibor/
- HKMA mortgage API documentation: https://apidocs.hkma.gov.hk/documentation/market-data-and-statistics/monthly-statistical-bulletin/banking/residential-mortgage-survey/
- HKMA API: https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin/banking/residential-mortgage-survey
- RVD property statistics: https://www.rvd.gov.hk/en/publications/property_market_statistics.html
- C&SD labor-market statistics: https://www.censtatd.gov.hk/en/scode200.html

No Bloomberg tickers, fields, or native units were fabricated for new variables before Terminal validation. Some RVD stock, vacancy, or take-up statistics may be annual releases even if they appear in a “monthly supplement” publication, so they should not be treated as monthly observations without checking. Colliers vacancy and RVD vacancy may also use different definitions and should not be stitched together mechanically.

## 4. Concrete equations for variable-extension experiments

An example that preserves the original error structure:

\[
p_t=a+X_t'\beta+\delta_1 HIBOR_{t-\ell_1}
+\delta_2\log MortgageApprovals_{t-\ell_2}
+\delta_3 CNYExposure_t+u_t,
\quad \phi(B)u_t=\epsilon_t.
\]

Including Fed, HIBOR, and mortgage rates together may create collinearity. Compare economically motivated alternatives such as replacing Fed with HIBOR, including a Fed-HIBOR spread, or using HIBOR only. Do not accumulate new variables by default. After adding new X, revisit sample length, stationarity or cointegration, and the residual structure. “Keeping the same model family” does not mean all previous diagnostics automatically remain valid.

Office example:

\[
r_t=a+X_t'\beta+\delta_A Absorption_{t-\ell}
+\delta_C Completions_{t-\ell}
+\delta_J ProfessionalHiring_{t-\ell}+u_t,
\quad \phi_2(B)u_t=\theta_1(B)\epsilon_t.
\]

Active job postings are a stock, while newly posted openings are a flow. Use fixed sources and firm cohorts rather than broad market totals, and track duplicates, reposts, and multi-location postings. Do not classify all jobs visible on the first collection date as “new hiring that day.” Foreign-hiring demand should be handled through observable attributes such as English-language posting, visa sponsorship disclosure, or Chinese-language requirement, not by imputing nationality or visa eligibility.

## 5. New method 1: ARDL / ECM

If a level regression is unstable and cointegration is supported among the relevant variables, then separate long-run equilibrium from short-run shocks.

\[
\Delta p_t=\alpha\big(p_{t-1}-c-\theta'x_{t-1}\big)
+\sum_i\psi_i\Delta p_{t-i}+\sum_j\Gamma_j'\Delta x_{t-j}
+\gamma'Calendar_t+\epsilon_t.
\]

Why this helps: it can reduce spurious-regression risk caused by trends while preserving a long-run relationship between prices, rents, and related variables. It also separates long-run elasticity \(\theta\), adjustment speed \(\alpha\), and short-run changes \(\Gamma\).

Requirements: check integration order, whether any variable may be I(2), cointegration support, adjustment direction, root stability, and structural breaks first. If cointegration is not supported, do not force an ECM. On the supplied data it is not supported (Engle–Granger p ≈ 0.98 in levels; ECM gate p ≈ 0.92 in logs), so the ECM is withheld and the growth model `Δlog y = c + φ Δlog y(t−1) + ΔX γ` serves as the short-run specification. Johansen and Gregory–Hansen tests are the next check. `fit_home_ecm` includes a testing implementation and a cointegration gate, but that gate alone is not a full proof that the assumptions hold. Without standard bootstrap treatment, the uncertainty in the two-step long-run estimate \(\theta\) is not fully reflected. Recursive ECM forecasting and dashboard support are not part of the default UI package.

## 6. New method 2: time-varying coefficients and structural change

\[
y_t=\mu_t+X_t'\beta_t+s_t+\epsilon_t,
\quad \beta_t=\beta_{t-1}+\eta_t.
\]

If rate pass-through, financial-sector space use, or post-pandemic demand have changed, constant coefficients may be too restrictive. But increasing coefficient flexibility also raises overfitting and identification risk. Shrink the state innovation variance and compare OOS performance against rolling-window or fixed-\(\beta\) alternatives. Do not let random-walk coefficients “explain” every movement by construction.

A lower-cost intermediate step is to use pre-defined policy-regime interactions, rolling coefficient stability checks, and recursive estimation. If the break date is chosen by searching the data, incorporate uncertainty from that search. Markov-switching, TVP, and stochastic-volatility variants are design ideas, not verified built-in features of the current package.

## 7. New method 3: lag shrinkage and nonlinear challengers

If the effect of rate changes is spread across several months, consider restricted distributed lags or Almon-polynomial / shrinkage approaches. Increasing the AR order of the error term is not a substitute for modeling the mean response properly.

Monthly datasets covering 20-30 years still provide only a few hundred rows. Before moving to large deep-learning models, it is reasonable to compare challengers such as lagged Elastic Net models or small gradient-boosted trees. All lag selection, normalization, and hyperparameter search should be nested inside rolling validation. Feature importance or SHAP values should not be described as causal contributions. Tree-based models are also weak at level extrapolation, so out-of-range scenarios need separate checks. These remain proposed challengers, not validated defaults.

## 8. New method 4: mixed frequency and explanatory-variable forecasts

Simple linear interpolation of quarterly or annual income and supply data would manufacture monthly “actual” values that were never observed. Consider ragged-edge state-space nowcasting, MIDAS, or low-frequency indicators that become usable only after release. If the sample is short or real-time vintages are unavailable, preserving the lower frequency may be the more honest choice.

Future X paths can be compared under fixed-value, seasonal-naive, separate univariate forecasts, joint VAR or state-space systems, and user-defined scenarios. Keep oracle scoring, which feeds realized future X into Y, separate from operational forecast scoring, which must rely only on information available at the origin. Diagnosing whether X-path forecast error is the main bottleneck is itself an important research result.

## 9. Stage gates

| Stage | Deliverable | Pass condition |
|---|---|---|
| G0 | Source facts and assumptions ledger | No undisclosed item is presented as fact |
| G1 | Audit of actual BQNT data, units, and timing | Required columns are secured, internal missingness resolved, vintage limits stated |
| G2 | Baseline model A and naive benchmarks | Convergence, roots, innovation diagnostics, and unit-scale OOS loss recorded |
| G3 | Seasonality experiments S0-S4 | Common origins, same X-forecast rule, coverage and stability comparison |
| G4 | One-group-at-a-time new-X ablations | Separate the loss of sample length from the value of new information |
| G5 | ECM / TVP / other challengers | Assumptions checked, inner validation used, final holdout preserved |
| G6 | Dashboard / model card | Input units, path rules, extrapolation, and uncertainty clearly shown |

Do not choose a target such as “10% RMSE improvement” after seeing the data. Agree in advance on business-acceptable forecast error by horizon, the minimum number of valid origins, update feasibility, and interpretation constraints. If a new variable does not exist for the full long history, compare A versus B on the overlapping period and report the long-sample baseline separately as well.

## 10. How to present the research contribution in an interview

What the source already did: multivariate property forecasting with serially correlated dynamics.

What can legitimately be claimed as your contribution: separating disclosed from undisclosed specification elements, defining the data and information-timing contract, re-estimating without leakage, decomposing seasonality structurally, separating pure model contribution from feature contribution through ablation, giving the correct conditional interpretation of scenarios, and producing an operational model card and dashboard.

What cannot yet be claimed: stable forecasting gains from the extension variables, replication of the original coefficients or 2027 forecast numbers, real-time PIT performance, or identified causal effects. The common-origin comparisons on the supplied latest-vintage Home sample are recorded in `outputs/model_comparison_scores.csv` (23 origins for method challengers, 19 for the 2011+ transaction extensions), and results from different samples or horizons should not be compared directly.
