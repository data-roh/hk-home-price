# Project Description for a Data Scientist Interview

## 1. What can be stated truthfully at this stage

This project is a research prototype built around the publicly disclosed methodology screens for two Bloomberg Intelligence Hong Kong property models. It reconstructs the model families, evaluates them, and implements scenario tooling. It does not claim to replicate the actual BI source code or coefficients. The user-supplied latest-vintage monthly data supports estimation of the Home model, but the Office model is withheld because `office_rent` is unavailable.

An English 60-90 second introduction:

> I rebuilt the disclosed Bloomberg Intelligence Hong Kong home-price model, a GLS regression with autoregressive errors, on 199 months of data from 2010 to 2026. The first thing I checked was whether the level regression is valid. Prices and most drivers are non-stationary and not cointegrated, and the estimated AR coefficient is 0.993, so the level model is really a regression on monthly changes with leftover momentum in the residuals. I therefore added a growth-rate model with a momentum term. Both models agree on the robust drivers: the Fed rate is negative, transaction volume is positive, and rent matters only in levels. A 100bp Fed cut is associated with roughly +2.6 to +3.1% on prices; the growth model shows it building up over a year rather than jumping. Out of sample, on 23 common origins, the growth model beats a random walk at one month and has much better-calibrated intervals, but nothing beats the random walk at three to twelve months. I report that as a result. Extra Land Registry transaction data was significant in sample but added nothing out of sample. Next, I would add Hong Kong-specific channels: government policy stance, foreign finance and legal firm hiring, and district-level restaurant activity. The data are latest-vintage, so this is not a point-in-time backtest, and the scenarios are conditional, not causal.

Do not fabricate Office results or personal impact claims when the data is absent. In an interview, state the dates, sample, horizons, benchmark forecasts, performance, and vintage limitation with numbers.

## 2. Answer skeletons by question

### Why use GLS rather than OLS?

When residuals are serially correlated, OLS standard errors under independent errors can be misleading and the estimator can lose efficiency. If the error covariance is known or estimated, the model can be whitened and estimated with GLS. But GLS does not fix endogeneity or spurious regression. It serves a different purpose from OLS with HAC errors.

### How did you distinguish AR errors from a lagged dependent variable?

Expanding an AR-error regression introduces a coefficient restriction of `-rho × beta` on lagged X. That is not the same as an unrestricted ARDL. In the conditional scenario exercise, I changed only the future X path and did not divide beta by `1-rho` to manufacture a long-run multiplier.

### The source says SARIMAX. Why is the seasonal order set to 0?

Because the actual seasonal order was not disclosed. Only AR=2 and MA=1 were confirmed, while the rest was treated as an explicit assumption. Seasonal-pattern adjustment may also have entered through the forecast path of X rather than through the error process. I do not criticize the source for omitting seasonality; I compare transparent challenger specifications instead.

### Didn’t the source already say seasonal patterns were adjusted?

Yes. Both screens explicitly state that the latest actual values are automatically incorporated and that trend and seasonal patterns are adjusted through 2027. So this project should not be described as “adding seasonality that the original model missed.” What remains undisclosed is not the existence of seasonality, but **where it enters and which specification is used**. The correct explanation is that I separated future X-path seasonality, deterministic seasonality in the Y mean equation, pre-adjustment, seasonal ARMA structure, and Lunar New Year or business-day effects, and then compared them out of sample.

> Yes. The source explicitly says that trend and seasonal patterns are adjusted through 2027. What remains undisclosed is where and how that adjustment enters the system. I therefore treat seasonality as a specification-identification problem and compare factor-path seasonality, deterministic calendar terms and seasonal error dynamics separately.

### Why do you think seasonality matters, and how did you test it?

I separate seasonality in prices, rents, and transaction volume. Transaction activity is especially exposed to the moving timing of Lunar New Year and to business-day counts. I compare Fourier terms, monthly dummies, and official-calendar exposure measures at the same forecast origins, and I avoid future leakage such as applying STL to the full sample before validation.

### The source says it keeps only statistically significant variables. Did you do the same?

No. Because the original selection procedure and threshold were not disclosed, I did not mimic it with arbitrary stepwise deletion by p-value. Instead, I define economic hypotheses in advance, perform selection inside the training folds, and reserve a final holdout. After selection, I do not present conventional p-values as if they were fully confirmatory evidence.

### Is this forecast operationally usable?

The current code provides pseudo-OOS evaluation in which future X is generated from the last observed value in the training sample. Oracle evaluation using realized future X is labeled separately. But because the current vendor history is latest-vintage, I do not call the result a real-time release-vintage forecast.

### Why did you add a growth model?

The level series are I(1) and the level regression is not cointegrated (Engle–Granger p ≈ 0.98), so the level coefficients may be picking up shared trends. The GLSAR AR coefficient of 0.993 also says the model is effectively in changes already, and its innovations keep price momentum (Ljung–Box fails; AR(2) does not fix it). A growth model with one lag of price growth makes no long-run level claim, captures momentum (φ ≈ 0.61), and cuts residual autocorrelation sharply. Fed (−) and transactions (+) keep their signs, while rent drops out, which tells me the rent link was mostly a shared trend.

### Why can't any model beat a random walk beyond one month?

The home-price index is highly persistent, so the random walk is a strong benchmark. Future X is held at its last value, so the models cannot see future rate or market moves. The growth model also overshoots at turning points. I keep the random walk in every scorecard rather than hiding it. The next refinements are damped momentum, Diebold–Mariano tests on the 23 origins, and empirical intervals.

### What would be the first variable you would add?

Three Hong Kong-specific channels, each entered into the growth model as a change and kept only if it beats both the growth model and the random walk out of sample:

1. **Government policy stance.** Build a stamp-duty and LTV event layer plus a text-tone index from official releases. Every tightening came after a 6-month price rise of 11–13%, so stance is endogenous; I would use the surprise component or forward-looking tone.
2. **Foreign finance and legal firms.** Use regional-office counts, SFC licensees, Grade A office take-up, and a fixed-panel hiring series. I would test them on large units and rent, because Classes D–E rose only +58% since 2010 vs. +132% for A–C and follow their own cycle.
3. **District restaurant activity.** At the city level the signal is weak and even negative at a 6-month lead, so I would use it as a district panel against regional prices.

HIBOR and mortgage flows are the parallel, conventional additions.

### What if the level model is a spurious regression?

That is what I found. ADF/KPSS say the main series are I(1), and Engle–Granger does not support cointegration, so the ECM is withheld by a gate in the code. I keep the level model only as the BI reconstruction and use the growth model for short-run forecasts and scenarios. Engle–Granger has low power and a structural break could hide a long-run relation, so Johansen and Gregory–Hansen tests are the next check.

### Are the scenarios causal effects?

No. They are conditional associations under a fitted equation with coefficients and the historical state held fixed. Exogeneity of the policy shock and coherence of the path are not separately identified. Uncertainty in future X and in model parameters should also be discussed separately from the default interval bands.

### Does the project still have value if the model does not beat the baseline?

Yes. A valid research project can conclude that there is no stable improvement or that an extension does not help at a given horizon. The value here is in preventing leakage, disclosing failed origins, separating feature contribution from model contribution, and building an operational diagnostic and scenario tool. I do not rebrand implementation quality as forecasting superiority.

## 3. Evidence card to keep for the supplied-data experiment

| Question | What to record |
|---|---|
| Data | Actual tickers, fields, units, reference dates, publication dates, vintages, and sample |
| Evaluation | Origins, training window, horizons, X-path generation rule, failed origins |
| Performance | MAE/RMSE/MASE versus naive benchmarks, coverage, interval width, and regime stability |
| Research contribution | Seasonality ablation, new-variable ablation, new-model ablation |
| Limitations | Simultaneity, undisclosed source details, missing release history, and out-of-range scenarios |
| Deliverables | Reproducible code, versions, settings, data hashes, a model card, and the dashboard |

The goal is not to say “I matched the source better.” The goal is to explain exactly what was validated, under which sample and assumptions.
