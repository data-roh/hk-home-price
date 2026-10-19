# Hong Kong Property Models: Methodology Notes

Reference notes behind notebook 01. The notebook is the short version; this file holds the derivations.

## 0. Fix the research question first

This project has three goals:

1. reconstruct the disclosed model families;
2. evaluate their explanatory power, forecasting ability, and stability;
3. build scenarios conditional on future explanatory-variable paths.

Estimating regression coefficients, making forecasts, and identifying causal effects are different questions. The conditional forecast

\[
E[Y_{t+h}\mid\mathcal I_t,X_{t+1:t+h}=x^*]
\]

is not the interventional quantity \(E[Y_{t+h}\mid do(X)=x^*]\). The code contains no exogeneity design for policy shocks and no broader causal-identification strategy. The supplied BQNT monthly values are latest-vintage; BI's original coefficients and release-vintage history were not provided. `src/hkproperty/demo.py` generates synthetic data for software tests only. All reported results in `outputs/` use the supplied real data.

## 1. The contract for observation unit, measurement unit, and time

\(t\) is monthly. The Home target is the RVD All Classes price index. The Office target would be Colliers Hong Kong Overall net effective rent. Housing transactions, a housing price index, and office rent are not interchangeable dependent variables.

Aggregation matters:

- Aggregating daily HSI, market-cap, and FX series to month-end gives different regressors than monthly averages.
- A monthly average of the effective Fed funds rate reflects exposure to policy changes during the month; a month-end value reflects only the last day.

The source does not disclose its aggregation rule, so the code records the mean/last choice as a sensitivity point.

For every series, distinguish:

1. `reference_period`: the calendar month the economic quantity refers to.
2. `vendor_date`: the DATE returned by BQL. Whether this is the reference month or a publication date still needs to be verified.
3. `published_at`: the source's publication timestamp.
4. `first_seen_at` / `retrieved_at`: when the collection system captured the value.
5. `vintage_id`: the version identifier that distinguishes revisions.

Two consequences:

- **Vintage.** Extracting historical PX_LAST today does not recreate the value that was available at the original publication date, because the source may have been released late or revised.
- **Gaps.** If internal missing months are removed with `dropna()`, observations two months apart become adjacent to an AR(1) model. The code therefore trims only to the outer common window and raises an error on internal gaps.

## 2. A statistical representation that nests the source model

\[
y_t=a+\sum_{j=1}^{k}\sum_{\ell\in L_j}\beta_{j\ell}x_{j,t-\ell}+d_t'\gamma+u_t.
\]

- \(y\) and \(x\) are in native units or a pre-declared transformation.
- \(d_t\) collects seasonal, calendar, and policy variables.
- For Home, the source discloses an AR covariance structure for \(u_t\). For Office, it discloses AR = 2 and MA = 1.
- Transformations, lags, and seasonal orders not confirmed by the source are explicit settings.

**Standardisation.** For numerical stability, \(z_{jt}=(x_{jt}-\bar x_j)/s_j\), with \(\bar x_j\) and \(s_j\) estimated inside each training fold only. Native-unit coefficients are recovered as \(\beta_j=b_j/s_j\), and the native-unit intercept as \(a=b_0-\sum_j b_j\bar x_j/s_j\). The code reports the centred-design intercept and the native-unit slopes separately.

**Log models.** In \(\log P_t=a+\beta_r r_t+\beta_H\log H_t+u_t\):

- a 1-percentage-point change in the rate \(r\) gives a conditional price change of \(100(e^{\beta_r}-1)\%\);
- a 1% change in \(H\) gives \(100[(1.01)^{\beta_H}-1]\%\).

The approximation "\(\beta_H\%\)" is valid only for small changes. A 100bp rate move is 1pp, not +100 on the raw series. Percentage variables such as CPI YoY and the unemployment rate are not logged by default.

## 3. OLS, GLS, and the assumptions they need

In matrix form \(y=X\beta+u\), and

\[
\hat\beta_{OLS}=(X'X)^{-1}X'y.
\]

**OLS under autocorrelation.** If \(E[u\mid X]=0\) and suitable sample conditions hold, OLS slopes can be unbiased or consistent even with autocorrelated errors. But autocorrelation correction does not restore consistency under:

- simultaneity,
- omitted variables, or
- shared non-stationary trends.

Under autocorrelation the true OLS covariance is

\[
\operatorname{Var}(\hat\beta_{OLS}\mid X)
=(X'X)^{-1}X'\Omega X(X'X)^{-1}.
\]

If \(\Omega\neq\sigma^2I\) but independent-error standard errors are used, t and F inference can be wrong. HAC (Newey–West) estimates the long-run covariance consistently. It does not fix endogeneity and does not create an AR forecast state.

**GLS.** With \(\Omega\) known and positive definite,

\[
\hat\beta_{GLS}=(X'\Omega^{-1}X)^{-1}X'\Omega^{-1}y,
\qquad
\operatorname{Var}(\hat\beta_{GLS}\mid X)=(X'\Omega^{-1}X)^{-1}.
\]

A correct covariance model buys efficiency; an incorrect one distorts inference and forecasts. If a whitening matrix \(W\) satisfies \(W\Omega W'=I\), GLS equals OLS of \(Wy\) on \(WX\).

## 4. AR(1) error covariance and FGLS

\[
u_t=\rho u_{t-1}+\epsilon_t,\quad |\rho|<1,
\quad E[\epsilon_t]=0,\quad \operatorname{Var}(\epsilon_t)=\sigma_\epsilon^2.
\]

The stationary AR(1) covariance is

\[
\Omega_{ij}=\frac{\sigma_\epsilon^2}{1-\rho^2}\rho^{|i-j|},
\]

and conditional whitening gives

\[
y_t-\rho y_{t-1}=(1-\rho)a+(x_t-\rho x_{t-1})'\beta+\epsilon_t.
\]

**Initial conditions.** Variants of "AR(1)" can give different finite-sample coefficients because they treat the first observation and the objective function differently:

| Method | First observation |
|---|---|
| Exact stationary Gaussian GLS | Weighted by the stationary distribution |
| Prais–Winsten | First row scaled by \(\sqrt{1-\rho^2}\) |
| Conditional Cochrane–Orcutt family | Dropped |

**Implementation.** The code uses statsmodels `GLSAR`, an iterative feasible GLS:

1. fit the regression;
2. estimate the AR coefficients from the residuals;
3. whiten and re-fit;
4. repeat until convergence.

GLSAR whitening drops the first \(p\) observations, so it cannot be assumed to be BI's exact GLS. Yule–Walker AR estimation and joint Gaussian ML are also different procedures. If coefficients differ from the source, check the estimation objective and initial conditions, not only the data and transformations.

**Non-stationary estimates.** If the estimated AR coefficient is not stationary, the code does not clip it into range; the specification must be revisited (trend, differencing, cointegration, breaks). On the supplied data \(\hat\rho\approx0.993\). Beyond "prices are persistent", that is a warning sign of a misspecified level regression, which is consistent with the failed cointegration test in §9.

## 5. AR errors are not a lagged dependent variable

Expanding

\[
y_t=a+\beta x_t+u_t,\quad u_t=\rho u_{t-1}+\epsilon_t
\]

gives

\[
y_t=(1-\rho)a+\rho y_{t-1}+\beta x_t-\rho\beta x_{t-1}+\epsilon_t.
\]

This differs from the unrestricted \(y_t=c+\rho y_{t-1}+\beta x_t+\epsilon_t\): the AR-error model imposes the restriction \(-\rho\beta\) on lagged \(x\). Mechanically multiplying the AR-error slope by \(1/(1-\rho)\) invents a long-run multiplier that the model does not contain.

Holding the past error state fixed and changing only the future \(x\) path,

\[
\hat y^*_{T+h}-\hat y^0_{T+h}=\hat\beta'(x^*_{T+h}-x^0_{T+h}).
\]

If the mean equation has lagged regressors, apply those lags. Error persistence does not amplify a scenario shock. This is why the level GLSAR scenario in notebook 01 §6.1 is a flat +2.6% for a Fed −100bp shock.

The growth model is different: it has an explicit lag of \(\Delta\log y\), so a permanent shock builds up to \(-\hat\gamma/(1-\hat\phi)\) (≈ 3.1% for Fed −100bp).

## 6. Conditional forecasts from the Home GLSAR

For AR(1),

\[
\hat y_{T+h|T}=x_{T+h}'\hat\beta+\hat\rho^h\hat u_T.
\]

Showing only \(X\beta\) from `GLSAR.predict` drops the forecast of the last residual state, so the package adds the AR residual mean recursively. AR(p) uses the same recursion on the last \(p\) residuals.

**Mechanical drift.** With \(\hat u_T\approx-85\) and \(\hat\rho\approx0.993\), the baseline drifts slowly upward (≈ 320 → 327 over 12 months). That drift is the residual decaying, not a market view.

**Interval variance.** With fixed parameters and fixed future X, the AR(1) \(h\)-step innovation variance is

\[
V_h=\hat\sigma_\epsilon^2\sum_{j=0}^{h-1}\hat\rho^{2j}.
\]

For AR(p), write \(u_t=\sum_{j=0}^\infty\psi_j\epsilon_{t-j}\) with \(\psi_0=1\), so \(V_h=\sigma_\epsilon^2\sum_{j=0}^{h-1}\psi_j^2\).

This variance excludes uncertainty in the AR coefficients and \(\beta\), and treats the initial residual state as fixed. It is not the full forecast risk. That is one reason the empirical 95% coverage of the GLSAR intervals is only 43–74%.

## 7. Reading SARIMAX correctly

The Office model is a regression with SARIMA errors:

\[
y_t=X_t'\beta+u_t,
\]
\[
\phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D u_t
=\theta(B)\Theta(B^s)\epsilon_t.
\]

**Notation.** \(Bu_t=u_{t-1}\), and \(s=12\) is the usual monthly candidate. \(\phi(B)=1-\phi_1B-\phi_2B^2\) and \(\theta(B)=1+\theta_1B\). Check the MA sign convention of the software.

**What the source supports.** The disclosed two AR terms and one MA term justify \(p=2, q=1\). The screen does not justify \(d=0\), \(D=0\), or \(P=Q=1\). The minimal baseline is \(d=0\) with no seasonal ARMA; seasonal regressors are compared separately. First check whether the raw data are already seasonally adjusted and whether trend adjustment applies only to the exogenous forecast paths.

**Seasonal polynomials.** A seasonal polynomial is not the same as adding a lag-12 term. For example, \((1-\phi B)(1-\Phi B^{12})\) produces lag 1, lag 12, and the cross term lag 13. A 12-month difference takes the change between the same months of consecutive years, which is not the same operation as a deterministic month effect.

**Constants.** The regression intercept is a column of exog with `trend='n'`, so its meaning matches the Home mean equation. A SARIMAX `trend='c'` constant can be tied to the AR parameterisation, so the same symbol \(a\) can imply a different long-run mean. With positive differencing order, a level constant may be unidentified and a drift design is needed. The current code assumes \(d=0\).

## 8. State space, Kalman filter, and maximum likelihood

ARMA errors can be written in state-space form by putting the needed AR/MA memory into the state vector \(\alpha_t\):

\[
y_t=X_t'\beta+Z_t\alpha_t+\varepsilon_t,
\qquad
\alpha_{t+1}=T_t\alpha_t+R_t\eta_t.
\]

With scalar one-step prediction error \(v_t=y_t-\hat y_{t|t-1}\) and variance \(F_t=\operatorname{Var}(v_t\mid\mathcal I_{t-1})\), the Gaussian log likelihood is

\[
\ell(\vartheta)=-\frac12\sum_t\{\log(2\pi)+\log F_t+v_t^2/F_t\}.
\]

The filter loops: predict the state → observe the innovation → update. For the same innovation size, a smaller \(F\) costs more likelihood. Numerical optimisation fits \(\beta\), AR, MA, seasonal coefficients, and variances to this likelihood. If the distribution is wrong the estimates can be read as quasi-ML, but valid inference needs further conditions.

**Implementation choices.**

- Standardise X on the training sample only.
- Impose stationarity and invertibility.
- Store convergence status and warnings, and hide forecasts from non-converged fits.

Convergence does not guarantee a global optimum, economic validity, white-noise residuals, or forecasting skill. Multiple starting values, optimisers, and profile likelihoods are follow-up checks.

**Comparing fits.**

- Hessian-based and OPG standard errors can differ in finite samples.
- Do not compare AIC across FGLS and ML fits with different initial conditions, effective samples, or transformations.
- Comparing a log-\(y\) likelihood with a level-\(y\) likelihood requires the Jacobian of the transformation.

In practice this project ranks models by out-of-sample loss in native units.

## 9. Non-stationarity, spurious regression, and cointegration

House prices, rent indices, and equity indices can share trends. A large \(R^2\), small conventional p-values, and smooth long-run forecasts do not validate an economic relation.

**Unit-root tests.**

| Test | Null hypothesis |
|---|---|
| ADF | Unit root |
| KPSS | Stationarity |

Both are sensitive to trend terms, lag or bandwidth choice, structural breaks, and sample length, and they can disagree. They are not an automatic differencing switch. Ordinary unit-root tests also do not diagnose seasonal unit roots.

**Cointegration.** If the variables are I(1) and

\[
z_t=\log P_t-c-\theta'\log X_t\sim I(0),
\]

a cointegrating relation is a candidate. The Engle–Granger residual unit-root test uses MacKinnon critical values, not the standard ADF ones. The code's ECM candidate runs `coint` first and stops if cointegration is not supported. A significant p-value would still not verify the stability of the relation, the I(1) status of each variable, or the absence of breaks.

**On the supplied data:**

- `home_price`, `hsi`, `home_rent`, and `fed_rate` are I(1); `secondary_sales` and `cnyhkd` are ambiguous; all first differences are stationary.
- Engle–Granger on the level regression: ADF statistic ≈ −1.9 against a 5% critical value ≈ −4.8 (p ≈ 0.98).
- The ECM gate on log levels: p ≈ 0.92.
- The ECM is withheld, and the growth model is the short-run specification. Johansen and Gregory–Hansen (break-robust) tests are next, because Engle–Granger has low power and a break could hide a long-run relation.

**Error-correction form** (with \(p=\log P\)):

\[
\Delta p_t=\alpha[p_{t-1}-c-\theta'x_{t-1}]
+\sum_i\psi_i\Delta p_{t-i}+\sum_j\Gamma_j'\Delta x_{t-j}+\delta'W_t+\epsilon_t.
\]

With this sign convention, \(\alpha<0\) means restoring adjustment. The range \(-2<\alpha<0\) is the stability condition only for a very simple univariate adjustment equation, not for the full model with several lags. Separating the long-run relation from short-run changes can be more principled than "always difference instead of levels".

## 10. Endogeneity, simultaneity, and causal limits

Housing transactions and prices, and vacancy and rents, react to the same demand and supply shocks.

- Adding lags does not guarantee exogeneity.
- Mixing AR errors with lagged \(y\) can correlate lagged \(y\) with the current error.
- FGLS changes the error covariance; it does not create an instrument.

A causal goal needs, first: the target shock, a control group or instrument, a case for policy exogeneity, and exclusion restrictions. Candidates include high-frequency rate surprises, event designs around specific policies, and regional variation. This package implements none of these.

Two concrete cases from notebook 01:

- **Fed rate.** Under the HKD peg, the Fed rate is not exogenous to Hong Kong.
- **Hong Kong cooling measures.** These lean against the wind (§9.1 of the notebook): tightening followed six-month price rises of 11–13%. Their naive regression coefficient would be biased toward zero or have the wrong sign.

Scenario results are described as "the conditional difference implied by the fitted equation, holding coefficients and other paths fixed".

## 11. Seasonality: four separate questions

### Get the research question right first

The source says that both models automatically incorporate the latest actuals and adjust trend and seasonal patterns through 2027. Starting from "the original model ignored seasonality" would misrepresent the source. The right question is not **whether** seasonality was handled, but **where and in what form**, and whether that choice improves out-of-sample performance and interval calibration.

Final forecasts alone cannot identify where seasonality enters. A seasonal \(X_{t+h}\) path in a non-seasonal regression and a flat \(X_{t+h}\) path plus a deterministic seasonal term \(s_{t+h}\) can look alike in a given sample. Allowing pre-adjustment and seasonal ARMA as well makes this observational equivalence larger. Without the original code, design matrix, factor paths, or coefficient restrictions, the screen cannot pin down one specification.

This project therefore treats four locations as separate ablations, and does not read an improvement in one as evidence about another:

1. pre-adjustment of the raw data or of individual \(X\);
2. seasonal forecasts of future \(X\) paths;
3. deterministic seasonal terms in the \(Y\) mean equation;
4. stochastic seasonality in the innovation process.

### Deterministic seasonality

Monthly dummies (11 coefficients):

\[
s_t=\sum_{m=2}^{12}\gamma_m1\{month(t)=m\}.
\]

Fourier terms give a smooth annual pattern with few degrees of freedom:

\[
s_t=\sum_{k=1}^K[a_k\sin(2\pi kt/12)+b_k\cos(2\pi kt/12)].
\]

- Start from \(K=1\) or \(2\). At monthly frequency the highest-frequency sine term degenerates, so do not raise \(K\) carelessly.
- Fourier terms capture a fixed calendar-month pattern; they do not track the moving Lunar New Year.
- Result: Fourier K=1 is the best GLSAR variant out of sample, but only marginally, so seasonality is not the bottleneck.

### Stochastic seasonality

Residual ACF at lags 12 and 24, seasonal ARMA, and seasonal differencing are a separate question. First check whether monthly dummies already explain the pattern, and add a seasonal AR term only if needed. Stacking dummies, many Fourier terms, and seasonal differencing without evidence causes redundancy, over-differencing, and instability.

### Calendar exposure

- **Business days.** Transaction counts \(Q\) depend on the number of trading days \(N\) and underlying activity \(\lambda\). Since \(E[Q_t]\approx N_t\lambda_t\), consider business-day counts or a log offset. Do not apply the same offset to price or rent indices.
- **Lunar New Year.** It falls in January or February, so a February dummy may not be enough. Build an exposure measure, for example the share of the window ±7 days around the official Lunar New Year date that falls in each month. If registrations lag transactions, compare adjacent-month lags.
- **Data hygiene.** The future calendar is known, but coefficients must still be chosen inside the training window. Use the official calendar; do not generate holiday dates.

### Where seasonal adjustment enters, and leakage

Seasonally adjusting X, forecasting future X seasonally, adding seasonal terms to the \(y\) mean equation, and adding seasonal AR to the errors are different operations. Full-sample STL, centred moving averages, and any filter that uses future observations leak information if applied outside the training fold. Seasonal-adjustment vintages, including revisions, also need tracking.

Interview phrasing: not "I added seasonality missing from the original model", but "The source disclosed seasonal adjustment but not its location or specification, so I designed leakage-controlled challengers that separate seasonal factor paths, deterministic calendar terms, and seasonal error dynamics."

## 12. Variable selection and meaningful significance

The phrase "economic and statistical significance" does not define a reproducible procedure. Keeping variables with full-sample p < 0.05 and then reporting p-values and performance on the same data creates selection bias and multiple-testing problems. The correct procedure:

1. pre-register the economic hypotheses;
2. choose variables, lags, and seasonal orders inside the training window;
3. use the final holdout once.

**Collinearity.** VIF and condition numbers diagnose multicollinearity. A high VIF is not an automatic reason to drop a variable: individual coefficients of correlated variables can be unstable while their combined prediction is not. Conversely, good forecasts do not justify reading individual coefficients. Look at signs, magnitudes, rolling stability, and ablations together.

**Example from this project.** Land Registry agreements are significant in sample (p ≈ 0.02 and 0.001) but add nothing out of sample, because they are a 0.99-correlated substitute for Midland sales.

## 13. Forecast evaluation: four experiments not to confuse

| # | Experiment | What it measures |
|---|---|---|
| 1 | In-sample fit | Explanatory power within the estimation sample |
| 2 | Oracle-X holdout | Conditional fit of \(Y\) given realised future X. Not forecasting performance. |
| 3 | Forecast-X pseudo-OOS | Uses paths built from past X at each origin. Still not point-in-time if run on latest-vintage data. |
| 4 | Release-vintage real-time OOS | Uses only data published and available at the time, with the target limited to what was observable |

**What the code does.** The default is experiment 3 (`PSEUDO_OOS`):

- re-fit the scaler and the model at every origin, using an expanding window from month 120 with a step of 3;
- 23 origins on the Home sample;
- future X follows a last-value path (the source's own X-forecast method is undisclosed);
- oracle runs are labelled in names and output files and kept out of the official scorecard.

A bulk BQL history alone does not achieve experiment 4.

**Design choices.**

- Score h = 1, 3, 6, 12 and compare expanding and rolling windows. A single chronological train/test split can land in a lucky regime, and random K-fold destroys time dependence and information timing.
- Do not hide failed origins; compare models on their common successful origins. Many dropped origins would itself show a model is not operationally stable.

**Loss functions.**

\[
MAE_h=N_h^{-1}\sum_o|y_{o+h}-\hat y_{o+h|o}|,
\quad RMSE_h=\sqrt{N_h^{-1}\sum_o e_{o,h}^2}.
\]

- The MASE denominator is the mean absolute seasonal-naive error within the training window; out-of-sample \(Y\) is never used for it.
- Do not rank Home index RMSE against HKD rent RMSE.
- Always include the random walk and seasonal naive, to control for the apparent accuracy created by high level persistence.

**Result.** The growth model wins only at h = 1; the random walk wins at h = 3, 6, 12.

**Significance of rankings.** Overlapping \(h\)-step losses are autocorrelated. Diebold–Mariano standardises the mean loss differential \(d_o=L(e^A_o)-L(e^B_o)\) by its long-run variance. With few origins, nested models, and many candidates searched, the approximation and selection bias matter. Do not pick a winner on a DM p-value alone; non-overlapping-origin sensitivity and block bootstrap are further checks.

## 14. Residual checks and structural stability

In ARMA models the raw regression error \(u\) and the one-step innovation \(\epsilon\) differ, so state which one is being diagnosed. GLSAR uses whitened residuals; SARIMAX uses standardised one-step prediction errors after burn-in.

| Check | What it diagnoses | Note |
|---|---|---|
| Ljung–Box | Remaining autocorrelation | State the lags and degrees-of-freedom adjustment |
| ARCH LM | Conditional heteroskedasticity | A good mean fit does not imply a stable variance |
| QQ / Jarque–Bera | Tails and distribution | Rejecting normality does not invalidate the mean forecast, but it affects Gaussian interval coverage |
| Rolling β, recursive residuals, CUSUM, break candidates | Coefficient stability | |
| Performance by regime (crisis, pandemic, before/after policy) | Regime dependence | |

**On the supplied data:**

- GLSAR innovations fail Ljung–Box (ACF 0.49 at lag 1, 0.36 at lag 2) and ARCH LM, and pass Jarque–Bera.
- AR(2) errors do not help.
- The explicit growth lag reduces Ljung–Box(12) from about 99 to about 20.

"Diagnostic p > 0.05" does not prove the null. Searching over many lags, periods, and models also selects the test results. Judge residual tests, out-of-sample loss, economic logic, and operational missingness together.

## 15. Scenarios and the decomposition of uncertainty

\[
\operatorname{Var}(Y_{T+h}\mid\mathcal I_T)
=E[\operatorname{Var}(Y_{T+h}\mid X^*,\vartheta,\mathcal I_T)]
+\operatorname{Var}(E[Y_{T+h}\mid X^*,\vartheta,\mathcal I_T)].
\]

This simplified law of total variance puts uncertainty about the X path, the parameters, and the state into the second term. Model-selection and structural-change risk are not captured automatically.

The app's default band is an observation interval conditional on fixed X and the estimated parameters. It is not a confidence interval for \(\beta\), and not a scenario probability that reflects all risks.

**Log forecasts.** If \(\log Y\sim N(m,V)\), the median is \(e^m\) and the mean is \(e^{m+V/2}\). The app displays the median, and the CSV also provides the mean. Mixing them biases level-scale forecasts.

**Better intervals.** Realistic intervals would combine:

- parameter re-estimation bootstrap;
- residual block bootstrap;
- joint simulation of the explanatory-variable paths.

Combining independent shocks while ignoring correlation among X and policy reactions can produce implausible paths. Out-of-range scenarios are flagged separately, and staying within the historical min/max does not by itself make a scenario economically plausible.

## 16. What can be said in an interview

"I separated the disclosed model family from its actual specification and did not fill undisclosed items in as facts. I separated the reference month from the publication date and did all transformations and estimation inside the training folds. I tested whether the level relation is cointegrated; it is not, so I added a growth model with momentum as the short-run specification. I wrote the scenario equations so that AR-error persistence is not mistaken for a causal multiplier. I split seasonality into calendar effects, deterministic seasonality, and stochastic seasonality. Finally, I benchmarked everything against a random walk on common origins and distinguished conditional fit, pseudo-OOS, and real-time point-in-time evaluation."

**Done:**

- model-family reconstruction;
- estimation on the supplied Home data;
- a cointegration check and the growth challenger;
- common-origin pseudo-OOS comparison;
- the scenario app.

**Not claimed:** real-time forecasting performance or numerical replication of BI. There are no release vintages and the original specification is undisclosed. The Office model is withheld because the `office_rent` target is missing.

## References

- User-supplied BI methodology screenshots (2026-10-17).
- statsmodels GLSAR: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.html
- GLSAR whitening: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLSAR.whiten.html
- SARIMAX API: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- SARIMAX intercept / exog FAQ: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_faq.html
- Forecasting: Principles and Practice, dynamic regression: https://otexts.com/fpp3/dynamic-regression.html
- Calendar/Fourier predictors: https://otexts.com/fpp3/useful-predictors.html
- statsmodels cointegration API: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
- Newey, W.K. and West, K.D. (1987), A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix, Econometrica.
- Engle, R.F. and Granger, C.W.J. (1987), Co-integration and Error Correction: Representation, Estimation, and Testing, Econometrica.
- Diebold, F.X. and Mariano, R.S. (1995), Comparing Predictive Accuracy, Journal of Business & Economic Statistics.
