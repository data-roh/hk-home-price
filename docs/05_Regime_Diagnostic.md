# Regime Diagnostic: Block Moments vs Sampling Error

Reference notes for the dashboard's **Regime** tab. The tab implements a simple, largely
non-parametric diagnostic for whether the monthly home-price series contains more than one
(mean, variance) regime. This document records the logic behind every control, every plot
element, and every statistic shown, plus the theory that justifies them.

## 1. Where this sits among regime methods

| | This diagnostic | Markov-switching model | State-space model |
|---|---|---|---|
| Type | Non-parametric (moment-based) descriptive test | **Parametric** | **Parametric** |
| What is assumed | Moments exist; blocks are long enough for the CLT | Per-regime distributions (usually Gaussian), a fixed number of regimes, a Markov transition matrix | Observation and state equations with specified error distributions |
| What it delivers | Evidence that ≥ 2 regimes exist (yes/no with a p-value) | Number-of-regime fit, regime probabilities per month, transition dates, forecasts | Latent state estimates, filtering/smoothing, forecasts |
| Failure mode | Low power with few blocks; block-length sensitivity | Misspecified likelihood silently gives wrong regimes | Misspecified dynamics bias the state estimates |

The intended workflow is: run this diagnostic first; only if it signals two regimes is it
worth paying the specification cost of a Markov-switching or state-space model to estimate
*when* the switches happened and with what probabilities. The diagnostic itself never
estimates transition dates or probabilities.

## 2. Pipeline

1. **Returns.** Monthly log returns $r_t = 100\,(\ln P_t - \ln P_{t-1})$ of the home-price
   index. Levels are non-stationary, so moments of levels are not comparable across time;
   returns make the single-regime null meaningful.
2. **Blocks.** The return series is cut into **non-overlapping** blocks of $n$ months,
   aligned so the latest month ends the last block (any remainder is dropped from the
   start). Non-overlapping blocks keep the block statistics (nearly) independent, which the
   distance interpretation requires. At least 4 complete blocks are required.
3. **Block moments.** Each block $i$ contributes a sample mean $\bar r_i$ and a sample
   variance $s_i^2$ — one point in the (mean, variance) plane.
4. **Single-regime null.** The pooled mean $\hat\mu$ and pooled variance $\hat\sigma^2$ over
   all usable months define the null centre, with sampling errors (Section 3).
5. **Standardise.** $z_i = \left(\frac{\bar r_i - \hat\mu}{SE_{\bar r}},\;
   \frac{s_i^2 - \hat\sigma^2}{SE_{s^2}}\right)$.
6. **Cluster.** k-means with $k = 2$ splits the standardised points into clusters A and B;
   the centroid distance $D$ measures separation in sampling-error units.
7. **Permutation test.** The same statistic is recomputed on block-permuted series to get
   the null distribution of $D$; the p-value is the share of permuted $D$ at least as large.
8. **Kurtosis cross-check.** Pooled vs within-block excess kurtosis distinguishes
   "two variance regimes" from "one fat-tailed distribution".

## 3. Theory

### 3.1 Sampling distributions of the block statistics

For $n$ draws from one regime with mean $\mu$ and variance $\sigma^2$:

$$\bar r \;\dot\sim\; \mathcal N\!\left(\mu,\ \tfrac{\sigma^2}{n}\right)
\qquad\text{(central limit theorem)},$$

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
\;\Rightarrow\;
s^2 \;\dot\sim\; \mathcal N\!\left(\sigma^2,\ \tfrac{2\sigma^4}{n-1}\right)
\qquad\text{(near-normal returns)}.$$

So the sampling errors are $SE_{\bar r} = \sigma/\sqrt{n}$ and
$SE_{s^2} = \sigma^2\sqrt{2/(n-1)}$. For normal data $\bar r$ and $s^2$ are independent
(Basu's theorem), so under a single regime the block points scatter around
$(\mu, \sigma^2)$ in an **axis-aligned ellipse** whose half-widths are exactly these
sampling errors. Sampling error is the yardstick: it quantifies how far apart two block
points can sit *by chance alone*.

### 3.2 Autocorrelation correction (the AR(1) toggle)

Monthly price-index returns are strongly autocorrelated (index smoothing; lag-1
autocorrelation ρ ≈ 0.67 in the current data). Positive autocorrelation means each block
contains fewer independent observations than its nominal length, so the naive standard
errors above are too small and would manufacture false regimes. The standard first-order
correction replaces $n$ with the effective sample size

$$n_{\text{eff}} = n\,\frac{1-\hat\rho_1}{1+\hat\rho_1},$$

which follows from the variance of the mean of an AR(1) process. With ρ ≈ 0.67 a 24-month
block carries only ≈ 4.7 effective observations — the correction is large, not cosmetic.
The same $n_{\text{eff}}$ is applied inside $SE_{s^2}$ as a first-order approximation
(the exact variance-of-variance correction under dependence is messier and would exceed
the precision this diagnostic needs).

### 3.3 Distance

After standardisation, every block point is approximately standard bivariate normal under
the single-regime null. The separation statistic is the Euclidean distance between the two
k-means centroids in standardised space:

$$D = \lVert \bar z_A - \bar z_B \rVert_2
= \sqrt{\left(\frac{\bar r_A - \bar r_B}{SE_{\bar r}}\right)^2
      + \left(\frac{s^2_A - s^2_B}{SE_{s^2}}\right)^2}.$$

This is a **Mahalanobis distance with a diagonal covariance matrix**: dividing each axis by
its own standard error is exactly the whitening step, and the diagonal form is justified by
the independence of $\bar r$ and $s^2$ under near-normality. The mean axis alone is a
two-sample $t$-statistic; the variance axis alone is an $F$-test in disguise; $D$ combines
both in one number. Alternatives exist — Hotelling's $T^2$ with an estimated full
covariance, or energy/Wasserstein distances between block distributions — but with as few
as 4–16 blocks, estimating a covariance matrix from the points themselves is noisier than
using the theoretical diagonal. The simple $D$ is the right tool at this sample size.

Two caveats make the raw value of $D$ uninterpretable on its own, which is why the
permutation test exists:

* k-means **always** produces two clusters, even in pure noise, so $D > 0$ always;
* the blocks sharing the pooled $\hat\mu,\hat\sigma^2$ induces slight dependence.

### 3.4 Permutation test

Under the null "one regime", the time ordering of the data carries no information about
moments. The null distribution of $D$ is therefore built by destroying the ordering:
the usable months are cut into 6-month sub-blocks, the sub-blocks are shuffled (preserving
short-run autocorrelation inside each sub-block), blocks and $D$ are recomputed with the
identical procedure, and this is repeated $N_{\text{perm}}$ times. The p-value uses the
standard add-one form

$$p = \frac{1 + \#\{D^{\text{perm}} \ge D^{\text{obs}}\}}{1 + N_{\text{perm}}},$$

which is exact-level and never returns 0. Because the null is simulated from the data
itself, no normality or variance-formula assumption enters the final inference — this is
what keeps the overall procedure essentially non-parametric even though normal-theory
standard errors are used as the (shared, order-preserving) scaling.

### 3.5 Kurtosis cross-check: the mixture signature

Mixing two normal distributions with **different variances** produces excess kurtosis even
though each component is exactly normal. For a 50/50 mixture of
$\mathcal N(0,\sigma_1^2)$ and $\mathcal N(0,\sigma_2^2)$:

$$\text{kurtosis} = \frac{6\,(\sigma_1^4+\sigma_2^4)}{(\sigma_1^2+\sigma_2^2)^2} \ge 3,
\qquad\text{with equality iff } \sigma_1=\sigma_2 .$$

This yields a diagnostic that separates two hypotheses that the scatter alone cannot:

| Observation | Reading |
|---|---|
| Pooled excess kurtosis ≫ mean within-block excess kurtosis (positive gap) | Fat pooled tails are *created by mixing* blocks with different variances → supports variance-regime switching |
| Within-block kurtosis also high | The distribution is fat-tailed *inside* each block → a single heavy-tailed regime (e.g. t-distributed returns) explains the data without regime switching |

This is historically why ARCH/GARCH and regime-switching models were introduced for
financial returns: unconditional fat tails with near-Gaussian conditional behaviour.

## 4. The controls

| Control | What it does | Why it matters |
|---|---|---|
| **Block length · months** | Sets $n$, the non-overlapping block size (12–48, step 6 so blocks divide evenly into 6-month permutation sub-blocks) | The central bias–variance trade-off: long blocks → small SEs and high power, but blocks straddle regime boundaries and blur them; short blocks → sharp boundaries but noisy moments. Results must be checked across several lengths; a conclusion that appears only at one length is weak. |
| **Permutations** | Number of block-permuted replicas used to build the null of $D$ (200–2000) | More permutations → smoother null and finer p-value resolution (minimum possible p is $1/(N+1)$). 500 is adequate for a 5% test; 2000 for quoting small p-values. |
| **AR(1)-adjust sampling errors** | Replaces $n$ with $n_{\text{eff}}$ in both SEs (Section 3.2) | Off, the ellipses shrink and the distance inflates — with ρ ≈ 0.67 this manufactures regimes out of smoothing. Leave on unless returns have been pre-whitened. |
| **Rolling trajectory overlay** | Draws overlapping rolling windows of the same length as a grey path in the scatter | Visual context only (see below). Never enters the test. |

## 5. The plot elements

### Scatter (mean–variance plane)

* **Cluster A / Cluster B points** — each point is one complete block; its coordinates are
  the block's sample mean (x) and sample variance (y), labelled by its date span. Colours
  are the k-means assignment in standardised space; A is defined as the cluster with the
  lower mean return so colours are stable across reruns. The clusters are *candidate*
  regimes — they are meaningful only if the permutation p-value is small.
* **Pooled (single-regime null) ×** — the pooled mean and variance of all usable months:
  the centre the blocks would scatter around if one regime generated everything.
* **1·SE and 2·SE ellipses** — the sampling-error ellipses from Section 3.1 around the
  pooled point. Under one regime roughly 39% / 86% of block points should fall inside
  (bivariate normal containment); points far outside the 2·SE ellipse, or two clusters
  separated beyond the permutation threshold, indicate distinct regimes.
* **Rolling trajectory** — the (mean, variance) of an $n$-month window stepped one month at
  a time, drawn as a connected path. It shows *when* the market travelled from one cluster
  to the other, which non-overlapping blocks are too coarse to show. Because consecutive
  windows share $n-1$ months they are heavily dependent: the path's smoothness is an
  artefact, and using these points in the test would overstate the sample — hence
  visual-only.

### Blocks-in-time panel

Two stacked panels sharing the time axis — the **original price index** (top) and the
monthly log returns the blocks are computed from (bottom) — with each block's span shaded
by its cluster colour. Regimes are a temporal claim: contiguous runs of one colour support
regime structure; colours alternating block-by-block point to noise. This panel is the
bridge back to the raw series: it shows which episodes of the actual price history each
cluster corresponds to.

### Null-distribution histogram

The permutation null of $D$ with the observed value (orange line) and the null's 95th
percentile (dashed). The observed distance is evidence of two regimes only when it falls
beyond the bulk of this histogram.

## 6. Limitations

* The test asks only whether **two** clusters separate; it does not estimate the number of
  regimes, transition dates, or probabilities — that is the job of the parametric models in
  Section 1.
* Block-length sensitivity is real: in the current data, 24-month blocks are not
  significant while 30-month blocks are. Report conclusions across lengths.
* $SE_{s^2}$ assumes near-normal blocks; the AR(1) correction is first-order; the index's
  smoothing inflates ρ. The SEs are order-preserving scalings, not exact standard errors —
  the permutation p-value, not the ellipses, carries the inference.
* Moment-based kurtosis is sensitive to single outliers; with heavier contamination,
  quantile-based spread/tail measures (IQR/MAD, Moors kurtosis) or distribution distances
  (Kolmogorov–Smirnov, Wasserstein) between blocks are more robust substitutes.
* The current implementation is univariate (price returns only); the same construction
  extends to any stationary transform of the dashboard's other series.
