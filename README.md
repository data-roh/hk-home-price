# Hong Kong Home-Price Model: Reconstruction, Challengers, and Scenario Lab

**Live App:** [HK Home · Scenario Lab](https://hk-home-price.streamlit.app/)

A self-contained research package built around the publicly disclosed methodology of the Bloomberg Intelligence (BI) Hong Kong home-price model. It:

1. reconstructs the disclosed model family (GLS with autoregressive errors) on user-supplied monthly BQNT data;
2. tests whether the level relation is valid, and adds a growth-rate challenger;
3. benchmarks every model against a random walk under a common pseudo-out-of-sample protocol;
4. serves conditional scenarios through a Streamlit app; and
5. proposes the next Hong Kong-specific variables to collect.

It is **not** a copy of BI's code, coefficients, or forecasts. Those were never available. The BQNT runtime is not required; the supplied CSV files are fixed inputs.

## Headline results

Sample: 199 complete months, 2010-02 to 2026-08. Out-of-sample: 23 expanding-window origins, 2020-01 to 2025-07, re-fitted every 3 months from month 120, future X held at its last value.

| Question | Answer |
|---|---|
| Is the level regression valid? | Prices and most predictors are I(1) and **not cointegrated** (Engle–Granger p ≈ 0.98; ECM gate p ≈ 0.92). The level GLSAR is kept as a reconstruction, not as a long-run relation. |
| What does the level GLSAR(1) say? | $\hat\rho \approx 0.993$, so it behaves like a regression on monthly changes. Fed rate −8.4 pts per 1pp; secondary sales +1.2 pts per 1,000 deals; rent +1.1 pts per index point; HSI and CNYHKD not significant. |
| What does the growth model say? | $\Delta\log y_t = c + \phi\,\Delta\log y_{t-1} + \Delta X_t\gamma$ with HAC errors: momentum $\phi \approx 0.61$, $R^2 \approx 0.59$. Fed (−) and transactions (+) survive; rent does not. |
| Are the residuals clean? | GLSAR innovations fail Ljung–Box and ARCH; AR(2) does not fix it. The growth lag does (Ljung–Box(12) ≈ 99 → ≈ 20). |
| Fed −100bp scenario | Level GLSAR: +2.6% at once. Growth model: +1.2% in month 1, building to ≈ +3.1%. Conditional projections, not causal effects. |
| Out-of-sample | Growth model wins at h = 1 (RMSE 4.53 vs. random walk 5.38). **Nothing beats the random walk at 3–12 months.** Growth-model intervals are best calibrated (87/91/83/70% vs. GLSAR 74/70/74/43%). |
| Extra variables tested | Land Registry transaction counts and values are significant in sample but give no out-of-sample gain (19 common origins, 2011+ sample). |
| What next? | Government policy stance, foreign finance/legal firm expansion and hiring, district restaurant activity (notebook §9). |

Model roles: **level GLSAR(1)** = BI reconstruction and interpretation baseline; **Δlog + momentum** = short-horizon forecasts and scenario paths; **random walk** = the benchmark every forecast claim must beat.

## Quick start

Python ≥ 3.10. Run everything from this folder.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python build_monthly_master.py     # source CSVs -> data/*.csv
python run_research_audit.py       # variable audit, model comparison, diagnostics -> outputs/
python -m streamlit run app.py     # dashboard
```

On Windows, `rebuild_outputs.ps1` runs the first two scripts and `run_dashboard.ps1` starts the app.

If the existing `.venv` uses Python older than 3.10, keep it and create a separate dashboard environment with an installed 64-bit Python 3.13:

```powershell
py -3.13 -m venv .venv-streamlit
& .\.venv-streamlit\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv-streamlit\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` while the server is running. Do not launch the dashboard with `python app.py` or by double-clicking the Python file. `run_dashboard.ps1` prefers `.venv-streamlit` when present and checks the Python version and Streamlit installation. If PowerShell blocks the script, use the direct Python command above; no execution-policy change is needed.

For Streamlit Community Cloud, use a GitHub repository, `app.py` as the entrypoint, and Python 3.13 in the deployment settings. The root `requirements.txt` installs dependencies, and the app loads `src/` itself. Include the required `data/` and `outputs/` files only after confirming redistribution rights. A local server is not an external deployment.

### Community Cloud deployment

1. Confirm permission to upload and serve the supplied data. Do not publish Bloomberg-derived inputs without redistribution rights.
2. Push the project to an authorized GitHub repository. Include `app.py`, `requirements.txt`, `src/hkproperty/`, `.streamlit/config.toml`, and `data/extension_model_input.csv`.
3. Include `outputs/home_stationarity.csv` and `outputs/research_status.json` for the Model tab's stationarity evidence and ECM status. The dashboard does not rebuild them during deployment. Other audit outputs remain available for notebook research but are not needed by the dashboard.
4. In Streamlit Community Cloud, create an app from that repository and branch, select `app.py` as the main file, and choose Python 3.13 under advanced settings.
5. Deploy, then verify the five tabs (Data, Model, Scenario, Extensions, Bubble index) and the controls within Scenario at the assigned app URL. The committed theme configuration keeps widgets and tables light; charts explicitly set their text colors and disable Streamlit's Plotly theme override.

Never upload virtual environments, `.env` files, or `.streamlit/secrets.toml`. If secrets are needed later, configure them in Community Cloud settings instead. Optional source exports, notebooks, interview materials, and screenshots are not needed by the dashboard.

Tests (run from `tests/`, because the brackets in the folder name `[HK]home_sales` confuse pytest's path handling):

```bash
cd tests && python -m pytest -q -p no:warnings --rootdir=. .
```

Execute the main notebook end to end:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/01_home_model_reconstruction.ipynb \
  --output 01_home_model_reconstruction.ipynb --output-dir notebooks
```

If you edit anything under `src/`, restart the notebook kernel; the `hkproperty` modules are cached.

## Project layout

```text
.
├─ app.py                     Streamlit scenario lab (5 tabs)
├─ build_monthly_master.py    Builds the monthly master and model inputs from source CSVs
├─ run_research_audit.py      Variable integration audit, common-origin model comparison, status file
├─ src/hkproperty/
│  ├─ specs.py                Series metadata; HOME (level GLSAR) and GROWTH specifications
│  ├─ models.py               Fitting and forecasting (GLSAR, SARIMAX, seasonal terms)
│  ├─ growth.py               Δlog + momentum model with HAC errors
│  ├─ validation.py           Rolling-origin evaluation; shared PSEUDO_OOS protocol
│  ├─ research.py             HOME_RESEARCH_SPECS registry of challengers and extensions
│  ├─ scenarios.py            Baseline X paths, shocks, extrapolation checks
│  ├─ diagnostics.py          ADF/KPSS, residual tests, VIF
│  ├─ extensions.py           Cointegration-gated ECM
│  ├─ data.py, demo.py        Data loading and auditing; synthetic demo data
│  └─ bqnt_export.py          Original BQNT extraction logic
├─ data/
│  ├─ source/bqnt/            Supplied BQNT monthly CSV and data dictionary
│  ├─ source/altdata/         Alternative data (RVD classes, Land Registry, FEHD restaurants, job boards)
│  └─ *.csv                   Monthly master, model inputs, dictionaries, coverage
├─ notebooks/                 See below
├─ outputs/                   Comparison scores, predictions, diagnostics, audit, scenario CSVs
├─ docs/                      Methodology and review documents (see below)
└─ tests/                     Research-contract tests
```

## App

`python -m streamlit run app.py` opens five tabs:

| Tab | Content |
|---|---|
| Data | Level and monthly growth charts, sample coverage, RVD definition, variables and units |
| Model | Notebook-reference Level/Growth estimated equations, coefficient/p-value tables, shared interpretation and estimation notes; expandable stationarity evidence |
| Scenario | Growth by default, optional Level comparison, stress presets, 100 seeded residual-simulated Growth paths with bootstrap bands, end-horizon medians and CSV downloads |
| Extensions | Explicit aggregate-model limitation and required regional disaggregation; notebook section 9 channels, policy layers, validation rules, seasonal regressors and ECM |
| Bubble index | Four thresholded cycle states, shared monthly history bands, smoothed Markov transitions and 1/3/6-month state probabilities, transition counts and signal CSV download |

The Model tab consolidates the notebook's rounded reference estimates without repeating generic equations or live coefficient tables. Its 199-month reference window describes input data; Growth estimation loses initial rows to differences and lagged growth. Scenario models are refitted to the supplied data. Detailed forecast scorecards, residual diagnostics and extension audits remain in the notebooks and precomputed research outputs rather than separate dashboard tabs.

Scenario offers only Level and Growth, with Growth selected by default. Model visibility is controlled by the persistent model selector, not by legend clicks that reset when Streamlit rebuilds a plot. Shocks are permanent shifts in factor levels starting next month, not fresh monthly increments; there is no shock-shape selector. Fed effective-rate shocks range from -100 to +100 bp in 25 bp steps; other shocks range from -10% to +10%. A 25 bp shock is 0.25 percentage points. FOMC target-range moves are commonly in 25 bp increments, but the observed effective rate is market-determined, not mechanically fixed to those increments. Model metrics report the prediction median at the selected final month, not an average across the forecast horizon.

Stress presets are illustrative joint assumptions: Baseline sets every shock to zero; Adverse 5% lowers HSI, secondary sales, rent and CNYHKD by 5% and raises the Fed rate by 25 bp; Supportive reverses those signs. These are not worst-case bounds, events with 5% probability, or scenarios guaranteed to produce a 5% price loss. Manual slider edits switch the preset label to Custom; model and horizon selections are preserved.

When Growth is selected, a toggle shows 100 paths drawn by independently resampling centred fitted monthly growth residuals. Errors propagate through the fitted momentum coefficient and accumulate in log prices. Seed 42 is fixed so positive/negative shock comparisons share the same random innovations. The bold line and headline metrics remain the fitted scenario median; Growth shading uses the simulated 2.5th/97.5th percentiles when enabled. Level retains analytical intervals. The IID bootstrap holds parameters and factor paths fixed, does not preserve residual serial dependence or volatility clustering, and gives coarse tail estimates with only 100 paths. Simulated paths can be downloaded separately.

Seasonality is described as an optional regressor experiment (month dummies or annual Fourier terms), not a third scenario model. Neither core model is silently refitted with seasonal terms; the notebook's seasonal comparison did not improve out-of-sample performance.

The Bubble index is a descriptive price-cycle view, not a statistical bubble test or a fair-value estimate. It shows `100 * (price / MA5 - 1)` and `100 * (price / price.shift(3) - 1)` on separate time-series charts. Their thresholded sides define four states: Heating up (positive/positive; red), Cooling (positive/negative; amber), Weakening (negative/negative; blue), and Recovery (negative/positive; green). Default dead bands are +/-0.25% for the MA5 gap and +/-0.5% for three-month momentum. Each side changes only beyond its band and retains its prior sign inside the band. Until both sides are established the state is unclassified; missing inputs reset the sides. All four definitions are always visible, the current state is highlighted, and the same thresholded monthly state bands appear behind all three histories.

The first-order Markov outlook estimates monthly transitions from all available classified history, excluding missing/unclassified months and calendar gaps. It adds 0.5 pseudo-count per cell; unseen rows therefore default to a uniform prior, not empirical evidence. Observed transition counts and sparse rows are displayed. State probabilities at 1, 3 and 6 months are calculated exactly by matrix propagation and sum to 100%. They assume stable transition probabilities, are fitted in sample, and are not validated real-time forecasts, price-return probabilities or speculative-bubble probabilities. Changing the thresholds refits the classifications and transitions.

The price/rent signal is the ratio of the price and rent indices relative to its own trailing 60-month mean, not a rental yield or a separate state classifier. Each chart retains a separate y-axis, a dashed zero reference and a dotted latest-value reference. Trailing averages include the current month, missing months are not filled, and no future observations enter the calculations. State shading is not a calibrated overheating threshold. The displayed date is the latest supplied observation, not necessarily the current month.

## Notebooks

| Notebook | Status |
|---|---|
| `01_home_model_reconstruction.ipynb` | **Main walkthrough.** §0–2 data, §3 unit roots and cointegration, §4 level and growth models, §5 residuals, §6 scenario and out-of-sample test, §7 challengers and extensions, §8 summary, §9 next variables. |
| `00_BQNT_Data_Export.ipynb` | Reference only. Documents the original BQL extraction; needs a BQNT runtime and does not need to be re-run. |
| `02_Seasonality_and_Extensions.ipynb` | Legacy. Fully merged into notebook 01 §7. |
| `ds_interview_prep.ipynb` | Application-form helper, separate from the model story. Maps the form's concept checkboxes to evidence and writes `outputs/answers_draft.*` and `outputs/interview_*`. |

## Documents

| File | Content |
|---|---|
| `docs/01_Source_Reconstruction.md` | What the BI screenshots disclose, what they do not, and the replication boundary |
| `docs/02_Methodology_Notes.md` | Methodology reference: GLS/FGLS, AR errors vs. lagged Y, SARIMAX, cointegration, seasonality, evaluation design |
| `docs/03_Extensions_and_New_Variables.md` | Extension roadmap and experimental design (model change vs. new variables) |
| `docs/04_Interview_Discussion.md` | Interview pitch and answer skeletons |
| `docs/analysis_review.md` | Review of the real-data analysis with current numbers |
| `docs/extension_review.md` | AltData candidates, decisions, and current extension results |

## Data and interpretation boundary

- **Data vintage.** BQNT values are latest-vintage. Results are pseudo-out-of-sample, not a point-in-time (release-vintage) backtest.
- **Office model.** Withheld: the target `office_rent` is not in the supplied data.
- **Undisclosed source details.** Transformations, lags, AR order, and seasonal treatment are explicit assumptions (levels, contemporaneous X, AR(1) errors).
- **Causality.** Scenarios are conditional projections with coefficients and other inputs held fixed. Under the HKD peg the Fed rate is not exogenous to Hong Kong, so they are not causal effects.
- **Data licensing.** `data/source/bqnt/` contains Bloomberg-derived data. Check redistribution terms before pushing to a shared or public remote.
- **Manifest.** `package_manifest.json` is a snapshot of the originally delivered package. No script regenerates it.

## Environment

Dependencies are pinned to ranges in `pyproject.toml` and `requirements.txt`. NumPy is kept below 2.5 for SciPy compatibility.
