from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
from scipy.cluster.vq import kmeans2


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hkproperty.data import read_monthly
from hkproperty.research import (
    FEATURE_LABELS,
    available_research_specs,
    fit_research_spec,
    scenario_path,
)
from hkproperty.specs import HOME


st.set_page_config(page_title="HK Home · Scenario Lab", page_icon="⌂", layout="wide")
st.markdown(
    """
    <style>
    :root { --ink:#132b2e; --muted:#465e62; --teal:#087f83; --orange:#d56734; --paper:#f2f5f3; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(242,245,243,.82); }
    [data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 1.4rem; }
    h1,h2,h3 { color: var(--ink); font-family: Georgia,"Times New Roman",serif; }
    h1 { font-size: 2.55rem; letter-spacing: 0; margin-bottom: .25rem; }
    .eyebrow { color:#07676b; font-size:.8rem; font-weight:800; letter-spacing:0; text-transform:uppercase; }
    .dek { color:var(--muted); font-size:1rem; max-width:860px; }
    .rule { border-bottom:1px solid #c8d4d1; margin:.9rem 0 1.15rem; }
    [role="tablist"] { gap:.25rem; border-bottom:2px solid #becfca; overflow-x:auto; }
    [data-testid="stTab"] { color:#344f53; background:transparent; min-height:3rem; padding:.65rem 1rem; white-space:nowrap; }
    [data-testid="stTab"] p { color:inherit; font-size:1rem; font-weight:600; }
    [data-testid="stTab"]:hover { color:#075c60; background:#e0ebe7; }
    [data-testid="stTab"][aria-selected="true"] { color:#075c60; background:#d9ebe5; }
    [data-testid="stTab"]:focus-visible { outline:2px solid #087f83; outline-offset:-2px; }
    [data-baseweb="tab-highlight"] { background:#087f83; height:3px; }
    [data-testid="stCaptionContainer"] { color:var(--muted); font-size:.9rem; }
    [data-testid="stWidgetLabel"] { color:var(--ink); }
    [data-testid="stMetric"] { background:#fff; border:1px solid #ccd8d5; border-top:3px solid var(--teal); padding:.75rem 1rem; border-radius:4px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricLabel"] p { white-space:normal; overflow-wrap:anywhere; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:Georgia,"Times New Roman",serif; font-size:2rem; }
    div[data-testid="stDataFrame"] { border:1px solid #ccd8d5; }
    .spec-note { background:#e5efec; border-left:4px solid var(--teal); padding:.8rem 1rem; color:#294347; }
    .cycle-states { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin:.8rem 0 1rem; }
    .cycle-state { border:1px solid #becfca; border-left:5px solid var(--state-color); border-radius:4px; padding:.7rem; background:#fff; }
    .cycle-state.current { outline:2px solid var(--state-color); background:var(--state-fill); }
    .cycle-state strong { display:block; color:var(--state-color); font-size:1rem; }
    .cycle-state span { display:block; color:#344f53; font-size:.85rem; margin-top:.3rem; }
    .cycle-state small { display:block; color:#132b2e; min-height:1.3rem; margin-top:.3rem; font-weight:700; }
    @media(max-width:700px){ .cycle-states{grid-template-columns:repeat(2,minmax(0,1fr));} }
    @media(max-width:700px){ h1{font-size:2rem;} [data-testid="stMainBlockContainer"]{padding:1rem .7rem;} [data-testid="stTab"]{padding:.65rem .75rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_PATH = PROJECT_DIR / "data" / "extension_model_input.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
STATIONARITY_PATH = OUTPUT_DIR / "home_stationarity.csv"
STATUS_PATH = OUTPUT_DIR / "research_status.json"
ASIA_MONTHLY_PATH = PROJECT_DIR / "data" / "asia_home_prices_monthly.csv"
ASIA_MONTHLY_SOURCE_PATH = PROJECT_DIR / "data" / "asia_home_prices_monthly_sources.json"
AUDIT_HINT = "Run `python run_research_audit.py` from the project root to create this output."
PALETTE = ["#087f83", "#d56734", "#5b4a9e"]
REGIME_STYLES = {
    "Heating up": {"color": "#aa3039", "fill": "#fbe4e5", "rule": "Positive MA5 side · rising side"},
    "Cooling": {"color": "#886000", "fill": "#fff1cb", "rule": "Positive MA5 side · falling side"},
    "Weakening": {"color": "#305f96", "fill": "#e4eef9", "rule": "Negative MA5 side · falling side"},
    "Recovery": {"color": "#10735b", "fill": "#ddf1e7", "rule": "Negative MA5 side · rising side"},
}
CHART = dict(template="plotly_white", paper_bgcolor="#f2f5f3", plot_bgcolor="#ffffff",
             font={"family": "Verdana, sans-serif", "size": 14, "color": "#132b2e"},
             title={"font": {"size": 18, "color": "#132b2e"}},
             colorway=PALETTE,
             xaxis={"automargin": True, "tickfont": {"size": 13, "color": "#344f53"}, "gridcolor": "#dce5e1"},
             yaxis={"automargin": True, "tickfont": {"size": 13, "color": "#344f53"}, "gridcolor": "#dce5e1"},
             hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#132b2e", "size": 14}},
             margin={"l": 30, "r": 25, "t": 65, "b": 50})
SERIES_LABELS = {"home_price": "Home price index (target)", **FEATURE_LABELS}

if not DATA_PATH.exists():
    st.error("Model input is missing. Run `python build_monthly_master.py` first.")
    st.stop()


@st.cache_resource(show_spinner="Estimating specification…")
def load_fit(path: str, modified_ns: int, spec_key: str):
    return fit_research_spec(read_monthly(path), spec_key)


def fit_or_stop(key: str):
    try:
        fitted = load_fit(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns, key)
    except Exception as exc:
        st.error(f"{available[key].label} could not be estimated: {exc}")
        st.stop()
    if not fitted.converged:
        st.error(f"{available[key].label} did not converge, so its forecasts are withheld.")
        st.stop()
    return fitted


def eyebrow(text: str):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


raw_frame = read_monthly(DATA_PATH)
available = available_research_specs(raw_frame)
scenario_keys = [key for key in ("baseline", "growth") if key in available]
target = raw_frame[HOME.target].dropna()
last_date, last_value = target.index[-1], float(target.iloc[-1])

eyebrow("HONG KONG RESIDENTIAL · USER-SUPPLIED LATEST-VINTAGE DATA")
st.title("Home price model & scenario lab")
st.markdown(
    '<p class="dek">A research perspective on Hong Kong housing: the forces shaping prices, '
    'the state of the market, and the paths that could lie ahead.</p><div class="rule"></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Data", "Model", "Scenario", "Extensions", "Bubble index", "Regime"])


def render_data_notes():
    st.subheader("How to read the Hong Kong price index")
    st.markdown(
        "**RVD is the Rating and Valuation Department of the Hong Kong SAR Government.** It publishes official "
        "private domestic property price and rent indices using transaction and tenancy information. "
        "Private domestic means private residential property, not public housing.\n\n"
        "This project uses the **all-classes** price index as the target and the corresponding rent index "
        "as a factor. The five saleable-area classes are A: below 40 m²; B: 40–69.9 m²; C: 70–99.9 m²; "
        "D: 100–159.9 m²; E: 160 m² or above. All classes is an aggregate, not a district-specific or "
        "individual-flat valuation.\n\n"
        "**1999 = 100 is an index base, not a currency price or the price of a typical flat.** An index "
        "of 320 means the measured level is 3.2 times its 1999 base. A move from 320 to 321 is +1 index "
        "point but only about +0.31%, not +1%. Percentage change is $100(P_t/P_{t-1}-1)$; the growth "
        "chart above uses $100\\Delta\\log P_t$, a close approximation for small changes.\n\n"
        "The index is **nominal**: it does not remove consumer inflation, exchange-rate changes, "
        "financing costs or floor-area differences. An index change describes the aggregate market, "
        "not the appreciation of a particular flat. Recent releases may be provisional or revised. "
        "The rent index measures rents, not rental yields."
    )
    st.caption("Official reference: [RVD Hong Kong Property Review, private domestic price indices](https://www.rvd.gov.hk/doc/en/HKPR2026_Preliminary_Findings_Eng.pdf).")

    st.subheader("Variables and units")
    st.markdown(
        "| Symbol | Variable | Level units |\n|---|---|---|\n"
        "| $y_t$ | RVD private domestic price index · home_price | 1999 = 100 |\n"
        "| $F_t$ | US effective federal funds rate · fed_rate | Percent; changes in percentage points |\n"
        "| $H_t$ | Hang Seng Index · hsi | Index points |\n"
        "| $S_t$ | Midland secondary-market transactions · secondary_sales | Number of transactions |\n"
        "| $R_t$ | RVD private domestic rent index · home_rent | 1999 = 100 |\n"
        "| $C_t$ | Renminbi exchange rate · cnyhkd | HKD per 1 CNY |"
    )


def render_asia_monthly_comparison():
    st.markdown("**Monthly housing-price indices · 2011 onward**")
    if not ASIA_MONTHLY_PATH.exists() or not ASIA_MONTHLY_SOURCE_PATH.exists():
        st.info("The monthly snapshot is unavailable. Run `python build_asia_monthly_prices.py` to create it.")
        return
    comparison = pd.read_csv(ASIA_MONTHLY_PATH).set_index("month")
    metadata = json.loads(ASIA_MONTHLY_SOURCE_PATH.read_text(encoding="utf-8"))
    names = {"hong_kong": "Hong Kong · private", "china_70_city": "China · 70-city resale composite",
             "seoul": "Seoul · apartments"}
    colors = {"hong_kong": "#087f83", "china_70_city": "#5b4a9e", "seoul": "#305f96"}
    bases = [period for period in ("2011-01", "2016-01", "2020-01", "2023-01") if period in comparison.index]
    if not bases:
        st.info("No common monthly comparison base is available.")
        return
    base = st.selectbox("Set every monthly series to 100 in", bases, key="asia_monthly_base")
    rebased = 100 * comparison.loc[base:] / comparison.loc[base]
    figure = go.Figure()
    for code in names:
        figure.add_trace(go.Scatter(
            x=pd.to_datetime(rebased.index), y=rebased[code], customdata=rebased.index,
            name=names[code], mode="lines", line={"color": colors[code], "width": 2.6},
            hovertemplate="%{customdata}<br>%{y:.1f} · rebased<extra>" + names[code] + "</extra>",
        ))
    figure.add_hline(y=100, line_color="#607276", line_dash="dot", line_width=1)
    figure.update_layout(height=440, yaxis_title=f"Index · {base} = 100", hovermode="x unified",
                         legend={"orientation": "h", "y": -0.22}, **CHART)
    st.plotly_chart(figure, width="stretch", theme=None, key="asia_monthly_comparison")
    st.caption(
        f"All three series are monthly; their common observed window is {comparison.index[0]}–{comparison.index[-1]} "
        f"({len(comparison)} months). Snapshot retrieved {metadata['retrieved_utc'][:10]}. "
        f"The chart sets {base} to 100, so 120 means a 20% increase since that month. "
        "The rebased levels do not equalise home type, geography, currency or measurement method."
    )
    st.info("China's line is our equal-weighted composite of monthly indices for 70 cities, not an official "
            "national aggregate. These comparisons are descriptive and are outside the fitted Hong Kong models.")
    changes = 100 * np.log(comparison[list(names)]).diff()
    changes = changes.dropna(how="any")
    st.markdown("**Monthly change correlation · exploratory**")
    st.caption("Pearson correlations use monthly log changes (100 × Δlog index), not trending index levels. "
               "The full common sample is used regardless of the chart's display base.")
    matrix = changes.corr()
    labels = [names[code] for code in names]
    heatmap = go.Figure(go.Heatmap(
        z=matrix.values, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0, "#305f96"], [0.5, "#f2f5f3"], [1, "#b64c3d"]],
        text=np.array([[f"{value:+.2f}" for value in row] for row in matrix.values]),
        texttemplate="%{text}", textfont={"size": 14},
        hovertemplate="%{y} / %{x}<br>Pearson r = %{z:.3f}<extra></extra>",
        colorbar={"title": "r"},
    ))
    heatmap.update_layout(height=335, **CHART)
    heatmap.update_layout(margin={"l": 8, "r": 8, "t": 12, "b": 8})
    st.plotly_chart(heatmap, width="stretch", theme=None, key="asia_monthly_correlation")
    rows = []
    for code in ("china_70_city", "seoul"):
        for lag, timing in ((0, "Same month"), (1, "Foreign change 1 month earlier")):
            pair = pd.concat([changes["hong_kong"], changes[code].shift(lag)], axis=1).dropna()
            rows.append({"Market vs Hong Kong": names[code], "Timing": timing,
                         "Pearson r": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "Months": len(pair)})
    st.dataframe(pd.DataFrame(rows).style.format({"Pearson r": "{:+.2f}"}),
                 width="stretch", hide_index=True)
    st.caption("Same-month comparisons have no direction. A one-month lead is descriptive, not a feasible "
               "forecast until release dates and data vintages are checked. Correlation does not establish causation.")
    with st.expander("Monthly series definitions and construction"):
        details = pd.DataFrame([
            {"Market": names[code], "Coverage": metadata["series"][code]["coverage"],
             "Latest original value": f"{comparison[code].iloc[-1]:.1f}",
             "Type": "Official index" if code in ("hong_kong", "seoul") else "Constructed composite"}
            for code in names
        ])
        st.dataframe(details, width="stretch", hide_index=True)
        st.markdown(
            "**China:** geometric mean of the 70 cities' monthly existing-home price changes, with one "
            "equal weight per city, chained from 2011. Source: [NBS monthly city indices](https://www.stats.gov.cn/sj/zxfb/202609/t20260915_1965304.html), "
            "[machine-readable transcription](https://github.com/changao1/70-China-cities-housing-index-data-by-national-bureau-of-statistics).\n\n"
            "**Hong Kong:** [RVD all-classes private domestic](https://www.rvd.gov.hk/en/publications/property_market_statistics.html). "
            "**Seoul:** [REB apartment actual transactions](https://www.reb.or.kr/r-one/portal/stat/easyStatPage/A_2024_00178.do). "
            "The original published index bases differ; the chart's common base is only a display transformation."
        )


def render_model_equations():
    st.subheader("Final models at a glance")
    st.markdown(
        "The notebook's reference input window is **2010-02 to 2026-08 (199 monthly observations)**. "
        "Both models use the same five factors; the Growth model loses initial observations to differencing "
        "and its lag. The equations and tables below are rounded notebook reference estimates. "
        "Scenario forecasts are fitted to the currently supplied data. Symbols and native units are defined in Data."
    )
    st.subheader("Level model · GLSAR(1)")
    st.markdown("Uses factor levels directly. Coefficients describe conditional associations in **price-index points**.")
    st.latex(
        r"\begin{aligned} y_t &\approx 184.4 - 8.38 F_t \\"
        r"&\quad + 0.00017 H_t + 0.00122 S_t \\"
        r"&\quad + 1.106 R_t + 8.5 C_t + u_t \\"
        r"u_t &\approx 0.993 u_{t-1} + \epsilon_t \\"
        r"\widehat{\sigma}_{\epsilon} &\approx 3.84\;\text{index points} \end{aligned}"
    )
    st.markdown(
        "| Term | Estimate | Per | p-value |\n|---|---|---|---|\n"
        "| Fed rate | **−8.38** | Index points per +1 percentage point | < 0.001 |\n"
        "| Secondary transactions | **+1.22** | Index points per +1,000 deals | < 0.001 |\n"
        "| Rent index | **+1.11** | Price-index points per +1 rent-index point | < 0.001 |\n"
        "| HSI | +0.17 | Price-index points per +1,000 HSI points | 0.50 |\n"
        "| CNYHKD | +8.5 | Price-index points per +1 HKD per CNY | 0.72 |\n"
        "| $\\rho$ | **0.993** | AR(1) error persistence, not price growth | — |"
    )

    st.subheader("Growth model · Δlog + momentum")
    st.markdown(
        "Used for short-run forecasts and scenarios. $g_t=\\Delta\\log y_t$; predictors enter in "
        "log differences except the Fed rate, which enters in percentage-point changes."
    )
    st.latex(
        r"\begin{aligned} g_t &\approx 0.0015 + 0.615 g_{t-1} \\"
        r"&\quad - 0.0119\Delta F_t \\"
        r"&\quad + 0.001\Delta\log H_t \\"
        r"&\quad + 0.0223\Delta\log S_t \\"
        r"&\quad + 0.105\Delta\log R_t \\"
        r"&\quad + 0.104\Delta\log C_t + \varepsilon_t \end{aligned}"
    )
    st.markdown(
        "| Term | Estimate | Reading, other terms fixed | p-value (HAC) |\n|---|---|---|---|\n"
        "| Momentum $\\phi$ | **0.615** | About 61.5% of last month's log growth carries over | < 0.001 |\n"
        "| ΔFed | **−0.0119** | +1 pp is associated with about −1.19 percentage points of monthly price growth | < 0.001 |\n"
        "| Δlog sales | **+0.0223** | +10% deals is associated with about +0.21 percentage points of monthly price growth | < 0.001 |\n"
        "| Δlog rent | +0.105 | Not statistically significant in this specification | 0.29 |\n"
        "| Δlog CNYHKD | +0.104 | Not statistically significant in this specification | 0.12 |\n"
        "| Δlog HSI | +0.001 | Not statistically significant in this specification | 0.94 |\n\n"
        "**Fit:** $R^2 \\approx 0.59$; residual standard deviation approximately 0.99% per month. "
        "R-squared is in-sample explained variation, not forecast accuracy. Log growth is in decimal units; "
        "the exact monthly percentage change is $100(e^{g_t}-1)$."
    )

    st.subheader("How to read the two models")
    st.markdown(
        "- **Consistent associations:** the Fed rate is negative and transactions positive in both specifications.\n"
        "- **Rent:** significant only in levels, which may reflect a shared trend; this does not establish causality or rule out an economic role.\n"
        "- **Level persistence:** $\\rho=0.993$ makes GLS quasi-differencing close to monthly changes in index points, not log growth. The intercept is weakly identified (standard error about 43).\n"
        "- **Growth propagation:** a lasting Fed −1 pp level shift is a one-time rate change followed by momentum, not a fresh cut each month. With the rounded coefficients, its limiting log-price effect is $0.0119/(1-0.615)$, or about +3.1% after conversion. This extrapolates short-run dynamics, not an estimated equilibrium.\n"
        "- **Limits:** neither model is causal. In the notebook, Growth beats a random walk at one month; neither beats it at 3–12 months."
    )
    st.subheader("Estimation and specification choice")
    st.markdown(
        "- **Level:** feasible GLS (`statsmodels.GLSAR`), with levels, contemporaneous factors and AR(1) errors as explicit project assumptions.\n"
        "- **Growth:** OLS with Newey–West HAC standard errors, 6 lags. HAC adjusts coefficient inference; it does not remove endogeneity.\n\n"
        "Key level series appear I(1). The notebook's Engle–Granger test does not reject no cointegration "
        "($p \\approx 0.98$), so the Level specification is not interpreted as a verified long-run equilibrium. "
        "Growth models monthly changes and momentum without imposing that equilibrium. Failure to reject "
        "is not proof that no relation exists. If cointegration were supported, an ECM could combine the "
        "level relation with short-run changes; that extension is described in Extensions."
    )
    st.info(
        "Level and Growth are alternative specifications, not a verdict on an external model. "
        "All diagnostics refer to this project's data, sample and assumptions; they do not evaluate "
        "Bloomberg Intelligence's proprietary implementation. Scenarios describe conditional associations, not causal effects."
    )
    st.caption(
        "Source: 01_home_model_reconstruction.ipynb, sections 0–1 and 3–6. Numerical equations use rounded "
        "notebook reference estimates, not live coefficients. "
        "Detailed forecast evaluation and diagnostics remain in the source notebook."
    )

# ---------------------------------------------------------------- Data
with tabs[0]:
    st.markdown(
        '<div class="spec-note"><strong>Start here:</strong> the Hong Kong target is an RVD private-home '
        'price <em>index</em> (1999 = 100). It tracks aggregate nominal price movements, not HKD per flat. '
        'Read month-to-month moves as percentage changes; the detailed definition is below.</div>',
        unsafe_allow_html=True,
    )
    home_columns = [HOME.target, *HOME.features]
    series = st.selectbox("Series", home_columns, format_func=lambda c: SERIES_LABELS.get(c, c))
    values = raw_frame[series].dropna()
    growth = values.diff() if series == "fed_rate" else 100 * np.log(values).diff()
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes="columns", shared_yaxes="rows",
        column_widths=[0.76, 0.24], horizontal_spacing=0.008, vertical_spacing=0.16,
        subplot_titles=("Level", "Distribution", "Change · pp" if series == "fed_rate" else "Growth · %", "Distribution"),
    )
    fig.add_trace(go.Scatter(x=values.index, y=values, mode="lines", line={"color": PALETTE[0]}, name="Level"), row=1, col=1)
    fig.add_trace(go.Bar(x=growth.index, y=growth, marker_color=PALETTE[1], name="Change"), row=2, col=1)
    for row, observations, color in [(1, values, PALETTE[0]), (2, growth.dropna(), PALETTE[1])]:
        counts, edges = np.histogram(observations, bins=20)
        fig.add_trace(go.Bar(
            x=counts, y=(edges[:-1] + edges[1:]) / 2, width=np.diff(edges), orientation="h",
            name="Monthly observations", customdata=np.column_stack((edges[:-1], edges[1:])),
            marker={"color": color, "line": {"width": 0}}, opacity=0.45,
            hovertemplate="Value bin: %{customdata[0]:.2f} to %{customdata[1]:.2f}<br>Months: %{x}<extra></extra>",
        ), row=row, col=2)
        if not observations.empty:
            mean = float(observations.mean())
            lower, upper = float(observations.min()), float(observations.max())
            padding = (upper - lower) * 0.08 if upper > lower else max(abs(mean) * 0.05, 0.01)
            for column in (1, 2):
                fig.update_yaxes(range=[lower - padding, upper + padding], row=row, col=column)
            fig.add_hline(y=mean, line_color="#88580a", line_dash="dot", line_width=1.5,
                          annotation_text=f"Mean {mean:,.2f}", annotation_position="top left", row=row, col=1)
            fig.add_hline(y=mean, line_color="#88580a", line_dash="dot", line_width=1.5, row=row, col=2)
        fig.update_xaxes(range=[values.index[0], values.index[-1]], showgrid=False, row=row, col=1)
        fig.update_xaxes(title_text="Months", title_font_size=12, nticks=3, rangemode="tozero",
                         range=[0, max(int(counts.max()), 1) * 1.08], matches=None,
                         showgrid=False, showline=False, zeroline=False, row=row, col=2)
        fig.update_yaxes(showgrid=True, gridcolor="#dce5e1", zeroline=False, row=row, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="#dce5e1", zeroline=False, row=row, col=2)
    fig.update_layout(height=620, showlegend=False, bargap=0, **CHART)
    fig.update_annotations(font_size=12)
    st.plotly_chart(fig, width="stretch", theme=None, key="data_series_chart")
    st.caption("Right-hand profiles count monthly observations in value bins, not trading volume. Dotted lines mark each sample mean. Growth is 100 × Δlog; the Fed rate uses percentage-point changes. These are historical distributions, not normal-distribution fits or forecast probabilities.")

    window = pd.DataFrame([
        {"series": c, "label": SERIES_LABELS.get(c, c), "first": raw_frame[c].first_valid_index(),
         "last": raw_frame[c].last_valid_index(), "observed_months": int(raw_frame[c].notna().sum())}
        for c in home_columns
    ])
    window[["first", "last"]] = window[["first", "last"]].apply(lambda s: s.dt.strftime("%Y-%m"))
    left, right = st.columns([1, 1.2])
    left.dataframe(window, width="stretch", hide_index=True)
    right.dataframe(raw_frame[home_columns].dropna().tail(6), width="stretch")
    st.caption("The model sample is the common fully observed block of all six series. Factors observed beyond the last "
               "target month are not used for estimation.")
    st.divider()
    st.subheader("Hong Kong, China and Seoul · monthly housing prices")
    render_asia_monthly_comparison()
    st.divider()
    render_data_notes()

# ---------------------------------------------------------------- Model evidence
with tabs[1]:
    render_model_equations()
    with st.expander("Stationarity test details"):
        if STATIONARITY_PATH.exists():
            stationarity = pd.read_csv(STATIONARITY_PATH)
            table = stationarity.pivot(index="series", columns="transform", values=["adf_pvalue", "kpss_pvalue"])
            table.columns = [f"{test.split('_')[0].upper()} p · {transform}" for test, transform in table.columns]
            st.dataframe(table.style.format("{:.3f}"), width="stretch")
            st.caption("ADF null: unit root. KPSS null: stationary. Interpret each series separately; p-values are exploratory.")
        else:
            st.info(AUDIT_HINT)
        if STATUS_PATH.exists():
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            st.info(f"Error-correction model: {status['ecm_status']}")

def apply_scenario_preset():
    preset = st.session_state["scenario_preset"]
    if preset == "Custom":
        return
    direction = {"Baseline (0 shock)": 0, "Adverse 5% stress": -1, "Supportive 5% stress": 1}[preset]
    for feature in HOME.features:
        st.session_state[f"scenario_shock_{feature}"] = -25 * direction if feature == "fed_rate" else 5.0 * direction


def mark_scenario_custom():
    st.session_state["scenario_preset"] = "Custom"


def render_scenario():
    settings = st.columns([2, 1])
    with settings[0]:
        model_keys = st.multiselect(
            "Models to compare", scenario_keys,
            default=["growth"] if "growth" in scenario_keys else scenario_keys,
            format_func=lambda key: available[key].label, max_selections=2, key="scenario_core_models",
        )
    with settings[1]:
        horizon = st.slider("Horizon · months", 1, 12, 12, key="scenario_horizon")
    st.selectbox(
        "Stress preset", ["Custom", "Baseline (0 shock)", "Adverse 5% stress", "Supportive 5% stress"],
        key="scenario_preset", on_change=apply_scenario_preset,
    )
    st.caption("Adverse 5%: HSI, sales, rent and CNYHKD −5%; Fed +25 bp. Supportive reverses these signs. These are illustrative joint shocks, not worst-case bounds, a 5% event probability or a guaranteed 5% price loss.")
    st.markdown("**Factor shocks**")
    st.caption("Fed effective-rate shock: 25 bp steps (25 bp = 0.25 percentage points). Other factors: percent of their last observed value.")
    st.caption("Each shock changes the factor level from next month onward and holds it there; it is not an additional change every month.")
    shocks = {}
    shock_columns = st.columns(len(HOME.features))
    for feature_index, feature in enumerate(HOME.features):
        label = FEATURE_LABELS.get(feature, feature)
        shock_key = f"scenario_shock_{feature}"
        if shock_key not in st.session_state:
            st.session_state[shock_key] = 0 if feature == "fed_rate" else 0.0
        with shock_columns[feature_index]:
            if feature == "fed_rate":
                shocks[feature] = st.slider(f"{label} · bp", -100, 100, step=25, key=shock_key,
                                            on_change=mark_scenario_custom) / 100.0
            else:
                shocks[feature] = st.slider(f"{label} · %", -10.0, 10.0, step=0.5, key=shock_key,
                                            on_change=mark_scenario_custom)
    if not model_keys:
        st.warning("Select at least one model to calculate a scenario.")
        return
    fits = {key: fit_or_stop(key) for key in model_keys}
    scenarios = {key: scenario_path(fit, shocks, horizon=horizon, shape="permanent") for key, fit in fits.items()}
    simulated_paths = None
    if "growth" in fits:
        if st.toggle("Show 100 residual-simulated paths · Growth", value=True, key="scenario_simulated_paths"):
            simulated_paths = fits["growth"].simulate(scenarios["growth"][1], n_paths=100, seed=42)
    else:
        st.caption("Select Growth to display residual-based path simulations; Level retains its analytical interval.")
    end_date = last_date + pd.offsets.MonthBegin(horizon)
    st.divider()
    metrics = st.columns(1 + len(fits))
    metrics[0].metric("Last observed Y", f"{last_value:,.2f}", help=f"{last_date:%Y-%m} home-price index")
    for column, (key, (_, _, base_fc, scenario_fc, _)) in zip(metrics[1:], scenarios.items()):
        base_end = float(base_fc["prediction_median"].iloc[-1])
        scenario_end = float(scenario_fc["prediction_median"].iloc[-1])
        column.metric(
            f"{available[key].label.split(' · ')[0]} · {end_date:%Y-%m} median",
            f"{scenario_end:,.2f}",
            delta=f"{100 * (scenario_end / base_end - 1):+.2f}% vs baseline path",
            help=f"End-horizon conditional prediction median, not an average across months. Baseline path: {base_end:,.2f}",
        )
    for key, fit in fits.items():
        if fit.spec.feature_lag:
            st.info(f"{available[key].label} uses X(t−{fit.spec.feature_lag}); a shock first moves its forecast "
                    f"{fit.spec.feature_lag} month(s) later.")

    history = target.tail(36)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history, mode="lines", name="Observed", line={"color": "#48635d", "width": 2}))
    for key, (_, _, base_fc, scenario_fc, _) in scenarios.items():
        color = PALETTE[0] if key == "baseline" else PALETTE[1]
        name = available[key].label.split(" · ")[0]
        x = [last_date, *scenario_fc.index]
        lower, upper = scenario_fc["lower_95"], scenario_fc["upper_95"]
        if key == "growth" and simulated_paths is not None:
            for path_name in simulated_paths:
                fig.add_trace(go.Scatter(
                    x=x, y=[last_value, *simulated_paths[path_name]], mode="lines",
                    line={"color": color, "width": 1}, opacity=0.12, name=path_name,
                    legendgroup=key, showlegend=False, hoverinfo="skip",
                ))
            lower = simulated_paths.quantile(0.025, axis=1)
            upper = simulated_paths.quantile(0.975, axis=1)
        fig.add_trace(go.Scatter(x=[*scenario_fc.index, *scenario_fc.index[::-1]],
                                 y=[*upper, *lower.iloc[::-1]],
                                 fill="toself", fillcolor=color, opacity=0.12, line={"width": 0},
                                 name=f"{name} · scenario 95%", hoverinfo="skip", legendgroup=key,
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[last_value, *base_fc["prediction_median"]], mode="lines",
                                 line={"color": color, "dash": "dot"}, name=f"{name} · baseline",
                                 legendgroup=key, showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=[last_value, *scenario_fc["prediction_median"]], mode="lines+markers",
                                 line={"color": color, "width": 3}, name=name, legendgroup=key))
    fig.update_layout(height=480, yaxis_title="Home price index · 1999=100",
                      legend={"orientation": "h", "y": -0.15, "itemclick": False, "itemdoubleclick": False},
                      hovermode="x unified", uirevision="scenario-comparison", **CHART)
    st.plotly_chart(fig, width="stretch", theme=None, key="scenario_forecast_chart")
    st.caption("Solid: fitted scenario median. Dotted: fitted baseline median. Model metrics remain fitted medians, not sample averages.")
    if simulated_paths is not None:
        st.caption("Thin lines: 100 Growth paths, resampling centred monthly residuals with momentum propagation. Growth shading: simulated 2.5th–97.5th percentiles; Level shading, if shown, is analytical. Seed 42 is fixed, so positive/negative shock comparisons use the same random innovations.")
        st.caption("Illustrative IID residual bootstrap: fixed coefficients and factor paths, no preservation of volatility clustering or residual serial dependence, no parameter uncertainty. 100 paths give coarse tail estimates, not calibrated loss probabilities.")
        st.download_button("Download 100 simulated Growth paths", simulated_paths.to_csv().encode("utf-8"),
                           "growth_simulated_paths.csv", "text/csv")
    else:
        st.caption("Shading: analytical conditional 95% prediction interval.")

    effect_table = pd.DataFrame({
        available[key].label.split(" · ")[0]: 100 * (scenario_fc["prediction_median"] / base_fc["prediction_median"] - 1)
        for key, (_, _, base_fc, scenario_fc, _) in scenarios.items()
    })
    effect_table.index = effect_table.index.strftime("%Y-%m")
    st.markdown("**Scenario vs. baseline path · % by month**")
    st.dataframe(effect_table.T.style.format("{:+.2f}"), width="stretch")

    download = pd.concat(
        {key: base_fc.add_prefix("baseline_").join(scenario_fc.add_prefix("scenario_"))
         for key, (_, _, base_fc, scenario_fc, _) in scenarios.items()},
        names=["model", "date"],
    )
    st.download_button("Download scenario paths", download.to_csv().encode("utf-8"), "home_scenario_paths.csv", "text/csv")
    st.caption("Scenario differences are relative to the same model's baseline path, not today's price. "
               "Growth propagates a one-time factor-level change through momentum. Intervals condition on fixed "
               "parameters and factor paths; they exclude parameter and future-factor uncertainty.")

with tabs[2]:
    render_scenario()

with tabs[3]:
    st.subheader("Research roadmap")
    st.caption("The current five predictors cover rates, equities, transactions, rent and FX. The cards show what can be explored next and what is still missing.")
    roadmap = st.columns(4)
    summaries = [
        ("01 · Policy", "Event dates needed", "Code stamp duties and LTV changes before testing price responses."),
        ("02 · Global firms", "Collection needed", "Track a stable panel of foreign finance and legal employers."),
        ("03 · District activity", "Partial data", "FEHD restaurant licences exist from 2020; match them to regional prices."),
        ("04 · Asian prices", "Monthly context", "China and Seoul comparisons and exploratory correlations are in Data; predictive value is untested."),
    ]
    for column, (heading, status, next_step) in zip(roadmap, summaries):
        with column:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.caption(status)
                st.write(next_step)
    st.info(
        "Current model, scenarios and Bubble index all use Hong Kong-wide prices. District data do not "
        "make them regional forecasts; the Asia comparison is descriptive and outside the fitted model."
    )
    st.subheader("Demand and supply · next research priorities")
    st.markdown(
        "**Demand is purchasing capacity, not just transaction activity; supply is available housing, "
        "not just announced land.** Prioritise local financing costs and household income alongside "
        "completed units and unsold inventory. The candidates below are research hypotheses, not fitted "
        "predictors or current market measurements. Proposed frequencies and regional detail require "
        "series-level verification before collection."
    )
    with st.expander("Demand-side variables · financing, income and households"):
        st.markdown(
            "| Priority | Candidate and measure | Why it may add information | Source / timing to verify |\n|---|---|---|---|\n"
            "| D1 · First test | **Local mortgage funding cost:** observed new-loan pricing where available; otherwise a documented HIBOR/prime-based proxy, plus its spread to the Fed rate | The US rate does not fully describe local funding conditions or mortgage caps; test replacement of Fed and then incremental spread information | HKMA interest-rate tables and Residential Mortgage Survey; HKAB HIBOR. Monthly aggregation, release dates and rate definitions to verify |\n"
            "| D2 · First test | **Jobs and purchasing power:** change in unemployment; real household income or earnings growth | Labour-market risk and payment capacity can move demand even when equities and rates are unchanged | C&SD labour and household surveys. Unemployment uses overlapping rolling three-month windows; income/earnings cadence differs by series |\n"
            "| D3 | **Mortgage affordability:** payment on a fixed reference unit and loan design divided by household income | Separates borrowing costs from household capacity; stress the assumed loan-to-value ratio and term | C&SD income plus verified mortgage pricing. Use only price/income information already available at the forecast origin; quarterly income cannot become new monthly information |\n"
            "| D4 | **Household formation and resident inflows:** growth in domestic households; net movement of residents as a separate measure | Household demand may affect rents before purchases; population growth is not equivalent to new households | C&SD population estimates and household statistics. Verify quarterly/semi-annual releases and census revisions |\n"
            "| D5 | **Mortgage credit availability:** new purchase-related approvals/drawdowns, kept separate from refinancing | Access to financing may convey information beyond its price; jointly determined with transactions and house prices | HKMA Residential Mortgage Survey. Monthly, using published rather than reference dates; lagged predictive test only, not a causal demand instrument |\n"
            "| D6 · Exploratory | **Non-local demand:** a consistently defined buyer-status or residency proxy, if a usable series can be obtained | External purchasing power is not fully measured by CNYHKD; strongest hypothesis for rents and large units | IRD duty-category statistics, Immigration Department arrivals and authorised transaction data. Verify coverage and policy breaks; these do not directly identify mainland buyers |\n\n"
            "**Construction rules:** rate changes enter in percentage points; use log changes or "
            "stock-normalised changes for positive quantities. Test local funding costs **instead of** "
            "Fed first, then a spread specification, rather than adding several near-identical rates. "
            "Use lagged information in affordability to avoid mechanically reintroducing the current "
            "target. Talent approvals, actual arrivals, residents and new households are different "
            "objects and must not be added as independent demand flows. Credit and duty-category "
            "proxies require endogeneity and measurement caveats even if they improve forecasts."
        )
    with st.expander("Supply-side variables · delivery, inventory and pipeline"):
        st.markdown(
            "| Priority | Candidate and measure | Why it may add information | Source / timing to verify |\n|---|---|---|---|\n"
            "| S1 · First test | **Private completions:** trailing 12-month completed units divided by last published private stock, by region and size where available | Measures supply reaching occupancy, not a policy promise; possible effects on rent and prices need separate tests | RVD property statistics and Property Review. Verify provisional completion series, annual stock vintage and regional coverage |\n"
            "| S2 · First test | **Unsold inventory and forward supply:** keep completed-unsold units separate from under-construction and ready-to-start buckets | Inventory captures immediate competition; pipeline is an expectation channel with a different horizon | Housing Bureau first-hand private supply statistics. Quarterly candidate; verify publication dates, bucket definitions and first-hand sales coverage |\n"
            "| S3 | **Private vacancy:** vacant units / private stock, separately by class and region where reported | Measures excess stock rather than construction flow; potentially more informative for local rents | RVD Property Review. Annual year-end measure with a later release; not a fresh monthly vacancy observation |\n"
            "| S4 | **Construction pipeline:** actual commencements, and presale or building consents as separate indicators | Signals future delivery before completion; permission is not the same as a start or an available flat | Buildings Department monthly digests and Lands Department consent statistics. Verify units versus floor area, scheme coverage and release dates |\n"
            "| S5 | **Land converted into future housing capacity:** realised disposals/lease modifications and estimated flat capacity, separate from announced programmes | A long-lag supply channel; tender prices themselves also reflect expected housing prices | Development Bureau and Lands Department. Quarterly/annual candidates; pre-specify multi-year lag hypotheses and distinguish planned from realised supply |\n"
            "| S6 · Segmented test | **Public housing delivery and access:** PRH/HOS completions and waiting-time indicators, kept separate from private stock | Public provision may substitute for some lower-cost private rental/purchase demand; effects need not match the aggregate market | Housing Authority and Housing Bureau. Quarterly/annual candidates; different eligibility and allocation rules require separate treatment |\n\n"
            "**Construction rules:** completed units are a flow; vacancy and unsold units are stocks. "
            "Pipeline buckets, completed-unsold stock and vacancy may overlap under their published "
            "definitions: do not add them together as independent supply. If first-hand sales coverage "
            "is consistent, test months of inventory as completed-unsold stock divided by trailing "
            "12-month first-hand sales / 12. Land, starts and completions are stages of a delivery "
            "chain; test their lags separately rather than flooding a short regression with correlated "
            "stages. Use the last **published** stock denominator, not a later annual revision."
        )
    with st.expander("Demand–supply research design · first four tests and sources"):
        st.markdown(
            "1. **Local-rate pass-through:** compare the Growth baseline with a Fed-to-local-rate replacement, "
            "then an incremental local/Fed spread variant. Evaluate whether it adds information rather than duplicating rates.\n"
            "2. **Income-risk channel:** add a lagged unemployment change first, then income growth separately; "
            "pre-specify a small lag grid and account for overlapping survey windows.\n"
            "3. **Delivered supply:** add trailing completions / last-known stock, matched to the region and "
            "size segment of the price or rent target. Short-run and longer-horizon effects need not have the same sign.\n"
            "4. **Inventory pressure:** test completed-unsold inventory or one forward pipeline measure separately "
            "from completions; use mixed-frequency handling and check whether the history supports the chosen horizons.\n\n"
            "**Validation gate:** use release-date-aware expanding windows and identical origins for the "
            "candidate, Growth baseline and random walk at 1/3/6/12 months where sample length permits. "
            "Freeze lag and transformation choices before the holdout, report RMSE/MAE and the number "
            "of common origins, and limit repeated specification searches. Retain a channel only after "
            "incremental out-of-sample improvement against both benchmarks. Without archived vintages, "
            "label results latest-vintage pseudo-out-of-sample, not a real-time backtest.\n\n"
            "**Regional requirement:** begin with Hong Kong Island / Kowloon / New Territories only where "
            "matched target histories can be built, then districts if coverage supports them. RVD size "
            "classes are a separate segmentation. National rates/FX can be common factors; local supply, "
            "transactions and income exposure must match local targets. No regional series is assumed "
            "available merely because the national series exists.\n\n"
            "**Official discovery sources**\n"
            "- [HKMA Monthly Statistical Bulletin](https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/monthly-statistical-bulletin/) "
            "and [Residential Mortgage Survey releases](https://www.hkma.gov.hk/eng/news-and-media/press-releases/monthly-residential-mortgage-survey-results/).\n"
            "- [C&SD statistics](https://www.censtatd.gov.hk/en/) and "
            "[General Household Survey](https://www.censtatd.gov.hk/en/page_1299.html).\n"
            "- [RVD](https://www.rvd.gov.hk/), [Housing Bureau](https://www.hb.gov.hk/), "
            "[Buildings Department](https://www.bd.gov.hk/) and [Lands Department](https://www.landsd.gov.hk/).\n"
            "- [Development Bureau](https://www.devb.gov.hk/), [Housing Authority](https://www.housingauthority.gov.hk/), "
            "[HKAB](https://www.hkab.org.hk/), [IRD](https://www.ird.gov.hk/) and "
            "[Immigration Department](https://www.immd.gov.hk/).\n\n"
            "The HKMA bulletin and C&SD portal were checked for source discovery. The remaining links "
            "are collection starting points, not confirmation that every proposed series, regional split "
            "or vintage archive is available. Record exact table IDs, units, coverage, licensing and "
            "release calendars before modelling. **No new demand/supply variable has been fitted or "
            "silently added to the current forecasts.**"
        )
    st.markdown("**Open a section below for the research design and limitations.**")
    with st.expander("Regional targets and validation rule"):
        st.subheader("Required next step · regional disaggregation")
        st.markdown(
            "**Regional analysis is required before drawing local housing-market conclusions.** "
            "Hong Kong-wide averages can conceal different price cycles, demand and policy exposure.\n\n"
            "- **Start with Hong Kong Island, Kowloon and the New Territories.** Build separate price and rent targets, "
            "then move to the 18 districts only where consistent target coverage is available.\n"
            "- **Match the geography of target and predictors.** Align regional prices with local transactions, "
            "restaurant activity, firm locations and hiring exposure. Do not attach district predictors to the "
            "aggregate target and present the result as a district forecast. Fed rates and FX can remain common factors.\n"
            "- **Keep geography and unit size distinct.** RVD Classes A–E are size groups, not regions; large-unit "
            "and regional segmentation answer different questions.\n"
            "- **Validate by region on common dates.** Check sample length, boundary consistency and publication "
            "timing; compare each regional model with its own Growth and random-walk benchmarks. A district "
            "panel is a candidate when individual histories are short, not a substitute for missing local targets."
        )
        st.markdown(
            "**Test rule**\n"
            "- Add changes in one channel at a time to the **Growth model**.\n"
            "- Compare on **common forecast origins**, using only information available at each origin.\n"
            "- Retain a channel only if it improves out-of-sample performance against **both the Growth model and a random walk**."
        )
    with st.expander("Government policy · event and sentiment design"):
        st.subheader("Government policy stance and sentiment")
        st.markdown("**Question: is the government trying to cool, hold or support prices?**")
        st.markdown(
            "| Layer | Construction | Frequency |\n|---|---|---|\n"
            "| Events | Stamp-duty and HKMA LTV changes: +1 easing / −1 tightening, assigned to announcement month | Irregular |\n"
            "| Text sentiment | Label housing-related official releases cooling / neutral / supportive; monthly index = (supportive − cooling) / total | Monthly |\n"
            "| Supply | Land-sale programme and public-housing targets | Annual / quarterly |\n\n"
            "Text sources: Policy Address, Budget, Hong Kong Monetary Authority (HKMA), Housing Bureau and "
            "info.gov.hk press releases. Dictionary or LLM labels require validation against a hand-audited sample. "
            "Months without releases should be missing, not automatically neutral.\n\n"
            "**Timing:** keep announcement and effective dates separate; prices may react at announcement. "
            "Verify policy dates against the Gazette or Inland Revenue Department (IRD) before modelling. "
            "**Method:** with only a few major events, begin with an event study rather than a large regression. "
            "The proposed index is not yet a validated or integrated predictor."
        )
    with st.expander("Foreign finance and legal firms · collection plan"):
        st.markdown(
            "Track overseas-parent regional headquarters (C&SD/InvestHK), SFC licensed firms and persons, "
            "foreign law firms and lawyers (Law Society of Hong Kong), Grade A office vacancy and take-up, "
            "and hiring at a fixed panel of employers through authorised sources. Pair the signals with "
            "rents and RVD Classes D–E, covering units of at least 100 m².\n\n"
            "Government job ads and a single employer snapshot cannot proxy this channel. Respect publication "
            "lags and mixed frequencies; do not treat annual observations as new monthly information. "
            "Keep employer coverage fixed so panel expansion is not mistaken for hiring growth."
        )
    with st.expander("District restaurant activity · collection plan"):
        st.markdown(
            "FEHD is the Food and Environmental Hygiene Department. Use its district restaurant licences "
            "to construct a 12-month net opening rate: (openings − closures) / stock a year earlier. "
            "Aggregate to Hong Kong Island, Kowloon and the New Territories and pair with regional prices. "
            "Review flow and place counts are possible additions only after licensing is established.\n\n"
            "A licence does not necessarily measure business activity. The 2020+ sample is shorter than "
            "the main 120-month training protocol, so pre-specify a short-sample test, for example a "
            "48-month initial window and 1–3-month horizons, alongside a district panel."
        )
    st.caption("Source: 01_home_model_reconstruction.ipynb §9–9.3. Data availability describes the supplied notebook, not a live source audit.")
    with st.expander("Asian housing markets · candidate factor"):
        st.markdown(
            "Data compares monthly Hong Kong, China and Seoul housing-price indices over their common "
            "observed period. China is a constructed 70-city existing-home composite; Seoul measures "
            "apartments and Hong Kong measures all private domestic classes. The Data tab reports "
            "same-month and one-month-lead correlations in log changes. Co-movement may reflect common "
            "rates or capital flows and does not establish an effect on Hong Kong."
        )
        st.markdown(
            "**First test:** add one lagged monthly log-price change from Seoul or the China composite "
            "to the Hong Kong Growth model at a time. Choose lags from publication dates; compare common "
            "forecast origins and horizons against the existing Growth "
            "model and a random walk. Use a holdout or account for trying multiple markets. The saved "
            "series are latest-vintage snapshots; a true point-in-time backtest needs archived releases."
        )
        st.caption("Status: data collected for comparison; coefficients, incremental forecast value and causal effects have not been estimated.")
    with st.expander("Seasonality · tested control"):
        st.subheader("Seasonality as additional regressors")
        st.markdown(
            "Seasonality is a **set of explanatory variables**, not a third economic model. Candidate controls "
            "are 11 month-of-year dummies (with an intercept) or a parsimonious annual sine/cosine pair. "
            "They can be added to a Level or Growth specification and tested on common forecast origins. "
            "The two core scenario models currently omit seasonal controls; the notebook's seasonal experiment "
            "did not improve out-of-sample performance. No untested seasonal term is silently added here."
        )
    with st.expander("ECM and further econometric checks"):
        st.subheader("If cointegration were supported: ECM")
        st.markdown(
            "An **error-correction model** combines a long-run level relation with short-run growth dynamics. "
            "Estimate the long-run relation, test its residual using cointegration-specific critical values, "
            "then include the lagged deviation in the growth equation. The equations below are illustrative, "
            "not fitted results for this sample."
        )
        st.latex(
            r"\begin{aligned} \log y_t &= \theta_0 + \theta^{\top}w_t + u_t \\"
            r"g_t &= c + \alpha\widehat{u}_{t-1} \\"
            r"&\quad + \phi g_{t-1} + \gamma^{\top}\Delta v_{t-1} + \varepsilon_t \end{aligned}"
        )
        st.markdown(
            "Here $w_t$ contains the long-run factors, $v_t$ the short-run factors, and "
            "$\\widehat{u}_{t-1}$ is last month's actual minus fitted log price. **$\\theta$ describes the "
            "equilibrium, $\\gamma$ the short-run response, and $\\alpha$ the adjustment speed.** A positive "
            "deviation is a price above the fitted relation, not necessarily fundamental overvaluation. "
            "A negative adjustment coefficient is consistent with correction; stability and statistical "
            "uncertainty still need checking.\n\n"
            "For illustration, $\\alpha=-0.05$ closes 5% of the gap each month in a fixed-equilibrium model "
            "without other dynamics: the half-life is about 13.5 months. Momentum and moving factors change "
            "this calculation. The Growth model's accumulated response extrapolates short-run dynamics; "
            "an ECM's long-run destination instead comes from the estimated equilibrium relation."
        )
        st.info(
            "The project's cointegration gate does not support estimating its ECM on the current sample. "
            "This is insufficient evidence for a long-run relation, not proof that none exists. The ECM "
            "implementation also uses different factor selection and lags from the Growth model; it is not "
            "identical apart from adding an error-correction term."
        )
        with st.expander("Further econometric checks"):
            st.markdown(
                "DOLS or FMOLS can support long-run coefficient inference under appropriate assumptions; "
                "ordinary OLS t-tests in a cointegrating regression are generally invalid. Stationary ECM "
                "terms alone do not guarantee valid inference: exogeneity, dynamics and residual behavior "
                "still matter. Johansen/VECM methods allow multiple cointegrating relations and examine "
                "which variables adjust. ARDL bounds testing can accommodate I(0)/I(1) mixtures, but not "
                "I(2) variables, and requires suitable assumptions and critical values. Gregory–Hansen "
                "tests allow a specified single structural break in the cointegrating relation."
            )
        st.caption("Detailed forecast evaluation, residual diagnostics and extension experiments remain in the source notebook.")


def price_cycle_signals(frame):
    monthly = frame[[HOME.target, "home_rent"]].sort_index().asfreq("MS")
    price = monthly[HOME.target].where(monthly[HOME.target] > 0)
    rent = monthly["home_rent"].where(monthly["home_rent"] > 0)
    signals = pd.DataFrame({"price": price, "ma5": price.rolling(5, min_periods=5).mean()})
    signals["ma5_gap_pct"] = 100 * (price / signals["ma5"] - 1)
    signals["momentum_3m_pct"] = 100 * price.pct_change(3, fill_method=None)
    price_rent = price / rent
    signals["price_rent_gap_pct"] = 100 * (price_rent / price_rent.rolling(60, min_periods=60).mean() - 1)
    signals["regime"] = np.select(
        [
            (signals["ma5_gap_pct"] > 0) & (signals["momentum_3m_pct"] > 0),
            (signals["ma5_gap_pct"] > 0) & (signals["momentum_3m_pct"] < 0),
            (signals["ma5_gap_pct"] < 0) & (signals["momentum_3m_pct"] < 0),
            (signals["ma5_gap_pct"] < 0) & (signals["momentum_3m_pct"] > 0),
        ],
        ["Heating up", "Cooling", "Weakening", "Recovery"],
        default="Neutral / unavailable",
    )
    return signals


def cycle_state_intervals(signals):
    groups = signals["regime"].ne(signals["regime"].shift()).cumsum()
    return [(group.index[0], group.index[-1] + pd.offsets.MonthBegin(1), group["regime"].iloc[0])
            for _, group in signals.groupby(groups)]


def threshold_cycle_states(signals, gap_threshold, momentum_threshold):
    if not np.isfinite([gap_threshold, momentum_threshold]).all() or min(gap_threshold, momentum_threshold) < 0:
        raise ValueError("State thresholds must be finite and non-negative")
    gap_side, momentum_side = 0, 0
    names = {(1, 1): "Heating up", (1, -1): "Cooling", (-1, -1): "Weakening", (-1, 1): "Recovery"}
    states = []
    for gap, momentum in signals[["ma5_gap_pct", "momentum_3m_pct"]].itertuples(index=False, name=None):
        if not np.isfinite([gap, momentum]).all():
            gap_side, momentum_side = 0, 0
        else:
            if gap > gap_threshold:
                gap_side = 1
            elif gap < -gap_threshold:
                gap_side = -1
            if momentum > momentum_threshold:
                momentum_side = 1
            elif momentum < -momentum_threshold:
                momentum_side = -1
        states.append(names.get((gap_side, momentum_side), "Neutral / unavailable"))
    return pd.Series(states, index=signals.index, name="regime")


def cycle_transition_probabilities(states):
    names = list(REGIME_STYLES)
    counts = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for position in range(1, len(states)):
        previous, current = states.iloc[position - 1], states.iloc[position]
        consecutive = states.index[position] == states.index[position - 1] + pd.offsets.MonthBegin(1)
        if consecutive and previous in names and current in names:
            counts.loc[previous, current] += 1
    probabilities = counts.add(0.5).div(counts.sum(axis=1) + 0.5 * len(names), axis=0)
    return counts, probabilities


def render_markov_transition_matrix(probabilities, counts):
    heatmap = go.Figure(go.Heatmap(
        z=100 * probabilities.to_numpy(), x=list(probabilities.columns), y=list(probabilities.index),
        customdata=counts.to_numpy(), zmin=0, zmax=100,
        colorscale=[[0, "#edf3f1"], [0.5, "#8ec7c2"], [1, "#075c60"]],
        colorbar={"title": "%", "ticksuffix": "%"},
        hovertemplate="Current: %{y}<br>Next: %{x}<br>Smoothed probability: %{z:.1f}%"
                      "<br>Observed transitions: %{customdata}<extra></extra>",
    ))
    for row, state in enumerate(probabilities.index):
        for col, next_state in enumerate(probabilities.columns):
            probability = 100 * probabilities.iloc[row, col]
            heatmap.add_annotation(
                x=next_state, y=state, text=f"{probability:.0f}%", showarrow=False,
                font={"color": "#ffffff" if probability >= 58 else "#132b2e", "size": 16},
            )
    heatmap.update_layout(height=370, xaxis_title="Next month", yaxis_title="Current month", **CHART)
    heatmap.update_yaxes(autorange="reversed")
    st.plotly_chart(heatmap, width="stretch", theme=None, key="bubble_markov_heatmap")
    st.caption("Each row sums to 100%. Hover over a cell for its observed count; displayed percentages include 0.5 pseudo-count per cell.")


def render_markov_transition_graph(probabilities, counts, current):
    names = list(probabilities.index)
    positions = {
        "Heating up": np.array([0.0, 2.0]),
        "Cooling": np.array([2.4, 0.0]),
        "Weakening": np.array([0.0, -2.0]),
        "Recovery": np.array([-2.4, 0.0]),
    }
    figure = go.Figure()
    radius = 0.43
    for source in names:
        origin = positions[source]
        color = REGIME_STYLES[source]["color"]
        for destination in names:
            target = positions[destination]
            probability = 100 * float(probabilities.loc[source, destination])
            width = 1.7 + probability / 25
            if destination == source:
                outward = origin / np.linalg.norm(origin)
                tangent = np.array([-outward[1], outward[0]])
                start = origin + 0.36 * outward - 0.23 * tangent
                end = origin + 0.36 * outward + 0.23 * tangent
                control_one = origin + 1.22 * outward - 0.85 * tangent
                control_two = origin + 1.22 * outward + 0.85 * tangent
                steps = np.linspace(0, 1, 51)
                points = np.array([
                    (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * control_one
                    + 3 * (1 - t) * t ** 2 * control_two + t ** 3 * end
                    for t in steps
                ])
                label = origin + 1.15 * outward
            else:
                direction = (target - origin) / np.linalg.norm(target - origin)
                start = origin + radius * direction
                end = target - radius * direction
                steps = np.linspace(0, 1, 51)
                bend = 0.55 * np.array([-direction[1], direction[0]])
                control = (start + end) / 2 + bend
                points = np.array([(1 - t) ** 2 * start + 2 * (1 - t) * t * control
                                   + t ** 2 * end for t in steps])
                label = points[17]
            figure.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1], mode="lines",
                line={"color": color, "width": width}, showlegend=False,
                hovertemplate=f"{source} → {destination}<br>Smoothed probability: {probability:.1f}%"
                              f"<br>Observed transitions: {int(counts.loc[source, destination])}<extra></extra>",
            ))
            figure.add_annotation(
                x=float(points[-1, 0]), y=float(points[-1, 1]),
                ax=float(points[-6, 0]), ay=float(points[-6, 1]),
                xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                arrowhead=3, arrowsize=1.4, arrowwidth=width, arrowcolor=color,
            )
            figure.add_annotation(
                x=float(label[0]), y=float(label[1]), text=f"{probability:.1f}%",
                showarrow=False, font={"size": 15, "color": "#132b2e"},
                bgcolor="#ffffff", bordercolor=color, borderwidth=1, borderpad=4,
            )
    for name, center in positions.items():
        style = REGIME_STYLES[name]
        figure.add_shape(
            type="circle", x0=center[0] - radius, x1=center[0] + radius,
            y0=center[1] - radius, y1=center[1] + radius,
            fillcolor=style["fill"], line={"color": style["color"], "width": 4 if name == current else 2},
        )
        figure.add_annotation(
            x=float(center[0]), y=float(center[1]), text=name.replace(" ", "<br>"),
            showarrow=False, font={"size": 17, "color": "#132b2e"},
        )
    figure.update_layout(height=820, showlegend=False, **CHART)
    figure.update_layout(margin={"l": 10, "r": 10, "t": 20, "b": 10})
    figure.update_xaxes(visible=False, range=[-4.25, 4.25], fixedrange=True)
    figure.update_yaxes(visible=False, range=[-3.8, 3.8], fixedrange=True,
                        scaleanchor="x", scaleratio=1)
    st.plotly_chart(figure, width="stretch", theme=None, key="bubble_markov_graph")
    latest_note = ("The outlined circle is the latest state; the forecast below starts there."
                   if current in positions else "The latest state is unclassified, so no circle is outlined.")
    st.caption("All 16 one-month transitions are shown. Arrow colour identifies the starting state; "
               "outer loops return to the same state. Labels are smoothed probabilities, and arrow "
               f"width follows probability. Hover for observed counts. {latest_note}")


def render_markov_outlook(signals):
    st.subheader("Markov state outlook")
    counts, probabilities = cycle_transition_probabilities(signals["regime"])
    st.caption("First-order monthly transitions estimated from all available classified history, not only the plotted 60 months. Each cell receives 0.5 pseudo-count to stabilise rare rows; a row with no observed transitions is uniform prior information, not measured evidence.")
    current = signals["regime"].iloc[-1]
    totals = counts.sum(axis=1)
    if (totals < 10).any():
        sparse = ", ".join(f"{name}: {int(totals[name])}" for name in totals.index if totals[name] < 10)
        st.info(f"Sparse transition rows (fewer than 10 observations): {sparse}. Treat these probabilities as exploratory.")
    st.markdown("**One-month transition matrix**")
    render_markov_transition_matrix(probabilities, counts)
    st.markdown("**All-state transition diagram**")
    render_markov_transition_graph(probabilities, counts, current)
    if current in probabilities.index:
        distribution = np.zeros(len(probabilities))
        distribution[list(probabilities.index).index(current)] = 1.0
        outlook = []
        for month in range(1, 7):
            distribution = distribution @ probabilities.to_numpy()
            if month in (1, 3, 6):
                outlook.append(pd.Series(distribution.copy(), index=probabilities.columns, name=f"{month}m"))
        forecasts = pd.DataFrame(outlook)
        figure = go.Figure()
        for state, style in REGIME_STYLES.items():
            figure.add_trace(go.Bar(
                x=forecasts.index, y=100 * forecasts[state], name=state,
                marker={"color": style["fill"], "line": {"color": style["color"], "width": 1}},
                texttemplate="%{y:.0f}%", textposition="inside", insidetextfont={"color": "#132b2e"},
                hovertemplate=state + "<br>%{x}: %{y:.1f}%<extra></extra>",
            ))
        figure.update_layout(height=380, barmode="stack", yaxis_title="State probability · %",
                             legend={"orientation": "h", "y": -0.2}, **CHART)
        figure.update_yaxes(range=[0, 100])
        st.plotly_chart(figure, width="stretch", theme=None, key="bubble_markov_probabilities")
        st.caption(f"Starting state: {current}. Exact matrix propagation at 1, 3 and 6 months; these are state probabilities, not probabilities of a speculative bubble or future price returns.")
    else:
        st.info("A current four-state classification is needed to initialise the Markov outlook.")
    with st.expander("Transition probabilities and observed counts"):
        st.markdown("**Smoothed transition probabilities · rows: current, columns: next month**")
        st.dataframe(probabilities.style.format("{:.1%}"), width="stretch")
        st.markdown("**Observed monthly transitions · excludes gaps and unclassified months**")
        st.dataframe(counts.assign(observed_total=totals), width="stretch")
    st.caption("This assumes stable transition probabilities and dependence only on the current state. Threshold choice, overlapping momentum windows, revisions and structural changes limit interpretation. This is an in-sample descriptive fit, not a validated point-in-time forecast.")


def render_bubble_index():
    st.subheader("Bubble index · simple price-cycle signals")
    st.caption("Descriptive indicators, not a bubble probability, a fair-value estimate or an investment recommendation.")
    signals = price_cycle_signals(raw_frame.loc[:last_date])
    thresholds = st.columns(2)
    with thresholds[0]:
        gap_threshold = st.number_input("MA5 dead band · %", min_value=0.0, max_value=5.0, value=0.25,
                                        step=0.05, key="bubble_gap_threshold")
    with thresholds[1]:
        momentum_threshold = st.number_input("3-month momentum dead band · %", min_value=0.0, max_value=10.0,
                                             value=0.5, step=0.1, key="bubble_momentum_threshold")
    signals["regime"] = threshold_cycle_states(signals, gap_threshold, momentum_threshold)
    st.caption("A side changes only beyond the positive or negative threshold. Inside each dead band its previous side is retained. States are unclassified until both sides are established; missing inputs reset them. The same thresholded states drive tiles, history colours and Markov probabilities.")
    latest = signals.iloc[-1]
    if latest[["ma5_gap_pct", "momentum_3m_pct"]].isna().any():
        st.warning("The latest month needs five consecutive price observations for the signals. Missing months are not filled.")
        return
    gap = float(latest["ma5_gap_pct"])
    momentum = float(latest["momentum_3m_pct"])
    rent_gap = latest["price_rent_gap_pct"]
    st.markdown(f"**As of {signals.index[-1]:%Y-%m} · {latest['regime']}**")
    state_tiles = []
    for name, style in REGIME_STYLES.items():
        current = name == latest["regime"]
        state_tiles.append(
            f'<div class="cycle-state{" current" if current else ""}" '
            f'style="--state-color:{style["color"]};--state-fill:{style["fill"]}">'
            f'<strong>{name}</strong><span>{style["rule"]}</span>'
            f'<small>{"Current state" if current else ""}</small></div>'
        )
    st.markdown('<div class="cycle-states">' + "".join(state_tiles) + "</div>", unsafe_allow_html=True)
    metrics = st.columns(3)
    metrics[0].metric("Price vs MA5", f"{gap:+.2f}%")
    metrics[1].metric("3-month momentum", f"{momentum:+.2f}%")
    metrics[2].metric("Price/rent vs 60-month mean", "N/A" if pd.isna(rent_gap) else f"{rent_gap:+.2f}%")
    readings = {
        "Heating up": "The thresholded MA5 side and momentum side are both positive. A heating phase is not evidence of a speculative bubble.",
        "Cooling": "The thresholded MA5 side is positive and momentum side negative: a possible cooling phase.",
        "Weakening": "Both thresholded sides are negative: a weakening phase, not proof of undervaluation.",
        "Recovery": "The thresholded MA5 side is negative and momentum side positive: a possible recovery phase.",
        "Neutral / unavailable": "Both directional sides have not yet been established; no four-state label is assigned.",
    }
    st.info(readings[latest["regime"]])
    render_markov_outlook(signals)

    st.subheader("Price-cycle history")
    st.caption("Monthly thresholded states: Heating up (red), Cooling (amber), Weakening (blue), Recovery (green). All three histories share the same state. Grey denotes an unclassified month. Dashed line: zero; dotted line: latest reading.")
    recent = signals.tail(60)
    intervals = cycle_state_intervals(recent)
    for column, label, interpretation in [
        ("ma5_gap_pct", "Price vs MA5", "Above zero: price exceeds its five-month mean. Below zero: price is below that mean."),
        ("momentum_3m_pct", "3-month momentum", "Above zero: prices have risen over three months. Below zero: prices have fallen."),
        ("price_rent_gap_pct", "Price/rent vs 60-month mean", "Above zero: the price/rent ratio exceeds its trailing mean. Below zero: it is below that mean, not necessarily undervalued."),
    ]:
        st.markdown(f"**{label}**")
        st.caption(interpretation)
        values = recent[column]
        if values.notna().sum() == 0:
            st.info("Not enough consecutive observations for this reference window.")
            continue
        limit = max(1.0, float(values.abs().max()) * 1.2)
        timeline = go.Figure()
        for start, end, regime in intervals:
            fill = REGIME_STYLES.get(regime, {"fill": "#edf0f1"})["fill"]
            timeline.add_vrect(x0=start, x1=end, fillcolor=fill, opacity=0.8, line_width=0, layer="below")
        timeline.add_hline(y=0, line_color="#607276", line_dash="dash", line_width=1)
        timeline.add_trace(go.Scatter(x=recent.index, y=values, mode="lines", name=label,
                                     line={"color": "#24484b", "width": 2.5}, connectgaps=False,
                                     customdata=recent["regime"],
                                     hovertemplate="%{x|%Y-%m}<br>%{y:+.2f}%<br>%{customdata}<extra></extra>"))
        if pd.notna(latest[column]):
            timeline.add_hline(y=float(latest[column]), line_color="#087f83", line_dash="dot", line_width=1,
                               annotation_text=f"{latest[column]:+.2f}%", annotation_position="top right")
            timeline.add_trace(go.Scatter(x=[signals.index[-1]], y=[latest[column]], mode="markers",
                             marker={"color": REGIME_STYLES.get(latest["regime"], {"color": "#607276"})["color"],
                                 "size": 10}, name="Latest",
                                         hovertemplate="Latest · %{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>"))
        timeline.update_layout(height=290, yaxis_title="%", showlegend=False, hovermode="x unified", **CHART)
        timeline.update_xaxes(range=[recent.index[0], recent.index[-1] + pd.offsets.MonthBegin(1)])
        timeline.update_yaxes(range=[-limit, limit], zeroline=False)
        st.plotly_chart(timeline, width="stretch", theme=None, key=f"bubble_history_{column}")
    with st.expander("Definitions and limitations"):
        st.markdown(
            "MA5 includes the current month and the previous four calendar months. Momentum is a "
            "three-month percentage change, not an annualised rate. Price/rent is a ratio of rebased "
            "indices: its gap is measured against its own trailing 60-month mean, not a rental yield "
            "or a fair-value benchmark. A positive value means price has outpaced rent relative to that mean. "
            "The current month is included in both trailing means. No future observations or forward fills are used."
        )
        st.latex(r"\text{MA5 gap}_t = 100\left(\frac{P_t}{\overline{P}_{t,5}}-1\right)")
        st.latex(r"\text{Momentum}_t = 100\left(\frac{P_t}{P_{t-3}}-1\right)")
        st.latex(r"\text{Price/rent gap}_t = 100\left(\frac{P_t/R_t}{\overline{(P/R)}_{t,60}}-1\right)")
        st.markdown(
            "Four states use thresholded MA5 and momentum sides, carrying prior sides inside the dead bands; "
            "price/rent is additional context, not a third classification input. Background colours show "
            "observed monthly states, not forecasts. Zero lines are descriptive boundaries, not calibrated overheating thresholds. The two price "
            "signals share the same underlying series and are not independent evidence. This is not a "
            "statistical bubble test; latest-vintage data may be revised, and the display is not a real-time backtest."
        )
    st.download_button("Download price-cycle signals", signals.to_csv().encode("utf-8"),
                       "home_price_cycle_signals.csv", "text/csv")


with tabs[4]:
    render_bubble_index()


@st.cache_data(show_spinner="Running the block-moment regime diagnostic…")
def regime_moment_analysis(path: str, modified_ns: int, block_len: int, n_perm: int,
                           adjust_ar1: bool, sub_block: int = 6, seed: int = 0):
    frame = read_monthly(path)
    price = frame[HOME.target].dropna()
    returns = (100 * np.log(price).diff()).dropna()
    n_blocks = len(returns) // block_len
    if n_blocks < 4:
        return None
    usable = returns.iloc[len(returns) - n_blocks * block_len:]
    values = usable.to_numpy().reshape(n_blocks, block_len)

    pooled_mean = float(usable.mean())
    pooled_var = float(usable.var(ddof=1))
    rho = float(max(0.0, usable.autocorr(1))) if adjust_ar1 else 0.0
    n_eff = block_len * (1 - rho) / (1 + rho)
    se_mean = float(np.sqrt(pooled_var / n_eff))
    se_var = float(pooled_var * np.sqrt(2 / (n_eff - 1)))

    block_means = values.mean(axis=1)
    block_vars = values.var(axis=1, ddof=1)
    z = np.column_stack([(block_means - pooled_mean) / se_mean, (block_vars - pooled_var) / se_var])
    centroids, labels = kmeans2(z, 2, minit="++", seed=seed)
    distance = float(np.linalg.norm(centroids[0] - centroids[1]))
    # Deterministic label order: cluster A has the lower mean return.
    if block_means[labels == 0].mean() > block_means[labels == 1].mean():
        labels = 1 - labels

    rng = np.random.default_rng(seed)
    flat = usable.to_numpy()
    n_chunks = len(flat) // sub_block
    chunks = flat[: n_chunks * sub_block].reshape(n_chunks, sub_block)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = chunks[rng.permutation(n_chunks)].ravel()[: n_blocks * block_len].reshape(n_blocks, block_len)
        z_perm = np.column_stack([
            (perm.mean(axis=1) - pooled_mean) / se_mean,
            (perm.var(axis=1, ddof=1) - pooled_var) / se_var,
        ])
        cent, _ = kmeans2(z_perm, 2, minit="++", seed=seed + 1)
        null[i] = float(np.linalg.norm(cent[0] - cent[1]))
    p_value = float((1 + (null >= distance).sum()) / (1 + n_perm))

    block_table = pd.DataFrame({
        "start": [usable.index[i * block_len] for i in range(n_blocks)],
        "end": [usable.index[(i + 1) * block_len - 1] for i in range(n_blocks)],
        "mean": block_means,
        "variance": block_vars,
        "z_mean": z[:, 0],
        "z_variance": z[:, 1],
        "excess_kurtosis": [float(sp_stats.kurtosis(row)) for row in values],
        "cluster": np.where(labels == 0, "A", "B"),
    })
    rolling = pd.DataFrame({
        "mean": usable.rolling(block_len).mean(),
        "variance": usable.rolling(block_len).var(ddof=1),
    }).dropna()
    pos = price.index.get_loc(usable.index[0])
    return {
        "returns": usable, "price": price.iloc[max(0, pos - 1):],
        "blocks": block_table, "rolling": rolling,
        "pooled_mean": pooled_mean, "pooled_var": pooled_var,
        "se_mean": se_mean, "se_var": se_var, "rho": rho, "n_eff": float(n_eff),
        "distance": distance, "null": null, "p_value": p_value,
        "kurtosis_full": float(sp_stats.kurtosis(flat)),
        "kurtosis_blocks": float(block_table["excess_kurtosis"].mean()),
    }


def render_regime_moments():
    st.subheader("Regime diagnostic · block moments vs sampling error")
    st.caption(
        "Monthly log returns of the home-price index are split into non-overlapping blocks aligned to the "
        "latest month. Each block is one point: x is its sample mean, y its sample variance. The scatter of "
        "these points is the empirical sampling distribution of (mean, variance); if one regime generated the "
        "data, the points should sit inside the sampling-error ellipses. This is a descriptive diagnostic, "
        "not a fitted regime-switching model."
    )
    block_len, n_perm = 24, 500
    adjust_ar1 = bool(st.session_state.get("regime_ar1", True))
    show_rolling = bool(st.session_state.get("regime_rolling", True))
    st.caption(
        "Fixed settings: **24-month blocks** — long enough for stable block moments while keeping at least "
        "8 blocks for clustering — and **500 permutations**, a p-value resolution of ~0.002, ample for a 5% "
        "test. Theory and further options are in the expanders at the bottom."
    )
    result = regime_moment_analysis(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns, block_len, n_perm, adjust_ar1)
    if result is None:
        st.warning("Fewer than four complete blocks are available at this block length. Choose a shorter block.")
        return
    blocks = result["blocks"]
    significant = result["p_value"] < 0.05
    threshold = float(np.percentile(result["null"], 95))

    scatter = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 120)
    for radius, dash in [(1, "dot"), (2, "dash")]:
        scatter.add_trace(go.Scatter(
            x=result["pooled_mean"] + radius * result["se_mean"] * np.cos(theta),
            y=result["pooled_var"] + radius * result["se_var"] * np.sin(theta),
            mode="lines", line={"color": "#8aa6a1", "width": 1.3, "dash": dash},
            name=f"{radius}·SE ellipse", hoverinfo="skip"))
    if show_rolling:
        rolling = result["rolling"]
        scatter.add_trace(go.Scatter(
            x=rolling["mean"], y=rolling["variance"], mode="lines",
            line={"color": "#b9c8c4", "width": 1}, name="Rolling trajectory",
            text=[f"{d:%Y-%m}" for d in rolling.index],
            hovertemplate="window ending %{text}<br>mean %{x:+.2f} · var %{y:.2f}<extra></extra>"))
    for cluster, colour in [("A", PALETTE[0]), ("B", PALETTE[1])]:
        part = blocks[blocks["cluster"] == cluster]
        scatter.add_trace(go.Scatter(
            x=part["mean"], y=part["variance"], mode="markers+text",
            marker={"color": colour, "size": 13, "line": {"color": "#ffffff", "width": 1}},
            text=[f"{s:%y.%m}–{e:%y.%m}" for s, e in zip(part["start"], part["end"])],
            textposition="top center", textfont={"size": 11, "color": "#465e62"},
            name=f"Cluster {cluster}",
            customdata=np.column_stack([part["z_mean"], part["z_variance"]]),
            hovertemplate="%{text}<br>mean %{x:+.3f} · var %{y:.3f}"
                          "<br>z = (%{customdata[0]:+.2f}, %{customdata[1]:+.2f})<extra></extra>"))
    scatter.add_trace(go.Scatter(x=[result["pooled_mean"]], y=[result["pooled_var"]], mode="markers",
                                 marker={"symbol": "x", "color": "#132b2e", "size": 11},
                                 name="Pooled (single-regime null)",
                                 hovertemplate="pooled mean %{x:+.3f} · pooled var %{y:.3f}<extra></extra>"))
    scatter.update_layout(height=520, xaxis_title="Block mean · monthly log return %",
                          yaxis_title="Block variance", legend={"orientation": "h", "y": 1.08}, **CHART)
    st.plotly_chart(scatter, width="stretch", theme=None, key="regime_scatter")
    st.caption("Dotted and dashed ellipses: one and two sampling errors around the pooled moments under a "
               "single-regime null. Points far outside the 2·SE ellipse, or two clusters separated by more "
               "than the permutation threshold, indicate distinct regimes.")

    columns = st.columns((3, 2))
    with columns[0]:
        timeline = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                 row_heights=[0.5, 0.5])
        fills = {"A": "#d2e8e6", "B": "#f8e2d6"}
        timeline.add_trace(go.Scatter(x=result["price"].index, y=result["price"], mode="lines",
                                      line={"color": "#087f83", "width": 2}, name="Price index",
                                      hovertemplate="%{x|%Y-%m}<br>index %{y:.1f}<extra></extra>"),
                           row=1, col=1)
        timeline.add_trace(go.Scatter(x=result["returns"].index, y=result["returns"], mode="lines",
                                      line={"color": "#24484b", "width": 1.6}, name="Monthly return",
                                      hovertemplate="%{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>"),
                           row=2, col=1)
        timeline.add_hline(y=0, line_color="#607276", line_dash="dash", line_width=1, row=2, col=1)
        for row in blocks.itertuples():
            timeline.add_vrect(x0=row.start, x1=row.end + pd.offsets.MonthBegin(1),
                               fillcolor=fills[row.cluster], opacity=0.85, line_width=0, layer="below",
                               row="all", col=1)
        timeline.update_layout(height=470, showlegend=False,
                               **{key: value for key, value in CHART.items() if key not in ("xaxis", "yaxis")})
        timeline.update_layout(title_text="Blocks in time · teal = cluster A (lower mean), orange = cluster B")
        timeline.update_xaxes(automargin=True, tickfont={"size": 13, "color": "#344f53"}, gridcolor="#dce5e1")
        timeline.update_yaxes(automargin=True, tickfont={"size": 13, "color": "#344f53"}, gridcolor="#dce5e1")
        timeline.update_yaxes(title_text="Price index", row=1, col=1)
        timeline.update_yaxes(title_text="Monthly log return %", row=2, col=1)
        st.plotly_chart(timeline, width="stretch", theme=None, key="regime_timeline")
        st.caption("Top: the original home-price index, so each block can be read against the raw series it "
                   "summarises. Bottom: the monthly log returns the blocks are computed from. Regimes are a "
                   "temporal claim: contiguous runs of one colour support regime structure, while colours "
                   "alternating block by block point to noise rather than regimes.")
    with columns[1]:
        null_fig = go.Figure()
        null_fig.add_trace(go.Histogram(x=result["null"], nbinsx=30, marker_color="#9fbcb7",
                                        name="Permutation null"))
        null_fig.add_vline(x=result["distance"], line_color="#d56734", line_width=2,
                           annotation_text=f"observed {result['distance']:.2f}",
                           annotation_position="top right")
        null_fig.add_vline(x=threshold, line_color="#607276", line_dash="dash", line_width=1.5,
                           annotation_text="95%", annotation_position="top left")
        null_fig.update_layout(height=470, xaxis_title="Centroid distance · SE units", yaxis_title="Count",
                               showlegend=False, **CHART)
        null_fig.update_layout(title_text="Null distribution of cluster separation")
        st.plotly_chart(null_fig, width="stretch", theme=None, key="regime_null")
        st.caption(f"{n_perm} block permutations (6-month sub-blocks reshuffled to preserve short-run "
                   "autocorrelation), each re-clustered the same way as the observed data.")

    st.markdown("**Test result**")
    metrics = st.columns(2)
    metrics[0].metric("Cluster separation", f"{result['distance']:.2f} SE",
                      help="Distance between the two k-means centroids after dividing each axis by its "
                           "single-regime sampling error.")
    metrics[1].metric("Permutation p-value", f"{result['p_value']:.3f}",
                      help="Share of block-permuted series whose best two-cluster split separates at least "
                           "as far. Small values are evidence of two regimes.")
    if significant:
        st.success("The two-cluster separation exceeds what single-regime sampling error produces in 95% of "
                   "block permutations: evidence of at least two (mean, variance) regimes.")
    else:
        st.info("The separation between the two best-fitting clusters is within the range that sampling error "
                "alone produces under one regime. These blocks do not establish two regimes.")

    with st.expander("Theory · why two moments and a distance are enough"):
        st.markdown(
            "**1 · Each block statistic has a known sampling distribution.** Take $n$ returns from one "
            "regime with mean $\\mu$ and variance $\\sigma^2$. The central limit theorem gives the block mean, "
            "and (for near-normal returns) a $\\chi^2$ law gives the block variance:"
        )
        st.latex(r"\bar r \;\dot\sim\; \mathcal N\!\left(\mu,\ \tfrac{\sigma^2}{n}\right), \qquad "
                 r"\frac{(n-1)\,s^2}{\sigma^2} \sim \chi^2_{\,n-1} \;\Rightarrow\; "
                 r"s^2 \;\dot\sim\; \mathcal N\!\left(\sigma^2,\ \tfrac{2\sigma^4}{n-1}\right)")
        st.markdown(
            "**2 · Under one regime the (mean, variance) cloud is a single ellipse.** For normal data "
            "$\\bar r$ and $s^2$ are independent, so block points scatter around the true $(\\mu, \\sigma^2)$ "
            "in an axis-aligned ellipse whose half-widths are exactly the sampling errors $SE_{\\bar r}$ and "
            "$SE_{s^2}$. Sampling error is therefore the yardstick: it says how far apart two block points may "
            "sit *by chance alone*."
        )
        st.markdown(
            "**3 · Standardising turns separation into a testable distance.** Dividing each axis by its "
            "sampling error makes every block point approximately standard bivariate normal under the "
            "single-regime null. The distance between two groups of blocks,"
        )
        st.latex(r"D=\sqrt{\left(\frac{\bar r_1-\bar r_2}{SE_{\bar r}}\right)^{2}"
                 r"+\left(\frac{s_1^{2}-s_2^{2}}{SE_{s^2}}\right)^{2}},")
        st.markdown(
            "is then a Mahalanobis-type statistic: values near $\\sqrt{2}$ are what chance produces, while "
            "$D \\gg 2\\text{–}3$ cannot be explained by sampling error. The mean axis alone is a two-sample "
            "$t$-test and the variance axis alone an $F$-test; the distance combines both in one picture."
        )
        st.markdown(
            "**4 · Two regimes split the cloud in two.** If the series alternates between regimes "
            "$(\\mu_1,\\sigma_1^2)$ and $(\\mu_2,\\sigma_2^2)$, blocks concentrate around two distinct centres "
            "and $D$ grows with $n$ — sampling errors shrink like $1/\\sqrt{n}$ while the true gap stays fixed. "
            "One regime: one cloud inside the ellipse. Two regimes: two clouds the ellipse cannot cover."
        )
    with st.expander("Advanced options"):
        st.checkbox("AR(1)-adjust sampling errors", value=True, key="regime_ar1",
                    help="Shrinks the effective sample size to n·(1−ρ)/(1+ρ) using the lag-1 "
                         "autocorrelation of monthly returns. Index smoothing makes this large.")
        st.checkbox("Rolling trajectory overlay", value=True, key="regime_rolling",
                    help="Overlapping rolling windows of the same length, for visualising when "
                         "the market moved between clusters. Not used in the test.")
    with st.expander("Kurtosis cross-check · mixture signature"):
        kurt_cols = st.columns(3)
        kurt_cols[0].metric("Full-sample excess kurtosis", f"{result['kurtosis_full']:+.2f}")
        kurt_cols[1].metric("Mean within-block excess kurtosis", f"{result['kurtosis_blocks']:+.2f}")
        kurt_gap = result["kurtosis_full"] - result["kurtosis_blocks"]
        kurt_cols[2].metric("Gap (full − within)", f"{kurt_gap:+.2f}",
                            help="Mixing blocks with different variances fattens the pooled tails even when each "
                                 "block is near-normal, so a clearly positive gap supports variance-regime mixing.")
        if kurt_gap > 0.5 and result["kurtosis_full"] > 0:
            st.caption("Pooled tails are fatter than within-block tails: consistent with variance-regime mixing.")
        else:
            st.caption("No clear mixture signature: pooled kurtosis is not meaningfully above within-block "
                       "kurtosis.")

    with st.expander("Method, formulas and limitations"):
        st.markdown(
            "Each complete block of monthly log returns contributes one point. Under a single-regime null "
            "the block statistics scatter around the pooled moments with standard errors"
        )
        st.latex(r"SE_{\bar r} = \sqrt{\hat\sigma^2 / n_{\text{eff}}}, \qquad "
                 r"SE_{s^2} = \hat\sigma^2\sqrt{2/(n_{\text{eff}}-1)}, \qquad "
                 r"n_{\text{eff}} = n\,\frac{1-\hat\rho_1}{1+\hat\rho_1}")
        st.markdown(
            "Blocks are standardised by these errors, split into two k-means clusters, and the centroid "
            "distance is compared with the same statistic on block-permuted series (6-month sub-blocks, so "
            "short-run autocorrelation survives under the null). k-means always finds *some* split, which is "
            "why only the permutation p-value — not the raw distance — carries evidential weight."
        )
        st.latex(r"D = \left\lVert \bar z_{A} - \bar z_{B} \right\rVert_2, \qquad "
                 r"p = \frac{1 + \#\{D^{\text{perm}} \ge D\}}{1 + N_{\text{perm}}}")
        st.markdown(
            "**Limitations.** The variance standard error assumes near-normal blocks; the AR(1) effective-sample "
            "correction is a first-order approximation and the index's smoothing inflates ρ, so sampling errors "
            "are indicative rather than exact. The test asks only whether *two* clusters separate — it does not "
            "estimate the number of regimes, their transition dates or probabilities (a Markov-switching model "
            "would). Overlapping rolling points are shown for context only; using them in the test would "
            "overstate the sample. Results depend on block length, so check robustness across several lengths "
            "before concluding anything."
        )
    export = blocks.copy()
    export["start"] = export["start"].dt.strftime("%Y-%m")
    export["end"] = export["end"].dt.strftime("%Y-%m")
    st.download_button("Download block moments", export.to_csv(index=False).encode("utf-8"),
                       "regime_block_moments.csv", "text/csv")


with tabs[5]:
    render_regime_moments()
