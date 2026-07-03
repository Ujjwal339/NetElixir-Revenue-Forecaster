

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT PATH
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ==========================================================
# PROJECT IMPORTS
# ==========================================================

from generate_features import (

    load_and_normalize,

    build_campaign_features,

)

from predict import (

    predict_campaign,

    aggregate_forecasts,

)

from llm_insights import (

    generate_forecast_summary,

    generate_anomaly_report,

    compute_historical_stats,

)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="NetElixir Revenue Forecaster",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded",

)
# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <h1 style="color:#E05A2B;">
        📈 NetElixir Revenue Forecaster
    </h1>

    <p style="font-size:17px;color:#666;">
        AI-powered probabilistic revenue and ROAS forecasting
        for Google Ads, Microsoft Ads and Meta Ads.
    </p>

    <hr>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Forecast Configuration")

st.sidebar.markdown(
    """
Configure the forecasting model, upload advertising
data and optionally enable AI-generated insights.
"""
)

# ----------------------------------------------------------
# Data Upload
# ----------------------------------------------------------

uploaded_files = st.sidebar.file_uploader(

    "Upload Advertising CSV Files",

    type=["csv"],

    accept_multiple_files=True,

    help=(
        "Upload one or more exported reports "
        "from Google Ads, Microsoft Ads "
        "or Meta Ads."
    ),

)

# ----------------------------------------------------------
# Forecast Horizon
# ----------------------------------------------------------

forecast_horizon = st.sidebar.selectbox(

    "Forecast Horizon",

    options=[30, 60, 90],

    index=0,

    help="Choose the prediction horizon.",

)

# ----------------------------------------------------------
# Gemini API Key
# ----------------------------------------------------------

api_key = st.sidebar.text_input(

    "Gemini API Key (Optional)",

    type="password",

    help=(
        "Leave blank to use the built-in "
        "rule-based insight generator."
    ),

)

if api_key.strip():

    os.environ["GEMINI_API_KEY"] = api_key.strip()
    st.sidebar.success("Gemini key loaded")
    
else:
    os.environ.pop("GEMINI_API_KEY", None)   
    
    
# ----------------------------------------------------------
# Model Information
# ----------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("Model")

st.sidebar.success(
    "XGBoost Quantile Regression"
)

st.sidebar.write(
    "**Prediction Intervals:**"
)

st.sidebar.write(
    "- P10"
)

st.sidebar.write(
    "- P50 (Median)"
)

st.sidebar.write(
    "- P90"
)

st.sidebar.write(
    "**Supported Horizons:**"
)

st.sidebar.write(
    "30 • 60 • 90 Days"
)

st.sidebar.divider()

st.sidebar.caption(
    "NetElixir AIgnition 3.0"
)
# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data(show_spinner=False)
def load_uploaded_data(uploaded_files):
    """
    Load and normalize uploaded CSV files.
    """

    import tempfile

    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:

        file_path = Path(temp_dir) / uploaded_file.name

        with open(file_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

    return load_and_normalize(temp_dir)


@st.cache_data(show_spinner=False)
def load_default_data():
    """
    Load bundled sample data.
    """

    data_dir = ROOT / "data"

    if not data_dir.exists():

        return None

    csv_files = list(
        data_dir.glob("*.csv")
    )

    if len(csv_files) == 0:

        return None

    return load_and_normalize(
        str(data_dir)
    )


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load trained forecasting models.
    """

    model_path = (
        ROOT /
        "pickle" /
        "model.pkl"
    )

    if not model_path.exists():

        return None

    with open(
        model_path,
        "rb",
    ) as file:

        models = pickle.load(
            file
        )

    return models


# ==========================================================
# LOAD USER DATA
# ==========================================================

if uploaded_files:

    df = load_uploaded_data(
        uploaded_files
    )

    st.success(

        f"Loaded {len(df):,} rows "
        f"from {len(uploaded_files)} file(s)."

    )

else:

    df = load_default_data()

    if df is not None:

        st.info(
            "Using bundled sample dataset."
        )


models = load_model()


if df is None:

    st.warning(
        "Upload one or more CSV files to continue."
    )

    st.stop()


if models is None:

    st.error(
        "No trained model found.\n\n"
        "Run train.py first."
    )

    st.stop()
    
# ==========================================================
# GENERATE INFERENCE FEATURES
# ==========================================================

@st.cache_data(show_spinner=False)
def make_inference_features(
    data: pd.DataFrame,
):
    """
    Build inference features for every campaign.
    """

    feature_frames = []

    required_columns =  {
        "channel",
        "campaign_name",
    }

    missing = required_columns - set(data.columns)

    if missing:
        st.error(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )  
        st.stop()

    grouped = data.groupby(
        [
            "channel",
            "campaign_name",
        ]
    )
    

    for _, campaign_df in grouped:

        feature_df = build_campaign_features(

            campaign_df,

            mode="inference",

        )

        if not feature_df.empty:

            feature_frames.append(

                feature_df

            )

    if len(feature_frames) == 0:

        return pd.DataFrame()

    return pd.concat(

        feature_frames,

        ignore_index=True,

    )


feature_df = make_inference_features(
    df
)


if feature_df.empty:

    st.error(
        "Unable to generate inference features."
    )

    st.stop()


# ==========================================================
# BUDGET SIMULATION
# ==========================================================

st.markdown("---")

st.subheader("Budget Simulation")

columns = st.columns(3)

channels = sorted(

    feature_df["channel"].unique()

)

channel_budgets = {}

for index, channel in enumerate(channels):

    historical_budget = (

        feature_df[
            feature_df["channel"] == channel
        ]["future_budget"]
        .sum()

    )

    with columns[index % 3]:

        channel_budgets[channel] = st.number_input(

            label=f"{channel} Budget",

            min_value=0.0,

            value=float(

                round(

                    historical_budget,

                    0,

                )

            ),

            step=500.0,

            format="%.0f",

            help=(
                f"Historical 30-day spend: "
                f"${historical_budget:,.0f}"
            ),

        )


# ==========================================================
# CAMPAIGN BUDGET ALLOCATION
# ==========================================================

campaign_budget_override = {}

for channel, total_budget in channel_budgets.items():

    channel_features = feature_df[

        feature_df["channel"] == channel

    ]

    historical_total = (

        channel_features["future_budget"]

        .sum()

    )

    for _, campaign in channel_features.iterrows():

        if historical_total > 0:

            allocation = (

                campaign["future_budget"]

                / historical_total

            )

        else:

            allocation = (

                1.0 /

                max(

                    len(channel_features),

                    1,

                )

            )

        monthly_budget = (

            total_budget

            * allocation

            / (

                forecast_horizon /

                30

            )

        )

        campaign_budget_override[

            campaign["campaign_name"]

        ] = monthly_budget
# ==========================================================
# RUN FORECAST
# ==========================================================

@st.cache_data(show_spinner=False)
def run_forecast(
    feature_data,
    budget_override,
    _models,
):

    campaign_forecasts = []

    forecast_date = str(
        date.today()
    )

    for _, campaign in feature_data.iterrows():

        campaign_name = campaign["campaign_name"]

        custom_budget = budget_override.get(
            campaign_name
        )

        prediction = predict_campaign(

            _models,

            campaign,

            custom_budget,

        )

        prediction["forecast_date"] = forecast_date

        prediction["channel"] = campaign["channel"]

        prediction["campaign_type"] = campaign["campaign_type"]

        prediction["campaign_name"] = campaign_name

        prediction["aggregation_level"] = "campaign"

        campaign_forecasts.append(
            prediction
        )

    if len(campaign_forecasts) == 0:

        return pd.DataFrame()

    campaign_forecasts = pd.concat(

        campaign_forecasts,

        ignore_index=True,

    )

    channel_forecasts = aggregate_forecasts(

        campaign_forecasts,

        "channel",

        [

            "forecast_date",

            "channel",

        ],

    )

    channel_forecasts["campaign_type"] = "ALL"

    channel_forecasts["campaign_name"] = "ALL"

    campaign_type_forecasts = aggregate_forecasts(

        campaign_forecasts,

        "campaign_type",

        [

            "forecast_date",

            "channel",

            "campaign_type",

        ],

    )

    campaign_type_forecasts["campaign_name"] = "ALL"

    overall_forecasts = aggregate_forecasts(

        campaign_forecasts,

        "overall",

        [

            "forecast_date",

        ],

    )

    overall_forecasts["channel"] = "ALL"

    overall_forecasts["campaign_type"] = "ALL"

    overall_forecasts["campaign_name"] = "ALL"

    forecast = pd.concat(

        [

            overall_forecasts,

            channel_forecasts,

            campaign_type_forecasts,

            campaign_forecasts,

        ],

        ignore_index=True,

    )

    return forecast


predictions = run_forecast(
    feature_df,
    campaign_budget_override,
    models,
)


if predictions.empty:

    st.error(
        "Forecast generation failed."
    )

    st.stop()
# ==========================================================
# FORECAST SUMMARY
# ==========================================================

st.markdown("---")

st.subheader(
    f"{forecast_horizon}-Day Forecast Summary"
)

overall_forecast = predictions[

    (predictions["aggregation_level"] == "overall")

    &

    (predictions["period_days"] == forecast_horizon)

]

if overall_forecast.empty:

    st.error(
        "No overall forecast available."
    )

    st.stop()

overall = overall_forecast.iloc[0]

revenue_width = (

    overall["revenue_p90"]

    -

    overall["revenue_p10"]

)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(

        label="Median Revenue",

        value=f"${overall['revenue_p50']:,.0f}",

        delta=(
            f"P10 ${overall['revenue_p10']:,.0f}"
            f" | "
            f"P90 ${overall['revenue_p90']:,.0f}"
        ),

    )

with metric2:

    st.metric(

        label="Median ROAS",

        value=f"{overall['roas_p50']:.2f}x",

        delta=(
            f"{overall['roas_p10']:.2f}x"
            f" – "
            f"{overall['roas_p90']:.2f}x"
        ),

    )

with metric3:

    st.metric(

        label="Forecast Budget",

        value=f"${overall['budget_input']:,.0f}",

    )

with metric4:

    st.metric(

        label="Prediction Interval",

        value=f"${revenue_width:,.0f}",

        delta="P10 → P90",

    )
# ==========================================================
# FORECAST VISUALIZATIONS
# ==========================================================

left_column, right_column = st.columns(2)

# ----------------------------------------------------------
# Revenue Forecast
# ----------------------------------------------------------

with left_column:

    st.subheader(
        "Revenue Forecast"
    )

    overall_all = (

        predictions[
            predictions["aggregation_level"] == "overall"
        ]

        .sort_values(
            "period_days"
        )

    )

    revenue_chart = go.Figure()

    revenue_chart.add_trace(

        go.Scatter(

            x=overall_all["period_days"],

            y=overall_all["revenue_p90"],

            mode="lines",

            line=dict(
                color="rgba(224,90,43,0.25)"
            ),

            name="P90",

            showlegend=False,

        )

    )

    revenue_chart.add_trace(

        go.Scatter(

            x=overall_all["period_days"],

            y=overall_all["revenue_p10"],

            mode="lines",

            fill="tonexty",

            fillcolor="rgba(224,90,43,0.15)",

            line=dict(
                color="rgba(224,90,43,0.25)"
            ),

            name="Prediction Interval",

        )

    )

    revenue_chart.add_trace(

        go.Scatter(

            x=overall_all["period_days"],

            y=overall_all["revenue_p50"],

            mode="lines+markers",

            line=dict(

                color="#E05A2B",

                width=3,

            ),

            marker=dict(
                size=8
            ),

            name="Median Forecast",

        )

    )

    revenue_chart.update_layout(

        height=360,

        margin=dict(

            l=10,

            r=10,

            t=10,

            b=10,

        ),

        xaxis_title="Forecast Horizon (Days)",

        yaxis_title="Revenue (USD)",

        template="plotly_white",

    )

    st.plotly_chart(

        revenue_chart,

        use_container_width=True,

    )


# ----------------------------------------------------------
# Channel ROAS
# ----------------------------------------------------------

with right_column:

    st.subheader(
        "Channel ROAS"
    )

    channel_predictions = predictions[

        (predictions["aggregation_level"] == "channel")

        &

        (predictions["period_days"] == forecast_horizon)

    ]

    roas_chart = go.Figure()

    for _, row in channel_predictions.iterrows():

        roas_chart.add_trace(

            go.Bar(

                x=[row["channel"]],

                y=[row["roas_p50"]],

                name=row["channel"],

                error_y=dict(

                    type="data",

                    symmetric=False,

                    array=[
                        row["roas_p90"] -
                        row["roas_p50"]
                    ],

                    arrayminus=[
                        row["roas_p50"] -
                        row["roas_p10"]
                    ],

                ),

            )

        )

    roas_chart.update_layout(

        height=360,

        margin=dict(

            l=10,

            r=10,

            t=10,

            b=10,

        ),

        showlegend=False,

        yaxis_title="ROAS",

        template="plotly_white",

    )

    st.plotly_chart(

        roas_chart,

        use_container_width=True,

    )


# ----------------------------------------------------------
# Campaign Type Breakdown
# ----------------------------------------------------------

st.subheader(
    "Campaign Type Revenue"
)

campaign_type_predictions = (

    predictions[

        (predictions["aggregation_level"] == "campaign_type")

        &

        (predictions["period_days"] == forecast_horizon)

    ]

    .sort_values(

        "revenue_p50",

        ascending=False,

    )

)

campaign_chart = px.bar(

    campaign_type_predictions,

    x="campaign_type",

    y=[

        "revenue_p10",

        "revenue_p50",

        "revenue_p90",

    ],

    barmode="group",

    labels={

        "value": "Revenue",

        "variable": "",

    },

    color_discrete_map={

        "revenue_p10": "#F7C59F",

        "revenue_p50": "#E05A2B",

        "revenue_p90": "#7A2D0E",

    },

)

campaign_chart.update_layout(

    height=380,

    margin=dict(

        l=10,

        r=10,

        t=10,

        b=10,

    ),

    template="plotly_white",

)

st.plotly_chart(

    campaign_chart,

    use_container_width=True,

)
# ==========================================================
# TOP CAMPAIGNS
# ==========================================================

st.markdown("---")

st.subheader(
    f"Top Campaign Forecasts ({forecast_horizon} Days)"
)

campaign_predictions = (

    predictions[

        (predictions["aggregation_level"] == "campaign")

        &

        (predictions["period_days"] == forecast_horizon)

    ]

    .sort_values(

        "revenue_p50",

        ascending=False,

    )

    .head(10)

)

display_table = campaign_predictions[

    [

        "channel",

        "campaign_type",

        "campaign_name",

        "budget_input",

        "revenue_p10",

        "revenue_p50",

        "revenue_p90",

        "roas_p10",

        "roas_p50",

        "roas_p90",

    ]

].copy()

display_table.columns = [

    "Channel",

    "Campaign Type",

    "Campaign",

    "Budget",

    "Revenue P10",

    "Revenue P50",

    "Revenue P90",

    "ROAS P10",

    "ROAS P50",

    "ROAS P90",

]

currency_columns = [

    "Budget",

    "Revenue P10",

    "Revenue P50",

    "Revenue P90",

]

for column in currency_columns:

    display_table[column] = (

        display_table[column]

        .map(

            lambda value:

            f"${value:,.0f}"

        )

    )

roas_columns = [

    "ROAS P10",

    "ROAS P50",

    "ROAS P90",

]

for column in roas_columns:

    display_table[column] = (

        display_table[column]

        .map(

            lambda value:

            f"{value:.2f}x"

        )

    )

st.dataframe(

    display_table,

    hide_index=True,

    use_container_width=True,

)


# ==========================================================
# AI INSIGHTS
# ==========================================================

st.markdown("---")

st.subheader("AI Insights")

# ----------------------------------------------------------
# Session State
# ----------------------------------------------------------

if "forecast_summary" not in st.session_state:
    st.session_state.forecast_summary = ""

if "anomaly_report" not in st.session_state:
    st.session_state.anomaly_report = ""

# ----------------------------------------------------------
# Buttons
# ----------------------------------------------------------

left_ai, right_ai = st.columns(2)

# ===========================
# Forecast Summary
# ===========================

with left_ai:

    if st.button(
        "Generate Forecast Summary",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating executive summary..."
        ):

            historical_stats = compute_historical_stats(df)

            st.session_state.forecast_summary = (
                generate_forecast_summary(
                    predictions,
                    historical_stats,
                )
            )

        st.success("Forecast summary generated.")

# ===========================
# Historical Anomalies
# ===========================

with right_ai:

    if st.button(
        "Detect Historical Anomalies",
        use_container_width=True,
    ):

        with st.spinner(
            "Analyzing historical data..."
        ):

            daily_data = (
                df.groupby(
                    [
                        "date",
                        "channel",
                    ]
                )[
                    [
                        "spend",
                        "revenue",
                    ]
                ]
                .sum()
                .reset_index()
            )

            # Always use deterministic anomaly detection
            st.session_state.anomaly_report = (
                generate_anomaly_report(
                    daily_data
                )
            )

        st.success("Historical anomaly analysis completed.")

# ----------------------------------------------------------
# Results
# ----------------------------------------------------------

summary_col, anomaly_col = st.columns(2)

with summary_col:

    if st.session_state.forecast_summary:

        st.markdown("### Executive Forecast Summary")

        st.markdown(
            st.session_state.forecast_summary
        )

with anomaly_col:

    if st.session_state.anomaly_report:

        st.markdown("### Historical Performance Anomalies")

        st.markdown(
            st.session_state.anomaly_report
        )

# ----------------------------------------------------------
# Clear Button
# ----------------------------------------------------------

st.markdown("")

_, center, _ = st.columns([1, 2, 1])

with center:

    if st.button(
        "🗑 Clear AI Insights",
        use_container_width=True,
    ):

        st.session_state.forecast_summary = ""

        st.session_state.anomaly_report = ""

        st.rerun()
# ==========================================================
# DOWNLOAD RESULTS
# ==========================================================

st.markdown("---")

st.subheader(
    "Export Forecast"
)

prediction_csv = predictions.to_csv(
    index=False,
).encode("utf-8")

st.download_button(

    label="Download Forecast (CSV)",

    data=prediction_csv,

    file_name="predictions.csv",

    mime="text/csv",

    use_container_width=True,

)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

left_footer, center_footer, right_footer = st.columns(3)

with left_footer:

    st.caption(
        "NetElixir AIgnition 3.0 Hackathon"
    )

with center_footer:

    st.caption(
        "Forecast Model: XGBoost Quantile Regression"
    )

with right_footer:

    if "GEMINI_API_KEY" in os.environ:

        st.caption(
            "AI Insights"
        )

    else:

        st.caption(
            "AI Insights: Rule-Based"
        )


st.caption(
    "Developed for probabilistic revenue forecasting across Google Ads, Microsoft Ads and Meta Ads."
)
