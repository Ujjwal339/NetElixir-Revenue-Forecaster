"""
llm_insights.py

AI-assisted business insights for NetElixir AIgnition 3.0

Features
--------
✓ Gemini 2.5 Flash integration (Free)
✓ Automatic offline fallback
✓ Isolation Forest anomaly detection
✓ Executive forecast summaries
✓ Business recommendations
✓ Marketing risk analysis
✓ Confidence scoring

Author: Ujjwal Kumar
"""

import os
import textwrap
from typing import Dict

import numpy as np
import pandas as pd

import google.generativeai as genai

from sklearn.ensemble import IsolationForest


# ==========================================================
# GEMINI CONFIGURATION
# ==========================================================


MODEL_NAME = "gemini-2.5-flash"



# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are a Senior Ecommerce Marketing Analyst working at a global
digital marketing agency.

You specialize in:

• Google Ads
• Meta Ads
• Microsoft Ads (Bing)

Your job is to explain revenue forecasts in business language.

Always:

- Explain WHY the forecast looks this way.
- Mention seasonality whenever relevant.
- Mention risks.
- Mention uncertainty.
- Recommend practical marketing actions.
- Recommend budget allocation changes.

Never invent numbers.

Keep answers below 300 words.

Use Markdown.

Your audience is marketing managers, not data scientists.
""").strip()


# ==========================================================
# GEMINI API
# ==========================================================

def _call_gemini(prompt: str) -> str:

    api_key = os.getenv("GEMINI_API_KEY", "")

    print("Gemini API Key Present:", bool(api_key))

    print("API Key Prefix:", api_key[:10] if api_key else "None")

    if not api_key:
        return ""

    try:

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:

        print(e)

        return f"[LLM unavailable: {e}]"
        
        
def _rule_based_summary(
    predictions: pd.DataFrame,
    stats: dict,
) -> str:
    """
    Generate a deterministic executive summary without an LLM.
    """

    overall = predictions[
        (predictions["aggregation_level"] == "overall")
        &
        (predictions["period_days"] == 30)
    ]

    if overall.empty:
        return "No forecast available."

    overall = overall.iloc[0]

    channel_df = predictions[
        (predictions["aggregation_level"] == "channel")
        &
        (predictions["period_days"] == 30)
    ].copy()

    lines = []

    lines.append("# Executive Forecast Summary\n")

    revenue = overall["revenue_p50"]
    budget = overall["budget_input"]
    roas = overall["roas_p50"]

    lines.append(
        f"Expected 30-day revenue is approximately "
        f"${revenue:,.0f} from a planned budget of "
        f"${budget:,.0f}, producing a forecast ROAS of "
        f"{roas:.2f}x."
    )

    historical_roas = stats.get(
        "blended_roas_30d",
        roas,
    )

    if roas > historical_roas * 1.05:

        lines.append(
            "\nPerformance is expected to improve compared with recent history."
        )

    elif roas < historical_roas * 0.95:

        lines.append(
            "\nPerformance is expected to decline compared with recent history."
        )

    else:

        lines.append(
            "\nPerformance is expected to remain broadly stable."
        )

    interval = (
        overall["revenue_p90"]
        -
        overall["revenue_p10"]
    )

    uncertainty = interval / max(revenue, 1)

    if uncertainty < 0.20:

        lines.append(
            "\nForecast confidence is high because the prediction interval is narrow."
        )

    elif uncertainty < 0.40:

        lines.append(
            "\nForecast confidence is moderate."
        )

    else:

        lines.append(
            "\nForecast uncertainty is relatively high. Monitor campaign performance closely."
        )

    if not channel_df.empty:

        best = channel_df.sort_values(
            "roas_p50",
            ascending=False,
        ).iloc[0]

        worst = channel_df.sort_values(
            "roas_p50",
            ascending=True,
        ).iloc[0]

        lines.append(
            f"\nBest projected channel: **{best['channel']}** "
            f"({best['roas_p50']:.2f}x ROAS)."
        )

        lines.append(
            f"Worst projected channel: **{worst['channel']}** "
            f"({worst['roas_p50']:.2f}x ROAS)."
        )

    month = stats.get("current_month", 1)

    if month in [10, 11, 12]:

        lines.append(
            "\nQ4 seasonality is likely to increase both opportunity and volatility."
        )

    elif month in [1, 2]:

        lines.append(
            "\nPost-holiday demand may reduce conversion rates."
        )

    lines.append("\nRecommended actions:")

    if roas > historical_roas:

        lines.append(
            "- Increase investment in high-performing campaigns."
        )

    else:

        lines.append(
            "- Review creatives and bidding strategy before increasing spend."
        )

    if uncertainty > 0.35:

        lines.append(
            "- Monitor campaign performance weekly due to forecast uncertainty."
        )

    if not channel_df.empty:

        lines.append(
            f"- Prioritize budget allocation toward {best['channel']} campaigns."
        )

    return "\n".join(lines)


def generate_forecast_summary(
    predictions: pd.DataFrame,
    historical_stats: Dict,
) -> str:
    """
    Generates an executive business summary for the forecast.

    Falls back to a deterministic rule-based summary whenever
    Gemini is unavailable.
    """

    overall = predictions[
        (predictions["aggregation_level"] == "overall") &
        (predictions["period_days"] == 30)
    ]

    channels = predictions[
        (predictions["aggregation_level"] == "channel") &
        (predictions["period_days"] == 30)
    ]

    if overall.empty:
        return _rule_based_summary(
            predictions,
            historical_stats,
        )

    overall = overall.iloc[0]

    confidence = 1 - (
        (overall["revenue_p90"] - overall["revenue_p10"])
        / max(overall["revenue_p50"], 1)
    )

    confidence = max(0.0, min(confidence, 1.0))

    channel_summary = []

    for _, row in channels.iterrows():

        channel_summary.append(
            f"""
Channel : {row['channel']}
Budget : ${row['budget_input']:,.0f}
Revenue Range : ${row['revenue_p10']:,.0f} to ${row['revenue_p90']:,.0f}
Median Revenue : ${row['revenue_p50']:,.0f}
ROAS : {row['roas_p50']:.2f}
"""
        )

    prompt = f"""
You are an experienced ecommerce marketing consultant.

Analyze the following revenue forecast.

Overall Forecast

Budget
${overall['budget_input']:,.0f}

Revenue

Conservative
${overall['revenue_p10']:,.0f}

Expected
${overall['revenue_p50']:,.0f}

Optimistic
${overall['revenue_p90']:,.0f}

Expected ROAS

{overall['roas_p50']:.2f}

Forecast Confidence

{confidence:.2%}

Historical Performance

Trailing 30 Day ROAS
{historical_stats['blended_roas_30d']:.2f}

Trailing 60 Day ROAS
{historical_stats['blended_roas_60d']:.2f}

Highest Performing Channel

{historical_stats['top_channel_by_roas']}

Current Month

{historical_stats['current_month']}

Channel Forecast

{''.join(channel_summary)}

Write a concise business report.

The report must contain exactly these sections.

Executive Summary

Performance Drivers

Channel Analysis

Business Risks

Recommended Budget Allocation

Recommended Actions

Keep the response under 300 words.

Do not invent values.

Base every statement on the supplied forecast.
"""

    result = _call_gemini(prompt)

    if result.strip():
        return result

    return _rule_based_summary(
        predictions,
        historical_stats,
    )
    
    
def generate_anomaly_report(daily_df: pd.DataFrame) -> str:
    """
    Detect anomalous campaign behaviour using Isolation Forest.
    """

    if daily_df.empty:
        return "No historical data available."

    daily_df = daily_df.copy()

    daily_df["roas"] = (
        daily_df["revenue"] /
        daily_df["spend"].replace(0, np.nan)
    ).fillna(0)

    features = daily_df[
        [
            "spend",
            "revenue",
            "roas"
        ]
    ].fillna(0)

    detector = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
    )

    daily_df["anomaly"] = detector.fit_predict(features)

    anomalies = (
        daily_df[daily_df["anomaly"] == -1]
        .sort_values("date", ascending=False)
        .head(10)
    )

    if anomalies.empty:
        return (
            "No significant anomalies were detected in the available "
            "historical campaign data."
        )

    report = "# Historical Performance Anomaly Report\n\n"

    for _, row in anomalies.iterrows():

        if row["roas"] >= daily_df["roas"].median():
            status = "Higher than expected ROAS"
        else:
            status = "Lower than expected ROAS"

        report += (
            f"### {row['date'].date()} | {row['channel']}\n"
            f"- **Spend:** ${row['spend']:,.2f}\n"
            f"- **Revenue:** ${row['revenue']:,.2f}\n"
            f"- **ROAS:** {row['roas']:.2f}\n"
            f"- **Observation:** {status}\n\n"
        )

    report += (
        "### Overall Observation\n"
        "These anomalies represent campaign performance that differs "
        "significantly from historical behaviour. Review campaign settings, "
        "budget changes, attribution, tracking configuration, and seasonal "
        "events before making optimization decisions."
    )

    return report

def compute_historical_stats(unified_df: pd.DataFrame) -> Dict:
    """
    Computes historical business metrics used by the AI insight engine.
    """

    if unified_df.empty:
        return {}

    cutoff = unified_df["date"].max()

    last30 = unified_df[
        unified_df["date"] >= cutoff - pd.Timedelta(days=30)
    ]

    last60 = unified_df[
        unified_df["date"] >= cutoff - pd.Timedelta(days=60)
    ]

    def blended_roas(df):
        spend = df["spend"].sum()
        revenue = df["revenue"].sum()

        if spend == 0:
            return 0

        return revenue / spend

    def safe_growth(current, previous):
        if previous == 0:
            return 0
        return ((current - previous) / previous) * 100

    revenue30 = last30["revenue"].sum()
    revenue60 = last60["revenue"].sum()

    spend30 = last30["spend"].sum()
    spend60 = last60["spend"].sum()

    roas30 = blended_roas(last30)
    roas60 = blended_roas(last60)

    channel_summary = (
        last30
        .groupby("channel")
        .agg(
            revenue=("revenue", "sum"),
            spend=("spend", "sum")
        )
        .reset_index()
    )

    channel_summary["roas"] = (
        channel_summary["revenue"] /
        channel_summary["spend"].replace(0, np.nan)
    ).fillna(0)

    if len(channel_summary):

        top_roas_channel = (
            channel_summary
            .sort_values("roas", ascending=False)
            .iloc[0]["channel"]
        )

        top_revenue_channel = (
            channel_summary
            .sort_values("revenue", ascending=False)
            .iloc[0]["channel"]
        )

        top_spend_channel = (
            channel_summary
            .sort_values("spend", ascending=False)
            .iloc[0]["channel"]
        )

    else:

        top_roas_channel = "Google"
        top_revenue_channel = "Google"
        top_spend_channel = "Google"

    previous30 = revenue60 - revenue30

    if previous30 <= 0:
       revenue_growth = 0
    else:
        revenue_growth = ((revenue30 - previous30) / previous30) * 100

    spend_growth = safe_growth(
        spend30,
        max(spend60 - spend30, 1)
    )

    season = "Normal"

    if cutoff.month in [10, 11, 12]:
        season = "Holiday"

    elif cutoff.month in [1, 2]:
        season = "Post Holiday"

    elif cutoff.month in [6, 7]:
        season = "Mid Year"

    confidence = 1.0

    if roas60 > 0:
        confidence = max(
            0.0,
            1 - abs(roas30 - roas60) / roas60
        )

    return {

        "current_month": cutoff.month,

        "season": season,

        "blended_roas_30d": round(roas30, 3),

        "blended_roas_60d": round(roas60, 3),

        "roas_30d_historical": round(roas30, 3),

        "roas_60d_historical": round(roas60, 3),

        "total_revenue_30d": round(revenue30, 2),

        "total_spend_30d": round(spend30, 2),

        "revenue_growth": round(revenue_growth, 2),

        "spend_growth": round(spend_growth, 2),

        "forecast_confidence": round(confidence, 3),

        "top_channel_by_roas": top_roas_channel,

        "top_channel_by_revenue": top_revenue_channel,

        "top_channel_by_spend": top_spend_channel,
    }
