import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==========================================================
# PLATFORM DETECTION
# ==========================================================

def detect_platform(df: pd.DataFrame, filepath: str) -> str:
    """
    Detect advertising platform from filename and column names.
    """

    filename = Path(filepath).stem.lower()
    columns = {c.lower() for c in df.columns}

    if any(name in filename for name in ["google", "google_ads"]):
        return "google"

    if any(name in filename for name in ["bing", "microsoft"]):
        return "bing"

    if any(name in filename for name in ["meta", "facebook"]):
        return "meta"

    if {
        "segments_date",
        "metrics_cost_micros",
        "campaign_budget_amount",
    }.issubset(columns):
        return "google"

    if {
        "timeperiod",
        "campaignid",
        "revenue",
    }.issubset(columns):
        return "bing"

    if {
        "date_start",
        "campaign_id",
        "daily_budget",
    }.issubset(columns):
        return "meta"

    raise ValueError(
        f"Unable to determine advertising platform for file: {filepath}"
    )


# ==========================================================
# CAMPAIGN TYPE INFERENCE
# ==========================================================

def infer_meta_campaign_type(name: str) -> str:
    """
    Infer Meta campaign type from campaign name.
    """

    name = str(name).lower()

    if "prospecting" in name:
        return "Prospecting"

    if "remarketing" in name:
        return "Remarketing"

    if "retarget" in name:
        return "Remarketing"

    if "catalog" in name:
        return "Shopping"

    if "video" in name:
        return "Video"

    return "Generic"


GOOGLE_TYPE_MAP = {
    "SEARCH": "Search",
    "SHOPPING": "Shopping",
    "DISPLAY": "Display",
    "VIDEO": "Video",
    "PERFORMANCE_MAX": "PerformanceMax",
    "DEMAND_GEN": "DemandGen",
}

# ==========================================================
# DATA NORMALIZATION
# ==========================================================

def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Meta Ads dataset into the common schema.
    """

    budget = pd.to_numeric(
        df.get("daily_budget", np.nan),
        errors="coerce"
    )

    return pd.DataFrame({

        "channel": "Meta",

        "campaign_id":
            df["campaign_id"].astype(str),

        "campaign_name":
            df["campaign_name"].astype(str),

        "campaign_type":
            df["campaign_name"].apply(
                infer_meta_campaign_type
            ),

        "date":
            pd.to_datetime(df["date_start"]),

        "spend":
            pd.to_numeric(
                df["spend"],
                errors="coerce"
            ).fillna(0),

        "revenue":
            pd.to_numeric(
                df["conversion"],
                errors="coerce"
            ).fillna(0),

        "clicks":
            pd.to_numeric(
                df["clicks"],
                errors="coerce"
            ).fillna(0),

        "impressions":
            pd.to_numeric(
                df["impressions"],
                errors="coerce"
            ).fillna(0),

        "conversions":
            pd.to_numeric(
                df["conversion"],
                errors="coerce"
            ).fillna(0),

        "budget":
            budget.fillna(budget.median())
    })


def normalize_bing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Microsoft Bing Ads dataset.
    """

    budget = pd.to_numeric(
        df["DailyBudget"],
        errors="coerce"
    )

    budget = budget.fillna(budget.median())

    return pd.DataFrame({

        "channel": "Bing",

        "campaign_id":
            df["CampaignId"].astype(str),

        "campaign_name":
            df["CampaignName"].astype(str),

        "campaign_type":
            df["CampaignType"].fillna("Other"),

        "date":
            pd.to_datetime(df["TimePeriod"]),

        "spend":
            pd.to_numeric(
                df["Spend"],
                errors="coerce"
            ).fillna(0),

        "revenue":
            pd.to_numeric(
                df["Revenue"],
                errors="coerce"
            ).fillna(0),

        "clicks":
            pd.to_numeric(
                df["Clicks"],
                errors="coerce"
            ).fillna(0),

        "impressions":
            pd.to_numeric(
                df["Impressions"],
                errors="coerce"
            ).fillna(0),

        "conversions":
            pd.to_numeric(
                df["Conversions"],
                errors="coerce"
            ).fillna(0),

        "budget":
            budget
    })


def normalize_google(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Google Ads dataset.
    """

    budget = pd.to_numeric(
        df["campaign_budget_amount"],
        errors="coerce"
    )

    budget = budget.fillna(budget.median())

    return pd.DataFrame({

        "channel": "Google",

        "campaign_id":
            df["campaign_id"].astype(str),

        "campaign_name":
            df["campaign_name"].astype(str),

        "campaign_type":
            df["campaign_advertising_channel_type"]
            .map(GOOGLE_TYPE_MAP)
            .fillna("Other"),

        "date":
            pd.to_datetime(df["segments_date"]),

        "spend":
            (
                pd.to_numeric(
                    df["metrics_cost_micros"],
                    errors="coerce"
                ).fillna(0) / 1e6
            ),

        "revenue":
            pd.to_numeric(
                df["metrics_conversions_value"],
                errors="coerce"
            ).fillna(0),

        "clicks":
            pd.to_numeric(
                df["metrics_clicks"],
                errors="coerce"
            ).fillna(0),

        "impressions":
            pd.to_numeric(
                df["metrics_impressions"],
                errors="coerce"
            ).fillna(0),

        "conversions":
            pd.to_numeric(
                df["metrics_conversions"],
                errors="coerce"
            ).fillna(0),

        "budget":
            budget
    })


NORMALIZERS = {
    "google": normalize_google,
    "bing": normalize_bing,
    "meta": normalize_meta,
}


def load_and_normalize(data_dir: str) -> pd.DataFrame:
    """
    Load every CSV file from the supplied directory and
    normalize all advertising platforms into one schema.
    """

    data_path = Path(data_dir)

    frames = []

    for csv_file in sorted(data_path.glob("*.csv")):

        raw = pd.read_csv(csv_file)

        try:

            platform = detect_platform(
                raw,
                str(csv_file)
            )

            normalized = NORMALIZERS[platform](raw)

            frames.append(normalized)

            print(
                f"Loaded {csv_file.name} "
                f"({platform}) "
                f"{len(normalized)} rows"
            )

        except Exception as e:

            print(
                f"Skipping {csv_file.name}: {e}"
            )

    if not frames:

        raise RuntimeError(
            "No supported CSV files were found."
        )

    df = (
        pd.concat(
            frames,
            ignore_index=True
        )
        .sort_values(
            [
                "channel",
                "campaign_name",
                "date"
            ]
        )
        .reset_index(drop=True)
    )

    df["budget"] = (
        df.groupby("channel")["budget"]
        .transform(
            lambda x: x.fillna(x.median())
        )
        .fillna(50)
    )

    return df


# ==========================================================
# CATEGORICAL ENCODING
# ==========================================================

CHANNEL_CODES = {
    "Google": 0,
    "Bing": 1,
    "Meta": 2,
}

CAMP_TYPE_CODES = {
    "Search": 0,
    "PerformanceMax": 1,
    "Shopping": 2,
    "Display": 3,
    "Video": 4,
    "DemandGen": 5,
    "Audience": 6,
    "Prospecting": 7,
    "Remarketing": 8,
    "Generic": 9,
    "Other": 10,
}


# ==========================================================
# FEATURE ENGINEERING HELPERS
# ==========================================================

def safe_divide(numerator, denominator):
    """
    Safe division that returns zero when denominator is zero.
    """

    numerator = float(numerator)
    denominator = float(denominator)

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def safe_roas(revenue, spend):
    """
    Calculate Return on Advertising Spend.
    """

    return safe_divide(revenue, spend)


def calculate_ctr(clicks, impressions):
    """
    Click Through Rate
    """

    return safe_divide(clicks, impressions)


def calculate_cvr(conversions, clicks):
    """
    Conversion Rate
    """

    return safe_divide(conversions, clicks)


def calculate_cpc(spend, clicks):
    """
    Cost Per Click
    """

    return safe_divide(spend, clicks)


def calculate_cpm(spend, impressions):
    """
    Cost Per 1000 Impressions
    """

    return safe_divide(spend * 1000, impressions)


def calculate_rpc(revenue, clicks):
    """
    Revenue Per Click
    """

    return safe_divide(revenue, clicks)


def calculate_aov(revenue, conversions):
    """
    Average Order Value
    """

    return safe_divide(revenue, conversions)


def calculate_budget_utilization(spend, budget, days):
    """
    Average budget utilization over a rolling window.
    """

    return safe_divide(
        spend,
        budget * max(days, 1)
    )


def calculate_trend(current_value, previous_value):
    """
    Relative trend between two periods.
    """

    return safe_divide(
        current_value - previous_value,
        previous_value + 1
    )


def calculate_volatility(series):
    """
    Rolling standard deviation.
    """

    if len(series) <= 1:
        return 0.0

    return float(series.std())


def campaign_age(current_date, first_date):
    """
    Campaign age in days.
    """

    return (current_date - first_date).days
    
    
    
    
def build_campaign_features(grp: pd.DataFrame, mode: str):
    """
    Generate rolling features for a single campaign.

    Parameters
    ----------
    grp : Daily campaign history sorted by date

    mode :
        train      -> create features + targets
        inference  -> create latest feature row only
    """

    grp = grp.sort_values("date").reset_index(drop=True)

    n = len(grp)

    rows = []

    if mode == "train":
        date_range = range(29, n)
    else:
        date_range = [n - 1]

    first_campaign_date = grp["date"].min()

    for i in date_range:

        current_date = grp.loc[i, "date"]

        # -------------------------------------------------
        # Rolling Windows
        # -------------------------------------------------

        w7 = grp.loc[max(0, i - 6):i]

        w14 = grp.loc[max(0, i - 13):i]

        w30 = grp.loc[max(0, i - 29):i]

        # -------------------------------------------------
        # Spend
        # -------------------------------------------------

        spend7 = w7["spend"].sum()

        spend14 = w14["spend"].sum()

        spend30 = w30["spend"].sum()

        # -------------------------------------------------
        # Revenue
        # -------------------------------------------------

        revenue7 = w7["revenue"].sum()

        revenue14 = w14["revenue"].sum()

        revenue30 = w30["revenue"].sum()

        # -------------------------------------------------
        # Engagement
        # -------------------------------------------------

        clicks30 = w30["clicks"].sum()

        impressions30 = w30["impressions"].sum()

        conversions30 = w30["conversions"].sum()

        # -------------------------------------------------
        # ROAS
        # -------------------------------------------------

        roas7 = safe_roas(
            revenue7,
            spend7
        )

        roas14 = safe_roas(
            revenue14,
            spend14
        )

        roas30 = safe_roas(
            revenue30,
            spend30
        )

        # -------------------------------------------------
        # Marketing Metrics
        # -------------------------------------------------

        ctr30 = calculate_ctr(
            clicks30,
            impressions30
        )

        cvr30 = calculate_cvr(
            conversions30,
            clicks30
        )

        cpc30 = calculate_cpc(
            spend30,
            clicks30
        )

        cpm30 = calculate_cpm(
            spend30,
            impressions30
        )

        rpc30 = calculate_rpc(
            revenue30,
            clicks30
        )

        aov30 = calculate_aov(
            revenue30,
            conversions30
        )

        # -------------------------------------------------
        # Budget Metrics
        # -------------------------------------------------

        avg_budget30 = w30["budget"].mean()

        budget_utilization30 = calculate_budget_utilization(
            spend30,
            avg_budget30,
            len(w30)
        )

        # -------------------------------------------------
        # Trend Features
        # -------------------------------------------------

        avg_spend7 = spend7 / max(len(w7), 1)
        avg_spend30 = spend30 / max(len(w30), 1)

        spend_trend = calculate_trend(
            avg_spend7,
            avg_spend30
        )

        avg_revenue7 = revenue7 / max(len(w7), 1)
        avg_revenue30 = revenue30 / max(len(w30), 1)

        revenue_trend = calculate_trend(
            avg_revenue7,
            avg_revenue30
        )

        roas_trend = calculate_trend(
            roas7,
            roas30
        )

        # -------------------------------------------------
        # Volatility Features
        # -------------------------------------------------

        revenue_std30 = calculate_volatility(
            w30["revenue"]
        )

        spend_std30 = calculate_volatility(
            w30["spend"]
        )

        roas_daily = (
            w30["revenue"] /
            w30["spend"].replace(0, np.nan)
        ).fillna(0)

        roas_std30 = calculate_volatility(
            roas_daily
        )

        # -------------------------------------------------
        # Lag Features
        # -------------------------------------------------

        revenue_lag1 = (
            grp.loc[i - 1, "revenue"]
            if i >= 1 else 0
        )

        revenue_lag7 = (
            grp.loc[i - 7, "revenue"]
            if i >= 7 else revenue7
        )

        spend_lag1 = (
            grp.loc[i - 1, "spend"]
            if i >= 1 else 0
        )

        spend_lag7 = (
            grp.loc[i - 7, "spend"]
            if i >= 7 else spend7
        )

        roas_lag7 = safe_roas(
            revenue_lag7,
            spend_lag7
        )

        # -------------------------------------------------
        # Campaign Features
        # -------------------------------------------------
        age_days = campaign_age(
            current_date,
            first_campaign_date
        )

        efficiency = (
            revenue30
            / 
            (spend30 + 1)
        ) * cvr30

        day_of_week = current_date.weekday()

        day_of_month = current_date.day

        week_of_year = current_date.isocalendar().week

        is_weekend = int(
            day_of_week >= 5
        )
        # -------------------------------------------------
        # Feature Dictionary
        # -------------------------------------------------

        row = {

            # ---------------------------------------------
            # Campaign Information
            # ---------------------------------------------

            "date": current_date,

            "channel": grp.loc[i, "channel"],

            "campaign_name": grp.loc[i, "campaign_name"],

            "campaign_type": grp.loc[i, "campaign_type"],

            # ---------------------------------------------
            # Rolling Spend
            # ---------------------------------------------

            "spend_7d": spend7,

            "spend_14d": spend14,

            "spend_30d": spend30,

            "spend_trend": spend_trend,

            "spend_std_30": spend_std30,

            # ---------------------------------------------
            # Rolling Revenue
            # ---------------------------------------------

            "revenue_7d": revenue7,

            "revenue_14d": revenue14,

            "revenue_30d": revenue30,

            "revenue_trend": revenue_trend,

            "revenue_std_30": revenue_std30,

            # ---------------------------------------------
            # ROAS
            # ---------------------------------------------

            "roas_7d": roas7,

            "roas_14d": roas14,

            "roas_30d": roas30,

            "roas_trend": roas_trend,

            "roas_std_30": roas_std30,

            # ---------------------------------------------
            # Marketing Metrics
            # ---------------------------------------------

            "ctr_30d": ctr30,

            "cvr_30d": cvr30,

            "cpc_30d": cpc30,

            "cpm_30d": cpm30,

            "rpc_30d": rpc30,

            "aov_30d": aov30,

            # ---------------------------------------------
            # Budget Features
            # ---------------------------------------------

            "budget_avg_30d": avg_budget30,

            "budget_util_30d": budget_utilization30,

            # ---------------------------------------------
            # Lag Features
            # ---------------------------------------------

            "revenue_lag1": revenue_lag1,

            "revenue_lag7": revenue_lag7,

            "spend_lag1": spend_lag1,

            "spend_lag7": spend_lag7,

            "roas_lag7": roas_lag7,

            # ---------------------------------------------
            # Campaign Features
            # ---------------------------------------------

            "campaign_age": min(age_days, 365),

            "campaign_efficiency": efficiency,

            # ---------------------------------------------
            # Calendar Features
            # ---------------------------------------------

            "day_of_week": day_of_week,

            "day_of_month": day_of_month,

            "week_of_year": week_of_year,

            "month": current_date.month,

            "quarter": current_date.quarter,

            "is_weekend": is_weekend,
            "is_holiday": int(current_date.month in [10, 11, 12]),
            "is_q4": int(current_date.month >= 10),

            # ---------------------------------------------
            # Encoded Features
            # ---------------------------------------------

            "channel_enc": CHANNEL_CODES.get(
                grp.loc[i, "channel"],
                0
            ),

            "camp_type_enc": CAMP_TYPE_CODES.get(
                grp.loc[i, "campaign_type"],
                10
            ),

            # ---------------------------------------------
            # Future Budget
            # ---------------------------------------------

            "future_budget": spend30,  # trailing 30d spend = best proxy for next 30d budget
        }
        # -------------------------------------------------
        # Training Targets
        # -------------------------------------------------

        if mode == "train":

            future30 = grp.loc[
                i + 1:min(i + 30, n - 1)
            ]

            future60 = grp.loc[
                i + 1:min(i + 60, n - 1)
            ]

            future90 = grp.loc[
                i + 1:min(i + 90, n - 1)
            ]

            # Skip rows with insufficient future history

            if len(future30) < 20:
                continue

            # -----------------------------
            # Revenue Targets
            # -----------------------------

            row["revenue_target_30d"] = (
                future30["revenue"].sum()
            )

            row["revenue_target_60d"] = (
                future60["revenue"].sum()
                if len(future60) >= 40
                else np.nan
            )

            row["revenue_target_90d"] = (
                future90["revenue"].sum()
                if len(future90) >= 60
                else np.nan
            )

            # -----------------------------
            # Future Spend
            # -----------------------------

            row["spend_future_30d"] = (
                future30["spend"].sum()
            )

            row["spend_future_60d"] = (
                future60["spend"].sum()
                if len(future60) >= 40
                else np.nan
            )

            row["spend_future_90d"] = (
                future90["spend"].sum()
                if len(future90) >= 60
                else np.nan
            )

            # -----------------------------
            # Future ROAS
            # -----------------------------

            spend30_future = row["spend_future_30d"]

            spend60_future = row["spend_future_60d"]

            spend90_future = row["spend_future_90d"]

            row["roas_target_30d"] = safe_roas(
                row["revenue_target_30d"],
                spend30_future
            )

            row["roas_target_60d"] = (
                safe_roas(
                    row["revenue_target_60d"],
                    spend60_future
                )
                if not np.isnan(spend60_future)
                else np.nan
            )

            row["roas_target_90d"] = (
                safe_roas(
                    row["revenue_target_90d"],
                    spend90_future
                )
                if not np.isnan(spend90_future)
                else np.nan
            )

            # -----------------------------
            # Future Budget
            # -----------------------------

        # -------------------------------------------------
        # Store Feature Row
        # -------------------------------------------------

        rows.append(row)

    # -------------------------------------------------
    # Return Feature DataFrame
    # -------------------------------------------------

    if not rows:

        return pd.DataFrame()

    feature_df = pd.DataFrame(rows)

    numeric_columns = feature_df.select_dtypes(
        include=[np.number]
    ).columns

    feature_df[numeric_columns] = (
        feature_df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    feature_df = feature_df.sort_values(
        "date"
    ).reset_index(drop=True)

    return feature_df
FEATURE_COLS = [

    "spend_7d",
    "spend_14d",
    "spend_30d",

    "revenue_7d",
    "revenue_14d",
    "revenue_30d",

    "roas_7d",
    "roas_14d",
    "roas_30d",

    "ctr_30d",
    "cvr_30d",
    "cpc_30d",
    "cpm_30d",
    "rpc_30d",
    "aov_30d",

    "budget_avg_30d",
    "budget_util_30d",

    "spend_trend",
    "revenue_trend",
    "roas_trend",

    "spend_std_30",
    "revenue_std_30",
    "roas_std_30",

    "revenue_lag1",
    "revenue_lag7",

    "spend_lag1",
    "spend_lag7",

    "roas_lag7",

    "campaign_age",
    "campaign_efficiency",

    "day_of_week",
    "day_of_month",
    "week_of_year",

    "month",
    "quarter",

    "is_weekend",
    "is_holiday",

    "channel_enc",
    "camp_type_enc",

    "future_budget"

]
# ==========================================================
# FEATURE GENERATION PIPELINE
# ==========================================================

def generate_features(
    data_dir: str,
    output_file: str,
    mode: str = "train",
):
    """
    Generate feature dataset from raw campaign CSV files.
    """

    print(f"Loading data from {data_dir}")

    df = load_and_normalize(data_dir)

    print(f"Loaded {len(df):,} rows")

    feature_frames = []

    grouped = df.groupby(

        [

            "channel",

            "campaign_name",

        ]

    )

    for _, campaign_df in grouped:

        features = build_campaign_features(

            campaign_df,

            mode=mode,

        )

        if not features.empty:

            feature_frames.append(

                features

            )

    if len(feature_frames) == 0:

        raise RuntimeError(

            "No features were generated."

        )

    features = pd.concat(

        feature_frames,

        ignore_index=True,

    )

    features.to_parquet(

        output_file,

        index=False,

    )

    print()

    print(f"Saved {len(features):,} rows")

    print(f"Output : {output_file}")

    return features



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        default="train",
    )

    args = parser.parse_args()

    output = (
        "features_train.parquet"
        if args.mode == "train"
        else "features.parquet"
    )

    generate_features(
        data_dir="data",
        output_file=output,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
