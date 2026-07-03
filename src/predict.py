"""
predict.py

Generate probabilistic revenue and ROAS forecasts
for 30, 60 and 90 day horizons using trained
quantile regression models.
"""

import argparse
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from generate_features import FEATURE_COLS

warnings.filterwarnings("ignore")


HORIZONS = [30, 60, 90]


def load_models(model_path: str) -> dict:
    """
    Load trained models and verify that every
    required quantile model exists.
    """

    model_path = Path(model_path)

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    with open(model_path, "rb") as f:

        models = pickle.load(f)

    required_models = []

    for horizon in HORIZONS:

        required_models.extend([
            f"{horizon}_p10",
            f"{horizon}_p50",
            f"{horizon}_p90",
        ])

    missing = [
        model
        for model in required_models
        if model not in models
    ]

    if missing:

       raise ValueError(
            "Missing trained models:\n"
             + "\n".join(missing)
       )

    for model_name in required_models:

        model = models[model_name]

        if not hasattr(model, "predict"):

            raise TypeError(
                  f"Invalid model '{model_name}'. "
                  "Expected a trained model with a predict() method."
            )  

    print(
         f"Loaded {len(models)} models successfully."
    )

    return models

def scale_budget(
    features: np.ndarray,
    future_budget: float,
    feature_index: int,
) -> np.ndarray:
    """
    Update the future budget feature while keeping all
    remaining features unchanged.
    """

    scaled = features.copy()

    scaled[0, feature_index] = future_budget

    return scaled

def predict_campaign(
    models: dict,
    features_row: pd.Series,
    custom_budget: float | None = None,
) -> pd.DataFrame:
    """
    Generate probabilistic revenue and ROAS forecasts
    for a single campaign across all forecast horizons.
    """

    feature_index = FEATURE_COLS.index("future_budget")

    base_budget = float(features_row["future_budget"])
    missing = [
        c
        for c in FEATURE_COLS
        if c not in features_row.index
    ]   

    if missing:

       raise ValueError(
           "Missing feature columns:\n"
           + "\n".join(missing)
       )
    X = (
        features_row[FEATURE_COLS]
        .astype(float)
        .values
        .reshape(1, -1)
    )

    forecasts = []

    for horizon in HORIZONS:

        if custom_budget is None:

            budget = base_budget * (horizon / 30)

        else:

            budget = custom_budget * (horizon / 30)

        X_scaled = scale_budget(
            X,
            budget,
            feature_index,
        )

        try:

            p10 = float(
                models[f"{horizon}_p10"].predict(X_scaled)[0]
            )

            p50 = float(
                models[f"{horizon}_p50"].predict(X_scaled)[0]
            )

            p90 = float(
                models[f"{horizon}_p90"].predict(X_scaled)[0]
            )

        except KeyError as e:

            raise KeyError(
                f"Missing model for {horizon} days: {e}"
            )

        p10 = max(0.0, p10)
        p50 = max(0.0, p50)
        p90 = max(0.0, p90)

        p10, p50, p90 = sorted(
            [
                p10,
                p50,
                p90,
            ]
        )

        if budget > 0:

            roas_p10 = p10 / budget
            roas_p50 = p50 / budget
            roas_p90 = p90 / budget

        else:

            roas_p10 = 0.0
            roas_p50 = 0.0
            roas_p90 = 0.0

        forecasts.append({

            "period_days": horizon,

            "budget_input": round(
                budget,
                2,
            ),

            "revenue_p10": round(
                p10,
                2,
            ),

            "revenue_p50": round(
                p50,
                2,
            ),

            "revenue_p90": round(
                p90,
                2,
            ),

            "roas_p10": round(
                roas_p10,
                4,
            ),

            "roas_p50": round(
                roas_p50,
                4,
            ),

            "roas_p90": round(
                roas_p90,
                4,
            ),
        })

    return pd.DataFrame(forecasts)
def aggregate_forecasts(
    forecast_df: pd.DataFrame,
    aggregation_level: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Aggregate campaign forecasts into higher reporting levels.

    Revenue values are summed and blended ROAS is
    recomputed from aggregated revenue and budget.
    """

    aggregated = (

        forecast_df
        .groupby(
            group_columns + ["period_days"],
            as_index=False,
        )
        .agg(

            budget_input=("budget_input", "sum"),

            revenue_p10=("revenue_p10", "sum"),

            revenue_p50=("revenue_p50", "sum"),

            revenue_p90=("revenue_p90", "sum"),
        )

    )

    budget = aggregated["budget_input"].replace(
        0,
        np.nan,
    )

    aggregated["roas_p10"] = (
        aggregated["revenue_p10"] /
        budget
    ).fillna(0)

    aggregated["roas_p50"] = (
        aggregated["revenue_p50"] /
        budget
    ).fillna(0)

    aggregated["roas_p90"] = (
        aggregated["revenue_p90"] /
        budget
    ).fillna(0)

    aggregated["aggregation_level"] = aggregation_level

    for column in [

        "channel",

        "campaign_type",

        "campaign_name",

    ]:

        if column not in aggregated.columns:

            aggregated[column] = "ALL"

    return aggregated

def run_predictions(
    features_path: str,
    model_path: str,
    output_path: str,
    budget_overrides: dict | None = None,
):
    """
    Run the complete forecasting pipeline.
    """

    print(f"\nLoading features: {features_path}")

    feature_df = pd.read_parquet(
        features_path
    )

    if feature_df.empty:

        raise RuntimeError(
            "Feature file is empty."
        )

    print(
        f"Loaded {len(feature_df)} campaigns."
    )

    print(f"\nLoading models: {model_path}")

    models = load_models(
        model_path
    )

    forecast_date = str(
        date.today()
    )

    campaign_results = []

    for _, campaign in feature_df.iterrows():

        campaign_name = campaign["campaign_name"]

        if budget_overrides:

            custom_budget = budget_overrides.get(
                campaign_name
            )

        else:

            custom_budget = None

        prediction = predict_campaign(

            models,

            campaign,

            custom_budget,

        )

        prediction["forecast_date"] = forecast_date

        prediction["channel"] = campaign["channel"]

        prediction["campaign_type"] = campaign["campaign_type"]

        prediction["campaign_name"] = campaign_name

        prediction["aggregation_level"] = "campaign"

        campaign_results.append(
            prediction
        )

    if len(campaign_results) == 0:

        raise RuntimeError(
            "No campaign forecasts generated."
        )

    campaign_forecasts = pd.concat(

        campaign_results,

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

    final = pd.concat(

        [

            overall_forecasts,

            channel_forecasts,

            campaign_type_forecasts,

            campaign_forecasts,

        ],

        ignore_index=True,

    )

    final = final[

        [

            "forecast_date",

            "aggregation_level",

            "channel",

            "campaign_type",

            "campaign_name",

            "period_days",

            "budget_input",

            "revenue_p10",

            "revenue_p50",

            "revenue_p90",

            "roas_p10",

            "roas_p50",

            "roas_p90",

        ]

    ]

    Path(output_path).parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    final.to_csv(

        output_path,

        index=False,

    )

    print(
        f"\nForecast saved to {output_path}"
    )

    overall = (

        final[
            final["aggregation_level"] == "overall"
        ]
        .sort_values(
            "period_days"
        )

    )

    print("\nOverall Forecast")

    for _, row in overall.iterrows():

        print(

            f"{row.period_days:>3} Days | "

            f"Budget ${row.budget_input:,.0f} | "

            f"Revenue ${row.revenue_p10:,.0f} - "

            f"${row.revenue_p50:,.0f} - "

            f"${row.revenue_p90:,.0f} | "

            f"ROAS "

            f"{row.roas_p10:.2f} "

            f"{row.roas_p50:.2f} "

            f"{row.roas_p90:.2f}"

        )

    return final

def main():
    """
    Command-line entry point for the forecasting pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Generate probabilistic revenue and ROAS forecasts."
    )

    parser.add_argument(
        "--features",
        type=str,
        default="features.parquet",
        help="Path to the inference feature file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="./pickle/model.pkl",
        help="Path to the trained model file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output/predictions.csv",
        help="Path to save prediction results.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Revenue Forecast Prediction Pipeline")
    print("=" * 60)

    print(f"Feature File : {args.features}")
    print(f"Model File   : {args.model}")
    print(f"Output File  : {args.output}")

    run_predictions(
        features_path=args.features,
        model_path=args.model,
        output_path=args.output,
    )

    print("\nPrediction completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
