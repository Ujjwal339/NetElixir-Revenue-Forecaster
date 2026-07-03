import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import shap

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from xgboost import XGBRegressor

from generate_features import FEATURE_COLS

warnings.filterwarnings("ignore")


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

SEED = 42

HORIZONS = {
    30: "revenue_target_30d",
    60: "revenue_target_60d",
    90: "revenue_target_90d",
}

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p90": 0.90,
}

# ==========================================================
# MODEL CREATION
# ==========================================================

def make_model(
    quantile_alpha: float,
) -> XGBRegressor:
    """
    Create a quantile XGBoost regression model.
    """

    model = XGBRegressor(

        objective="reg:quantileerror",

        quantile_alpha=quantile_alpha,

        n_estimators=500,

        learning_rate=0.03,

        max_depth=6,

        min_child_weight=3,

        subsample=0.85,

        colsample_bytree=0.85,

        gamma=0.5,

        reg_alpha=0.1,

        reg_lambda=1.5,

        random_state=SEED,

        tree_method="hist",

        n_jobs=-1,

        verbosity=0,

    )

    return model
# ==========================================================
# TRAINING PIPELINE
# ==========================================================

def train_all(
    features_path: str,
    model_path: str,
):
    """
    Train quantile regression models for all forecast
    horizons and evaluate their performance.
    """

    print(f"\nLoading features: {features_path}")

    df = pd.read_parquet(
        features_path
    )

    if df.empty:

        raise RuntimeError(
            "Training feature file is empty."
        )

    print(
        f"Loaded {len(df):,} training samples."
    )

    models = {}

    metrics = []

    feature_importance = []

    for horizon, target_column in HORIZONS.items():

        print("\n" + "=" * 60)

        print(
            f"Training {horizon}-Day Forecast Models"
        )

        print("=" * 60)

        df_h = (
            df
            .dropna(
                subset=[target_column]
            )
            .copy()
        )

        if len(df_h) < 50:

            print(
                f"Skipping {horizon}-day model "
                f"(only {len(df_h)} samples)."
            )

            continue

        df_h = df_h.sort_values(
            "date"
        ).reset_index(
            drop=True
        )

        split_index = int(
            len(df_h) * 0.80
        )

        train_df = df_h.iloc[
            :split_index
        ]

        valid_df = df_h.iloc[
            split_index:
        ]

        X_train = (
            train_df[FEATURE_COLS]
            .astype(float)
            .values
        )

        y_train = (
            train_df[target_column]
            .clip(lower=0)
            .values
        )

        X_valid = (
            valid_df[FEATURE_COLS]
            .astype(float)
            .values
        )

        y_valid = (
            valid_df[target_column]
            .clip(lower=0)
            .values
        )

        print(
            f"Training Samples   : {len(X_train):,}"
        )

        print(
            f"Validation Samples : {len(X_valid):,}"
        )
        # --------------------------------------------------
        # Train Quantile Models
        # --------------------------------------------------

        validation_predictions = {}

        training_predictions = {}

        for quantile_name, quantile_alpha in QUANTILES.items():

            print(
                f"\nTraining {quantile_name.upper()} "
                f"(α={quantile_alpha:.2f})"
            )

            model = make_model(
                quantile_alpha
            )

            model.fit(

                X_train,

                y_train,

                eval_set=[
                    (
                        X_valid,
                        y_valid,
                    )
                ],

                verbose=False,

            )

            model_key = (
                f"{horizon}_{quantile_name}"
            )

            models[model_key] = model

            train_pred = (
                model
                .predict(X_train)
                .clip(0)
            )

            valid_pred = (
                model
                .predict(X_valid)
                .clip(0)
            )

            training_predictions[
                quantile_name
            ] = train_pred

            validation_predictions[
                quantile_name
            ] = valid_pred
            # --------------------------------------------------
            # Evaluation Metrics
            # --------------------------------------------------

            mae = mean_absolute_error(

                y_valid,

                valid_pred,

            )

            rmse = np.sqrt(

                mean_squared_error(

                    y_valid,

                    valid_pred,

                )

            )

            mape = mean_absolute_percentage_error(

                y_valid,

                valid_pred,

            ) * 100

            residuals = (
                y_valid -
                valid_pred
            )

            pinball_loss = np.mean(

                np.where(

                    residuals >= 0,

                    quantile_alpha * residuals,

                    (quantile_alpha - 1) * residuals,

                )

            )

            metrics.append({

                "horizon": horizon,

                "quantile": quantile_name,

                "mae": round(
                    mae,
                    4,
                ),

                "rmse": round(
                    rmse,
                    4,
                ),

                "mape": round(
                    mape,
                    2,
                ),

                "pinball_loss": round(
                    pinball_loss,
                    4,
                ),

            })

            print(

                f"Validation MAE          : "
                f"{mae:,.2f}"

            )

            print(

                f"Validation RMSE         : "
                f"{rmse:,.2f}"

            )

            print(

                f"Validation MAPE         : "
                f"{mape:.2f}%"

            )

            print(

                f"Validation Pinball Loss : "
                f"{pinball_loss:.4f}"

            )
        # --------------------------------------------------
        # Coverage Evaluation
        # --------------------------------------------------

        p10_predictions = validation_predictions["p10"]

        p50_predictions = validation_predictions["p50"]

        p90_predictions = validation_predictions["p90"]

        coverage = np.mean(

            (y_valid >= p10_predictions) &

            (y_valid <= p90_predictions)

        ) * 100

        interval_width = np.mean(

            p90_predictions -

            p10_predictions

        )

        print()

        print(f"Prediction Coverage : {coverage:.2f}%")

        print(f"Average Interval Width : {interval_width:.2f}")

        metrics.append({

            "horizon": horizon,

            "quantile": "overall",

            "mae": np.nan,

            "rmse": np.nan,

            "mape": np.nan,

            "pinball_loss": np.nan,

            "coverage": round(

                coverage,

                2,

            ),

            "interval_width": round(

                interval_width,

                2,

            ),

        })

        # --------------------------------------------------
        # SHAP Feature Importance
        # --------------------------------------------------

        print("Computing SHAP feature importance...")

        sample_size = min(
            500,
            len(X_valid),
        )

        sample_index = np.random.choice(
            len(X_valid),
            sample_size,
            replace=False,
        )

        X_shap = X_valid[sample_index]

        explainer = shap.Explainer(
            models[f"{horizon}_p50"]
        )

        shap_values = explainer(
            X_shap
        )

        importance = np.abs(
            shap_values.values
        ).mean(axis=0)

        importance_df = pd.DataFrame({

            "feature": FEATURE_COLS,

            "importance": importance,

            "horizon": horizon,

        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        feature_importance.append(
            importance_df
        )
            # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_path = (
        Path(model_path)
        .parent
        / "training_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print()

    print(
        f"Training metrics saved to "
        f"{metrics_path}"
    )

    # --------------------------------------------------
    # Save Feature Importance
    # --------------------------------------------------

    if feature_importance:

        importance_df = pd.concat(

            feature_importance,

            ignore_index=True,

        )

        importance_path = (

            Path(model_path)

            .parent

            / "feature_importance.csv"

        )

        importance_df.to_csv(

            importance_path,

            index=False,

        )

        print(

            f"Feature importance saved to "
            f"{importance_path}"

        )

    # --------------------------------------------------
    # Save Models
    # --------------------------------------------------

    Path(model_path).parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        model_path,

        "wb",

    ) as file:

        pickle.dump(

            models,

            file,

            protocol=pickle.HIGHEST_PROTOCOL,

        )

    print()

    print(

        f"Saved {len(models)} trained models."

    )

    print(

        f"Model path : {model_path}"

    )

    print()

    print("=" * 60)

    print("Training Completed Successfully")

    print("=" * 60)

    return models

# ==========================================================
# MAIN
# ==========================================================

def main():
    """
    Command-line entry point for model training.
    """

    parser = argparse.ArgumentParser(
        description="Train probabilistic XGBoost revenue forecasting models."
    )

    parser.add_argument(
        "--features",
        type=str,
        default="features_train.parquet",
        help="Path to the training feature file.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="./pickle/model.pkl",
        help="Path to save the trained models.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Revenue Forecast Training Pipeline")
    print("=" * 60)

    print(f"Training Features : {args.features}")
    print(f"Model Output      : {args.model_path}")

    train_all(
        features_path=args.features,
        model_path=args.model_path,
    )

    print()
    print("=" * 60)
    print("Training Finished Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()


